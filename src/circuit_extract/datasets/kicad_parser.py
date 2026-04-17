"""Parse KiCad schematic JSON into our :class:`Netlist` schema.

The ``bshada/open-schematics`` dataset ships each row with a ``json`` field
that is a pre-parsed dump of the ``.kicad_sch`` source. We use that directly
rather than re-parsing the S-expressions ourselves.

Phase A scope: components only. Net topology (ARI) requires additional work
(label extraction from the raw ``.kicad_sch`` plus union-find over wires +
pin positions); see :mod:`circuit_extract.datasets.kicad_nets`.
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any

from circuit_extract.schema import Component, ComponentType, Netlist

logger = logging.getLogger(__name__)


# Library-symbol name prefixes / exact matches that represent power rails,
# ground, or mechanical / informational markers — not real circuit elements.
# We exclude these from the component list.
_POWER_EXACT: frozenset[str] = frozenset(
    {
        "GND",
        "GNDA",
        "GNDD",
        "GNDPWR",
        "GNDREF",
        "GNDS",
        "Earth",
        "VCC",
        "VDD",
        "VSS",
        "VEE",
        "VBUS",
        "PWR_FLAG",
        "MountingHole",
        "MountingHole_Pad",
        "TestPoint",
        "TestPoint_Alt",
        "TestPoint_2Pole",
        "Fiducial",
        "NoConnect",
        "NC",
    }
)


def _is_power_symbol(entry_name: str) -> bool:
    """Return True if a KiCad library symbol represents a net/power/mechanical marker."""
    name = entry_name.strip()
    if name in _POWER_EXACT:
        return True
    # Power rail prefixes: +5V, +3.3V, -12V, +3V3, +VIN, etc.
    if name.startswith(("+", "-")) and any(ch.isdigit() for ch in name):
        return True
    # KiCad convention: PWR_* for power-related symbols
    if name.startswith("PWR_"):
        return True
    # Mechanical variants
    return name.startswith(("MountingHole", "TestPoint", "Fiducial"))


# Map KiCad library entry names / keywords to our canonical ComponentType.
# Keys are checked as case-insensitive substrings against
# ``libraryNickname:entryName``; first match wins so ORDER MATTERS (specific
# before generic).
_TYPE_KEYWORDS: list[tuple[str, ComponentType]] = [
    # --- Opamps (match before generic "ic") ----------------------------
    ("opamp", "opamp"),
    ("operational", "opamp"),
    ("lm358", "opamp"),
    ("lm324", "opamp"),
    ("lm741", "opamp"),
    ("tl07", "opamp"),
    ("tl08", "opamp"),
    ("ne5532", "opamp"),
    # --- Voltage regulators / ICs (match before "r" / "c") -------------
    ("ams1117", "ic"),
    ("lm317", "ic"),
    ("lm78", "ic"),
    ("regulator_linear", "ic"),
    ("regulator_switching", "ic"),
    ("mcp", "ic"),  # Microchip MCP* part numbers
    ("74hc", "ic"),
    ("74ls", "ic"),
    ("74ahc", "ic"),
    ("atmega", "ic"),
    ("attiny", "ic"),
    ("stm32", "ic"),
    ("esp32", "ic"),
    ("esp8266", "ic"),
    ("ch340", "ic"),
    # --- MOSFETs -------------------------------------------------------
    ("pmos", "pmos"),
    ("nmos", "nmos"),
    ("mosfet", "nmos"),
    ("transistor_fet", "nmos"),
    ("bss138", "nmos"),
    ("2n7002", "nmos"),
    ("ao3400", "nmos"),
    ("ao3401", "pmos"),
    ("dmn", "nmos"),
    # --- BJT -----------------------------------------------------------
    ("mmbt3904", "bjt_npn"),
    ("mmbt3906", "bjt_pnp"),
    ("2n2222", "bjt_npn"),
    ("2n3904", "bjt_npn"),
    ("2n3906", "bjt_pnp"),
    ("bc846", "bjt_npn"),
    ("bc847", "bjt_npn"),
    ("bc856", "bjt_pnp"),
    ("bc857", "bjt_pnp"),
    ("npn", "bjt_npn"),
    ("pnp", "bjt_pnp"),
    ("bjt", "bjt_npn"),
    # --- Optocouplers (treat as IC) ------------------------------------
    ("optocoupler", "ic"),
    ("tlp", "ic"),
    # --- Diodes / LEDs -------------------------------------------------
    ("zener", "zener"),
    ("schottky", "diode"),
    ("d_tvs", "diode"),
    ("tvs", "diode"),
    ("led", "led"),
    ("diode", "diode"),
    # --- Passives ------------------------------------------------------
    ("r_potentiometer", "resistor"),
    ("r_small", "resistor"),
    ("r_us", "resistor"),
    ("r-us", "resistor"),
    ("r-eu", "resistor"),
    ("r_eu", "resistor"),
    ("0603r", "resistor"),
    ("0805r", "resistor"),
    ("resistor", "resistor"),
    ("c_small", "capacitor"),
    ("c_polarized", "capacitor"),
    ("c_us", "capacitor"),
    ("c-eu", "capacitor"),
    ("c_eu", "capacitor"),
    ("c_generic", "capacitor"),
    ("capacitor", "capacitor"),
    ("l_small", "inductor"),
    ("l_core", "inductor"),
    ("ferritebead", "inductor"),
    ("ferrite_bead", "inductor"),
    ("inductor", "inductor"),
    # --- Switches / buttons --------------------------------------------
    ("sw_push", "switch"),
    ("sw_spst", "switch"),
    ("sw_spdt", "switch"),
    ("sw_dpst", "switch"),
    ("keysw", "switch"),
    ("keyboard", "switch"),
    ("mx_sw", "switch"),  # Cherry MX switch
    ("choc", "switch"),  # Kailh Choc low-profile switch
    ("alps", "switch"),
    ("tgate", "switch"),
    ("switch", "switch"),
    # --- Other discrete ------------------------------------------------
    ("battery", "battery"),
    ("transformer", "transformer"),
    ("crystal", "crystal"),
    ("resonator", "crystal"),
    ("fuse", "fuse"),
    # --- Logic / digital (match after opamps/regulators) ---------------
    ("flipflop", "ic"),
    ("d_flipflop", "ic"),
    ("inverter", "ic"),
    # --- Connectors ----------------------------------------------------
    ("conn_", "connector"),
    ("connector", "connector"),
    ("header", "connector"),
    ("usb", "connector"),
    ("screw_terminal", "connector"),
    ("jumper", "connector"),
    ("jack", "connector"),
    ("socket", "connector"),
    ("barrel", "connector"),
]


def _classify(library_nickname: str, entry_name: str) -> ComponentType:
    """Map a KiCad library symbol to our :class:`ComponentType`.

    Ordering matters: specific (opamp, LM358) before generic (ic).
    """
    needle = f"{library_nickname}:{entry_name}".lower()

    # Exact-single-letter canonical prefixes from KiCad's Device library
    # ("Device:R", "Device:C", "Device:L", "Device:D") come through as short
    # entry names. Handle them explicitly so we don't mis-classify.
    short = entry_name.strip()
    if short in ("R", "R_Small", "R_US", "R_Variable"):
        return "resistor"
    if short in ("C", "C_Small", "C_Polarized", "C_Polarized_Small"):
        return "capacitor"
    if short in ("L", "L_Small", "L_Core_Iron", "L_Core_Ferrite"):
        return "inductor"
    if short in ("D", "D_Small", "D_Zener", "D_Schottky"):
        if "zener" in short.lower():
            return "zener"
        return "diode"
    if short in ("LED", "LED_Small"):
        return "led"

    for needle_keyword, ctype in _TYPE_KEYWORDS:
        if needle_keyword in needle:
            return ctype
    # Fallback: if it looks IC-like, mark it so; else "other".
    if "_ic" in needle or "74hc" in needle or "74ls" in needle or "ic_" in needle:
        return "ic"
    return "other"


def _get_property(sym: dict[str, Any], key: str) -> str | None:
    for prop in sym.get("properties", []):
        if prop.get("key") == key:
            val = prop.get("value")
            if isinstance(val, str):
                return val.strip() or None
    return None


def parse_kicad_components(
    kicad_json: str | dict[str, Any],
    *,
    include_power: bool = False,
) -> list[Component]:
    """Extract a deduplicated list of :class:`Component` from a KiCad json dump.

    Multi-unit symbols (e.g. LM358 shown as two opamp sub-symbols plus a power
    sub-symbol) appear once per unit in ``schematicSymbols`` with a shared
    Reference. We dedupe by Reference so the user's ``IC1`` counts once.

    Parameters
    ----------
    kicad_json:
        Either the raw JSON string from the dataset's ``json`` column, or the
        already-parsed dict.
    include_power:
        If False (default), drop GND/VCC/+5V/MountingHole/etc. markers.
    """
    data = kicad_json if isinstance(kicad_json, dict) else _json.loads(kicad_json)

    symbols = data.get("schematicSymbols", [])
    seen_refs: dict[str, Component] = {}

    for sym in symbols:
        entry_name = sym.get("entryName", "") or ""
        if not include_power and _is_power_symbol(entry_name):
            continue

        ref = _get_property(sym, "Reference") or ""
        # Auto-refs for hidden power parts look like "#GND03" or "#PWR01" — skip
        # them too when they slipped past the library-name filter.
        if ref.startswith("#") and not include_power:
            continue
        if not ref:
            # Give it a synthetic ref so it doesn't collide on empty-string key
            ref = f"_{entry_name}_{len(seen_refs)}"

        if ref in seen_refs:
            # Multi-unit symbol (same refdes, different unit). Skip subsequent units.
            continue

        value = _get_property(sym, "Value")
        # KiCad convention: if Value == entryName, treat as "no value".
        if value == entry_name:
            value = None

        library_nickname = sym.get("libraryNickname", "") or ""
        comp_type = _classify(library_nickname, entry_name)
        notes = f"{library_nickname}:{entry_name}" if library_nickname else entry_name

        seen_refs[ref] = Component(
            id=ref,
            type=comp_type,
            value=value,
            pins=[],  # Filled in Phase B when we derive pin positions.
            notes=notes,
        )

    return list(seen_refs.values())


def parse_kicad_json(
    kicad_json: str | dict[str, Any],
    *,
    include_power: bool = False,
    stem: str | None = None,
) -> Netlist:
    """Produce a component-only :class:`Netlist` from a KiCad json dump.

    Net topology is left empty here — see :mod:`kicad_nets` (Phase B).
    """
    components = parse_kicad_components(kicad_json, include_power=include_power)
    return Netlist(
        components=components,
        nets=[],
        source_image=stem,
        extractor="ground_truth:kicad_json",
    )
