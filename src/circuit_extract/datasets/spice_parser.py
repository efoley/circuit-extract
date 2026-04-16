"""Parse HSPICE ``.sp`` netlists into :class:`~circuit_extract.schema.Netlist`.

This is *not* a general-purpose SPICE parser — it targets the ground-truth
``.sp`` files from the ``hanky2397/schematic_images`` dataset, which are simple
flat netlists (no nested subcircuits, no expressions, no parameter sweeps).

Supported element prefixes:

    R  resistor          2 nodes + value
    C  capacitor         2 nodes + value
    L  inductor          2 nodes + value
    D  diode             2 nodes + model
    Q  BJT               3 nodes (C B E) + model
    M  MOSFET            4 nodes (D G S B) + model
    J  JFET              3 nodes (D G S) + model
    V  voltage source    2 nodes + value/spec
    I  current source    2 nodes + value/spec
    X  subcircuit inst   N nodes + subckt name
    E  VCVS              4 nodes + gain
    F  CCCS              2 nodes + Vname + gain
    G  VCCS              4 nodes + gain
    H  CCVS              2 nodes + Vname + gain
"""

from __future__ import annotations

import re
from pathlib import Path

from circuit_extract.schema import Component, ComponentType, Net, Netlist, PinRef

# Prefix → (canonical type, number of nodes, has_value)
# For types with variable node counts (X), we handle them specially.
_PREFIX_MAP: dict[str, tuple[ComponentType, int | None, bool]] = {
    "R": ("resistor", 2, True),
    "C": ("capacitor", 2, True),
    "L": ("inductor", 2, True),
    "D": ("diode", 2, False),
    "Q": ("bjt_npn", 3, False),  # Q defaults to npn; we refine below if possible
    "M": ("nmos", 4, False),  # M defaults to nmos; refined below if possible
    "J": ("jfet", 3, False),
    "V": ("voltage_source", 2, True),
    "I": ("current_source", 2, True),
    "X": ("ic", None, False),  # subcircuit — node count determined at parse time
    "E": ("other", 4, True),  # VCVS
    "F": ("other", 2, True),  # CCCS (+ controlling source name)
    "G": ("other", 4, True),  # VCCS
    "H": ("other", 2, True),  # CCVS
}

# Standard pin names per node count (positional)
_PIN_NAMES: dict[str, list[str]] = {
    "R": ["1", "2"],
    "C": ["1", "2"],
    "L": ["1", "2"],
    "D": ["+", "-"],
    "Q": ["C", "B", "E"],
    "M": ["D", "G", "S", "B"],
    "J": ["D", "G", "S"],
    "V": ["+", "-"],
    "I": ["+", "-"],
    "E": ["out+", "out-", "in+", "in-"],
    "G": ["out+", "out-", "in+", "in-"],
    "F": ["+", "-"],
    "H": ["+", "-"],
}


def _normalise_net(name: str) -> str:
    """Normalise a SPICE net name to a canonical form."""
    low = name.lower().strip()
    if low in ("0", "gnd", "gnd!"):
        return "GND"
    return name


def _join_continuation_lines(text: str) -> list[str]:
    """Join SPICE continuation lines (``+`` at start) with their predecessor."""
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("+") and lines:
            lines[-1] += " " + stripped[1:].strip()
        else:
            lines.append(stripped)
    return lines


def _extract_value(tokens: list[str]) -> str | None:
    """Pull a value string from remaining tokens after nodes.

    Handles things like ``10k``, ``100nF``, ``DC 5V``, ``AC 1``, model names.
    Returns the joined remainder or None if empty.
    """
    if not tokens:
        return None
    # Drop known SPICE keywords that aren't values
    cleaned = [t for t in tokens if not re.match(r"^[wl]=$", t, re.IGNORECASE)]
    # Rejoin parameter assignments (W=1u L=0.18u) and plain values
    val = " ".join(cleaned).strip()
    return val or None


def parse_spice(path: str | Path) -> Netlist:
    """Parse a ``.sp`` file and return a :class:`Netlist`."""
    text = Path(path).read_text()
    lines = _join_continuation_lines(text)

    components: list[Component] = []
    # net_name → set of (component_id, pin)
    net_pins: dict[str, list[PinRef]] = {}

    def _add_to_net(net_name: str, comp_id: str, pin: str) -> None:
        net_name = _normalise_net(net_name)
        ref = PinRef(component_id=comp_id, pin=pin)
        net_pins.setdefault(net_name, []).append(ref)

    for line in lines:
        # Skip comments and directives
        if line.startswith("*") or line.startswith("."):
            continue

        tokens = line.split()
        if not tokens:
            continue

        inst_name = tokens[0]
        prefix = inst_name[0].upper()

        if prefix not in _PREFIX_MAP:
            continue

        comp_type, node_count, has_value = _PREFIX_MAP[prefix]

        if prefix == "X":
            # Subcircuit: last non-param token is the subcircuit name,
            # everything between inst_name and subckt_name are nodes.
            # Heuristic: find the last token that doesn't contain '='
            non_param = [t for t in tokens[1:] if "=" not in t]
            if len(non_param) < 2:
                continue
            subckt_name = non_param[-1]
            node_tokens = non_param[:-1]
            pin_names = [str(i + 1) for i in range(len(node_tokens))]
            comp = Component(
                id=inst_name,
                type="ic",
                value=subckt_name,
                pins=pin_names,
            )
        else:
            assert node_count is not None
            if len(tokens) < 1 + node_count:
                continue
            node_tokens = tokens[1 : 1 + node_count]
            remaining = tokens[1 + node_count :]
            pin_names = _PIN_NAMES.get(prefix, [str(i + 1) for i in range(node_count)])

            value = _extract_value(remaining) if has_value else _extract_value(remaining)
            comp = Component(
                id=inst_name,
                type=comp_type,
                value=value,
                pins=list(pin_names),
            )

        components.append(comp)
        for pin_name, net_name in zip(pin_names, node_tokens, strict=False):
            _add_to_net(net_name, comp.id, pin_name)

    # Build nets
    nets: list[Net] = []
    for net_name in sorted(net_pins):
        pins = net_pins[net_name]
        nets.append(Net(name=net_name, pins=pins))

    return Netlist(
        components=components,
        nets=nets,
        extractor="ground_truth:spice",
    )
