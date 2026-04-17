"""Tests for the KiCad JSON parser.

Uses handcrafted json fixtures that mimic the shape of the ``bshada/open-
schematics`` dataset's ``json`` column, so the parser can be exercised
without any network access or parquet files.
"""

from __future__ import annotations

from typing import Any

from circuit_extract.datasets.kicad_parser import (
    _is_power_symbol,
    parse_kicad_components,
    parse_kicad_json,
)


def _symbol(
    entry_name: str,
    reference: str,
    value: str = "",
    library_nickname: str = "MyLib",
) -> dict[str, Any]:
    return {
        "libraryNickname": library_nickname,
        "entryName": entry_name,
        "position": {"x": 0, "y": 0, "angle": 0},
        "properties": [
            {"key": "Reference", "value": reference},
            {"key": "Value", "value": value or entry_name},
        ],
    }


def _kicad_json(symbols: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "libSymbols": [],
        "schematicSymbols": symbols,
        "junctions": [],
        "noConnects": [],
        "busEntries": [],
        "graphicalItems": [],
        "globalLabels": [],
        "sheetInstances": [],
    }


# ---------------------------------------------------------------------------
# Power / mechanical symbol detection
# ---------------------------------------------------------------------------


def test_power_symbol_detection_exact() -> None:
    assert _is_power_symbol("GND")
    assert _is_power_symbol("VCC")
    assert _is_power_symbol("VBUS")
    assert _is_power_symbol("PWR_FLAG")
    assert _is_power_symbol("MountingHole")
    assert _is_power_symbol("TestPoint")


def test_power_symbol_detection_prefixes() -> None:
    assert _is_power_symbol("+5V")
    assert _is_power_symbol("+3.3V")
    assert _is_power_symbol("+3V3")
    assert _is_power_symbol("-12V")
    assert _is_power_symbol("+VIN1")
    assert _is_power_symbol("PWR_ANALOG")
    assert _is_power_symbol("MountingHole_Pad")


def test_power_symbol_detection_false_positives() -> None:
    assert not _is_power_symbol("R")
    assert not _is_power_symbol("C")
    assert not _is_power_symbol("LM358")
    assert not _is_power_symbol("Conn_01x04")
    # Values that contain "V" but aren't rails
    assert not _is_power_symbol("Varistor")


# ---------------------------------------------------------------------------
# Basic component extraction
# ---------------------------------------------------------------------------


def test_parse_components_rejects_power_symbols_by_default() -> None:
    doc = _kicad_json(
        [
            _symbol("R", "R1", "10k"),
            _symbol("GND", "#GND01"),
            _symbol("+5V", "#PWR05"),
            _symbol("C", "C1", "100nF"),
        ]
    )
    comps = parse_kicad_components(doc)
    assert {c.id for c in comps} == {"R1", "C1"}


def test_parse_components_include_power() -> None:
    doc = _kicad_json(
        [
            _symbol("R", "R1", "10k"),
            _symbol("GND", "#GND01"),
            _symbol("+5V", "#PWR05"),
        ]
    )
    comps = parse_kicad_components(doc, include_power=True)
    assert len(comps) == 3


def test_parse_components_accepts_json_string() -> None:
    import json as _json

    doc = _kicad_json([_symbol("R", "R1", "10k")])
    comps = parse_kicad_components(_json.dumps(doc))
    assert comps[0].id == "R1"


# ---------------------------------------------------------------------------
# Multi-unit symbol deduplication
# ---------------------------------------------------------------------------


def test_multi_unit_symbol_dedup() -> None:
    """LM358 (dual opamp) shows up 3x in KiCad (opamp-A, opamp-B, power pins)
    all sharing Reference=IC1. We must count it once."""
    doc = _kicad_json(
        [
            _symbol("LM358D", "IC1", "LM358D"),
            _symbol("LM358D", "IC1", "LM358D"),
            _symbol("LM358D", "IC1", "LM358D"),
            _symbol("R", "R1", "10k"),
        ]
    )
    comps = parse_kicad_components(doc)
    assert len(comps) == 2
    assert {c.id for c in comps} == {"IC1", "R1"}


# ---------------------------------------------------------------------------
# Type classification
# ---------------------------------------------------------------------------


def test_classifies_short_device_names() -> None:
    doc = _kicad_json(
        [
            _symbol("R", "R1", "10k"),
            _symbol("R_Small", "R2", "1k"),
            _symbol("R_US", "R3", "4.7k"),
            _symbol("C", "C1", "100nF"),
            _symbol("C_Small", "C2", "10nF"),
            _symbol("C_Polarized", "C3", "10uF"),
            _symbol("L", "L1", "10uH"),
            _symbol("D", "D1", "1N4148"),
            _symbol("D_Zener", "D2", "BZX"),
            _symbol("LED", "LED1"),
        ]
    )
    comps = {c.id: c for c in parse_kicad_components(doc)}
    assert comps["R1"].type == "resistor"
    assert comps["R2"].type == "resistor"
    assert comps["R3"].type == "resistor"
    assert comps["C1"].type == "capacitor"
    assert comps["C2"].type == "capacitor"
    assert comps["C3"].type == "capacitor"
    assert comps["L1"].type == "inductor"
    assert comps["D1"].type == "diode"
    assert comps["D2"].type == "zener"
    assert comps["LED1"].type == "led"


def test_classifies_transistors() -> None:
    doc = _kicad_json(
        [
            _symbol("BC846", "Q1"),
            _symbol("2N2222", "Q2"),
            _symbol("2N3906", "Q3"),
            _symbol("MMBT3904", "Q4"),
        ]
    )
    comps = {c.id: c for c in parse_kicad_components(doc)}
    assert comps["Q1"].type == "bjt_npn"
    assert comps["Q2"].type == "bjt_npn"
    assert comps["Q3"].type == "bjt_pnp"
    assert comps["Q4"].type == "bjt_npn"


def test_classifies_opamps_and_ics() -> None:
    doc = _kicad_json(
        [
            _symbol("LM358D", "IC1"),
            _symbol("TL072", "IC2"),
            _symbol("Amplifier_Operational:LM324", "IC3", library_nickname="Amp"),
        ]
    )
    comps = {c.id: c for c in parse_kicad_components(doc)}
    assert comps["IC1"].type == "opamp"
    assert comps["IC2"].type == "opamp"
    assert comps["IC3"].type == "opamp"


def test_classifies_connectors() -> None:
    doc = _kicad_json(
        [
            _symbol("Conn_01x04", "J1"),
            _symbol("USB_A", "J2"),
            _symbol("Screw_Terminal_01x02", "J3"),
        ]
    )
    comps = {c.id: c for c in parse_kicad_components(doc)}
    assert comps["J1"].type == "connector"
    assert comps["J2"].type == "connector"
    assert comps["J3"].type == "connector"


def test_classifies_us_style_passives() -> None:
    """KiCad US-style library variants should map cleanly."""
    doc = _kicad_json(
        [
            _symbol("R_Small_US", "R1"),
            _symbol("R-US_0204/7", "R2", library_nickname="mainboard-eagle-import"),
            _symbol("C_Polarized_US", "C1"),
            _symbol("C_Generic", "C2", library_nickname="Library"),
            _symbol("R_Potentiometer", "RV1"),
        ]
    )
    comps = {c.id: c for c in parse_kicad_components(doc)}
    assert comps["R1"].type == "resistor"
    assert comps["R2"].type == "resistor"
    assert comps["RV1"].type == "resistor"
    assert comps["C1"].type == "capacitor"
    assert comps["C2"].type == "capacitor"


def test_classifies_mosfets_by_part_number() -> None:
    doc = _kicad_json(
        [
            _symbol("BSS138", "Q1", library_nickname="Transistor_FET"),
            _symbol("2N7002", "Q2", library_nickname="Transistor_FET"),
            _symbol("AO3401", "Q3"),
        ]
    )
    comps = {c.id: c for c in parse_kicad_components(doc)}
    assert comps["Q1"].type == "nmos"
    assert comps["Q2"].type == "nmos"
    assert comps["Q3"].type == "pmos"


def test_classifies_regulators_as_ic() -> None:
    doc = _kicad_json(
        [
            _symbol("AMS1117-3.3", "U1", library_nickname="Regulator_Linear"),
            _symbol("LM7805", "U2"),
            _symbol("LM317", "U3"),
        ]
    )
    comps = {c.id: c for c in parse_kicad_components(doc)}
    assert comps["U1"].type == "ic"
    assert comps["U2"].type == "ic"
    assert comps["U3"].type == "ic"


def test_classifies_keyboard_switches() -> None:
    """Mechanical keyboard projects are a large slice of the dataset."""
    doc = _kicad_json(
        [
            _symbol("SW_Push", "SW1"),
            _symbol("KEYSW", "SW2", library_nickname="keyboard_parts"),
            _symbol("MX_SW_solder", "SW3", library_nickname="PCM_marbastlib-mx"),
            _symbol("Choc", "SW4", library_nickname="mntcomp-keyboard"),
        ]
    )
    comps = {c.id: c for c in parse_kicad_components(doc)}
    for ref in ("SW1", "SW2", "SW3", "SW4"):
        assert comps[ref].type == "switch", f"{ref} should be switch"


def test_classifies_common_mcus_and_chips_as_ic() -> None:
    doc = _kicad_json(
        [
            _symbol("ATmega328", "U1"),
            _symbol("ESP32-WROOM", "U2"),
            _symbol("STM32F103", "U3"),
            _symbol("CH340C", "U4"),
            _symbol("74HC595", "U5"),
        ]
    )
    comps = {c.id: c for c in parse_kicad_components(doc)}
    for ref in ("U1", "U2", "U3", "U4", "U5"):
        assert comps[ref].type == "ic", f"{ref} should be ic"


def test_classifies_ferrite_bead_as_inductor() -> None:
    doc = _kicad_json([_symbol("FerriteBead_Small", "FB1")])
    assert parse_kicad_components(doc)[0].type == "inductor"


def test_classifies_battery() -> None:
    doc = _kicad_json([_symbol("Battery_Cell", "BT1"), _symbol("Battery", "BT2")])
    comps = {c.id: c for c in parse_kicad_components(doc)}
    assert comps["BT1"].type == "battery"
    assert comps["BT2"].type == "battery"


def test_tvs_diode() -> None:
    doc = _kicad_json(
        [
            _symbol("D_TVS", "D1"),
            _symbol("D_Schottky", "D2"),
        ]
    )
    comps = {c.id: c for c in parse_kicad_components(doc)}
    assert comps["D1"].type == "diode"
    assert comps["D2"].type == "diode"


def test_unknown_symbol_falls_back_to_other() -> None:
    doc = _kicad_json([_symbol("SomeRandomWidget", "U99")])
    comps = parse_kicad_components(doc)
    assert comps[0].type == "other"


# ---------------------------------------------------------------------------
# Value handling
# ---------------------------------------------------------------------------


def test_value_equals_entry_name_treated_as_none() -> None:
    # Common in KiCad: ICs often have Value = the symbol name (LM358D=LM358D)
    doc = _kicad_json([_symbol("LM358D", "IC1", "LM358D")])
    comp = parse_kicad_components(doc)[0]
    assert comp.value is None


def test_value_when_distinct_from_entry() -> None:
    doc = _kicad_json([_symbol("R", "R1", "10k")])
    comp = parse_kicad_components(doc)[0]
    assert comp.value == "10k"


def test_missing_value_is_none() -> None:
    # Symbol with no Value property at all
    sym = {
        "libraryNickname": "Device",
        "entryName": "R",
        "position": {"x": 0, "y": 0, "angle": 0},
        "properties": [{"key": "Reference", "value": "R1"}],
    }
    doc = _kicad_json([sym])
    comp = parse_kicad_components(doc)[0]
    assert comp.value is None


# ---------------------------------------------------------------------------
# Reference designator handling
# ---------------------------------------------------------------------------


def test_hash_prefixed_refs_skipped() -> None:
    """KiCad internal refs like #PWR01, #GND03 are hidden markers — skip."""
    doc = _kicad_json(
        [
            _symbol("GND", "#GND01"),
            _symbol("+5V", "#PWR05"),
        ]
    )
    assert parse_kicad_components(doc) == []


def test_hash_refs_kept_with_include_power() -> None:
    doc = _kicad_json(
        [
            _symbol("GND", "#GND01"),
            _symbol("+5V", "#PWR05"),
        ]
    )
    comps = parse_kicad_components(doc, include_power=True)
    assert {c.id for c in comps} == {"#GND01", "#PWR05"}


# ---------------------------------------------------------------------------
# Full parse_kicad_json
# ---------------------------------------------------------------------------


def test_parse_kicad_json_produces_valid_netlist() -> None:
    doc = _kicad_json(
        [
            _symbol("R", "R1", "10k"),
            _symbol("C", "C1", "100nF"),
        ]
    )
    netlist = parse_kicad_json(doc, stem="test_circuit")
    assert len(netlist.components) == 2
    assert netlist.nets == []
    assert netlist.source_image == "test_circuit"
    assert netlist.extractor == "ground_truth:kicad_json"
    # Schema consistency: no nets means no dangling pin refs
    assert netlist.validate_consistency() == []
