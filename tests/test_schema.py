"""Tests for the netlist schema and consistency checks."""

from __future__ import annotations

from circuit_extract.schema import Component, Net, Netlist, PinRef


def _voltage_divider() -> Netlist:
    return Netlist(
        components=[
            Component(id="V1", type="voltage_source", value="5V", pins=["+", "-"]),
            Component(id="R1", type="resistor", value="10k", pins=["1", "2"]),
            Component(id="R2", type="resistor", value="10k", pins=["1", "2"]),
            Component(id="GND1", type="ground", pins=["1"]),
        ],
        nets=[
            Net(
                name="VCC",
                pins=[PinRef(component_id="V1", pin="+"), PinRef(component_id="R1", pin="1")],
            ),
            Net(
                name="VOUT",
                pins=[PinRef(component_id="R1", pin="2"), PinRef(component_id="R2", pin="1")],
            ),
            Net(
                name="GND",
                pins=[
                    PinRef(component_id="V1", pin="-"),
                    PinRef(component_id="R2", pin="2"),
                    PinRef(component_id="GND1", pin="1"),
                ],
            ),
        ],
    )


def test_voltage_divider_is_consistent() -> None:
    nl = _voltage_divider()
    assert nl.validate_consistency() == []


def test_round_trip_json() -> None:
    nl = _voltage_divider()
    blob = nl.model_dump_json()
    restored = Netlist.model_validate_json(blob)
    assert restored == nl


def test_unknown_component_reference_is_flagged() -> None:
    nl = _voltage_divider()
    nl.nets[0].pins.append(PinRef(component_id="R99", pin="1"))
    warnings = nl.validate_consistency()
    assert any("R99" in w for w in warnings)


def test_unknown_pin_reference_is_flagged() -> None:
    nl = _voltage_divider()
    nl.nets[0].pins.append(PinRef(component_id="R1", pin="99"))
    warnings = nl.validate_consistency()
    assert any("R1" in w and "99" in w for w in warnings)


def test_singleton_net_is_flagged() -> None:
    nl = _voltage_divider()
    nl.nets.append(Net(name="NC_R1_2", pins=[PinRef(component_id="R1", pin="2")]))
    warnings = nl.validate_consistency()
    assert any("fewer than 2 pins" in w for w in warnings)
