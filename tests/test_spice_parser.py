"""Tests for the HSPICE netlist parser."""

from __future__ import annotations

from pathlib import Path

from circuit_extract.datasets.spice_parser import parse_spice
from circuit_extract.schema import Netlist

VOLTAGE_DIVIDER_SP = """\
* Voltage divider
V1 VCC 0 DC 5V
R1 VCC VOUT 10k
R2 VOUT 0 10k
.end
"""

MOSFET_INVERTER_SP = """\
* CMOS inverter
VDD VDD 0 DC 1.8
VIN IN 0 PULSE(0 1.8 0 1n 1n 5n 10n)
M1 OUT IN VDD VDD PMOS W=2u L=0.18u
M2 OUT IN 0 0 NMOS W=1u L=0.18u
.end
"""

BJT_AMP_SP = """\
* Common-emitter amplifier
VCC VCC 0 DC 12V
R1 VCC BASE 100k
R2 BASE 0 47k
RC VCC COLLECTOR 4.7k
RE EMITTER 0 1k
Q1 COLLECTOR BASE EMITTER 2N2222
.end
"""

SUBCIRCUIT_SP = """\
* Opamp circuit
V1 VCC 0 DC 15
V2 0 VEE DC 15
X1 INP INN OUT OPAMP741
R1 INP 0 10k
R2 INN OUT 100k
.end
"""

CONTINUATION_SP = """\
* Test continuation lines
R1 N1 N2
+ 10k
V1 VCC 0
+ DC 5V
.end
"""


def _parse_string(sp: str, tmp_path: Path) -> Netlist:
    f = tmp_path / "test.sp"
    f.write_text(sp)
    return parse_spice(f)


def test_voltage_divider(tmp_path: Path) -> None:
    nl = _parse_string(VOLTAGE_DIVIDER_SP, tmp_path)
    assert len(nl.components) == 3
    types = {c.id: c.type for c in nl.components}
    assert types["V1"] == "voltage_source"
    assert types["R1"] == "resistor"
    assert types["R2"] == "resistor"

    net_names = {n.name for n in nl.nets}
    assert "GND" in net_names
    assert "VCC" in net_names
    assert "VOUT" in net_names
    assert nl.validate_consistency() == []


def test_mosfet_inverter(tmp_path: Path) -> None:
    nl = _parse_string(MOSFET_INVERTER_SP, tmp_path)
    types = {c.id: c.type for c in nl.components}
    assert types["M1"] == "nmos"
    assert types["M2"] == "nmos"
    # M1 should have 4 pins: D G S B
    m1 = next(c for c in nl.components if c.id == "M1")
    assert m1.pins == ["D", "G", "S", "B"]


def test_bjt_amplifier(tmp_path: Path) -> None:
    nl = _parse_string(BJT_AMP_SP, tmp_path)
    types = {c.id: c.type for c in nl.components}
    assert types["Q1"] == "bjt_npn"
    q1 = next(c for c in nl.components if c.id == "Q1")
    assert q1.pins == ["C", "B", "E"]
    assert q1.value == "2N2222"


def test_subcircuit_instance(tmp_path: Path) -> None:
    nl = _parse_string(SUBCIRCUIT_SP, tmp_path)
    x1 = next(c for c in nl.components if c.id == "X1")
    assert x1.type == "ic"
    assert x1.value == "OPAMP741"
    assert len(x1.pins) == 3


def test_continuation_lines(tmp_path: Path) -> None:
    nl = _parse_string(CONTINUATION_SP, tmp_path)
    r1 = next(c for c in nl.components if c.id == "R1")
    assert r1.value == "10k"
    v1 = next(c for c in nl.components if c.id == "V1")
    assert v1.value is not None and "5V" in v1.value


def test_ground_normalisation(tmp_path: Path) -> None:
    nl = _parse_string(VOLTAGE_DIVIDER_SP, tmp_path)
    gnd_net = next(n for n in nl.nets if n.name == "GND")
    comp_ids = {ref.component_id for ref in gnd_net.pins}
    assert "V1" in comp_ids
    assert "R2" in comp_ids
