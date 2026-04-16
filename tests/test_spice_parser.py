"""Tests for the HSPICE netlist parser.

Covers both the hanky2397 dataset format (.subckt-wrapped, topology-only)
and standard HSPICE (flat with values).
"""

from __future__ import annotations

from pathlib import Path

from circuit_extract.datasets.spice_parser import parse_spice
from circuit_extract.schema import Netlist

# -----------------------------------------------------------------------
# hanky2397 dataset format (topology-only, .subckt-wrapped)
# -----------------------------------------------------------------------

DATASET_MOSFETS_AND_BJTS = """\

.subckt 000001_output
q0 gnd gnd net9 pnp
q1 gnd gnd net8 pnp
q2 gnd gnd net6 pnp
m3 net3 net3 net0 net0 pmos4
m4 net1 net1 net6 net6 nmos4
m5 net3 net1 net7 net7 nmos4
m6 net5 net4 net0 net0 pmos4
m7 net1 net2 net0 net0 pmos4
r8 net5 net9 r
r9 net7 net8 r
.ends
"""

DATASET_SOURCES_AND_CAP = """\

.subckt 000299_output
v0 net4 gnd v
r1 net1 net3 r
i2 net5 gnd i
r3 net0 net2 r
i4 net6 gnd i
q5 net2 net4 net5 npn
q6 net3 net2 net6 npn
c9 net5 net6 c
.ends
"""

DATASET_CAPS_AND_MOSFETS = """\

.subckt 000026_output
m0 net1 net2 net0 net0 pmos4
m1 net4 net3 net0 net0 pmos4
m2 net1 net5 net8 net8 nmos4
m3 net8 net12 gnd gnd nmos4
m4 net4 net7 net8 net8 nmos4
c5 net1 net12 c
c7 net12 net4 c
.ends
"""


# -----------------------------------------------------------------------
# Standard HSPICE format (flat with values)
# -----------------------------------------------------------------------

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


# -----------------------------------------------------------------------
# Dataset format tests
# -----------------------------------------------------------------------


def test_dataset_mosfets_and_bjts(tmp_path: Path) -> None:
    nl = _parse_string(DATASET_MOSFETS_AND_BJTS, tmp_path)
    assert len(nl.components) == 10
    types = {c.id: c.type for c in nl.components}
    assert types["q0"] == "bjt_pnp"
    assert types["m3"] == "pmos"
    assert types["m4"] == "nmos"
    assert types["r8"] == "resistor"


def test_dataset_mosfet_pins(tmp_path: Path) -> None:
    nl = _parse_string(DATASET_MOSFETS_AND_BJTS, tmp_path)
    m3 = next(c for c in nl.components if c.id == "m3")
    assert m3.pins == ["D", "G", "S", "B"]
    assert m3.type == "pmos"


def test_dataset_bjt_pins(tmp_path: Path) -> None:
    nl = _parse_string(DATASET_MOSFETS_AND_BJTS, tmp_path)
    q0 = next(c for c in nl.components if c.id == "q0")
    assert q0.pins == ["C", "B", "E"]
    assert q0.type == "bjt_pnp"


def test_dataset_sources_and_cap(tmp_path: Path) -> None:
    nl = _parse_string(DATASET_SOURCES_AND_CAP, tmp_path)
    assert len(nl.components) == 8
    types = {c.id: c.type for c in nl.components}
    assert types["v0"] == "voltage_source"
    assert types["i2"] == "current_source"
    assert types["c9"] == "capacitor"
    assert types["q5"] == "bjt_npn"


def test_dataset_gnd_net(tmp_path: Path) -> None:
    nl = _parse_string(DATASET_SOURCES_AND_CAP, tmp_path)
    gnd_net = next(n for n in nl.nets if n.name == "GND")
    comp_ids = {ref.component_id for ref in gnd_net.pins}
    assert "v0" in comp_ids
    assert "i2" in comp_ids
    assert "i4" in comp_ids


def test_dataset_no_values(tmp_path: Path) -> None:
    """Dataset format components should have no value (topology-only)."""
    nl = _parse_string(DATASET_CAPS_AND_MOSFETS, tmp_path)
    for comp in nl.components:
        if comp.type != "ic":
            assert comp.value is None, f"{comp.id} should have no value"


def test_dataset_net_connectivity(tmp_path: Path) -> None:
    nl = _parse_string(DATASET_CAPS_AND_MOSFETS, tmp_path)
    net_map = {n.name: {(r.component_id, r.pin) for r in n.pins} for n in nl.nets}
    assert ("m0", "S") in net_map["net0"]
    assert ("m1", "S") in net_map["net0"]


# -----------------------------------------------------------------------
# Standard HSPICE format tests
# -----------------------------------------------------------------------


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
