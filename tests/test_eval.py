"""Tests for the evaluation metrics."""

from __future__ import annotations

from typing import cast

from circuit_extract.eval.metrics import (
    component_metrics,
    net_metrics,
)
from circuit_extract.schema import Component, ComponentType, Net, Netlist, PinRef


def _make_netlist(
    components: list[tuple[str, str]],
    nets: list[tuple[str, list[tuple[str, str]]]],
) -> Netlist:
    """Shorthand: components=[(id, type), ...], nets=[(name, [(comp, pin), ...]), ...]."""
    return Netlist(
        components=[
            Component(id=cid, type=cast(ComponentType, ctype), pins=[]) for cid, ctype in components
        ],
        nets=[
            Net(name=name, pins=[PinRef(component_id=c, pin=p) for c, p in pins])
            for name, pins in nets
        ],
    )


def test_perfect_component_match() -> None:
    nl = _make_netlist(
        [("R1", "resistor"), ("C1", "capacitor")],
        [],
    )
    m = component_metrics(nl, nl)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_partial_component_match() -> None:
    pred = _make_netlist([("R1", "resistor"), ("R2", "resistor"), ("R3", "resistor")], [])
    gt = _make_netlist([("R1", "resistor"), ("C1", "capacitor")], [])
    m = component_metrics(pred, gt)
    assert m.matched == 1
    assert m.predicted == 3
    assert m.ground_truth == 2


def test_type_normalisation_bjt() -> None:
    pred = _make_netlist([("Q1", "bjt_pnp")], [])
    gt = _make_netlist([("Q1", "bjt_npn")], [])
    m = component_metrics(pred, gt)
    assert m.matched == 1


def test_type_normalisation_mosfet() -> None:
    pred = _make_netlist([("M1", "pmos")], [])
    gt = _make_netlist([("M1", "nmos")], [])
    m = component_metrics(pred, gt)
    assert m.matched == 1


def test_perfect_net_ari() -> None:
    nl = _make_netlist(
        [("R1", "resistor"), ("R2", "resistor")],
        [
            ("N1", [("R1", "1"), ("R2", "1")]),
            ("N2", [("R1", "2"), ("R2", "2")]),
        ],
    )
    m = net_metrics(nl, nl)
    assert m.adjusted_rand_index == 1.0


def test_completely_wrong_nets() -> None:
    pred = _make_netlist(
        [("R1", "resistor"), ("R2", "resistor")],
        [
            ("N1", [("R1", "1"), ("R2", "2")]),
            ("N2", [("R1", "2"), ("R2", "1")]),
        ],
    )
    gt = _make_netlist(
        [("R1", "resistor"), ("R2", "resistor")],
        [
            ("N1", [("R1", "1"), ("R2", "1")]),
            ("N2", [("R1", "2"), ("R2", "2")]),
        ],
    )
    m = net_metrics(pred, gt)
    assert m.adjusted_rand_index < 0.01


def test_no_common_pins_gives_zero_ari() -> None:
    pred = _make_netlist(
        [("R1", "resistor")],
        [("N1", [("R1", "1"), ("R1", "2")])],
    )
    gt = _make_netlist(
        [("C1", "capacitor")],
        [("N1", [("C1", "1"), ("C1", "2")])],
    )
    m = net_metrics(pred, gt)
    assert m.adjusted_rand_index == 0.0
    assert m.common_pins == 0


def test_empty_netlists() -> None:
    pred = _make_netlist([], [])
    gt = _make_netlist([], [])
    cm = component_metrics(pred, gt)
    assert cm.f1 == 0.0
    nm = net_metrics(pred, gt)
    assert nm.adjusted_rand_index == 0.0
