"""Tests for the detection pipeline.

Doesn't load Grounding DINO (which would be a ~170 MB download). Instead
exercises the pure-Python pieces: label → ComponentType mapping, refdes
generation, NMS, and pipeline shape via a fake detector.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from circuit_extract.detect.grounding_dino import Detection, _build_prompt
from circuit_extract.detect.pipeline import (
    DetectionPipeline,
    _iou,
    _nms,
    classify_detection_label,
    detections_to_netlist,
)

# ---------------------------------------------------------------------------
# Label → ComponentType mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("resistor", "resistor"),
        ("a resistor", "resistor"),
        ("capacitor", "capacitor"),
        ("inductor", "inductor"),
        ("diode", "diode"),
        ("light emitting diode", "led"),
        ("led", "led"),
        ("zener diode", "zener"),
        ("transistor", "bjt_npn"),  # generic transistor → bjt_npn fallback
        ("bjt transistor", "bjt_npn"),
        ("npn transistor", "bjt_npn"),
        ("pnp transistor", "bjt_pnp"),
        ("nmos", "nmos"),
        ("pmos", "pmos"),
        ("mosfet", "nmos"),
        ("integrated circuit", "ic"),
        ("microcontroller chip", "ic"),
        ("connector", "connector"),
        ("usb connector", "connector"),
        ("switch", "switch"),
        ("push button", "switch"),
        ("battery", "battery"),
        ("fuse", "fuse"),
        ("crystal", "crystal"),
        ("transformer", "transformer"),
    ],
)
def test_classify_detection_label(label: str, expected: str) -> None:
    assert classify_detection_label(label) == expected


def test_classify_unknown_falls_back_to_other() -> None:
    assert classify_detection_label("spaceship") == "other"
    assert classify_detection_label("") == "other"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def test_build_prompt_format() -> None:
    prompt = _build_prompt(["resistor", "capacitor", "diode"])
    assert prompt == "resistor. capacitor. diode."


def test_build_prompt_lowercases_and_strips() -> None:
    prompt = _build_prompt(["  Resistor ", "LED"])
    assert prompt == "resistor. led."


# ---------------------------------------------------------------------------
# IoU and NMS
# ---------------------------------------------------------------------------


def _det(x1: int, y1: int, x2: int, y2: int, score: float, label: str = "resistor") -> Detection:
    return Detection(label=label, score=score, x1=x1, y1=y1, x2=x2, y2=y2)


def test_iou_overlapping_boxes() -> None:
    a = _det(0, 0, 10, 10, 0.9)
    b = _det(5, 5, 15, 15, 0.8)
    # Intersection = 25, union = 175
    assert abs(_iou(a, b) - 25 / 175) < 1e-6


def test_iou_disjoint_is_zero() -> None:
    a = _det(0, 0, 10, 10, 0.9)
    b = _det(20, 20, 30, 30, 0.8)
    assert _iou(a, b) == 0.0


def test_iou_identical_is_one() -> None:
    a = _det(0, 0, 10, 10, 0.9)
    b = _det(0, 0, 10, 10, 0.5)
    assert _iou(a, b) == 1.0


def test_nms_keeps_highest_score_among_overlaps() -> None:
    # Two near-identical boxes — NMS should keep the higher-scoring one
    # (standard NMS: each candidate is compared only against already-kept
    # boxes, so transitively-overlapping non-maxima may still survive).
    dets = [
        _det(0, 0, 10, 10, 0.9, "resistor"),
        _det(0, 0, 10, 10, 0.7, "integrated circuit"),
    ]
    kept = _nms(dets, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].score == 0.9
    assert kept[0].label == "resistor"


def test_nms_preserves_non_overlapping() -> None:
    dets = [
        _det(0, 0, 10, 10, 0.9),
        _det(100, 100, 110, 110, 0.8),
        _det(200, 200, 210, 210, 0.7),
    ]
    kept = _nms(dets, iou_threshold=0.5)
    assert len(kept) == 3


def test_nms_empty_input() -> None:
    assert _nms([]) == []


# ---------------------------------------------------------------------------
# detections_to_netlist
# ---------------------------------------------------------------------------


def test_detections_to_netlist_assigns_refdes() -> None:
    dets = [
        _det(0, 0, 10, 10, 0.9, "resistor"),
        _det(20, 0, 30, 10, 0.8, "resistor"),
        _det(40, 0, 50, 10, 0.8, "capacitor"),
    ]
    nl = detections_to_netlist(dets, source_image="foo.png")
    ids = [c.id for c in nl.components]
    types = [c.type for c in nl.components]
    assert ids == ["R1", "R2", "C1"]
    assert types == ["resistor", "resistor", "capacitor"]
    assert nl.extractor is not None and "detect" in nl.extractor
    assert nl.source_image == "foo.png"


def test_detections_to_netlist_keeps_bboxes() -> None:
    dets = [_det(5, 10, 25, 30, 0.9, "resistor")]
    nl = detections_to_netlist(dets)
    comp = nl.components[0]
    assert comp.bbox is not None
    assert (comp.bbox.x1, comp.bbox.y1, comp.bbox.x2, comp.bbox.y2) == (5, 10, 25, 30)


def test_detections_to_netlist_empty() -> None:
    nl = detections_to_netlist([])
    assert nl.components == []
    assert nl.nets == []


def test_detections_to_netlist_other_label_gets_x_prefix() -> None:
    dets = [_det(0, 0, 10, 10, 0.9, "some unrecognized thing")]
    nl = detections_to_netlist(dets)
    assert nl.components[0].id == "X1"
    assert nl.components[0].type == "other"


# ---------------------------------------------------------------------------
# Pipeline with a fake detector
# ---------------------------------------------------------------------------


_CallRecord = tuple[str | Path, tuple[str, ...], float, float]


class _FakeDetector:
    model_name = "fake/grounding-dino"

    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections
        self.calls: list[_CallRecord] = []

    def detect(
        self,
        image: str | Path,
        classes: tuple[str, ...] | list[str] = (),
        *,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        self.calls.append((image, tuple(classes), box_threshold, text_threshold))
        return list(self._detections)


def test_pipeline_runs_detector_and_returns_netlist(tmp_path: Path) -> None:
    img = tmp_path / "fake.png"
    img.write_bytes(b"")  # the fake detector doesn't actually read the file

    dets = [
        _det(0, 0, 10, 10, 0.9, "resistor"),
        _det(20, 0, 30, 10, 0.7, "capacitor"),
    ]
    detector = _FakeDetector(dets)
    pipeline = DetectionPipeline(detector=detector)
    nl = pipeline.run(img)

    assert len(detector.calls) == 1
    assert detector.calls[0][0] == img
    assert len(nl.components) == 2


def test_pipeline_applies_nms(tmp_path: Path) -> None:
    img = tmp_path / "fake.png"
    img.write_bytes(b"")

    # Two overlapping boxes at the same position, different labels
    dets = [
        _det(0, 0, 10, 10, 0.9, "resistor"),
        _det(1, 1, 11, 11, 0.7, "integrated circuit"),
    ]
    detector = _FakeDetector(dets)
    pipeline = DetectionPipeline(detector=detector, nms_iou=0.3)
    nl = pipeline.run(img)

    assert len(nl.components) == 1


def test_pipeline_nms_disabled(tmp_path: Path) -> None:
    img = tmp_path / "fake.png"
    img.write_bytes(b"")

    dets = [
        _det(0, 0, 10, 10, 0.9, "resistor"),
        _det(1, 1, 11, 11, 0.7, "resistor"),
    ]
    detector = _FakeDetector(dets)
    pipeline = DetectionPipeline(detector=detector, nms_iou=None)
    nl = pipeline.run(img)

    assert len(nl.components) == 2
