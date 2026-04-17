"""Tests for the overlay / visualisation module."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from circuit_extract.overlay import (
    YOLO_CLASS_NAMES,
    YoloAnnotation,
    load_yolo_annotations,
    overlay_netlist,
    overlay_yolo,
    side_by_side,
)
from circuit_extract.schema import BBox, Component, Net, Netlist, PinRef


def _blank_image(tmp_path: Path, size: tuple[int, int] = (700, 364)) -> Path:
    img = Image.new("RGB", size, (255, 255, 255))
    p = tmp_path / "schematic.png"
    img.save(p)
    return p


# ---------------------------------------------------------------------------
# YOLO annotation loader
# ---------------------------------------------------------------------------


SAMPLE_YOLO = """\
5 0.5 0.5 0.2 0.1 R0
11 0.25 0.75 0.1 0.1 R90
0 0.9 0.9 0.05 0.05 R0
"""


def test_yolo_loader_parses_pixel_coords(tmp_path: Path) -> None:
    f = tmp_path / "circuit0.txt"
    f.write_text(SAMPLE_YOLO)
    anns = load_yolo_annotations(f, image_size=(700, 364))
    assert len(anns) == 3

    r = anns[0]
    assert r.class_id == 5
    assert r.class_name == "resistor"
    assert r.orientation == "R0"
    # cx=0.5*700=350, cy=0.5*364=182, w=140, h=36
    # so bbox = (280, 164, 420, 200)
    assert r.x1 == 280
    assert r.y1 == 164
    assert r.x2 == 420
    assert r.y2 == 200


def test_yolo_loader_unknown_class(tmp_path: Path) -> None:
    f = tmp_path / "odd.txt"
    f.write_text("99 0.5 0.5 0.1 0.1 R0\n")
    anns = load_yolo_annotations(f, image_size=(100, 100))
    assert anns[0].class_name == "class_99"


def test_yolo_loader_missing_orientation(tmp_path: Path) -> None:
    """Orientation is optional; default to R0."""
    f = tmp_path / "minimal.txt"
    f.write_text("5 0.5 0.5 0.1 0.1\n")
    anns = load_yolo_annotations(f, image_size=(100, 100))
    assert anns[0].orientation == "R0"


def test_yolo_class_names_cover_documented_classes() -> None:
    # Sanity check that our class list matches the dataset's comoponent_id.txt
    assert YOLO_CLASS_NAMES[0] == "gnd"
    assert YOLO_CLASS_NAMES[5] == "resistor"
    assert YOLO_CLASS_NAMES[16] == "tgate"


def test_yolo_annotation_center_properties() -> None:
    ann = YoloAnnotation(
        class_id=5, class_name="resistor", orientation="R0", x1=10, y1=20, x2=30, y2=40
    )
    assert ann.cx == 20
    assert ann.cy == 30


# ---------------------------------------------------------------------------
# Netlist overlay
# ---------------------------------------------------------------------------


def test_overlay_netlist_with_bboxes(tmp_path: Path) -> None:
    img_path = _blank_image(tmp_path)
    netlist = Netlist(
        components=[
            Component(id="R1", type="resistor", bbox=BBox(x1=100, y1=50, x2=200, y2=150)),
            Component(id="C1", type="capacitor", bbox=BBox(x1=300, y1=100, x2=350, y2=200)),
        ],
        nets=[
            Net(
                name="N1",
                pins=[PinRef(component_id="R1", pin="1"), PinRef(component_id="C1", pin="1")],
            )
        ],
    )
    result = overlay_netlist(img_path, netlist, title="test")
    # Canvas is wider than input only when we add title/footer bars
    assert result.width == 700
    assert result.height >= 364


def test_overlay_netlist_without_bboxes_lists_in_footer(tmp_path: Path) -> None:
    img_path = _blank_image(tmp_path)
    netlist = Netlist(
        components=[
            Component(id="R1", type="resistor"),  # no bbox
            Component(id="R2", type="resistor"),
        ],
        nets=[],
    )
    result = overlay_netlist(img_path, netlist)
    # Footer adds rows below the image
    assert result.height > 364


def test_overlay_netlist_empty(tmp_path: Path) -> None:
    img_path = _blank_image(tmp_path)
    netlist = Netlist(components=[], nets=[])
    result = overlay_netlist(img_path, netlist)
    assert result.width == 700


# ---------------------------------------------------------------------------
# YOLO overlay
# ---------------------------------------------------------------------------


def test_overlay_yolo(tmp_path: Path) -> None:
    img_path = _blank_image(tmp_path)
    anns = [
        YoloAnnotation(
            class_id=5, class_name="resistor", orientation="R0", x1=10, y1=20, x2=50, y2=60
        ),
    ]
    result = overlay_yolo(img_path, anns, title="gt")
    assert result.width == 700


# ---------------------------------------------------------------------------
# Side-by-side
# ---------------------------------------------------------------------------


def test_side_by_side() -> None:
    left = Image.new("RGB", (100, 50), (255, 0, 0))
    right = Image.new("RGB", (200, 50), (0, 255, 0))
    combined = side_by_side(left, right, gap=10)
    assert combined.width == 310
    assert combined.height == 50


def test_side_by_side_unequal_heights() -> None:
    left = Image.new("RGB", (100, 50), (255, 0, 0))
    right = Image.new("RGB", (100, 100), (0, 255, 0))
    combined = side_by_side(left, right, gap=0)
    assert combined.width == 200
    assert combined.height == 100


# ---------------------------------------------------------------------------
# Skip test that exercises font loading if PIL default fonts are unavailable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(50, 50), (1024, 768)])
def test_overlay_netlist_various_sizes(tmp_path: Path, size: tuple[int, int]) -> None:
    img_path = _blank_image(tmp_path, size=size)
    netlist = Netlist(
        components=[Component(id="R1", type="resistor", bbox=BBox(x1=5, y1=5, x2=20, y2=20))],
        nets=[],
    )
    result = overlay_netlist(img_path, netlist)
    assert result.width == size[0]
