"""Visualisation helpers for inspecting circuit extractions.

Draws bounding boxes + labels onto schematic images so failure modes are
eyeball-inspectable. Supports:

- Predicted netlists from our VLM pipeline (bbox coordinates come from
  :class:`~circuit_extract.schema.Component.bbox`).
- Ground-truth YOLO annotations from the ``hanky2397/schematic_images``
  dataset's ``components.zip``.
- Side-by-side pred vs. GT for direct comparison.
"""

from circuit_extract.overlay.annotations import (
    YOLO_CLASS_NAMES,
    YoloAnnotation,
    load_yolo_annotations,
)
from circuit_extract.overlay.render import overlay_netlist, overlay_yolo, side_by_side

__all__ = [
    "YOLO_CLASS_NAMES",
    "YoloAnnotation",
    "load_yolo_annotations",
    "overlay_netlist",
    "overlay_yolo",
    "side_by_side",
]
