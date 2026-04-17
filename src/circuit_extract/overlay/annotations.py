"""Ground-truth YOLO annotation loader for ``hanky2397/schematic_images``.

Each annotation file is one component per line:

    <class_id> <cx> <cy> <w> <h> <orientation>

where ``cx cy w h`` are normalised to ``[0, 1]`` relative to the image size,
and ``orientation`` is one of ``R0``, ``R90``, ``R180``, ``R270``, ``MX``, ``MY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Class IDs come from the dataset's comoponent_id.txt (typo'd upstream).
# Order is significant — index = class id.
YOLO_CLASS_NAMES: list[str] = [
    "gnd",
    "pmos",
    "nmos",
    "pnp",
    "npn",
    "resistor",
    "capacitor",
    "voltage",
    "current",
    "diode",
    "inductor",
    "and",
    "xor",
    "inverter",
    "dflipflop",
    "opamp",
    "tgate",
]


@dataclass(frozen=True)
class YoloAnnotation:
    """One YOLO-format component annotation resolved to pixel coordinates."""

    class_id: int
    class_name: str
    orientation: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2


def load_yolo_annotations(
    annotation_path: str | Path,
    image_size: tuple[int, int],
) -> list[YoloAnnotation]:
    """Parse a YOLO annotation file into pixel-space bboxes.

    Parameters
    ----------
    annotation_path:
        Path to a ``.txt`` file (one component per line).
    image_size:
        ``(width, height)`` of the source image, used to convert normalised
        coordinates to pixels.
    """
    width, height = image_size
    results: list[YoloAnnotation] = []

    for raw_line in Path(annotation_path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        cx_norm = float(parts[1])
        cy_norm = float(parts[2])
        w_norm = float(parts[3])
        h_norm = float(parts[4])
        orientation = parts[5] if len(parts) >= 6 else "R0"

        cx_px = cx_norm * width
        cy_px = cy_norm * height
        w_px = w_norm * width
        h_px = h_norm * height

        x1 = max(0, int(round(cx_px - w_px / 2)))
        y1 = max(0, int(round(cy_px - h_px / 2)))
        x2 = min(width, int(round(cx_px + w_px / 2)))
        y2 = min(height, int(round(cy_px + h_px / 2)))

        class_name = (
            YOLO_CLASS_NAMES[class_id]
            if 0 <= class_id < len(YOLO_CLASS_NAMES)
            else f"class_{class_id}"
        )

        results.append(
            YoloAnnotation(
                class_id=class_id,
                class_name=class_name,
                orientation=orientation,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )

    return results
