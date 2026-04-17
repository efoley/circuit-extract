"""Detection-based extraction pipeline.

Takes raw :class:`Detection` results from an open-vocabulary detector and
turns them into a :class:`~circuit_extract.schema.Netlist`. Because
detection alone produces no refdes, values, pins, or nets, the resulting
netlist has:

- Auto-generated ids (``R1`` / ``R2`` / ``C1`` / ... per canonical type)
- Pixel bboxes
- No values
- No pins, no nets

That's still enough to compute component P/R/F1 and to visualise overlays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from circuit_extract.detect.grounding_dino import (
    DEFAULT_CLASSES,
    Detection,
    GroundingDinoDetector,
)
from circuit_extract.schema import BBox, Component, ComponentType, Netlist


class _Detector(Protocol):
    """Structural interface for any open-vocabulary detector.

    Lets us swap in fake detectors in tests or different backends later
    without touching :class:`DetectionPipeline`.
    """

    model_name: str

    def detect(
        self,
        image: str | Path,
        classes: tuple[str, ...] | list[str] = ...,
        *,
        box_threshold: float = ...,
        text_threshold: float = ...,
    ) -> list[Detection]: ...


logger = logging.getLogger(__name__)


# Map detector text labels to our canonical ComponentType. Substring-matched
# case-insensitively; first hit wins (order matters — specific before generic).
_LABEL_KEYWORDS: list[tuple[str, ComponentType]] = [
    # Keep this aligned with DEFAULT_CLASSES in grounding_dino.py.
    ("light emitting diode", "led"),
    ("zener", "zener"),
    ("led", "led"),
    ("diode", "diode"),
    ("opamp", "opamp"),
    ("operational amplifier", "opamp"),
    ("mosfet", "nmos"),
    ("nmos", "nmos"),
    ("pmos", "pmos"),
    ("npn", "bjt_npn"),
    ("pnp", "bjt_pnp"),
    ("bjt", "bjt_npn"),
    ("transistor", "bjt_npn"),  # fallback for unspecified
    ("integrated circuit", "ic"),
    ("microcontroller", "ic"),
    ("chip", "ic"),
    ("resistor", "resistor"),
    ("capacitor", "capacitor"),
    ("inductor", "inductor"),
    ("transformer", "transformer"),
    ("crystal", "crystal"),
    ("fuse", "fuse"),
    ("switch", "switch"),
    ("button", "switch"),
    ("connector", "connector"),
    ("header", "connector"),
    ("usb", "connector"),
    ("jack", "connector"),
    ("battery", "battery"),
    ("ground", "ground"),
    ("vcc", "vcc"),
]

# Reference-designator prefix per canonical type.
_REFDES_PREFIX: dict[ComponentType, str] = {
    "resistor": "R",
    "capacitor": "C",
    "inductor": "L",
    "diode": "D",
    "led": "D",
    "zener": "D",
    "bjt_npn": "Q",
    "bjt_pnp": "Q",
    "nmos": "M",
    "pmos": "M",
    "jfet": "J",
    "opamp": "U",
    "ic": "U",
    "switch": "SW",
    "voltage_source": "V",
    "current_source": "I",
    "ac_source": "V",
    "battery": "BT",
    "ground": "GND",
    "vcc": "VCC",
    "node_label": "N",
    "transformer": "T",
    "crystal": "Y",
    "fuse": "F",
    "connector": "J",
    "other": "X",
}


def classify_detection_label(label: str) -> ComponentType:
    """Map a detector label (free text) to our canonical :class:`ComponentType`."""
    needle = label.lower()
    for key, ctype in _LABEL_KEYWORDS:
        if key in needle:
            return ctype
    return "other"


def _nms(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """Greedy non-maximum suppression across all classes.

    Grounding DINO returns overlapping boxes when a single component matches
    multiple prompt classes (e.g. "resistor" and "integrated circuit" both
    fire on a chip). We collapse overlaps keeping the highest-scoring box.
    """
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for det in ordered:
        if any(_iou(det, k) > iou_threshold for k in kept):
            continue
        kept.append(det)
    return kept


def _iou(a: Detection, b: Detection) -> float:
    """Intersection-over-union of two axis-aligned boxes."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    area_b = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def detections_to_netlist(
    detections: list[Detection],
    *,
    source_image: str | None = None,
    model_name: str = "grounding-dino",
) -> Netlist:
    """Convert detection results to a :class:`Netlist` with auto refdes."""
    # Per-type counters so refdes come out consistent (R1, R2, ..., C1, C2, ...)
    type_counters: dict[ComponentType, int] = {}
    components: list[Component] = []

    for det in detections:
        ctype = classify_detection_label(det.label)
        prefix = _REFDES_PREFIX.get(ctype, "X")
        type_counters[ctype] = type_counters.get(ctype, 0) + 1
        idx = type_counters[ctype]
        components.append(
            Component(
                id=f"{prefix}{idx}",
                type=ctype,
                value=None,
                pins=[],
                bbox=BBox(x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2),
                notes=f"detector_label={det.label} score={det.score:.2f}",
            )
        )

    return Netlist(
        components=components,
        nets=[],
        source_image=source_image,
        extractor=f"detect:{model_name}",
    )


@dataclass
class DetectionPipeline:
    """End-to-end detect-only extraction.

    Parameters
    ----------
    detector:
        Any detector exposing ``detect(image, classes=...) -> list[Detection]``.
        Defaults to a lazily-constructed :class:`GroundingDinoDetector`.
    classes:
        Prompt classes passed to the detector.
    box_threshold, text_threshold:
        Detector thresholds.
    nms_iou:
        Cross-class NMS IoU threshold. Set to ``None`` to disable NMS.
    """

    detector: _Detector | None = None
    classes: tuple[str, ...] = DEFAULT_CLASSES
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    nms_iou: float | None = 0.5

    _raw_detections: list[Detection] = field(default_factory=list, init=False, repr=False)

    def _ensure_detector(self) -> _Detector:
        if self.detector is None:
            self.detector = GroundingDinoDetector()
        return self.detector

    def run(self, image: str | Path) -> Netlist:
        image_path = Path(image)
        det = self._ensure_detector()

        logger.info("running detector on %s...", image_path.name)
        raw = det.detect(
            image_path,
            classes=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        logger.info("  -> %d raw detections", len(raw))

        if self.nms_iou is not None:
            before = len(raw)
            raw = _nms(raw, iou_threshold=self.nms_iou)
            logger.info("  -> %d after NMS (from %d)", len(raw), before)

        self._raw_detections = raw
        return detections_to_netlist(
            raw,
            source_image=str(image_path),
            model_name=det.model_name.split("/")[-1],
        )


def extract_netlist_via_detection(
    image: str | Path,
    *,
    model: str = "IDEA-Research/grounding-dino-tiny",
    classes: tuple[str, ...] = DEFAULT_CLASSES,
) -> Netlist:
    """Convenience: build a detector + pipeline and run it once."""
    detector = GroundingDinoDetector(model=model)
    return DetectionPipeline(detector=detector, classes=classes).run(image)
