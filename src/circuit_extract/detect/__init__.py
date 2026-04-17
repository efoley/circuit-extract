"""Component detection backends for Approach 2.

We start with **zero-shot open-vocabulary detection** (Grounding DINO), since
no fine-tuned weights exist for the realistic KiCad renderings in
``bshada/open-schematics`` and fine-tuning requires a training pipeline we
don't have yet. Fine-tuned YOLO is a separate follow-up.
"""

from circuit_extract.detect.grounding_dino import (
    DEFAULT_CLASSES,
    Detection,
    GroundingDinoDetector,
)
from circuit_extract.detect.pipeline import (
    DetectionPipeline,
    classify_detection_label,
    extract_netlist_via_detection,
)

__all__ = [
    "DEFAULT_CLASSES",
    "Detection",
    "DetectionPipeline",
    "GroundingDinoDetector",
    "classify_detection_label",
    "extract_netlist_via_detection",
]
