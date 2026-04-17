"""Zero-shot object detection on schematics via Grounding DINO.

Grounding DINO is a text-conditioned detector: you pass an image plus a
period-separated prompt of class names, and it returns bounding boxes with
class labels drawn from that prompt. No training data required.

We use ``IDEA-Research/grounding-dino-tiny`` by default (~170 MB download).
Swap to ``-base`` for better accuracy at higher cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"


# Default prompt: what we're looking for on a circuit schematic.
# Grounding DINO expects single-word-ish, lowercase, period-separated terms.
DEFAULT_CLASSES: tuple[str, ...] = (
    "resistor",
    "capacitor",
    "inductor",
    "diode",
    "light emitting diode",
    "transistor",
    "integrated circuit",
    "connector",
    "switch",
    "battery",
    "fuse",
    "transformer",
    "crystal",
)


def _build_prompt(classes: tuple[str, ...] | list[str]) -> str:
    """Build a Grounding DINO prompt string from a list of class names.

    Format: lowercase, period-and-space separated, trailing period.
    """
    return ". ".join(c.strip().lower() for c in classes) + "."


@dataclass(frozen=True)
class Detection:
    """One detected object in an image."""

    label: str  # The text label the detector returned (e.g. "resistor")
    score: float  # Confidence [0, 1]
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class GroundingDinoDetector:
    """Open-vocabulary detector wrapping HuggingFace's Grounding DINO."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        # Imported lazily so installing without the [detect] extra doesn't
        # crash the rest of the package.
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        self.model_name = model
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("loading Grounding DINO: %s (device=%s)", model, device)
        # Typed as Any — transformers' HF Auto* classes have dynamic __call__
        # and post_process_* methods that static checkers can't see through.
        self._processor: Any = AutoProcessor.from_pretrained(model)
        self._model: Any = AutoModelForZeroShotObjectDetection.from_pretrained(model).to(device)
        self._model.eval()

    def detect(
        self,
        image: str | Path | Image.Image,
        classes: tuple[str, ...] | list[str] = DEFAULT_CLASSES,
        *,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        """Run detection and return a list of :class:`Detection` results.

        Parameters
        ----------
        image:
            Path or PIL image.
        classes:
            Candidate class names. Passed to the model as a period-separated
            prompt. The detector may return labels that are substrings or
            concatenations of these names (a quirk of text-conditioned
            detectors); downstream code should match loosely.
        box_threshold:
            Minimum objectness score for a box to be kept.
        text_threshold:
            Minimum text-alignment score for the prompt match.
        """
        import torch  # type: ignore[import-not-found]

        img = _open_image(image)
        prompt = _build_prompt(classes)

        inputs = self._processor(images=img, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results_list = self._postprocess(
            outputs=outputs,
            inputs=inputs,
            image_size=img.size,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        detections: list[Detection] = []
        for res in results_list:
            boxes = res["boxes"].tolist()
            scores = res["scores"].tolist()
            labels = res.get("labels") or res.get("text_labels") or []
            for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels, strict=False):
                detections.append(
                    Detection(
                        label=str(label).strip().lower(),
                        score=float(score),
                        x1=max(0, int(round(x1))),
                        y1=max(0, int(round(y1))),
                        x2=int(round(x2)),
                        y2=int(round(y2)),
                    )
                )
        return detections

    def _postprocess(
        self,
        *,
        outputs: Any,
        inputs: Any,
        image_size: tuple[int, int],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict[str, Any]]:
        """Run the model-specific post-processing.

        The Grounding DINO processor exposes ``post_process_grounded_object_detection``;
        some transformers releases renamed it.
        """
        width, height = image_size
        target_sizes = [(height, width)]
        processor: Any = self._processor
        # The post-processor's keyword changed from `box_threshold` → `threshold`
        # around transformers 5.0. Try the new name first, fall back for older
        # installs.
        fn = processor.post_process_grounded_object_detection
        try:
            return fn(
                outputs=outputs,
                input_ids=inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            return fn(
                outputs=outputs,
                input_ids=inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )


def _open_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")
