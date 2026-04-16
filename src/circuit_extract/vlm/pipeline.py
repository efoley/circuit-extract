"""Multi-step VLM extraction pipeline.

The pipeline runs the prompts in :mod:`circuit_extract.vlm.prompts` against a
provider-agnostic :class:`VLMProvider`. Each stage's output is a pydantic
model so we can inspect, log, or persist intermediate state.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from circuit_extract.providers.base import VLMProvider
from circuit_extract.schema import Component, Netlist
from circuit_extract.vlm.prompts import (
    COMPONENTS_PROMPT,
    NETS_PROMPT,
    ONESHOT_PROMPT,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class _ComponentList(BaseModel):
    """Intermediate schema for stage 1 (components only, no nets)."""

    components: list[Component]


@dataclass
class StageResult:
    """Captured output of a single pipeline stage."""

    name: str
    payload: BaseModel


@dataclass
class VLMExtractionPipeline:
    """End-to-end VLM extraction.

    Parameters
    ----------
    provider:
        Any :class:`VLMProvider` implementation.
    multi_step:
        If True (default), run the components-then-nets two-stage pipeline.
        If False, run a single-shot prompt as a baseline for comparison.
    """

    provider: VLMProvider
    multi_step: bool = True
    stages: list[StageResult] = field(default_factory=list)

    def run(self, image: str | Path) -> Netlist:
        image_path = Path(image)
        self.stages.clear()

        if not self.multi_step:
            logger.info("stage 1/1 (oneshot): calling VLM...")
            t0 = time.perf_counter()
            netlist = self.provider.generate_json(
                ONESHOT_PROMPT,
                schema=Netlist,
                images=[image_path],
                system=SYSTEM_PROMPT,
            )
            logger.info(
                "  -> %d components, %d nets (%.1fs)",
                len(netlist.components),
                len(netlist.nets),
                time.perf_counter() - t0,
            )
            self.stages.append(StageResult("oneshot", netlist))
            return self._finalise(netlist, image_path)

        # Stage 1: components
        logger.info("stage 1/2: extracting components...")
        t0 = time.perf_counter()
        comp_result = self.provider.generate_json(
            COMPONENTS_PROMPT,
            schema=_ComponentList,
            images=[image_path],
            system=SYSTEM_PROMPT,
        )
        logger.info(
            "  -> %d components (%.1fs)",
            len(comp_result.components),
            time.perf_counter() - t0,
        )
        self.stages.append(StageResult("components", comp_result))

        # Stage 2: nets, conditioned on the components from stage 1
        logger.info("stage 2/2: tracing nets...")
        t1 = time.perf_counter()
        components_json = _ComponentList(components=comp_result.components).model_dump_json(
            indent=2
        )
        nets_prompt = NETS_PROMPT.format(components_json=components_json)
        netlist = self.provider.generate_json(
            nets_prompt,
            schema=Netlist,
            images=[image_path],
            system=SYSTEM_PROMPT,
        )
        logger.info("  -> %d nets (%.1fs)", len(netlist.nets), time.perf_counter() - t1)
        self.stages.append(StageResult("nets", netlist))

        return self._finalise(netlist, image_path)

    def _finalise(self, netlist: Netlist, image_path: Path) -> Netlist:
        netlist.source_image = str(image_path)
        netlist.extractor = f"vlm:{self.provider.name}:{self.provider.model}"
        return netlist


def extract_netlist(
    image: str | Path,
    provider: VLMProvider,
    *,
    multi_step: bool = True,
) -> Netlist:
    """Convenience function: build a pipeline and run it once."""
    return VLMExtractionPipeline(provider=provider, multi_step=multi_step).run(image)
