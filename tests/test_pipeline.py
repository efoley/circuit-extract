"""Tests for the multi-step VLM pipeline using a fake provider.

These tests do not call any real API. The fake provider returns canned
pydantic objects keyed off the requested schema, which lets us exercise the
pipeline plumbing in isolation from Gemini.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar, cast

from pydantic import BaseModel

from circuit_extract.providers.base import ImageInput, VLMProvider, VLMResponse
from circuit_extract.schema import Component, Net, Netlist, PinRef
from circuit_extract.vlm.pipeline import VLMExtractionPipeline

T = TypeVar("T", bound=BaseModel)


class FakeProvider(VLMProvider):
    name = "fake"
    model = "fake-1"

    def __init__(self, responses: dict[type[BaseModel], BaseModel]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def generate_text(
        self,
        prompt: str,
        images: list[ImageInput] | None = None,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> VLMResponse:
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        schema: type[T],
        images: list[ImageInput] | None = None,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        self.calls.append(schema.__name__)
        return cast(T, self.responses[schema])


def test_multi_step_pipeline_runs_two_stages(tmp_path: Path) -> None:
    img = tmp_path / "fake.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # not a real png, just needs to exist

    components = [
        Component(id="R1", type="resistor", value="1k", pins=["1", "2"]),
        Component(id="GND1", type="ground", pins=["1"]),
    ]
    final = Netlist(
        components=components,
        nets=[
            Net(
                name="N1",
                pins=[
                    PinRef(component_id="R1", pin="1"),
                    PinRef(component_id="R1", pin="2"),
                ],
            ),
            Net(
                name="GND",
                pins=[
                    PinRef(component_id="R1", pin="2"),
                    PinRef(component_id="GND1", pin="1"),
                ],
            ),
        ],
    )

    # Stage 1 returns a _ComponentList; stage 2 returns the full Netlist.
    from circuit_extract.vlm.pipeline import _ComponentList

    provider = FakeProvider(
        {
            _ComponentList: _ComponentList(components=components),
            Netlist: final,
        }
    )

    pipeline = VLMExtractionPipeline(provider=provider, multi_step=True)
    result = pipeline.run(img)

    assert provider.calls == ["_ComponentList", "Netlist"]
    assert [s.name for s in pipeline.stages] == ["components", "nets"]
    assert result.source_image == str(img)
    assert result.extractor == "vlm:fake:fake-1"
    assert len(result.components) == 2
    assert len(result.nets) == 2


def test_oneshot_pipeline_runs_single_stage(tmp_path: Path) -> None:
    img = tmp_path / "fake.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    final = Netlist(components=[], nets=[])
    provider = FakeProvider({Netlist: final})

    pipeline = VLMExtractionPipeline(provider=provider, multi_step=False)
    result = pipeline.run(img)

    assert provider.calls == ["Netlist"]
    assert [s.name for s in pipeline.stages] == ["oneshot"]
    assert result.extractor == "vlm:fake:fake-1"
