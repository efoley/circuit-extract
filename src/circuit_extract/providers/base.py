"""Abstract VLM provider interface.

Concrete providers (Gemini, Claude, OpenAI, ...) implement :class:`VLMProvider`.
The interface is intentionally minimal:

- ``generate_text``  : free-form text completion from text + images.
- ``generate_json``  : structured output validated against a pydantic model.

Anything model-specific (safety settings, thinking budgets, ...) lives behind
provider-specific kwargs at construction time so the pipeline code stays clean.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# An "image input" is either a path on disk or raw bytes (already-loaded PNG/JPEG).
ImageInput = Path | bytes


@dataclass
class VLMResponse:
    """Container for a VLM call's output and minimal metadata."""

    text: str
    model: str
    # Token usage is optional because not every provider exposes it identically.
    input_tokens: int | None = None
    output_tokens: int | None = None


class VLMProvider(ABC):
    """Minimal interface every multimodal provider must implement."""

    name: str
    model: str

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        images: list[ImageInput] | None = None,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> VLMResponse:
        """Run a text + image prompt and return the model's text response."""

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        schema: type[T],
        images: list[ImageInput] | None = None,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        """Run a prompt and parse the result into ``schema``.

        Implementations should use the provider's native structured-output mode
        when available (Gemini ``response_schema``, OpenAI ``response_format``,
        Claude tool-use), and fall back to text-mode + manual JSON parsing
        otherwise.
        """
