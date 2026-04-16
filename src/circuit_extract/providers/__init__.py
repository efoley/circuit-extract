"""VLM provider abstraction.

A provider knows how to send (text + image) prompts to a multimodal model and
return either free-form text or a JSON object validated against a pydantic
schema. The goal is to keep approach-specific code (Gemini, Claude, OpenAI,
local) out of the extraction pipeline so we can swap models freely.
"""

from typing import Any

from circuit_extract.providers.base import VLMProvider, VLMResponse
from circuit_extract.providers.gemini import GeminiProvider

__all__ = ["GeminiProvider", "VLMProvider", "VLMResponse", "get_provider"]


def get_provider(name: str, **kwargs: Any) -> VLMProvider:
    """Look up a provider by short name.

    Currently only ``gemini`` is implemented; this is the seam where we'll wire
    in Claude / OpenAI / local providers later. ``kwargs`` are forwarded to the
    provider's constructor (e.g. ``model``, ``api_key``).
    """
    name = name.lower()
    if name in ("gemini", "google"):
        return GeminiProvider(**kwargs)
    raise ValueError(f"Unknown VLM provider: {name!r}")
