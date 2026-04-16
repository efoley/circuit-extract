"""Gemini implementation of :class:`VLMProvider`.

Uses the modern ``google-genai`` SDK (``google.genai``), which natively
supports pydantic models as the structured-output schema. The default model is
Gemini 3 Flash (preview).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from PIL import Image
from pydantic import BaseModel

from circuit_extract.providers.base import ImageInput, VLMProvider, VLMResponse

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-3-flash-preview"


def _load_image(img: ImageInput) -> Image.Image:
    """Coerce an :data:`ImageInput` into a PIL ``Image``.

    The google-genai SDK accepts ``PIL.Image.Image`` objects directly in the
    ``contents`` list, which keeps this layer free of base64 encoding noise.
    """
    if isinstance(img, (str, Path)):
        return Image.open(img)
    if isinstance(img, bytes):
        from io import BytesIO

        return Image.open(BytesIO(img))
    raise TypeError(f"Unsupported image input type: {type(img)!r}")


class GeminiProvider(VLMProvider):
    """Multimodal provider backed by Google's Gemini API."""

    name = "gemini"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        # Imported lazily so test/dev environments without the SDK installed
        # can still import the rest of the package.
        from google import genai

        self.model = model
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GeminiProvider requires an API key. Set GEMINI_API_KEY (or pass api_key=...)."
            )
        self._client = genai.Client(api_key=key)

    # ------------------------------------------------------------------ helpers

    def _build_contents(self, prompt: str, images: list[ImageInput] | None) -> list[Any]:
        contents: list[Any] = []
        if images:
            contents.extend(_load_image(img) for img in images)
        contents.append(prompt)
        return contents

    def _build_config(
        self,
        *,
        system: str | None,
        temperature: float,
        response_schema: type[BaseModel] | None = None,
    ) -> Any:
        from google.genai import types

        kwargs: dict[str, Any] = {"temperature": temperature}
        if system is not None:
            kwargs["system_instruction"] = system
        if response_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_schema
        return types.GenerateContentConfig(**kwargs)

    def _usage(self, response: Any) -> tuple[int | None, int | None]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return None, None
        return (
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )

    # ----------------------------------------------------------------- public

    def generate_text(
        self,
        prompt: str,
        images: list[ImageInput] | None = None,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> VLMResponse:
        response = self._client.models.generate_content(
            model=self.model,
            contents=self._build_contents(prompt, images),
            config=self._build_config(system=system, temperature=temperature),
        )
        in_tok, out_tok = self._usage(response)
        return VLMResponse(
            text=response.text or "",
            model=self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    def generate_json(
        self,
        prompt: str,
        schema: type[T],
        images: list[ImageInput] | None = None,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        response = self._client.models.generate_content(
            model=self.model,
            contents=self._build_contents(prompt, images),
            config=self._build_config(
                system=system, temperature=temperature, response_schema=schema
            ),
        )
        # The SDK exposes the parsed pydantic instance directly when a
        # response_schema is supplied.
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        # Fallback: parse the raw JSON text ourselves.
        return schema.model_validate_json(response.text or "")
