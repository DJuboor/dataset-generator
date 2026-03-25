"""OpenAI-compatible provider. Works with OpenAI, vLLM, Ollama, LiteLLM, Together, Groq."""

import json
import logging
from typing import Any

import openai

from dataset_generator.providers.base import CompletionResult

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """Provider using any OpenAI-compatible API."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str = ""):
        self.model = model
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key or "no-key")

    def _extract_usage(self, response) -> dict[str, int | None]:
        """Extract token usage from API response."""
        usage = getattr(response, "usage", None)
        if usage:
            return {
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
            }
        return {"input_tokens": None, "output_tokens": None}

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Generate a text completion."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = self._extract_usage(response)
        return CompletionResult(content=content, model=self.model, **usage)

    def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON completion with response_format enforcement."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
