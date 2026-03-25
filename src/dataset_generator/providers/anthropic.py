"""Anthropic provider using the native anthropic SDK."""

import json
import logging
from typing import Any

from dataset_generator.providers.base import CompletionResult

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]


class AnthropicProvider:
    """Provider using the Anthropic Messages API.

    Handles the format differences from OpenAI: system prompt is a top-level
    parameter, not a message in the array. max_tokens is required.
    """

    def __init__(
        self,
        model: str,
        api_key: str = "",
        timeout: float = 600.0,
        max_tokens: int = 8192,
    ):
        if anthropic is None:
            raise ImportError("Install anthropic extra: uv add 'dataset-generator[anthropic]'")
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key or None, timeout=timeout)

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Generate a text completion via the Anthropic Messages API."""
        system, user_messages = _split_system(messages)

        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=user_messages,
            temperature=temperature,
            max_tokens=max_tokens or self.max_tokens,
        )

        content = response.content[0].text if response.content else ""
        return CompletionResult(
            content=content,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON completion. Appends a JSON instruction to the system prompt."""
        # Anthropic doesn't have response_format — nudge via system prompt
        system, user_messages = _split_system(messages)
        if system and "JSON" not in system:
            system += "\n\nRespond with valid JSON only."

        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=user_messages,
            temperature=temperature,
            max_tokens=max_tokens or self.max_tokens,
        )

        content = response.content[0].text if response.content else "{}"
        return json.loads(content)


def _split_system(
    messages: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """Split OpenAI-format messages into Anthropic system + messages.

    Anthropic requires system as a separate parameter, not in the messages array.
    """
    system = ""
    user_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            user_messages.append(msg)
    return system, user_messages
