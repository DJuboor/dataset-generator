"""Provider protocol — the only interface tasks interact with."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class CompletionResult:
    """Result from a provider completion, including usage stats for cost tracking."""

    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return None


class Provider(Protocol):
    """LLM provider that can generate text from messages."""

    model: str

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Generate a completion from messages.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            CompletionResult with content and optional usage stats.
        """
        ...

    def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON completion with response_format enforcement.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Parsed JSON response.
        """
        ...
