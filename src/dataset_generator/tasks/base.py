"""Base task protocol and sample model."""

from __future__ import annotations

import re
from typing import Any, Protocol

from pydantic import BaseModel


def clean_llm_response(response: str) -> str:
    """Clean common LLM output artifacts before JSON parsing.

    Handles: thinking tags (<think>...</think>), markdown code blocks,
    EOS tokens (<|endoftext|>, <|im_start|>, etc.), and leading/trailing whitespace.
    """
    # Strip thinking/reasoning tags (qwen, deepseek, etc.)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"<reasoning>.*?</reasoning>", "", response, flags=re.DOTALL)

    # Strip EOS / special tokens
    response = re.sub(r"<\|[^>]+\|>.*", "", response, flags=re.DOTALL)

    response = response.strip()

    # Strip markdown code blocks
    if response.startswith("```"):
        response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return response


class Sample(BaseModel):
    """A single generated sample. All tasks produce these.

    Core fields are ``text`` and optional ``label``. Task-specific data
    (entities, chosen/rejected, context, etc.) lives in ``metadata`` and
    is flattened to top-level keys on serialization so output formats are
    clean and task-appropriate.
    """

    text: str
    label: str | None = None
    metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict. Metadata is flattened to top-level keys."""
        d: dict[str, Any] = {"text": self.text}
        if self.label is not None:
            d["label"] = self.label
        # Flatten metadata so output is clean per-task
        d.update(self.metadata)
        return d


class Task(Protocol):
    """Protocol for dataset generation tasks."""

    @classmethod
    def from_config(cls, config: dict) -> Task:
        """Create task from config dict."""
        ...

    def build_messages(self, batch_size: int = 1) -> list[dict[str, str]]:
        """Build LLM messages for generating samples.

        Args:
            batch_size: Number of samples to request per LLM call.

        Returns:
            Messages in OpenAI chat format.
        """
        ...

    def parse_response(self, response: str) -> list[Sample]:
        """Parse LLM response into samples.

        Args:
            response: Raw LLM response text.

        Returns:
            List of parsed samples.
        """
        ...
