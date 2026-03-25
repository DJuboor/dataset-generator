"""Strategy protocol."""

from typing import Protocol


class Strategy(Protocol):
    """Controls how messages are modified for diversity across batches."""

    def apply(self, messages: list[dict[str, str]], batch_index: int) -> list[dict[str, str]]:
        """Transform messages for a specific batch to increase diversity.

        Args:
            messages: Base messages from the task.
            batch_index: Which batch number this is (for varying prompts).

        Returns:
            Modified messages.
        """
        ...
