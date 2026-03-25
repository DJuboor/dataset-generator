"""Direct strategy — passes messages through with minor variation."""


class DirectStrategy:
    """No transformation. Adds batch index hint for diversity."""

    def apply(self, messages: list[dict[str, str]], batch_index: int) -> list[dict[str, str]]:
        if batch_index == 0:
            return messages
        # Add a diversity hint on subsequent batches
        modified = [m.copy() for m in messages]
        modified[-1] = {
            **modified[-1],
            "content": modified[-1]["content"]
            + f"\n\nThis is batch {batch_index + 1}. Generate completely different examples "
            "from any previous batches. Use different vocabulary, topics, and phrasing.",
        }
        return modified
