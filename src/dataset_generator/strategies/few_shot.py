"""Few-shot strategy — injects example samples into the prompt."""

import json


class FewShotStrategy:
    """Adds few-shot examples to improve output quality and format adherence."""

    def __init__(self, examples: list[dict] | None = None):
        self.examples = examples or []

    def apply(self, messages: list[dict[str, str]], batch_index: int) -> list[dict[str, str]]:
        if not self.examples:
            # No examples — fall back to direct strategy behavior inline
            if batch_index == 0:
                return messages
            modified = [m.copy() for m in messages]
            modified[-1] = {
                **modified[-1],
                "content": modified[-1]["content"]
                + f"\n\nThis is batch {batch_index + 1}. Generate completely different examples "
                "from any previous batches. Use different vocabulary, topics, and phrasing.",
            }
            return modified

        modified = [m.copy() for m in messages]
        example_text = "\n\nHere are some example outputs for reference:\n"
        for ex in self.examples[:3]:
            example_text += f"- {json.dumps(ex) if isinstance(ex, dict) else ex}\n"
        example_text += (
            "\nGenerate new examples that are similar in quality but different in content."
        )

        modified[-1] = {
            **modified[-1],
            "content": modified[-1]["content"] + example_text,
        }

        if batch_index > 0:
            modified[-1]["content"] += (
                f"\n\nBatch {batch_index + 1}: vary your examples significantly."
            )
        return modified
