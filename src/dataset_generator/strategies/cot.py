"""Chain-of-thought strategy — injects step-by-step reasoning instructions."""

from __future__ import annotations

REASONING_PROMPTS: dict[str, str] = {
    "brief": (
        "Before generating each example, briefly outline your reasoning "
        "in 1-2 sentences, then produce the example."
    ),
    "moderate": (
        "Before generating each example, think step-by-step:\n"
        "1. Consider what makes a good example for this task\n"
        "2. Identify the key characteristics and constraints\n"
        "3. Plan the example content\n"
        "Then generate the example."
    ),
    "detailed": (
        "Before generating each example, perform detailed reasoning:\n"
        "1. Analyze the task requirements and constraints thoroughly\n"
        "2. Consider multiple possible approaches and pick the strongest\n"
        "3. Identify potential pitfalls or quality issues to avoid\n"
        "4. Plan the structure, tone, and content carefully\n"
        "5. Verify the example meets all criteria before finalizing\n"
        "Then generate the example."
    ),
}


class ChainOfThoughtStrategy:
    """Adds chain-of-thought reasoning instructions to improve example quality."""

    def __init__(self, reasoning_depth: str = "moderate") -> None:
        if reasoning_depth not in REASONING_PROMPTS:
            raise ValueError(
                f"Unknown reasoning_depth: {reasoning_depth}. Available: {list(REASONING_PROMPTS)}"
            )
        self.reasoning_depth = reasoning_depth

    def apply(self, messages: list[dict[str, str]], batch_index: int) -> list[dict[str, str]]:
        """Inject chain-of-thought instructions, with diversity hint on later batches."""
        modified = [m.copy() for m in messages]
        cot_instruction = REASONING_PROMPTS[self.reasoning_depth]
        suffix = f"\n\n{cot_instruction}"

        if batch_index > 0:
            suffix += (
                f"\n\nThis is batch {batch_index + 1}. Generate completely different examples "
                "from any previous batches. Use different vocabulary, topics, and phrasing."
            )

        modified[-1] = {
            **modified[-1],
            "content": modified[-1]["content"] + suffix,
        }
        return modified
