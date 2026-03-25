"""Adversarial strategy — generates hard/tricky examples near decision boundaries."""

from __future__ import annotations

DEFAULT_MODES: list[str] = [
    "ambiguous examples near decision boundaries",
    "examples with misleading surface features",
    "examples requiring world knowledge",
    "examples with negation or double negation",
    "examples with sarcasm or irony",
    "edge cases with unusual formatting",
    "examples that challenge common assumptions",
    "examples with subtle errors or inconsistencies",
]


class AdversarialStrategy:
    """Rotates through adversarial modes to generate challenging examples."""

    def __init__(self, modes: list[str] | None = None) -> None:
        self.modes = modes or DEFAULT_MODES

    def apply(self, messages: list[dict[str, str]], batch_index: int) -> list[dict[str, str]]:
        """Inject adversarial mode instruction for the current batch."""
        modified = [m.copy() for m in messages]
        mode = self.modes[batch_index % len(self.modes)]

        modified[-1] = {
            **modified[-1],
            "content": modified[-1]["content"] + f"\n\nGenerate adversarial/challenging examples. "
            f"Focus specifically on: {mode}. "
            "These should be difficult cases that would trip up a naive model — "
            "tricky, realistic, and not easily solvable with simple heuristics.",
        }
        return modified
