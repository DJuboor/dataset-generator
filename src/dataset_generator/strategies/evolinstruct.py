"""EvolInstruct strategy — WizardLM-style progressive example evolution."""

from __future__ import annotations

EVOLUTION_PROMPTS: list[str] = [
    "Add constraints or requirements to make this more challenging",
    "Increase the reasoning depth required",
    "Combine with another topic or domain",
    "Make more specific and concrete with real-world details",
    "Rewrite for a different audience or expertise level",
    "Add multi-step dependencies between parts",
    "Introduce edge cases or special conditions",
]


class EvolInstructStrategy:
    """Generates progressively more complex examples via evolution prompts."""

    def __init__(self, evolution_rounds: int = 1) -> None:
        self.evolution_rounds = evolution_rounds

    def apply(self, messages: list[dict[str, str]], batch_index: int) -> list[dict[str, str]]:
        """Inject evolution instructions that increase in complexity across batches."""
        modified = [m.copy() for m in messages]

        # Stack multiple evolution prompts based on rounds
        evolutions: list[str] = []
        for r in range(self.evolution_rounds):
            idx = (batch_index + r) % len(EVOLUTION_PROMPTS)
            evolutions.append(EVOLUTION_PROMPTS[idx])

        evolution_text = "\n".join(f"- {e}" for e in evolutions)

        modified[-1] = {
            **modified[-1],
            "content": modified[-1]["content"]
            + "\n\nEvolve the complexity of the generated examples. "
            "Apply the following evolution instructions:\n"
            + evolution_text
            + "\n\nThe resulting examples should be noticeably more sophisticated "
            "and challenging than basic examples for this task.",
        }
        return modified
