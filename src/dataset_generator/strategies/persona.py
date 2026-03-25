"""Persona strategy — generates from different perspectives for diversity."""

PERSONAS = [
    "a technical writer creating documentation examples",
    "a customer writing product reviews",
    "a journalist writing news articles",
    "a student writing academic essays",
    "a social media user posting casual updates",
    "a professional writing business emails",
    "a researcher writing scientific abstracts",
    "a creative writer crafting fiction excerpts",
    "a support agent handling customer tickets",
    "a blogger writing opinion pieces",
]


class PersonaStrategy:
    """Rotates through personas across batches for natural diversity."""

    def __init__(self, personas: list[str] | None = None):
        self.personas = personas or PERSONAS

    def apply(self, messages: list[dict[str, str]], batch_index: int) -> list[dict[str, str]]:
        persona = self.personas[batch_index % len(self.personas)]
        modified = [m.copy() for m in messages]
        modified[-1] = {
            **modified[-1],
            "content": modified[-1]["content"]
            + f"\n\nWrite these examples from the perspective of {persona}. "
            "The writing style and vocabulary should reflect this persona.",
        }
        return modified
