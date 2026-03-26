"""Knowledge distillation dataset generation."""

from __future__ import annotations

import json
import logging

from dataset_generator.tasks.base import Sample, clean_llm_response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a dataset generation assistant acting as a teacher model. Generate \
high-quality instruction-response pairs with explicit reasoning, suitable for distilling \
knowledge into smaller student models.

Each example should demonstrate thorough reasoning and a polished final response that a \
smaller model can learn to emulate.

Respond with a JSON array of objects, each with "instruction", "teacher_response", and \
"reasoning" fields.
Do NOT include any text outside the JSON array."""

USER_PROMPT_TEMPLATE = """Generate {batch_size} teacher-quality examples for knowledge distillation.

{domain_context}
Teacher style: {teacher_style}
Complexity: {complexity}

Requirements:
- Instructions should be challenging enough to benefit from teacher guidance
- "reasoning" should show the step-by-step thought process (chain-of-thought)
- "teacher_response" should be the final polished answer
- For detailed style: comprehensive explanations with examples
- For concise style: precise answers with minimal but sufficient detail
- For step_by_step style: numbered steps with clear logical progression
- Vary instruction types: analysis, problem-solving, explanation, comparison
- Each example should teach something a smaller model would struggle with alone

Respond with a JSON array:
[{{"instruction": "Explain why...", "teacher_response": "The answer is...", \
"reasoning": "Let me think through this step by step..."}}]"""


class DistillationTask:
    """Generate teacher-quality responses for distilling to smaller models."""

    def __init__(
        self,
        domain: str = "",
        teacher_style: str = "detailed",
        complexity: str = "mixed",
    ):
        self.domain = domain
        self.teacher_style = teacher_style
        self.complexity = complexity

    def required_keys(self) -> set[str]:
        return {"text", "instruction", "teacher_response", "reasoning"}

    @classmethod
    def from_config(cls, config: dict) -> DistillationTask:
        """Create from config dict."""
        task_config = config.get("task", config)
        return cls(
            domain=task_config.get("domain", ""),
            teacher_style=task_config.get("teacher_style", "detailed"),
            complexity=task_config.get("complexity", "mixed"),
        )

    def build_messages(self, batch_size: int = 3) -> list[dict[str, str]]:
        """Build messages for generating distillation examples."""
        domain_context = f"Domain: {self.domain}" if self.domain else ""

        user_msg = USER_PROMPT_TEMPLATE.format(
            batch_size=batch_size,
            domain_context=domain_context,
            teacher_style=self.teacher_style,
            complexity=self.complexity,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def parse_response(self, response: str) -> list[Sample]:
        """Parse JSON array of distillation examples."""
        response = clean_llm_response(response)

        try:
            items = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse distillation response as JSON")
            return []

        samples = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if not all(k in item for k in ("instruction", "teacher_response", "reasoning")):
                continue
            samples.append(
                Sample(
                    text=item["instruction"],
                    metadata={
                        "instruction": item["instruction"],
                        "teacher_response": item["teacher_response"],
                        "reasoning": item["reasoning"],
                    },
                )
            )
        return samples
