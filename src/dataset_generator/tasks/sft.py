"""SFT / instruction-following dataset generation."""

from __future__ import annotations

import json
import logging

from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a dataset generation assistant. Generate diverse, high-quality \
instruction-response pairs suitable for supervised fine-tuning (SFT) of language models.

Each pair should have a clear instruction and a helpful, accurate response. Vary the \
instruction types: questions, tasks, creative writing, analysis, coding, math, etc.

Respond with a JSON array of objects, each with "instruction" and "response" fields.
Do NOT include any text outside the JSON array."""

USER_PROMPT_TEMPLATE = """Generate {batch_size} diverse instruction-response pairs for SFT training.

{domain_context}
{system_prompt_context}
Complexity: {complexity}
Response style: {response_style}

Requirements:
- Instructions should span different topics and formats within the domain
- Responses should match the requested complexity and style
- Vary instruction types: open-ended, factual, creative, analytical, procedural
- Each response must fully address its instruction
- Do NOT repeat or closely paraphrase instructions

Respond with a JSON array:
[{{"instruction": "Explain how...", "response": "Here is an explanation..."}}]"""


class SFTTask:
    """Generate instruction-response pairs for supervised fine-tuning."""

    def __init__(
        self,
        domain: str = "",
        system_prompt: str = "",
        complexity: str = "mixed",
        response_style: str = "detailed",
    ):
        self.domain = domain
        self.system_prompt = system_prompt
        self.complexity = complexity
        self.response_style = response_style

    @classmethod
    def from_config(cls, config: dict) -> SFTTask:
        """Create from config dict."""
        task_config = config.get("task", config)
        return cls(
            domain=task_config.get("domain", ""),
            system_prompt=task_config.get("system_prompt", ""),
            complexity=task_config.get("complexity", "mixed"),
            response_style=task_config.get("response_style", "detailed"),
        )

    def build_messages(self, batch_size: int = 5) -> list[dict[str, str]]:
        """Build messages for generating SFT pairs."""
        domain_context = f"Domain: {self.domain}" if self.domain else ""
        system_prompt_context = (
            f"System prompt for the model being trained: {self.system_prompt}"
            if self.system_prompt
            else ""
        )

        user_msg = USER_PROMPT_TEMPLATE.format(
            batch_size=batch_size,
            domain_context=domain_context,
            system_prompt_context=system_prompt_context,
            complexity=self.complexity,
            response_style=self.response_style,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def parse_response(self, response: str) -> list[Sample]:
        """Parse JSON array of instruction-response pairs."""
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            items = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse SFT response as JSON")
            return []

        samples = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if "instruction" not in item or "response" not in item:
                continue
            samples.append(
                Sample(
                    text=item["instruction"],
                    metadata={
                        "instruction": item["instruction"],
                        "response": item["response"],
                        "system_prompt": self.system_prompt,
                    },
                )
            )
        return samples
