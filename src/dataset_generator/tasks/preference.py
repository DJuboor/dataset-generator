"""Preference pair generation for DPO/RLHF training."""

from __future__ import annotations

import json
import logging

from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a dataset generation assistant. Generate preference pairs for \
reinforcement learning from human feedback (RLHF) / direct preference optimization (DPO).

Each example has a prompt, a chosen (better) response, and a rejected (worse) response.
The quality difference should be clear but realistic — not trivially bad rejected responses.

Respond with a JSON array of objects with "prompt", "chosen", and "rejected" fields.
Do NOT include any text outside the JSON array."""

USER_PROMPT_TEMPLATE = """Generate {batch_size} preference pairs.

{domain_context}
{criteria}

Requirements:
- Prompts should be realistic user requests
- "chosen" responses should be clearly better than "rejected"
- Quality differences: accuracy, helpfulness, safety, conciseness
- Rejected responses should be plausibly wrong, not obviously terrible
- Vary the type of quality difference across examples

Respond with a JSON array:
[{{"prompt": "user request", "chosen": "good response", "rejected": "worse response"}}]"""


class PreferenceTask:
    """Generate preference pair datasets for DPO/RLHF."""

    def __init__(self, domain: str = "", criteria: str = ""):
        self.domain = domain
        self.criteria = criteria

    @classmethod
    def from_config(cls, config: dict) -> PreferenceTask:
        task_config = config.get("task", config)
        return cls(
            domain=task_config.get("domain", ""),
            criteria=task_config.get("criteria", ""),
        )

    def build_messages(self, batch_size: int = 5) -> list[dict[str, str]]:
        domain_context = f"Domain: {self.domain}" if self.domain else ""
        criteria = f"Preference criteria: {self.criteria}" if self.criteria else ""
        user_msg = USER_PROMPT_TEMPLATE.format(
            batch_size=batch_size,
            domain_context=domain_context,
            criteria=criteria,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def parse_response(self, response: str) -> list[Sample]:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            items = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse preference response as JSON")
            return []

        samples = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if not all(k in item for k in ("prompt", "chosen", "rejected")):
                continue
            samples.append(
                Sample(
                    text=item["prompt"],
                    metadata={"chosen": item["chosen"], "rejected": item["rejected"]},
                )
            )
        return samples
