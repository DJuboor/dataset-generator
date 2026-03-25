"""Text classification dataset generation."""

from __future__ import annotations

import json
import logging

from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a dataset generation assistant. Generate realistic, diverse text \
classification examples. Each example should be a natural text that clearly belongs to the \
specified label.

Respond with a JSON array of objects, each with "text" and "label" fields.
Do NOT include any text outside the JSON array."""

USER_PROMPT_TEMPLATE = """Generate {batch_size} diverse text classification examples.

Labels: {labels}
{domain_context}
{label_descriptions}

Requirements:
- Each example should be realistic and clearly belong to its label
- Vary length, style, and vocabulary across examples
- Distribute examples roughly equally across labels
- Do NOT repeat or closely paraphrase examples

Respond with a JSON array:
[{{"text": "example text", "label": "label_name"}}, ...]"""


class ClassificationTask:
    """Generate labeled text classification datasets."""

    def __init__(
        self,
        labels: list[str],
        domain: str = "",
        label_descriptions: dict[str, str] | None = None,
    ):
        self.labels = labels
        self.domain = domain
        self.label_descriptions = label_descriptions or {}

    @classmethod
    def from_config(cls, config: dict) -> ClassificationTask:
        """Create from config dict."""
        task_config = config.get("task", config)
        labels = task_config.get("labels", [])
        if isinstance(labels, str):
            labels = [lbl.strip() for lbl in labels.split(",")]
        return cls(
            labels=labels,
            domain=task_config.get("domain", ""),
            label_descriptions=task_config.get("label_descriptions", {}),
        )

    def build_messages(self, batch_size: int = 10) -> list[dict[str, str]]:
        """Build messages for generating classification samples."""
        domain_context = f"Domain: {self.domain}" if self.domain else ""
        label_desc = ""
        if self.label_descriptions:
            lines = [f"- {k}: {v}" for k, v in self.label_descriptions.items()]
            label_desc = "Label descriptions:\n" + "\n".join(lines)

        user_msg = USER_PROMPT_TEMPLATE.format(
            batch_size=batch_size,
            labels=", ".join(self.labels),
            domain_context=domain_context,
            label_descriptions=label_desc,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def parse_response(self, response: str) -> list[Sample]:
        """Parse JSON array of classification samples."""
        response = response.strip()
        # Handle markdown code blocks
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            items = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse classification response as JSON")
            return []

        samples = []
        for item in items:
            if isinstance(item, dict) and "text" in item and "label" in item:
                if item["label"] in self.labels:
                    samples.append(Sample(text=item["text"], label=item["label"]))
                else:
                    logger.debug(f"Skipping sample with invalid label: {item['label']}")
        return samples
