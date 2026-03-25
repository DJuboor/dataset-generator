"""Named entity recognition dataset generation."""

from __future__ import annotations

import json
import logging

from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a dataset generation assistant. Generate realistic text with named \
entity annotations. Each example should contain natural text with entities clearly present.

Respond with a JSON array of objects, each with "text" and "entities" fields.
Each entity has "text", "label", "start", and "end" (character offsets).
Do NOT include any text outside the JSON array."""

USER_PROMPT_TEMPLATE = """Generate {batch_size} diverse text examples with named entity annotations.

Entity types: {entity_types}
{domain_context}

Requirements:
- Each example should read naturally and contain 1-5 entities
- Vary sentence length and complexity
- Include examples where entities appear in different positions
- Character offsets (start, end) must be accurate

Respond with a JSON array:
[{{"text": "John works at Google in NYC", "entities": [{{"text": "John", "label": "PERSON", \
"start": 0, "end": 4}}, {{"text": "Google", "label": "ORG", "start": 15, "end": 21}}, \
{{"text": "NYC", "label": "LOC", "start": 25, "end": 28}}]}}]"""


class NERTask:
    """Generate NER-annotated datasets."""

    def __init__(self, entity_types: list[str], domain: str = ""):
        self.entity_types = entity_types
        self.domain = domain

    @classmethod
    def from_config(cls, config: dict) -> NERTask:
        task_config = config.get("task", config)
        entity_types = task_config.get("entity_types", [])
        if isinstance(entity_types, str):
            entity_types = [e.strip() for e in entity_types.split(",")]
        return cls(
            entity_types=entity_types,
            domain=task_config.get("domain", ""),
        )

    def build_messages(self, batch_size: int = 5) -> list[dict[str, str]]:
        domain_context = f"Domain: {self.domain}" if self.domain else ""
        user_msg = USER_PROMPT_TEMPLATE.format(
            batch_size=batch_size,
            entity_types=", ".join(self.entity_types),
            domain_context=domain_context,
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
            logger.warning("Failed to parse NER response as JSON")
            return []

        samples = []
        for item in items:
            if not isinstance(item, dict) or "text" not in item:
                continue
            entities = item.get("entities", [])
            # Validate entity offsets
            valid_entities = []
            for ent in entities:
                if all(k in ent for k in ("text", "label", "start", "end")):
                    # Verify the span matches
                    start, end = ent["start"], ent["end"]
                    if item["text"][start:end] == ent["text"]:
                        valid_entities.append(ent)
                    else:
                        logger.debug(
                            f"Entity offset mismatch: '{ent['text']}' vs '{item['text'][start:end]}'"
                        )

            samples.append(
                Sample(
                    text=item["text"],
                    metadata={"entities": valid_entities},
                )
            )
        return samples
