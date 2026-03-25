"""Question-answering dataset generation."""

from __future__ import annotations

import json
import logging

from dataset_generator.tasks.base import Sample, clean_llm_response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a dataset generation assistant. Generate realistic question-answer \
pairs. Each pair should have a clear, factual question and a concise answer.

Respond with a JSON array of objects with "question", "answer", and optionally "context" fields.
Do NOT include any text outside the JSON array."""

USER_PROMPT_TEMPLATE = """Generate {batch_size} diverse question-answer pairs.

{domain_context}
{context_instruction}

Requirements:
- Questions should be clear and specific
- Answers should be concise and accurate
- Vary question types (what, how, why, when, where, who)
- Include both simple factual and reasoning questions

Respond with a JSON array:
[{{"question": "What is...", "answer": "It is...", "context": "optional background text"}}]"""


class QATask:
    """Generate question-answer pair datasets."""

    def __init__(self, domain: str = "", contexts: list[str] | None = None):
        self.domain = domain
        self.contexts = contexts

    @classmethod
    def from_config(cls, config: dict) -> QATask:
        task_config = config.get("task", config)
        return cls(
            domain=task_config.get("domain", ""),
            contexts=task_config.get("contexts"),
        )

    def build_messages(self, batch_size: int = 5) -> list[dict[str, str]]:
        domain_context = f"Domain: {self.domain}" if self.domain else ""
        context_instruction = ""
        if self.contexts:
            context_instruction = (
                "Generate questions based on these reference texts:\n"
                + "\n---\n".join(self.contexts[:5])
            )

        user_msg = USER_PROMPT_TEMPLATE.format(
            batch_size=batch_size,
            domain_context=domain_context,
            context_instruction=context_instruction,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def parse_response(self, response: str) -> list[Sample]:
        response = clean_llm_response(response)

        try:
            items = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse QA response as JSON")
            return []

        samples = []
        for item in items:
            if not isinstance(item, dict) or "question" not in item or "answer" not in item:
                continue
            metadata = {}
            if "context" in item:
                metadata["context"] = item["context"]
            samples.append(
                Sample(
                    text=item["question"],
                    label=item["answer"],
                    metadata=metadata,
                )
            )
        return samples
