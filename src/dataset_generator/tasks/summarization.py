"""Text summarization dataset generation."""

from __future__ import annotations

import json
import logging

from dataset_generator.tasks.base import Sample, clean_llm_response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a dataset generation assistant. Generate realistic document-summary \
pairs for training summarization models. Each pair should have a substantial source document \
and an accurate summary.

Respond with a JSON array of objects, each with "document" and "summary" fields.
Do NOT include any text outside the JSON array."""

USER_PROMPT_TEMPLATE = """Generate {batch_size} diverse document-summary pairs.

{domain_context}
Document length: approximately {min_doc_length} to {max_doc_length} words
Summary style: {summary_style}

Requirements:
- Documents should be realistic and substantive (articles, reports, descriptions, etc.)
- Each document should be {min_doc_length}-{max_doc_length} words long
- Summaries should accurately capture the key points of their document
- For abstractive: rephrase and condense in new words
- For extractive: use key sentences from the document
- For mixed: combine both approaches
- Vary topics, writing styles, and document types
- Do NOT generate trivially short documents

Respond with a JSON array:
[{{"document": "Full document text here...", "summary": "Concise summary here..."}}]"""


class SummarizationTask:
    """Generate document-summary pair datasets."""

    def __init__(
        self,
        domain: str = "",
        min_doc_length: int = 200,
        max_doc_length: int = 1000,
        summary_style: str = "abstractive",
    ):
        self.domain = domain
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
        self.summary_style = summary_style

    @classmethod
    def from_config(cls, config: dict) -> SummarizationTask:
        """Create from config dict."""
        task_config = config.get("task", config)
        return cls(
            domain=task_config.get("domain", ""),
            min_doc_length=task_config.get("min_doc_length", 200),
            max_doc_length=task_config.get("max_doc_length", 1000),
            summary_style=task_config.get("summary_style", "abstractive"),
        )

    def build_messages(self, batch_size: int = 3) -> list[dict[str, str]]:
        """Build messages for generating summarization pairs."""
        domain_context = f"Domain: {self.domain}" if self.domain else ""

        user_msg = USER_PROMPT_TEMPLATE.format(
            batch_size=batch_size,
            domain_context=domain_context,
            min_doc_length=self.min_doc_length,
            max_doc_length=self.max_doc_length,
            summary_style=self.summary_style,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def parse_response(self, response: str) -> list[Sample]:
        """Parse JSON array of document-summary pairs."""
        response = clean_llm_response(response)

        try:
            items = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse summarization response as JSON")
            return []

        samples = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if "document" not in item or "summary" not in item:
                continue
            samples.append(
                Sample(
                    text=item["document"],
                    label=item["summary"],
                )
            )
        return samples
