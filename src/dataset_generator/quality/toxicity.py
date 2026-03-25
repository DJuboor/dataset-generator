"""Toxicity filtering — keyword-based detection of harmful content."""

from __future__ import annotations

import logging
import re

from dataset_generator.quality.pipeline import StepReport
from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)

# Minimal default keyword list. Uses word boundaries to reduce false positives.
# Intentionally kept short — users should extend via the keywords parameter.
_DEFAULT_KEYWORDS: list[str] = [
    "fuck",
    "shit",
    "asshole",
    "bitch",
    "bastard",
    "damn",
    "crap",
    "dick",
    "piss",
    "slut",
    "whore",
    "faggot",
    "nigger",
    "retard",
    "kike",
    "spic",
    "chink",
    "kill yourself",
    "kys",
]


def _build_pattern(keywords: list[str]) -> re.Pattern[str]:
    """Build a single regex alternation with word boundaries."""
    escaped = [re.escape(kw) for kw in keywords]
    # Use word boundaries for single-word, looser match for multi-word
    parts: list[str] = []
    for kw, esc in zip(keywords, escaped, strict=True):
        if " " in kw:
            parts.append(esc)
        else:
            parts.append(rf"\b{esc}\b")
    return re.compile("|".join(parts), re.IGNORECASE)


class ToxicityFilter:
    """Detect toxic/harmful content via keyword matching."""

    name: str = "toxicity"

    def __init__(
        self,
        method: str = "keywords",
        action: str = "flag",
        keywords: list[str] | None = None,
    ) -> None:
        """Init toxicity filter.

        Args:
            method: Detection method. Currently only "keywords".
            action: "flag" (annotate but keep) or "remove" (drop sample).
            keywords: Custom keyword list. None uses defaults.
        """
        if method != "keywords":
            raise ValueError(f"Invalid method: {method!r}. Only 'keywords' is supported.")
        if action not in ("flag", "remove"):
            raise ValueError(f"Invalid action: {action!r}. Must be 'flag' or 'remove'.")
        self.method = method
        self.action = action
        word_list = keywords if keywords is not None else _DEFAULT_KEYWORDS
        self._pattern = _build_pattern(word_list)

    def process(self, samples: list[Sample]) -> tuple[list[Sample], StepReport]:
        """Scan samples for toxic content."""
        output: list[Sample] = []
        removed = 0
        flagged = 0
        total_matches = 0

        for sample in samples:
            matches = self._pattern.findall(sample.text)
            if not matches:
                output.append(sample)
                continue

            total_matches += len(matches)

            if self.action == "remove":
                removed += 1
                continue

            # Flag
            flagged += 1
            unique_matches = sorted(set(m.lower() for m in matches))
            sample = sample.model_copy(
                update={
                    "metadata": {
                        **sample.metadata,
                        "toxic_keywords": unique_matches,
                    },
                },
            )
            output.append(sample)

        if total_matches:
            logger.info(
                "Toxicity filter: %d matches found, %d removed, %d flagged",
                total_matches,
                removed,
                flagged,
            )

        return output, StepReport(
            name=self.name,
            input_count=len(samples),
            output_count=len(output),
            removed=removed,
            flagged=flagged,
            details={"total_matches": total_matches},
        )
