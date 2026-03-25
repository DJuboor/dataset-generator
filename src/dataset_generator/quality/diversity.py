"""Diversity metrics — report-only analysis of sample variety."""

from __future__ import annotations

import logging
from collections import Counter

from dataset_generator.quality.pipeline import StepReport
from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Whitespace tokenizer with lowercasing."""
    return text.lower().split()


def _distinct_n(all_tokens: list[str], n: int) -> float:
    """Compute distinct-n: fraction of unique n-grams out of total n-grams."""
    if len(all_tokens) < n:
        return 0.0
    ngrams = [tuple(all_tokens[i : i + n]) for i in range(len(all_tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


class DiversityReporter:
    """Compute diversity metrics without modifying samples."""

    name: str = "diversity"

    def process(self, samples: list[Sample]) -> tuple[list[Sample], StepReport]:
        """Compute diversity metrics and return all samples unmodified."""
        if not samples:
            return samples, StepReport(
                name=self.name,
                input_count=0,
                output_count=0,
                removed=0,
                details={"skipped": "no samples"},
            )

        all_tokens: list[str] = []
        vocab: Counter[str] = Counter()
        for sample in samples:
            tokens = _tokenize(sample.text)
            all_tokens.extend(tokens)
            vocab.update(tokens)

        d1 = _distinct_n(all_tokens, 1)
        d2 = _distinct_n(all_tokens, 2)
        d3 = _distinct_n(all_tokens, 3)

        details = {
            "distinct_1": round(d1, 4),
            "distinct_2": round(d2, 4),
            "distinct_3": round(d3, 4),
            "vocabulary_size": len(vocab),
            "total_tokens": len(all_tokens),
        }

        logger.info(
            "Diversity: d1=%.4f, d2=%.4f, d3=%.4f, vocab=%d, tokens=%d",
            d1,
            d2,
            d3,
            len(vocab),
            len(all_tokens),
        )

        return samples, StepReport(
            name=self.name,
            input_count=len(samples),
            output_count=len(samples),
            removed=0,
            details=details,
        )
