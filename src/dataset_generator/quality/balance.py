"""Label balance enforcement — detect and optionally correct class imbalance."""

from __future__ import annotations

import logging
from collections import Counter

from dataset_generator.quality.pipeline import StepReport
from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)


class BalanceChecker:
    """Check and optionally enforce label balance."""

    name: str = "balance"

    def __init__(
        self,
        strategy: str = "report",
        max_ratio: float = 3.0,
    ) -> None:
        """Init balance checker.

        Args:
            strategy: "report" (just report) or "undersample" (trim majority classes).
            max_ratio: Maximum allowed ratio between most and least common labels.
        """
        if strategy not in ("report", "undersample"):
            raise ValueError(f"Invalid strategy: {strategy!r}. Must be 'report' or 'undersample'.")
        if max_ratio < 1.0:
            raise ValueError("max_ratio must be >= 1.0")
        self.strategy = strategy
        self.max_ratio = max_ratio

    def process(self, samples: list[Sample]) -> tuple[list[Sample], StepReport]:
        """Check label balance and optionally undersample."""
        labeled = [s for s in samples if s.label is not None]
        unlabeled = [s for s in samples if s.label is None]

        # Nothing to balance if no labels
        if not labeled:
            return samples, StepReport(
                name=self.name,
                input_count=len(samples),
                output_count=len(samples),
                removed=0,
                details={"skipped": "no labeled samples"},
            )

        counts = Counter(s.label for s in labeled)
        min_count = min(counts.values())
        max_count = max(counts.values())
        ratio = max_count / min_count if min_count > 0 else float("inf")
        is_imbalanced = ratio > self.max_ratio

        details: dict = {
            "label_counts": dict(counts),
            "ratio": round(ratio, 2),
            "imbalanced": is_imbalanced,
        }

        if not is_imbalanced or self.strategy == "report":
            if is_imbalanced:
                logger.warning(
                    "Label imbalance detected (ratio=%.1f, max_ratio=%.1f): %s",
                    ratio,
                    self.max_ratio,
                    dict(counts),
                )
            return samples, StepReport(
                name=self.name,
                input_count=len(samples),
                output_count=len(samples),
                removed=0,
                details=details,
            )

        # Undersample: cap each label at max_ratio * min_count
        cap = int(min_count * self.max_ratio)
        label_budget: dict[str, int] = {}
        for label_val, count in counts.items():
            label_budget[label_val] = min(count, cap)

        balanced: list[Sample] = []
        used: Counter[str] = Counter()
        for sample in labeled:
            assert sample.label is not None
            if used[sample.label] < label_budget[sample.label]:
                balanced.append(sample)
                used[sample.label] += 1

        output = balanced + unlabeled
        removed = len(samples) - len(output)

        details["cap_per_label"] = cap
        details["label_counts_after"] = dict(used)
        logger.info("Undersampled: removed %d samples (cap=%d per label)", removed, cap)

        return output, StepReport(
            name=self.name,
            input_count=len(samples),
            output_count=len(output),
            removed=removed,
            details=details,
        )
