"""Quality pipeline — composable post-generation filtering and annotation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)


@dataclass
class StepReport:
    """Report from a single quality step."""

    name: str
    input_count: int
    output_count: int
    removed: int
    flagged: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Aggregate report from the full quality pipeline."""

    input_count: int
    output_count: int
    step_reports: list[StepReport] = field(default_factory=list)


class QualityStep(Protocol):
    """Protocol for quality pipeline steps."""

    name: str

    def process(self, samples: list[Sample]) -> tuple[list[Sample], StepReport]: ...


class QualityPipeline:
    """Runs a sequence of quality steps, accumulating reports."""

    def __init__(self, steps: list[QualityStep]) -> None:
        self.steps = steps

    def run(self, samples: list[Sample]) -> tuple[list[Sample], QualityReport]:
        """Run all steps in sequence.

        Returns:
            Filtered/annotated samples and aggregate report.
        """
        report = QualityReport(input_count=len(samples), output_count=0)
        current = samples

        for step in self.steps:
            current, step_report = step.process(current)
            report.step_reports.append(step_report)
            logger.info(
                "Step '%s': %d -> %d samples (removed=%d, flagged=%d)",
                step_report.name,
                step_report.input_count,
                step_report.output_count,
                step_report.removed,
                step_report.flagged,
            )

        report.output_count = len(current)
        return current, report
