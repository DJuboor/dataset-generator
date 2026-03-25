"""PII detection — regex-based detection of personal identifiable information."""

from __future__ import annotations

import logging
import re

from dataset_generator.quality.pipeline import StepReport
from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)

# Default PII patterns: name -> compiled regex
_DEFAULT_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone_us": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)"),
    "ip_address": re.compile(
        r"(?<!\d)(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}(?!\d)"
    ),
}

_REDACTED = "[REDACTED]"


class PIIFilter:
    """Detect and handle PII in generated samples."""

    name: str = "pii"

    def __init__(
        self,
        action: str = "flag",
        patterns: list[str] | None = None,
    ) -> None:
        """Init PII filter.

        Args:
            action: "flag" (annotate but keep), "redact" (replace matches), or "remove" (drop sample).
            patterns: Subset of default pattern names to use. None means all.
        """
        if action not in ("flag", "redact", "remove"):
            raise ValueError(f"Invalid action: {action!r}. Must be 'flag', 'redact', or 'remove'.")
        self.action = action
        if patterns is not None:
            unknown = set(patterns) - set(_DEFAULT_PATTERNS)
            if unknown:
                raise ValueError(f"Unknown PII patterns: {unknown}")
            self.patterns = {k: v for k, v in _DEFAULT_PATTERNS.items() if k in patterns}
        else:
            self.patterns = dict(_DEFAULT_PATTERNS)

    def _find_pii(self, text: str) -> dict[str, list[str]]:
        """Return dict of pattern_name -> list of matches found in text."""
        found: dict[str, list[str]] = {}
        for name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                found[name] = matches
        return found

    def process(self, samples: list[Sample]) -> tuple[list[Sample], StepReport]:
        """Scan samples for PII and apply configured action."""
        output: list[Sample] = []
        removed = 0
        flagged = 0
        pii_counts: dict[str, int] = {}

        for sample in samples:
            found = self._find_pii(sample.text)
            if not found:
                output.append(sample)
                continue

            # Tally matches per pattern type
            for ptype, matches in found.items():
                pii_counts[ptype] = pii_counts.get(ptype, 0) + len(matches)

            if self.action == "remove":
                removed += 1
                continue

            if self.action == "redact":
                text = sample.text
                for pattern in self.patterns.values():
                    text = pattern.sub(_REDACTED, text)
                sample = sample.model_copy(
                    update={"text": text, "metadata": {**sample.metadata, "pii_redacted": True}},
                )
                output.append(sample)

            elif self.action == "flag":
                flagged += 1
                detected_types = list(found.keys())
                sample = sample.model_copy(
                    update={"metadata": {**sample.metadata, "pii_detected": detected_types}},
                )
                output.append(sample)

        if pii_counts:
            logger.info("PII detected: %s", pii_counts)

        return output, StepReport(
            name=self.name,
            input_count=len(samples),
            output_count=len(output),
            removed=removed,
            flagged=flagged,
            details={"pii_counts": pii_counts},
        )
