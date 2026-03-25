"""Sample validation — checks schema and constraint conformance."""

import logging

from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)


def validate_samples(
    samples: list[Sample],
    min_length: int = 10,
    max_length: int = 10000,
    allowed_labels: list[str] | None = None,
) -> list[Sample]:
    """Filter samples that fail validation constraints.

    Args:
        samples: Input samples.
        min_length: Minimum text length (chars).
        max_length: Maximum text length (chars).
        allowed_labels: If set, reject samples with labels not in this list.

    Returns:
        Valid samples only.
    """
    valid: list[Sample] = []
    rejected = 0

    for sample in samples:
        if len(sample.text) < min_length:
            rejected += 1
            continue
        if len(sample.text) > max_length:
            rejected += 1
            continue
        if allowed_labels and sample.label and sample.label not in allowed_labels:
            rejected += 1
            continue
        valid.append(sample)

    if rejected:
        logger.info(f"Rejected {rejected}/{len(samples)} samples during validation")

    return valid
