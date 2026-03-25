"""Deduplication — exact and fuzzy (n-gram shingling, O(n) approximate)."""

import logging

from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    return " ".join(text.lower().split())


def _shingles(text: str, n: int = 3) -> set[str]:
    """Compute character n-gram shingle set."""
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two shingle sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(
    samples: list[Sample],
    similarity_threshold: float = 0.85,
) -> list[Sample]:
    """Remove exact and near-duplicate samples.

    Uses exact hash dedup (O(n)) then n-gram shingling with an inverted index
    for approximate fuzzy dedup. Much faster than SequenceMatcher at scale.

    Args:
        samples: Input samples.
        similarity_threshold: Fuzzy match threshold (0-1). Pairs above this are duplicates.

    Returns:
        Deduplicated samples.
    """
    if not samples:
        return samples

    # Phase 1: exact dedup on normalized text
    seen_exact: set[str] = set()
    unique: list[Sample] = []
    for sample in samples:
        key = _normalize(sample.text)
        if key not in seen_exact:
            seen_exact.add(key)
            unique.append(sample)

    exact_removed = len(samples) - len(unique)
    if exact_removed:
        logger.info(f"Removed {exact_removed} exact duplicates")

    # Phase 2: fuzzy dedup with n-gram shingling
    if similarity_threshold >= 1.0:
        return unique

    # Build inverted index: shingle -> list of sample indices
    kept_shingles: list[set[str]] = []
    # Map from shingle to indices of kept samples that contain it
    inv_index: dict[str, list[int]] = {}
    final: list[Sample] = []

    for sample in unique:
        normalized = _normalize(sample.text)
        sample_shingles = _shingles(normalized)

        # Find candidate duplicates via inverted index
        candidate_indices: set[int] = set()
        for shingle in sample_shingles:
            if shingle in inv_index:
                candidate_indices.update(inv_index[shingle])

        is_dup = False
        for idx in candidate_indices:
            if _jaccard(sample_shingles, kept_shingles[idx]) >= similarity_threshold:
                is_dup = True
                break

        if not is_dup:
            new_idx = len(final)
            kept_shingles.append(sample_shingles)
            for shingle in sample_shingles:
                inv_index.setdefault(shingle, []).append(new_idx)
            final.append(sample)

    fuzzy_removed = len(unique) - len(final)
    if fuzzy_removed:
        logger.info(f"Removed {fuzzy_removed} fuzzy duplicates (threshold={similarity_threshold})")

    return final
