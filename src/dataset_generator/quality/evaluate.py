"""Dataset evaluation metrics — text stats, diversity, and quality scoring."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from dataset_generator.tasks.base import Sample


def evaluate_dataset(samples: list[Sample]) -> dict[str, Any]:
    """Compute comprehensive quality metrics for a dataset.

    Returns a dict of metrics suitable for display or JSON export.
    """
    if not samples:
        return {"error": "No samples to evaluate"}

    texts = [s.text for s in samples]
    labels = [s.label for s in samples if s.label is not None]

    metrics: dict[str, Any] = {}

    # Text statistics
    lengths = [len(t) for t in texts]
    metrics["sample_count"] = len(samples)
    metrics["text_length"] = {
        "mean": round(sum(lengths) / len(lengths), 1),
        "min": min(lengths),
        "max": max(lengths),
        "std": round(_std(lengths), 1),
    }

    # Vocabulary
    all_tokens = _tokenize_all(texts)
    metrics["vocabulary_size"] = len(set(all_tokens))
    metrics["total_tokens"] = len(all_tokens)
    metrics["type_token_ratio"] = (
        round(len(set(all_tokens)) / len(all_tokens), 4) if all_tokens else 0
    )

    # Distinct-N diversity
    metrics["distinct_1"] = round(_distinct_n(all_tokens, 1), 4)
    metrics["distinct_2"] = round(_distinct_n(all_tokens, 2), 4)
    metrics["distinct_3"] = round(_distinct_n(all_tokens, 3), 4)

    # Self-BLEU (lower = more diverse, computed on random sample for efficiency)
    if len(texts) >= 4:
        metrics["self_bleu"] = round(_self_bleu(texts, sample_size=min(100, len(texts))), 4)

    # Label distribution
    if labels:
        label_counts = Counter(labels)
        metrics["label_distribution"] = dict(label_counts.most_common())
        metrics["label_entropy"] = round(_entropy(list(label_counts.values())), 4)
        metrics["num_labels"] = len(label_counts)

    return metrics


def _tokenize_all(texts: list[str]) -> list[str]:
    """Simple whitespace tokenization across all texts."""
    tokens = []
    for text in texts:
        tokens.extend(text.lower().split())
    return tokens


def _distinct_n(tokens: list[str], n: int) -> float:
    """Fraction of unique n-grams out of total n-grams."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def _self_bleu(texts: list[str], sample_size: int = 100) -> float:
    """Average pairwise BLEU score (simplified). Lower = more diverse."""
    if len(texts) < 2:
        return 0.0

    sampled = random.sample(texts, min(sample_size, len(texts)))
    scores = []

    for i in range(min(50, len(sampled))):
        ref = sampled[i].lower().split()
        # Compare against a random other text
        j = (i + 1) % len(sampled)
        hyp = sampled[j].lower().split()
        scores.append(_bleu_score(ref, hyp))

    return sum(scores) / len(scores) if scores else 0.0


def _bleu_score(reference: list[str], hypothesis: list[str]) -> float:
    """Simplified unigram BLEU (precision of hypothesis tokens in reference)."""
    if not hypothesis or not reference:
        return 0.0
    ref_counts = Counter(reference)
    matches = sum(
        min(count, ref_counts.get(token, 0)) for token, count in Counter(hypothesis).items()
    )
    return matches / len(hypothesis)


def _entropy(counts: list[int]) -> float:
    """Shannon entropy of a distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _std(values: list[int | float]) -> float:
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)
