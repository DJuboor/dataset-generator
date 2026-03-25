"""Tests for quality pipeline — dedup and validation."""

from dataset_generator.quality.dedup import deduplicate
from dataset_generator.quality.validate import validate_samples
from dataset_generator.tasks.base import Sample


class TestDeduplicate:
    def test_exact_duplicates(self):
        samples = [
            Sample(text="hello world", label="a"),
            Sample(text="hello world", label="a"),
            Sample(text="different text", label="b"),
        ]
        result = deduplicate(samples, similarity_threshold=1.0)
        assert len(result) == 2

    def test_case_insensitive_exact(self):
        samples = [
            Sample(text="Hello World", label="a"),
            Sample(text="hello world", label="a"),
        ]
        result = deduplicate(samples, similarity_threshold=1.0)
        assert len(result) == 1

    def test_fuzzy_duplicates(self):
        samples = [
            Sample(text="This is a great product, I love it", label="pos"),
            Sample(text="This is a great product, I love it!", label="pos"),
            Sample(text="Completely different text here", label="neg"),
        ]
        result = deduplicate(samples, similarity_threshold=0.85)
        assert len(result) == 2

    def test_empty_input(self):
        assert deduplicate([]) == []

    def test_threshold_1_skips_fuzzy(self):
        samples = [
            Sample(text="almost same text here", label="a"),
            Sample(text="almost same text here!", label="a"),
        ]
        result = deduplicate(samples, similarity_threshold=1.0)
        assert len(result) == 2  # fuzzy dedup skipped


class TestValidateSamples:
    def test_min_length(self):
        samples = [
            Sample(text="short", label="a"),
            Sample(text="this is long enough to pass", label="b"),
        ]
        result = validate_samples(samples, min_length=10)
        assert len(result) == 1
        assert result[0].label == "b"

    def test_max_length(self):
        samples = [
            Sample(text="ok", label="a"),
            Sample(text="x" * 100, label="b"),
        ]
        result = validate_samples(samples, min_length=1, max_length=50)
        assert len(result) == 1

    def test_allowed_labels(self):
        samples = [
            Sample(text="valid label text", label="positive"),
            Sample(text="invalid label text", label="unknown"),
        ]
        result = validate_samples(samples, min_length=1, allowed_labels=["positive", "negative"])
        assert len(result) == 1
        assert result[0].label == "positive"

    def test_no_label_passes_label_check(self):
        samples = [Sample(text="no label here", label=None)]
        result = validate_samples(samples, min_length=1, allowed_labels=["a"])
        assert len(result) == 1
