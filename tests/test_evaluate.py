"""Tests for dataset evaluation metrics."""

from dataset_generator.quality.evaluate import evaluate_dataset
from dataset_generator.tasks.base import Sample


class TestEvaluateDataset:
    def test_basic_metrics(self):
        samples = [
            Sample(text="The quick brown fox jumps over the lazy dog", label="positive"),
            Sample(text="A completely different sentence with unique words", label="negative"),
            Sample(text="Yet another example for diversity testing", label="positive"),
        ]
        metrics = evaluate_dataset(samples)

        assert metrics["sample_count"] == 3
        assert metrics["vocabulary_size"] > 0
        assert metrics["type_token_ratio"] > 0
        assert 0 < metrics["distinct_1"] <= 1
        assert 0 < metrics["distinct_2"] <= 1
        assert "text_length" in metrics

    def test_label_distribution(self):
        samples = [
            Sample(text="pos text one", label="positive"),
            Sample(text="pos text two", label="positive"),
            Sample(text="neg text one", label="negative"),
        ]
        metrics = evaluate_dataset(samples)

        assert metrics["num_labels"] == 2
        assert metrics["label_distribution"]["positive"] == 2
        assert metrics["label_entropy"] > 0

    def test_self_bleu(self):
        samples = [Sample(text=f"Sample number {i} with text content") for i in range(10)]
        metrics = evaluate_dataset(samples)

        assert "self_bleu" in metrics
        assert 0 <= metrics["self_bleu"] <= 1

    def test_empty_dataset(self):
        metrics = evaluate_dataset([])
        assert "error" in metrics

    def test_no_labels(self):
        samples = [Sample(text="just text")]
        metrics = evaluate_dataset(samples)

        assert "label_distribution" not in metrics
        assert metrics["sample_count"] == 1
