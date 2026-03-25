"""Tests for HuggingFace Hub card generation and publishing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dataset_generator.hub import _map_task_category, _size_category, generate_card, push_to_hub
from dataset_generator.tasks.base import Sample


class TestGenerateCard:
    def test_card_contains_repo_name(self):
        samples = [Sample(text="example text", label="positive")]
        card = generate_card(samples, repo_id="user/my-dataset")

        assert "my-dataset" in card

    def test_card_contains_sample_count(self):
        samples = [Sample(text=f"sample {i}") for i in range(5)]
        card = generate_card(samples, repo_id="user/test-ds")

        assert "5" in card

    def test_card_contains_model_info(self):
        samples = [Sample(text="text")]
        config = {
            "provider": {"model": "gpt-4o-mini"},
            "task": {"type": "classification"},
            "generation": {"strategy": "persona"},
        }
        card = generate_card(samples, repo_id="user/ds", config=config)

        assert "gpt-4o-mini" in card
        assert "classification" in card
        assert "persona" in card

    def test_card_contains_examples(self):
        samples = [
            Sample(text="First example", label="A"),
            Sample(text="Second example", label="B"),
        ]
        card = generate_card(samples, repo_id="user/ds")

        assert "First example" in card
        assert "Example 1" in card
        assert "Example 2" in card

    def test_card_contains_label_distribution(self):
        samples = [
            Sample(text="a", label="positive"),
            Sample(text="b", label="positive"),
            Sample(text="c", label="negative"),
        ]
        card = generate_card(samples, repo_id="user/ds")

        assert "Label Distribution" in card
        assert "positive" in card
        assert "negative" in card

    def test_card_has_yaml_frontmatter(self):
        samples = [Sample(text="text")]
        card = generate_card(samples, repo_id="user/ds")

        assert card.startswith("---\n")
        assert "synthetic" in card
        assert "dataset-generator" in card

    def test_card_with_no_config(self):
        """Card should still generate with default/unknown values when no config given."""
        samples = [Sample(text="hello")]
        card = generate_card(samples, repo_id="user/ds")

        assert "unknown" in card  # default model/task type
        assert "1" in card  # sample count

    def test_card_includes_temperature(self):
        samples = [Sample(text="text")]
        config = {"generation": {"temperature": 0.9}, "provider": {}, "task": {}}
        card = generate_card(samples, repo_id="user/ds", config=config)

        assert "0.9" in card
        assert "Temperature" in card


class TestSizeCategory:
    def test_small(self):
        assert _size_category(500) == "n<1K"

    def test_1k(self):
        assert _size_category(5_000) == "1K<n<10K"

    def test_10k(self):
        assert _size_category(50_000) == "10K<n<100K"

    def test_100k(self):
        assert _size_category(500_000) == "100K<n<1M"

    def test_1m(self):
        assert _size_category(2_000_000) == "n>1M"


class TestMapTaskCategory:
    def test_known_types(self):
        assert _map_task_category("classification") == "text-classification"
        assert _map_task_category("ner") == "token-classification"
        assert _map_task_category("qa") == "question-answering"

    def test_unknown_falls_back(self):
        assert _map_task_category("custom") == "text-generation"


class TestPushToHub:
    def test_push_calls_hub_apis(self):
        mock_dataset_cls = MagicMock()
        mock_ds_instance = MagicMock()
        mock_dataset_cls.from_list.return_value = mock_ds_instance
        mock_hf_api_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "datasets": MagicMock(Dataset=mock_dataset_cls),
                "huggingface_hub": MagicMock(HfApi=mock_hf_api_cls),
            },
        ):
            # Re-import to pick up mocks
            from importlib import reload

            import dataset_generator.hub as hub_mod

            reload(hub_mod)

            samples = [Sample(text="test", label="a")]
            url = hub_mod.push_to_hub(samples, repo_id="user/test-ds", token="tok")

            assert "user/test-ds" in url
            mock_dataset_cls.from_list.assert_called_once()
            mock_ds_instance.push_to_hub.assert_called_once()

    def test_push_import_error(self):
        with (
            patch.dict("sys.modules", {"datasets": None, "huggingface_hub": None}),
            pytest.raises(ImportError, match="huggingface"),
        ):
            push_to_hub([Sample(text="x")], repo_id="u/d")
