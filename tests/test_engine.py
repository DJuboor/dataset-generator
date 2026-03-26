"""Tests for the generation engine."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dataset_generator.engine import _estimate_cost, _generate_batch, estimate_run, generate
from dataset_generator.providers.base import CompletionResult
from dataset_generator.tasks.base import Sample


def _make_classification_config(tmp_path, **overrides):
    """Build a minimal classification config dict for testing."""
    cfg = {
        "type": "classification",
        "task": {"labels": ["positive", "negative"]},
        "provider": {
            "kind": "openai",
            "model": "gpt-4o",
            "base_url": "http://test",
            "api_key": "test",
        },
        "generation": {
            "num_samples": 10,
            "batch_size": 5,
            "max_workers": 1,
            "max_retries": 3,
            "temperature": 0.7,
            "strategy": "direct",
            "strategy_config": {},
        },
        "output": {
            "path": str(tmp_path / "output.jsonl"),
            "format": "jsonl",
        },
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and key in cfg:
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


def _mock_provider(responses: list[CompletionResult] | None = None):
    """Create a mock provider that returns canned CompletionResults."""
    provider = MagicMock()
    provider.model = "gpt-4o"
    if responses:
        provider.complete.side_effect = responses
    return provider


def _valid_json_response(n: int = 5, labels: list[str] | None = None) -> str:
    """Build a JSON string that ClassificationTask can parse."""
    labels = labels or ["positive", "negative"]
    items = []
    for i in range(n):
        label = labels[i % len(labels)]
        items.append(
            {"text": f"Sample text number {i} is sufficiently long for validation.", "label": label}
        )
    return json.dumps(items)


# ---------------------------------------------------------------------------
# _generate_batch
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    def test_valid_json_returns_samples(self):
        """Given a provider that returns valid classification JSON,
        _generate_batch should parse and return Sample objects."""
        # Given
        from dataset_generator.strategies.direct import DirectStrategy
        from dataset_generator.tasks.classification import ClassificationTask

        task = ClassificationTask(labels=["positive", "negative"])
        strategy = DirectStrategy()
        content = _valid_json_response(5)
        provider = _mock_provider(
            [
                CompletionResult(content=content, input_tokens=100, output_tokens=200),
            ]
        )

        # When
        samples, in_tok, out_tok = _generate_batch(
            provider,
            task,
            strategy,
            batch_index=0,
            batch_size=5,
            temperature=0.7,
            max_retries=3,
        )

        # Then
        assert len(samples) == 5
        assert all(isinstance(s, Sample) for s in samples)
        assert {s.label for s in samples} == {"positive", "negative"}
        assert in_tok == 100
        assert out_tok == 200

    def test_retry_on_failure_then_success(self):
        """Given a provider that fails once then succeeds,
        _generate_batch should retry and return results."""
        from dataset_generator.strategies.direct import DirectStrategy
        from dataset_generator.tasks.classification import ClassificationTask

        task = ClassificationTask(labels=["positive", "negative"])
        strategy = DirectStrategy()
        content = _valid_json_response(3)

        provider = _mock_provider(
            [
                RuntimeError("API timeout"),
                CompletionResult(content=content, input_tokens=50, output_tokens=80),
            ]
        )

        # When
        samples, in_tok, out_tok = _generate_batch(
            provider,
            task,
            strategy,
            batch_index=0,
            batch_size=3,
            temperature=0.7,
            max_retries=3,
        )

        # Then — recovered on second attempt
        assert len(samples) == 3
        assert provider.complete.call_count == 2
        assert in_tok == 50
        assert out_tok == 80

    def test_empty_response_returns_empty_list(self):
        """Given a provider that returns invalid/empty JSON,
        _generate_batch should exhaust retries and return []."""
        from dataset_generator.strategies.direct import DirectStrategy
        from dataset_generator.tasks.classification import ClassificationTask

        task = ClassificationTask(labels=["positive", "negative"])
        strategy = DirectStrategy()

        provider = _mock_provider(
            [
                CompletionResult(content="not valid json", input_tokens=10, output_tokens=5),
                CompletionResult(content="still garbage", input_tokens=10, output_tokens=5),
                CompletionResult(content="nope", input_tokens=10, output_tokens=5),
                CompletionResult(content="", input_tokens=10, output_tokens=5),
            ]
        )

        # When
        samples, in_tok, out_tok = _generate_batch(
            provider,
            task,
            strategy,
            batch_index=0,
            batch_size=5,
            temperature=0.7,
            max_retries=3,
        )

        # Then — all retries exhausted, empty result
        assert samples == []
        assert provider.complete.call_count == 4  # 1 initial + 3 retries
        assert in_tok == 40
        assert out_tok == 20

    def test_token_counts_accumulated_across_retries(self):
        """Given a provider that returns empty JSON arrays (parsed but 0 samples),
        token counts from all attempts should be accumulated."""
        from dataset_generator.strategies.direct import DirectStrategy
        from dataset_generator.tasks.classification import ClassificationTask

        task = ClassificationTask(labels=["positive", "negative"])
        strategy = DirectStrategy()

        # Empty arrays parse successfully but yield 0 samples, triggering retries
        provider = _mock_provider(
            [
                CompletionResult(content="[]", input_tokens=15, output_tokens=3),
                CompletionResult(content="[]", input_tokens=15, output_tokens=3),
                CompletionResult(
                    content=_valid_json_response(2),
                    input_tokens=20,
                    output_tokens=50,
                ),
            ]
        )

        # When
        samples, in_tok, out_tok = _generate_batch(
            provider,
            task,
            strategy,
            batch_index=0,
            batch_size=2,
            temperature=0.7,
            max_retries=3,
        )

        # Then — tokens summed across all calls
        assert len(samples) == 2
        assert in_tok == 15 + 15 + 20
        assert out_tok == 3 + 3 + 50

    def test_none_tokens_treated_as_zero(self):
        """Given a provider returning None for token counts,
        they should be treated as 0 without errors."""
        from dataset_generator.strategies.direct import DirectStrategy
        from dataset_generator.tasks.classification import ClassificationTask

        task = ClassificationTask(labels=["positive", "negative"])
        strategy = DirectStrategy()
        content = _valid_json_response(2)

        provider = _mock_provider(
            [
                CompletionResult(content=content, input_tokens=None, output_tokens=None),
            ]
        )

        # When
        samples, in_tok, out_tok = _generate_batch(
            provider,
            task,
            strategy,
            batch_index=0,
            batch_size=2,
            temperature=0.7,
            max_retries=3,
        )

        # Then
        assert len(samples) == 2
        assert in_tok == 0
        assert out_tok == 0


# ---------------------------------------------------------------------------
# generate (full pipeline)
# ---------------------------------------------------------------------------


class TestGenerate:
    @patch("dataset_generator.engine.create_provider")
    def test_full_pipeline(self, mock_create_provider, tmp_path):
        """Given a mock provider, generate() should produce deduplicated,
        validated samples and call write_output."""
        # Given — provider returns distinct samples per batch
        content_batch = _valid_json_response(5)
        mock_provider = _mock_provider()
        mock_provider.complete.return_value = CompletionResult(
            content=content_batch,
            input_tokens=100,
            output_tokens=200,
        )
        mock_create_provider.return_value = mock_provider

        config = _make_classification_config(tmp_path)

        # When
        samples = generate(config=config)

        # Then — samples were generated, output file exists
        assert len(samples) > 0
        assert all(isinstance(s, Sample) for s in samples)
        assert (tmp_path / "output.jsonl").exists()

    @patch("dataset_generator.engine.create_provider")
    def test_config_overrides(self, mock_create_provider, tmp_path):
        """Given config with num_samples=4 and batch_size=2,
        generate() should respect those parameters."""
        content = _valid_json_response(2)
        mock_provider = _mock_provider()
        mock_provider.complete.return_value = CompletionResult(
            content=content,
            input_tokens=50,
            output_tokens=100,
        )
        mock_create_provider.return_value = mock_provider

        config = _make_classification_config(
            tmp_path,
            generation={
                "num_samples": 4,
                "batch_size": 2,
                "max_workers": 1,
                "max_retries": 1,
                "temperature": 0.5,
                "strategy": "direct",
                "strategy_config": {},
            },
        )

        # When
        samples = generate(config=config)

        # Then — respects num_samples cap
        assert len(samples) <= 4

    @patch("dataset_generator.engine.create_provider")
    def test_fewer_samples_than_requested_logs_warning(
        self, mock_create_provider, tmp_path, caplog
    ):
        """Given a provider that returns very few samples,
        generate() should log a warning about the shortfall."""
        # Return only 1 sample per batch — not enough to fill 10 requested
        single = json.dumps(
            [
                {
                    "text": "This is a unique sample that is sufficiently long for validation checks.",
                    "label": "positive",
                }
            ]
        )
        mock_provider = _mock_provider()
        mock_provider.complete.return_value = CompletionResult(
            content=single,
            input_tokens=10,
            output_tokens=20,
        )
        mock_create_provider.return_value = mock_provider

        config = _make_classification_config(tmp_path)

        # When
        with caplog.at_level(logging.WARNING, logger="dataset_generator.engine"):
            samples = generate(config=config)

        # Then — warning about shortfall
        assert len(samples) < 10
        assert any("Delivered" in msg and "requested" in msg for msg in caplog.messages)

    @patch("dataset_generator.engine.create_provider")
    def test_output_file_written(self, mock_create_provider, tmp_path):
        """Given a successful generation, the output JSONL file should be written."""
        content = _valid_json_response(5)
        mock_provider = _mock_provider()
        mock_provider.complete.return_value = CompletionResult(
            content=content,
            input_tokens=100,
            output_tokens=200,
        )
        mock_create_provider.return_value = mock_provider

        output_path = tmp_path / "out" / "dataset.jsonl"
        config = _make_classification_config(
            tmp_path,
            output={"path": str(output_path), "format": "jsonl"},
        )

        # When
        generate(config=config)

        # Then
        assert output_path.exists()
        lines = output_path.read_text().strip().splitlines()
        assert len(lines) > 0
        # Each line should be valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "text" in parsed


# ---------------------------------------------------------------------------
# estimate_run
# ---------------------------------------------------------------------------


class TestEstimateRun:
    def test_calculates_batches_and_cost(self, tmp_path):
        """Given a config, estimate_run should calculate correct batch count and cost."""
        config = _make_classification_config(tmp_path)

        # When
        est = estimate_run(config)

        # Then
        assert est["model"] == "gpt-4o"
        # 10 samples * 1.2 overshoot = 12, / 5 batch_size = ceil(2.4) = 3 batches
        assert est["num_batches"] == 3
        assert est["batch_size"] == 5
        assert est["estimated_input_tokens"] > 0
        assert est["estimated_output_tokens"] > 0
        assert est["estimated_total_tokens"] == (
            est["estimated_input_tokens"] + est["estimated_output_tokens"]
        )
        # gpt-4o is a known model so cost should be > 0
        assert est["estimated_cost_usd"] > 0.0

    def test_unknown_model_zero_cost(self, tmp_path):
        """Given an unknown model, estimate_run should return 0.0 cost."""
        config = _make_classification_config(
            tmp_path,
            provider={
                "kind": "openai",
                "model": "my-custom-local-model",
                "base_url": "http://test",
                "api_key": "test",
            },
        )

        # When
        est = estimate_run(config)

        # Then
        assert est["model"] == "my-custom-local-model"
        assert est["estimated_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# _estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_known_model_pricing(self):
        """Given a known model (gpt-4o), _estimate_cost should apply correct pricing."""
        # gpt-4o: $2.50 / 1M input, $10.00 / 1M output
        cost = _estimate_cost(1_000_000, 1_000_000, "gpt-4o")
        assert cost == pytest.approx(2.50 + 10.00)

    def test_partial_match_pricing(self):
        """Given a model name that partially matches a known model,
        _estimate_cost should still find pricing."""
        cost = _estimate_cost(1_000_000, 1_000_000, "gpt-4o-2024-08-06")
        assert cost > 0.0

    def test_unknown_model_returns_zero(self):
        """Given an unknown model, _estimate_cost should return 0.0."""
        cost = _estimate_cost(1_000_000, 1_000_000, "totally-unknown-model")
        assert cost == 0.0

    def test_zero_tokens_returns_zero(self):
        """Given zero tokens, cost should be zero even for known models."""
        cost = _estimate_cost(0, 0, "gpt-4o")
        assert cost == 0.0

    def test_scaling(self):
        """Cost should scale linearly with token count."""
        cost_1x = _estimate_cost(500_000, 500_000, "gpt-4o")
        cost_2x = _estimate_cost(1_000_000, 1_000_000, "gpt-4o")
        assert cost_2x == pytest.approx(cost_1x * 2)


# ---------------------------------------------------------------------------
# --from-docs integration
# ---------------------------------------------------------------------------


class TestFromDocs:
    @patch("dataset_generator.engine.create_provider")
    def test_from_docs_injects_context_into_messages(self, mock_create_provider, tmp_path):
        """Given --from-docs pointing to a text file, doc content should
        appear in the LLM messages."""
        # Create a document
        doc_path = tmp_path / "docs"
        doc_path.mkdir()
        (doc_path / "knowledge.txt").write_text("Python was created by Guido van Rossum in 1991.")

        content = _valid_json_response(5)
        mock_provider = _mock_provider()
        mock_provider.complete.return_value = CompletionResult(
            content=content, input_tokens=100, output_tokens=200
        )
        mock_create_provider.return_value = mock_provider

        config = _make_classification_config(tmp_path, from_docs=str(doc_path))

        generate(config=config)

        # The provider should have been called with messages containing doc text
        call_args = mock_provider.complete.call_args_list[0]
        messages = call_args[0][0]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Guido van Rossum" in user_msg["content"]


# ---------------------------------------------------------------------------
# --resume integration
# ---------------------------------------------------------------------------


class TestResume:
    @patch("dataset_generator.engine.create_provider")
    def test_resume_creates_and_cleans_checkpoint(self, mock_create_provider, tmp_path):
        """Given resume=True, generate() should save checkpoints and clean up after."""
        content = _valid_json_response(5)
        mock_provider = _mock_provider()
        mock_provider.complete.return_value = CompletionResult(
            content=content, input_tokens=100, output_tokens=200
        )
        mock_create_provider.return_value = mock_provider

        config = _make_classification_config(tmp_path)
        samples = generate(config=config, resume=True)

        assert len(samples) > 0
        # Checkpoint should be cleaned up after success
        checkpoint_dir = Path(".dg_checkpoints")
        assert not checkpoint_dir.exists() or not list(checkpoint_dir.iterdir())


# ---------------------------------------------------------------------------
# quality pipeline steps config
# ---------------------------------------------------------------------------


class TestQualityPipelineConfig:
    @patch("dataset_generator.engine.create_provider")
    def test_quality_steps_from_config(self, mock_create_provider, tmp_path):
        """Given quality.steps in config, the pipeline should run those steps."""
        content = _valid_json_response(5)
        mock_provider = _mock_provider()
        mock_provider.complete.return_value = CompletionResult(
            content=content, input_tokens=100, output_tokens=200
        )
        mock_create_provider.return_value = mock_provider

        config = _make_classification_config(
            tmp_path,
            quality={
                "similarity_threshold": 0.85,
                "min_length": 10,
                "max_length": 10000,
                "steps": [
                    {"diversity": {}},
                ],
            },
        )

        samples = generate(config=config)

        # Diversity is report-only, so all samples should still be there
        assert len(samples) > 0


class TestAsyncGenerate:
    @patch("dataset_generator.engine.create_provider")
    def test_async_full_pipeline(self, mock_create_provider, tmp_path):
        """async_generate() should produce samples just like sync generate()."""
        import asyncio

        from dataset_generator.engine import async_generate

        content = _valid_json_response(5)
        mock_provider = _mock_provider()
        mock_provider.complete.return_value = CompletionResult(
            content=content, input_tokens=100, output_tokens=200
        )
        # async_complete falls back to sync via asyncio.to_thread
        mock_provider.async_complete = MagicMock(
            return_value=CompletionResult(content=content, input_tokens=100, output_tokens=200)
        )

        # Make async_complete a coroutine
        async def _async_complete(*args, **kwargs):
            return CompletionResult(content=content, input_tokens=100, output_tokens=200)

        mock_provider.async_complete = _async_complete
        mock_create_provider.return_value = mock_provider

        config = _make_classification_config(tmp_path)
        samples = asyncio.run(async_generate(config=config))

        assert len(samples) > 0
        assert all(isinstance(s, Sample) for s in samples)
        assert (tmp_path / "output.jsonl").exists()


class TestBuildQualityPipeline:
    def test_builds_known_steps(self):
        from dataset_generator.engine import _build_quality_pipeline

        pipeline = _build_quality_pipeline(
            [
                {"pii": {"action": "flag"}},
                {"language": {"expected": "en"}},
                {"diversity": {}},
            ]
        )

        assert len(pipeline.steps) == 3

    def test_skips_unknown_steps(self, caplog):
        from dataset_generator.engine import _build_quality_pipeline

        with caplog.at_level(logging.WARNING):
            pipeline = _build_quality_pipeline(
                [
                    {"diversity": {}},
                    {"nonexistent": {}},
                ]
            )

        assert len(pipeline.steps) == 1
        assert any("Unknown quality step" in msg for msg in caplog.messages)
