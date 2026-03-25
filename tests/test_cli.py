"""Tests for CLI commands."""

import json
from pathlib import Path

from typer.testing import CliRunner

from dataset_generator.cli import app

runner = CliRunner()


def _write_sample_jsonl(path: Path, n: int = 5) -> Path:
    """Write a temp JSONL file with classification samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(n):
            label = "positive" if i % 2 == 0 else "negative"
            sample = {
                "text": f"This is sample number {i} with enough text for validation.",
                "label": label,
            }
            f.write(json.dumps(sample) + "\n")
    return path


# ---------------------------------------------------------------------------
# dg init
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_classification_creates_config(self, tmp_path, monkeypatch):
        """Given 'dg init classification', a config.yaml should be created."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["init", "classification"], catch_exceptions=False)

        assert result.exit_code == 0
        assert (tmp_path / "config.yaml").exists()
        content = (tmp_path / "config.yaml").read_text()
        assert "classification" in content

    def test_init_classification_with_labels(self, tmp_path, monkeypatch):
        """Given '--labels a,b', the config should contain those labels."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app,
            ["init", "classification", "--labels", "spam,ham"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        content = (tmp_path / "config.yaml").read_text()
        assert "'spam'" in content
        assert "'ham'" in content

    def test_init_ner_creates_config(self, tmp_path, monkeypatch):
        """Given 'dg init ner', a NER config should be created."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["init", "ner"], catch_exceptions=False)

        assert result.exit_code == 0
        content = (tmp_path / "config.yaml").read_text()
        assert "ner" in content
        assert "entity_types" in content

    def test_init_sft_creates_config(self, tmp_path, monkeypatch):
        """Given 'dg init sft', an SFT config should be created."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["init", "sft"], catch_exceptions=False)

        assert result.exit_code == 0
        content = (tmp_path / "config.yaml").read_text()
        assert "sft" in content

    def test_init_unknown_task_fails(self, tmp_path, monkeypatch):
        """Given 'dg init foobar', it should fail with an error."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["init", "foobar"], catch_exceptions=False)

        assert result.exit_code != 0
        assert "Unknown task type" in result.output or "foobar" in result.output

    def test_init_custom_output_path(self, tmp_path, monkeypatch):
        """Given '--output custom.yaml', the config should be written there."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app,
            ["init", "classification", "--output", "custom.yaml"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert (tmp_path / "custom.yaml").exists()

    def test_init_ner_with_labels_injects_entity_types(self, tmp_path, monkeypatch):
        """Given 'dg init ner --labels FOOD,DRINK', entity_types should be replaced."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app,
            ["init", "ner", "--labels", "FOOD,DRINK"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        content = (tmp_path / "config.yaml").read_text()
        assert "'FOOD'" in content
        assert "'DRINK'" in content


# ---------------------------------------------------------------------------
# dg info
# ---------------------------------------------------------------------------


class TestInfo:
    def test_info_shows_task_types(self):
        """Given 'dg info', it should display task types, strategies, and formats."""
        result = runner.invoke(app, ["info"], catch_exceptions=False)

        assert result.exit_code == 0
        assert "classification" in result.output
        assert "ner" in result.output
        assert "qa" in result.output
        assert "preference" in result.output
        assert "sft" in result.output

    def test_info_shows_strategies(self):
        """Given 'dg info', strategy table should be present."""
        result = runner.invoke(app, ["info"], catch_exceptions=False)

        assert result.exit_code == 0
        assert "direct" in result.output
        assert "few_shot" in result.output
        assert "persona" in result.output

    def test_info_shows_formats(self):
        """Given 'dg info', output formats table should be present."""
        result = runner.invoke(app, ["info"], catch_exceptions=False)

        assert result.exit_code == 0
        assert "jsonl" in result.output
        assert "csv" in result.output
        assert "parquet" in result.output


# ---------------------------------------------------------------------------
# dg validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_produces_clean_output(self, tmp_path):
        """Given a valid JSONL file, 'dg validate' should produce a cleaned file."""
        input_path = _write_sample_jsonl(tmp_path / "data.jsonl", n=5)
        output_path = tmp_path / "data_clean.jsonl"

        result = runner.invoke(
            app,
            ["validate", str(input_path), "--output", str(output_path)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert output_path.exists()
        lines = output_path.read_text().strip().splitlines()
        assert len(lines) > 0
        # Each line should be valid JSON with text field
        for line in lines:
            parsed = json.loads(line)
            assert "text" in parsed

    def test_validate_reports_stats(self, tmp_path):
        """Given a JSONL file, 'dg validate' should display a validation report."""
        input_path = _write_sample_jsonl(tmp_path / "data.jsonl", n=5)

        result = runner.invoke(
            app,
            ["validate", str(input_path), "--output", str(tmp_path / "clean.jsonl")],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "Validation Report" in result.output or "samples" in result.output.lower()

    def test_validate_default_output_path(self, tmp_path):
        """Given no --output flag, validate should write to <input>_clean.jsonl."""
        input_path = _write_sample_jsonl(tmp_path / "dataset.jsonl", n=3)

        result = runner.invoke(
            app,
            ["validate", str(input_path)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        expected = tmp_path / "dataset_clean.jsonl"
        assert expected.exists()

    def test_validate_with_threshold(self, tmp_path):
        """Given --threshold flag, validate should accept it."""
        input_path = _write_sample_jsonl(tmp_path / "data.jsonl", n=5)

        result = runner.invoke(
            app,
            [
                "validate",
                str(input_path),
                "--threshold",
                "0.9",
                "--output",
                str(tmp_path / "out.jsonl"),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# dg export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_jsonl_to_csv(self, tmp_path):
        """Given a JSONL file, 'dg export --format csv' should produce a CSV."""
        input_path = _write_sample_jsonl(tmp_path / "data.jsonl", n=5)
        output_path = tmp_path / "data.csv"

        result = runner.invoke(
            app,
            ["export", str(input_path), "--format", "csv", "--output", str(output_path)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert output_path.exists()
        content = output_path.read_text()
        assert "text" in content  # CSV header
        # Should have header + data rows
        lines = content.strip().splitlines()
        assert len(lines) == 6  # header + 5 samples

    def test_export_default_output_path(self, tmp_path):
        """Given no --output flag, export should derive path from input."""
        input_path = _write_sample_jsonl(tmp_path / "dataset.jsonl", n=3)

        result = runner.invoke(
            app,
            ["export", str(input_path), "--format", "csv"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        expected = tmp_path / "dataset.csv"
        assert expected.exists()

    def test_export_reports_count(self, tmp_path):
        """Given a JSONL file, export should report the number of samples."""
        input_path = _write_sample_jsonl(tmp_path / "data.jsonl", n=4)

        result = runner.invoke(
            app,
            ["export", str(input_path), "--format", "csv", "--output", str(tmp_path / "out.csv")],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "4" in result.output


# ---------------------------------------------------------------------------
# dg generate --dry-run
# ---------------------------------------------------------------------------


class TestGenerateDryRun:
    def test_dry_run_shows_estimate(self, tmp_path, monkeypatch):
        """Given '--dry-run --task classification --labels a,b',
        generate should show an estimate table without actually running."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app,
            [
                "generate",
                "--dry-run",
                "--task",
                "classification",
                "--labels",
                "happy,sad",
                "--num-samples",
                "20",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "Dry Run Estimate" in result.output
        assert "Model" in result.output
        assert "Batches" in result.output
        assert "cost" in result.output.lower()

    def test_dry_run_does_not_create_output_file(self, tmp_path, monkeypatch):
        """Given --dry-run, no output file should be created."""
        monkeypatch.chdir(tmp_path)

        runner.invoke(
            app,
            [
                "generate",
                "--dry-run",
                "--task",
                "classification",
                "--labels",
                "a,b",
                "--output",
                str(tmp_path / "should_not_exist.jsonl"),
            ],
            catch_exceptions=False,
        )

        assert not (tmp_path / "should_not_exist.jsonl").exists()

    def test_dry_run_with_model_override(self, tmp_path, monkeypatch):
        """Given --model override, the estimate should reflect the specified model."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app,
            [
                "generate",
                "--dry-run",
                "--task",
                "classification",
                "--labels",
                "a,b",
                "--model",
                "gpt-4o-mini",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "gpt-4o-mini" in result.output


# ---------------------------------------------------------------------------
# dg push
# ---------------------------------------------------------------------------


class TestPush:
    def test_push_calls_hub(self, tmp_path):
        """Given a JSONL file and repo ID, 'dg push' should call push_to_hub."""
        from unittest.mock import patch

        input_path = _write_sample_jsonl(tmp_path / "data.jsonl", n=3)

        with patch("dataset_generator.hub.push_to_hub") as mock_push:
            mock_push.return_value = "https://huggingface.co/datasets/user/test-ds"

            result = runner.invoke(
                app,
                ["push", "user/test-ds", "--file", str(input_path)],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        assert "Published" in result.output or "huggingface" in result.output.lower()
        mock_push.assert_called_once()

    def test_push_with_private_flag(self, tmp_path):
        """Given --private, push should pass private=True."""
        from unittest.mock import patch

        input_path = _write_sample_jsonl(tmp_path / "data.jsonl", n=2)

        with patch("dataset_generator.hub.push_to_hub") as mock_push:
            mock_push.return_value = "https://huggingface.co/datasets/user/ds"

            runner.invoke(
                app,
                ["push", "user/ds", "--file", str(input_path), "--private"],
                catch_exceptions=False,
            )

        assert mock_push.call_args[1]["private"] is True
