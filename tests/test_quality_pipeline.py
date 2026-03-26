"""Tests for quality pipeline, PII filter, balance checker, diversity reporter, toxicity, language."""

from __future__ import annotations

import pytest

from dataset_generator.quality.balance import BalanceChecker
from dataset_generator.quality.diversity import DiversityReporter
from dataset_generator.quality.language import LanguageFilter, _detect_language, _text_trigrams
from dataset_generator.quality.pii import PIIFilter
from dataset_generator.quality.pipeline import QualityPipeline, StepReport
from dataset_generator.quality.toxicity import ToxicityFilter
from dataset_generator.tasks.base import Sample


def _make_sample(text: str, label: str | None = None) -> Sample:
    return Sample(text=text, label=label)


# ---------------------------------------------------------------------------
# QualityPipeline
# ---------------------------------------------------------------------------


class TestQualityPipeline:
    def test_runs_steps_in_sequence(self):
        class PassthroughStep:
            name = "passthrough"

            def process(self, samples: list[Sample]) -> tuple[list[Sample], StepReport]:
                return samples, StepReport(
                    name=self.name,
                    input_count=len(samples),
                    output_count=len(samples),
                    removed=0,
                )

        class DropFirstStep:
            name = "drop_first"

            def process(self, samples: list[Sample]) -> tuple[list[Sample], StepReport]:
                output = samples[1:]
                return output, StepReport(
                    name=self.name,
                    input_count=len(samples),
                    output_count=len(output),
                    removed=1,
                )

        pipeline = QualityPipeline(steps=[PassthroughStep(), DropFirstStep()])
        samples = [_make_sample("a"), _make_sample("b"), _make_sample("c")]

        result, report = pipeline.run(samples)

        assert len(result) == 2
        assert report.input_count == 3
        assert report.output_count == 2
        assert len(report.step_reports) == 2
        assert report.step_reports[1].removed == 1

    def test_empty_pipeline(self):
        pipeline = QualityPipeline(steps=[])
        samples = [_make_sample("hello")]

        result, report = pipeline.run(samples)

        assert result == samples
        assert report.output_count == 1


# ---------------------------------------------------------------------------
# PIIFilter
# ---------------------------------------------------------------------------


class TestPIIFilter:
    def test_detects_email(self):
        filt = PIIFilter(action="flag", patterns=["email"])
        samples = [_make_sample("Contact me at john@example.com for details.")]

        result, report = filt.process(samples)

        assert len(result) == 1
        assert "pii_detected" in result[0].metadata
        assert report.flagged == 1

    def test_detects_phone(self):
        filt = PIIFilter(action="flag", patterns=["phone_us"])
        samples = [_make_sample("Call me at 555-123-4567 anytime.")]

        result, report = filt.process(samples)

        assert len(result) == 1
        assert "pii_detected" in result[0].metadata
        assert report.flagged == 1

    def test_remove_action_drops_sample(self):
        filt = PIIFilter(action="remove", patterns=["email"])
        samples = [
            _make_sample("My email is test@test.com"),
            _make_sample("No PII here"),
        ]

        result, report = filt.process(samples)

        assert len(result) == 1
        assert result[0].text == "No PII here"
        assert report.removed == 1

    def test_flag_action_keeps_sample(self):
        filt = PIIFilter(action="flag", patterns=["email"])
        samples = [_make_sample("Reach me at user@domain.org")]

        result, report = filt.process(samples)

        assert len(result) == 1
        assert report.flagged == 1
        assert report.removed == 0

    def test_clean_text_passes_through(self):
        filt = PIIFilter(action="remove")
        samples = [_make_sample("Just a normal sentence.")]

        result, report = filt.process(samples)

        assert len(result) == 1
        assert report.removed == 0


# ---------------------------------------------------------------------------
# BalanceChecker
# ---------------------------------------------------------------------------


class TestBalanceChecker:
    def test_report_mode_reports_imbalance(self):
        checker = BalanceChecker(strategy="report", max_ratio=2.0)
        samples = [_make_sample("pos text", label="positive")] * 10 + [
            _make_sample("neg text", label="negative")
        ] * 2

        result, report = checker.process(samples)

        assert len(result) == 12
        assert report.removed == 0
        assert report.details["imbalanced"] is True

    def test_undersample_mode_trims_majority(self):
        checker = BalanceChecker(strategy="undersample", max_ratio=2.0)
        samples = [_make_sample("pos text", label="positive")] * 10 + [
            _make_sample("neg text", label="negative")
        ] * 2

        result, report = checker.process(samples)

        positive_count = sum(1 for s in result if s.label == "positive")
        negative_count = sum(1 for s in result if s.label == "negative")
        assert positive_count == 4
        assert negative_count == 2
        assert report.removed == 6

    def test_balanced_data_unchanged(self):
        checker = BalanceChecker(strategy="undersample", max_ratio=2.0)
        samples = [_make_sample("a", label="A")] * 5 + [_make_sample("b", label="B")] * 5

        result, report = checker.process(samples)

        assert len(result) == 10
        assert report.removed == 0

    def test_no_labels_skips(self):
        checker = BalanceChecker(strategy="undersample")
        samples = [_make_sample("no label")]

        result, report = checker.process(samples)

        assert len(result) == 1
        assert "skipped" in report.details


# ---------------------------------------------------------------------------
# DiversityReporter
# ---------------------------------------------------------------------------


class TestDiversityReporter:
    def test_computes_distinct_n_metrics(self):
        reporter = DiversityReporter()
        samples = [
            _make_sample("the quick brown fox jumps over the lazy dog"),
            _make_sample("a completely different sentence with unique words"),
        ]

        _, report = reporter.process(samples)

        assert report.details["distinct_1"] > 0
        assert report.details["distinct_2"] > 0
        assert report.details["distinct_3"] > 0
        assert report.details["vocabulary_size"] > 0

    def test_returns_all_samples_unchanged(self):
        reporter = DiversityReporter()
        samples = [_make_sample("hello world"), _make_sample("foo bar")]

        result, report = reporter.process(samples)

        assert len(result) == 2
        assert report.removed == 0

    def test_empty_samples(self):
        reporter = DiversityReporter()
        result, report = reporter.process([])

        assert result == []
        assert report.input_count == 0


# ---------------------------------------------------------------------------
# ToxicityFilter
# ---------------------------------------------------------------------------


class TestToxicityFilter:
    def test_detects_toxic_keyword(self):
        filt = ToxicityFilter(action="flag", keywords=["badword"])
        samples = [_make_sample("This contains a badword in it.")]

        result, report = filt.process(samples)

        assert len(result) == 1
        assert report.flagged == 1
        assert "toxic_keywords" in result[0].metadata

    def test_remove_drops_sample(self):
        filt = ToxicityFilter(action="remove", keywords=["toxic"])
        samples = [
            _make_sample("This is toxic content"),
            _make_sample("This is clean content"),
        ]

        result, report = filt.process(samples)

        assert len(result) == 1
        assert result[0].text == "This is clean content"
        assert report.removed == 1

    def test_clean_text_passes(self):
        filt = ToxicityFilter(action="remove", keywords=["badword"])
        samples = [_make_sample("Nothing bad here at all.")]

        result, report = filt.process(samples)

        assert len(result) == 1
        assert report.removed == 0

    def test_case_insensitive(self):
        filt = ToxicityFilter(action="flag", keywords=["badword"])
        samples = [_make_sample("This has BADWORD in it.")]

        _, report = filt.process(samples)

        assert report.flagged == 1

    def test_word_boundary_prevents_false_positive(self):
        filt = ToxicityFilter(action="flag", keywords=["bad"])
        samples = [_make_sample("This badge is nice.")]

        _, report = filt.process(samples)

        assert report.flagged == 0


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------


class TestTextTrigrams:
    def test_extracts_trigrams(self):
        tri = _text_trigrams("hello")
        assert tri["hel"] == 1
        assert tri["ell"] == 1
        assert tri["llo"] == 1

    def test_lowercases(self):
        tri = _text_trigrams("HELLO")
        assert "hel" in tri

    def test_short_text(self):
        tri = _text_trigrams("ab")
        assert len(tri) == 0


class TestDetectLanguage:
    def test_english(self):
        text = "The quick brown fox jumps over the lazy dog and the cat was sitting there"
        lang, confidence = _detect_language(text)
        assert lang == "en"
        assert confidence > 0.2

    def test_spanish(self):
        text = "El rápido zorro marrón salta sobre el perro perezoso que estaba en la casa"
        lang, confidence = _detect_language(text)
        assert lang == "es"
        assert confidence > 0.1

    def test_short_text_returns_unknown(self):
        lang, confidence = _detect_language("hi")
        assert lang == "unknown"
        assert confidence == 0.0

    def test_german(self):
        text = "Die schnelle braune Fuchs springt über den faulen Hund und die Katze"
        lang, _confidence = _detect_language(text)
        assert lang == "de"


# ---------------------------------------------------------------------------
# LanguageFilter
# ---------------------------------------------------------------------------


class TestLanguageFilter:
    def test_keeps_matching_language(self):
        filt = LanguageFilter(expected="en", action="remove")
        samples = [_make_sample("The quick brown fox jumps over the lazy dog and the cat")]

        result, report = filt.process(samples)

        assert len(result) == 1
        assert report.removed == 0

    def test_removes_non_matching(self):
        filt = LanguageFilter(expected="en", action="remove")
        samples = [
            _make_sample("The quick brown fox jumps over the lazy dog and the cat"),
            _make_sample("El rápido zorro marrón salta sobre el perro perezoso en la casa"),
        ]

        result, report = filt.process(samples)

        assert len(result) == 1
        assert report.removed == 1

    def test_flag_action_keeps_but_annotates(self):
        filt = LanguageFilter(expected="en", action="flag")
        samples = [
            _make_sample("El rápido zorro marrón salta sobre el perro perezoso en la casa"),
        ]

        result, report = filt.process(samples)

        assert len(result) == 1
        assert report.flagged == 1
        assert "detected_language" in result[0].metadata
        assert "language_confidence" in result[0].metadata

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="Invalid action"):
            LanguageFilter(expected="en", action="explode")

    def test_report_contains_distribution(self):
        filt = LanguageFilter(expected="en")
        samples = [_make_sample("The quick brown fox jumps over the lazy dog and the cat")]

        _, report = filt.process(samples)

        assert "language_distribution" in report.details


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class TestLLMJudge:
    def test_scores_and_filters(self):
        from unittest.mock import MagicMock

        from dataset_generator.providers.base import CompletionResult
        from dataset_generator.quality.llm_judge import LLMJudge

        judge = LLMJudge(model="gpt-4o", threshold=3.0, action="remove")
        # Mock the provider creation
        mock_provider = MagicMock()
        mock_provider.complete.return_value = CompletionResult(
            content='[{"index": 0, "score": 4.5}, {"index": 1, "score": 2.0}]',
            input_tokens=50,
            output_tokens=20,
        )

        from unittest.mock import patch

        with patch(
            "dataset_generator.quality.llm_judge.create_provider", return_value=mock_provider
        ):
            samples = [_make_sample("Good sample"), _make_sample("Bad sample")]
            result, report = judge.process(samples)

        assert len(result) == 1  # Only the 4.5 score passes threshold 3.0
        assert report.removed == 1

    def test_flag_action_keeps_with_score(self):
        from unittest.mock import MagicMock, patch

        from dataset_generator.providers.base import CompletionResult
        from dataset_generator.quality.llm_judge import LLMJudge

        judge = LLMJudge(model="gpt-4o", threshold=3.0, action="flag")
        mock_provider = MagicMock()
        mock_provider.complete.return_value = CompletionResult(
            content='[{"index": 0, "score": 2.0}]',
            input_tokens=10,
            output_tokens=10,
        )

        with patch(
            "dataset_generator.quality.llm_judge.create_provider", return_value=mock_provider
        ):
            samples = [_make_sample("Mediocre sample")]
            result, report = judge.process(samples)

        assert len(result) == 1
        assert report.flagged == 1
        assert "judge_score" in result[0].metadata

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="Invalid action"):
            from dataset_generator.quality.llm_judge import LLMJudge

            LLMJudge(model="gpt-4o", action="explode")

    def test_requires_model_or_config(self):
        with pytest.raises(ValueError, match="requires"):
            from dataset_generator.quality.llm_judge import LLMJudge

            LLMJudge()
