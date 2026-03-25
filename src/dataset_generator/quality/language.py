"""Language detection — trigram-based heuristic, no external dependencies."""

from __future__ import annotations

import logging
from collections import Counter

from dataset_generator.quality.pipeline import StepReport
from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)

# Top trigram profiles per language. Built from frequency analysis of representative
# text corpora. Each list contains the ~30 most common character trigrams (lowercased).
# This is a lightweight heuristic — not a substitute for proper language detection.
_LANGUAGE_PROFILES: dict[str, list[str]] = {
    "en": [
        " th",
        "the",
        "he ",
        "nd ",
        "ing",
        " an",
        "and",
        "ed ",
        " in",
        "ion",
        "tion",
        " of",
        "of ",
        "er ",
        " to",
        "to ",
        "in ",
        "ent",
        "es ",
        "is ",
        "re ",
        "on ",
        "an ",
        " is",
        " re",
        " co",
        "or ",
        "at ",
        " be",
        "ng ",
    ],
    "es": [
        " de",
        "de ",
        " la",
        " en",
        "en ",
        "la ",
        "ión",
        "ció",
        " el",
        "el ",
        "os ",
        "es ",
        " co",
        "ent",
        " qu",
        "que",
        "ue ",
        "as ",
        " lo",
        "los",
        "do ",
        "on ",
        " se",
        "se ",
        "al ",
        " un",
        "er ",
        "re ",
        "ón ",
        "aci",
    ],
    "fr": [
        " de",
        "de ",
        " le",
        "les",
        "es ",
        "ent",
        " la",
        "la ",
        " co",
        "le ",
        "on ",
        "tion",
        "ion",
        " en",
        "en ",
        " pa",
        " qu",
        "que",
        "ue ",
        "re ",
        "ons",
        " et",
        "et ",
        "ns ",
        " un",
        "nt ",
        "ne ",
        " le",
        "men",
        " de",
    ],
    "de": [
        " de",
        "die",
        "der",
        "en ",
        " di",
        "er ",
        "nd ",
        "ein",
        " ei",
        "und",
        " un",
        "den",
        "sch",
        "ich",
        "che",
        "ung",
        "in ",
        " da",
        " de",
        "te ",
        "gen",
        "ine",
        "ch ",
        "ie ",
        " be",
        "eit",
        " au",
        "ber",
        "ier",
        "ver",
    ],
    "pt": [
        " de",
        "de ",
        " co",
        " qu",
        "que",
        "ue ",
        "os ",
        "ção",
        "ão ",
        " do",
        "do ",
        "ent",
        " se",
        " da",
        "da ",
        " em",
        "em ",
        "as ",
        " um",
        "es ",
        "men",
        " pa",
        "nte",
        "to ",
        " no",
        " os",
        "al ",
        "er ",
        "com",
        "ade",
    ],
    "it": [
        " di",
        "di ",
        " de",
        "la ",
        " la",
        " in",
        "che",
        "ion",
        " co",
        "lla",
        "to ",
        " il",
        "il ",
        "del",
        "ell",
        "ent",
        "ato",
        "one",
        "ne ",
        "re ",
        "per",
        " pe",
        "le ",
        " ch",
        "con",
        " un",
        "in ",
        "ta ",
        "zione",
        "tti",
    ],
    "ru": [
        " \u043f\u0440",
        " \u043d\u0430",
        " \u043f\u043e",
        "\u0441\u0442\u0430",
        "\u043d\u0430 ",
        "\u0442\u043e ",
        " \u043a\u043e",
        "\u0435\u043d\u0438",
        " \u043d\u0435",
        " \u0432 ",
        "\u043e\u0433\u043e",
        " \u043e\u0431",
        " \u0432\u0441",
        "\u043d\u0438\u0435",
        "\u043d\u044b\u0445",
        "\u0430\u0442\u044c",
        "\u0435\u0441\u0442",
        " \u0438 ",
        " \u0441\u043e",
        "\u0442\u044c ",
    ],
    "zh": [
        "的 ",
        " 的",
        "是 ",
        " 是",
        "了 ",
        " 了",
        "在 ",
        " 在",
        "有 ",
        " 有",
        "人 ",
        " 人",
        "不 ",
        " 不",
        "他 ",
        " 他",
        "这 ",
        " 这",
        "我 ",
        " 我",
        "们 ",
        " 们",
        "中 ",
        " 中",
        "大 ",
        " 大",
        "来 ",
        " 来",
        "上 ",
        " 上",
    ],
    "ja": [
        "の ",
        " の",
        "は ",
        " は",
        "に ",
        " に",
        "を ",
        " を",
        "た ",
        " た",
        "て ",
        " て",
        "で ",
        " で",
        "と ",
        " と",
        "が ",
        " が",
        "する",
        "ない",
        "って",
        "れる",
        "いる",
        "から",
        "ます",
        "した",
        "こと",
        "ある",
        "です",
        "られ",
    ],
    "ko": [
        "이 ",
        " 이",
        "의 ",
        " 의",
        "은 ",
        " 은",
        "는 ",
        " 는",
        "에 ",
        " 에",
        "를 ",
        " 를",
        "한 ",
        " 한",
        "고 ",
        " 고",
        "다 ",
        " 다",
        "로 ",
        " 로",
        "하는",
        "에서",
        "으로",
        "것이",
        "하고",
        "되는",
        "있는",
        "하여",
        "대한",
        "있다",
    ],
}


def _text_trigrams(text: str) -> Counter[str]:
    """Extract character trigram frequency profile from text."""
    text = text.lower()
    trigrams: Counter[str] = Counter()
    for i in range(len(text) - 2):
        trigrams[text[i : i + 3]] += 1
    return trigrams


def _detect_language(text: str) -> tuple[str, float]:
    """Detect most likely language from trigram overlap.

    Returns:
        (language_code, confidence) where confidence is fraction of top
        profile trigrams found in the text.
    """
    if len(text) < 10:
        return "unknown", 0.0

    text_tri = set(_text_trigrams(text).keys())
    best_lang = "unknown"
    best_score = 0.0

    for lang, profile in _LANGUAGE_PROFILES.items():
        overlap = sum(1 for tri in profile if tri in text_tri)
        score = overlap / len(profile)
        if score > best_score:
            best_score = score
            best_lang = lang

    return best_lang, round(best_score, 3)


class LanguageFilter:
    """Filter samples by detected language."""

    name: str = "language"

    def __init__(
        self,
        expected: str = "en",
        action: str = "remove",
    ) -> None:
        """Init language filter.

        Args:
            expected: Expected language code (e.g., "en", "es", "fr").
            action: "remove" (drop non-matching) or "flag" (annotate but keep).
        """
        if action not in ("remove", "flag"):
            raise ValueError(f"Invalid action: {action!r}. Must be 'remove' or 'flag'.")
        self.expected = expected
        self.action = action

    def process(self, samples: list[Sample]) -> tuple[list[Sample], StepReport]:
        """Detect language and filter/flag samples."""
        output: list[Sample] = []
        removed = 0
        flagged = 0
        lang_counts: Counter[str] = Counter()

        for sample in samples:
            detected, confidence = _detect_language(sample.text)
            lang_counts[detected] += 1

            if detected == self.expected:
                output.append(sample)
                continue

            if self.action == "remove":
                removed += 1
            else:
                flagged += 1
                sample = sample.model_copy(
                    update={
                        "metadata": {
                            **sample.metadata,
                            "detected_language": detected,
                            "language_confidence": confidence,
                        },
                    },
                )
                output.append(sample)

        if removed or flagged:
            logger.info(
                "Language filter: %d removed, %d flagged (expected=%s, distribution=%s)",
                removed,
                flagged,
                self.expected,
                dict(lang_counts),
            )

        return output, StepReport(
            name=self.name,
            input_count=len(samples),
            output_count=len(output),
            removed=removed,
            flagged=flagged,
            details={"language_distribution": dict(lang_counts)},
        )
