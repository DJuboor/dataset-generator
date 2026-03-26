"""LLM-as-judge quality step — use a (possibly stronger) model to score and filter samples."""

from __future__ import annotations

import json
import logging
import re

from dataset_generator.providers import create_provider
from dataset_generator.quality.pipeline import StepReport
from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """Rate each sample on a scale of 1-5 for quality, relevance, and correctness.

Respond with a JSON array of objects: [{{"index": 0, "score": 4}}, ...]
Only include the JSON array, no other text.

Samples to evaluate:
{samples}"""


class LLMJudge:
    """Score samples using an LLM and filter/flag based on threshold."""

    name: str = "llm_judge"

    def __init__(
        self,
        provider_config: dict | None = None,
        model: str = "",
        prompt_template: str = "",
        threshold: float = 3.0,
        action: str = "remove",
        batch_size: int = 10,
    ) -> None:
        if action not in ("remove", "flag"):
            raise ValueError(f"Invalid action: {action!r}. Must be 'remove' or 'flag'.")

        # Build provider config from shorthand or full config
        if provider_config:
            self._provider_config = provider_config
        elif model:
            self._provider_config = {"kind": "openai", "model": model}
        else:
            raise ValueError("LLMJudge requires either 'provider_config' or 'model'.")

        self.prompt_template = prompt_template or DEFAULT_PROMPT
        self.threshold = threshold
        self.action = action
        self.batch_size = batch_size

    def process(self, samples: list[Sample]) -> tuple[list[Sample], StepReport]:
        """Score samples in batches and filter/flag below threshold."""
        provider = create_provider(self._provider_config)

        scores: list[float] = []
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i : i + self.batch_size]
            batch_scores = self._score_batch(provider, batch)
            scores.extend(batch_scores)

        output: list[Sample] = []
        removed = 0
        flagged = 0

        for sample, score in zip(samples, scores, strict=False):
            if score >= self.threshold:
                output.append(sample)
            elif self.action == "remove":
                removed += 1
            else:
                flagged += 1
                sample = sample.model_copy(
                    update={"metadata": {**sample.metadata, "judge_score": score}},
                )
                output.append(sample)

        if removed or flagged:
            avg_score = sum(scores) / len(scores) if scores else 0
            logger.info(
                "LLM judge: %d removed, %d flagged (threshold=%.1f, avg_score=%.2f)",
                removed,
                flagged,
                self.threshold,
                avg_score,
            )

        return output, StepReport(
            name=self.name,
            input_count=len(samples),
            output_count=len(output),
            removed=removed,
            flagged=flagged,
            details={"scores": scores, "threshold": self.threshold},
        )

    def _score_batch(self, provider, batch: list[Sample]) -> list[float]:
        """Send a batch of samples to the judge model and parse scores."""
        samples_text = "\n".join(f"[{i}] {json.dumps(s.to_dict())}" for i, s in enumerate(batch))
        prompt = self.prompt_template.format(samples=samples_text)

        try:
            result = provider.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return self._parse_scores(result.content, len(batch))
        except Exception as e:
            logger.warning(f"LLM judge scoring failed: {e}")
            # On failure, give all samples the threshold score (don't drop them)
            return [self.threshold] * len(batch)

    def _parse_scores(self, response: str, expected_count: int) -> list[float]:
        """Parse score array from judge response."""
        # Strip thinking tags and markdown
        from dataset_generator.tasks.base import clean_llm_response

        response = clean_llm_response(response)

        try:
            items = json.loads(response)
            score_map = {item["index"]: float(item["score"]) for item in items}
            return [score_map.get(i, self.threshold) for i in range(expected_count)]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: try to extract numbers
            numbers = re.findall(r"\b([1-5](?:\.\d)?)\b", response)
            if len(numbers) >= expected_count:
                return [float(n) for n in numbers[:expected_count]]
            # Can't parse — default to threshold
            logger.debug("Could not parse judge scores, defaulting to threshold")
            return [self.threshold] * expected_count
