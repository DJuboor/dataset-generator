"""Auto-prompt optimization — wraps any strategy with a calibration + refinement cycle."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class AutoOptimizeStrategy:
    """Wraps a base strategy with automatic prompt refinement.

    On the first batch (batch_index=0), runs a calibration cycle:
    1. Generate a small batch with the base strategy
    2. Self-critique the outputs via the same LLM
    3. Extract improvement suggestions
    4. Prepend the refinement to all subsequent batches

    This is a strategy *wrapper*, not a standalone strategy.
    """

    def __init__(self, base_strategy, provider, task, calibration_size: int = 5):
        self.base = base_strategy
        self.provider = provider
        self.task = task
        self.calibration_size = calibration_size
        self._refinement: str = ""
        self._calibrated = False

    def apply(self, messages: list[dict[str, str]], batch_index: int) -> list[dict[str, str]]:
        """Apply base strategy + inject refinement from calibration."""
        messages = self.base.apply(messages, batch_index)

        # Run calibration on first batch
        if not self._calibrated:
            self._calibrate(messages)
            self._calibrated = True

        # Inject refinement into all batches
        if self._refinement:
            modified = [m.copy() for m in messages]
            for m in modified:
                if m["role"] == "system":
                    m["content"] += f"\n\n{self._refinement}"
                    break
            return modified

        return messages

    def _calibrate(self, messages: list[dict[str, str]]) -> None:
        """Run calibration: generate small batch, self-critique, extract refinement."""
        try:
            # Generate calibration samples
            result = self.provider.complete(messages, temperature=0.7)
            samples = self.task.parse_response(result.content)

            if not samples:
                logger.info("Auto-optimize: calibration produced 0 samples, skipping refinement")
                return

            # Self-critique
            samples_text = "\n".join(
                f"- {json.dumps(s.to_dict())}" for s in samples[: self.calibration_size]
            )
            critique_prompt = (
                f"Rate these generated samples 1-5 on quality, relevance, and diversity. "
                f"Then provide 2-3 specific, actionable improvements.\n\n{samples_text}\n\n"
                f"Respond with:\nRating: X/5\nImprovements:\n- improvement 1\n- improvement 2"
            )

            critique_result = self.provider.complete(
                messages=[{"role": "user", "content": critique_prompt}],
                temperature=0.3,
            )

            # Extract improvements
            content = critique_result.content
            if "Improvement" in content or "improvement" in content:
                # Extract everything after "Improvements:" or similar
                for marker in ["Improvements:", "improvements:", "Improvement:"]:
                    if marker in content:
                        improvements = content.split(marker, 1)[1].strip()
                        self._refinement = (
                            "IMPORTANT: Apply these quality improvements to your generation:\n"
                            + improvements[:500]  # Cap length
                        )
                        logger.info(
                            "Auto-optimize: calibration complete, refinement applied (%d chars)",
                            len(self._refinement),
                        )
                        return

            logger.info("Auto-optimize: could not extract improvements from critique")

        except Exception as e:
            logger.warning(f"Auto-optimize calibration failed: {e}")
