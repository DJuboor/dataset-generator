"""Tests for generation strategies."""

from __future__ import annotations

from dataset_generator.strategies.adversarial import AdversarialStrategy
from dataset_generator.strategies.cot import ChainOfThoughtStrategy
from dataset_generator.strategies.direct import DirectStrategy
from dataset_generator.strategies.evolinstruct import EvolInstructStrategy
from dataset_generator.strategies.few_shot import FewShotStrategy
from dataset_generator.strategies.persona import PersonaStrategy

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Generate examples."},
]


class TestDirectStrategy:
    def test_first_batch_unchanged(self):
        strategy = DirectStrategy()
        result = strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        assert result == SAMPLE_MESSAGES

    def test_subsequent_batches_add_diversity_hint(self):
        strategy = DirectStrategy()
        result = strategy.apply(SAMPLE_MESSAGES, batch_index=1)
        assert "batch 2" in result[-1]["content"].lower()

    def test_does_not_mutate_original(self):
        strategy = DirectStrategy()
        original_content = SAMPLE_MESSAGES[-1]["content"]
        strategy.apply(SAMPLE_MESSAGES, batch_index=5)
        assert SAMPLE_MESSAGES[-1]["content"] == original_content


class TestPersonaStrategy:
    def test_injects_persona(self):
        strategy = PersonaStrategy()
        result = strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        assert "perspective of" in result[-1]["content"]

    def test_rotates_personas(self):
        strategy = PersonaStrategy(personas=["writer", "engineer"])
        r0 = strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        r1 = strategy.apply(SAMPLE_MESSAGES, batch_index=1)
        assert "writer" in r0[-1]["content"]
        assert "engineer" in r1[-1]["content"]


class TestFewShotStrategy:
    def test_with_examples(self):
        strategy = FewShotStrategy(examples=[{"text": "example"}])
        result = strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        assert "example" in result[-1]["content"]

    def test_without_examples_falls_back(self):
        strategy = FewShotStrategy()
        result = strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        # Falls back to DirectStrategy behavior
        assert result == SAMPLE_MESSAGES


class TestChainOfThoughtStrategy:
    def test_injects_reasoning_hint(self):
        strategy = ChainOfThoughtStrategy(reasoning_depth="moderate")
        result = strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        # Should contain chain-of-thought instruction in the last message
        assert (
            "step-by-step" in result[-1]["content"].lower()
            or "step" in result[-1]["content"].lower()
        )
        # Original messages should not be mutated
        assert SAMPLE_MESSAGES[-1]["content"] == "Generate examples."

    def test_diversity_hint_on_batch_gt_0(self):
        strategy = ChainOfThoughtStrategy(reasoning_depth="brief")
        result = strategy.apply(SAMPLE_MESSAGES, batch_index=2)
        content = result[-1]["content"]
        # Should contain both reasoning and diversity hints
        assert "reasoning" in content.lower() or "outline" in content.lower()
        assert "batch 3" in content.lower()
        assert "different" in content.lower()

    def test_no_diversity_hint_on_batch_0(self):
        strategy = ChainOfThoughtStrategy(reasoning_depth="brief")
        result = strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        content = result[-1]["content"]
        assert "batch" not in content.lower() or "batch 1" not in content.lower()

    def test_invalid_depth_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown reasoning_depth"):
            ChainOfThoughtStrategy(reasoning_depth="extreme")


class TestAdversarialStrategy:
    def test_injects_adversarial_mode(self):
        strategy = AdversarialStrategy()
        result = strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        content = result[-1]["content"]
        assert "adversarial" in content.lower()
        assert "challenging" in content.lower()

    def test_rotates_modes(self):
        strategy = AdversarialStrategy(modes=["ambiguous", "sarcasm", "negation"])
        r0 = strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        r1 = strategy.apply(SAMPLE_MESSAGES, batch_index=1)
        r2 = strategy.apply(SAMPLE_MESSAGES, batch_index=2)
        # Each batch should use a different mode
        assert "ambiguous" in r0[-1]["content"]
        assert "sarcasm" in r1[-1]["content"]
        assert "negation" in r2[-1]["content"]

    def test_wraps_around(self):
        strategy = AdversarialStrategy(modes=["modeA", "modeB"])
        r2 = strategy.apply(SAMPLE_MESSAGES, batch_index=2)
        assert "modeA" in r2[-1]["content"]

    def test_does_not_mutate_original(self):
        strategy = AdversarialStrategy()
        original_content = SAMPLE_MESSAGES[-1]["content"]
        strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        assert SAMPLE_MESSAGES[-1]["content"] == original_content


class TestEvolInstructStrategy:
    def test_injects_evolution_prompt(self):
        strategy = EvolInstructStrategy(evolution_rounds=1)
        result = strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        content = result[-1]["content"]
        assert "evolve" in content.lower() or "evolution" in content.lower()
        assert "sophisticated" in content.lower() or "challenging" in content.lower()

    def test_rotates_prompts(self):
        strategy = EvolInstructStrategy(evolution_rounds=1)
        r0 = strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        r1 = strategy.apply(SAMPLE_MESSAGES, batch_index=1)
        # Different batches should get different evolution instructions
        assert r0[-1]["content"] != r1[-1]["content"]

    def test_stacks_rounds(self):
        strategy = EvolInstructStrategy(evolution_rounds=3)
        result = strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        content = result[-1]["content"]
        # Should contain multiple evolution bullet points
        assert content.count("- ") >= 3

    def test_does_not_mutate_original(self):
        strategy = EvolInstructStrategy()
        original_content = SAMPLE_MESSAGES[-1]["content"]
        strategy.apply(SAMPLE_MESSAGES, batch_index=0)
        assert SAMPLE_MESSAGES[-1]["content"] == original_content
