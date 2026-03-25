"""Generation strategies — control how prompts are varied for diversity."""

from dataset_generator.strategies.adversarial import AdversarialStrategy
from dataset_generator.strategies.base import Strategy
from dataset_generator.strategies.cot import ChainOfThoughtStrategy
from dataset_generator.strategies.direct import DirectStrategy
from dataset_generator.strategies.evolinstruct import EvolInstructStrategy
from dataset_generator.strategies.few_shot import FewShotStrategy
from dataset_generator.strategies.persona import PersonaStrategy

__all__ = [
    "AdversarialStrategy",
    "ChainOfThoughtStrategy",
    "DirectStrategy",
    "EvolInstructStrategy",
    "FewShotStrategy",
    "PersonaStrategy",
    "Strategy",
    "create_strategy",
]

STRATEGY_REGISTRY: dict[str, type] = {
    "direct": DirectStrategy,
    "few_shot": FewShotStrategy,
    "persona": PersonaStrategy,
    "cot": ChainOfThoughtStrategy,
    "adversarial": AdversarialStrategy,
    "evolinstruct": EvolInstructStrategy,
}


def create_strategy(name: str, config: dict | None = None) -> Strategy:
    """Create a strategy by name, passing through strategy-specific config.

    Args:
        name: Strategy name (direct, few_shot, persona, cot, adversarial, evolinstruct).
        config: Strategy-specific config dict (e.g. examples for few_shot, personas for persona).
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")
    kwargs = config or {}
    return STRATEGY_REGISTRY[name](**kwargs)
