"""LLM provider abstraction."""

from dataset_generator.providers.base import CompletionResult, Provider
from dataset_generator.providers.openai_compat import OpenAIProvider

__all__ = ["CompletionResult", "OpenAIProvider", "Provider", "create_provider"]


def create_provider(config: dict) -> Provider:
    """Create a provider from config dict.

    Args:
        config: Provider config with 'kind', 'base_url', 'api_key', 'model'.

    Returns:
        Configured Provider instance.
    """
    kind = config.get("kind", "openai")
    if kind in ("openai", "vllm", "ollama", "litellm", "together", "groq"):
        return OpenAIProvider(
            base_url=config.get("base_url"),
            api_key=config.get("api_key", ""),
            model=config["model"],
        )
    raise ValueError(f"Unknown provider kind: {kind}")
