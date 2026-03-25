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
            timeout=config.get("timeout", 600.0),
        )
    if kind == "anthropic":
        from dataset_generator.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            api_key=config.get("api_key", ""),
            model=config.get("model", "claude-sonnet-4-6"),
            timeout=config.get("timeout", 600.0),
            max_tokens=config.get("max_tokens", 8192),
        )
    raise ValueError(f"Unknown provider kind: {kind}")
