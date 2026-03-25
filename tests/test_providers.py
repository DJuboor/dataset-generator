"""Tests for provider creation and OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dataset_generator.providers import create_provider
from dataset_generator.providers.base import CompletionResult
from dataset_generator.providers.openai_compat import OpenAIProvider


class TestCreateProvider:
    def test_openai_kind(self):
        provider = create_provider(
            {
                "kind": "openai",
                "base_url": "http://localhost:11434/v1",
                "api_key": "test",
                "model": "gpt-4o",
            }
        )
        assert isinstance(provider, OpenAIProvider)

    def test_ollama_kind(self):
        provider = create_provider(
            {
                "kind": "ollama",
                "base_url": "http://localhost:11434/v1",
                "model": "llama3.1:8b",
            }
        )
        assert isinstance(provider, OpenAIProvider)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            create_provider({"kind": "unknown", "model": "x"})


class TestCompletionResult:
    def test_total_tokens(self):
        r = CompletionResult(content="hi", input_tokens=10, output_tokens=20)
        assert r.total_tokens == 30

    def test_total_tokens_none_when_missing(self):
        r = CompletionResult(content="hi")
        assert r.total_tokens is None

    def test_total_tokens_none_when_partial(self):
        r = CompletionResult(content="hi", input_tokens=10)
        assert r.total_tokens is None


def _mock_response(content: str = "hello", prompt_tokens: int = 5, completion_tokens: int = 10):
    """Build a mock OpenAI API response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class TestOpenAIProviderComplete:
    def test_complete_returns_result(self):
        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        provider.client = MagicMock()
        provider.client.chat.completions.create.return_value = _mock_response("generated text")

        result = provider.complete([{"role": "user", "content": "hi"}])

        assert isinstance(result, CompletionResult)
        assert result.content == "generated text"
        assert result.input_tokens == 5
        assert result.output_tokens == 10
        assert result.model == "gpt-4o"

    def test_complete_with_max_tokens(self):
        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        provider.client = MagicMock()
        provider.client.chat.completions.create.return_value = _mock_response()

        provider.complete([{"role": "user", "content": "hi"}], max_tokens=100)

        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 100

    def test_complete_without_max_tokens(self):
        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        provider.client = MagicMock()
        provider.client.chat.completions.create.return_value = _mock_response()

        provider.complete([{"role": "user", "content": "hi"}])

        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert "max_tokens" not in call_kwargs

    def test_complete_none_content_returns_empty_string(self):
        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        provider.client = MagicMock()
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=None,
        )
        provider.client.chat.completions.create.return_value = resp

        result = provider.complete([{"role": "user", "content": "hi"}])
        assert result.content == ""
        assert result.input_tokens is None


class TestOpenAIProviderCompleteJSON:
    def test_complete_json(self):
        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        provider.client = MagicMock()
        provider.client.chat.completions.create.return_value = _mock_response('{"key": "value"}')

        result = provider.complete_json([{"role": "user", "content": "hi"}])

        assert result == {"key": "value"}
        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_complete_json_none_content_returns_empty(self):
        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        provider.client = MagicMock()
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=None,
        )
        provider.client.chat.completions.create.return_value = resp

        result = provider.complete_json([{"role": "user", "content": "hi"}])
        assert result == {}

    def test_complete_json_with_max_tokens(self):
        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        provider.client = MagicMock()
        provider.client.chat.completions.create.return_value = _mock_response('{"a": 1}')

        provider.complete_json([{"role": "user", "content": "hi"}], max_tokens=50)

        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 50


class TestExtractUsage:
    def test_with_usage(self):
        provider = OpenAIProvider(model="m", api_key="k")
        resp = _mock_response()
        usage = provider._extract_usage(resp)
        assert usage == {"input_tokens": 5, "output_tokens": 10}

    def test_without_usage(self):
        provider = OpenAIProvider(model="m", api_key="k")
        resp = SimpleNamespace(choices=[])
        usage = provider._extract_usage(resp)
        assert usage == {"input_tokens": None, "output_tokens": None}
