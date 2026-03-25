"""YAML configuration with environment variable substitution."""

import os
import re
from pathlib import Path
from typing import Any

import yaml

ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")

DEFAULT_CONFIG: dict[str, Any] = {
    "provider": {
        "kind": "openai",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "llama3.1:8b",
    },
    "generation": {
        "num_samples": 100,
        "batch_size": 10,
        "max_workers": 10,
        "max_retries": 3,
        "temperature": 0.7,
        "strategy": "direct",
        "strategy_config": {},
    },
    "output": {
        "path": "data/output.jsonl",
        "format": "jsonl",
    },
}


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR:-default} patterns with environment variable values."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(var_name, default if default is not None else "")

    return ENV_VAR_PATTERN.sub(_replace, value)


def _walk_and_substitute(obj: Any) -> Any:
    """Recursively substitute env vars in all string values."""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_substitute(item) for item in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, preferring override values."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config from YAML file, merge with defaults, substitute env vars.

    Args:
        path: Path to YAML config file. If None, looks for config.yaml in cwd.

    Returns:
        Merged configuration dictionary.
    """
    config = DEFAULT_CONFIG.copy()

    path = Path.cwd() / "config.yaml" if path is None else Path(path)

    if path.exists():
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)

    return _walk_and_substitute(config)
