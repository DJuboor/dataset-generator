"""Tests for config loading and env var substitution."""

import os

from dataset_generator.config import (
    DEFAULT_CONFIG,
    _deep_merge,
    _substitute_env_vars,
    _walk_and_substitute,
    load_config,
)


class TestEnvVarSubstitution:
    def test_simple_var(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "hello")
        assert _substitute_env_vars("${TEST_VAR}") == "hello"

    def test_default_value(self):
        # Ensure var doesn't exist
        os.environ.pop("NONEXISTENT_VAR", None)
        assert _substitute_env_vars("${NONEXISTENT_VAR:-fallback}") == "fallback"

    def test_missing_no_default(self):
        os.environ.pop("NONEXISTENT_VAR", None)
        assert _substitute_env_vars("${NONEXISTENT_VAR}") == ""

    def test_embedded_in_string(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        assert _substitute_env_vars("http://${HOST}:8080") == "http://localhost:8080"

    def test_no_vars(self):
        assert _substitute_env_vars("plain string") == "plain string"


class TestWalkAndSubstitute:
    def test_nested_dict(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret")
        result = _walk_and_substitute({"a": {"b": "${MY_KEY}"}})
        assert result == {"a": {"b": "secret"}}

    def test_list(self, monkeypatch):
        monkeypatch.setenv("X", "val")
        result = _walk_and_substitute(["${X}", "plain"])
        assert result == ["val", "plain"]

    def test_non_string_passthrough(self):
        assert _walk_and_substitute(42) == 42
        assert _walk_and_substitute(True) is True


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        assert _deep_merge(base, override) == {"a": 1, "b": 3}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        assert _deep_merge(base, override) == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_new_keys(self):
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}


class TestLoadConfig:
    def test_defaults_when_no_file(self, tmp_path):
        # Given a nonexistent config path
        config = load_config(tmp_path / "nonexistent.yaml")
        # Then defaults are returned
        assert config["generation"]["num_samples"] == DEFAULT_CONFIG["generation"]["num_samples"]

    def test_loads_yaml(self, tmp_path):
        # Given a config file with overrides
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("generation:\n  num_samples: 500\n")

        # When loaded
        config = load_config(cfg_file)

        # Then override is applied, defaults preserved
        assert config["generation"]["num_samples"] == 500
        assert config["generation"]["max_workers"] == DEFAULT_CONFIG["generation"]["max_workers"]
