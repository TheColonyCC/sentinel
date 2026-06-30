"""Env-var configuration: the _env_str / _env_int helpers and that the
Ollama/scan constants actually honour their environment overrides.
"""
from __future__ import annotations

import importlib

import sentinel as s


class TestEnvInt:
    def test_valid(self, monkeypatch):
        monkeypatch.setenv("SENTINEL_TEST_INT", "42")
        assert s._env_int("SENTINEL_TEST_INT", 1) == 42

    def test_invalid_falls_back_and_warns(self, monkeypatch, capsys):
        monkeypatch.setenv("SENTINEL_TEST_INT", "not-a-number")
        assert s._env_int("SENTINEL_TEST_INT", 9) == 9
        assert "ignoring invalid" in capsys.readouterr().err

    def test_missing(self, monkeypatch):
        monkeypatch.delenv("SENTINEL_TEST_INT", raising=False)
        assert s._env_int("SENTINEL_TEST_INT", 5) == 5

    def test_empty(self, monkeypatch):
        monkeypatch.setenv("SENTINEL_TEST_INT", "")
        assert s._env_int("SENTINEL_TEST_INT", 5) == 5


class TestEnvStr:
    def test_override(self, monkeypatch):
        monkeypatch.setenv("SENTINEL_TEST_STR", "hello")
        assert s._env_str("SENTINEL_TEST_STR", "def") == "hello"

    def test_empty_falls_back(self, monkeypatch):
        monkeypatch.setenv("SENTINEL_TEST_STR", "")
        assert s._env_str("SENTINEL_TEST_STR", "def") == "def"

    def test_missing_falls_back(self, monkeypatch):
        monkeypatch.delenv("SENTINEL_TEST_STR", raising=False)
        assert s._env_str("SENTINEL_TEST_STR", "def") == "def"


def test_env_overrides_module_constants(monkeypatch):
    """The module-level constants pick up env at import time. Reload under a
    patched environment, assert, then reload back to restore for other tests.
    """
    import sentinel
    monkeypatch.setenv("OLLAMA_HOST", "http://example:1234")
    monkeypatch.setenv("SENTINEL_MODEL", "custom-model:1b")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "42")
    monkeypatch.setenv("SCAN_SAVE_EVERY", "3")
    try:
        importlib.reload(sentinel)
        assert sentinel.OLLAMA_HOST == "http://example:1234"
        assert sentinel.DEFAULT_MODEL == "custom-model:1b"
        assert sentinel.OLLAMA_TIMEOUT == 42
        assert sentinel.SCAN_SAVE_EVERY == 3
    finally:
        for k in ("OLLAMA_HOST", "SENTINEL_MODEL", "OLLAMA_TIMEOUT", "SCAN_SAVE_EVERY"):
            monkeypatch.delenv(k, raising=False)
        importlib.reload(sentinel)  # restore module defaults for the rest of the suite

    # Defaults are back after the restoring reload.
    assert sentinel.OLLAMA_TIMEOUT == 180
    assert sentinel.DEFAULT_MODEL == "qwen3.5:9b-q4_k_m"
