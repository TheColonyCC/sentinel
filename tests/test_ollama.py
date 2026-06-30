"""Robustness pins for the Ollama call (the per-post scan engine).

A wedged model used to hang a single scan for ten minutes — ``OLLAMA_TIMEOUT``
was 600s and generation was unbounded — before timing out and giving up.
These tests pin the guards that bound a scan:

* a tight wall-clock read timeout (≈180s, was 600),
* a separate short connect timeout so a dead daemon fails fast,
* a ``num_predict`` cap so a runaway decode can't burn the whole budget,
* and that every failure mode returns ``None`` (post retries next run)
  rather than raising and killing the scan loop.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

import sentinel as s


# ─── Config pins ─────────────────────────────────────────────────────────

def test_timeout_is_bounded_to_180s():
    # The headline fix: a single scan can no longer hang for 10 minutes.
    assert s.OLLAMA_TIMEOUT == 180
    assert s.OLLAMA_TIMEOUT <= 180


def test_connect_timeout_is_short():
    assert 0 < s.OLLAMA_CONNECT_TIMEOUT <= 10


def test_generation_is_capped():
    # num_predict bounds tokens => bounds time; without it a runaway decode
    # generates until the wall-clock timeout.
    assert "num_predict" in s.OLLAMA_OPTIONS
    assert 0 < s.OLLAMA_OPTIONS["num_predict"] <= 4096


def test_slow_warn_below_timeout():
    # The early-warning threshold must fire before the hard timeout.
    assert s.OLLAMA_SLOW_WARN_SECONDS < s.OLLAMA_TIMEOUT


# ─── call_ollama behaviour ───────────────────────────────────────────────

def _fake_response(content: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"message": {"content": content}}
    return resp


def test_passes_split_connect_read_timeout(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["timeout"] = timeout
        return _fake_response('{"category": "OK"}')

    monkeypatch.setattr(s.requests, "post", fake_post)
    s.call_ollama("m", [{"role": "user", "content": "hi"}])

    assert captured["timeout"] == (s.OLLAMA_CONNECT_TIMEOUT, s.OLLAMA_TIMEOUT)


def test_sends_num_predict_in_options(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return _fake_response('{"category": "OK"}')

    monkeypatch.setattr(s.requests, "post", fake_post)
    s.call_ollama("m", [{"role": "user", "content": "hi"}])

    assert captured["json"]["options"]["num_predict"] == s.OLLAMA_OPTIONS["num_predict"]


def test_happy_path_returns_parsed_json(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        s.requests, "post",
        lambda *a, **k: _fake_response('{"category": "JUNK", "score": 0}'),
    )
    out = s.call_ollama("m", [{"role": "user", "content": "hi"}])
    assert out == {"category": "JUNK", "score": 0}


def test_timeout_returns_none_not_raise(monkeypatch: pytest.MonkeyPatch):
    def fake_post(*a, **k):
        raise requests.exceptions.Timeout("read timed out")

    monkeypatch.setattr(s.requests, "post", fake_post)
    assert s.call_ollama("m", [{"role": "user", "content": "hi"}]) is None


def test_connection_error_returns_none_not_raise(monkeypatch: pytest.MonkeyPatch):
    def fake_post(*a, **k):
        raise requests.exceptions.ConnectionError("daemon down")

    monkeypatch.setattr(s.requests, "post", fake_post)
    assert s.call_ollama("m", [{"role": "user", "content": "hi"}]) is None


def test_500_returns_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        s.requests, "post", lambda *a, **k: _fake_response("", status=500),
    )
    assert s.call_ollama("m", [{"role": "user", "content": "hi"}]) is None


def test_malformed_json_returns_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        s.requests, "post", lambda *a, **k: _fake_response("not json at all"),
    )
    assert s.call_ollama("m", [{"role": "user", "content": "hi"}]) is None


def test_slow_call_warns_but_still_returns(monkeypatch: pytest.MonkeyPatch, caplog):
    # Simulate a call that crosses the slow-warn threshold: monotonic jumps
    # past OLLAMA_SLOW_WARN_SECONDS between start and finish.
    ticks = iter([100.0, 100.0 + s.OLLAMA_SLOW_WARN_SECONDS + 5])
    monkeypatch.setattr(s.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        s.requests, "post", lambda *a, **k: _fake_response('{"category": "OK"}'),
    )
    import logging
    with caplog.at_level(logging.WARNING):
        out = s.call_ollama("m", [{"role": "user", "content": "hi"}])
    assert out == {"category": "OK"}
    assert any("took" in r.getMessage() for r in caplog.records)
