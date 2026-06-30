"""Tests for scan-mode resilience: model preflight, the shared per-post
pipeline, and crash-safe incremental memory checkpointing.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

import sentinel as s


def _args(**over):
    base = dict(
        dry_run=False, no_vote=False, no_pii=False, model="qwen3.5:9b-q4_K_M",
        username=None, force=False, post_id=None, sort="new", limit=50,
        days=7, confirm=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _tags_response(names):
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = MagicMock()
    r.json.return_value = {"models": [{"name": n} for n in names]}
    return r


# ─── #2 ensure_model_available ───────────────────────────────────────────

class TestEnsureModelAvailable:
    def test_exact_match_ok(self, monkeypatch):
        monkeypatch.setattr(s.requests, "get",
                            lambda *a, **k: _tags_response(["qwen3.5:9b-q4_K_M"]))
        s.ensure_model_available("qwen3.5:9b-q4_K_M")  # no raise

    def test_base_name_match_ok(self, monkeypatch):
        monkeypatch.setattr(s.requests, "get",
                            lambda *a, **k: _tags_response(["llama3:latest"]))
        s.ensure_model_available("llama3")  # base name matches

    def test_implicit_latest_ok(self, monkeypatch):
        monkeypatch.setattr(s.requests, "get",
                            lambda *a, **k: _tags_response(["llama3:latest"]))
        s.ensure_model_available("llama3")

    def test_missing_model_exits(self, monkeypatch):
        monkeypatch.setattr(s.requests, "get",
                            lambda *a, **k: _tags_response(["other:7b"]))
        with pytest.raises(SystemExit) as e:
            s.ensure_model_available("qwen3.5:9b-q4_K_M")
        assert e.value.code == 1

    def test_daemon_error_exits(self, monkeypatch):
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError("down")
        monkeypatch.setattr(s.requests, "get", boom)
        with pytest.raises(SystemExit) as e:
            s.ensure_model_available("m")
        assert e.value.code == 1


# ─── #3 _process_post ────────────────────────────────────────────────────

class TestProcessPost:
    def _post(self):
        return {"post": {"id": "p1", "title": "t", "colony_id": "c"}, "comments": []}

    def test_success_records_and_acts(self, monkeypatch, mock_client):
        monkeypatch.setattr(s, "analyze_post",
                            lambda data, model: {"category": "OKAY", "score": 5})
        act = MagicMock(return_value=[])
        monkeypatch.setattr(s, "act_on_judgement", act)
        memory, results = {}, []
        ok = s._process_post(mock_client, self._post(), _args(), memory, results)
        assert ok is True
        assert memory["p1"]["category"] == "OKAY"
        assert "analyzed_at" in memory["p1"]
        assert len(results) == 1
        act.assert_called_once()

    def test_dry_run_skips_actions(self, monkeypatch, mock_client):
        monkeypatch.setattr(s, "analyze_post", lambda d, m: {"category": "OKAY"})
        act = MagicMock(return_value=[])
        monkeypatch.setattr(s, "act_on_judgement", act)
        memory, results = {}, []
        ok = s._process_post(mock_client, self._post(), _args(dry_run=True), memory, results)
        assert ok is True
        assert "p1" in memory
        act.assert_not_called()

    def test_model_failure_records_nothing(self, monkeypatch, mock_client):
        monkeypatch.setattr(s, "analyze_post", lambda d, m: None)
        act = MagicMock()
        monkeypatch.setattr(s, "act_on_judgement", act)
        memory, results = {}, []
        ok = s._process_post(mock_client, self._post(), _args(), memory, results)
        assert ok is False
        assert memory == {} and results == []
        act.assert_not_called()


# ─── #1 incremental + crash-safe checkpointing in cmd_scan ───────────────

def _wire_scan(monkeypatch, mock_client, posts, *, analyze):
    """Stub cmd_scan's collaborators; return the list that records each
    save_memory() call's snapshot of memory keys."""
    monkeypatch.setattr(s, "ensure_model_available", lambda model: None)
    monkeypatch.setattr(s, "get_or_register_client",
                        lambda u: (mock_client, {"username": "sentinel-bot"}))
    monkeypatch.setattr(s, "load_memory", lambda: {})
    monkeypatch.setattr(s, "prune_memory", lambda m, **k: m)
    monkeypatch.setattr(s, "retry_pending_actions", lambda c, m: 0)
    monkeypatch.setattr(s, "is_within_days", lambda *a, **k: True)
    monkeypatch.setattr(s, "fetch_post_with_comments",
                        lambda c, pid: {"post": {"id": pid, "title": "t"}, "comments": []})
    monkeypatch.setattr(s, "act_on_judgement", lambda *a, **k: [])
    monkeypatch.setattr(s, "log_results", lambda r: None)
    monkeypatch.setattr(s, "analyze_post", analyze)
    mock_client.iter_posts = MagicMock(return_value=posts)
    saves = []
    monkeypatch.setattr(s, "save_memory", lambda m: saves.append(set(m.keys())))
    return saves


def test_scan_checkpoints_periodically(monkeypatch, mock_client):
    posts = [{"id": f"p{i}", "title": "t", "created_at": "2026-06-30",
              "author": {"username": "someone"}} for i in range(25)]
    saves = _wire_scan(monkeypatch, mock_client, posts,
                       analyze=lambda d, m: {"category": "OKAY"})
    s.cmd_scan(_args())
    # 25 posts, SCAN_SAVE_EVERY=10 → checkpoints at 10, 20, + final finally = 3.
    assert len(saves) >= 3
    assert len(saves[-1]) == 25  # final save has everything


def test_scan_persists_on_crash(monkeypatch, mock_client):
    posts = [{"id": f"p{i}", "title": "t", "created_at": "2026-06-30",
              "author": {"username": "someone"}} for i in range(5)]
    calls = {"n": 0}

    def exploding_analyze(data, model):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated crash mid-scan")
        return {"category": "OKAY"}

    saves = _wire_scan(monkeypatch, mock_client, posts, analyze=exploding_analyze)
    with pytest.raises(RuntimeError):
        s.cmd_scan(_args())
    # The finally block must have saved the two posts finished before the crash.
    assert saves, "memory was never checkpointed on crash"
    assert saves[-1] == {"p0", "p1"}


def test_dry_run_never_saves(monkeypatch, mock_client):
    posts = [{"id": "p0", "title": "t", "created_at": "2026-06-30",
              "author": {"username": "someone"}}]
    saves = _wire_scan(monkeypatch, mock_client, posts,
                       analyze=lambda d, m: {"category": "OKAY"})
    s.cmd_scan(_args(dry_run=True))
    assert saves == []


def test_scan_save_every_constant_sane():
    assert isinstance(s.SCAN_SAVE_EVERY, int)
    assert 1 <= s.SCAN_SAVE_EVERY <= 100
