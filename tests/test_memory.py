"""Tests for the on-disk memory + config persistence helpers.

Sentinel's only durable state lives in two JSON files written to
CWD: ``colony_analyzed.json`` (the post→judgement skip-cache) and
``colony_config.json`` (api_key + username). Both go through
``_atomic_json_write`` which writes to a tempfile and ``os.replace``s
so a crash mid-write can't leave a half-written file on disk.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import sentinel as s


class TestAtomicJsonWrite:
    def test_round_trips_simple_payload(self, isolated_cwd: Path):
        target = isolated_cwd / "thing.json"
        s._atomic_json_write(target, {"hello": "world", "n": 3})
        assert json.loads(target.read_text()) == {"hello": "world", "n": 3}

    def test_overwrites_existing_file(self, isolated_cwd: Path):
        target = isolated_cwd / "thing.json"
        target.write_text(json.dumps({"old": True}))
        s._atomic_json_write(target, {"new": True})
        assert json.loads(target.read_text()) == {"new": True}

    def test_preserves_unicode(self, isolated_cwd: Path):
        """``ensure_ascii=False`` in the writer means non-ASCII titles
        stay readable in the file rather than landing as ``\\uXXXX``."""
        target = isolated_cwd / "thing.json"
        s._atomic_json_write(target, {"title": "héllo wörld 🌟"})
        # Read raw bytes to confirm the UTF-8 character is intact.
        assert "héllo wörld 🌟" in target.read_text(encoding="utf-8")


class TestLoadMemory:
    def test_returns_empty_dict_when_file_missing(self):
        assert s.load_memory() == {}

    def test_returns_empty_dict_on_corrupt_json(self):
        """A garbage memory file shouldn't crash sentinel on startup
        — it should reset to empty and re-analyze posts."""
        s.MEMORY_FILE.write_text("{ not valid json")
        assert s.load_memory() == {}

    def test_returns_parsed_contents_on_good_file(self):
        s.MEMORY_FILE.write_text(json.dumps({"p1": {"score": 8}}))
        assert s.load_memory() == {"p1": {"score": 8}}


class TestSaveMemory:
    def test_creates_file_with_payload(self):
        s.save_memory({"p1": {"score": 9}})
        assert json.loads(s.MEMORY_FILE.read_text()) == {"p1": {"score": 9}}


class TestPruneMemory:
    def test_drops_failed_analyses_with_score_zero(self):
        """``score=0`` is the marker for a failed Ollama call kept
        only so the post can be retried. They have no analytic value
        and should be pruned on every run."""
        m = {"p1": {"score": 0}, "p2": {"score": 7}}
        pruned = s.prune_memory(m)
        assert "p1" not in pruned
        assert "p2" in pruned

    def test_drops_entries_older_than_max_age(self):
        recent = datetime.now() - timedelta(days=1)
        ancient = datetime.now() - timedelta(days=s.MEMORY_MAX_AGE_DAYS + 1)
        m = {
            "p_recent": {"score": 5, "analyzed_at": recent.isoformat()},
            "p_ancient": {"score": 5, "analyzed_at": ancient.isoformat()},
        }
        pruned = s.prune_memory(m)
        assert "p_recent" in pruned
        assert "p_ancient" not in pruned

    def test_keeps_entries_with_unparseable_timestamp(self):
        """An entry written by an older sentinel version that didn't
        record ``analyzed_at`` (or wrote it in a stale format) should
        survive pruning — better to keep a stale-cache entry than to
        re-burn LLM cycles on a post we likely already analyzed."""
        m = {"p1": {"score": 5, "analyzed_at": "not-a-date"}}
        pruned = s.prune_memory(m)
        assert "p1" in pruned

    def test_keeps_entries_missing_analyzed_at(self):
        m = {"p1": {"score": 5}}  # no analyzed_at at all
        pruned = s.prune_memory(m)
        assert "p1" in pruned


class TestGetProcessedIds:
    def test_returns_ids_for_scored_entries(self):
        m = {
            "p1": {"post_id": "p1", "score": 7},
            "p2": {"post_id": "p2", "score": 0},   # failed analysis
            "p3": {"post_id": "p3", "score": 9},
            "p4": {"score": 8},                     # missing post_id
        }
        assert s.get_processed_ids(m) == {"p1", "p3"}


class TestConfigPersistence:
    def test_load_config_empty_when_missing(self):
        assert s.load_config() == {}

    def test_save_then_load_round_trips(self):
        s.save_config({"api_key": "col_xyz", "username": "agent-1"})
        assert s.load_config() == {"api_key": "col_xyz", "username": "agent-1"}
