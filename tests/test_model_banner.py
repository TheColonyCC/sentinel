"""Both modes announce the model in use at the start of the run.

The ordering is the substance of these tests, not the string. ``cmd_scan`` and
``cmd_webhook`` both call ``ensure_model_available``, which calls ``sys.exit(1)``
when the model isn't pulled — so a banner emitted after the preflight would be
missing from exactly the logs where "which model was it?" is the question being
asked. The banner therefore has to come first, and "first" is asserted against
the preflight rather than assumed from reading the source.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import sentinel as s


def _scan_args(**over):
    base = dict(
        dry_run=True, no_vote=True, no_pii=True, model="test-model:9b",
        username=None, force=False, post_id=None, sort="new", limit=1,
        days=7, confirm=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _webhook_args(**over):
    base = dict(
        dry_run=True, no_vote=True, no_pii=True, model="test-model:9b",
        username=None, port=8000, path="/webhook", secret="x" * 16,
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestLogModelInUse:
    def test_names_the_model_and_the_host(self, caplog):
        with caplog.at_level(logging.INFO, logger="sentinel"):
            s.log_model_in_use("qwen3.5:9b-q4_k_m")
        line = caplog.text
        assert "qwen3.5:9b-q4_k_m" in line
        # The host disambiguates which daemon served the run.
        assert s.OLLAMA_HOST in line

    def test_reports_the_model_actually_passed_not_the_default(self, caplog):
        """Control: a banner hard-coded to DEFAULT_MODEL would pass the test
        above whenever the caller happened to use the default."""
        override = "some-other-model:70b"
        assert override != s.DEFAULT_MODEL
        with caplog.at_level(logging.INFO, logger="sentinel"):
            s.log_model_in_use(override)
        assert override in caplog.text
        assert s.DEFAULT_MODEL not in caplog.text


class TestBannerPrecedesPreflight:
    """The banner must survive a run that dies in the model preflight."""

    def test_scan_logs_model_before_preflight_exits(self, monkeypatch, caplog):
        def exit_preflight(model):
            raise SystemExit(1)

        monkeypatch.setattr(s, "ensure_model_available", exit_preflight)
        with caplog.at_level(logging.INFO, logger="sentinel"):
            with pytest.raises(SystemExit):
                s.cmd_scan(_scan_args())
        assert "test-model:9b" in caplog.text

    def test_webhook_logs_model_before_preflight_exits(self, monkeypatch, caplog):
        def exit_preflight(model):
            raise SystemExit(1)

        monkeypatch.setattr(s, "ensure_model_available", exit_preflight)
        with caplog.at_level(logging.INFO, logger="sentinel"):
            with pytest.raises(SystemExit):
                s.cmd_webhook(_webhook_args())
        assert "test-model:9b" in caplog.text

    def test_scan_banner_is_emitted_on_a_normal_run(self, monkeypatch, caplog):
        """Not just on the failure path — the ordinary run says it too."""
        monkeypatch.setattr(s, "ensure_model_available", lambda m: None)
        client = MagicMock()
        client.iter_posts.return_value = iter([])
        monkeypatch.setattr(s, "get_or_register_client", lambda u: (client, {}))
        monkeypatch.setattr(s, "load_memory", lambda: {})
        monkeypatch.setattr(s, "prune_memory", lambda m, **k: m)
        with caplog.at_level(logging.INFO, logger="sentinel"):
            s.cmd_scan(_scan_args())
        assert "test-model:9b" in caplog.text
