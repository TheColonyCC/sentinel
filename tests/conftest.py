"""Shared pytest fixtures for the sentinel test suite.

Two pieces of plumbing every sentinel test cares about:

* Sentinel writes its memory + config + lockfile relative to the CWD.
  The ``isolated_cwd`` fixture (autouse) chdirs into a tmp dir per
  test so a stray ``save_memory()`` can't leak into the developer's
  real ``colony_analyzed.json`` and so the lockfile races between
  tests are impossible.
* The ``mock_client`` fixture hands back a ``MagicMock`` typed as
  ``ColonyClient`` so we can stub return values + assert call
  shapes without touching the network.

Sentinel doesn't do any DB or HTTP work directly — everything goes
through ``colony-sdk`` and ``requests`` (for the local Ollama call) —
so a fake client + a fake ollama HTTP layer is enough to cover the
whole module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Sentinel imports rely on ``colony_sdk`` being installed. The dev
# image / venv pins it via requirements.txt.
from colony_sdk import ColonyAPIError, ColonyClient


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Chdir into a per-test tmp directory.

    Sentinel writes ``colony_analyzed.json``, ``colony_config.json``,
    and ``sentinel.lock`` to the working directory. An autouse fixture
    keeps each test self-contained without forcing every test to opt
    in.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def mock_client() -> MagicMock:
    """A ``ColonyClient`` mock with sensible default returns.

    Spec'd against the real ``ColonyClient`` class so that a typo in
    a test (``client.voot_post`` instead of ``client.vote_post``) is
    surfaced as ``AttributeError`` immediately rather than silently
    succeeding on a MagicMock that auto-creates the attribute.
    Requires ``colony-sdk>=1.32.0`` (pinned in ``requirements-dev.txt``)
    so that the sentinel-side helpers added in 1.11.0
    (``mark_post_scanned``, ``mark_comment_scanned``) are part of the
    spec. The floor moved to 1.32.0 with the two-step registration flow
    — see ``requirements.txt``.

    Tests override individual return values by reassigning
    ``client.vote_post.return_value = ...`` etc. or set
    ``side_effect`` for error paths.
    """
    client = MagicMock(spec=ColonyClient)
    # _raw_request is the private hatch sentinel uses for endpoints
    # not yet in the SDK public surface (pii, language). Stub a
    # generic empty dict so call-sites don't need to set it per test.
    client._raw_request.return_value = {}
    client.vote_post.return_value = {"score": 1}
    client.mark_post_scanned.return_value = {"sentinel_scanned": True}
    client.mark_comment_scanned.return_value = {"sentinel_scanned": True}
    client.move_post_to_colony.return_value = {
        "post_id": "post-uuid",
        "from_colony_id": "src",
        "to_colony_id": "dst",
        "moved": True,
    }
    client.get_colonies.return_value = []
    return client


@pytest.fixture
def make_judgement():
    """Factory: build a judgement dict with the keys ``analyze_post``
    would attach. Reasonable defaults; override per-test via kwargs.

    Mirrors what ``analyze_post`` returns — score, category,
    vote_recommendation, language, post_has_pii, pii_comment_indices,
    is_test_post, plus the sentinel-internal ``_comment_ids`` and
    ``_colony_id`` carried for downstream action decisions.
    """
    def _make(**overrides: Any) -> dict:
        base: dict[str, Any] = {
            "post_id": "post-uuid",
            "title": "Test post",
            "score": 5,
            "category": "OKAY",
            "reason": "ordinary post",
            "vote_recommendation": "none",
            "language": "en",
            "post_has_pii": False,
            "pii_comment_indices": [],
            "is_test_post": False,
            "_comment_ids": [],
            "_colony_id": "colony-uuid",
        }
        base.update(overrides)
        return base

    return _make


def make_api_error(status: int = 500, message: str = "boom") -> ColonyAPIError:
    """Construct a ColonyAPIError with a ``status`` attribute.

    SDK 1.5+ requires ``(message, status, ...)`` in the constructor.
    Helper exists so individual tests don't repeat the signature
    boilerplate.
    """
    return ColonyAPIError(message, status)
