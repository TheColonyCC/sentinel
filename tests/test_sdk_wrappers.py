"""Tests for the sentinel-side SDK wrappers.

Each wrapper wraps a single SDK call and exists to centralise:
* uniform info-level logging on success
* 403 logging (so a missing-role misconfiguration is loud)
* generic warning + False return on other errors

The wrappers are thin but they're the only place sentinel decides
whether to surface vs. swallow a failure, so they're worth pinning
down. Also covers the lazy ``_is_sandbox_colony`` cache.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from colony_sdk import ColonyAPIError

import sentinel as s


def make_api_error(status: int = 500, message: str = "boom") -> ColonyAPIError:
    return ColonyAPIError(message, status)


# ────────────────────────────────────────────────────────────────────
# Language
# ────────────────────────────────────────────────────────────────────


class TestSetPostLanguage:
    def test_short_circuits_for_english(self, mock_client: MagicMock):
        """Posts already in English shouldn't hit the API at all."""
        assert s.set_post_language(mock_client, "p1", "en") is False
        mock_client._raw_request.assert_not_called()

    def test_short_circuits_for_blank_lang(self, mock_client: MagicMock):
        assert s.set_post_language(mock_client, "p1", "") is False
        mock_client._raw_request.assert_not_called()

    def test_short_circuits_for_too_short_lang(self, mock_client: MagicMock):
        """ISO 639-1 is 2 chars; anything shorter is the LLM
        hallucinating ('?', 'x') and should be dropped."""
        assert s.set_post_language(mock_client, "p1", "x") is False
        mock_client._raw_request.assert_not_called()

    def test_happy_path_puts_to_api(self, mock_client: MagicMock):
        assert s.set_post_language(mock_client, "p1", "fr") is True
        mock_client._raw_request.assert_called_once_with(
            "PUT", "/posts/p1/language?language=fr"
        )

    def test_409_already_set_treated_as_success(self, mock_client: MagicMock):
        """The server returns 409 when the language was already set
        by an earlier run — that's idempotent success, not a failure
        to retry."""
        mock_client._raw_request.side_effect = make_api_error(409)
        assert s.set_post_language(mock_client, "p1", "fr") is True

    def test_422_invalid_code_returns_false(self, mock_client: MagicMock):
        mock_client._raw_request.side_effect = make_api_error(422)
        assert s.set_post_language(mock_client, "p1", "zz") is False

    def test_other_error_returns_false(self, mock_client: MagicMock):
        mock_client._raw_request.side_effect = make_api_error(500)
        assert s.set_post_language(mock_client, "p1", "fr") is False


# ────────────────────────────────────────────────────────────────────
# Junk / PII flags
# ────────────────────────────────────────────────────────────────────


class TestMarkPostJunk:
    def test_happy_path(self, mock_client: MagicMock):
        assert s.mark_post_junk(mock_client, "p1", True) is True
        mock_client._raw_request.assert_called_once_with(
            "PUT", "/posts/p1/junk?junk=true"
        )

    def test_false_param_unmarks(self, mock_client: MagicMock):
        s.mark_post_junk(mock_client, "p1", False)
        mock_client._raw_request.assert_called_once_with(
            "PUT", "/posts/p1/junk?junk=false"
        )

    def test_403_returns_false(self, mock_client: MagicMock):
        mock_client._raw_request.side_effect = make_api_error(403)
        assert s.mark_post_junk(mock_client, "p1", True) is False


class TestFlagPostPii:
    def test_happy_path(self, mock_client: MagicMock):
        assert s.flag_post_pii(mock_client, "p1", True) is True
        mock_client._raw_request.assert_called_once_with(
            "PUT", "/posts/p1/pii?has_pii=true"
        )

    def test_403_returns_false(self, mock_client: MagicMock):
        mock_client._raw_request.side_effect = make_api_error(403)
        assert s.flag_post_pii(mock_client, "p1", True) is False


class TestFlagCommentPii:
    def test_happy_path(self, mock_client: MagicMock):
        assert s.flag_comment_pii(mock_client, "c1", True) is True
        mock_client._raw_request.assert_called_once_with(
            "PUT", "/comments/c1/pii?has_pii=true"
        )


# ────────────────────────────────────────────────────────────────────
# Move to sandbox
# ────────────────────────────────────────────────────────────────────


class TestMovePostToSandbox:
    def test_happy_path_returns_true(self, mock_client: MagicMock):
        ok = s.move_post_to_sandbox(mock_client, "p1")
        assert ok is True
        mock_client.move_post_to_colony.assert_called_once_with(
            "p1", s.TEST_POSTS_COLONY
        )

    def test_idempotent_no_move_still_returns_true(self, mock_client: MagicMock):
        """The endpoint returns ``moved: False`` when the post is
        already in the target colony — that's success."""
        mock_client.move_post_to_colony.return_value = {
            "post_id": "p1", "moved": False
        }
        assert s.move_post_to_sandbox(mock_client, "p1") is True

    def test_403_returns_false(self, mock_client: MagicMock):
        mock_client.move_post_to_colony.side_effect = make_api_error(403)
        assert s.move_post_to_sandbox(mock_client, "p1") is False


# ────────────────────────────────────────────────────────────────────
# Mark-scanned
# ────────────────────────────────────────────────────────────────────


class TestMarkScanned:
    def test_post_happy_path(self, mock_client: MagicMock):
        assert s.mark_post_scanned(mock_client, "p1") is True
        mock_client.mark_post_scanned.assert_called_once_with("p1")

    def test_post_403_returns_false(self, mock_client: MagicMock):
        mock_client.mark_post_scanned.side_effect = make_api_error(403)
        assert s.mark_post_scanned(mock_client, "p1") is False

    def test_post_500_returns_false(self, mock_client: MagicMock):
        mock_client.mark_post_scanned.side_effect = make_api_error(500)
        assert s.mark_post_scanned(mock_client, "p1") is False

    def test_comment_happy_path(self, mock_client: MagicMock):
        assert s.mark_comment_scanned(mock_client, "c1") is True
        mock_client.mark_comment_scanned.assert_called_once_with("c1")


# ────────────────────────────────────────────────────────────────────
# Sandbox-colony lookup + cache
# ────────────────────────────────────────────────────────────────────


class TestIsSandboxColony:
    def setup_method(self) -> None:
        # The cache is module-level so each test must wipe it to start
        # from a known state — otherwise the order of tests affects the
        # first-call assertions.
        s._SANDBOX_CACHE.clear()

    def test_returns_false_for_empty_id(self, mock_client: MagicMock):
        assert s._is_sandbox_colony(mock_client, "") is False
        mock_client.get_colonies.assert_not_called()

    def test_populates_cache_on_first_lookup(self, mock_client: MagicMock):
        mock_client.get_colonies.return_value = [
            {"id": "c1", "is_sandbox": True},
            {"id": "c2", "is_sandbox": False},
        ]
        assert s._is_sandbox_colony(mock_client, "c1") is True
        assert s._is_sandbox_colony(mock_client, "c2") is False
        # One round-trip serves both lookups.
        mock_client.get_colonies.assert_called_once_with(limit=200)

    def test_unknown_id_after_lookup_returns_false(self, mock_client: MagicMock):
        """A colony id not in the /colonies response shouldn't crash
        — just report not-sandbox so the caller proceeds with the
        move (worst case: a 400 from the server, which the wrapper
        handles gracefully)."""
        mock_client.get_colonies.return_value = [{"id": "c1", "is_sandbox": True}]
        assert s._is_sandbox_colony(mock_client, "c-unknown") is False

    def test_paginated_envelope_response(self, mock_client: MagicMock):
        """The SDK may return ``{"colonies": [...]}`` instead of a
        bare list depending on version — both should work."""
        mock_client.get_colonies.return_value = {
            "colonies": [{"id": "c-sb", "is_sandbox": True}]
        }
        assert s._is_sandbox_colony(mock_client, "c-sb") is True

    def test_api_failure_returns_false_without_caching(
        self, mock_client: MagicMock
    ):
        """If /colonies blows up, we don't want to false-positive a
        sandbox lookup (would suppress a legitimate move). Return
        False and don't cache the empty result — next call will retry."""
        mock_client.get_colonies.side_effect = make_api_error(503)
        assert s._is_sandbox_colony(mock_client, "c1") is False
        # No caching of the failed lookup — next call retries.
        mock_client.get_colonies.return_value = [
            {"id": "c1", "is_sandbox": True}
        ]
        mock_client.get_colonies.side_effect = None
        assert s._is_sandbox_colony(mock_client, "c1") is True
