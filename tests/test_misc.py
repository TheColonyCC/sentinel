"""Tests for the small isolated helpers — ``is_within_days``,
``build_analysis_text``, and the ``_pending_actions`` translator's
behaviour on malformed judgement fields.

These don't talk to the network and aren't on a hot path, but they
encode assumptions that the LLM prompt and the scan loop rely on.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import sentinel as s


class TestIsWithinDays:
    def test_recent_post_returns_true(self):
        ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        assert s.is_within_days(ts, days=7) is True

    def test_old_post_returns_false(self):
        ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        assert s.is_within_days(ts, days=7) is False

    def test_empty_string_returns_false(self):
        assert s.is_within_days("", days=7) is False

    def test_malformed_timestamp_returns_false(self):
        """An unparseable timestamp must not crash the scan loop —
        it should silently skip the post."""
        assert s.is_within_days("not-a-date", days=7) is False

    def test_z_suffix_is_handled(self):
        """The Colony's API returns ISO timestamps with a ``Z``
        suffix; ``fromisoformat`` only accepts ``+00:00``. The helper
        does the conversion — verify it does."""
        ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        ts_zulu = ts.replace("+00:00", "Z")
        assert s.is_within_days(ts_zulu, days=1) is True


class TestBuildAnalysisText:
    def _post(self, **overrides) -> dict:
        base = {
            "post": {
                "id": "p1",
                "title": "Hello world",
                "body": "Some body text",
                "author": {"username": "alice"},
                "created_at": "2026-05-18T12:00:00Z",
            },
            "comments": [],
        }
        base["post"].update(overrides.get("post", {}))
        if "comments" in overrides:
            base["comments"] = overrides["comments"]
        return base

    def test_includes_title_body_author(self):
        text = s.build_analysis_text(self._post())
        assert "alice" in text
        assert "Hello world" in text
        assert "Some body text" in text

    def test_no_replies_line_when_zero_comments(self):
        text = s.build_analysis_text(self._post(comments=[]))
        assert "No replies yet" in text
        assert "TOP REPLIES" not in text

    def test_top_replies_section_renders(self):
        text = s.build_analysis_text(self._post(comments=[
            {"author": {"username": "bob"}, "body": "first reply"},
            {"author": {"username": "carol"}, "body": "second reply"},
        ]))
        assert "TOP REPLIES" in text
        assert "1. bob: first reply" in text
        assert "2. carol: second reply" in text

    def test_long_comment_body_truncated_to_400_chars(self):
        """The prompt's per-comment cap is 400 chars — going over
        bloats num_ctx and degrades latency without adding signal."""
        long_body = "x" * 1000
        text = s.build_analysis_text(self._post(comments=[
            {"author": {"username": "spammer"}, "body": long_body}
        ]))
        # The truncated rendering should contain at most 400 'x' in a row.
        assert "x" * 400 in text
        assert "x" * 401 not in text

    def test_anonymous_author_handled(self):
        """The author dict may be absent for some post types — must
        not crash, should render as 'anonymous'."""
        text = s.build_analysis_text(self._post(post={"author": None}))
        assert "anonymous" in text

    def test_falls_back_to_content_field_if_no_body(self):
        """``body`` is the canonical field but some agent clients
        post with ``content`` — sentinel reads either so a third-party
        client doesn't get silently empty-analyzed."""
        text = s.build_analysis_text({
            "post": {
                "title": "T",
                "body": "",
                "content": "from content field",
                "author": {"username": "x"},
            },
            "comments": [],
        })
        assert "from content field" in text


class TestPendingActionsMalformedInputs:
    """Robustness tests for ``_pending_actions``: an LLM judgement is
    untrusted input, and a sentinel rolled back to an older version
    might encounter judgement fields with unexpected shapes from a
    saved memory entry written by a newer version."""

    def test_missing_vote_recommendation_treated_as_none(self):
        j = {"_comment_ids": [], "_colony_id": "c1"}
        kinds = [a["kind"] for a in s._pending_actions(j)]
        assert "vote" not in kinds  # default no-vote

    def test_score_with_string_value_treated_as_zero(self):
        """A non-numeric score from a misbehaving model should not
        crash — the upvote gate just sees 0 < UPVOTE_MIN_SCORE and
        drops the vote."""
        j = {
            "vote_recommendation": "upvote",
            "score": "high",
            "_comment_ids": [],
            "_colony_id": "c1",
        }
        kinds = [a["kind"] for a in s._pending_actions(j)]
        assert "vote" not in kinds

    def test_no_colony_id_still_emits_move_to_sandbox_action(self):
        """Move-to-sandbox is emitted purely on ``is_test_post`` —
        missing ``_colony_id`` just means ``source_colony_id=None``
        in the action, and _apply_action handles that path (falls
        through to the move, which may then 400 if already in
        target)."""
        j = {
            "is_test_post": True,
            "_comment_ids": [],
        }
        actions = s._pending_actions(j)
        move = next(a for a in actions if a["kind"] == "move_to_sandbox")
        assert move["source_colony_id"] is None
