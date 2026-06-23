"""Tests for the action pipeline — ``_pending_actions``,
``_apply_action``, ``act_on_judgement``.

The most important regression surface in sentinel: every new judgement
field landing in the system prompt is a chance to (a) silently miss
emitting an action, (b) silently mis-gate an action, or (c) break
replay of an old memory entry written by a previous sentinel version.

These tests pin down the contract of each transition.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from colony_sdk import ColonyAPIError

import sentinel as s


def make_api_error(status: int = 500, message: str = "boom") -> ColonyAPIError:
    """Inlined helper — ``ColonyAPIError(message, status, ...)`` shape
    since SDK 1.5+. Duplicated across test modules rather than
    cross-imported from conftest so test files stay independent."""
    return ColonyAPIError(message, status)


# ────────────────────────────────────────────────────────────────────
# _pending_actions: judgement -> [action dict]
# ────────────────────────────────────────────────────────────────────


class TestPendingActions:
    def test_okay_post_emits_only_mark_scanned(self, make_judgement):
        """Plain OKAY post with no vote / no language flip / no PII:
        the only action should be the post mark-scanned (every scanned
        post is marked, regardless of category)."""
        j = make_judgement()
        actions = s._pending_actions(j)
        assert actions == [{"kind": "mark_scanned_post"}]

    def test_strong_upvote_emits_vote_plus_mark(self, make_judgement):
        """Score ≥ UPVOTE_MIN_SCORE + upvote recommendation -> +1 vote."""
        j = make_judgement(vote_recommendation="upvote", score=s.UPVOTE_MIN_SCORE)
        actions = s._pending_actions(j)
        kinds = [a["kind"] for a in actions]
        assert "vote" in kinds
        vote = next(a for a in actions if a["kind"] == "vote")
        assert vote["value"] == 1

    def test_weak_upvote_below_floor_is_dropped(self, make_judgement):
        """Score below UPVOTE_MIN_SCORE silently downgrades upvote to no-vote.

        The floor is the whole reason upvotes are scarce — without
        this branch the LLM's looseness inflates them. Tests for the
        floor live here so a future bump to UPVOTE_MIN_SCORE (or a
        rename) immediately surfaces.
        """
        j = make_judgement(vote_recommendation="upvote", score=s.UPVOTE_MIN_SCORE - 1)
        kinds = [a["kind"] for a in s._pending_actions(j)]
        assert "vote" not in kinds

    def test_downvote_is_not_gated(self, make_judgement):
        """Downvotes have no score floor — spam should be suppressed
        promptly even on a 'low confidence' BAD judgement."""
        j = make_judgement(vote_recommendation="downvote", score=2)
        vote = next(a for a in s._pending_actions(j) if a["kind"] == "vote")
        assert vote["value"] == -1

    def test_junk_category_emits_junk_action(self, make_judgement):
        j = make_judgement(category="JUNK", vote_recommendation="downvote", score=1)
        kinds = [a["kind"] for a in s._pending_actions(j)]
        assert "junk" in kinds

    def test_non_english_language_emits_language_action(self, make_judgement):
        j = make_judgement(language="fr")
        lang = next(a for a in s._pending_actions(j) if a["kind"] == "language")
        assert lang["code"] == "fr"

    def test_english_language_does_not_emit_action(self, make_judgement):
        """Most posts are English; the default ``en`` should not
        produce a no-op API call."""
        j = make_judgement(language="en")
        kinds = [a["kind"] for a in s._pending_actions(j)]
        assert "language" not in kinds

    def test_post_pii_emits_post_pii_action(self, make_judgement):
        j = make_judgement(post_has_pii=True)
        kinds = [a["kind"] for a in s._pending_actions(j)]
        assert "post_pii" in kinds

    def test_comment_pii_indices_map_to_comment_ids(self, make_judgement):
        """1-based ``pii_comment_indices`` from the LLM should resolve
        to the right comment_id from ``_comment_ids``."""
        j = make_judgement(
            pii_comment_indices=[1, 3],
            _comment_ids=["c1-uuid", "c2-uuid", "c3-uuid"],
        )
        flagged = [a for a in s._pending_actions(j) if a["kind"] == "comment_pii"]
        assert {a["comment_id"] for a in flagged} == {"c1-uuid", "c3-uuid"}

    def test_comment_pii_index_out_of_range_silently_ignored(self, make_judgement):
        """Robustness: an LLM-supplied index pointing past the
        ``_comment_ids`` length should be dropped, not raise. Same for
        non-integer entries."""
        j = make_judgement(
            pii_comment_indices=[7, "garbage", 0],  # 0 -> -1 index, also dropped
            _comment_ids=["c1"],
        )
        flagged = [a for a in s._pending_actions(j) if a["kind"] == "comment_pii"]
        assert flagged == []

    def test_is_test_post_emits_move_to_sandbox_with_source(self, make_judgement):
        """When the judgement flags a test post, the action carries
        the source colony_id so ``_apply_action`` can short-circuit
        without re-fetching the post (the source-sandbox check happens
        at apply time)."""
        j = make_judgement(is_test_post=True, _colony_id="src-colony-uuid")
        move = next(a for a in s._pending_actions(j) if a["kind"] == "move_to_sandbox")
        assert move["source_colony_id"] == "src-colony-uuid"

    def test_mark_scanned_emitted_for_every_top_comment(self, make_judgement):
        """Mark-scanned should fire for the post AND every comment
        included in the prompt — that's the unit of 'sentinel saw it'."""
        j = make_judgement(_comment_ids=["c1", "c2", "c3"])
        actions = s._pending_actions(j)
        post_marks = [a for a in actions if a["kind"] == "mark_scanned_post"]
        comment_marks = [a for a in actions if a["kind"] == "mark_scanned_comment"]
        assert len(post_marks) == 1
        assert [a["comment_id"] for a in comment_marks] == ["c1", "c2", "c3"]

    def test_mark_scanned_comment_drops_falsy_ids(self, make_judgement):
        """An empty / None comment_id slot shouldn't produce an
        action that would then PUT to /comments//sentinel-scanned."""
        j = make_judgement(_comment_ids=["c1", None, "", "c2"])
        comment_marks = [
            a for a in s._pending_actions(j) if a["kind"] == "mark_scanned_comment"
        ]
        assert [a["comment_id"] for a in comment_marks] == ["c1", "c2"]


# ────────────────────────────────────────────────────────────────────
# _apply_action: one action -> one client call
# ────────────────────────────────────────────────────────────────────


class TestApplyAction:
    def test_vote_calls_vote_post(self, mock_client: MagicMock):
        ok = s._apply_action(mock_client, "p1", {"kind": "vote", "value": 1})
        assert ok is True
        mock_client.vote_post.assert_called_once_with("p1", 1)

    def test_vote_returns_false_on_api_error(self, mock_client: MagicMock):
        mock_client.vote_post.side_effect = make_api_error(429)
        ok = s._apply_action(mock_client, "p1", {"kind": "vote", "value": -1})
        assert ok is False

    def test_mark_scanned_post_calls_sdk(self, mock_client: MagicMock):
        ok = s._apply_action(mock_client, "p1", {"kind": "mark_scanned_post"})
        assert ok is True
        mock_client.mark_post_scanned.assert_called_once_with("p1")

    def test_mark_scanned_comment_calls_sdk(self, mock_client: MagicMock):
        ok = s._apply_action(
            mock_client, "p1", {"kind": "mark_scanned_comment", "comment_id": "c1"}
        )
        assert ok is True
        mock_client.mark_comment_scanned.assert_called_once_with("c1")

    def test_mark_scanned_comment_without_id_returns_false(
        self, mock_client: MagicMock
    ):
        """Defensive: an action dict missing the comment_id shouldn't
        crash the apply loop; should just report failure so the loop
        moves on."""
        ok = s._apply_action(mock_client, "p1", {"kind": "mark_scanned_comment"})
        assert ok is False
        mock_client.mark_comment_scanned.assert_not_called()

    def test_unknown_kind_logs_warning_and_returns_false(
        self, mock_client: MagicMock, caplog: pytest.LogCaptureFixture
    ):
        """A future sentinel version saving a new kind, then rolling
        back to this version, would replay an unknown kind from
        memory. Return False so the retry budget eventually expires
        rather than looping forever, and log so it's visible."""
        with caplog.at_level("WARNING", logger="sentinel"):
            ok = s._apply_action(mock_client, "p1", {"kind": "totally-new-thing"})
        assert ok is False
        assert any("Unknown action kind" in r.message for r in caplog.records)

    def test_move_to_sandbox_skips_when_source_is_sandbox(
        self, mock_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        """If the post is already in a sandbox colony, the move
        endpoint isn't called — just a log line and a True return."""
        monkeypatch.setattr(
            s, "_is_sandbox_colony", lambda client, cid: True
        )
        ok = s._apply_action(
            mock_client,
            "p1",
            {"kind": "move_to_sandbox", "source_colony_id": "already-sandbox"},
        )
        assert ok is True
        mock_client.move_post_to_colony.assert_not_called()

    def test_move_to_sandbox_calls_sdk_when_source_is_community(
        self, mock_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            s, "_is_sandbox_colony", lambda client, cid: False
        )
        ok = s._apply_action(
            mock_client,
            "p1",
            {"kind": "move_to_sandbox", "source_colony_id": "regular-colony"},
        )
        assert ok is True
        mock_client.move_post_to_colony.assert_called_once_with("p1", s.TEST_POSTS_COLONY)


# ────────────────────────────────────────────────────────────────────
# act_on_judgement: end-to-end gating + apply
# ────────────────────────────────────────────────────────────────────


class TestActOnJudgement:
    def test_full_run_applies_everything_returns_no_failures(
        self, mock_client: MagicMock, make_judgement
    ):
        j = make_judgement(
            vote_recommendation="downvote",  # downvotes bypass UPVOTE_MIN_SCORE
            score=3,
            category="JUNK",
            language="fr",
            post_has_pii=True,
            _comment_ids=["c1"],
        )
        failed = s.act_on_judgement(mock_client, "p1", j)
        assert failed == []
        # vote, junk (via _raw_request), language (via _raw_request),
        # post_pii (via _raw_request), mark_scanned_post, mark_scanned_comment
        mock_client.vote_post.assert_called_once()
        mock_client.mark_post_scanned.assert_called_once_with("p1")
        mock_client.mark_comment_scanned.assert_called_once_with("c1")

    def test_no_vote_gate_skips_vote_and_junk_and_move(
        self, mock_client: MagicMock, make_judgement
    ):
        """--no-vote is the umbrella moderation-actions gate. Vote,
        junk-marking, and move-to-sandbox all share it."""
        j = make_judgement(
            vote_recommendation="downvote",
            score=2,
            category="JUNK",
            is_test_post=True,
        )
        s.act_on_judgement(mock_client, "p1", j, allow_vote=False)
        mock_client.vote_post.assert_not_called()
        mock_client.move_post_to_colony.assert_not_called()
        # junk endpoint is reached via _raw_request — confirm no PUT.
        calls = [c.args for c in mock_client._raw_request.call_args_list]
        assert not any("/junk" in path for _, path in calls)

    def test_no_lang_gate_skips_language(self, mock_client: MagicMock, make_judgement):
        j = make_judgement(language="es")
        s.act_on_judgement(mock_client, "p1", j, allow_lang=False)
        calls = [c.args for c in mock_client._raw_request.call_args_list]
        assert not any("/language" in path for _, path in calls)

    def test_no_pii_gate_skips_post_and_comment_pii(
        self, mock_client: MagicMock, make_judgement
    ):
        j = make_judgement(
            post_has_pii=True,
            pii_comment_indices=[1],
            _comment_ids=["c1"],
        )
        s.act_on_judgement(mock_client, "p1", j, allow_pii=False)
        calls = [c.args for c in mock_client._raw_request.call_args_list]
        assert not any("/pii" in path for _, path in calls)

    def test_no_mark_scanned_gate_skips_marks(
        self, mock_client: MagicMock, make_judgement
    ):
        j = make_judgement(_comment_ids=["c1"])
        s.act_on_judgement(mock_client, "p1", j, allow_mark_scanned=False)
        mock_client.mark_post_scanned.assert_not_called()
        mock_client.mark_comment_scanned.assert_not_called()

    def test_failed_actions_returned_for_persistence(
        self, mock_client: MagicMock, make_judgement
    ):
        """The whole point of returning ``failed`` is so the caller
        can persist it under ``_pending_actions`` for next-run retry.
        A failed mark-scanned should show up in the returned list."""
        mock_client.mark_post_scanned.side_effect = make_api_error(502)
        j = make_judgement()
        failed = s.act_on_judgement(mock_client, "p1", j)
        assert len(failed) == 1
        assert failed[0]["kind"] == "mark_scanned_post"

    def test_one_action_failing_does_not_block_others(
        self, mock_client: MagicMock, make_judgement
    ):
        """Each action is independent — if mark-scanned 502s but the
        vote succeeds, the vote should still have landed."""
        mock_client.mark_post_scanned.side_effect = make_api_error(502)
        j = make_judgement(vote_recommendation="downvote", score=3)
        failed = s.act_on_judgement(mock_client, "p1", j)
        mock_client.vote_post.assert_called_once()
        assert {a["kind"] for a in failed} == {"mark_scanned_post"}


# ────────────────────────────────────────────────────────────────────
# Advertisement flag + ads-colony downvote carve-out
# ────────────────────────────────────────────────────────────────────
class TestAdActions:
    def test_is_ad_emits_ad_action(self, make_judgement):
        j = make_judgement(is_ad=True)
        kinds = [a["kind"] for a in s._pending_actions(j)]
        assert "ad" in kinds

    def test_not_ad_emits_no_ad_action(self, make_judgement):
        j = make_judgement(is_ad=False)
        kinds = [a["kind"] for a in s._pending_actions(j)]
        assert "ad" not in kinds

    def test_ad_flag_is_colony_independent(self, make_judgement):
        """The ``is_ad`` flag itself is recorded everywhere — it's only the
        downvote that the ads colony exempts, not the flag."""
        j = make_judgement(is_ad=True, _colony_name="general")
        kinds = [a["kind"] for a in s._pending_actions(j)]
        assert "ad" in kinds

    def test_downvote_suppressed_for_ad_in_ads_colony(self, make_judgement):
        j = make_judgement(
            vote_recommendation="downvote", score=3, category="BAD",
            is_ad=True, _colony_name="ads",
        )
        kinds = [a["kind"] for a in s._pending_actions(j)]
        # No downvote — being an ad is not a downvote-worthy offence in /c/ads.
        assert "vote" not in kinds
        # ...but the post is still flagged as an ad.
        assert "ad" in kinds

    def test_downvote_kept_for_ad_in_other_colony(self, make_judgement):
        j = make_judgement(
            vote_recommendation="downvote", score=3, category="BAD",
            is_ad=True, _colony_name="general",
        )
        vote = next(a for a in s._pending_actions(j) if a["kind"] == "vote")
        assert vote["value"] == -1

    def test_downvote_kept_for_non_ad_in_ads_colony(self, make_judgement):
        """The carve-out only fires for ads. A genuinely bad NON-ad post in
        the ads colony is still downvoted."""
        j = make_judgement(
            vote_recommendation="downvote", score=3, category="BAD",
            is_ad=False, _colony_name="ads",
        )
        vote = next(a for a in s._pending_actions(j) if a["kind"] == "vote")
        assert vote["value"] == -1

    def test_upvote_unaffected_for_ad_in_ads_colony(self, make_judgement):
        """The carve-out only suppresses downvotes — a great ad can still be
        upvoted."""
        j = make_judgement(
            vote_recommendation="upvote", score=9, category="GOOD",
            is_ad=True, _colony_name="ads",
        )
        vote = next(a for a in s._pending_actions(j) if a["kind"] == "vote")
        assert vote["value"] == 1

    def test_junk_still_marked_for_ad_in_ads_colony(self, make_judgement):
        """A scam/gibberish post the model still rates JUNK in the ads colony
        is marked junk — the carve-out is downvote-only, not a free pass."""
        j = make_judgement(
            vote_recommendation="downvote", score=1, category="JUNK",
            is_ad=True, _colony_name="ads",
        )
        kinds = [a["kind"] for a in s._pending_actions(j)]
        assert "junk" in kinds
        assert "vote" not in kinds  # downvote still suppressed

    def test_apply_ad_action_calls_endpoint(self, mock_client):
        ok = s._apply_action(mock_client, "p1", {"kind": "ad"})
        assert ok is True
        mock_client._raw_request.assert_called_once_with("PUT", "/posts/p1/ad?is_ad=true")

    def test_flag_post_ad_false_on_403(self, mock_client):
        mock_client._raw_request.side_effect = make_api_error(403)
        assert s.flag_post_ad(mock_client, "p1", True) is False

    def test_colony_name_resolution_matches_ads(self, mock_client):
        mock_client.get_colonies.return_value = [
            {"id": "c-ads", "name": "Ads"},
            {"id": "c-gen", "name": "general"},
        ]
        s._COLONY_NAME_CACHE.clear()
        assert s._colony_name_for(mock_client, "c-ads") == "ads"
        assert s._colony_name_for(mock_client, "c-gen") == "general"
        s._COLONY_NAME_CACHE.clear()

    def test_build_analysis_text_includes_colony(self):
        text = s.build_analysis_text({
            "post": {"title": "Buy now", "body": "cheap widgets", "author": {"username": "x"}},
            "comments": [],
            "colony_name": "ads",
        })
        assert "Colony: ads" in text
