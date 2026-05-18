"""Tests for ``retry_pending_actions``.

Failed actions from a previous run get persisted in
``memory[post_id]["_pending_actions"]`` for replay on the next run /
webhook startup. The retry path must (a) replay each pending action
without re-invoking the LLM, (b) clear the pending slot on full
success, (c) increment an attempt counter on partial failure, and
(d) drop entries entirely after the attempt-ceiling so a permanently
broken target (e.g. a deleted post) doesn't burn API calls forever.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from colony_sdk import ColonyAPIError

import sentinel as s


def make_api_error(status: int = 500, message: str = "boom") -> ColonyAPIError:
    return ColonyAPIError(message, status)


class TestRetryPendingActions:
    def test_no_pending_means_nothing_retried(self, mock_client: MagicMock):
        memory = {"p1": {"score": 7, "category": "OKAY"}}  # no _pending_actions
        retried = s.retry_pending_actions(mock_client, memory)
        assert retried == 0
        mock_client.vote_post.assert_not_called()

    def test_successful_retry_clears_pending_slot(self, mock_client: MagicMock):
        """A previously-failed vote that succeeds on retry should
        drop out of ``_pending_actions`` so the next run doesn't
        replay it again."""
        memory = {
            "p1": {
                "score": 9,
                "_pending_actions": [{"kind": "vote", "value": 1}],
            }
        }
        retried = s.retry_pending_actions(mock_client, memory)
        assert retried == 1
        assert "_pending_actions" not in memory["p1"]
        assert "_pending_attempts" not in memory["p1"]
        mock_client.vote_post.assert_called_once_with("p1", 1)

    def test_partial_failure_persists_remaining_and_increments_attempts(
        self, mock_client: MagicMock
    ):
        """Vote succeeds but mark-scanned fails: the vote action
        clears, mark-scanned stays in the pending slot, attempt count
        ticks up so the ceiling can eventually evict it."""
        mock_client.mark_post_scanned.side_effect = make_api_error(502)
        memory = {
            "p1": {
                "score": 9,
                "_pending_actions": [
                    {"kind": "vote", "value": 1},
                    {"kind": "mark_scanned_post"},
                ],
            }
        }
        s.retry_pending_actions(mock_client, memory)
        remaining = memory["p1"]["_pending_actions"]
        assert [a["kind"] for a in remaining] == ["mark_scanned_post"]
        assert memory["p1"]["_pending_attempts"] == 1

    def test_attempts_carry_forward_across_runs(self, mock_client: MagicMock):
        """Sequential calls of retry_pending_actions on a still-failing
        action should increment ``_pending_attempts`` each time."""
        mock_client.vote_post.side_effect = make_api_error(429)
        memory = {
            "p1": {
                "_pending_actions": [{"kind": "vote", "value": -1}],
            }
        }
        s.retry_pending_actions(mock_client, memory)
        assert memory["p1"]["_pending_attempts"] == 1
        s.retry_pending_actions(mock_client, memory)
        assert memory["p1"]["_pending_attempts"] == 2

    def test_ceiling_drops_pending_after_too_many_attempts(
        self, mock_client: MagicMock
    ):
        """After 5 failed attempts the entry is dropped entirely —
        the post was probably deleted between runs, no point trying
        forever."""
        mock_client.vote_post.side_effect = make_api_error(404)
        memory = {
            "p1": {
                "_pending_actions": [{"kind": "vote", "value": 1}],
                "_pending_attempts": 5,  # already at the ceiling
            }
        }
        s.retry_pending_actions(mock_client, memory)
        assert "_pending_actions" not in memory["p1"]
        assert "_pending_attempts" not in memory["p1"]
        # And critically: when we hit the ceiling we DON'T burn
        # another API call on a doomed target.
        mock_client.vote_post.assert_not_called()

    def test_multiple_posts_are_independent(self, mock_client: MagicMock):
        """One post's success / failure shouldn't leak into another's
        pending slot."""
        mock_client.vote_post.side_effect = [
            make_api_error(429),       # p1 vote: fails
            {"score": 1},               # p2 vote: succeeds
        ]
        memory = {
            "p1": {"_pending_actions": [{"kind": "vote", "value": 1}]},
            "p2": {"_pending_actions": [{"kind": "vote", "value": 1}]},
        }
        s.retry_pending_actions(mock_client, memory)
        assert memory["p1"]["_pending_actions"]  # still pending
        assert memory["p1"]["_pending_attempts"] == 1
        assert "_pending_actions" not in memory["p2"]
