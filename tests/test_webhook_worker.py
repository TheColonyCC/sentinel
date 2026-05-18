"""Tests for ``WebhookWorker.enqueue`` — the gatekeeper that decides
whether a webhook delivery enters the work queue.

Three branches matter:
* in-flight dedup: a redelivery while the worker is mid-LLM-call
  must not double-queue.
* already-analyzed dedup: a redelivery after the post was
  successfully scored (and has no pending retries) must not re-run
  the LLM.
* queue-full: under load we return ``"full"`` so the Colony marks
  the delivery as failed and retries later instead of dropping it.

The worker thread itself is NOT started in these tests — we exercise
``enqueue`` synchronously since that's where all the policy lives.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import sentinel as s


def _build_worker(client: MagicMock | None = None) -> s.WebhookWorker:
    """Construct a worker without starting the background thread.

    ``__init__`` does a warm-start retry pass over memory; the
    isolated_cwd fixture (autouse) means memory is empty so that's
    a no-op.
    """
    return s.WebhookWorker(
        client=client or MagicMock(),
        own_username="sentinel-test",
        model="qwen3.5:9b-q4_K_M",
        allow_vote=True,
        allow_lang=True,
        allow_pii=True,
        allow_mark_scanned=True,
    )


class TestEnqueue:
    def test_first_call_queues(self):
        worker = _build_worker()
        assert worker.enqueue("post-1") == "queued"
        assert "post-1" in worker.inflight
        assert worker.q.qsize() == 1

    def test_duplicate_while_inflight_returns_duplicate(self):
        """Two webhook deliveries for the same post during a slow
        Ollama run should collapse to one enqueue. The Colony
        retries idempotently — sentinel must idempotently de-dupe."""
        worker = _build_worker()
        worker.enqueue("post-1")
        assert worker.enqueue("post-1") == "duplicate"
        assert worker.q.qsize() == 1

    def test_already_analyzed_returns_duplicate(self):
        """If memory already has a successful judgement (score > 0,
        no pending retries), a redelivery shouldn't re-burn an LLM
        call — the post hasn't changed, our verdict hasn't either."""
        s.MEMORY_FILE.write_text(json.dumps({
            "post-1": {"post_id": "post-1", "score": 8, "category": "OKAY"}
        }))
        worker = _build_worker()
        assert worker.enqueue("post-1") == "duplicate"
        assert worker.q.qsize() == 0

    def test_previously_failed_analysis_is_not_a_duplicate(self):
        """A score-0 entry is the marker for a failed Ollama call;
        the next webhook should be allowed to re-try."""
        s.MEMORY_FILE.write_text(json.dumps({
            "post-1": {"post_id": "post-1", "score": 0}
        }))
        worker = _build_worker()
        assert worker.enqueue("post-1") == "queued"

    def test_pending_actions_means_not_a_duplicate(self):
        """Successful analysis but actions failed -> the entry has
        ``_pending_actions``. A redelivery should re-enter the queue
        so the failed actions can be retried.

        Setup wrinkle: the worker's warm-start replays pending
        actions, so we hand it a client that keeps failing them —
        otherwise the pending slot would clear at construction and
        the enqueue call would (correctly) treat the post as a
        duplicate.
        """
        from colony_sdk import ColonyAPIError
        client = MagicMock()
        client.vote_post.side_effect = ColonyAPIError("boom", 502)

        s.MEMORY_FILE.write_text(json.dumps({
            "post-1": {
                "post_id": "post-1",
                "score": 7,
                "_pending_actions": [{"kind": "vote", "value": 1}],
            }
        }))
        worker = _build_worker(client=client)
        assert worker.enqueue("post-1") == "queued"

    def test_queue_full_returns_full_and_does_not_track_inflight(
        self, monkeypatch
    ):
        """When the queue is saturated, the post must NOT land in
        ``inflight`` — otherwise a future retry would be silently
        rejected as 'duplicate' even though we never enqueued it."""
        monkeypatch.setattr(s, "WEBHOOK_QUEUE_SIZE", 1)
        worker = _build_worker()
        # Saturate the queue.
        assert worker.enqueue("post-1") == "queued"
        # Second distinct post should bounce.
        assert worker.enqueue("post-2") == "full"
        assert "post-2" not in worker.inflight

    def test_distinct_posts_both_enqueue(self):
        worker = _build_worker()
        assert worker.enqueue("post-1") == "queued"
        assert worker.enqueue("post-2") == "queued"
        assert worker.q.qsize() == 2
        assert worker.inflight == {"post-1", "post-2"}
