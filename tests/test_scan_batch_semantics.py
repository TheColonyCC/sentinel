"""`--limit N` means "analyse up to N posts", and the fetch is materialised.

Two properties that look unrelated and are not: fixing the first naively
breaks the second.

**--limit is a budget of ANALYSES.** Several filters run AFTER the fetch —
local memory, `--days`, and the sentinel's own posts — so fetching exactly N
routinely analysed fewer than N and said nothing about it. `--limit 10`
delivering three is the same shape as the bug in #22: a run that reports
success having done less work than asked. So the scan over-fetches a
candidate window and stops at N analyses.

**The window is materialised before the first mark.** The obvious way to make
--limit honest is to consume `iter_posts` lazily until N are analysed. That is
wrong here, and quietly so. `_process_post` marks each post scanned, which
REMOVES it from the `sentinel_scanned=false` set the server is paginating, and
`iter_posts` pages by offset. A lazy generator would have the window slide
underneath it: mark k posts, and the next page starts k posts further into a
set that just lost k members, skipping k posts that were never seen. Collecting
the whole window before the first mark is what keeps the offsets stable.

That is why `list(...)` is load-bearing rather than incidental, and why
`test_the_window_is_fully_read_before_anything_is_marked` exists — it is the
only thing standing between a future "optimisation" and a scanner that
silently skips a post for every post it completes.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import sentinel as s


def _args(**over):
    base = dict(
        dry_run=True, no_vote=True, no_pii=True, model="m", username=None,
        force=False, post_id=None, sort="new", limit=3, days=7, confirm=False,
        include_scanned=False, allow_cpu=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _post(i, *, author="someone", created=None):
    from datetime import datetime, timedelta, timezone
    when = created or (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    return {"id": f"post-{i:03d}", "title": f"Post {i}",
            "created_at": when, "author": {"username": author}}


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Drive cmd_scan with every side effect stubbed except the batch logic."""
    monkeypatch.setattr(s, "MEMORY_FILE", tmp_path / "mem.json")
    monkeypatch.setattr(s, "ensure_model_available", lambda m: None)
    monkeypatch.setattr(s, "ensure_gpu_available", lambda m: None, raising=False)
    monkeypatch.setattr(s, "log_vote_exemptions", lambda: None)

    state = SimpleNamespace(requested=[], analysed=[], yielded=[], events=[])

    client = SimpleNamespace()

    def iter_posts(**kw):
        state.requested.append(kw)
        for p in state.available:
            state.yielded.append(p["id"])
            state.events.append(("yield", p["id"]))
            yield p

    client.iter_posts = iter_posts
    monkeypatch.setattr(s, "get_or_register_client",
                        lambda u: (client, {"username": "sentinel-bot"}))
    monkeypatch.setattr(s, "fetch_post_with_comments",
                        lambda c, pid: {"post": {"id": pid}})

    def process(client_, data, args, memory, results):
        pid = data["post"]["id"]
        state.analysed.append(pid)
        state.events.append(("analyse", pid))
        memory[pid] = {"score": 5}
        return True

    monkeypatch.setattr(s, "_process_post", process)
    state.available = []
    return state


class TestLimitIsABudgetOfAnalyses:
    def test_it_stops_at_the_limit(self, harness):
        harness.available = [_post(i) for i in range(10)]
        s.cmd_scan(_args(limit=3))
        assert harness.analysed == ["post-000", "post-001", "post-002"]

    def test_it_over_fetches_so_skips_do_not_shrink_the_batch(self, harness):
        """The whole point: ask for more candidates than the budget, because
        some will be filtered out after the fetch."""
        harness.available = [_post(i) for i in range(30)]
        s.cmd_scan(_args(limit=5))
        assert harness.requested[0]["max_results"] == 5 * s.SCAN_FETCH_MULTIPLIER

    def test_the_over_fetch_is_capped(self, harness):
        """A large --limit must not turn into an enormous request."""
        harness.available = []
        s.cmd_scan(_args(limit=10_000))
        assert harness.requested[0]["max_results"] == s.SCAN_FETCH_CAP

    def test_skipped_posts_do_not_consume_the_budget(self, harness):
        """Three of the first four are the sentinel's own; a budget of 3 must
        still analyse 3, not 1. This is the behaviour that was broken."""
        harness.available = [
            _post(0, author="sentinel-bot"),
            _post(1),
            _post(2, author="sentinel-bot"),
            _post(3),
            _post(4),
            _post(5),
        ]
        s.cmd_scan(_args(limit=3))
        assert harness.analysed == ["post-001", "post-003", "post-004"]

    def test_posts_beyond_the_budget_are_left_untouched(self, harness):
        """Anything past the limit must stay unanalysed AND unmarked, so the
        next run picks it up rather than it being silently consumed."""
        harness.available = [_post(i) for i in range(10)]
        s.cmd_scan(_args(limit=2))
        assert "post-002" not in harness.analysed
        assert len(harness.analysed) == 2

    def test_a_short_batch_is_reported(self, harness, caplog):
        """Silence here is what let 'analysed 3 of a requested 10' read as a
        healthy scan."""
        harness.available = [_post(0), _post(1)]
        with caplog.at_level("INFO"):
            s.cmd_scan(_args(limit=10))
        assert any("Analysed 2 of a requested 10" in r.message for r in caplog.records)

    def test_a_full_batch_is_not_reported_as_short(self, harness, caplog):
        harness.available = [_post(i) for i in range(10)]
        with caplog.at_level("INFO"):
            s.cmd_scan(_args(limit=3))
        assert not any("of a requested" in r.message for r in caplog.records)


class TestTheWindowIsMaterialised:
    def test_the_window_is_fully_read_before_anything_is_marked(self, harness):
        """The load-bearing one.

        Every post must be pulled off the generator BEFORE the first analysis
        (and therefore before the first mark_post_scanned). If this ever
        interleaves, the server-side offset window shifts under the scanner
        and it skips one unseen post per post completed — silently, with a
        healthy-looking log.
        """
        harness.available = [_post(i) for i in range(6)]
        s.cmd_scan(_args(limit=6))

        kinds = [k for k, _ in harness.events]
        first_analyse = kinds.index("analyse")
        assert "yield" not in kinds[first_analyse:], (
            "the fetch is being consumed lazily — posts were still being "
            "pulled after marking began, so the offset window will slide and "
            "the scan will skip work. Keep the list(...) around iter_posts."
        )
        assert harness.yielded == [f"post-{i:03d}" for i in range(6)]

    def test_the_whole_window_is_read_even_when_the_budget_is_smaller(self, harness):
        """A consequence worth stating: over-fetching means we read more than
        we analyse. That is the price of stable offsets, and it is cheap — the
        fetch is one request, the analysis is an LLM call per post."""
        harness.available = [_post(i) for i in range(9)]
        s.cmd_scan(_args(limit=3))
        assert len(harness.yielded) == 9
        assert len(harness.analysed) == 3
