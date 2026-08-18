"""`make run` must fetch work not yet done, not the newest N posts.

Reported 2026-08-17: a routine `make run` appeared to do nothing. It fetched
the ten newest posts, recognised all ten from its own local memory, skipped
them client-side, and exited reporting success — because the Sentinel keeps up
with the front page, so the newest ten are exactly the ten it has already
moderated. `--limit 10` meant "ten posts, probably none of which need
moderating".

The server has supported the filter for some time. Verified against production
the day this was written, where it partitions the corpus exactly:

    no filter               total = 15,485
    sentinel_scanned=false  total = 12,830
    sentinel_scanned=true   total =  2,655

12,830 + 2,655 = 15,485. So the backlog was real, reachable, and 12,830 posts
deep while the agent re-read the front page. The gap was that colony-sdk had
no way to send the parameter, so `iter_posts` could not express the request.

Note the shape of the failure, because it is the reason this went unnoticed:
nothing errored. An undeclared query param is dropped by the API rather than
rejected, so the unfiltered request returns a perfectly good 200 — just with
the wrong rows. A filter that silently does not apply returns MORE than asked
for, and more looks like success.
"""
from __future__ import annotations

import argparse

import sentinel as s


def _args(**overrides) -> argparse.Namespace:
    base = {"force": False, "include_scanned": False}
    base.update(overrides)
    return argparse.Namespace(**base)


class TestDefaultAsksForTheBacklog:
    def test_default_restricts_to_unscanned(self):
        assert s.scanned_filter_for(_args()) is False

    def test_it_is_false_not_none(self):
        """`None` means "no filter" on the wire. Returning it here would
        reinstate the exact bug while still looking like a filter is set."""
        result = s.scanned_filter_for(_args())
        assert result is not None
        assert result is False


class TestExplicitRevisitDisablesTheFilter:
    def test_force_clears_the_filter(self):
        """`--force` already discards local memory to re-analyse finished
        work; restricting to unscanned would leave it with nothing to do."""
        assert s.scanned_filter_for(_args(force=True)) is None

    def test_include_scanned_clears_the_filter(self):
        assert s.scanned_filter_for(_args(include_scanned=True)) is None

    def test_both_together_clears_the_filter(self):
        assert s.scanned_filter_for(_args(force=True, include_scanned=True)) is None


class TestTheFilterReachesTheClient:
    """The policy helper returning the right value is worth nothing if the
    fetch does not pass it — that was the original defect one layer up, where
    the server supported the filter and the caller never sent it."""

    def _run_scan_capturing_fetch(self, monkeypatch, args) -> dict:
        seen: dict = {}

        def _fake_iter_posts(**kwargs):
            seen.update(kwargs)
            return iter([])

        class _FakeClient:
            iter_posts = staticmethod(_fake_iter_posts)

        monkeypatch.setattr(s, "log_model_in_use", lambda *a, **k: None)
        monkeypatch.setattr(s, "log_vote_exemptions", lambda *a, **k: None)
        monkeypatch.setattr(s, "ensure_model_available", lambda *a, **k: None)
        monkeypatch.setattr(
            s, "get_or_register_client",
            lambda *a, **k: (_FakeClient(), {"username": "sentinel"}),
        )
        monkeypatch.setattr(s, "load_memory", lambda *a, **k: {})
        monkeypatch.setattr(s, "prune_memory", lambda m, *a, **k: m)
        monkeypatch.setattr(s, "save_memory", lambda *a, **k: None)
        monkeypatch.setattr(s, "get_processed_ids", lambda *a, **k: set())
        s.cmd_scan(args)
        return seen

    def test_scan_sends_sentinel_scanned_false_by_default(self, monkeypatch):
        seen = self._run_scan_capturing_fetch(
            monkeypatch,
            _args(
                model="m", limit=10, sort="new", days=7, post_id=None,
                no_vote=True, no_pii=True, confirm=False, dry_run=True,
                username=None,
            ),
        )
        assert seen.get("sentinel_scanned") is False, (
            f"cmd_scan fetched with {seen!r} — without sentinel_scanned=False "
            f"it re-reads the front page and moderates nothing"
        )

    def test_scan_omits_the_filter_under_force(self, monkeypatch):
        seen = self._run_scan_capturing_fetch(
            monkeypatch,
            _args(
                force=True, model="m", limit=10, sort="new", days=7,
                post_id=None, no_vote=True, no_pii=True, confirm=False,
                dry_run=True, username=None,
            ),
        )
        assert seen.get("sentinel_scanned") is None


class TestTheFlagIsWiredIntoTheCli:
    def test_include_scanned_is_accepted_and_defaults_off(self):
        parser = s.build_parser()
        assert parser.parse_args(["scan"]).include_scanned is False
        assert parser.parse_args(["scan", "--include-scanned"]).include_scanned is True

    def test_bare_invocation_still_gets_the_flag(self):
        """`make run` passes `scan` explicitly, but a bare invocation is
        normalised to it — and would AttributeError in cmd_scan if the flag
        lived on a different subparser."""
        parser = s.build_parser()
        args = parser.parse_args(s._normalize_argv([]))
        assert args.command == "scan"
        assert s.scanned_filter_for(args) is False
