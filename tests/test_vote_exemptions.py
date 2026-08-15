"""Local vote-exemption lists — ``never_upvote.txt`` / ``never_downvote.txt``.

Two gitignored, operator-owned files on the sentinel's own filesystem name
users this instance must never upvote / never downvote. The whole value of
the feature is that it is *absolute*: an operator who lists a name has said
"whatever the model decides, not this". So the tests below are mostly about
the ways an exemption list can silently fail open —

* matching that is too literal (case, a pasted ``@``, trailing whitespace),
* the two lists leaking into each other, so listing someone as
  never-downvote quietly also stops upvotes (or worse, the reverse),
* a cached read that never notices the operator edited the file, which in
  webhook mode means the list only takes effect on restart,
* a vote queued to ``_pending_actions`` BEFORE the name was listed, then
  replayed days later and cast anyway.

Every one of those fails quietly in production: the sentinel keeps running,
logs nothing unusual, and votes on somebody it was told not to.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import sentinel as s


@pytest.fixture(autouse=True)
def exemption_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point both lists at per-test paths and clear the read cache.

    The cache is keyed on path + (mtime, size) and lives for the life of the
    process, so without this reset one test's list would be visible to the
    next — and, worse, a test could pass because of a *stale cached* value
    rather than the file it just wrote.
    """
    up = tmp_path / "never_upvote.txt"
    down = tmp_path / "never_downvote.txt"
    monkeypatch.setattr(s, "NEVER_UPVOTE_FILE", up)
    monkeypatch.setattr(s, "NEVER_DOWNVOTE_FILE", down)
    s._VOTE_EXEMPT_CACHE.clear()
    yield up, down
    s._VOTE_EXEMPT_CACHE.clear()


# ────────────────────────────────────────────────────────────────────
# Parsing
# ────────────────────────────────────────────────────────────────────


class TestParsing:
    def test_one_username_per_line(self):
        assert s._parse_username_list("alice\nbob\n") == {"alice", "bob"}

    def test_blank_lines_and_whitespace_are_ignored(self):
        assert s._parse_username_list("\n  alice  \n\n\tbob\n   \n") == {"alice", "bob"}

    def test_whole_line_and_trailing_comments_are_stripped(self):
        text = "# who not to touch\nalice   # a human\nbob\n"
        assert s._parse_username_list(text) == {"alice", "bob"}

    def test_a_comment_only_file_yields_nothing(self):
        """The shipped ``.example`` files are entirely comments. If a
        comment leaked through as a username, copying the example
        verbatim would exempt a user literally called ``# my-other-agent``
        — harmless — but would also report a non-zero count at startup and
        tell the operator the list was live when it holds nobody."""
        assert s._parse_username_list("# nobody\n#  also nobody\n") == frozenset()

    def test_leading_at_is_stripped(self):
        """An operator copying a mention out of the site pastes ``@name``."""
        assert s._parse_username_list("@alice\n") == {"alice"}

    def test_matching_is_case_insensitive(self):
        assert s._parse_username_list("Alice\nBOB\n") == {"alice", "bob"}

    def test_crlf_line_endings(self):
        """A file edited on Windows / pasted through a browser. ``\\r``
        surviving into the name means nothing ever matches."""
        assert s._parse_username_list("alice\r\nbob\r\n") == {"alice", "bob"}


# ────────────────────────────────────────────────────────────────────
# Loading from disk
# ────────────────────────────────────────────────────────────────────


class TestLoading:
    def test_missing_file_is_no_exemptions(self, exemption_files):
        """The default state for every operator who never uses this."""
        up, _down = exemption_files
        assert not up.exists()
        assert s._load_username_list(up) == frozenset()

    def test_unreadable_file_is_empty_and_does_not_raise(self, tmp_path, caplog):
        """A path that exists but can't be read (here: a directory of that
        name) is an operator mistake. It must not take the scan down — but
        it must not pass silently either, or the operator sees "no
        exemptions" and assumes the list is simply empty."""
        bogus = tmp_path / "a_directory_not_a_file.txt"
        bogus.mkdir()
        with caplog.at_level("WARNING"):
            assert s._load_username_list(bogus) == frozenset()
        assert any("Could not read" in r.message for r in caplog.records)

    def test_edits_are_picked_up_without_a_restart(self, exemption_files):
        """The webhook process runs for days. If the list were cached for
        the life of the process, adding a username would do nothing until
        the next restart — and the operator would have no way to tell."""
        up, _down = exemption_files
        up.write_text("alice\n", encoding="utf-8")
        assert s._load_username_list(up) == {"alice"}
        # Same size, different content: a cache keyed on size alone would
        # miss this. Bump mtime explicitly — a same-second rewrite in a
        # fast test can otherwise land on an identical coarse timestamp.
        up.write_text("bobby\n", encoding="utf-8")
        import os
        st = up.stat()
        os.utime(up, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        assert s._load_username_list(up) == {"bobby"}

    def test_a_deleted_list_stops_exempting(self, exemption_files):
        """Removing the file is how an operator turns the feature off."""
        up, _down = exemption_files
        up.write_text("alice\n", encoding="utf-8")
        assert s.vote_is_exempt("alice", 1) is True
        up.unlink()
        assert s.vote_is_exempt("alice", 1) is False


# ────────────────────────────────────────────────────────────────────
# vote_is_exempt — the two lists must not leak into each other
# ────────────────────────────────────────────────────────────────────


class TestExemptionLookup:
    def test_never_upvote_blocks_only_upvotes(self, exemption_files):
        up, _down = exemption_files
        up.write_text("alice\n", encoding="utf-8")
        assert s.vote_is_exempt("alice", 1) is True
        assert s.vote_is_exempt("alice", -1) is False, (
            "a never-UPVOTE entry must not also suppress downvotes — the "
            "two lists exist separately so each can be expressed alone"
        )

    def test_never_downvote_blocks_only_downvotes(self, exemption_files):
        _up, down = exemption_files
        down.write_text("alice\n", encoding="utf-8")
        assert s.vote_is_exempt("alice", -1) is True
        assert s.vote_is_exempt("alice", 1) is False

    def test_listing_in_both_blocks_both(self, exemption_files):
        up, down = exemption_files
        up.write_text("alice\n", encoding="utf-8")
        down.write_text("alice\n", encoding="utf-8")
        assert s.vote_is_exempt("alice", 1) is True
        assert s.vote_is_exempt("alice", -1) is True

    def test_an_unlisted_author_is_never_exempt(self, exemption_files):
        up, down = exemption_files
        up.write_text("alice\n", encoding="utf-8")
        down.write_text("alice\n", encoding="utf-8")
        assert s.vote_is_exempt("bob", 1) is False
        assert s.vote_is_exempt("bob", -1) is False

    @pytest.mark.parametrize("written,queried", [
        ("Alice", "alice"), ("alice", "ALICE"), ("@alice", "alice"),
        ("alice", "@alice"), ("  alice  ", "alice"),
    ])
    def test_matching_survives_casing_at_signs_and_whitespace(
        self, exemption_files, written, queried,
    ):
        up, _down = exemption_files
        up.write_text(f"{written}\n", encoding="utf-8")
        assert s.vote_is_exempt(queried, 1) is True

    def test_unknown_author_is_not_exempt(self, exemption_files):
        """``None`` cannot match a list. Fail-open is deliberate and
        bounded — see ``_apply_action``'s replay note — because the
        alternative (fail-closed on unknown) would silently stop the
        sentinel voting at all if the author field ever went missing."""
        up, _down = exemption_files
        up.write_text("alice\n", encoding="utf-8")
        assert s.vote_is_exempt(None, 1) is False
        assert s.vote_is_exempt("", 1) is False

    def test_a_zero_vote_is_not_a_vote(self, exemption_files):
        up, _down = exemption_files
        up.write_text("alice\n", encoding="utf-8")
        assert s.vote_is_exempt("alice", 0) is False


# ────────────────────────────────────────────────────────────────────
# _pending_actions — the vote is never queued in the first place
# ────────────────────────────────────────────────────────────────────


class TestPendingActionsRespectsTheLists:
    def test_listed_author_gets_no_upvote_action(self, exemption_files, make_judgement):
        up, _down = exemption_files
        up.write_text("alice\n", encoding="utf-8")
        j = make_judgement(
            vote_recommendation="upvote", score=s.UPVOTE_MIN_SCORE,
            _author_username="alice",
        )
        kinds = [a["kind"] for a in s._pending_actions(j)]
        assert "vote" not in kinds

    def test_listed_author_gets_no_downvote_action(self, exemption_files, make_judgement):
        _up, down = exemption_files
        down.write_text("alice\n", encoding="utf-8")
        j = make_judgement(
            vote_recommendation="downvote", score=2, category="BAD",
            _author_username="alice",
        )
        kinds = [a["kind"] for a in s._pending_actions(j)]
        assert "vote" not in kinds

    def test_the_rest_of_the_pipeline_is_untouched(self, exemption_files, make_judgement):
        """An exemption suppresses the VOTE and nothing else. A JUNK post
        by a listed author still gets marked, still gets its language set,
        still gets marked scanned — the operator asked us not to vote on
        them, not to stop moderating the platform."""
        _up, down = exemption_files
        down.write_text("alice\n", encoding="utf-8")
        j = make_judgement(
            vote_recommendation="downvote", score=1, category="JUNK",
            language="fr", post_has_pii=True, _author_username="alice",
        )
        kinds = [a["kind"] for a in s._pending_actions(j)]
        assert "vote" not in kinds
        assert "junk" in kinds
        assert "language" in kinds
        assert "post_pii" in kinds
        assert "mark_scanned_post" in kinds

    def test_an_unlisted_author_still_gets_voted_on(self, exemption_files, make_judgement):
        """The control. Without this, a bug that suppressed EVERY vote
        would pass every other test in this file."""
        up, down = exemption_files
        up.write_text("alice\n", encoding="utf-8")
        down.write_text("alice\n", encoding="utf-8")
        j = make_judgement(
            vote_recommendation="upvote", score=s.UPVOTE_MIN_SCORE,
            _author_username="bob",
        )
        votes = [a for a in s._pending_actions(j) if a["kind"] == "vote"]
        assert len(votes) == 1
        assert votes[0]["value"] == 1

    def test_a_queued_vote_carries_its_author(self, exemption_files, make_judgement):
        """Load-bearing for replay: ``_apply_action`` re-checks the lists
        when a saved vote comes back, and it can only do that if the
        author travelled with the action into memory."""
        j = make_judgement(
            vote_recommendation="upvote", score=s.UPVOTE_MIN_SCORE,
            _author_username="bob",
        )
        vote = next(a for a in s._pending_actions(j) if a["kind"] == "vote")
        assert vote["author"] == "bob"

    def test_no_author_key_when_the_username_is_unknown(self, make_judgement):
        """Keeps the persisted shape identical to what older sentinels
        wrote, so a memory file stays readable in both directions."""
        j = make_judgement(
            vote_recommendation="upvote", score=s.UPVOTE_MIN_SCORE,
            _author_username=None,
        )
        vote = next(a for a in s._pending_actions(j) if a["kind"] == "vote")
        assert "author" not in vote


# ────────────────────────────────────────────────────────────────────
# _apply_action — the replay path
# ────────────────────────────────────────────────────────────────────


class TestReplayIsRecheckedAgainstTheCurrentList:
    def test_a_vote_queued_before_the_listing_is_not_cast(
        self, exemption_files, mock_client,
    ):
        """The scenario the second check exists for: the vote failed with a
        502 last week and was persisted; since then the operator listed the
        author. Replaying the stored decision verbatim would cast exactly
        the vote they asked us never to cast."""
        _up, down = exemption_files
        down.write_text("alice\n", encoding="utf-8")
        ok = s._apply_action(
            mock_client, "p1", {"kind": "vote", "value": -1, "author": "alice"},
        )
        assert ok is True, (
            "must report SUCCESS — a False here re-queues the forbidden "
            "vote as a failure and retries it for five more runs"
        )
        mock_client.vote_post.assert_not_called()

    def test_an_unlisted_author_still_replays(self, exemption_files, mock_client):
        _up, down = exemption_files
        down.write_text("alice\n", encoding="utf-8")
        ok = s._apply_action(
            mock_client, "p1", {"kind": "vote", "value": -1, "author": "bob"},
        )
        assert ok is True
        mock_client.vote_post.assert_called_once_with("p1", -1)

    def test_a_pre_upgrade_action_with_no_author_still_replays(
        self, exemption_files, mock_client,
    ):
        """Memory written by a sentinel older than this feature has no
        ``author`` on the vote dict. It can't be re-checked, so it is
        applied; the gap is bounded by the 5-attempt pending ceiling. The
        assertion pins that this is a DECISION, not an accident."""
        _up, down = exemption_files
        down.write_text("alice\n", encoding="utf-8")
        ok = s._apply_action(mock_client, "p1", {"kind": "vote", "value": -1})
        assert ok is True
        mock_client.vote_post.assert_called_once_with("p1", -1)

    def test_the_direction_is_respected_on_replay(self, exemption_files, mock_client):
        """A never-DOWNVOTE listing must not block a replayed upvote."""
        _up, down = exemption_files
        down.write_text("alice\n", encoding="utf-8")
        ok = s._apply_action(
            mock_client, "p1", {"kind": "vote", "value": 1, "author": "alice"},
        )
        assert ok is True
        mock_client.vote_post.assert_called_once_with("p1", 1)


# ────────────────────────────────────────────────────────────────────
# End-to-end through act_on_judgement, and the startup announcement
# ────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_act_on_judgement_casts_no_vote_for_a_listed_author(
        self, exemption_files, mock_client, make_judgement,
    ):
        """The seam that matters — every earlier test could pass while the
        wiring between ``_pending_actions`` and the client was wrong."""
        up, _down = exemption_files
        up.write_text("alice\n", encoding="utf-8")
        failed = s.act_on_judgement(
            mock_client, "p1",
            make_judgement(
                vote_recommendation="upvote", score=s.UPVOTE_MIN_SCORE,
                _author_username="alice",
            ),
        )
        mock_client.vote_post.assert_not_called()
        assert failed == [], "a suppressed vote is not a failure to retry"

    def test_act_on_judgement_still_votes_for_everyone_else(
        self, exemption_files, mock_client, make_judgement,
    ):
        up, _down = exemption_files
        up.write_text("alice\n", encoding="utf-8")
        s.act_on_judgement(
            mock_client, "p1",
            make_judgement(
                vote_recommendation="upvote", score=s.UPVOTE_MIN_SCORE,
                _author_username="bob",
            ),
        )
        mock_client.vote_post.assert_called_once_with("p1", 1)


class TestAnalyzePostCarriesTheAuthor:
    def test_the_username_is_stamped_onto_the_judgement(self, monkeypatch):
        """Without this the lists can never match anything: the judgement
        is the only thing ``_pending_actions`` sees."""
        monkeypatch.setattr(
            s, "call_ollama",
            lambda model, messages: {
                "score": 9, "category": "GOOD", "vote_recommendation": "upvote",
                "language": "en",
            },
        )
        post_data = {
            "post": {"id": "p1", "title": "t", "body": "b",
                     "author": {"username": "alice"}, "colony_id": "c1"},
            "comments": [],
            "colony_name": "general",
        }
        j = s.analyze_post(post_data, "model")
        assert j is not None
        assert j["_author_username"] == "alice"

    def test_a_missing_author_block_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(
            s, "call_ollama",
            lambda model, messages: {"score": 5, "category": "OKAY"},
        )
        post_data = {"post": {"id": "p1"}, "comments": [], "colony_name": None}
        j = s.analyze_post(post_data, "model")
        assert j is not None
        assert j["_author_username"] is None


class TestStartupAnnouncement:
    def test_both_lists_are_logged_with_absolute_paths(
        self, exemption_files, caplog,
    ):
        """"No exemptions" and "the list is in the wrong directory" look
        identical from the outside. Logging the resolved path is the only
        thing that lets an operator tell them apart."""
        up, _down = exemption_files
        up.write_text("alice\nbob\n", encoding="utf-8")
        with caplog.at_level("INFO"):
            s.log_vote_exemptions()
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "never-upvote" in text and "never-downvote" in text
        assert "2 entries" in text
        assert "not present" in text, "the absent list must say so, not stay silent"
        assert str(up.resolve()) in text
