"""Local never-upvote strings — ``never_upvote_strings.txt``.

The content-keyed sibling of ``never_upvote.txt``: phrases which, if present
in a post's title or body, mean this sentinel never upvotes it. Like the
username lists, its whole value is that it is absolute, so these tests are
mostly about the ways it could silently fail open —

* matching that is too literal (case), or a parser that eats a phrase
  containing ``#`` as a comment,
* the list leaking into the downvote direction or the other actions,
* the wiring: a check that exists but is never handed the post text,
* a vote queued to ``_pending_actions`` BEFORE the phrase was listed (or
  before the post was edited to contain it), then replayed and cast anyway,

plus the one cost the design must not quietly pay: post bodies piling up in
the persisted memory file.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from colony_sdk import ColonyAPIError

import sentinel as s


@pytest.fixture(autouse=True)
def strings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point every list at per-test paths and clear the read cache, so one
    test's list (or a stale cached parse) can never leak into the next."""
    path = tmp_path / "never_upvote_strings.txt"
    monkeypatch.setattr(s, "NEVER_UPVOTE_STRINGS_FILE", path)
    monkeypatch.setattr(s, "NEVER_UPVOTE_FILE", tmp_path / "never_upvote.txt")
    monkeypatch.setattr(s, "NEVER_DOWNVOTE_FILE", tmp_path / "never_downvote.txt")
    s._VOTE_EXEMPT_CACHE.clear()
    yield path
    s._VOTE_EXEMPT_CACHE.clear()


def _post(title: str = "A post", body: str = "Ordinary text.") -> dict:
    return {"id": "p1", "title": title, "body": body, "author": {"username": "bob"}}


# ────────────────────────────────────────────────────────────────────
# Parsing
# ────────────────────────────────────────────────────────────────────


class TestParsing:
    def test_one_phrase_per_line_casefolded(self):
        assert s._parse_string_list("Buy Now\nlimited time OFFER\n") == {
            "buy now", "limited time offer",
        }

    def test_blank_lines_whitespace_and_crlf(self):
        assert s._parse_string_list("\n  buy now  \r\n\t\r\nfree\r\n") == {"buy now", "free"}

    def test_whole_line_comments_are_ignored(self):
        assert s._parse_string_list("# a note\n   # indented note\nbuy now\n") == {"buy now"}

    def test_a_hash_inside_a_phrase_is_kept(self):
        """No trailing comments: the username lists strip ``# ...``, and doing
        the same here would silently truncate a phrase like this one."""
        assert s._parse_string_list("issue #42 fixed\n") == {"issue #42 fixed"}

    def test_quotes_allow_a_leading_hash_and_surrounding_spaces(self):
        assert s._parse_string_list('"#sponsored"\n" scam "\n') == {"#sponsored", " scam "}

    def test_empty_or_whitespace_only_entries_are_dropped(self):
        """``" "`` would otherwise match nearly every post on the platform."""
        assert s._parse_string_list('""\n" "\n"\t"\n') == frozenset()


# ────────────────────────────────────────────────────────────────────
# Matching + reload
# ────────────────────────────────────────────────────────────────────


class TestMatching:
    def test_absent_file_matches_nothing(self):
        assert s.never_upvote_string_in("buy now") is None

    def test_case_insensitive_substring_in_title_or_body(self, strings_file):
        strings_file.write_text("buy now\n", encoding="utf-8")
        assert s.never_upvote_string_in(s.post_text_for_lists(_post(title="BUY NOW!"))) == "buy now"
        assert s.never_upvote_string_in(s.post_text_for_lists(_post(body="…then Buy Now."))) == "buy now"
        assert s.never_upvote_string_in(s.post_text_for_lists(_post())) is None

    def test_a_phrase_does_not_match_across_title_and_body(self, strings_file):
        """Title and body are joined with a newline, so a phrase cannot be
        assembled from the end of one and the start of the other."""
        strings_file.write_text("now offer\n", encoding="utf-8")
        assert s.never_upvote_string_in(
            s.post_text_for_lists(_post(title="buy now", body="offer inside"))
        ) is None

    def test_post_text_falls_back_to_content_and_tolerates_none(self):
        assert s.post_text_for_lists({"title": None, "content": "legacy body"}) == "\nlegacy body"

    def test_edits_take_effect_without_restart(self, strings_file):
        strings_file.write_text("alpha\n", encoding="utf-8")
        assert s.never_upvote_string_in("x alpha x") == "alpha"
        strings_file.write_text("bravo-longer\n", encoding="utf-8")  # size changes too
        assert s.never_upvote_string_in("x alpha x") is None
        assert s.never_upvote_string_in("x bravo-longer x") == "bravo-longer"

    def test_same_file_for_two_lists_is_not_served_the_other_parse(self, strings_file):
        """The cache is keyed on (parser, path). Keyed on path alone, pointing
        a username list and the strings list at one file would hand one of
        them the other's parse."""
        strings_file.write_text("Buy Now # note\n", encoding="utf-8")
        assert s._load_username_list(strings_file) == {"buy now"}
        assert s._load_string_list(strings_file) == {"buy now # note"}


# ────────────────────────────────────────────────────────────────────
# _pending_actions — the queue-time decision
# ────────────────────────────────────────────────────────────────────


class TestQueueTime:
    def test_a_listed_phrase_suppresses_the_upvote(self, strings_file, make_judgement):
        strings_file.write_text("buy now\n", encoding="utf-8")
        j = make_judgement(vote_recommendation="upvote", score=s.UPVOTE_MIN_SCORE)
        actions = s._pending_actions(j, "please BUY NOW")
        assert not [a for a in actions if a["kind"] == "vote"]
        assert {"kind": "mark_scanned_post"} in actions, "only the vote is suppressed"

    def test_an_unlisted_post_is_still_upvoted(self, strings_file, make_judgement):
        strings_file.write_text("buy now\n", encoding="utf-8")
        j = make_judgement(vote_recommendation="upvote", score=s.UPVOTE_MIN_SCORE)
        votes = [a for a in s._pending_actions(j, "a careful write-up") if a["kind"] == "vote"]
        assert votes and votes[0]["value"] == 1

    def test_downvotes_are_not_affected(self, strings_file, make_judgement):
        """Upvotes only: a listed phrase must not stop a downvote either way."""
        strings_file.write_text("buy now\n", encoding="utf-8")
        j = make_judgement(vote_recommendation="downvote", score=2)
        votes = [a for a in s._pending_actions(j, "buy now buy now") if a["kind"] == "vote"]
        assert votes and votes[0]["value"] == -1

    def test_the_vote_action_never_carries_the_post_text(self, make_judgement):
        """A failed vote is persisted verbatim to memory. The text stays out
        of it; a replay re-reads the post instead."""
        j = make_judgement(vote_recommendation="upvote", score=s.UPVOTE_MIN_SCORE)
        vote = next(a for a in s._pending_actions(j, "body text") if a["kind"] == "vote")
        assert "body text" not in repr(vote)


# ────────────────────────────────────────────────────────────────────
# _apply_action — given text, and the replay path
# ────────────────────────────────────────────────────────────────────


class TestApplyTime:
    def test_given_text_blocks_the_upvote_without_a_fetch(self, strings_file, mock_client):
        strings_file.write_text("buy now\n", encoding="utf-8")
        ok = s._apply_action(mock_client, "p1", {"kind": "vote", "value": 1}, "buy now")
        assert ok is True, "resolved, not deferred — False would retry it 5 more times"
        mock_client.vote_post.assert_not_called()
        mock_client.get_post.assert_not_called()

    def test_replay_rereads_the_post_and_blocks(self, strings_file, mock_client):
        """Queued before the phrase was listed (or before the post was edited
        to contain it): the replay must judge the post as it stands now."""
        strings_file.write_text("buy now\n", encoding="utf-8")
        mock_client.get_post.return_value = _post(body="edited: Buy Now")
        ok = s._apply_action(mock_client, "p1", {"kind": "vote", "value": 1, "author": "bob"})
        assert ok is True
        mock_client.get_post.assert_called_once_with("p1")
        mock_client.vote_post.assert_not_called()

    def test_replay_of_a_clean_post_is_cast(self, strings_file, mock_client):
        strings_file.write_text("buy now\n", encoding="utf-8")
        mock_client.get_post.return_value = _post()
        assert s._apply_action(mock_client, "p1", {"kind": "vote", "value": 1}) is True
        mock_client.vote_post.assert_called_once_with("p1", 1)

    def test_replay_fails_closed_when_the_post_cannot_be_read(self, strings_file, mock_client):
        strings_file.write_text("buy now\n", encoding="utf-8")
        mock_client.get_post.side_effect = ColonyAPIError("bad gateway", 502)
        ok = s._apply_action(mock_client, "p1", {"kind": "vote", "value": 1})
        assert ok is False, "deferred: an upvote we could not check stays pending"
        mock_client.vote_post.assert_not_called()

    def test_no_list_means_no_extra_request(self, mock_client):
        assert s._apply_action(mock_client, "p1", {"kind": "vote", "value": 1}) is True
        mock_client.get_post.assert_not_called()
        mock_client.vote_post.assert_called_once_with("p1", 1)

    def test_downvote_replay_never_rereads(self, strings_file, mock_client):
        strings_file.write_text("buy now\n", encoding="utf-8")
        assert s._apply_action(mock_client, "p1", {"kind": "vote", "value": -1}) is True
        mock_client.get_post.assert_not_called()
        mock_client.vote_post.assert_called_once_with("p1", -1)

    def test_retry_pending_actions_goes_through_the_check(self, strings_file, mock_client):
        strings_file.write_text("buy now\n", encoding="utf-8")
        mock_client.get_post.return_value = _post(title="Buy now")
        memory = {"p1": {"score": 9, "_pending_actions": [{"kind": "vote", "value": 1}]}}
        s.retry_pending_actions(mock_client, memory)
        mock_client.vote_post.assert_not_called()
        assert "_pending_actions" not in memory["p1"], "a blocked vote is resolved, not retried"


# ────────────────────────────────────────────────────────────────────
# Wiring — the seams where a correct check can still never run
# ────────────────────────────────────────────────────────────────────


def _args(**over):
    base = dict(dry_run=False, no_vote=False, no_pii=False, model="m", confirm=False)
    base.update(over)
    return SimpleNamespace(**base)


def _upvote(monkeypatch):
    monkeypatch.setattr(
        s, "call_ollama",
        lambda model, messages: {
            "score": 9, "category": "GOOD", "vote_recommendation": "upvote", "language": "en",
        },
    )


class TestWiring:
    def test_scan_path_hands_the_post_text_to_the_check(
        self, strings_file, monkeypatch, mock_client,
    ):
        strings_file.write_text("buy now\n", encoding="utf-8")
        _upvote(monkeypatch)
        post_data = {"post": _post(body="Buy now!"), "comments": [], "colony_name": "general"}
        memory: dict = {}
        assert s._process_post(mock_client, post_data, _args(), memory, []) is True
        mock_client.vote_post.assert_not_called()
        mock_client.get_post.assert_not_called()  # text was passed, so no re-read

    def test_scan_path_still_upvotes_a_clean_post(self, strings_file, monkeypatch, mock_client):
        strings_file.write_text("buy now\n", encoding="utf-8")
        _upvote(monkeypatch)
        post_data = {"post": _post(), "comments": [], "colony_name": "general"}
        s._process_post(mock_client, post_data, _args(), {}, [])
        mock_client.vote_post.assert_called_once_with("p1", 1)

    def test_memory_does_not_store_the_post_body(self, strings_file, monkeypatch, mock_client):
        """The reason the text is passed alongside the judgement rather than
        stamped onto it: the judgement is what memory persists."""
        _upvote(monkeypatch)
        mock_client.vote_post.side_effect = ColonyAPIError("bad gateway", 502)  # force a pending vote
        body = "A distinctive body sentence that must not be persisted."
        post_data = {"post": _post(body=body), "comments": [], "colony_name": "general"}
        memory: dict = {}
        s._process_post(mock_client, post_data, _args(), memory, [])
        assert memory["p1"].get("_pending_actions"), "precondition: a vote was persisted"
        assert body not in repr(memory)

    def test_webhook_path_hands_the_post_text_to_the_check(
        self, strings_file, monkeypatch, mock_client,
    ):
        strings_file.write_text("buy now\n", encoding="utf-8")
        _upvote(monkeypatch)
        monkeypatch.setattr(
            s, "fetch_post_with_comments",
            lambda client, pid: {"post": _post(body="BUY NOW"), "comments": [],
                                 "colony_name": "general"},
        )
        worker = s.WebhookWorker(
            client=mock_client, own_username="sentinel-test", model="m",
            allow_vote=True, allow_lang=True, allow_pii=True, allow_mark_scanned=True,
        )
        worker._process("p1")
        mock_client.vote_post.assert_not_called()
        mock_client.get_post.assert_not_called()


class TestStartupAnnouncement:
    def test_the_strings_list_is_logged_with_its_absolute_path(self, strings_file, caplog):
        strings_file.write_text("buy now\n# note\nfree\n", encoding="utf-8")
        with caplog.at_level("INFO"):
            s.log_vote_exemptions()
        line = next(
            r.getMessage() for r in caplog.records if "never-upvote-strings" in r.getMessage()
        )
        assert "2 entries" in line and str(strings_file.resolve()) in line
