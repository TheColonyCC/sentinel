"""The remote inference backend (Nous Portal) and the backend switch itself.

Three groups, and the first is the one that matters most:

* **Selection.** Adding a second backend is only safe if the first one is
  untouched by default, and if choosing the second one also moves everything
  that hangs off it — the default model, the preflight, the GPU check, the
  banner. A half-switched run is worse than no switch: it reports the model it
  did not use, or ships an Ollama tag to a remote catalogue.
* **The call.** ``call_nous`` mirrors ``call_ollama``'s contract exactly —
  every failure returns ``None`` so the post is retried next run and no error
  path can raise out of the scan loop. Each status code that needs a different
  operator action is pinned separately, because "HTTP 402" in a log does not
  say "top up".
* **Secrets.** The key is read from the environment and must never reach a
  log. That is asserted against a transport error that *contains* the key, not
  against a well-behaved one — a test where nothing could have leaked proves
  nothing about the redactor.
"""
from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

import sentinel as s

MESSAGES = [{"role": "user", "content": "hi"}]
FAKE_KEY = "sk-nous-TESTKEY-must-never-be-logged"


@pytest.fixture
def nous(monkeypatch: pytest.MonkeyPatch):
    """Active backend = nous, with a key set. Undone after each test."""
    monkeypatch.setattr(s, "BACKEND", s.BACKEND_NOUS)
    monkeypatch.setenv(s.NOUS_API_KEY_ENV, FAKE_KEY)


def _chat_response(content: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _catalogue(rows: list[dict], status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"data": rows}
    return resp


# ─── Backend selection ───────────────────────────────────────────────────

class TestBackendSelection:
    def test_default_backend_is_ollama(self):
        """The whole point: an existing install is unaffected by this feature
        until it opts in."""
        assert s.BACKEND == s.BACKEND_OLLAMA
        assert s.BACKENDS == (s.BACKEND_OLLAMA, s.BACKEND_NOUS)

    def test_env_selects_the_backend(self, monkeypatch):
        import sentinel
        monkeypatch.setenv("SENTINEL_BACKEND", "nous")
        try:
            importlib.reload(sentinel)
            assert sentinel.BACKEND == "nous"
        finally:
            monkeypatch.delenv("SENTINEL_BACKEND", raising=False)
            importlib.reload(sentinel)
        assert sentinel.BACKEND == "ollama"

    def test_invalid_env_warns_and_falls_back(self, monkeypatch, capsys):
        # Import must not fail on a typo'd env var — same contract as _env_int.
        assert s._env_choice("SENTINEL_TEST_BACKEND", "ollama", s.BACKENDS) == "ollama"
        monkeypatch.setenv("SENTINEL_TEST_BACKEND", "nuos")
        assert s._env_choice("SENTINEL_TEST_BACKEND", "ollama", s.BACKENDS) == "ollama"
        assert "ignoring invalid" in capsys.readouterr().err

    def test_env_choice_is_case_and_space_tolerant(self, monkeypatch):
        monkeypatch.setenv("SENTINEL_TEST_BACKEND", "  NOUS ")
        assert s._env_choice("SENTINEL_TEST_BACKEND", "ollama", s.BACKENDS) == "nous"

    def test_apply_backend_none_leaves_it_alone(self, monkeypatch):
        monkeypatch.setattr(s, "BACKEND", s.BACKEND_NOUS)
        assert s._apply_backend(None) == s.BACKEND_NOUS

    def test_apply_backend_switches(self, monkeypatch):
        monkeypatch.setattr(s, "BACKEND", s.BACKEND_OLLAMA)
        assert s._apply_backend("NOUS") == s.BACKEND_NOUS
        assert s.BACKEND == s.BACKEND_NOUS

    def test_apply_backend_rejects_unknown(self, monkeypatch):
        monkeypatch.setattr(s, "BACKEND", s.BACKEND_OLLAMA)
        with pytest.raises(SystemExit) as e:
            s._apply_backend("nuos")
        assert e.value.code == 2
        # ...and left the process on the backend it already had.
        assert s.BACKEND == s.BACKEND_OLLAMA

    @pytest.mark.parametrize("argv,expected", [
        (["scan", "--backend", "nous"], "nous"),
        (["scan", "--backend=nous"], "nous"),
        (["scan", "--limit", "5"], None),
        (["scan", "--backend"], None),          # trailing flag, no value
    ])
    def test_argv_backend(self, argv, expected):
        assert s._argv_backend(argv) == expected

    def test_inference_not_required_for_webhook_register(self):
        assert s._inference_required(["scan"]) is True
        assert s._inference_required(["webhook"]) is True
        assert s._inference_required(["webhook-register"]) is False


# ─── Default model resolution ────────────────────────────────────────────

class TestDefaultModel:
    def test_defaults_differ_per_backend(self):
        """The defect this guards: --backend nous with no --model shipping
        'qwen3.5:9b-q4_k_m' to a catalogue that has never heard of it."""
        assert s.default_model_for(s.BACKEND_OLLAMA) == s.DEFAULT_MODEL
        assert s.default_model_for(s.BACKEND_NOUS) == s.DEFAULT_NOUS_MODEL
        assert s.DEFAULT_NOUS_MODEL != s.DEFAULT_MODEL

    def test_model_flag_default_is_none_not_baked_in(self):
        """Control for the test above: an argparse default of DEFAULT_MODEL is
        fixed before --backend is read, so resolution could never fire."""
        parser = s.build_parser()
        for argv in (["scan"], ["webhook"]):
            assert parser.parse_args(argv).model is None, argv

    def test_scan_resolves_the_nous_default(self, monkeypatch, caplog, nous):
        monkeypatch.setattr(s, "ensure_model_available", lambda m: None)
        monkeypatch.setattr(s, "ensure_gpu_available", lambda m: None)
        client = MagicMock()
        client.iter_posts.return_value = iter([])
        monkeypatch.setattr(s, "get_or_register_client", lambda u: (client, {}))
        monkeypatch.setattr(s, "load_memory", lambda: {})
        monkeypatch.setattr(s, "prune_memory", lambda m, **k: m)
        args = SimpleNamespace(
            dry_run=True, no_vote=True, no_pii=True, model=None, backend="nous",
            username=None, force=False, post_id=None, sort="new", limit=1,
            days=7, confirm=False, allow_cpu=False, include_scanned=False,
        )
        with caplog.at_level(logging.INFO, logger="sentinel"):
            s.cmd_scan(args)
        assert args.model == s.DEFAULT_NOUS_MODEL
        assert s.DEFAULT_MODEL not in caplog.text


# ─── Banner ──────────────────────────────────────────────────────────────

class TestBanner:
    def test_names_nous_and_its_endpoint(self, caplog, nous):
        with caplog.at_level(logging.INFO, logger="sentinel"):
            s.log_model_in_use("nousresearch/hermes-4-70b")
        assert "nousresearch/hermes-4-70b" in caplog.text
        assert s.NOUS_API_BASE in caplog.text
        # A banner that still named the local daemon would misattribute every
        # judgement in the log to a machine that did not produce it.
        assert s.OLLAMA_HOST not in caplog.text


# ─── Dispatch ────────────────────────────────────────────────────────────

class TestCallModelDispatch:
    def test_routes_to_nous_when_selected(self, monkeypatch, nous):
        monkeypatch.setattr(s, "call_nous", lambda m, msgs: {"via": "nous"})
        monkeypatch.setattr(s, "call_ollama", lambda m, msgs: {"via": "ollama"})
        assert s.call_model("m", MESSAGES) == {"via": "nous"}

    def test_routes_to_ollama_by_default(self, monkeypatch):
        """Control for the test above — the dispatcher must move both ways."""
        monkeypatch.setattr(s, "BACKEND", s.BACKEND_OLLAMA)
        monkeypatch.setattr(s, "call_nous", lambda m, msgs: {"via": "nous"})
        monkeypatch.setattr(s, "call_ollama", lambda m, msgs: {"via": "ollama"})
        assert s.call_model("m", MESSAGES) == {"via": "ollama"}


# ─── call_nous: request shape ────────────────────────────────────────────

class TestRequestShape:
    def test_url_headers_and_payload(self, monkeypatch, nous):
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured.update(url=url, json=json, headers=headers, timeout=timeout)
            return _chat_response('{"category": "OK"}')

        monkeypatch.setattr(s.requests, "post", fake_post)
        s.call_nous("nousresearch/hermes-4-70b", MESSAGES)

        assert captured["url"] == f"{s.NOUS_API_BASE}/chat/completions"
        assert captured["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
        assert captured["json"]["model"] == "nousresearch/hermes-4-70b"
        assert captured["json"]["stream"] is False
        # The OpenAI-shaped counterpart of Ollama's format:"json".
        assert captured["json"]["response_format"] == {"type": "json_object"}
        assert captured["json"]["temperature"] == s.NOUS_OPTIONS["temperature"]

    def test_no_max_tokens_cap(self, monkeypatch, nous):
        """Same regression as num_predict on the local backend: a token cap is
        spent on a reasoning model's thinking and starves the JSON answer."""
        captured = {}
        monkeypatch.setattr(
            s.requests, "post",
            lambda url, json=None, headers=None, timeout=None: (
                captured.update(json=json) or _chat_response('{"category": "OK"}')),
        )
        s.call_nous("m", MESSAGES)
        assert "max_tokens" not in captured["json"]
        assert s.NOUS_OPTIONS.get("max_tokens") is None

    def test_split_connect_read_timeout(self, monkeypatch, nous):
        captured = {}
        monkeypatch.setattr(
            s.requests, "post",
            lambda url, json=None, headers=None, timeout=None: (
                captured.update(timeout=timeout) or _chat_response('{"a": 1}')),
        )
        s.call_nous("m", MESSAGES)
        assert captured["timeout"] == (s.NOUS_CONNECT_TIMEOUT, s.NOUS_TIMEOUT)

    def test_connect_timeout_covers_a_wan_handshake(self):
        # Loopback's 5s is too tight for a TLS handshake over the internet,
        # and the slow-warn must still fire before the hard timeout.
        assert s.NOUS_CONNECT_TIMEOUT >= s.OLLAMA_CONNECT_TIMEOUT
        assert s.NOUS_SLOW_WARN_SECONDS < s.NOUS_TIMEOUT


# ─── call_nous: outcomes ─────────────────────────────────────────────────

class TestCallOutcomes:
    def test_happy_path_parses_choices_content(self, monkeypatch, nous):
        monkeypatch.setattr(
            s.requests, "post",
            lambda *a, **k: _chat_response('{"category": "JUNK", "score": 0}'))
        assert s.call_nous("m", MESSAGES) == {"category": "JUNK", "score": 0}

    def test_missing_key_returns_none_without_calling_out(self, monkeypatch, caplog):
        monkeypatch.setattr(s, "BACKEND", s.BACKEND_NOUS)
        monkeypatch.delenv(s.NOUS_API_KEY_ENV, raising=False)

        def must_not_be_called(*a, **k):  # pragma: no cover - the assertion
            raise AssertionError("no request may be made without a key")

        monkeypatch.setattr(s.requests, "post", must_not_be_called)
        with caplog.at_level(logging.ERROR, logger="sentinel"):
            assert s.call_nous("m", MESSAGES) is None
        assert s.NOUS_API_KEY_ENV in caplog.text

    @pytest.mark.parametrize("status", [401, 403, 402, 429, 500, 503])
    def test_error_statuses_return_none(self, monkeypatch, status, nous):
        monkeypatch.setattr(
            s.requests, "post",
            lambda *a, **k: _chat_response("", status=status))
        assert s.call_nous("m", MESSAGES) is None

    def test_402_names_the_remedy(self, monkeypatch, caplog, nous):
        monkeypatch.setattr(
            s.requests, "post", lambda *a, **k: _chat_response("", status=402))
        with caplog.at_level(logging.ERROR, logger="sentinel"):
            s.call_nous("m", MESSAGES)
        assert "credit" in caplog.text.lower()

    def test_401_names_the_env_var(self, monkeypatch, caplog, nous):
        monkeypatch.setattr(
            s.requests, "post", lambda *a, **k: _chat_response("", status=401))
        with caplog.at_level(logging.ERROR, logger="sentinel"):
            s.call_nous("m", MESSAGES)
        assert s.NOUS_API_KEY_ENV in caplog.text

    def test_timeout_returns_none_not_raise(self, monkeypatch, nous):
        def boom(*a, **k):
            raise requests.exceptions.Timeout("read timed out")
        monkeypatch.setattr(s.requests, "post", boom)
        assert s.call_nous("m", MESSAGES) is None

    def test_connection_error_returns_none_not_raise(self, monkeypatch, nous):
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError("dns failure")
        monkeypatch.setattr(s.requests, "post", boom)
        assert s.call_nous("m", MESSAGES) is None

    def test_unparseable_body_returns_none(self, monkeypatch, nous):
        monkeypatch.setattr(
            s.requests, "post", lambda *a, **k: _chat_response("not json at all"))
        assert s.call_nous("m", MESSAGES) is None

    def test_unexpected_envelope_returns_none(self, monkeypatch, nous):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"unexpected": "shape"}
        monkeypatch.setattr(s.requests, "post", lambda *a, **k: resp)
        assert s.call_nous("m", MESSAGES) is None

    def test_slow_call_warns_but_still_returns(self, monkeypatch, caplog, nous):
        ticks = iter([100.0, 100.0 + s.NOUS_SLOW_WARN_SECONDS + 5])
        monkeypatch.setattr(s.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(
            s.requests, "post", lambda *a, **k: _chat_response('{"category": "OK"}'))
        with caplog.at_level(logging.WARNING, logger="sentinel"):
            out = s.call_nous("m", MESSAGES)
        assert out == {"category": "OK"}
        assert any("took" in r.getMessage() for r in caplog.records)


# ─── Secret handling ─────────────────────────────────────────────────────

class TestKeyIsNeverLogged:
    def test_transport_error_carrying_the_key_is_redacted(
        self, monkeypatch, caplog, nous,
    ):
        """The control is the third assertion. A test that only checked the
        key's absence would pass on a run that logged nothing at all — and so
        would prove nothing about the redactor."""
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError(
                f"failed sending header Authorization: Bearer {FAKE_KEY}")

        monkeypatch.setattr(s.requests, "post", boom)
        with caplog.at_level(logging.ERROR, logger="sentinel"):
            assert s.call_nous("m", MESSAGES) is None
        assert caplog.text.strip(), "control: the failure must log something"
        assert FAKE_KEY not in caplog.text
        assert "<redacted>" in caplog.text

    def test_redact_is_a_noop_without_a_key(self, monkeypatch):
        monkeypatch.delenv(s.NOUS_API_KEY_ENV, raising=False)
        assert s._redact("plain text") == "plain text"

    def test_redact_replaces_every_occurrence(self, monkeypatch):
        monkeypatch.setenv(s.NOUS_API_KEY_ENV, FAKE_KEY)
        out = s._redact(f"{FAKE_KEY} and again {FAKE_KEY}")
        assert FAKE_KEY not in out
        assert out.count("<redacted>") == 2


# ─── JSON recovery ───────────────────────────────────────────────────────

class TestParseJsonAnswer:
    def test_plain_object(self):
        assert s._parse_json_answer('{"a": 1}') == {"a": 1}

    def test_recovers_from_a_reasoning_preamble(self, caplog):
        raw = 'Let me think about this.\n{"category": "OK", "score": 7}'
        with caplog.at_level(logging.WARNING, logger="sentinel"):
            assert s._parse_json_answer(raw) == {"category": "OK", "score": 7}
        # Recovery must be visible: a silent fallback would hide a backend
        # that has quietly started prefixing every answer.
        assert "before the JSON verdict" in caplog.text

    def test_no_object_raises(self):
        with pytest.raises(ValueError):
            s._parse_json_answer("no braces here")

    def test_non_object_json_raises(self):
        # json.loads("[1,2]") succeeds but the pipeline needs a dict; a list
        # would blow up later, at result["post_id"] = ...
        for text in ("[1, 2]", "null", "42"):
            with pytest.raises(ValueError):
                s._parse_json_answer(text)


# ─── Preflight ───────────────────────────────────────────────────────────

CATALOGUE = [
    {"id": "nousresearch/hermes-4-70b",
     "aliases": ["nousresearch/hermes-4-70b", "Hermes-4-70B"]},
    {"id": "nousresearch/hermes-4-405b", "aliases": ["Hermes-4-405B"]},
    {"id": "tencent/hy4-preview", "aliases": []},
]


class TestNousPreflight:
    def test_exact_id_ok(self, monkeypatch, nous):
        monkeypatch.setattr(s.requests, "get", lambda *a, **k: _catalogue(CATALOGUE))
        s.ensure_model_available("nousresearch/hermes-4-70b")  # no raise

    def test_alias_and_casing_ok(self, monkeypatch, nous):
        # The portal publishes both forms; a preflight that took only the
        # canonical id would reject a name the catalogue itself hands out.
        monkeypatch.setattr(s.requests, "get", lambda *a, **k: _catalogue(CATALOGUE))
        s.ensure_model_available("Hermes-4-70B")
        s.ensure_model_available("hermes-4-405b".upper())

    def test_unknown_model_exits(self, monkeypatch, caplog, nous):
        monkeypatch.setattr(s.requests, "get", lambda *a, **k: _catalogue(CATALOGUE))
        with caplog.at_level(logging.ERROR, logger="sentinel"):
            with pytest.raises(SystemExit) as e:
                s.ensure_model_available("qwen3.5:9b-q4_k_m")
        assert e.value.code == 1
        # The hint names the Nous-published models, not all 350.
        assert "nousresearch/hermes-4-70b" in caplog.text
        assert "tencent/hy4-preview" not in caplog.text

    def test_catalogue_unreachable_exits(self, monkeypatch, nous):
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError("down")
        monkeypatch.setattr(s.requests, "get", boom)
        with pytest.raises(SystemExit) as e:
            s.ensure_model_available("nousresearch/hermes-4-70b")
        assert e.value.code == 1

    def test_missing_key_exits_before_the_catalogue_call(self, monkeypatch, caplog):
        """The catalogue is public, so a key-less host would otherwise pass
        the model check and only discover the real problem one error per post
        later."""
        monkeypatch.setattr(s, "BACKEND", s.BACKEND_NOUS)
        monkeypatch.delenv(s.NOUS_API_KEY_ENV, raising=False)

        def must_not_be_called(*a, **k):  # pragma: no cover - the assertion
            raise AssertionError("catalogue must not be queried without a key")

        monkeypatch.setattr(s.requests, "get", must_not_be_called)
        with caplog.at_level(logging.ERROR, logger="sentinel"):
            with pytest.raises(SystemExit) as e:
                s.ensure_model_available("nousresearch/hermes-4-70b")
        assert e.value.code == 1
        assert s.NOUS_API_KEY_ENV in caplog.text

    def test_ollama_preflight_still_reached_by_default(self, monkeypatch):
        """Control: the dispatcher must not have stolen the local path."""
        called = {}
        monkeypatch.setattr(s, "BACKEND", s.BACKEND_OLLAMA)
        monkeypatch.setattr(s, "_ensure_ollama_model", lambda m: called.setdefault("ollama", m))
        monkeypatch.setattr(s, "_ensure_nous_model", lambda m: called.setdefault("nous", m))
        s.ensure_model_available("qwen3.5:9b-q4_k_m")
        assert called == {"ollama": "qwen3.5:9b-q4_k_m"}


class TestGpuCheckIsLocalOnly:
    def test_skipped_on_the_remote_backend(self, monkeypatch, caplog, nous):
        """/api/ps would answer about a daemon nothing is going to call, and
        whether THIS box has a GPU is not a fact about a remote run."""
        def must_not_be_called(*a, **k):  # pragma: no cover - the assertion
            raise AssertionError("no Ollama call may be made on the nous backend")

        monkeypatch.setattr(s.requests, "get", must_not_be_called)
        monkeypatch.setattr(s.requests, "post", must_not_be_called)
        with caplog.at_level(logging.INFO, logger="sentinel"):
            s.ensure_gpu_available("nousresearch/hermes-4-70b")  # no raise
        assert "skipped" in caplog.text
