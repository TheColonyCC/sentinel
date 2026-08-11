"""First-run agent registration: the two-step register_begin / register_confirm flow.

WHY THIS FILE EXISTS
--------------------
``get_or_register_client`` was the only untested function in sentinel's startup
path — every other test monkeypatches it out (see ``test_scan_resilience.py``).
That gap is what let a real break sit on master: the code called the one-shot
``ColonyClient.register``, which colony-sdk has since removed, so first-run
registration raised ``AttributeError`` while 140 tests stayed green. A path no
test exercises is a path where the SDK can change underneath you silently.

The ordering assertions below are the point. ``register_begin`` mints a
*pending* account; ``register_confirm`` activates it by proving the caller still
holds the issued key. Confirming before the key is durably stored would defeat
the gate entirely — so "saved to disk, and read back, strictly before confirm"
is a behavioural requirement, not an implementation detail, and it is asserted
directly rather than inferred from the happy path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sentinel as s

BEGUN = {
    "api_key": "col_test_key_abc123",
    "claim_token": "claim-tok-1",
    "expires_at": "2026-08-11T12:00:00Z",
}


@pytest.fixture
def sdk(monkeypatch):
    """Patch both halves of the SDK registration flow, recording call order.

    Order matters here and is otherwise invisible, so every call appends to a
    shared list that the tests assert against.
    """
    calls: list[tuple] = []

    def begin(**kwargs):
        calls.append(("begin", kwargs))
        # The key must not be on disk before begin returns it.
        calls.append(("disk_at_begin", Path(s.CONFIG_FILE).exists()))
        return dict(BEGUN)

    def confirm(claim_token, fingerprint):
        # Snapshot what was durably stored at the moment of confirm.
        on_disk = (
            json.loads(Path(s.CONFIG_FILE).read_text()).get("api_key")
            if Path(s.CONFIG_FILE).exists()
            else None
        )
        calls.append(("confirm", claim_token, fingerprint, on_disk))
        return {"status": "active", "id": "agent-uuid", "username": "sentinel-test"}

    monkeypatch.setattr(s.ColonyClient, "register_begin", staticmethod(begin))
    monkeypatch.setattr(s.ColonyClient, "register_confirm", staticmethod(confirm))
    return calls


class TestRegisterAgent:
    def test_uses_two_step_flow_and_persists_before_confirming(self, sdk):
        config: dict = {}
        api_key = s.register_agent("sentinel-test", config)

        assert api_key == BEGUN["api_key"]
        assert config["api_key"] == BEGUN["api_key"]
        assert config["username"] == "sentinel-test"

        kinds = [c[0] for c in sdk]
        assert kinds == ["begin", "disk_at_begin", "confirm"], (
            "expected exactly one begin then one confirm"
        )

        # The key was NOT on disk before begin, and WAS on disk at confirm.
        # Both directions are asserted: a test that only checks the second
        # would pass against an implementation that wrote the key at import.
        assert sdk[1][1] is False
        confirm_call = sdk[2]
        assert confirm_call[3] == BEGUN["api_key"], (
            "register_confirm ran before the key reached durable storage"
        )

    def test_confirms_with_the_last_six_chars_of_the_key(self, sdk):
        s.register_agent("sentinel-test", {})
        _, claim_token, fingerprint, _ = sdk[2]
        assert claim_token == BEGUN["claim_token"]
        assert fingerprint == BEGUN["api_key"][-6:]
        # Guard the fingerprint against being the whole key by accident.
        assert len(fingerprint) == 6
        assert fingerprint != BEGUN["api_key"]

    def test_passes_agent_identity_to_begin(self, sdk):
        s.register_agent("sentinel-test", {})
        kwargs = sdk[0][1]
        assert kwargs["username"] == "sentinel-test"
        assert kwargs["display_name"] == s.DEFAULT_DISPLAY_NAME
        assert kwargs["bio"] == s.DEFAULT_BIO
        assert kwargs["capabilities"] == s.AGENT_CAPABILITIES

    def test_begin_failure_exits_without_writing_config(self, monkeypatch):
        def boom(**kwargs):
            raise s.ColonyAPIError("username taken", 409)

        monkeypatch.setattr(s.ColonyClient, "register_begin", staticmethod(boom))
        with pytest.raises(SystemExit):
            s.register_agent("sentinel-test", {})
        assert not Path(s.CONFIG_FILE).exists()

    def test_missing_claim_token_aborts_before_confirm(self, monkeypatch):
        """A 200 that omits claim_token must not be treated as success.

        The key alone is useless while the account is pending, so silently
        continuing would hand back an api_key that cannot post.
        """
        monkeypatch.setattr(
            s.ColonyClient, "register_begin",
            staticmethod(lambda **k: {"api_key": "col_x"}),
        )
        confirmed = []
        monkeypatch.setattr(
            s.ColonyClient, "register_confirm",
            staticmethod(lambda *a: confirmed.append(a)),
        )
        with pytest.raises(SystemExit):
            s.register_agent("sentinel-test", {})
        assert confirmed == []

    def test_already_active_is_treated_as_success(self, monkeypatch):
        """REGISTER_ALREADY_ACTIVE is the SDK's documented idempotent guard.

        It means a previous attempt confirmed and then died before recording
        it. The account works, so exiting would strand a usable agent.
        """
        monkeypatch.setattr(
            s.ColonyClient, "register_begin", staticmethod(lambda **k: dict(BEGUN))
        )

        def already(*args):
            raise s.ColonyAPIError(
                "already active", 409, code="REGISTER_ALREADY_ACTIVE"
            )

        monkeypatch.setattr(s.ColonyClient, "register_confirm", staticmethod(already))
        assert s.register_agent("sentinel-test", {}) == BEGUN["api_key"]

    def test_other_confirm_failure_exits(self, monkeypatch):
        """Control for the test above: a *different* API error must still exit.

        Without this, an over-broad except that swallowed every ColonyAPIError
        would pass the already-active test and look correct.
        """
        monkeypatch.setattr(
            s.ColonyClient, "register_begin", staticmethod(lambda **k: dict(BEGUN))
        )

        def expired(*args):
            raise s.ColonyAPIError(
                "claim expired", 410, code="REGISTER_CLAIM_EXPIRED"
            )

        monkeypatch.setattr(s.ColonyClient, "register_confirm", staticmethod(expired))
        with pytest.raises(SystemExit):
            s.register_agent("sentinel-test", {})

    def test_unreadable_persisted_key_aborts_before_confirm(self, monkeypatch):
        """If the key doesn't survive the round-trip to disk, don't activate.

        Leaving the account pending releases the username after the claim
        window, so the retry is clean instead of colliding with a half-made
        account whose key nobody holds.
        """
        monkeypatch.setattr(
            s.ColonyClient, "register_begin", staticmethod(lambda **k: dict(BEGUN))
        )
        confirmed = []
        monkeypatch.setattr(
            s.ColonyClient, "register_confirm",
            staticmethod(lambda *a: confirmed.append(a)),
        )
        # Simulate a config that reads back empty (disk full, wrong cwd, etc.)
        monkeypatch.setattr(s, "load_config", lambda: {})
        with pytest.raises(SystemExit):
            s.register_agent("sentinel-test", {})
        assert confirmed == []


class TestGetOrRegisterClient:
    def test_existing_key_skips_registration(self, monkeypatch):
        """The common path: a saved key means no registration calls at all."""
        s.save_config({"api_key": "col_saved", "username": "already-there"})

        def fail(**kwargs):
            raise AssertionError("register_begin must not be called")

        monkeypatch.setattr(s.ColonyClient, "register_begin", staticmethod(fail))
        built: list = []
        monkeypatch.setattr(
            s, "ColonyClient",
            type("C", (), {
                "__init__": lambda self, api_key: built.append(api_key),
                "register_begin": staticmethod(fail),
            }),
        )
        _, config = s.get_or_register_client("already-there")
        assert built == ["col_saved"]
        assert config["username"] == "already-there"

    def test_no_key_triggers_registration(self, monkeypatch):
        called: list = []
        monkeypatch.setattr(
            s, "register_agent",
            lambda username, config: (called.append(username), "col_new")[1],
        )
        monkeypatch.setattr(
            s, "ColonyClient",
            type("C", (), {"__init__": lambda self, api_key: None}),
        )
        s.get_or_register_client("brand-new")
        assert called == ["brand-new"]
