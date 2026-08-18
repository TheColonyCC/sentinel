"""The scan refuses to run when the model would answer from CPU.

The failure this prevents is not a crash. Ollama falls back to CPU silently
when a model does not fit in VRAM, and the run then "works" at roughly a
hundredth of the speed: minutes per post instead of seconds, most of them
hitting OLLAMA_TIMEOUT. The only pre-existing signal was the per-call
OLLAMA_SLOW_WARN_SECONDS warning, which fires after a minute of grinding,
once per post, for the whole run.

**Why the gate is Ollama's /api/ps rather than nvidia-smi.** They answer
different questions. nvidia-smi answers "does this box have an NVIDIA GPU the
driver can see"; the question that matters is "did Ollama put this model on a
GPU". Those come apart in three ways that all occur in practice: OLLAMA_HOST
can point at another machine, so the local GPU is irrelevant; a present GPU
can still be full, sending the model to CPU; and Ollama runs on ROCm and Metal,
where nvidia-smi is absent on a perfectly good machine. A gate on nvidia-smi
would pass the first two and fail the third. It is kept only to enrich the
error message.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import sentinel as s


def _ps(models):
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = {"models": models}
    return r


def _loaded(name, size, vram):
    return {"name": name, "size": size, "model": name, "size_vram": vram}


@pytest.fixture(autouse=True)
def _no_preload(monkeypatch):
    """The preload POST is a side effect, not the thing under test."""
    monkeypatch.setattr(s.requests, "post", lambda *a, **k: MagicMock())
    monkeypatch.setattr(s, "REQUIRE_GPU", True)


class TestOffloadFraction:
    def test_fully_on_gpu(self, monkeypatch):
        monkeypatch.setattr(s.requests, "get",
                            lambda *a, **k: _ps([_loaded("m:27b", 100, 100)]))
        assert s.gpu_offload_fraction("m:27b") == 1.0

    def test_fully_on_cpu(self, monkeypatch):
        monkeypatch.setattr(s.requests, "get",
                            lambda *a, **k: _ps([_loaded("m:27b", 100, 0)]))
        assert s.gpu_offload_fraction("m:27b") == 0.0

    def test_partial_offload(self, monkeypatch):
        monkeypatch.setattr(s.requests, "get",
                            lambda *a, **k: _ps([_loaded("m:27b", 100, 40)]))
        assert s.gpu_offload_fraction("m:27b") == pytest.approx(0.4)

    def test_case_and_latest_are_matched_like_the_model_preflight(self, monkeypatch):
        """Ollama lowercases tags on pull and a bare name means :latest — the
        same normalisation ensure_model_available already does. Getting this
        wrong reports 'unknown' forever on a perfectly healthy box."""
        monkeypatch.setattr(s.requests, "get",
                            lambda *a, **k: _ps([_loaded("m:latest", 10, 10)]))
        assert s.gpu_offload_fraction("M") == 1.0

    @pytest.mark.parametrize("models,why", [
        ([], "model not loaded"),
        ([_loaded("other:7b", 10, 10)], "a different model is loaded"),
        ([{"name": "m:27b", "size": 10}], "no size_vram field (older Ollama)"),
        ([{"name": "m:27b", "size": 0, "size_vram": 0}], "zero size, undivisible"),
    ])
    def test_unknown_is_none_not_zero(self, monkeypatch, models, why):
        """None and 0.0 must stay distinct: 0.0 aborts the run, None does not.
        Collapsing them would brick a scan on an older Ollama that simply does
        not report the field."""
        monkeypatch.setattr(s.requests, "get", lambda *a, **k: _ps(models))
        assert s.gpu_offload_fraction("m:27b") is None, why

    def test_unreachable_daemon_is_unknown(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("connection refused")
        monkeypatch.setattr(s.requests, "get", boom)
        assert s.gpu_offload_fraction("m:27b") is None


class TestTheGate:
    def test_cpu_only_aborts(self, monkeypatch):
        monkeypatch.setattr(s, "gpu_offload_fraction", lambda m: 0.0)
        monkeypatch.setattr(s, "_local_gpu_hint", lambda: "(hint)")
        with pytest.raises(SystemExit) as e:
            s.ensure_gpu_available("m:27b")
        assert e.value.code == 1

    def test_full_gpu_proceeds(self, monkeypatch):
        monkeypatch.setattr(s, "gpu_offload_fraction", lambda m: 1.0)
        s.ensure_gpu_available("m:27b")  # no raise

    def test_partial_offload_warns_but_proceeds(self, monkeypatch, caplog):
        """Partial offload is legal and sometimes fine. Aborting on it would
        make the gate refuse setups that work."""
        monkeypatch.setattr(s, "gpu_offload_fraction", lambda m: 0.5)
        with caplog.at_level("WARNING"):
            s.ensure_gpu_available("m:27b")
        assert any("in VRAM" in r.message for r in caplog.records)

    def test_unknown_proceeds_with_a_warning(self, monkeypatch, caplog):
        """An older Ollama, or a model dropped between the preload and the
        read, must not be fatal — 'we cannot tell' is not 'it is on CPU'."""
        monkeypatch.setattr(s, "gpu_offload_fraction", lambda m: None)
        with caplog.at_level("WARNING"):
            s.ensure_gpu_available("m:27b")
        assert any("Could not determine" in r.message for r in caplog.records)

    def test_require_gpu_off_skips_entirely(self, monkeypatch):
        monkeypatch.setattr(s, "REQUIRE_GPU", False)
        called = []
        monkeypatch.setattr(s, "gpu_offload_fraction",
                            lambda m: called.append(m) or 0.0)
        s.ensure_gpu_available("m:27b")  # would SystemExit if it consulted
        assert called == [], "the escape hatch still queried and could abort"


class TestScanHonoursTheFlag:
    def _args(self, **over):
        base = dict(
            dry_run=True, no_vote=True, no_pii=True, model="m:27b", username=None,
            force=False, post_id=None, sort="new", limit=1, days=7, confirm=False,
            include_scanned=False, allow_cpu=False,
        )
        base.update(over)
        return SimpleNamespace(**base)

    def test_scan_calls_the_gate(self, monkeypatch):
        monkeypatch.setattr(s, "ensure_model_available", lambda m: None)
        seen = []
        monkeypatch.setattr(s, "ensure_gpu_available", lambda m: seen.append(m))
        monkeypatch.setattr(s, "get_or_register_client",
                            lambda u: (_ for _ in ()).throw(SystemExit(0)))
        with pytest.raises(SystemExit):
            s.cmd_scan(self._args())
        assert seen == ["m:27b"]

    def test_allow_cpu_skips_the_gate(self, monkeypatch):
        monkeypatch.setattr(s, "ensure_model_available", lambda m: None)
        seen = []
        monkeypatch.setattr(s, "ensure_gpu_available", lambda m: seen.append(m))
        monkeypatch.setattr(s, "get_or_register_client",
                            lambda u: (_ for _ in ()).throw(SystemExit(0)))
        with pytest.raises(SystemExit):
            s.cmd_scan(self._args(allow_cpu=True))
        assert seen == [], "--allow-cpu did not bypass the GPU gate"
