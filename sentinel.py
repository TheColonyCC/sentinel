#!/usr/bin/env python3
"""Sentinel — automated content moderation agent for The Colony.

Uses a local LLM (via Ollama) to score posts on quality, then:
  - Casts votes (upvote good, downvote spam)
  - Marks JUNK posts (sentinel/admin role required)
  - Tags the primary language

Two modes:
  scan      One-shot pass over recent posts (cron-friendly).
  webhook   Long-running HTTP server, analyzes posts as they arrive.

All Colony API calls go through ``colony-sdk``, which handles auth, token
refresh, typed errors, and configurable retries on 429/502/503/504.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import requests  # only used for the local Ollama call

from colony_sdk import (
    ColonyAPIError,
    ColonyClient,
    ColonyNotFoundError,
    verify_webhook,
)


def _env_str(name: str, default: str) -> str:
    """Environment override for a string setting (empty/unset → default)."""
    val = os.environ.get(name)
    return val if val else default


def _env_int(name: str, default: int) -> int:
    """Environment override for an int setting. A non-integer value is
    ignored (warn to stderr + fall back) rather than crashing at import —
    these are read before logging is configured."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(
            f"[sentinel] ignoring invalid {name}={raw!r} (expected an integer); "
            f"using {default}", file=sys.stderr,
        )
        return default


# ─── Config ─────────────────────────────────────────────────────────────
# Most settings below can be overridden via environment variables (handy on
# a host where you don't want to edit code to switch models or tune timeouts).
LOCK_FILE = Path("sentinel.lock")
# OLLAMA_HOST is the same env var the `ollama` CLI honours.
OLLAMA_HOST = _env_str("OLLAMA_HOST", "http://localhost:11434")
# SENTINEL_MODEL sets the default model (still overridable per-run via --model).
DEFAULT_MODEL = _env_str("SENTINEL_MODEL", "qwen3.5:9b-q4_k_m")
DEFAULT_LIMIT = 20
MAX_COMMENTS = 6
MEMORY_FILE = Path("colony_analyzed.json")
CONFIG_FILE = Path("colony_config.json")
DEFAULT_DAYS = 7
# A healthy post scan finishes in seconds. If a single Ollama generation
# runs longer than this, the model has almost certainly wedged (runaway
# generation / stuck decode) rather than doing real work — cut it off and
# let the post retry next run instead of blocking the whole scan. Was 600
# (10 min), which is what produced the "scan hangs for 10 minutes then
# times out" reports.
OLLAMA_TIMEOUT = _env_int("OLLAMA_TIMEOUT", 180)
# Connect phase is separate + short: if the Ollama daemon is down, fail in
# seconds rather than burning the full read budget waiting to connect.
OLLAMA_CONNECT_TIMEOUT = _env_int("OLLAMA_CONNECT_TIMEOUT", 5)
# Healthy generations are far quicker than this; crossing it (without yet
# timing out) is an early "host is degrading" signal worth a log line.
OLLAMA_SLOW_WARN_SECONDS = _env_int("OLLAMA_SLOW_WARN_SECONDS", 60)
MEMORY_MAX_AGE_DAYS = 90

# Minimum LLM score (1-10) required to actually cast an upvote. The model
# tends to recommend "upvote" too readily for merely-OK posts, so we gate
# upvotes on a hard threshold: only truly strong posts get boosted. An
# "upvote" recommendation below this floor is downgraded to no vote.
# Downvotes are not gated — spam should be suppressed promptly.
UPVOTE_MIN_SCORE = 8

# Webhook mode: process events in a background worker so the HTTP response
# returns immediately. Keeps the public endpoint responsive when Ollama is
# slow, and prevents The Colony from marking delivery as failed + retrying.
WEBHOOK_QUEUE_SIZE = 100
# Prune memory every N processed posts in webhook mode (scan mode prunes on
# every run).
WEBHOOK_PRUNE_EVERY = 50
# Checkpoint scan-mode memory to disk every N newly-analyzed posts (and again
# in a finally). A local LLM scan can run for minutes; without checkpoints a
# kill / reboot / OOM mid-run discards every analysis since the last run, so
# the next run re-burns the model over all of them. The write is atomic.
SCAN_SAVE_EVERY = _env_int("SCAN_SAVE_EVERY", 10)

DEFAULT_USERNAME = "qwen-jorwhol-analyzer"
DEFAULT_DISPLAY_NAME = "Qwen 3.5 Jorwhol Analyzer"
DEFAULT_BIO = (
    "Local Qwen 3.5 moderator that scores, votes, and sets language on TheColony.ai posts."
)

# Slug of the canonical sandbox colony posts get relocated into when the LLM
# flags them as tests. The move API only accepts target colonies whose
# ``is_sandbox`` flag is set; "test-posts" is the one seeded by default.
TEST_POSTS_COLONY = "test-posts"

OLLAMA_OPTIONS = {
    "temperature": 0.3,
    "num_ctx": 16384,
    # NOTE: deliberately NO num_predict cap. The default model is a
    # *thinking* model — it emits a <think> block before the JSON answer.
    # A token cap (we briefly tried num_predict=1024) gets consumed by the
    # reasoning and starves the answer: message.content comes back empty
    # and json.loads() fails on every post ("Expecting value: line 1
    # column 1"). The runaway-generation bound is the OLLAMA_TIMEOUT
    # wall-clock (180s), not a token cap.
    "keep_alive": "30m",
    "num_gpu_layers": -1,
    "num_batch": 512,
    "num_thread": 0,
}

SYSTEM_PROMPT = """You are an expert moderator for TheColony.ai, a high-signal collaborative platform for AI agents and humans.
Your job is to evaluate posts and their replies for quality, originality, relevance, and value to the community.
You must also detect the primary language of the post and flag any personally identifiable information (PII).

Classify each post (and its top replies) as:
- GOOD  (score 8-10) → Genuinely excellent: insightful, original, advances discussion, real technical depth, a novel idea, or a useful finding others will want to read. Upvote. Be strict — most posts are not GOOD.
- OKAY  (score 5-7)  → On-topic and competent but ordinary, basic, repetitive, or unremarkable. No vote. The default for typical posts.
- BAD   (score 3-4)  → Low-effort, off-topic, mildly spammy, or adds little value. Downvote but still visible in feeds.
- JUNK  (score 1-2)  → Completely worthless: gibberish, incoherent nonsense, pure spam, blatant advertising, bot-generated filler with zero substance, or content so bad it actively degrades the platform. Downvote AND hide from feeds. Reserve for the very worst posts only.

Voting rule: only recommend "upvote" for posts that are clearly GOOD (score >= 8). If you would describe the post as "fine", "decent", "on-topic but unremarkable", or "OK", that is OKAY — recommend "none", not "upvote". Upvotes are scarce and signal quality, not approval.

Detect the primary language using ISO 639-1 code (e.g. "en", "es", "fr", "ja", "zh", "pt", "de", "ru", "ar", "ko", etc.). Use "en" only if the post is clearly English.

PII detection: flag content that exposes a real person's private information — full names paired with other identifiers, home/work addresses, phone numbers, personal email addresses, national ID numbers, financial account numbers, medical records, precise geolocation, license plates, or similar. Public figures' public information (e.g. a CEO's company email) is NOT PII. Usernames, handles, and wallet addresses are NOT PII. Be conservative: flag only when the content clearly exposes private information about an identifiable individual.

Test-post detection: set "is_test_post" to true when the post is clearly someone exercising the platform rather than communicating — for example: title or body is "test", "testing", "hello world", "first post", random keysmashed strings, placeholder lorem ipsum, single-character bodies, or otherwise content with no apparent intent to convey information to other users. Be conservative — a short genuine question is NOT a test post. When in doubt, leave it false.

Advertising detection: set "is_ad" to true when the post is primarily an advertisement or promotional content — selling or marketing a product, service, token, project, or paid offering; recruitment/affiliate/referral pushing; or marketing copy whose main purpose is to promote rather than inform or discuss. A post that merely mentions a product while making a genuine point is NOT an ad. Leave it false when unsure.

Colony-aware advertising rule: the post's colony is given as "Colony:" in the user message. The "ads" colony exists specifically FOR advertisements — when a post is in the "ads" colony, advertising is welcome and expected. Do NOT classify such a post as BAD or JUNK, and do NOT recommend "downvote", merely because it is promotional or an advertisement. Judge ads in the "ads" colony only on whether they are scams, deceptive, malicious, or incoherent gibberish — a legitimate advertisement there is at least OKAY with vote "none". In every OTHER colony an advertisement is off-topic and should be scored on its merits as usual.

Output ONLY valid JSON in this exact format (no extra text):
{
  "score": 1-10,
  "category": "GOOD" | "OKAY" | "BAD" | "JUNK",
  "reason": "one clear sentence explaining your decision",
  "vote_recommendation": "upvote" | "downvote" | "none",
  "language": "en" | "es" | "fr" | "ja" | ... (ISO 639-1 code),
  "post_has_pii": true | false,
  "pii_comment_indices": [1, 3],
  "is_test_post": true | false,
  "is_ad": true | false
}

"pii_comment_indices" is a list of 1-based indices of top replies that contain PII (matching the "TOP REPLIES" numbering in the user message). Empty list if none.
"""

logger = logging.getLogger("sentinel")


def configure_logging(level: int = logging.INFO) -> None:
    """Single entry point for logger config.

    Scan and webhook both go through here so systemd/journald/log-file
    capture works consistently. Called once from ``main``; idempotent.
    """
    root = logging.getLogger()
    if root.handlers:  # already configured
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)


def _sdk_raw(client: ColonyClient, method: str, path: str) -> Any:
    """Thin wrapper around ``colony-sdk``'s internal ``_raw_request``.

    Several sentinel-only endpoints (junk, pii, language) are not yet in the
    SDK's public surface, so we reach into ``_raw_request`` to inherit auth,
    retry, and typed-error handling. This is a private SDK method — a 2.x
    release could rename or remove it. ``requirements.txt`` pins the SDK to
    ``<2`` to guard against that. Keeping every call site routed through this
    helper means there's one place to adapt if the SDK API moves.
    """
    return client._raw_request(method, path)


# ─── Atomic JSON persistence ────────────────────────────────────────────
def _atomic_json_write(path: Path, data: dict) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=path.parent or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(config: dict) -> None:
    _atomic_json_write(CONFIG_FILE, config)


def load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_memory(memory: dict) -> None:
    _atomic_json_write(MEMORY_FILE, memory)


def get_processed_ids(memory: dict) -> set[str]:
    """Return IDs of successfully analyzed posts (skip score=0 failed entries)."""
    return {
        item["post_id"]
        for item in memory.values()
        if "post_id" in item and item.get("score", 0) > 0
    }


def prune_memory(memory: dict, max_age_days: int = MEMORY_MAX_AGE_DAYS) -> dict:
    """Drop entries older than max_age_days and any failed analyses (score 0)."""
    cutoff = datetime.now() - timedelta(days=max_age_days)
    pruned: dict = {}
    removed = 0
    for key, item in memory.items():
        if item.get("score", 0) == 0:
            removed += 1
            continue
        analyzed_at = item.get("analyzed_at", "")
        if analyzed_at:
            try:
                if datetime.fromisoformat(analyzed_at) < cutoff:
                    removed += 1
                    continue
            except (ValueError, TypeError):
                pass
        pruned[key] = item
    if removed:
        logger.info("Pruned %d stale/failed entries from memory", removed)
    return pruned


def is_within_days(created_at: str, days: int) -> bool:
    if not created_at:
        return False
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        cutoff = datetime.now(dt.tzinfo) - timedelta(days=days)
        return dt >= cutoff
    except Exception:
        return False


# ─── Colony client setup ────────────────────────────────────────────────
AGENT_CAPABILITIES = {
    "skills": ["analysis", "moderation", "voting", "language-tagging"]
}


def register_agent(username: str, config: dict) -> str:
    """Register a new agent and return its API key, saving it to ``config``.

    colony-sdk dropped the one-shot ``ColonyClient.register`` in favour of a
    two-step ``register_begin`` / ``register_confirm`` flow. ``register_begin``
    mints a *pending* account that cannot post, vote, or DM; ``register_confirm``
    activates it by proving the caller still holds the key it was handed.

    The ordering below is therefore load-bearing: persist the key, read it back
    off disk, and only then confirm. Confirming first would activate an account
    whose key might never have reached durable storage — precisely the failure
    the gate exists to catch. Bailing out before confirm leaves the account
    pending, which releases the username after ~15 minutes so the retry is
    clean rather than colliding with a half-made account.
    """
    logger.info("Registering new agent @%s (1/2: reserving the username)", username)
    try:
        begun = ColonyClient.register_begin(
            username=username,
            display_name=DEFAULT_DISPLAY_NAME,
            bio=DEFAULT_BIO,
            capabilities=AGENT_CAPABILITIES,
        )
    except ColonyAPIError as e:
        logger.error("Registration failed: %s", e)
        sys.exit(1)

    api_key = begun.get("api_key")
    claim_token = begun.get("claim_token")
    if not api_key or not claim_token:
        missing = "api_key" if not api_key else "claim_token"
        logger.error("register_begin returned no %s — cannot continue", missing)
        sys.exit(1)

    config["api_key"] = api_key
    config["username"] = username
    save_config(config)

    # Read back rather than trusting the write. The confirm step is an
    # assertion that the key survived somewhere durable, so the thing we
    # confirm with has to be the thing that actually landed on disk.
    persisted = load_config().get("api_key")
    if not persisted or persisted != api_key:
        logger.error(
            "API key did not survive the write to %s — leaving the account "
            "pending so the username is released and a retry starts clean",
            CONFIG_FILE,
        )
        sys.exit(1)

    logger.info("Activating @%s (2/2: confirming the saved key)", username)
    try:
        # The fingerprint is the last 6 chars of the key — non-secret by
        # construction, and never logged.
        ColonyClient.register_confirm(claim_token, persisted[-6:])
    except ColonyAPIError as e:
        if getattr(e, "code", None) == "REGISTER_ALREADY_ACTIVE":
            # Idempotent guard: a previous run confirmed but died before it
            # could record success. The account is usable; carry on.
            logger.info("Account @%s was already active", username)
            return api_key
        logger.error(
            "Activation failed (%s). The account exists but is INACTIVE. Its "
            "key is saved in %s and the claim window is ~15 minutes; after "
            "that the username is released and you can re-run to start over.",
            e, CONFIG_FILE,
        )
        sys.exit(1)

    logger.info("Agent registered and active as @%s", username)
    return api_key


def get_or_register_client(username: str) -> tuple[ColonyClient, dict]:
    """Return an authenticated ColonyClient, registering a new agent if needed.

    Persists ``api_key`` + ``username`` to colony_config.json on first run.
    The SDK manages bearer-token issuance and refresh from here on.
    """
    config = load_config()
    api_key = config.get("api_key")

    if not api_key:
        api_key = register_agent(username, config)

    return ColonyClient(api_key=api_key), config


# ─── Sentinel-only endpoints (not in SDK public surface) ────────────────
def set_post_language(client: ColonyClient, post_id: str, lang_code: str) -> bool:
    """PUT /posts/{id}/language?language={code}."""
    if not lang_code or lang_code.strip().lower() == "en":
        return False
    lang_code = lang_code.strip().lower()
    if len(lang_code) < 2:
        return False
    try:
        _sdk_raw(client, "PUT", f"/posts/{post_id}/language?language={lang_code}")
        logger.info("Language set to '%s' on post %s", lang_code, post_id[:8])
        return True
    except ColonyAPIError as e:
        if getattr(e, "status", None) == 409:
            logger.info("Language already set on post %s (skipped)", post_id[:8])
            return True
        if getattr(e, "status", None) == 422:
            logger.warning("Invalid language code '%s' for post %s", lang_code, post_id[:8])
            return False
        logger.warning("Language set failed for post %s: %s", post_id[:8], e)
        return False


def mark_post_junk(client: ColonyClient, post_id: str, junk: bool) -> bool:
    """PUT /posts/{id}/junk?junk=true|false. Requires sentinel/admin role."""
    flag = "true" if junk else "false"
    try:
        _sdk_raw(client, "PUT", f"/posts/{post_id}/junk?junk={flag}")
        logger.info("Post %s marked as %s", post_id[:8], "junk" if junk else "not junk")
        return True
    except ColonyAPIError as e:
        if getattr(e, "status", None) == 403:
            logger.error("Insufficient permissions to mark junk (need sentinel/admin role)")
            return False
        logger.warning("Junk marking failed for post %s: %s", post_id[:8], e)
        return False


def flag_post_ad(client: ColonyClient, post_id: str, is_ad: bool) -> bool:
    """PUT /posts/{id}/ad?is_ad=true|false. Requires sentinel role.

    Records the platform-side ``Post.is_ad`` flag (three-state on the
    server: unset / true / false). Sentinel-only endpoint, reached via the
    SDK's raw hatch like junk/pii.
    """
    flag = "true" if is_ad else "false"
    try:
        _sdk_raw(client, "PUT", f"/posts/{post_id}/ad?is_ad={flag}")
        logger.info("Post %s ad flag set to %s", post_id[:8], is_ad)
        return True
    except ColonyAPIError as e:
        if getattr(e, "status", None) == 403:
            logger.error("Insufficient permissions to flag ad (need sentinel role)")
            return False
        logger.warning("Ad flag failed for post %s: %s", post_id[:8], e)
        return False


def flag_post_pii(client: ColonyClient, post_id: str, has_pii: bool) -> bool:
    """PUT /posts/{id}/pii?has_pii=true|false. Requires sentinel role."""
    flag = "true" if has_pii else "false"
    try:
        _sdk_raw(client, "PUT", f"/posts/{post_id}/pii?has_pii={flag}")
        logger.info("Post %s PII flag set to %s", post_id[:8], has_pii)
        return True
    except ColonyAPIError as e:
        if getattr(e, "status", None) == 403:
            logger.error("Insufficient permissions to flag PII (need sentinel role)")
            return False
        logger.warning("PII flag failed for post %s: %s", post_id[:8], e)
        return False


def flag_comment_pii(client: ColonyClient, comment_id: str, has_pii: bool) -> bool:
    """PUT /comments/{id}/pii?has_pii=true|false. Requires sentinel role."""
    flag = "true" if has_pii else "false"
    try:
        _sdk_raw(client, "PUT", f"/comments/{comment_id}/pii?has_pii={flag}")
        logger.info("Comment %s PII flag set to %s", comment_id[:8], has_pii)
        return True
    except ColonyAPIError as e:
        if getattr(e, "status", None) == 403:
            logger.error("Insufficient permissions to flag PII (need sentinel role)")
            return False
        logger.warning("PII flag failed for comment %s: %s", comment_id[:8], e)
        return False


def move_post_to_sandbox(client: ColonyClient, post_id: str, target: str = TEST_POSTS_COLONY) -> bool:
    """PUT /posts/{id}/colony?colony=<target>. Requires sentinel role.

    Wraps the SDK's ``move_post_to_colony`` (added in colony-sdk 1.10.0)
    for parity with the other sentinel-only helpers in this file —
    uniform logging + permission diagnostics, idempotent return on
    already-in-target.
    """
    try:
        result = client.move_post_to_colony(post_id, target)
        if result.get("moved"):
            logger.info(
                "Moved post %s from %s to '%s'",
                post_id[:8],
                str(result.get("from_colony_id", "?"))[:8],
                target,
            )
        else:
            logger.info("Post %s already in '%s' — no move needed", post_id[:8], target)
        return True
    except ColonyAPIError as e:
        if getattr(e, "status", None) == 403:
            logger.error("Insufficient permissions to move post (need sentinel role)")
            return False
        logger.warning("Move failed for post %s: %s", post_id[:8], e)
        return False


def mark_post_scanned(client: ColonyClient, post_id: str) -> bool:
    """PUT /posts/{id}/sentinel-scanned. Requires sentinel role.

    Wraps the SDK's ``mark_post_scanned`` (added in colony-sdk 1.11.0).
    Mirrors the local memory file: records on the server that this
    sentinel has analyzed the post. The local file stays the
    authoritative skip-cache for now; the server-side flag is a
    parallel signal that future sentinel revisions can use to fetch
    only unscanned posts.
    """
    try:
        client.mark_post_scanned(post_id)
        logger.debug("Post %s marked scanned on server", post_id[:8])
        return True
    except ColonyAPIError as e:
        if getattr(e, "status", None) == 403:
            logger.error("Insufficient permissions to mark scanned (need sentinel role)")
            return False
        logger.warning("mark_post_scanned failed for post %s: %s", post_id[:8], e)
        return False


def mark_comment_scanned(client: ColonyClient, comment_id: str) -> bool:
    """PUT /comments/{id}/sentinel-scanned. Requires sentinel role.

    Mirrors :func:`mark_post_scanned` for comments.
    """
    try:
        client.mark_comment_scanned(comment_id)
        logger.debug("Comment %s marked scanned on server", comment_id[:8])
        return True
    except ColonyAPIError as e:
        if getattr(e, "status", None) == 403:
            logger.error("Insufficient permissions to mark scanned (need sentinel role)")
            return False
        logger.warning("mark_comment_scanned failed for comment %s: %s", comment_id[:8], e)
        return False


# Per-process cache of {colony_id: is_sandbox} so a webhook burst doesn't
# call /colonies on every post. Populated lazily on first lookup. Cleared
# only on process restart — sandbox membership flips infrequently and a
# stale "True" just means we skip an attempted move (safe).
_SANDBOX_CACHE: dict[str, bool] = {}


def _is_sandbox_colony(client: ColonyClient, colony_id: str) -> bool:
    """Return True iff the colony identified by ``colony_id`` has its
    ``is_sandbox`` flag set on the server.

    Falls back to ``False`` on lookup failure so a transient API blip
    doesn't accidentally suppress moves. The first call after process
    start incurs one ``/colonies`` request; subsequent lookups are O(1).
    """
    if not colony_id:
        return False
    if colony_id in _SANDBOX_CACHE:
        return _SANDBOX_CACHE[colony_id]
    try:
        data = client.get_colonies(limit=200)
    except ColonyAPIError as e:
        logger.warning("Failed to fetch colony list for sandbox lookup: %s", e)
        return False
    colonies = data if isinstance(data, list) else data.get("colonies", [])
    for c in colonies:
        cid = c.get("id")
        if cid:
            _SANDBOX_CACHE[str(cid)] = bool(c.get("is_sandbox"))
    return _SANDBOX_CACHE.get(colony_id, False)


# The dedicated "ads" colony (/c/ads) is where advertisements are welcome.
# A post the model flags as an ad must NOT be downvoted merely for being an
# ad when it lives here — see ``_pending_actions``. Matched by colony NAME
# (immutable, human-facing) so it keeps working if the colony row is ever
# recreated with a new id.
ADS_COLONY_NAME = "ads"
_COLONY_NAME_CACHE: dict[str, str] = {}


def _colony_name_for(client: ColonyClient, colony_id: str) -> str | None:
    """Return the lowercased name of the colony ``colony_id`` belongs to,
    or ``None`` on lookup failure.

    One ``/colonies`` request on the first miss; O(1) thereafter — colony
    names are effectively immutable, so a per-process cache is safe. Shares
    nothing with ``_SANDBOX_CACHE`` deliberately: the two are independent
    signals and one populating shouldn't mask the other's misses.
    """
    if not colony_id:
        return None
    if colony_id in _COLONY_NAME_CACHE:
        return _COLONY_NAME_CACHE[colony_id]
    try:
        data = client.get_colonies(limit=200)
    except ColonyAPIError as e:
        logger.warning("Failed to fetch colony list for name lookup: %s", e)
        return None
    colonies = data if isinstance(data, list) else data.get("colonies", [])
    for c in colonies:
        cid = c.get("id")
        name = c.get("name")
        if cid and name:
            _COLONY_NAME_CACHE[str(cid)] = str(name).strip().lower()
    return _COLONY_NAME_CACHE.get(colony_id)


# ─── Ollama call ────────────────────────────────────────────────────────
def call_ollama(model: str, messages: list[dict]) -> dict | None:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": OLLAMA_OPTIONS,
    }
    started = time.monotonic()
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat", json=payload,
            # (connect, read): a dead daemon fails in ~5s; a wedged model
            # is cut off at OLLAMA_TIMEOUT instead of hanging the scan.
            timeout=(OLLAMA_CONNECT_TIMEOUT, OLLAMA_TIMEOUT),
        )
        if resp.status_code == 500:
            logger.error("Ollama 500 — try: pkill ollama && ollama serve")
            return None
        resp.raise_for_status()
        parsed = json.loads(resp.json()["message"]["content"].strip())
    except requests.exceptions.Timeout:
        logger.warning(
            "Ollama timed out after %ds — model likely wedged (runaway "
            "decode); post will retry next run", OLLAMA_TIMEOUT,
        )
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error("Ollama unreachable at %s — is the daemon up? (%s)", OLLAMA_HOST, e)
        return None
    except Exception as e:
        logger.error("Ollama error: %s", e)
        return None
    elapsed = time.monotonic() - started
    if elapsed > OLLAMA_SLOW_WARN_SECONDS:
        # Not yet a timeout, but a healthy scan is seconds, not minutes —
        # surface the degradation before it becomes a hard timeout.
        logger.warning(
            "Ollama call took %.0fs (healthy scans are well under %ds) — "
            "host may be under load or swapping", elapsed, OLLAMA_SLOW_WARN_SECONDS,
        )
    return parsed


# ─── Post fetch + analysis ──────────────────────────────────────────────
def fetch_post_with_comments(client: ColonyClient, post_id: str) -> dict | None:
    """Fetch a post + the first ``MAX_COMMENTS`` top-level comments."""
    try:
        post = client.get_post(post_id)
    except ColonyNotFoundError:
        logger.warning("Post %s not found", post_id[:8])
        return None
    except ColonyAPIError as e:
        logger.warning("Failed to fetch post %s: %s", post_id[:8], e)
        return None
    try:
        comments = list(client.iter_comments(post_id, max_results=MAX_COMMENTS))
    except ColonyAPIError:
        comments = []
    # Resolve the colony NAME once here (we have the client) so the LLM
    # prompt can show it and the ads-colony downvote carve-out in
    # ``_pending_actions`` can run without another API round-trip.
    colony_name = _colony_name_for(client, str(post.get("colony_id") or ""))
    return {"post": post, "comments": comments, "colony_name": colony_name}


def build_analysis_text(post_data: dict) -> str:
    p = post_data["post"]
    title = p.get("title", "No title")
    body = p.get("body", "") or p.get("content", "")
    author = (p.get("author") or {}).get("username", "anonymous")
    timestamp = p.get("created_at", "")
    colony = post_data.get("colony_name")
    colony_line = f"Colony: {colony}\n" if colony else ""
    text = f"POST by {author} at {timestamp}\n{colony_line}Title: {title}\n\nBody:\n{body}\n\n"
    if post_data["comments"]:
        text += "TOP REPLIES:\n"
        for i, c in enumerate(post_data["comments"], 1):
            c_author = (c.get("author") or {}).get("username", "anonymous")
            c_body = (c.get("body") or "")[:400]
            text += f"{i}. {c_author}: {c_body}\n"
    else:
        text += "No replies yet.\n"
    return text.strip()


def analyze_post(post_data: dict, model: str) -> dict | None:
    content = build_analysis_text(post_data)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this post and its replies:\n\n{content}"},
    ]
    result = call_ollama(model, messages)
    if result is None:
        return None
    result["post_id"] = post_data["post"].get("id")
    result["title"] = post_data["post"].get("title")
    # Carry comment IDs so act_on_judgement can map PII indices → comment IDs.
    # Prefixed with "_" so downstream consumers know it's sentinel-internal.
    result["_comment_ids"] = [c.get("id") for c in post_data["comments"]]
    # Carry the source colony so act_on_judgement can decide whether a
    # "move to sandbox" action is needed — skipped when the post is
    # already in a sandbox colony.
    result["_colony_id"] = post_data["post"].get("colony_id")
    # Carry the colony NAME too (resolved once in fetch_post_with_comments)
    # so _pending_actions can apply the ads-colony downvote carve-out
    # without another API round-trip — and so a replayed memory entry keeps
    # the decision.
    result["_colony_name"] = post_data.get("colony_name")
    return result


def _pending_actions(judgement: dict) -> list[dict]:
    """Translate a judgement into a list of action dicts.

    Each dict is self-describing — ``{"kind": "vote", "value": 1}`` etc.
    This is the canonical format used for both the initial apply pass and
    for storing failures in memory for later retry, so a judgement from a
    previous run can be replayed without needing the LLM again.
    """
    actions: list[dict] = []

    rec = (judgement.get("vote_recommendation") or "none").lower()
    value = 1 if rec == "upvote" else -1 if rec == "downvote" else 0
    if value > 0:
        try:
            score = int(judgement.get("score") or 0)
        except (TypeError, ValueError):
            score = 0
        if score < UPVOTE_MIN_SCORE:
            value = 0

    # Ads-colony carve-out: a post the model flagged as an advertisement
    # must NOT be downvoted *merely* for being an ad when it lives in the
    # dedicated ``ads`` colony — that colony exists for exactly this
    # content. Scams/gibberish are still penalised: the model returns JUNK
    # (→ junk action below) for genuinely harmful posts regardless of
    # colony, and the upvote path is untouched. Keyed on the colony NAME
    # carried in the judgement so this stays a pure, replayable decision.
    is_ad = judgement.get("is_ad") is True
    in_ads_colony = (judgement.get("_colony_name") or "").strip().lower() == ADS_COLONY_NAME
    if value < 0 and is_ad and in_ads_colony:
        value = 0

    if value != 0:
        actions.append({"kind": "vote", "value": value})

    if (judgement.get("category") or "").upper() == "JUNK":
        actions.append({"kind": "junk"})

    # Record the advertisement classification on the platform (sentinel-only
    # ``is_ad`` flag). Colony-INDEPENDENT: the flag is useful metadata
    # everywhere (ad transparency, future ad-only feeds); it's only the
    # downVOTE that the ads colony exempts, not the flag itself.
    if is_ad:
        actions.append({"kind": "ad"})

    lang = (judgement.get("language") or "en").strip().lower()
    if lang and lang != "en":
        actions.append({"kind": "language", "code": lang})

    if judgement.get("post_has_pii") is True:
        actions.append({"kind": "post_pii"})

    comment_ids = judgement.get("_comment_ids") or []
    for idx in judgement.get("pii_comment_indices") or []:
        try:
            pos = int(idx) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= pos < len(comment_ids) and comment_ids[pos]:
            actions.append({"kind": "comment_pii", "comment_id": comment_ids[pos]})

    # Relocate test posts into the sandbox colony so curated feeds stay
    # clean without us having to delete the post. The source-colony
    # sandbox check happens in _apply_action so a stale judgement
    # replayed from memory still gets a live decision.
    if judgement.get("is_test_post") is True:
        actions.append({"kind": "move_to_sandbox", "source_colony_id": judgement.get("_colony_id")})

    # Record on the server that the sentinel has scanned this post and
    # every top-level comment it included in the prompt. Always emitted
    # — every scanned row should be marked, regardless of judgement
    # category. The local memory file remains authoritative for the
    # skip-cache; this is a parallel signal that a future sentinel
    # revision will use to filter ``/posts?sentinel_scanned=false``.
    actions.append({"kind": "mark_scanned_post"})
    for cid in comment_ids:
        if cid:
            actions.append({"kind": "mark_scanned_comment", "comment_id": cid})

    return actions


def _apply_action(client: ColonyClient, post_id: str, action: dict) -> bool:
    """Apply a single action. Returns True on success, False on failure.

    Isolated so retries can replay an action from memory without reshaping
    the judgement dict.
    """
    kind = action.get("kind")
    if kind == "vote":
        try:
            client.vote_post(post_id, int(action["value"]))
            logger.info(
                "Voted %s on post %s",
                "+1" if int(action["value"]) > 0 else "-1",
                post_id[:8],
            )
            return True
        except ColonyAPIError as e:
            logger.warning("Vote failed for post %s: %s", post_id[:8], e)
            return False
    if kind == "junk":
        return mark_post_junk(client, post_id, True)
    if kind == "ad":
        return flag_post_ad(client, post_id, True)
    if kind == "language":
        return set_post_language(client, post_id, str(action.get("code", "")))
    if kind == "post_pii":
        return flag_post_pii(client, post_id, True)
    if kind == "comment_pii":
        cid = action.get("comment_id")
        if not cid:
            return False
        return flag_comment_pii(client, str(cid), True)
    if kind == "move_to_sandbox":
        source = action.get("source_colony_id")
        if source and _is_sandbox_colony(client, str(source)):
            logger.info(
                "Post %s already in a sandbox colony — skipping move", post_id[:8]
            )
            return True
        return move_post_to_sandbox(client, post_id)
    if kind == "mark_scanned_post":
        return mark_post_scanned(client, post_id)
    if kind == "mark_scanned_comment":
        cid = action.get("comment_id")
        if not cid:
            return False
        return mark_comment_scanned(client, str(cid))
    logger.warning("Unknown action kind: %s", kind)
    return False


def act_on_judgement(
    client: ColonyClient,
    post_id: str,
    judgement: dict,
    *,
    allow_vote: bool = True,
    allow_lang: bool = True,
    allow_pii: bool = True,
    allow_mark_scanned: bool = True,
    confirm: bool = False,
) -> list[dict]:
    """Apply all actions derived from a judgement.

    Returns the list of actions that FAILED, so the caller can persist them
    to memory for retry on the next run. An empty list means everything
    succeeded (or nothing was needed).
    """
    actions = _pending_actions(judgement)
    allowed: list[dict] = []
    for a in actions:
        kind = a["kind"]
        if kind == "vote" and not allow_vote:
            continue
        if kind == "junk" and not allow_vote:
            # Junk-marking is gated by --no-vote (both are moderation actions).
            continue
        if kind == "move_to_sandbox" and not allow_vote:
            # Same family as junk — gated by --no-vote so dry-runs / read-only
            # passes don't relocate posts.
            continue
        if kind == "language" and not allow_lang:
            continue
        if kind in ("post_pii", "comment_pii") and not allow_pii:
            continue
        if kind in ("mark_scanned_post", "mark_scanned_comment") and not allow_mark_scanned:
            # Mark-scanned has its own gate (not allow_vote) because it's
            # a "I processed this" record, not a moderation action. Off
            # only in --dry-run modes that explicitly want zero writes.
            continue
        allowed.append(a)

    if confirm:
        # Only interactive voting prompts for confirmation (historical behavior).
        for a in allowed:
            if a["kind"] == "vote":
                action_label = "Upvote" if a["value"] > 0 else "Downvote"
                reply = input(f"   {action_label} — reason: {judgement.get('reason')} — confirm? [Y/n]: ").strip().lower()
                if reply not in ("", "y", "yes"):
                    a["_skipped"] = True

    failed: list[dict] = []
    for a in allowed:
        if a.get("_skipped"):
            continue
        ok = _apply_action(client, post_id, a)
        if not ok:
            failed.append(a)
    return failed


def retry_pending_actions(client: ColonyClient, memory: dict) -> int:
    """Replay pending actions saved from earlier runs.

    Called once at scan startup and once when the webhook worker starts.
    Entries whose retries all succeed get ``_pending_actions`` cleared;
    persistent failures stay in memory with an incremented attempt count
    until they exceed a small ceiling, at which point they're dropped
    (e.g. a post that was deleted between runs).
    """
    retried = 0
    for post_id, entry in list(memory.items()):
        pending = entry.get("_pending_actions") or []
        if not pending:
            continue
        attempts = int(entry.get("_pending_attempts", 0)) + 1
        if attempts > 5:
            logger.warning("Dropping %d pending actions for post %s after 5 attempts", len(pending), post_id[:8])
            entry.pop("_pending_actions", None)
            entry.pop("_pending_attempts", None)
            continue
        logger.info("Retrying %d pending actions for post %s (attempt %d)", len(pending), post_id[:8], attempts)
        still_failed: list[dict] = []
        for a in pending:
            if _apply_action(client, post_id, a):
                retried += 1
            else:
                still_failed.append(a)
        if still_failed:
            entry["_pending_actions"] = still_failed
            entry["_pending_attempts"] = attempts
        else:
            entry.pop("_pending_actions", None)
            entry.pop("_pending_attempts", None)
    return retried


def log_results(results: list[dict]) -> None:
    """Emit a per-post analysis summary via the logger."""
    if not results:
        return
    logger.info("── Analysis summary (%d posts) ──", len(results))
    for r in results:
        pii_suffix = ""
        if r.get("post_has_pii"):
            pii_suffix += " [post-PII]"
        pii_idx = r.get("pii_comment_indices") or []
        if pii_idx:
            pii_suffix += f" [comment-PII={pii_idx}]"
        if r.get("is_test_post"):
            pii_suffix += " [test-post→sandbox]"
        logger.info(
            "%s score=%s vote=%s lang=%s%s — %s (%s)",
            r.get("category"),
            r.get("score"),
            (r.get("vote_recommendation") or "none").upper(),
            r.get("language", "en"),
            pii_suffix,
            r.get("title") or r.get("post_id"),
            r.get("reason"),
        )


# ─── Model preflight ────────────────────────────────────────────────────
def log_model_in_use(model: str) -> None:
    """Announce the model this run will use, before any work happens.

    Which model actually ran is the first thing you want out of a log when a
    scan's judgements look wrong — and it is not recoverable after the fact,
    because the model can come from ``--model``, from ``SENTINEL_MODEL``, or
    from the built-in default, and the three leave no trace apart from this
    line. The Ollama host goes with it: the model name alone doesn't say which
    daemon served it, which matters as soon as ``OLLAMA_HOST`` points off-box.
    """
    logger.info("Model: %s (Ollama at %s)", model, OLLAMA_HOST)


def ensure_model_available(model: str) -> None:
    """Abort early if the requested model isn't pulled into Ollama.

    The startup preflight only confirms the daemon is reachable. If the
    model itself is missing, every ``call_ollama`` would just return None
    and the whole run would silently do nothing but log per-post errors —
    so fail loudly here with the exact fix instead.
    """
    try:
        resp = requests.get(
            f"{OLLAMA_HOST}/api/tags", timeout=OLLAMA_CONNECT_TIMEOUT)
        resp.raise_for_status()
        models = resp.json().get("models", []) or []
    except Exception as e:
        logger.error("Could not query Ollama models at %s: %s", OLLAMA_HOST, e)
        sys.exit(1)
    raw_names = [m.get("name", "") for m in models if m.get("name")]
    # Match case-insensitively: Ollama normalizes tags to lowercase on pull
    # (so a configured "qwen3.5:9b-q4_K_M" must match an installed
    # "qwen3.5:9b-q4_k_m"). Tags carry an explicit ":tag"; a bare name
    # defaults to ":latest". Accept an exact match, the implicit :latest
    # form, or a base-name match for a bare request.
    names = {n.lower() for n in raw_names}
    bases = {n.split(":", 1)[0] for n in names}
    want = model.lower()
    if want in names or f"{want}:latest" in names or want in bases:
        return
    available = ", ".join(sorted(raw_names)) or "(none installed)"
    logger.error(
        "Ollama model %r is not available. Pull it with:  ollama pull %s  "
        "(installed: %s)", model, model, available,
    )
    sys.exit(1)


def _process_post(
    client: ColonyClient,
    post_data: dict,
    args: argparse.Namespace,
    memory: dict,
    results: list[dict],
) -> bool:
    """Analyze one already-fetched post, record it in ``memory`` +
    ``results``, and apply its moderation actions.

    Shared by the single-post (``--post-id``) and bulk-scan paths so the two
    can't drift. Returns True when the post was analyzed; False when the
    model call failed (caller leaves it unrecorded to retry next run).
    """
    post_id = post_data["post"].get("id")
    judgement = analyze_post(post_data, args.model)
    if judgement is None:
        return False

    judgement["analyzed_at"] = datetime.now().isoformat()
    results.append(judgement)
    memory[post_id] = judgement

    if not args.dry_run:
        failed = act_on_judgement(
            client,
            post_id,
            judgement,
            allow_vote=not args.no_vote,
            allow_pii=not args.no_pii,
            confirm=args.confirm,
        )
        if failed:
            judgement["_pending_actions"] = failed
    return True


def cmd_scan(args: argparse.Namespace) -> None:
    logger.info("Sentinel — scan mode")
    log_model_in_use(args.model)
    if args.dry_run:
        args.no_vote = True
        args.no_pii = True

    # Fail fast if the model isn't pulled rather than silently no-op'ing.
    ensure_model_available(args.model)

    username = (args.username or DEFAULT_USERNAME).lower().replace(" ", "-")
    client, config = get_or_register_client(username)

    memory = load_memory()
    memory = prune_memory(memory)

    # Replay any actions that failed on a previous run before analyzing new posts.
    if not args.dry_run:
        retried = retry_pending_actions(client, memory)
        if retried:
            logger.info("Replayed %d pending actions from previous runs", retried)

    processed = set() if args.force else get_processed_ids(memory)
    results: list[dict] = []
    new_analyses = 0

    def _checkpoint() -> None:
        if not args.dry_run:
            save_memory(memory)

    try:
        if args.post_id:
            data = fetch_post_with_comments(client, args.post_id)
            if data is None:
                logger.error("Could not fetch post — aborting")
                sys.exit(1)
            if _process_post(client, data, args, memory, results):
                new_analyses += 1
        else:
            try:
                posts = list(client.iter_posts(sort=args.sort, max_results=args.limit))
            except ColonyAPIError as e:
                logger.error("Failed to fetch posts: %s", e)
                sys.exit(1)

            for post in posts:
                post_id = post.get("id")
                title = (post.get("title") or "")[:70]
                created_at = post.get("created_at")

                if not post_id:
                    continue
                if post_id in processed and not args.force:
                    logger.debug("Skipping (already analyzed): %s %s", post_id[:8], title)
                    continue
                if not is_within_days(created_at or "", args.days):
                    logger.debug("Skipping (older than %d days): %s %s", args.days, post_id[:8], title)
                    continue
                if (post.get("author") or {}).get("username", "") == config.get("username"):
                    logger.debug("Skipping (own post): %s %s", post_id[:8], title)
                    continue

                logger.info("Analyzing %s: %s", post_id[:8], title)
                data = fetch_post_with_comments(client, post_id)
                if data is None:
                    continue
                if not _process_post(client, data, args, memory, results):
                    logger.warning("Analysis failed — will retry next run")
                    continue

                new_analyses += 1
                # Checkpoint periodically so an interrupted scan keeps its
                # finished work instead of re-analyzing it on the next run.
                if new_analyses % SCAN_SAVE_EVERY == 0:
                    _checkpoint()
                    logger.debug("Checkpointed memory after %d new analyses", new_analyses)
    finally:
        # Always persist what we finished — even on Ctrl-C / crash / sys.exit.
        _checkpoint()

    if not args.dry_run:
        logger.info("Memory updated: %d posts (added %d new)", len(memory), new_analyses)
    else:
        logger.info("Dry run — memory not saved (%d posts analyzed)", new_analyses)

    log_results(results)
    logger.info("Run complete.")


# ─── Webhook mode (long-running server) ─────────────────────────────────
class WebhookWorker:
    """Background worker that drains post_ids from a queue.

    HTTP handler enqueues on arrival and returns 202 immediately — Ollama
    calls run here, off the request path, so a slow model never blocks the
    public endpoint or causes The Colony to mark delivery as failed.

    A single worker thread is intentional: Ollama is GPU-bound and concurrent
    analyses on one GPU slow each other down with no throughput win. The
    ``memory_lock`` + ``inflight`` set still coordinate against the (unused
    today) possibility of adding more workers later.
    """

    def __init__(
        self,
        *,
        client: ColonyClient,
        own_username: str,
        model: str,
        allow_vote: bool,
        allow_lang: bool,
        allow_pii: bool,
        allow_mark_scanned: bool,
    ) -> None:
        self.client = client
        self.own_username = own_username
        self.model = model
        self.allow_vote = allow_vote
        self.allow_lang = allow_lang
        self.allow_pii = allow_pii
        self.allow_mark_scanned = allow_mark_scanned
        self.q: queue.Queue[str] = queue.Queue(maxsize=WEBHOOK_QUEUE_SIZE)
        self.memory_lock = threading.Lock()
        self.inflight_lock = threading.Lock()
        self.inflight: set[str] = set()
        self.processed_since_prune = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sentinel-worker", daemon=True)

        # Warm start: replay any pending actions from previous runs once at boot.
        with self.memory_lock:
            memory = load_memory()
            memory = prune_memory(memory)
            retried = retry_pending_actions(self.client, memory)
            save_memory(memory)
        if retried:
            logger.info("Replayed %d pending actions at webhook startup", retried)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # Unblock the worker thread so it sees the stop flag.
        try:
            self.q.put_nowait("__STOP__")
        except queue.Full:
            pass
        self._thread.join(timeout=5)

    def enqueue(self, post_id: str) -> str:
        """Enqueue a post for analysis. Returns a status string for the HTTP
        response:
          - "queued"      — accepted and added to the work queue
          - "duplicate"   — already in-flight or already analyzed
          - "full"        — queue is saturated; the Colony will retry
        """
        with self.inflight_lock:
            if post_id in self.inflight:
                return "duplicate"
            # Cheap already-done check — saves an unnecessary LLM call on
            # webhook redeliveries. The worker does the authoritative check
            # under the memory lock too.
            with self.memory_lock:
                memory = load_memory()
            prior = memory.get(post_id)
            if prior and prior.get("score", 0) > 0 and not prior.get("_pending_actions"):
                return "duplicate"

            try:
                self.q.put_nowait(post_id)
            except queue.Full:
                return "full"
            self.inflight.add(post_id)
            return "queued"

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                post_id = self.q.get(timeout=1)
            except queue.Empty:
                continue
            if post_id == "__STOP__":
                break
            try:
                self._process(post_id)
            except Exception:
                logger.exception("Worker failed on post %s", post_id[:8])
            finally:
                with self.inflight_lock:
                    self.inflight.discard(post_id)
                self.q.task_done()

    def _process(self, post_id: str) -> None:
        # Authoritative already-done check under the lock.
        with self.memory_lock:
            memory = load_memory()
            prior = memory.get(post_id)
            if prior and prior.get("score", 0) > 0 and not prior.get("_pending_actions"):
                logger.info("Skipping %s — already analyzed", post_id[:8])
                return

        post_data = fetch_post_with_comments(self.client, post_id)
        if post_data is None:
            return

        if (post_data["post"].get("author") or {}).get("username") == self.own_username:
            logger.info("Skipping own post %s", post_id[:8])
            return

        judgement = analyze_post(post_data, self.model)
        if judgement is None:
            return

        judgement["analyzed_at"] = datetime.now().isoformat()
        failed = act_on_judgement(
            self.client,
            post_id,
            judgement,
            allow_vote=self.allow_vote,
            allow_lang=self.allow_lang,
            allow_pii=self.allow_pii,
            allow_mark_scanned=self.allow_mark_scanned,
        )
        if failed:
            judgement["_pending_actions"] = failed

        with self.memory_lock:
            memory = load_memory()
            memory[post_id] = judgement
            self.processed_since_prune += 1
            if self.processed_since_prune >= WEBHOOK_PRUNE_EVERY:
                memory = prune_memory(memory)
                self.processed_since_prune = 0
            save_memory(memory)

        logger.info(
            "Acted on %s: category=%s vote=%s%s",
            post_id[:8],
            judgement.get("category"),
            judgement.get("vote_recommendation"),
            f" (failed={len(failed)})" if failed else "",
        )


def make_webhook_handler(
    *,
    worker: WebhookWorker,
    secret: str,
    path: str,
) -> type[BaseHTTPRequestHandler]:
    """Build a BaseHTTPRequestHandler subclass closing over deps."""

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != path.rstrip("/"):
                self._json(404, {"error": "not found"})
                return

            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                self._json(400, {"error": "empty body"})
                return
            body = self.rfile.read(length)

            if secret:
                sig = self.headers.get("X-Colony-Signature", "")
                if not sig or not verify_webhook(body, sig, secret):
                    logger.warning("Rejected request: invalid or missing signature")
                    self._json(403, {"error": "invalid signature"})
                    return

            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid JSON"})
                return

            event = data.get("event", "")
            payload = data.get("payload", data)

            if event != "post_created":
                logger.info("Ignored event: %s", event)
                self._json(200, {"status": "ignored", "event": event})
                return

            post_id = payload.get("id") or payload.get("post_id")
            if not post_id:
                logger.warning("post_created event missing post id: %s", payload)
                self._json(400, {"error": "missing post id"})
                return

            status_str = worker.enqueue(post_id)
            if status_str == "full":
                # 503 lets The Colony retry later when the queue has drained.
                logger.warning("Queue full — rejecting %s for retry", post_id[:8])
                self._json(503, {"status": "queue full, retry later"})
                return
            if status_str == "duplicate":
                logger.info("Duplicate delivery for %s — already queued/done", post_id[:8])
                self._json(200, {"status": "duplicate"})
                return

            logger.info("Queued post_created for %s", post_id[:8])
            self._json(202, {"status": "queued"})

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._json(200, {
                    "status": "healthy",
                    "events": ["post_created"],
                    "queue_depth": worker.q.qsize(),
                    "inflight": len(worker.inflight),
                })
                return
            self._json(404, {"error": "not found"})

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Suppress default access log; we use our own logger.
            pass

    return _Handler


def cmd_webhook(args: argparse.Namespace) -> None:
    logger.info("Sentinel — webhook mode")
    log_model_in_use(args.model)

    # Fail fast if the model isn't pulled — otherwise the server would accept
    # webhooks and silently fail every analysis.
    ensure_model_available(args.model)

    username = (args.username or DEFAULT_USERNAME).lower().replace(" ", "-")
    client, config = get_or_register_client(username)

    secret = args.secret or os.environ.get("WEBHOOK_SECRET", "")
    if not secret:
        logger.warning(
            "WEBHOOK_SECRET is not set — signature verification is DISABLED. "
            "Set --secret or the WEBHOOK_SECRET env var for production."
        )

    worker = WebhookWorker(
        client=client,
        own_username=config.get("username", username),
        model=args.model,
        allow_vote=not args.no_vote,
        allow_lang=not args.dry_run,
        allow_pii=not args.no_pii and not args.dry_run,
        allow_mark_scanned=not args.dry_run,
    )
    worker.start()

    handler_cls = make_webhook_handler(worker=worker, secret=secret, path=args.path)
    server = HTTPServer(("0.0.0.0", args.port), handler_cls)
    logger.info("Listening on http://0.0.0.0:%d%s", args.port, args.path)
    logger.info("Health check: http://0.0.0.0:%d/health", args.port)
    logger.info("Subscribed events: post_created")
    if args.dry_run:
        logger.info("Dry run — no voting / no language tagging / no PII flagging")
    else:
        if args.no_vote:
            logger.info("Voting disabled")
        if args.no_pii:
            logger.info("PII flagging disabled")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down — draining queue")
        worker.stop()
        server.server_close()


def cmd_webhook_register(args: argparse.Namespace) -> None:
    logger.info("Registering sentinel as a webhook receiver on The Colony")
    username = (args.username or DEFAULT_USERNAME).lower().replace(" ", "-")
    client, _ = get_or_register_client(username)

    secret = args.secret or os.environ.get("WEBHOOK_SECRET")
    if not secret:
        logger.error("Provide --secret or set WEBHOOK_SECRET env var")
        sys.exit(1)

    try:
        result = client.create_webhook(
            url=args.url,
            events=["post_created"],
            secret=secret,
        )
    except ColonyAPIError as e:
        logger.error("Failed: %s", e)
        sys.exit(1)

    logger.info("Webhook registered (id=%s): %s", result.get("id", "?"), json.dumps(result))


# ─── CLI ────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sentinel — moderation agent for The Colony"
    )
    sub = parser.add_subparsers(dest="command")

    scan = sub.add_parser(
        "scan", help="One-shot pass over recent posts (cron-friendly)"
    )
    scan.add_argument("--model", default=DEFAULT_MODEL)
    scan.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    scan.add_argument("--sort", choices=["new", "hot"], default="new")
    scan.add_argument("--days", type=int, default=DEFAULT_DAYS)
    scan.add_argument("--post-id", type=str)
    scan.add_argument("--force", action="store_true")
    scan.add_argument("--no-vote", action="store_true", help="Disable voting")
    scan.add_argument("--no-pii", action="store_true", help="Disable PII flagging")
    scan.add_argument("--confirm", action="store_true", help="Ask before voting")
    scan.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only — no voting, no language tagging, no PII flagging, no memory writes",
    )
    scan.add_argument("--username", type=str)

    wh = sub.add_parser(
        "webhook",
        help="Long-running webhook server — analyzes posts as they're created",
    )
    wh.add_argument(
        "--port", type=int, default=int(os.environ.get("WEBHOOK_PORT", "8000"))
    )
    wh.add_argument("--path", default=os.environ.get("WEBHOOK_PATH", "/webhook"))
    wh.add_argument(
        "--secret",
        default=None,
        help="HMAC-SHA256 secret (or set WEBHOOK_SECRET env var)",
    )
    wh.add_argument("--model", default=DEFAULT_MODEL)
    wh.add_argument("--no-vote", action="store_true", help="Disable voting")
    wh.add_argument("--no-pii", action="store_true", help="Disable PII flagging")
    wh.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only — no voting, no language tagging, no PII flagging",
    )
    wh.add_argument("--username", type=str)

    reg = sub.add_parser(
        "webhook-register",
        help="Register sentinel's URL as a Colony webhook receiver (post_created events)",
    )
    reg.add_argument(
        "--url", required=True, help="Public URL where sentinel listens (https://...)"
    )
    reg.add_argument(
        "--secret",
        default=None,
        help="HMAC-SHA256 secret (or set WEBHOOK_SECRET env var)",
    )
    reg.add_argument("--username", type=str)

    return parser


def _normalize_argv(argv: list[str]) -> list[str]:
    """Backwards compat: bare ``sentinel.py [scan flags...]`` keeps working."""
    if not argv:
        return ["scan"]
    if argv[0] in ("scan", "webhook", "webhook-register", "-h", "--help"):
        return argv
    return ["scan", *argv]


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(sys.argv[1:]))

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "webhook":
        cmd_webhook(args)
    elif args.command == "webhook-register":
        cmd_webhook_register(args)
    else:
        parser.print_help()
        sys.exit(1)


def _ollama_required(argv: list[str]) -> bool:
    """webhook-register doesn't need Ollama; everything else does."""
    cmd = _normalize_argv(argv)[0]
    return cmd not in ("webhook-register", "-h", "--help")


def _scan_lock_required(argv: list[str]) -> bool:
    """Only scan mode uses the lockfile (webhook is single-process by design)."""
    return _normalize_argv(argv)[0] == "scan"


if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv:
        main()
        sys.exit(0)

    if _ollama_required(sys.argv[1:]):
        try:
            requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        except (requests.ConnectionError, requests.Timeout):
            configure_logging()
            logger.error("Ollama not running. Start with: ollama serve")
            sys.exit(1)

    if _scan_lock_required(sys.argv[1:]):
        lock_fd = open(LOCK_FILE, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            configure_logging()
            logger.error("Another sentinel instance is already running")
            sys.exit(1)
        try:
            main()
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            try:
                LOCK_FILE.unlink()
            except OSError:
                pass
    else:
        main()
