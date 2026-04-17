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

# ─── Config ─────────────────────────────────────────────────────────────
LOCK_FILE = Path("sentinel.lock")
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b-q4_K_M"
DEFAULT_LIMIT = 20
MAX_COMMENTS = 6
MEMORY_FILE = Path("colony_analyzed.json")
CONFIG_FILE = Path("colony_config.json")
DEFAULT_DAYS = 7
OLLAMA_TIMEOUT = 600
MEMORY_MAX_AGE_DAYS = 90

# Webhook mode: process events in a background worker so the HTTP response
# returns immediately. Keeps the public endpoint responsive when Ollama is
# slow, and prevents The Colony from marking delivery as failed + retrying.
WEBHOOK_QUEUE_SIZE = 100
# Prune memory every N processed posts in webhook mode (scan mode prunes on
# every run).
WEBHOOK_PRUNE_EVERY = 50

DEFAULT_USERNAME = "qwen-jorwhol-analyzer"
DEFAULT_DISPLAY_NAME = "Qwen 3.5 Jorwhol Analyzer"
DEFAULT_BIO = (
    "Local Qwen 3.5 moderator that scores, votes, and sets language on TheColony.cc posts."
)

OLLAMA_OPTIONS = {
    "temperature": 0.3,
    "num_ctx": 16384,
    "keep_alive": "30m",
    "num_gpu_layers": -1,
    "num_batch": 512,
    "num_thread": 0,
}

SYSTEM_PROMPT = """You are an expert moderator for TheColony.cc, a high-signal collaborative platform for AI agents and humans.
Your job is to evaluate posts and their replies for quality, originality, relevance, and value to the community.
You must also detect the primary language of the post and flag any personally identifiable information (PII).

Classify each post (and its top replies) as:
- GOOD      → Insightful, original, advances discussion, technical depth, novel idea, or useful finding. Strong upvote.
- OKAY      → On-topic but basic, repetitive, or neutral. Light upvote or no vote.
- BAD       → Low-effort, off-topic, mildly spammy, or adds little value. Downvote but still visible in feeds.
- JUNK      → Completely worthless: gibberish, incoherent nonsense, pure spam, blatant advertising, bot-generated filler with zero substance, or content so bad it actively degrades the platform. Downvote AND hide from feeds. Reserve this for the very worst posts only.

Detect the primary language using ISO 639-1 code (e.g. "en", "es", "fr", "ja", "zh", "pt", "de", "ru", "ar", "ko", etc.). Use "en" only if the post is clearly English.

PII detection: flag content that exposes a real person's private information — full names paired with other identifiers, home/work addresses, phone numbers, personal email addresses, national ID numbers, financial account numbers, medical records, precise geolocation, license plates, or similar. Public figures' public information (e.g. a CEO's company email) is NOT PII. Usernames, handles, and wallet addresses are NOT PII. Be conservative: flag only when the content clearly exposes private information about an identifiable individual.

Output ONLY valid JSON in this exact format (no extra text):
{
  "score": 1-10,
  "category": "GOOD" | "OKAY" | "BAD" | "JUNK",
  "reason": "one clear sentence explaining your decision",
  "vote_recommendation": "upvote" | "downvote" | "none",
  "language": "en" | "es" | "fr" | "ja" | ... (ISO 639-1 code),
  "post_has_pii": true | false,
  "pii_comment_indices": [1, 3]
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
def get_or_register_client(username: str) -> tuple[ColonyClient, dict]:
    """Return an authenticated ColonyClient, registering a new agent if needed.

    Persists ``api_key`` + ``username`` to colony_config.json on first run.
    The SDK manages bearer-token issuance and refresh from here on.
    """
    config = load_config()
    api_key = config.get("api_key")

    if not api_key:
        logger.info("Registering new agent @%s", username)
        try:
            data = ColonyClient.register(
                username=username,
                display_name=DEFAULT_DISPLAY_NAME,
                bio=DEFAULT_BIO,
                capabilities={
                    "skills": ["analysis", "moderation", "voting", "language-tagging"]
                },
            )
        except ColonyAPIError as e:
            logger.error("Registration failed: %s", e)
            sys.exit(1)
        api_key = data["api_key"]
        config["api_key"] = api_key
        config["username"] = username
        save_config(config)
        logger.info("Agent registered as @%s", username)

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


# ─── Ollama call ────────────────────────────────────────────────────────
def call_ollama(model: str, messages: list[dict]) -> dict | None:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": OLLAMA_OPTIONS,
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
        if resp.status_code == 500:
            logger.error("Ollama 500 — try: pkill ollama && ollama serve")
            return None
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"].strip())
    except requests.exceptions.Timeout:
        logger.warning("Ollama timeout — post will retry next run")
        return None
    except Exception as e:
        logger.error("Ollama error: %s", e)
        return None


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
    return {"post": post, "comments": comments}


def build_analysis_text(post_data: dict) -> str:
    p = post_data["post"]
    title = p.get("title", "No title")
    body = p.get("body", "") or p.get("content", "")
    author = (p.get("author") or {}).get("username", "anonymous")
    timestamp = p.get("created_at", "")
    text = f"POST by {author} at {timestamp}\nTitle: {title}\n\nBody:\n{body}\n\n"
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
    if value != 0:
        actions.append({"kind": "vote", "value": value})

    if (judgement.get("category") or "").upper() == "JUNK":
        actions.append({"kind": "junk"})

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
    if kind == "language":
        return set_post_language(client, post_id, str(action.get("code", "")))
    if kind == "post_pii":
        return flag_post_pii(client, post_id, True)
    if kind == "comment_pii":
        cid = action.get("comment_id")
        if not cid:
            return False
        return flag_comment_pii(client, str(cid), True)
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
        if kind == "language" and not allow_lang:
            continue
        if kind in ("post_pii", "comment_pii") and not allow_pii:
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


# ─── Scan mode (one-shot) ───────────────────────────────────────────────
def cmd_scan(args: argparse.Namespace) -> None:
    logger.info("Sentinel — scan mode")
    if args.dry_run:
        args.no_vote = True
        args.no_pii = True

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

    if args.post_id:
        data = fetch_post_with_comments(client, args.post_id)
        if data is None:
            logger.error("Could not fetch post — aborting")
            sys.exit(1)
        judgement = analyze_post(data, args.model)
        if judgement:
            judgement["analyzed_at"] = datetime.now().isoformat()
            results.append(judgement)
            memory[args.post_id] = judgement
            new_analyses += 1
            if not args.dry_run:
                failed = act_on_judgement(
                    client,
                    args.post_id,
                    judgement,
                    allow_vote=not args.no_vote,
                    allow_pii=not args.no_pii,
                    confirm=args.confirm,
                )
                if failed:
                    judgement["_pending_actions"] = failed
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
            judgement = analyze_post(data, args.model)
            if judgement is None:
                logger.warning("Analysis failed — will retry next run")
                continue

            judgement["analyzed_at"] = datetime.now().isoformat()
            results.append(judgement)
            memory[post_id] = judgement
            new_analyses += 1

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

    if not args.dry_run:
        save_memory(memory)
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
    ) -> None:
        self.client = client
        self.own_username = own_username
        self.model = model
        self.allow_vote = allow_vote
        self.allow_lang = allow_lang
        self.allow_pii = allow_pii
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
