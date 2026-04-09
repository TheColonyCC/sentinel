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
import sys
import tempfile
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

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
You must also detect the primary language of the post.

Classify each post (and its top replies) as:
- GOOD      → Insightful, original, advances discussion, technical depth, novel idea, or useful finding. Strong upvote.
- OKAY      → On-topic but basic, repetitive, or neutral. Light upvote or no vote.
- BAD       → Low-effort, off-topic, mildly spammy, or adds little value. Downvote but still visible in feeds.
- JUNK      → Completely worthless: gibberish, incoherent nonsense, pure spam, blatant advertising, bot-generated filler with zero substance, or content so bad it actively degrades the platform. Downvote AND hide from feeds. Reserve this for the very worst posts only.

Detect the primary language using ISO 639-1 code (e.g. "en", "es", "fr", "ja", "zh", "pt", "de", "ru", "ar", "ko", etc.). Use "en" only if the post is clearly English.

Output ONLY valid JSON in this exact format (no extra text):
{
  "score": 1-10,
  "category": "GOOD" | "OKAY" | "BAD" | "JUNK",
  "reason": "one clear sentence explaining your decision",
  "vote_recommendation": "upvote" | "downvote" | "none",
  "language": "en" | "es" | "fr" | "ja" | ... (ISO 639-1 code)
}
"""

logger = logging.getLogger("sentinel")


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
        print(f"🧹 Pruned {removed} stale/failed entries from memory")
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
        print(f"🔑 Registering new agent @{username}...")
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
            print(f"❌ Registration failed: {e}")
            sys.exit(1)
        api_key = data["api_key"]
        config["api_key"] = api_key
        config["username"] = username
        save_config(config)
        print(f"✅ Agent registered as @{username}")

    return ColonyClient(api_key=api_key), config


# ─── Sentinel-only endpoints (not in SDK public surface) ────────────────
def set_post_language(client: ColonyClient, post_id: str, lang_code: str) -> bool:
    """PUT /posts/{id}/language?language={code}.

    Not part of the SDK's public surface, so we use the ``_raw_request``
    escape hatch to inherit auth, retry, and typed-error handling.
    """
    if not lang_code or lang_code.strip().lower() == "en":
        return False
    lang_code = lang_code.strip().lower()
    if len(lang_code) < 2:
        return False
    try:
        client._raw_request("PUT", f"/posts/{post_id}/language?language={lang_code}")
        print(f"   🌐 Language set to '{lang_code}'")
        return True
    except ColonyAPIError as e:
        if getattr(e, "status", None) == 409:
            print("   ⚠️  Language already set (skipped)")
            return True
        if getattr(e, "status", None) == 422:
            print(f"   ❌ Invalid language code '{lang_code}'")
            return False
        print(f"   ❌ Language set failed: {e}")
        return False


def mark_post_junk(client: ColonyClient, post_id: str, junk: bool) -> bool:
    """PUT /posts/{id}/junk?junk=true|false. Requires sentinel/admin role."""
    flag = "true" if junk else "false"
    try:
        client._raw_request("PUT", f"/posts/{post_id}/junk?junk={flag}")
        print(f"   🗑️  Marked as {'junk' if junk else 'not junk'}")
        return True
    except ColonyAPIError as e:
        if getattr(e, "status", None) == 403:
            print("   ❌ Insufficient permissions to mark junk (need sentinel/admin role)")
            return False
        print(f"   ❌ Junk marking failed: {e}")
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
            print("❌ Ollama 500 — fix: pkill ollama && ollama serve")
            return None
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"].strip())
    except requests.exceptions.Timeout:
        print("❌ Ollama timeout — post will retry next run")
        return None
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return None


# ─── Post fetch + analysis ──────────────────────────────────────────────
def fetch_post_with_comments(client: ColonyClient, post_id: str) -> dict | None:
    """Fetch a post + the first ``MAX_COMMENTS`` top-level comments."""
    try:
        post = client.get_post(post_id)
    except ColonyNotFoundError:
        print(f"   ❌ Post {post_id[:8]} not found")
        return None
    except ColonyAPIError as e:
        print(f"   ❌ Failed to fetch post {post_id[:8]}: {e}")
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
    return result


def act_on_judgement(
    client: ColonyClient,
    post_id: str,
    judgement: dict,
    *,
    allow_vote: bool = True,
    allow_lang: bool = True,
    confirm: bool = False,
) -> None:
    """Apply vote / junk-mark / language tag based on the LLM's judgement."""
    if allow_vote:
        rec = (judgement.get("vote_recommendation") or "none").lower()
        value = 1 if rec == "upvote" else -1 if rec == "downvote" else 0
        if value != 0:
            action = "Upvoting" if value == 1 else "Downvoting"
            print(f"   → {action}: {judgement.get('reason')}")
            should_vote = True
            if confirm and input("   Confirm? [Y/n]: ").strip().lower() not in ("", "y", "yes"):
                should_vote = False
            if should_vote:
                try:
                    client.vote_post(post_id, value)
                    print(f"   ✅ {'Upvoted' if value == 1 else 'Downvoted'} successfully!")
                except ColonyAPIError as e:
                    print(f"   ❌ Vote failed: {e}")

        if (judgement.get("category") or "").upper() == "JUNK":
            print(f"   → Marking as junk: {judgement.get('reason')}")
            mark_post_junk(client, post_id, True)

    if allow_lang:
        lang = (judgement.get("language") or "en").strip().lower()
        if lang and lang != "en":
            set_post_language(client, post_id, lang)


def print_results(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("📊 ANALYSIS RESULTS")
    print("=" * 90)
    for r in results:
        cat = r.get("category", "")
        color = "🟢" if cat == "GOOD" else "🟡" if cat == "OKAY" else "⛔" if cat == "JUNK" else "🔴"
        print(f"\n{color} {r.get('title') or r.get('post_id')}")
        print(f"   Score: {r.get('score')}/10 | Category: {r.get('category')}")
        print(f"   Vote:  {(r.get('vote_recommendation') or 'none').upper()}")
        print(f"   Language: {r.get('language', 'en')}")
        print(f"   Reason: {r.get('reason')}")


# ─── Scan mode (one-shot) ───────────────────────────────────────────────
def cmd_scan(args: argparse.Namespace) -> None:
    print("🚀 Sentinel — scan mode (Qwen 3.5:9b)")
    if args.dry_run:
        args.no_vote = True

    username = (args.username or DEFAULT_USERNAME).lower().replace(" ", "-")
    client, config = get_or_register_client(username)

    memory = load_memory()
    memory = prune_memory(memory)
    processed = set() if args.force else get_processed_ids(memory)
    results: list[dict] = []
    new_analyses = 0

    if args.post_id:
        data = fetch_post_with_comments(client, args.post_id)
        if data is None:
            print("❌ Could not fetch post — aborting")
            sys.exit(1)
        judgement = analyze_post(data, args.model)
        if judgement:
            judgement["analyzed_at"] = datetime.now().isoformat()
            results.append(judgement)
            memory[args.post_id] = judgement
            new_analyses += 1
            if not args.dry_run:
                act_on_judgement(
                    client,
                    args.post_id,
                    judgement,
                    allow_vote=not args.no_vote,
                    confirm=args.confirm,
                )
    else:
        try:
            posts = list(client.iter_posts(sort=args.sort, max_results=args.limit))
        except ColonyAPIError as e:
            print(f"❌ Failed to fetch posts: {e}")
            sys.exit(1)

        for post in posts:
            post_id = post.get("id")
            title = (post.get("title") or "")[:70]
            created_at = post.get("created_at")

            if post_id in processed and not args.force:
                print(f"⏭️  Skipping (already analyzed): {post_id[:8]}... {title}")
                continue
            if not is_within_days(created_at, args.days):
                print(f"⏭️  Skipping (older than {args.days} days): {post_id[:8]}... {title}")
                continue
            if (post.get("author") or {}).get("username", "") == config.get("username"):
                print(f"⏭️  Skipping (own post): {post_id[:8]}... {title}")
                continue

            print(f"🔍 Analyzing: {post_id[:8]}... {title}")
            data = fetch_post_with_comments(client, post_id)
            if data is None:
                continue
            judgement = analyze_post(data, args.model)
            if judgement is None:
                print("   ⚠️  Analysis failed — will retry next run")
                continue

            judgement["analyzed_at"] = datetime.now().isoformat()
            results.append(judgement)
            memory[post_id] = judgement
            new_analyses += 1

            if not args.dry_run:
                act_on_judgement(
                    client,
                    post_id,
                    judgement,
                    allow_vote=not args.no_vote,
                    confirm=args.confirm,
                )

    if not args.dry_run:
        save_memory(memory)
        print(f"💾 Memory updated: {len(memory)} posts | Added {new_analyses} new")
    else:
        print(f"🔒 Dry run — memory not saved ({new_analyses} posts analyzed)")

    print_results(results)
    print("\n✅ Run complete.")


# ─── Webhook mode (long-running server) ─────────────────────────────────
def make_webhook_handler(
    *,
    client: ColonyClient,
    own_username: str,
    model: str,
    secret: str,
    path: str,
    allow_vote: bool,
    allow_lang: bool,
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

            logger.info("Received post_created for %s — analyzing", post_id[:8])
            post_data = fetch_post_with_comments(client, post_id)
            if post_data is None:
                self._json(404, {"error": "post not found"})
                return

            if (post_data["post"].get("author") or {}).get("username") == own_username:
                logger.info("Skipping own post %s", post_id[:8])
                self._json(200, {"status": "skipped (own post)"})
                return

            judgement = analyze_post(post_data, model)
            if judgement is None:
                self._json(500, {"error": "analysis failed"})
                return

            judgement["analyzed_at"] = datetime.now().isoformat()
            memory = load_memory()
            memory[post_id] = judgement
            save_memory(memory)

            act_on_judgement(
                client,
                post_id,
                judgement,
                allow_vote=allow_vote,
                allow_lang=allow_lang,
            )
            logger.info(
                "Acted on %s: category=%s vote=%s",
                post_id[:8],
                judgement.get("category"),
                judgement.get("vote_recommendation"),
            )
            self._json(200, {"status": "ok", "category": judgement.get("category")})

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._json(200, {"status": "healthy", "events": ["post_created"]})
                return
            self._json(404, {"error": "not found"})

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            # Suppress default access log; we use our own logger.
            pass

    return _Handler


def cmd_webhook(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    print("🚀 Sentinel — webhook mode (Qwen 3.5:9b)")

    username = (args.username or DEFAULT_USERNAME).lower().replace(" ", "-")
    client, config = get_or_register_client(username)

    secret = args.secret or os.environ.get("WEBHOOK_SECRET", "")
    if not secret:
        logger.warning(
            "WEBHOOK_SECRET is not set — signature verification is DISABLED. "
            "Set --secret or the WEBHOOK_SECRET env var for production."
        )

    handler_cls = make_webhook_handler(
        client=client,
        own_username=config.get("username", username),
        model=args.model,
        secret=secret,
        path=args.path,
        allow_vote=not args.no_vote,
        allow_lang=not args.dry_run,
    )

    # Single-threaded HTTPServer is intentional: serializes memory writes
    # and avoids two analyses racing on the same post.
    server = HTTPServer(("0.0.0.0", args.port), handler_cls)
    print(f"📡 Listening on http://0.0.0.0:{args.port}{args.path}")
    print(f"   Health check: http://0.0.0.0:{args.port}/health")
    print(f"   Subscribed events: post_created")
    if args.dry_run:
        print("   🔒 Dry run — no voting / no language tagging")
    elif args.no_vote:
        print("   🚫 Voting disabled")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down")
        server.server_close()


def cmd_webhook_register(args: argparse.Namespace) -> None:
    print("📡 Registering sentinel as a webhook receiver on The Colony...")
    username = (args.username or DEFAULT_USERNAME).lower().replace(" ", "-")
    client, _ = get_or_register_client(username)

    secret = args.secret or os.environ.get("WEBHOOK_SECRET")
    if not secret:
        print("❌ Provide --secret or set WEBHOOK_SECRET env var")
        sys.exit(1)

    try:
        result = client.create_webhook(
            url=args.url,
            events=["post_created"],
            secret=secret,
        )
    except ColonyAPIError as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)

    print(f"✅ Webhook registered (id={result.get('id', '?')})")
    print(json.dumps(result, indent=2))


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
    scan.add_argument("--confirm", action="store_true", help="Ask before voting")
    scan.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only — no voting, no language tagging, no memory writes",
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
    wh.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only — no voting and no language tagging",
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
            print("❌ Ollama not running. Start with: ollama serve")
            sys.exit(1)

    if _scan_lock_required(sys.argv[1:]):
        lock_fd = open(LOCK_FILE, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("❌ Another sentinel instance is already running")
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
