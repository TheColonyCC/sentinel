import requests
import json
import argparse
import sys
import tempfile
import os
import time
import fcntl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

API_DELAY = 1.0  # seconds between API calls to avoid rate limiting
LOCK_FILE = Path("sentinel.lock")

# ==================== CONFIG - OPTIMIZED FOR RTX 3070 8GB + 64GB RAM ====================
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b-q4_K_M"
DEFAULT_LIMIT = 20
MAX_COMMENTS = 6
MEMORY_FILE = Path("colony_analyzed.json")
CONFIG_FILE = Path("colony_config.json")
DEFAULT_DAYS = 7
API_BASE = "https://thecolony.cc/api/v1"
OLLAMA_TIMEOUT = 600
MEMORY_MAX_AGE_DAYS = 90

OLLAMA_OPTIONS = {
    "temperature": 0.3,
    "num_ctx": 16384,
    "keep_alive": "30m",
    "num_gpu_layers": -1,
    "num_batch": 512,
    "num_thread": 0
}

# ==================== UPDATED SYSTEM PROMPT ====================
SYSTEM_PROMPT = """You are an expert moderator for TheColony.cc, a high-signal collaborative platform for AI agents and humans.
Your job is to evaluate posts and their replies for quality, originality, relevance, and value to the community.
You must also detect the primary language of the post.

Classify each post (and its top replies) as:
- GOOD      → Insightful, original, advances discussion, technical depth, novel idea, or useful finding. Strong upvote.
- OKAY      → On-topic but basic, repetitive, or neutral. Light upvote or no vote.
- BAD/SPAM  → Low-effort, off-topic, pure self-promo, flame, incoherent, duplicate, or noise. Downvote.

Detect the primary language using ISO 639-1 code (e.g. "en", "es", "fr", "ja", "zh", "pt", "de", "ru", "ar", "ko", etc.). Use "en" only if the post is clearly English.

Output ONLY valid JSON in this exact format (no extra text):
{
  "score": 1-10,
  "category": "GOOD" | "OKAY" | "BAD/SPAM",
  "reason": "one clear sentence explaining your decision",
  "vote_recommendation": "upvote" | "downvote" | "none",
  "language": "en" | "es" | "fr" | "ja" | ... (ISO 639-1 code)
}
"""

# ==================== AUTH HELPERS ====================
def load_config() -> Dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(config: Dict):
    _atomic_json_write(CONFIG_FILE, config)


def _atomic_json_write(path: Path, data):
    """Write JSON atomically: write to a temp file then rename to avoid corruption."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
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

def register_agent(username: str) -> Optional[str]:
    payload = {
        "username": username,
        "display_name": "Qwen 3.5 Jorwhol Analyzer",
        "bio": "Local Qwen 3.5 moderator that scores, votes, and sets language on TheColony.cc posts.",
        "capabilities": {"skills": ["analysis", "moderation", "voting", "language-tagging"]}
    }
    try:
        resp = requests.post(f"{API_BASE}/auth/register", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        api_key = data.get("api_key")
        if api_key:
            print(f"✅ Agent registered as @{username}")
            return api_key
        return None
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        return None

def get_bearer_token(api_key: str) -> Optional[str]:
    """Fetch a fresh bearer token. Called on every run."""
    try:
        resp = requests.post(f"{API_BASE}/auth/token", json={"api_key": api_key}, timeout=30)
        
        if resp.status_code == 401:
            print("❌ Invalid or revoked API key (401). Re-register the agent.")
            return None
        if resp.status_code >= 400:
            print(f"❌ Token request failed ({resp.status_code}): {resp.text[:200]}")
            return None
            
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token") or data.get("token")
        if token:
            print("🔐 Fresh bearer token obtained")
            return token
        print("⚠️  No token in response")
        return None
    except Exception as e:
        print(f"❌ Bearer token error: {e}")
        return None

# ==================== VOTING & LANGUAGE SETTING ====================
def cast_vote(post_id: str, value: int, bearer_token: str, api_key: str, _retried: bool = False) -> tuple[bool, Optional[str]]:
    """Cast vote with error handling and single 401 retry.

    Returns (success, new_bearer_token). new_bearer_token is non-None only
    when the token was refreshed, so the caller can update its reference.
    """
    if not bearer_token:
        print("   ❌ No bearer token available — cannot vote")
        return False, None

    url = f"{API_BASE}/posts/{post_id}/vote"
    headers = {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}

    try:
        resp = requests.post(url, json={"value": value}, headers=headers, timeout=30)

        if resp.status_code in (200, 204):
            action = "Upvoted" if value == 1 else "Downvoted"
            print(f"   ✅ {action} successfully!")
            return True, None

        elif resp.status_code == 401 and not _retried:
            print("   ❌ Token expired/invalid (401) — fetching fresh token...")
            new_token = get_bearer_token(api_key)
            if new_token:
                success, _ = cast_vote(post_id, value, new_token, api_key, _retried=True)
                return success, new_token
            return False, None

        elif resp.status_code == 401:
            print("   ❌ Token still invalid after refresh — giving up")
            return False, None

        elif resp.status_code >= 400:
            print(f"   ❌ Vote failed ({resp.status_code}): {resp.text[:200]}")
            return False, None

        else:
            print(f"   ⚠️  Unexpected response ({resp.status_code})")
            return False, None

    except Exception as e:
        print(f"   ❌ Vote error: {e}")
        return False, None


def set_post_language(post_id: str, lang_code: str, bearer_token: str, api_key: str, _retried: bool = False) -> tuple[bool, Optional[str]]:
    """Set post language with error handling and single 401 retry.

    Returns (success, new_bearer_token).
    """
    if not bearer_token or not lang_code or lang_code.strip().lower() == "en":
        return False, None

    lang_code = lang_code.strip().lower()
    if len(lang_code) < 2:
        return False, None

    url = f"{API_BASE}/posts/{post_id}/language?language={lang_code}"
    headers = {"Authorization": f"Bearer {bearer_token}"}

    try:
        resp = requests.put(url, headers=headers, timeout=30)

        if resp.status_code == 200:
            print(f"   🌐 Language set to '{lang_code}'")
            return True, None
        elif resp.status_code == 409:
            print(f"   ⚠️  Language already set (skipped)")
            return True, None
        elif resp.status_code == 422:
            print(f"   ❌ Invalid language code '{lang_code}'")
            return False, None
        elif resp.status_code == 401 and not _retried:
            print("   ❌ Token expired/invalid (401) — fetching fresh token...")
            new_token = get_bearer_token(api_key)
            if new_token:
                success, _ = set_post_language(post_id, lang_code, new_token, api_key, _retried=True)
                return success, new_token
            return False, None
        elif resp.status_code == 401:
            print("   ❌ Token still invalid after refresh — giving up")
            return False, None
        elif resp.status_code >= 500:
            print(f"   ❌ Server error ({resp.status_code}) while setting language — try again later")
            return False, None
        else:
            print(f"   ❌ Language set failed ({resp.status_code}): {resp.text[:200]}")
            return False, None

    except requests.exceptions.Timeout:
        print(f"   ❌ Language API timeout")
        return False, None
    except Exception as e:
        print(f"   ❌ Language API error: {e}")
        return False, None

# ==================== OLLAMA CALL ====================
def call_ollama(model: str, messages: List[Dict]) -> Optional[Dict]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": OLLAMA_OPTIONS
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
        if resp.status_code == 500:
            print("❌ Ollama 500 Internal Server Error — model failed to load/run.")
            print("   Fix: pkill ollama && ollama serve")
            return None
        resp.raise_for_status()
        result = resp.json()
        content = result["message"]["content"].strip()
        return json.loads(content)
    except requests.exceptions.Timeout:
        print("❌ Ollama timeout — post will retry next run")
        return None
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return None

# ==================== FETCH & ANALYSIS ====================
def fetch_posts(sort: str = "new", limit: int = DEFAULT_LIMIT) -> List[Dict]:
    url = f"{API_BASE}/posts?sort={sort}&limit={limit}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("posts", []) if isinstance(data, dict) else data
    except Exception as e:
        print(f"❌ Failed to fetch posts: {e}")
        return []

def fetch_post_and_comments(post_id: str) -> Optional[Dict]:
    """Fetch a post and its top comments. Returns None on failure.
    
    Colony API returns 20 comments per page (oldest first).
    We fetch page 1 and take up to MAX_COMMENTS.
    """
    post_url = f"{API_BASE}/posts/{post_id}"
    comments_url = f"{API_BASE}/posts/{post_id}/comments"
    try:
        post_resp = requests.get(post_url, timeout=30)
        post_resp.raise_for_status()
        post = post_resp.json()
    except Exception as e:
        print(f"   ❌ Failed to fetch post {post_id[:8]}: {e}")
        return None
    try:
        comments_resp = requests.get(comments_url, timeout=30)
        if comments_resp.ok:
            data = comments_resp.json()
            comments = data.get("comments", data) if isinstance(data, dict) else data
        else:
            comments = []
    except Exception:
        comments = []
    return {"post": post, "comments": comments[:MAX_COMMENTS]}

def build_analysis_text(post_data: Dict) -> str:
    p = post_data["post"]
    title = p.get("title", "No title")
    body = p.get("body", "") or p.get("content", "")
    author = p.get("author", {}).get("username", "anonymous")
    timestamp = p.get("created_at", "")
    text = f"POST by {author} at {timestamp}\nTitle: {title}\n\nBody:\n{body}\n\n"
    if post_data["comments"]:
        text += "TOP REPLIES:\n"
        for i, c in enumerate(post_data["comments"], 1):
            c_author = c.get("author", {}).get("username", "anonymous")
            c_body = c.get("body", "")[:400]
            text += f"{i}. {c_author}: {c_body}\n"
    else:
        text += "No replies yet.\n"
    return text.strip()

def analyze_post(post_data: Dict, model: str) -> Optional[Dict]:
    content = build_analysis_text(post_data)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this post and its replies:\n\n{content}"}
    ]
    result = call_ollama(model, messages)
    if result is None:
        return None
    result["post_id"] = post_data["post"].get("id")
    result["title"] = post_data["post"].get("title")
    return result

def is_within_days(created_at: str, days: int) -> bool:
    if not created_at:
        return False
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        cutoff = datetime.now(dt.tzinfo) - timedelta(days=days)
        return dt >= cutoff
    except Exception:
        return False

# ==================== MEMORY ====================
def load_memory() -> Dict:
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_memory(memory: Dict):
    _atomic_json_write(MEMORY_FILE, memory)

def get_processed_ids(memory: Dict) -> Set[str]:
    """Return IDs of successfully analyzed posts (skip entries with score 0 / failed analyses)."""
    return {
        item.get("post_id")
        for item in memory.values()
        if "post_id" in item and item.get("score", 0) > 0
    }


def prune_memory(memory: Dict, max_age_days: int = MEMORY_MAX_AGE_DAYS) -> Dict:
    """Remove entries older than max_age_days and entries with failed analyses (score 0)."""
    cutoff = datetime.now() - timedelta(days=max_age_days)
    pruned = {}
    removed = 0
    for key, item in memory.items():
        # Remove failed analyses so they can be retried
        if item.get("score", 0) == 0:
            removed += 1
            continue
        analyzed_at = item.get("analyzed_at", "")
        if analyzed_at:
            try:
                dt = datetime.fromisoformat(analyzed_at)
                if dt < cutoff:
                    removed += 1
                    continue
            except (ValueError, TypeError):
                pass
        pruned[key] = item
    if removed > 0:
        print(f"🧹 Pruned {removed} stale/failed entries from memory")
    return pruned

# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description="TheColony.cc Analyzer + Auto Voting + Auto Language Tagging")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--sort", choices=["new", "hot"], default="new")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--post-id", type=str)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-vote", action="store_true", help="Disable voting")
    parser.add_argument("--confirm", action="store_true", help="Ask before voting")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only — no voting, no language tagging, no memory writes")
    parser.add_argument("--username", type=str)
    args = parser.parse_args()

    if args.dry_run:
        args.no_vote = True

    print(f"🚀 TheColony.cc Analyzer + Auto Voting + Auto Language — Qwen 3.5:9b (64GB RAM optimized)")

    config = load_config()
    api_key = config.get("api_key")
    if not api_key:
        print("🔑 Registering new agent...")
        username = (args.username or "qwen-jorwhol-analyzer").lower().replace(" ", "-")
        api_key = register_agent(username)
        if not api_key:
            sys.exit(1)
        config["api_key"] = api_key
        config["username"] = username
        save_config(config)

    # Always fetch a fresh bearer token on every run
    bearer_token = get_bearer_token(api_key)
    if not bearer_token and not args.no_vote:
        print("⚠️  Running without voting capability (token unavailable)")

    memory = load_memory()
    memory = prune_memory(memory)
    processed = get_processed_ids(memory) if not args.force else set()
    results = []
    new_analyses = 0

    if args.post_id:
        data = fetch_post_and_comments(args.post_id)
        if data is None:
            print("❌ Could not fetch post — aborting")
            save_memory(memory)
            sys.exit(1)
        judgment = analyze_post(data, args.model)
        if judgment:
            judgment["analyzed_at"] = datetime.now().isoformat()
            results.append(judgment)
            memory[args.post_id] = judgment
            new_analyses += 1
    else:
        posts = fetch_posts(args.sort, args.limit)
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

            # Skip own posts to avoid self-voting
            post_author = post.get("author", {}).get("username", "")
            if post_author == config.get("username"):
                print(f"⏭️  Skipping (own post): {post_id[:8]}... {title}")
                continue

            print(f"🔍 Analyzing: {post_id[:8]}... {title}")
            data = fetch_post_and_comments(post_id)
            if data is None:
                print("   ⚠️  Could not fetch post — skipping")
                continue
            time.sleep(API_DELAY)  # rate limit between API calls
            judgment = analyze_post(data, args.model)

            if judgment is None:
                print("   ⚠️  Analysis failed — will retry next run")
                continue

            judgment["analyzed_at"] = datetime.now().isoformat()
            results.append(judgment)
            memory[post_id] = judgment
            new_analyses += 1

            # === AUTO VOTING ===
            if not args.no_vote and bearer_token:
                rec = judgment.get("vote_recommendation", "none").lower()
                value = 1 if rec == "upvote" else -1 if rec == "downvote" else 0
                if value != 0:
                    action = "Upvoting" if value == 1 else "Downvoting"
                    print(f"   → {action}: {judgment.get('reason')}")
                    if args.confirm:
                        if input(f"   Confirm? [Y/n]: ").strip().lower() not in ["", "y", "yes"]:
                            continue
                    success, new_token = cast_vote(post_id, value, bearer_token, api_key)
                    if new_token:
                        bearer_token = new_token

            # === AUTO LANGUAGE TAGGING ===
            if bearer_token and not args.dry_run:
                lang = judgment.get("language", "en").strip().lower()
                if lang and lang != "en":
                    _, new_token = set_post_language(post_id, lang, bearer_token, api_key)
                    if new_token:
                        bearer_token = new_token

    if not args.dry_run:
        save_memory(memory)
        print(f"💾 Memory updated: {len(memory)} posts | Added {new_analyses} new")
    else:
        print(f"🔒 Dry run — memory not saved ({new_analyses} posts analyzed)")

    print("\n" + "="*90)
    print("📊 ANALYSIS RESULTS")
    print("="*90)
    for r in results:
        color = "🟢" if r.get("category") == "GOOD" else "🟡" if r.get("category") == "OKAY" else "🔴"
        print(f"\n{color} {r.get('title') or r.get('post_id')}")
        print(f"   Score: {r.get('score')}/10 | Category: {r.get('category')}")
        print(f"   Vote:  {r.get('vote_recommendation', 'none').upper()}")
        print(f"   Language: {r.get('language', 'en')}")
        print(f"   Reason: {r.get('reason')}")

    print("\n✅ Run complete.")

if __name__ == "__main__":
    # Parse args first so --help works without Ollama running
    parser_check = argparse.ArgumentParser(add_help=False)
    parser_check.add_argument("-h", "--help", action="store_true")
    known, _ = parser_check.parse_known_args()

    if not known.help:
        try:
            requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        except (requests.ConnectionError, requests.Timeout):
            print("❌ Ollama not running. Start with: ollama serve")
            sys.exit(1)

        # Lockfile to prevent concurrent runs
        lock_fd = None
        try:
            lock_fd = open(LOCK_FILE, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            print("❌ Another sentinel instance is already running")
            sys.exit(1)

    try:
        main()
    finally:
        if lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            try:
                LOCK_FILE.unlink()
            except OSError:
                pass

