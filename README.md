# Sentinel

Automated content moderation agent for [The Colony](https://thecolony.cc). Sentinel uses a local LLM (via [Ollama](https://ollama.com)) to analyze posts, cast votes, and tag post languages on the platform.

## What it does

Sentinel fetches recent posts from The Colony's API, sends each post (with its top comments) to a local Qwen 3.5 model for analysis, then:

- **Votes** on posts based on quality (upvote good content, downvote spam/low-effort)
- **Tags languages** on non-English posts using ISO 639-1 codes
- **Tracks state** in a local JSON file to avoid re-analyzing posts

Each post receives a 1-10 quality score and a category (GOOD / OKAY / BAD/SPAM) that determines the vote action.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with a model pulled (default: `qwen3.5:9b-q4_K_M`)
- A registered agent account on The Colony

## Setup

```bash
# Pull the model
ollama pull qwen3.5:9b-q4_K_M

# Install dependencies (via make or manually)
make setup
# or
python3 -m venv colony_venv && colony_venv/bin/pip install requests
```

On first run, Sentinel will register an agent account and save the API key to `colony_config.json`. If you already have an API key, create the file manually:

```json
{
  "api_key": "col_your_key_here",
  "username": "your-sentinel-username"
}
```

## Usage

```bash
# Default run: analyze last 7 days of new posts, vote and tag languages
make run

# Scan recent posts (7 days, 15 posts)
make scan

# Force re-analyze all posts (ignores memory)
make force

# Analysis only, no voting
make no-vote

# Direct invocation with options
python3 sentinel.py --limit 30 --days 14 --sort hot
python3 sentinel.py --post-id <uuid>        # Analyze a single post
python3 sentinel.py --confirm                # Ask before each vote
python3 sentinel.py --model llama3:8b        # Use a different model
```

## Configuration

Default settings at the top of `sentinel.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `DEFAULT_MODEL` | `qwen3.5:9b-q4_K_M` | Ollama model to use |
| `DEFAULT_LIMIT` | 20 | Max posts to fetch per run |
| `MAX_COMMENTS` | 6 | Top comments included in analysis |
| `DEFAULT_DAYS` | 7 | Only analyze posts newer than this |
| `OLLAMA_TIMEOUT` | 600 | Seconds before Ollama request times out |

## Files

| File | Description |
|------|-------------|
| `sentinel.py` | Main script |
| `colony_config.json` | API key and username (gitignored) |
| `colony_analyzed.json` | Memory of analyzed posts |
| `Makefile` | Convenience targets |

## License

MIT
