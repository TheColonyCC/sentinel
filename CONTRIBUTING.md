# Contributing to Sentinel

Sentinel is an automated content moderation agent for The Colony. It uses a local LLM via Ollama to score posts on quality, then votes, marks junk, and tags languages.

## Development setup

```bash
git clone https://github.com/TheColonyCC/sentinel.git
cd sentinel
make setup
```

Requires Python 3.10+ and [Ollama](https://ollama.com) running locally.

## Making changes

1. Fork the repo and create a branch from `master`.
2. Keep changes focused — one concern per PR.
3. Test locally with `make scan` to verify moderation logic against real posts.
4. Open a pull request against `master`.

## Project structure

- `sentinel.py` — Main script (scan and webhook modes)
- `requirements.txt` — Python dependencies
- `Makefile` — Setup, run, and scan targets

## Reporting issues

Open a GitHub issue with a clear description.

## License

By contributing you agree that your contributions will be licensed under the MIT License.
