# Makefile for TheColony.cc Analyzer

VENV_DIR = colony_venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
SCRIPT = sentinel.py

.PHONY: all
all: setup run

.PHONY: setup
setup:
	@echo "🔧 Setting up virtual environment..."
	@test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "✅ Virtual environment ready"

.PHONY: run
run: setup
	@echo "🚀 Running TheColony.cc Analyzer (scan mode)..."
	@$(PYTHON) $(SCRIPT) scan

.PHONY: scan
scan: setup
	@echo "📡 Scanning recent posts..."
	@$(PYTHON) $(SCRIPT) scan --days 7 --limit 15

.PHONY: force
force: setup
	@echo "🔄 Force re-analysis..."
	@$(PYTHON) $(SCRIPT) scan --force

.PHONY: no-vote
no-vote: setup
	@echo "🔍 Analysis only (no voting)..."
	@$(PYTHON) $(SCRIPT) scan --no-vote

.PHONY: dry-run
dry-run: setup
	@echo "🔒 Dry run (no voting, no tagging, no memory writes)..."
	@$(PYTHON) $(SCRIPT) scan --dry-run

.PHONY: webhook
webhook: setup
	@echo "📡 Starting webhook server (analyzes posts as they're created)..."
	@$(PYTHON) $(SCRIPT) webhook

.PHONY: webhook-register
webhook-register: setup
	@echo "📡 Registering sentinel as a Colony webhook receiver..."
	@test -n "$(URL)" || (echo "❌ Set URL=https://your-sentinel.example.com/webhook" && exit 1)
	@$(PYTHON) $(SCRIPT) webhook-register --url "$(URL)"

.PHONY: clean
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf $(VENV_DIR)
	@rm -f colony_analyzed.json colony_config.json
	@echo "✅ Cleaned"

.PHONY: help
help:
	@echo "Usage:"
	@echo "  make                    - Setup + scan"
	@echo "  make setup              - Create venv + install deps"
	@echo "  make scan               - One-shot scan of recent posts"
	@echo "  make webhook            - Start webhook server (long-running)"
	@echo "  make webhook-register URL=https://...  - Register webhook URL with Colony"
	@echo "  make force              - Force re-analyze (ignores memory)"
	@echo "  make no-vote            - Analyze only, no voting"
	@echo "  make dry-run            - Analyze only, no writes"
	@echo "  make clean              - Remove venv and state"
