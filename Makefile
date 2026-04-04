# Makefile for TheColony.cc Analyzer

VENV_DIR = colony_venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
SCRIPT = colony_analyzer.py

.PHONY: all
all: setup run

.PHONY: setup
setup:
	@echo "🔧 Setting up virtual environment..."
	@test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	@$(PIP) install --upgrade pip
	@$(PIP) install requests
	@echo "✅ Virtual environment ready"

.PHONY: run
run: setup
	@echo "🚀 Running TheColony.cc Analyzer..."
	@$(PYTHON) $(SCRIPT)

.PHONY: scan
scan: setup
	@echo "📡 Scanning recent posts..."
	@$(PYTHON) $(SCRIPT) --days 7 --limit 15

.PHONY: force
force: setup
	@echo "🔄 Force re-analysis..."
	@$(PYTHON) $(SCRIPT) --force

.PHONY: no-vote
no-vote: setup
	@echo "🔍 Analysis only (no voting)..."
	@$(PYTHON) $(SCRIPT) --no-vote

.PHONY: clean
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf $(VENV_DIR)
	@rm -f colony_analyzed.json colony_config.json
	@echo "✅ Cleaned"

.PHONY: help
help:
	@echo "Usage:"
	@echo "  make          - Setup + run"
	@echo "  make setup    - Create venv"
	@echo "  make scan     - Scan recent posts"
	@echo "  make force    - Force re-analyze"
	@echo "  make no-vote  - Analysis only"
	@echo "  make clean    - Remove venv"
