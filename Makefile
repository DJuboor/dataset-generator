.DEFAULT_GOAL := help
SHELL := /bin/bash

# Optional overrides: SAMPLES=1000 WORKERS=20 CONFIG=my_task.yaml FORMAT=csv
SAMPLES ?=
WORKERS ?=
CONFIG  ?= config.yaml
FORMAT  ?= jsonl
OUTPUT  ?=
TASK    ?=
LABELS  ?=
DOMAIN  ?=
MODEL   ?=
STRATEGY ?=
DOCS    ?=
SEED    ?=
LANG    ?=

_samples  = $(if $(SAMPLES),-n $(SAMPLES))
_workers  = $(if $(WORKERS),-w $(WORKERS))
_config   = -c $(CONFIG)
_format   = $(if $(filter-out jsonl,$(FORMAT)),-f $(FORMAT))
_output   = $(if $(OUTPUT),-o $(OUTPUT))
_task     = $(if $(TASK),-t $(TASK))
_labels   = $(if $(LABELS),--labels "$(LABELS)")
_domain   = $(if $(DOMAIN),-d "$(DOMAIN)")
_model    = $(if $(MODEL),-m $(MODEL))
_strategy = $(if $(STRATEGY),-s $(STRATEGY))
_docs     = $(if $(DOCS),--from-docs $(DOCS))
_seed     = $(if $(SEED),--seed-from $(SEED))
_lang     = $(if $(LANG),--language $(LANG))

# ── Workflow ─────────────────────────────────────────────────────────
.PHONY: setup init generate validate export push estimate

setup:       ## Install dependencies
	@uv sync --all-extras

init:        ## Scaffold config ─ TASK=classification LABELS="pos,neg"
	@uv run dg init $(or $(TASK),classification) $(if $(LABELS),-l "$(LABELS)") $(_output)

generate:    ## Generate dataset ─ TASK=sft DOMAIN="coding" SAMPLES=1000
	@uv run dg generate $(if $(TASK),$(_task),$(_config)) $(_samples) $(_workers) $(_output) $(_format) $(_domain) $(_labels) $(_model) $(_strategy) $(_docs) $(_seed) $(_lang)

estimate:    ## Dry-run: estimate cost ─ same flags as generate
	@uv run dg generate $(if $(TASK),$(_task),$(_config)) $(_samples) $(_workers) $(_domain) $(_labels) $(_model) --dry-run

validate:    ## Deduplicate + validate ─ FILE=data/output.jsonl
	@uv run dg validate $(or $(FILE),data/output.jsonl)

export:      ## Export to another format ─ FILE=data/output.jsonl FORMAT=csv
	@uv run dg export $(or $(FILE),data/output.jsonl) $(_format) $(_output)

push:        ## Push to HuggingFace Hub ─ REPO=user/dataset FILE=data/output.jsonl
	@uv run dg push $(REPO) $(if $(FILE),-f $(FILE))

# ── Dev ──────────────────────────────────────────────────────────────
.PHONY: quality format test clean

quality:     ## Lint + format check + tests
	@uv run ruff check src/ tests/
	@uv run ruff format --check src/ tests/
	@uv run pytest

format:      ## Auto-fix lint + formatting
	@uv run ruff format src/ tests/
	@uv run ruff check --fix src/ tests/

test:        ## Run tests only
	@uv run pytest

clean:       ## Remove build artifacts
	@rm -rf dist/ build/ *.egg-info .ruff_cache .pytest_cache .coverage htmlcov/ .dg_checkpoints/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── Help ─────────────────────────────────────────────────────────────
.PHONY: help
help:
	@printf "\n  \033[1mdataset-generator\033[0m — generate synthetic ML datasets with any LLM\n\n"
	@printf "  \033[33mWorkflow\033[0m\n"
	@printf "  \033[36m%-16s\033[0m %s\n" "make setup"      "Install dependencies"
	@printf "  \033[36m%-16s\033[0m %s\n" "make init"       "Scaffold a config.yaml"
	@printf "  \033[36m%-16s\033[0m %s\n" "make generate"   "Generate dataset from config or inline flags"
	@printf "  \033[36m%-16s\033[0m %s\n" "make estimate"   "Dry-run: estimate tokens and cost"
	@printf "  \033[36m%-16s\033[0m %s\n" "make validate"   "Deduplicate + validate dataset"
	@printf "  \033[36m%-16s\033[0m %s\n" "make export"     "Export to CSV, Parquet, OpenAI, Alpaca, ShareGPT"
	@printf "  \033[36m%-16s\033[0m %s\n" "make push"       "Push dataset to HuggingFace Hub"
	@printf "\n  \033[33mDev\033[0m\n"
	@printf "  \033[36m%-16s\033[0m %s\n" "make quality"    "Lint + format check + tests"
	@printf "  \033[36m%-16s\033[0m %s\n" "make format"     "Auto-fix lint + formatting"
	@printf "  \033[36m%-16s\033[0m %s\n" "make test"       "Run tests only"
	@printf "  \033[36m%-16s\033[0m %s\n" "make clean"      "Remove build artifacts"
	@printf "\n  \033[33mInline Generation\033[0m  (no config.yaml needed)\n"
	@printf "  \033[2m  make generate TASK=sft DOMAIN=\"coding\" SAMPLES=500\033[0m\n"
	@printf "  \033[2m  make generate TASK=classification LABELS=\"pos,neg\" SAMPLES=1000\033[0m\n"
	@printf "  \033[2m  make generate TASK=qa DOMAIN=\"ML\" DOCS=./papers/ SAMPLES=200\033[0m\n"
	@printf "  \033[2m  make generate TASK=classification LABELS=\"pos,neg\" SEED=data.jsonl SAMPLES=1000\033[0m\n"
	@printf "  \033[2m  make generate TASK=classification LABELS=\"pos,neg\" LANG=es SAMPLES=500\033[0m\n"
	@printf "\n  \033[33mOverrides\033[0m  (combine with any target)\n"
	@printf "  \033[36m%-16s\033[0m %s\n" "TASK=sft"        "Task: classification, ner, qa, preference, sft, conversation, summarization, distillation"
	@printf "  \033[36m%-16s\033[0m %s\n" "LABELS=\"a,b\""  "Comma-separated labels (classification/NER)"
	@printf "  \033[36m%-16s\033[0m %s\n" "DOMAIN=\"topic\"" "Domain context for generation"
	@printf "  \033[36m%-16s\033[0m %s\n" "MODEL=gpt-4o"   "Model name override"
	@printf "  \033[36m%-16s\033[0m %s\n" "STRATEGY=persona" "Strategy: direct, few_shot, persona, cot, adversarial, evolinstruct"
	@printf "  \033[36m%-16s\033[0m %s\n" "DOCS=./docs/"   "Document path for grounded generation"
	@printf "  \033[36m%-16s\033[0m %s\n" "SEED=data.jsonl" "Seed file for few-shot bootstrapping"
	@printf "  \033[36m%-16s\033[0m %s\n" "LANG=es"        "Generate in target language (es, fr, de, zh, ja, ...)"
	@printf "  \033[36m%-16s\033[0m %s\n" "SAMPLES=1000"   "Number of samples"
	@printf "  \033[36m%-16s\033[0m %s\n" "WORKERS=20"     "Parallel LLM requests"
	@printf "  \033[36m%-16s\033[0m %s\n" "FORMAT=openai"  "Output: jsonl, csv, parquet, openai, alpaca, sharegpt"
	@printf "  \033[36m%-16s\033[0m %s\n" "REPO=user/name" "HuggingFace repo for push"
	@printf "\n"
