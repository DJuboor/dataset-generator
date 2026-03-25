"""CLI entry point — the ``dg`` command."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="dg",
    help="Generate synthetic ML datasets using any LLM.",
    no_args_is_help=True,
)
console = Console()

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"

# All known task types (kept in sync with tasks/__init__.py TASK_REGISTRY)
_TASK_TYPES = [
    "classification",
    "ner",
    "qa",
    "preference",
    "sft",
    "conversation",
    "summarization",
    "distillation",
]

# All known strategies (kept in sync with strategies/__init__.py STRATEGY_REGISTRY)
_STRATEGIES = [
    "direct",
    "few_shot",
    "persona",
    "cot",
    "adversarial",
    "evolinstruct",
]

# All output formats
_FORMATS = ["jsonl", "csv", "parquet", "huggingface", "openai", "alpaca", "sharegpt"]


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init(
    task_type: str = typer.Argument("classification", help=f"Task type: {', '.join(_TASK_TYPES)}"),
    labels: str = typer.Option("", "--labels", "-l", help="Comma-separated labels"),
    output: str = typer.Option("config.yaml", "--output", "-o", help="Output config path"),
) -> None:
    """Scaffold a config.yaml for a dataset generation task."""
    template_path = TEMPLATES_DIR / f"{task_type}.yaml"
    if not template_path.exists():
        available = [f.stem for f in TEMPLATES_DIR.glob("*.yaml")]
        console.print(f"[red]Unknown task type: {task_type}[/red]")
        console.print(f"Available: {', '.join(available)}")
        raise typer.Exit(1)

    dest = Path(output)
    if dest.exists():
        overwrite = typer.confirm(f"{dest} already exists. Overwrite?")
        if not overwrite:
            raise typer.Exit(0)

    shutil.copy(template_path, dest)

    # Inject custom labels if provided
    if labels and task_type == "classification":
        content = dest.read_text()
        content = content.replace(
            'labels: ["positive", "negative", "neutral"]',
            f"labels: [{', '.join(repr(lbl.strip()) for lbl in labels.split(','))}]",
        )
        dest.write_text(content)
    elif labels and task_type == "ner":
        content = dest.read_text()
        content = content.replace(
            'entity_types: ["PERSON", "ORG", "LOC", "DATE"]',
            f"entity_types: [{', '.join(repr(lbl.strip()) for lbl in labels.split(','))}]",
        )
        dest.write_text(content)

    console.print(f"[green]Created {dest}[/green]")
    console.print("Edit the config, then run: [bold]dg generate[/bold]")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@app.command()
def generate(
    config: str | None = typer.Option(None, "--config", "-c", help="Config YAML path"),
    task: str | None = typer.Option(
        None, "--task", "-t", help=f"Task type: {', '.join(_TASK_TYPES)}"
    ),
    labels: str | None = typer.Option(
        None, "--labels", help="Comma-separated labels (classification)"
    ),
    entity_types: str | None = typer.Option(
        None, "--entity-types", help="Comma-separated entity types (NER)"
    ),
    domain: str | None = typer.Option(None, "--domain", "-d", help="Domain context"),
    model: str | None = typer.Option(None, "--model", "-m", help="Override model name"),
    base_url: str | None = typer.Option(None, "--base-url", help="Override provider base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="Override API key"),
    strategy: str | None = typer.Option(
        None, "--strategy", "-s", help=f"Strategy: {', '.join(_STRATEGIES)}"
    ),
    num_samples: int | None = typer.Option(None, "--num-samples", "-n", help="Number of samples"),
    max_workers: int | None = typer.Option(None, "--max-workers", "-w", help="Parallel workers"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output path"),
    fmt: str | None = typer.Option(
        None, "--format", "-f", help=f"Output format: {', '.join(_FORMATS)}"
    ),
    from_docs: str | None = typer.Option(
        None, "--from-docs", help="Path to docs directory/file for grounded generation"
    ),
    max_cost: float | None = typer.Option(
        None, "--max-cost", help="Budget cap in USD (stops generation when reached)"
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from checkpoint if available"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost without running"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Generate a synthetic dataset. Use --config for YAML or --task for inline mode."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    from dataset_generator.config import load_config
    from dataset_generator.engine import estimate_run
    from dataset_generator.engine import generate as run_generate

    # Build config: either from YAML or inline from CLI flags
    if task and not config:
        cfg = _build_inline_config(
            task=task,
            labels=labels,
            entity_types=entity_types,
            domain=domain,
            model=model,
            base_url=base_url,
            api_key=api_key,
            strategy=strategy,
            num_samples=num_samples,
            max_workers=max_workers,
            output_path=output,
            output_format=fmt,
            from_docs=from_docs,
            max_cost=max_cost,
        )
    else:
        cfg = load_config(config or "config.yaml")

    # CLI overrides on top of YAML config
    if num_samples is not None:
        cfg.setdefault("generation", {})["num_samples"] = num_samples
    if max_workers is not None:
        cfg.setdefault("generation", {})["max_workers"] = max_workers
    if output is not None:
        cfg.setdefault("output", {})["path"] = output
    if fmt is not None:
        cfg.setdefault("output", {})["format"] = fmt
    if strategy is not None:
        cfg.setdefault("generation", {})["strategy"] = strategy
    if model is not None:
        cfg.setdefault("provider", {})["model"] = model
    if base_url is not None:
        cfg.setdefault("provider", {})["base_url"] = base_url
    if api_key is not None:
        cfg.setdefault("provider", {})["api_key"] = api_key
    if max_cost is not None:
        cfg.setdefault("generation", {})["max_cost"] = max_cost
    if from_docs is not None:
        cfg["from_docs"] = from_docs

    # Dry run: estimate and exit
    if dry_run:
        est = estimate_run(cfg)
        table = Table(title="Dry Run Estimate")
        table.add_column("Parameter", style="bold")
        table.add_column("Value")
        table.add_row("Model", est["model"])
        table.add_row("Batches", str(est["num_batches"]))
        table.add_row("Batch size", str(est["batch_size"]))
        table.add_row("Est. input tokens", f"{est['estimated_input_tokens']:,}")
        table.add_row("Est. output tokens", f"{est['estimated_output_tokens']:,}")
        table.add_row("Est. total tokens", f"{est['estimated_total_tokens']:,}")
        table.add_row("Est. cost", f"${est['estimated_cost_usd']:.4f}")
        console.print(table)
        console.print("\nRun without --dry-run to proceed.")
        return

    samples = run_generate(config=cfg, resume=resume)
    output_path = cfg.get("output", {}).get("path", "data/output.jsonl")
    console.print(f"\n[green]Generated {len(samples)} samples → {output_path}[/green]")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@app.command()
def validate(
    input_path: str = typer.Argument(..., help="Path to JSONL dataset"),
    similarity_threshold: float = typer.Option(0.85, "--threshold", "-t", help="Dedup threshold"),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output path for cleaned dataset"
    ),
) -> None:
    """Validate and deduplicate an existing dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from dataset_generator.formats import read_jsonl, write_jsonl
    from dataset_generator.quality import deduplicate, validate_samples

    samples = read_jsonl(input_path)
    console.print(f"Loaded {len(samples)} samples from {input_path}")

    deduped = deduplicate(samples, similarity_threshold=similarity_threshold)
    validated = validate_samples(deduped)

    # Stats
    table = Table(title="Validation Report")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Input samples", str(len(samples)))
    table.add_row("After dedup", str(len(deduped)))
    table.add_row("After validation", str(len(validated)))
    table.add_row("Duplicates removed", str(len(samples) - len(deduped)))
    table.add_row("Invalid removed", str(len(deduped) - len(validated)))

    # Label distribution (only for samples that have labels)
    label_counts: dict[str, int] = {}
    for s in validated:
        if s.label:
            label_counts[s.label] = label_counts.get(s.label, 0) + 1
    if label_counts:
        table.add_row("---", "---")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            pct = count / len(validated) * 100
            table.add_row(f"  {label}", f"{count} ({pct:.1f}%)")

    # Entity stats for NER
    entity_counts: dict[str, int] = {}
    for s in validated:
        entities = s.metadata.get("entities", [])
        for ent in entities:
            lbl = ent.get("label", "unknown")
            entity_counts[lbl] = entity_counts.get(lbl, 0) + 1
    if entity_counts:
        table.add_row("---", "---")
        table.add_row("[bold]Entity Distribution[/bold]", "")
        for label, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
            table.add_row(f"  {label}", str(count))

    console.print(table)

    # Write cleaned output
    if output is None:
        p = Path(input_path)
        out_path = str(p.with_stem(p.stem + "_clean"))
    else:
        out_path = output
    write_jsonl(validated, out_path)
    console.print(f"[green]Cleaned dataset → {out_path}[/green]")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@app.command()
def export(
    input_path: str = typer.Argument(..., help="Path to JSONL dataset"),
    fmt: str = typer.Option("csv", "--format", "-f", help=f"Format: {', '.join(_FORMATS)}"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output path"),
) -> None:
    """Export a JSONL dataset to another format."""
    from dataset_generator.formats import read_jsonl, write_output

    samples = read_jsonl(input_path)

    if output is None:
        ext_map = {
            "csv": ".csv",
            "huggingface": "_hf",
            "parquet": ".parquet",
            "openai": "_openai.jsonl",
            "alpaca": "_alpaca.json",
            "sharegpt": "_sharegpt.jsonl",
        }
        suffix = ext_map.get(fmt, f".{fmt}")
        output = str(Path(input_path).with_suffix(suffix))

    write_output(samples, output, fmt=fmt)
    console.print(f"[green]Exported {len(samples)} samples → {output}[/green]")


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------


@app.command()
def push(
    repo_id: str = typer.Argument(..., help="HuggingFace repo ID (e.g. username/my-dataset)"),
    input_path: str = typer.Option("data/output.jsonl", "--file", "-f", help="JSONL file to push"),
    private: bool = typer.Option(False, "--private", help="Create private repo"),
    token: str | None = typer.Option(
        None, "--token", help="HuggingFace token (or use HF_TOKEN env)"
    ),
) -> None:
    """Push a dataset to HuggingFace Hub with an auto-generated dataset card."""
    from dataset_generator.formats import read_jsonl
    from dataset_generator.hub import push_to_hub

    samples = read_jsonl(input_path)
    console.print(f"Pushing {len(samples)} samples to [bold]{repo_id}[/bold]...")

    url = push_to_hub(samples, repo_id=repo_id, token=token, private=private)
    console.print(f"[green]Published → {url}[/green]")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@app.command()
def info() -> None:
    """Show available task types, strategies, and output formats."""
    table = Table(title="Task Types")
    table.add_column("Type", style="bold")
    table.add_column("Description")
    table.add_row("classification", "Labeled text classification (sentiment, topic, intent)")
    table.add_row("ner", "Named entity recognition with span annotations")
    table.add_row("qa", "Question-answer pairs with optional context")
    table.add_row("preference", "DPO/RLHF preference pairs (chosen vs rejected)")
    table.add_row("sft", "Instruction-response pairs for supervised fine-tuning")
    table.add_row("conversation", "Multi-turn conversations for chat model training")
    table.add_row("summarization", "Document-summary pairs")
    table.add_row("distillation", "Teacher-quality responses for knowledge distillation")
    console.print(table)

    table2 = Table(title="Strategies")
    table2.add_column("Strategy", style="bold")
    table2.add_column("Description")
    table2.add_row("direct", "Simple generation with diversity hints per batch")
    table2.add_row("few_shot", "Inject example outputs to guide format and quality")
    table2.add_row("persona", "Rotate writing personas for natural style diversity")
    table2.add_row("cot", "Chain-of-thought reasoning for complex tasks")
    table2.add_row("adversarial", "Generate edge cases, boundary examples, tricky inputs")
    table2.add_row(
        "evolinstruct", "Iteratively evolve instructions for complexity (WizardLM-style)"
    )
    console.print(table2)

    table3 = Table(title="Output Formats")
    table3.add_column("Format", style="bold")
    table3.add_column("Description")
    table3.add_row("jsonl", "JSON Lines (default)")
    table3.add_row("csv", "Comma-separated values")
    table3.add_row("parquet", "Apache Parquet (via datasets)")
    table3.add_row("huggingface", "HuggingFace Arrow dataset")
    table3.add_row("openai", "OpenAI fine-tuning JSONL ({messages: [...]})")
    table3.add_row("alpaca", "Alpaca JSON ({instruction, input, output})")
    table3.add_row("sharegpt", "ShareGPT JSONL ({conversations: [...]})")
    console.print(table3)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_inline_config(
    task: str,
    labels: str | None = None,
    entity_types: str | None = None,
    domain: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    strategy: str | None = None,
    num_samples: int | None = None,
    max_workers: int | None = None,
    output_path: str | None = None,
    output_format: str | None = None,
    from_docs: str | None = None,
    max_cost: float | None = None,
) -> dict:
    """Build a config dict from CLI flags for inline generation (no YAML needed)."""
    from dataset_generator.config import DEFAULT_CONFIG, _deep_merge, _walk_and_substitute

    cfg = _walk_and_substitute(_deep_merge(DEFAULT_CONFIG, {"type": task}))

    # Task config
    task_cfg: dict = {}
    if domain:
        task_cfg["domain"] = domain
    if labels:
        task_cfg["labels"] = [lbl.strip() for lbl in labels.split(",")]
    if entity_types:
        task_cfg["entity_types"] = [e.strip() for e in entity_types.split(",")]
    if task_cfg:
        cfg["task"] = task_cfg

    # Provider overrides
    if model:
        cfg["provider"]["model"] = model
    if base_url:
        cfg["provider"]["base_url"] = base_url
    if api_key:
        cfg["provider"]["api_key"] = api_key

    # Generation overrides
    if strategy:
        cfg["generation"]["strategy"] = strategy
    if num_samples is not None:
        cfg["generation"]["num_samples"] = num_samples
    if max_workers is not None:
        cfg["generation"]["max_workers"] = max_workers
    if max_cost is not None:
        cfg["generation"]["max_cost"] = max_cost

    # Output overrides
    if output_path:
        cfg["output"]["path"] = output_path
    if output_format:
        cfg["output"]["format"] = output_format

    # Document grounding
    if from_docs:
        cfg["from_docs"] = from_docs

    return cfg
