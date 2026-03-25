"""Generation engine — orchestrates LLM calls with concurrency, retries, and progress."""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from dataset_generator.config import load_config
from dataset_generator.formats import write_output
from dataset_generator.providers import create_provider
from dataset_generator.providers.base import CompletionResult
from dataset_generator.quality import QualityPipeline, deduplicate, validate_samples
from dataset_generator.strategies import create_strategy
from dataset_generator.tasks import create_task
from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)


def _generate_batch(
    provider,
    task,
    strategy,
    batch_index: int,
    batch_size: int,
    temperature: float,
    max_retries: int,
    doc_contexts: list[str] | None = None,
) -> tuple[list[Sample], int, int]:
    """Generate a single batch with retries.

    Returns:
        Tuple of (samples, input_tokens, output_tokens).
    """
    messages = task.build_messages(batch_size=batch_size)
    messages = strategy.apply(messages, batch_index)

    # Inject document context for grounded generation
    if doc_contexts:
        # Rotate through docs — each batch gets a different window
        window_size = min(3, len(doc_contexts))
        start = (batch_index * window_size) % len(doc_contexts)
        batch_docs = [doc_contexts[(start + i) % len(doc_contexts)] for i in range(window_size)]
        context_block = "\n\n---\n\n".join(batch_docs)
        for msg in messages:
            if msg["role"] == "user":
                msg["content"] += (
                    "\n\nBase your generation on these reference documents:\n\n" + context_block
                )
                break

    total_input = 0
    total_output = 0

    for attempt in range(max_retries + 1):
        try:
            result: CompletionResult = provider.complete(messages, temperature=temperature)
            total_input += result.input_tokens or 0
            total_output += result.output_tokens or 0
            samples = task.parse_response(result.content)
            if samples:
                return samples, total_input, total_output
            logger.warning(f"Batch {batch_index}: parsed 0 samples, attempt {attempt + 1}")
        except Exception as e:
            logger.warning(f"Batch {batch_index} attempt {attempt + 1} failed: {e}")

    return [], total_input, total_output


def generate(
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    resume: bool = False,
) -> list[Sample]:
    """Run the full generation pipeline.

    Args:
        config: Pre-loaded config dict. If None, loads from config_path.
        config_path: Path to config YAML. Used only if config is None.
        resume: If True, attempt to resume from a previous checkpoint.

    Returns:
        List of generated, deduplicated, validated samples.
    """
    if config is None:
        config = load_config(config_path)

    gen_config = config.get("generation", {})
    num_samples = gen_config.get("num_samples", 100)
    max_workers = gen_config.get("max_workers", 10)
    max_retries = gen_config.get("max_retries", 3)
    temperature = gen_config.get("temperature", 0.7)
    batch_size = gen_config.get("batch_size", 10)
    strategy_name = gen_config.get("strategy", "direct")
    strategy_config = gen_config.get("strategy_config", {})
    max_cost = gen_config.get("max_cost")

    provider = create_provider(config.get("provider", {}))
    task = create_task(config)
    strategy = create_strategy(strategy_name, config=strategy_config)

    # Load documents for grounded generation
    doc_contexts: list[str] | None = None
    from_docs = config.get("from_docs")
    if from_docs:
        from dataset_generator.loaders import load_documents

        chunks = load_documents(from_docs, chunk_size=1000, chunk_overlap=200)
        if chunks:
            doc_contexts = [chunk.text for chunk in chunks]
            logger.info(f"Loaded {len(chunks)} document chunks from {from_docs}")
        else:
            logger.warning(f"No documents loaded from {from_docs}")

    # Calculate batches needed (overshoot to account for dedup/validation losses)
    overshoot = 1.2
    total_needed = math.ceil(num_samples * overshoot)
    num_batches = math.ceil(total_needed / batch_size)

    logger.info(
        f"Generating {num_samples} samples ({num_batches} batches of {batch_size}, "
        f"{max_workers} workers, strategy={strategy_name})"
    )

    # Checkpoint support
    checkpoint_mgr = None
    start_batch = 0
    all_samples: list[Sample] = []
    total_input_tokens = 0
    total_output_tokens = 0

    if resume:
        from dataset_generator.checkpoint import CheckpointManager

        checkpoint_mgr = CheckpointManager()
        restored = checkpoint_mgr.load(config)
        if restored:
            all_samples, start_batch = restored
            start_batch += 1  # Resume from the next batch
            logger.info(f"Resuming from batch {start_batch} with {len(all_samples)} samples")
    elif resume is not False:
        # resume=True but no checkpoint — start fresh with manager for saving
        pass

    if checkpoint_mgr is None and resume:
        from dataset_generator.checkpoint import CheckpointManager

        checkpoint_mgr = CheckpointManager()

    remaining_batches = list(range(start_batch, num_batches))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_batch,
                provider,
                task,
                strategy,
                i,
                batch_size,
                temperature,
                max_retries,
                doc_contexts,
            ): i
            for i in remaining_batches
        }

        with tqdm(total=num_batches, desc="Generating", unit="batch", initial=start_batch) as pbar:
            for future in as_completed(futures):
                batch_idx = futures[future]
                samples, in_tok, out_tok = future.result()
                all_samples.extend(samples)
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                pbar.update(1)
                pbar.set_postfix(samples=len(all_samples))

                # Save checkpoint after each batch
                if checkpoint_mgr and samples:
                    checkpoint_mgr.save(samples, batch_idx, config)

                # Budget cap check
                if (
                    max_cost
                    and _estimate_cost(total_input_tokens, total_output_tokens, provider.model)
                    >= max_cost
                ):
                    logger.warning(f"Budget cap ${max_cost} reached, stopping generation")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    logger.info(f"Raw samples: {len(all_samples)}")

    if total_input_tokens:
        cost = _estimate_cost(total_input_tokens, total_output_tokens, provider.model)
        logger.info(
            f"Tokens: {total_input_tokens:,} input + {total_output_tokens:,} output"
            f" | Estimated cost: ${cost:.4f}"
        )

    # Quality pipeline
    quality_config = config.get("quality", {})
    similarity_threshold = quality_config.get("similarity_threshold", 0.85)
    all_samples = deduplicate(all_samples, similarity_threshold=similarity_threshold)

    allowed_labels = None
    task_config = config.get("task", {})
    if "labels" in task_config:
        labels = task_config["labels"]
        allowed_labels = (
            [lbl.strip() for lbl in labels.split(",")] if isinstance(labels, str) else labels
        )

    all_samples = validate_samples(
        all_samples,
        min_length=quality_config.get("min_length", 10),
        max_length=quality_config.get("max_length", 10000),
        allowed_labels=allowed_labels,
    )

    # Run additional quality steps if configured
    quality_steps_config = quality_config.get("steps", [])
    if quality_steps_config:
        pipeline = _build_quality_pipeline(quality_steps_config)
        all_samples, quality_report = pipeline.run(all_samples)
        logger.info(
            f"Quality pipeline: {quality_report.input_count} → {quality_report.output_count} samples"
        )

    # Trim to requested count
    all_samples = all_samples[:num_samples]

    if len(all_samples) < num_samples:
        logger.warning(
            f"Delivered {len(all_samples)} samples (requested {num_samples}). "
            "Increase overshoot or relax quality filters."
        )

    logger.info(f"Final dataset: {len(all_samples)} samples")

    # Clean up checkpoint after successful completion
    if checkpoint_mgr:
        checkpoint_mgr.cleanup()

    # Write output
    output_config = config.get("output", {})
    output_path = output_config.get("path", "data/output.jsonl")
    output_format = output_config.get("format", "jsonl")
    write_output(all_samples, output_path, fmt=output_format)

    return all_samples


def estimate_run(config: dict[str, Any]) -> dict[str, Any]:
    """Estimate cost and tokens for a generation run without executing it.

    Returns:
        Dict with estimated tokens, cost, batches, and model info.
    """
    gen_config = config.get("generation", {})
    num_samples = gen_config.get("num_samples", 100)
    batch_size = gen_config.get("batch_size", 10)
    model = config.get("provider", {}).get("model", "unknown")

    overshoot = 1.2
    total_needed = math.ceil(num_samples * overshoot)
    num_batches = math.ceil(total_needed / batch_size)

    # Build one representative batch to estimate prompt size
    task = create_task(config)
    messages = task.build_messages(batch_size=batch_size)
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    # ~4 chars per token for English
    est_input_per_batch = prompt_chars // 4
    # Output typically 2-4x input for generation tasks
    est_output_per_batch = est_input_per_batch * 3

    total_input = est_input_per_batch * num_batches
    total_output = est_output_per_batch * num_batches
    cost = _estimate_cost(total_input, total_output, model)

    return {
        "model": model,
        "num_batches": num_batches,
        "batch_size": batch_size,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_total_tokens": total_input + total_output,
        "estimated_cost_usd": cost,
    }


# Per-million-token pricing for common models
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    "claude-haiku-4-5": (0.80, 4.00),
}


def _build_quality_pipeline(steps_config: list[dict]) -> QualityPipeline:
    """Build a QualityPipeline from config.

    Config format:
        steps:
          - pii: {action: remove, patterns: [email, phone_us]}
          - language: {expected: en, action: remove}
          - toxicity: {action: flag}
          - balance: {strategy: report, max_ratio: 3.0}
          - diversity: {}
    """
    from dataset_generator.quality import (
        BalanceChecker,
        DiversityReporter,
        LanguageFilter,
        PIIFilter,
        QualityPipeline,
        ToxicityFilter,
    )

    step_builders = {
        "pii": lambda cfg: PIIFilter(**cfg),
        "language": lambda cfg: LanguageFilter(**cfg),
        "toxicity": lambda cfg: ToxicityFilter(**cfg),
        "balance": lambda cfg: BalanceChecker(**cfg),
        "diversity": lambda cfg: DiversityReporter(**cfg),
    }

    steps = []
    for step_entry in steps_config:
        for name, cfg in step_entry.items():
            builder = step_builders.get(name)
            if builder is None:
                logger.warning(f"Unknown quality step: {name}, skipping")
                continue
            steps.append(builder(cfg or {}))

    return QualityPipeline(steps=steps)


def _estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate USD cost from token counts and model name."""
    # Try exact match first, then prefix match
    pricing = _MODEL_PRICING.get(model)
    if not pricing:
        for key, val in _MODEL_PRICING.items():
            if key in model or model in key:
                pricing = val
                break
    if not pricing:
        return 0.0  # Unknown model, can't estimate

    input_cost = (input_tokens / 1_000_000) * pricing[0]
    output_cost = (output_tokens / 1_000_000) * pricing[1]
    return input_cost + output_cost
