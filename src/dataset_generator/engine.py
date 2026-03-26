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
from dataset_generator.tasks.base import Sample, validate_sample_schema

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
    language: str | None = None,
) -> tuple[list[Sample], int, int]:
    """Generate a single batch with retries.

    Returns:
        Tuple of (samples, input_tokens, output_tokens).
    """
    messages = task.build_messages(batch_size=batch_size)
    messages = strategy.apply(messages, batch_index)

    # Inject language instruction
    if language:
        from dataset_generator.tasks.base import LANGUAGE_NAMES

        lang_name = LANGUAGE_NAMES.get(language, language)
        for msg in messages:
            if msg["role"] == "user":
                msg["content"] += (
                    f"\n\nIMPORTANT: Generate ALL text content in {lang_name}. "
                    f"Every example must be written entirely in {lang_name}."
                )
                break

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

    last_error: str = ""
    parse_failures = 0

    for attempt in range(max_retries + 1):
        try:
            result: CompletionResult = provider.complete(messages, temperature=temperature)
            total_input += result.input_tokens or 0
            total_output += result.output_tokens or 0
            samples = task.parse_response(result.content)
            # Schema validation: drop samples missing required keys
            if samples and hasattr(task, "required_keys"):
                required = task.required_keys()
                valid = [s for s in samples if validate_sample_schema(s.to_dict(), required)]
                if len(valid) < len(samples):
                    logger.debug(
                        f"Batch {batch_index}: schema validation dropped "
                        f"{len(samples) - len(valid)}/{len(samples)} samples"
                    )
                samples = valid
            if samples:
                return samples, total_input, total_output
            parse_failures += 1
            logger.warning(f"Batch {batch_index}: parsed 0 samples, attempt {attempt + 1}")
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Batch {batch_index} attempt {attempt + 1} failed: {e}")

    # Emit actionable guidance after all retries exhausted
    if "onnect" in last_error or "refused" in last_error:
        logger.error(
            f"Cannot reach LLM at {provider.client.base_url}. "
            "Is Ollama/vLLM running? Check base_url and try: curl %s/v1/models",
            provider.client.base_url,
        )
    elif "timed out" in last_error.lower() or "timeout" in last_error.lower():
        logger.error(
            "Request timed out. For slow/large models, try: "
            "--timeout 1200 or lower --num-samples and batch_size in your config."
        )
    elif parse_failures >= max_retries:
        logger.error(
            "Model responded but output could not be parsed as JSON. "
            "Try a more capable model, or lower batch_size to reduce output complexity."
        )

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

    # Auto-detect local models and adjust defaults for single-GPU inference
    provider_config = config.get("provider", {})
    base_url = provider_config.get("base_url", "")
    _is_local = any(host in base_url for host in ("localhost", "127.0.0.1", "0.0.0.0"))

    if _is_local:
        # Only adjust values that weren't explicitly set by the user
        if "max_workers" not in gen_config:
            max_workers = 1
        if "batch_size" not in gen_config:
            batch_size = 5
        if "timeout" not in provider_config:
            provider_config.setdefault("timeout", 600.0)
        if max_workers == 1 or batch_size < 10:
            logger.info(
                f"Local model detected ({base_url}): using max_workers={max_workers}, "
                f"batch_size={batch_size}"
            )

    provider = create_provider(provider_config)
    task = create_task(config)

    # Load seed examples for few-shot bootstrapping
    seed_from = config.get("seed_from")
    if seed_from:
        from dataset_generator.formats import read_samples

        seed_samples = read_samples(seed_from)
        seed_dicts = [s.to_dict() for s in seed_samples]
        logger.info(f"Loaded {len(seed_dicts)} seed examples from {seed_from}")
        if strategy_name == "direct":
            strategy_name = "few_shot"
            logger.info("Auto-switching strategy to few_shot (seed examples provided)")
        strategy_config["examples"] = seed_dicts

    strategy = create_strategy(strategy_name, config=strategy_config)

    # Multi-language generation
    language = config.get("language")

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
                language,
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

                # Rich progress postfix: samples + cost or tokens
                cost = _estimate_cost(total_input_tokens, total_output_tokens, provider.model)
                if cost > 0:
                    pbar.set_postfix(samples=len(all_samples), cost=f"${cost:.4f}")
                else:
                    total_tok = total_input_tokens + total_output_tokens
                    pbar.set_postfix(samples=len(all_samples), tokens=f"{total_tok:,}")

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

    # Auto-inject language filter when --language is set
    quality_steps_config = quality_config.get("steps", [])
    if language:
        has_language_step = any("language" in step for step in quality_steps_config)
        if not has_language_step:
            quality_steps_config = [{"language": {"expected": language}}, *quality_steps_config]
            logger.info(f"Auto-added language filter (expected={language})")

    # Run additional quality steps if configured
    if quality_steps_config:
        pipeline = _build_quality_pipeline(quality_steps_config)
        all_samples, quality_report = pipeline.run(all_samples)
        logger.info(
            f"Quality pipeline: {quality_report.input_count} → {quality_report.output_count} samples"
        )

    # Trim to requested count
    all_samples = all_samples[:num_samples]

    if len(all_samples) < num_samples:
        if len(all_samples) == 0:
            logger.warning(
                f"Generated 0 samples (requested {num_samples}). "
                "Check the errors above — likely a connection, timeout, or parsing issue."
            )
        else:
            logger.warning(
                f"Delivered {len(all_samples)} samples (requested {num_samples}). "
                "Try increasing num_samples, lowering quality thresholds, or using a different strategy."
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


# ---------------------------------------------------------------------------
# Async generation (for remote/cloud providers)
# ---------------------------------------------------------------------------


async def _async_generate_batch(
    provider,
    task,
    strategy,
    batch_index: int,
    batch_size: int,
    temperature: float,
    max_retries: int,
    doc_contexts: list[str] | None = None,
    language: str | None = None,
) -> tuple[list[Sample], int, int]:
    """Async version of _generate_batch."""
    messages = task.build_messages(batch_size=batch_size)
    messages = strategy.apply(messages, batch_index)

    if language:
        from dataset_generator.tasks.base import LANGUAGE_NAMES

        lang_name = LANGUAGE_NAMES.get(language, language)
        for msg in messages:
            if msg["role"] == "user":
                msg["content"] += (
                    f"\n\nIMPORTANT: Generate ALL text content in {lang_name}. "
                    f"Every example must be written entirely in {lang_name}."
                )
                break

    if doc_contexts:
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
            result: CompletionResult = await provider.async_complete(
                messages, temperature=temperature
            )
            total_input += result.input_tokens or 0
            total_output += result.output_tokens or 0
            samples = task.parse_response(result.content)
            if samples and hasattr(task, "required_keys"):
                required = task.required_keys()
                samples = [s for s in samples if validate_sample_schema(s.to_dict(), required)]
            if samples:
                return samples, total_input, total_output
            logger.warning(f"Batch {batch_index}: parsed 0 samples, attempt {attempt + 1}")
        except Exception as e:
            logger.warning(f"Batch {batch_index} attempt {attempt + 1} failed: {e}")

    return [], total_input, total_output


async def async_generate(
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    resume: bool = False,
) -> list[Sample]:
    """Async generation pipeline — uses asyncio for concurrent API calls.

    Same interface and behavior as generate(), but uses async I/O for
    better throughput with cloud providers.
    """
    import asyncio

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

    provider_config = config.get("provider", {})
    provider = create_provider(provider_config)
    task = create_task(config)

    seed_from = config.get("seed_from")
    if seed_from:
        from dataset_generator.formats import read_samples

        seed_samples = read_samples(seed_from)
        seed_dicts = [s.to_dict() for s in seed_samples]
        if strategy_name == "direct":
            strategy_name = "few_shot"
        strategy_config["examples"] = seed_dicts

    strategy = create_strategy(strategy_name, config=strategy_config)
    language = config.get("language")

    doc_contexts: list[str] | None = None
    from_docs = config.get("from_docs")
    if from_docs:
        from dataset_generator.loaders import load_documents

        chunks = load_documents(from_docs, chunk_size=1000, chunk_overlap=200)
        if chunks:
            doc_contexts = [chunk.text for chunk in chunks]

    overshoot = 1.2
    total_needed = math.ceil(num_samples * overshoot)
    num_batches = math.ceil(total_needed / batch_size)

    logger.info(
        f"Async generating {num_samples} samples ({num_batches} batches of {batch_size}, "
        f"concurrency={max_workers}, strategy={strategy_name})"
    )

    all_samples: list[Sample] = []
    total_input_tokens = 0
    total_output_tokens = 0
    semaphore = asyncio.Semaphore(max_workers)
    budget_exceeded = False

    async def _run_batch(batch_idx: int) -> tuple[list[Sample], int, int]:
        nonlocal budget_exceeded
        if budget_exceeded:
            return [], 0, 0
        async with semaphore:
            return await _async_generate_batch(
                provider,
                task,
                strategy,
                batch_idx,
                batch_size,
                temperature,
                max_retries,
                doc_contexts,
                language,
            )

    with tqdm(total=num_batches, desc="Generating (async)", unit="batch") as pbar:
        tasks = [_run_batch(i) for i in range(num_batches)]
        for coro in asyncio.as_completed(tasks):
            samples, in_tok, out_tok = await coro
            all_samples.extend(samples)
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            pbar.update(1)

            cost = _estimate_cost(total_input_tokens, total_output_tokens, provider.model)
            if cost > 0:
                pbar.set_postfix(samples=len(all_samples), cost=f"${cost:.4f}")
            else:
                pbar.set_postfix(
                    samples=len(all_samples), tokens=f"{total_input_tokens + total_output_tokens:,}"
                )

            if max_cost and cost >= max_cost:
                logger.warning(f"Budget cap ${max_cost} reached, stopping generation")
                budget_exceeded = True
                break

    logger.info(f"Raw samples: {len(all_samples)}")

    # Quality pipeline (same as sync)
    quality_config = config.get("quality", {})
    all_samples = deduplicate(
        all_samples, similarity_threshold=quality_config.get("similarity_threshold", 0.85)
    )

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

    quality_steps_config = quality_config.get("steps", [])
    if language and not any("language" in step for step in quality_steps_config):
        quality_steps_config = [{"language": {"expected": language}}, *quality_steps_config]

    if quality_steps_config:
        pipeline = _build_quality_pipeline(quality_steps_config)
        all_samples, _ = pipeline.run(all_samples)

    all_samples = all_samples[:num_samples]
    logger.info(f"Final dataset: {len(all_samples)} samples")

    output_config = config.get("output", {})
    write_output(
        all_samples,
        output_config.get("path", "data/output.jsonl"),
        fmt=output_config.get("format", "jsonl"),
    )

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
        LLMJudge,
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
        "llm_judge": lambda cfg: LLMJudge(**cfg),
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
