"""One-command HuggingFace Hub publishing with auto-generated dataset cards."""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)


def push_to_hub(
    samples: list[Sample],
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    config: dict[str, Any] | None = None,
) -> str:
    """Push samples to HuggingFace Hub with an auto-generated dataset card.

    Args:
        samples: Generated samples to publish.
        repo_id: HuggingFace repo ID (e.g. "username/my-dataset").
        token: HuggingFace API token. Falls back to cached login.
        private: Whether the repo should be private.
        config: Generation config dict for card metadata.

    Returns:
        URL of the published dataset.
    """
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError as err:
        raise ImportError(
            "Install huggingface extra: uv add 'dataset-generator[huggingface]'"
        ) from err

    records = [s.to_dict() for s in samples]
    ds = Dataset.from_list(records)

    # Push dataset
    ds.push_to_hub(repo_id, token=token, private=private)
    logger.info(f"Pushed {len(samples)} samples to {repo_id}")

    # Generate and push dataset card
    card_content = generate_card(samples, repo_id, config=config)
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info(f"Pushed dataset card to {repo_id}")

    url = f"https://huggingface.co/datasets/{repo_id}"
    return url


def generate_card(
    samples: list[Sample],
    repo_id: str,
    config: dict[str, Any] | None = None,
) -> str:
    """Generate a YAML-frontmatter + markdown dataset card.

    Args:
        samples: Generated samples for stats/examples.
        repo_id: HuggingFace repo ID.
        config: Generation config dict for metadata.

    Returns:
        Dataset card as a string.
    """
    config = config or {}
    task_config = config.get("task", {})
    gen_config = config.get("generation", {})
    provider_config = config.get("provider", {})

    task_type = task_config.get("type", "unknown")
    model = provider_config.get("model", "unknown")
    strategy = gen_config.get("strategy", "direct")

    # Label distribution
    labels = [s.label for s in samples if s.label is not None]
    label_counts = Counter(labels)

    # YAML frontmatter
    frontmatter_data: dict[str, Any] = {
        "task_categories": [_map_task_category(task_type)],
        "size_categories": [_size_category(len(samples))],
        "tags": ["synthetic", "dataset-generator"],
    }
    if labels:
        frontmatter_data["tags"].append(task_type)

    frontmatter = "---\n"
    for key, value in frontmatter_data.items():
        if isinstance(value, list):
            frontmatter += f"{key}:\n"
            for item in value:
                frontmatter += f"  - {item}\n"
        else:
            frontmatter += f"{key}: {value}\n"
    frontmatter += "---\n"

    # Markdown body
    lines = [
        frontmatter,
        f"# {repo_id.split('/')[-1]}",
        "",
        f"Synthetic {task_type} dataset with **{len(samples):,}** samples.",
        "",
        "## Generation Config",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Model | `{model}` |",
        f"| Strategy | `{strategy}` |",
        f"| Task type | `{task_type}` |",
        f"| Samples | {len(samples):,} |",
    ]

    if gen_config.get("temperature") is not None:
        lines.append(f"| Temperature | {gen_config['temperature']} |")

    # Label distribution
    if label_counts:
        lines.extend(["", "## Label Distribution", ""])
        lines.append("| Label | Count | Percentage |")
        lines.append("|-------|-------|------------|")
        for label, count in label_counts.most_common():
            pct = count / len(labels) * 100
            lines.append(f"| {label} | {count:,} | {pct:.1f}% |")

    # Example samples
    lines.extend(["", "## Examples", ""])
    for i, sample in enumerate(samples[:3]):
        lines.append(f"**Example {i + 1}:**")
        lines.append("```json")
        lines.append(json.dumps(sample.to_dict(), indent=2))
        lines.append("```")
        lines.append("")

    # Credit
    lines.extend(
        [
            "---",
            "",
            "*Generated with [dataset-generator](https://github.com/datajuboor/dataset-generator)*",
        ]
    )

    return "\n".join(lines) + "\n"


def _map_task_category(task_type: str) -> str:
    """Map internal task type to HuggingFace task category."""
    mapping = {
        "classification": "text-classification",
        "ner": "token-classification",
        "qa": "question-answering",
        "preference": "text-generation",
        "summarization": "summarization",
    }
    return mapping.get(task_type, "text-generation")


def _size_category(n: int) -> str:
    """Map sample count to HuggingFace size category."""
    if n < 1_000:
        return "n<1K"
    if n < 10_000:
        return "1K<n<10K"
    if n < 100_000:
        return "10K<n<100K"
    if n < 1_000_000:
        return "100K<n<1M"
    return "n>1M"
