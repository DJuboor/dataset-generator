"""Output format serialization — JSONL, CSV, Parquet, HuggingFace, and SFT formats."""

import csv
import json
import logging
from pathlib import Path

from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core formats
# ---------------------------------------------------------------------------


class StreamingWriter:
    """Incrementally write samples to JSONL as they are generated."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0
        self._file = open(self.path, "w")  # noqa: SIM115

    def write_batch(self, samples: list[Sample]) -> None:
        """Append a batch of samples to the output file."""
        for sample in samples:
            self._file.write(json.dumps(sample.to_dict()) + "\n")
        self._file.flush()
        self._count += len(samples)

    def close(self) -> None:
        """Close the file and log the total."""
        self._file.close()
        logger.info(f"Streamed {self._count} samples to {self.path}")

    @property
    def count(self) -> int:
        return self._count

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def write_jsonl(samples: list[Sample], path: str | Path) -> None:
    """Write samples as JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")
    logger.info(f"Wrote {len(samples)} samples to {path}")


def write_csv(samples: list[Sample], path: str | Path) -> None:
    """Write samples as CSV with dynamic fieldnames derived from data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        return

    # Derive fieldnames from actual data
    fieldnames: list[str] = []
    for key in samples[0].to_dict():
        if key not in fieldnames:
            fieldnames.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row: dict[str, str] = {}
            for k, v in sample.to_dict().items():
                row[k] = (
                    json.dumps(v)
                    if isinstance(v, (dict, list))
                    else str(v)
                    if v is not None
                    else ""
                )
            writer.writerow(row)
    logger.info(f"Wrote {len(samples)} samples to {path}")


def write_huggingface(samples: list[Sample], path: str | Path) -> None:
    """Write as HuggingFace Dataset (Arrow format). Requires datasets extra."""
    try:
        from datasets import Dataset
    except ImportError as err:
        raise ImportError(
            "Install huggingface extra: uv add 'dataset-generator[huggingface]'"
        ) from err

    records = [s.to_dict() for s in samples]
    ds = Dataset.from_list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))
    logger.info(f"Wrote {len(samples)} samples to {path} (HuggingFace Dataset)")


def write_parquet(samples: list[Sample], path: str | Path) -> None:
    """Write as Parquet file. Requires pyarrow."""
    try:
        from datasets import Dataset
    except ImportError as err:
        raise ImportError(
            "Install huggingface extra: uv add 'dataset-generator[huggingface]'"
        ) from err

    records = [s.to_dict() for s in samples]
    ds = Dataset.from_list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(path))
    logger.info(f"Wrote {len(samples)} samples to {path} (Parquet)")


# ---------------------------------------------------------------------------
# SFT / fine-tuning formats
# ---------------------------------------------------------------------------


def write_openai_finetune(samples: list[Sample], path: str | Path) -> None:
    """Write in OpenAI fine-tuning JSONL format: {"messages": [...]}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            d = sample.to_dict()
            messages = _sample_to_messages(d)
            f.write(json.dumps({"messages": messages}) + "\n")
    logger.info(f"Wrote {len(samples)} samples to {path} (OpenAI fine-tune)")


def write_alpaca(samples: list[Sample], path: str | Path) -> None:
    """Write in Alpaca JSON format: [{"instruction": ..., "input": ..., "output": ...}]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in samples:
        d = sample.to_dict()
        records.append(
            {
                "instruction": d.get("instruction", d.get("text", "")),
                "input": d.get("input", ""),
                "output": d.get("response", d.get("label", d.get("answer", ""))),
            }
        )
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info(f"Wrote {len(samples)} samples to {path} (Alpaca)")


def write_sharegpt(samples: list[Sample], path: str | Path) -> None:
    """Write in ShareGPT format: {"conversations": [{"from": "human", ...}, ...]}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            d = sample.to_dict()
            conversations = []
            # Multi-turn: check for messages in metadata
            if "messages" in d:
                for msg in d["messages"]:
                    role_map = {"user": "human", "assistant": "gpt", "system": "system"}
                    conversations.append(
                        {
                            "from": role_map.get(msg.get("role", ""), msg.get("role", "")),
                            "value": msg.get("content", ""),
                        }
                    )
            else:
                # Single-turn: text is instruction, label/response is output
                conversations.append(
                    {"from": "human", "value": d.get("instruction", d.get("text", ""))}
                )
                output = d.get("response", d.get("label", d.get("answer", "")))
                if output:
                    conversations.append({"from": "gpt", "value": output})
            f.write(json.dumps({"conversations": conversations}) + "\n")
    logger.info(f"Wrote {len(samples)} samples to {path} (ShareGPT)")


def _sample_to_messages(d: dict) -> list[dict[str, str]]:
    """Convert a sample dict to OpenAI messages format."""
    messages = []
    # If the sample already has messages (multi-turn), use them
    if "messages" in d:
        return d["messages"]
    # Build from fields
    if d.get("system_prompt"):
        messages.append({"role": "system", "content": d["system_prompt"]})
    # Instruction/text as user message
    user_content = d.get("instruction", d.get("text", ""))
    if d.get("input"):
        user_content += f"\n\n{d['input']}"
    messages.append({"role": "user", "content": user_content})
    # Response/label/answer as assistant message
    assistant_content = d.get("response", d.get("label", d.get("answer", "")))
    if assistant_content:
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


# ---------------------------------------------------------------------------
# Dispatch + reading
# ---------------------------------------------------------------------------

WRITERS = {
    "jsonl": write_jsonl,
    "csv": write_csv,
    "huggingface": write_huggingface,
    "parquet": write_parquet,
    "openai": write_openai_finetune,
    "alpaca": write_alpaca,
    "sharegpt": write_sharegpt,
}


def write_output(samples: list[Sample], path: str | Path, fmt: str = "jsonl") -> None:
    """Write samples in the specified format.

    Args:
        samples: Samples to write.
        path: Output file/directory path.
        fmt: Format — 'jsonl', 'csv', 'parquet', 'huggingface', 'openai', 'alpaca', 'sharegpt'.
    """
    if fmt not in WRITERS:
        raise ValueError(f"Unknown format: {fmt}. Available: {list(WRITERS)}")
    WRITERS[fmt](samples, path)


def read_samples(path: str | Path) -> list[Sample]:
    """Read samples from JSONL or CSV based on file extension."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    return read_jsonl(path)


def read_jsonl(path: str | Path) -> list[Sample]:
    """Read samples from a JSONL file. Handles both flat and nested metadata formats."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                text = data.pop("text", "")
                label = data.pop("label", None)
                # If there's an explicit metadata key, use it as base
                metadata = data.pop("metadata", {})
                # Any remaining top-level keys are flattened task-specific fields
                metadata.update(data)
                samples.append(Sample(text=text, label=label, metadata=metadata))
    return samples


def read_csv(path: str | Path) -> list[Sample]:
    """Read samples from a CSV file."""
    samples = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.pop("text", "")
            label = row.pop("label", None)
            metadata = {k: v for k, v in row.items() if v}
            samples.append(Sample(text=text, label=label, metadata=metadata))
    return samples
