"""Checkpoint/resume support for long generation runs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint files for resumable generation runs.

    Checkpoints are keyed by a hash of the config dict (excluding output path)
    so that a resumed run matches the original parameters.
    """

    def __init__(self, checkpoint_dir: Path | str = ".dg_checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)

    def save(self, samples: list[Sample], batch_index: int, config: dict[str, Any]) -> None:
        """Append completed batch samples to checkpoint and update metadata.

        Args:
            samples: Samples from the completed batch.
            batch_index: Index of the completed batch.
            config: Generation config dict (used for hashing).
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config_hash = _config_hash(config)
        data_path = self.checkpoint_dir / f"{config_hash}.jsonl"
        meta_path = self.checkpoint_dir / f"{config_hash}.meta.json"

        # Append samples to JSONL
        with open(data_path, "a") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

        # Update metadata sidecar
        meta = {
            "config_hash": config_hash,
            "batch_index": batch_index,
            "timestamp": time.time(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        logger.debug(
            f"Checkpoint saved: batch {batch_index}, {len(samples)} samples appended ({data_path})"
        )

    def load(self, config: dict[str, Any]) -> tuple[list[Sample], int] | None:
        """Load checkpoint for the given config if one exists.

        Args:
            config: Generation config dict to match against.

        Returns:
            (accumulated_samples, last_completed_batch_index) or None if no checkpoint.
        """
        config_hash = _config_hash(config)
        data_path = self.checkpoint_dir / f"{config_hash}.jsonl"
        meta_path = self.checkpoint_dir / f"{config_hash}.meta.json"

        if not data_path.exists() or not meta_path.exists():
            return None

        # Read metadata
        with open(meta_path) as f:
            meta = json.load(f)

        batch_index: int = meta["batch_index"]

        # Read accumulated samples
        samples: list[Sample] = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data.pop("text", "")
                label = data.pop("label", None)
                metadata = data.pop("metadata", {})
                metadata.update(data)
                samples.append(Sample(text=text, label=label, metadata=metadata))

        logger.info(
            f"Resumed from checkpoint: {len(samples)} samples, "
            f"last batch {batch_index} ({data_path})"
        )
        return samples, batch_index

    def cleanup(self) -> None:
        """Remove all checkpoint files after successful completion."""
        if not self.checkpoint_dir.exists():
            return

        removed = 0
        for path in self.checkpoint_dir.iterdir():
            if path.suffix in (".jsonl", ".json"):
                path.unlink()
                removed += 1

        # Remove dir if empty
        with contextlib.suppress(OSError):
            self.checkpoint_dir.rmdir()

        if removed:
            logger.info(f"Cleaned up {removed} checkpoint files")


def _config_hash(config: dict[str, Any]) -> str:
    """Deterministic hash of config dict, excluding output path."""
    filtered = {k: v for k, v in config.items() if k != "output"}
    serialized = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
