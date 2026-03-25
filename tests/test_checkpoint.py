"""Tests for checkpoint/resume support."""

from __future__ import annotations

from pathlib import Path

from dataset_generator.checkpoint import CheckpointManager
from dataset_generator.tasks.base import Sample


class TestCheckpointManager:
    def test_save_load_roundtrip(self, tmp_path: Path):
        """Given saved samples, loading with the same config should recover them."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
        config = {"task": {"type": "classification"}, "count": 100}
        samples = [
            Sample(text="hello world", label="positive"),
            Sample(text="goodbye", label="negative", metadata={"extra": "data"}),
        ]

        # When: save then load
        mgr.save(samples, batch_index=2, config=config)
        result = mgr.load(config)

        # Then: should recover samples and batch index
        assert result is not None
        loaded_samples, batch_index = result
        assert batch_index == 2
        assert len(loaded_samples) == 2
        assert loaded_samples[0].text == "hello world"
        assert loaded_samples[0].label == "positive"
        assert loaded_samples[1].text == "goodbye"
        assert loaded_samples[1].label == "negative"

    def test_load_returns_none_when_no_checkpoint(self, tmp_path: Path):
        """Given no prior save, load should return None."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path / "empty_checkpoints")
        config = {"task": {"type": "ner"}}

        result = mgr.load(config)

        assert result is None

    def test_cleanup_removes_files(self, tmp_path: Path):
        """Given saved checkpoints, cleanup should remove all checkpoint files."""
        ckpt_dir = tmp_path / "checkpoints"
        mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
        config = {"task": {"type": "qa"}}
        samples = [Sample(text="question?", label="answer")]

        mgr.save(samples, batch_index=0, config=config)
        # Verify files exist
        assert any(ckpt_dir.iterdir())

        # When: cleanup
        mgr.cleanup()

        # Then: directory should be removed (or empty)
        assert not ckpt_dir.exists()

    def test_append_across_batches(self, tmp_path: Path):
        """Given multiple saves, load should return all accumulated samples."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
        config = {"task": {"type": "sft"}}

        batch_0 = [Sample(text="batch zero sample")]
        batch_1 = [Sample(text="batch one sample")]

        mgr.save(batch_0, batch_index=0, config=config)
        mgr.save(batch_1, batch_index=1, config=config)

        result = mgr.load(config)
        assert result is not None
        loaded_samples, batch_index = result
        assert batch_index == 1
        assert len(loaded_samples) == 2
        assert loaded_samples[0].text == "batch zero sample"
        assert loaded_samples[1].text == "batch one sample"

    def test_different_config_different_checkpoint(self, tmp_path: Path):
        """Given different configs, checkpoints should be independent."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
        config_a = {"task": {"type": "classification"}}
        config_b = {"task": {"type": "ner"}}

        mgr.save([Sample(text="a")], batch_index=0, config=config_a)

        # Config B should have no checkpoint
        assert mgr.load(config_b) is None

        # Config A should have its checkpoint
        assert mgr.load(config_a) is not None
