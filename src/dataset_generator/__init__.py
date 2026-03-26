"""Generate high-quality synthetic datasets for ML training using any LLM."""

__version__ = "0.2.0"

from dataset_generator.config import load_config
from dataset_generator.engine import async_generate, generate
from dataset_generator.formats import write_output as export

__all__ = ["async_generate", "export", "generate", "load_config"]
