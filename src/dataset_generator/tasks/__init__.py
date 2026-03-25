"""Built-in dataset generation task types."""

from dataset_generator.tasks.base import Sample, Task
from dataset_generator.tasks.classification import ClassificationTask
from dataset_generator.tasks.conversation import ConversationTask
from dataset_generator.tasks.distillation import DistillationTask
from dataset_generator.tasks.ner import NERTask
from dataset_generator.tasks.preference import PreferenceTask
from dataset_generator.tasks.qa import QATask
from dataset_generator.tasks.sft import SFTTask
from dataset_generator.tasks.summarization import SummarizationTask

__all__ = [
    "ClassificationTask",
    "ConversationTask",
    "DistillationTask",
    "NERTask",
    "PreferenceTask",
    "QATask",
    "SFTTask",
    "Sample",
    "SummarizationTask",
    "Task",
    "create_task",
]

TASK_REGISTRY: dict[str, type[Task]] = {
    "classification": ClassificationTask,
    "conversation": ConversationTask,
    "distillation": DistillationTask,
    "ner": NERTask,
    "qa": QATask,
    "preference": PreferenceTask,
    "sft": SFTTask,
    "summarization": SummarizationTask,
}


def create_task(config: dict) -> Task:
    """Create a task from config dict.

    Args:
        config: Task config with 'type' and task-specific params.

    Returns:
        Configured Task instance.
    """
    task_type = config.get("type", "classification")
    if task_type not in TASK_REGISTRY:
        raise ValueError(f"Unknown task type: {task_type}. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_type].from_config(config)
