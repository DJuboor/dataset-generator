"""Tests for task types — message building and response parsing."""

from __future__ import annotations

import json

from dataset_generator.tasks import create_task
from dataset_generator.tasks.base import clean_llm_response
from dataset_generator.tasks.classification import ClassificationTask
from dataset_generator.tasks.conversation import ConversationTask
from dataset_generator.tasks.distillation import DistillationTask
from dataset_generator.tasks.ner import NERTask
from dataset_generator.tasks.preference import PreferenceTask
from dataset_generator.tasks.qa import QATask
from dataset_generator.tasks.sft import SFTTask
from dataset_generator.tasks.summarization import SummarizationTask


class TestCleanLLMResponse:
    def test_strips_think_tags(self):
        raw = '<think>\nsome reasoning\n</think>\n[{"text": "hello", "label": "a"}]'
        assert clean_llm_response(raw) == '[{"text": "hello", "label": "a"}]'

    def test_strips_reasoning_tags(self):
        raw = '<reasoning>step by step</reasoning>[{"a": 1}]'
        assert clean_llm_response(raw) == '[{"a": 1}]'

    def test_strips_eos_tokens(self):
        raw = '[{"a": 1}]<|endoftext|><|im_start|>user\nmore stuff'
        assert clean_llm_response(raw) == '[{"a": 1}]'

    def test_strips_markdown_code_blocks(self):
        raw = '```json\n[{"a": 1}]\n```'
        assert clean_llm_response(raw) == '[{"a": 1}]'

    def test_handles_think_plus_markdown(self):
        raw = '<think>\nplanning...\n</think>\n```json\n[{"x": 1}]\n```<|endoftext|>'
        assert clean_llm_response(raw) == '[{"x": 1}]'

    def test_plain_json_unchanged(self):
        raw = '[{"text": "hello"}]'
        assert clean_llm_response(raw) == '[{"text": "hello"}]'

    def test_classification_with_think_tags(self):
        """Integration: a classification task should parse responses with think tags."""
        task = ClassificationTask(labels=["positive", "negative"])
        response = '<think>\nLet me generate examples...\n</think>\n[{"text": "Great product!", "label": "positive"}]'
        samples = task.parse_response(response)
        assert len(samples) == 1
        assert samples[0].label == "positive"


class TestClassificationTask:
    def setup_method(self):
        self.task = ClassificationTask(
            labels=["positive", "negative"],
            domain="reviews",
        )

    def test_build_messages(self):
        msgs = self.task.build_messages(batch_size=5)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "5" in msgs[1]["content"]
        assert "positive" in msgs[1]["content"]

    def test_parse_valid_response(self):
        response = json.dumps(
            [
                {"text": "Great product!", "label": "positive"},
                {"text": "Terrible quality", "label": "negative"},
            ]
        )
        samples = self.task.parse_response(response)
        assert len(samples) == 2
        assert samples[0].label == "positive"

    def test_parse_filters_invalid_labels(self):
        response = json.dumps(
            [
                {"text": "text", "label": "positive"},
                {"text": "text", "label": "unknown"},
            ]
        )
        samples = self.task.parse_response(response)
        assert len(samples) == 1

    def test_parse_markdown_code_block(self):
        response = '```json\n[{"text": "hello", "label": "positive"}]\n```'
        samples = self.task.parse_response(response)
        assert len(samples) == 1

    def test_parse_invalid_json(self):
        samples = self.task.parse_response("not json at all")
        assert samples == []

    def test_from_config_string_labels(self):
        task = ClassificationTask.from_config({"task": {"labels": "a, b, c"}})
        assert task.labels == ["a", "b", "c"]


class TestNERTask:
    def test_build_messages(self):
        task = NERTask(entity_types=["PERSON", "ORG"])
        msgs = task.build_messages(batch_size=3)
        assert "PERSON" in msgs[1]["content"]

    def test_build_messages_with_domain(self):
        task = NERTask(entity_types=["PERSON"], domain="medical")
        msgs = task.build_messages(batch_size=2)
        assert "medical" in msgs[1]["content"]

    def test_parse_valid_entities(self):
        task = NERTask(entity_types=["PERSON"])
        response = json.dumps(
            [
                {
                    "text": "John works here",
                    "entities": [{"text": "John", "label": "PERSON", "start": 0, "end": 4}],
                }
            ]
        )
        samples = task.parse_response(response)
        assert len(samples) == 1
        assert len(samples[0].metadata["entities"]) == 1

    def test_parse_filters_bad_offsets(self):
        task = NERTask(entity_types=["PERSON"])
        response = json.dumps(
            [
                {
                    "text": "John works here",
                    "entities": [{"text": "Jane", "label": "PERSON", "start": 0, "end": 4}],
                }
            ]
        )
        samples = task.parse_response(response)
        assert len(samples) == 1
        assert len(samples[0].metadata["entities"]) == 0

    def test_from_config_string_entity_types(self):
        task = NERTask.from_config({"task": {"entity_types": "PERSON, ORG, LOC"}})
        assert task.entity_types == ["PERSON", "ORG", "LOC"]

    def test_from_config_list_entity_types(self):
        task = NERTask.from_config({"task": {"entity_types": ["PERSON", "ORG"]}})
        assert task.entity_types == ["PERSON", "ORG"]

    def test_parse_markdown_wrapped(self):
        task = NERTask(entity_types=["PERSON"])
        response = '```json\n[{"text": "Alice runs", "entities": [{"text": "Alice", "label": "PERSON", "start": 0, "end": 5}]}]\n```'
        samples = task.parse_response(response)
        assert len(samples) == 1
        assert len(samples[0].metadata["entities"]) == 1

    def test_parse_invalid_json(self):
        task = NERTask(entity_types=["PERSON"])
        assert task.parse_response("not json") == []

    def test_parse_skips_non_dict_items(self):
        task = NERTask(entity_types=["PERSON"])
        response = json.dumps(["not a dict", {"text": "valid"}])
        samples = task.parse_response(response)
        assert len(samples) == 1

    def test_parse_entity_missing_fields(self):
        task = NERTask(entity_types=["PERSON"])
        response = json.dumps([{"text": "Hello world", "entities": [{"text": "Hello"}]}])
        samples = task.parse_response(response)
        assert len(samples) == 1
        assert len(samples[0].metadata["entities"]) == 0


class TestQATask:
    def test_parse_response(self):
        task = QATask(domain="python")
        response = json.dumps(
            [
                {
                    "question": "What is Python?",
                    "answer": "A programming language",
                    "context": "...",
                },
            ]
        )
        samples = task.parse_response(response)
        assert len(samples) == 1
        assert samples[0].text == "What is Python?"
        assert samples[0].label == "A programming language"
        assert "context" in samples[0].metadata

    def test_from_config(self):
        task = QATask.from_config({"task": {"domain": "science", "contexts": ["text1", "text2"]}})
        assert task.domain == "science"
        assert task.contexts == ["text1", "text2"]

    def test_from_config_no_contexts(self):
        task = QATask.from_config({"task": {"domain": "math"}})
        assert task.contexts is None

    def test_build_messages_with_domain(self):
        task = QATask(domain="biology")
        msgs = task.build_messages(batch_size=3)
        assert "biology" in msgs[1]["content"]

    def test_build_messages_with_contexts(self):
        task = QATask(domain="", contexts=["Reference text A", "Reference text B"])
        msgs = task.build_messages(batch_size=2)
        assert "Reference text A" in msgs[1]["content"]
        assert "Reference text B" in msgs[1]["content"]

    def test_parse_markdown_wrapped(self):
        task = QATask()
        response = '```json\n[{"question": "Q?", "answer": "A"}]\n```'
        samples = task.parse_response(response)
        assert len(samples) == 1

    def test_parse_invalid_json(self):
        task = QATask()
        assert task.parse_response("garbage") == []

    def test_parse_skips_missing_fields(self):
        task = QATask()
        response = json.dumps([{"question": "Q?"}, {"question": "Q2?", "answer": "A2"}])
        samples = task.parse_response(response)
        assert len(samples) == 1
        assert samples[0].label == "A2"

    def test_parse_without_context(self):
        task = QATask()
        response = json.dumps([{"question": "Q?", "answer": "A"}])
        samples = task.parse_response(response)
        assert samples[0].metadata == {}


class TestPreferenceTask:
    def test_parse_response(self):
        task = PreferenceTask(domain="coding")
        response = json.dumps(
            [
                {
                    "prompt": "Write hello world",
                    "chosen": "print('hello world')",
                    "rejected": "echo hello",
                }
            ]
        )
        samples = task.parse_response(response)
        assert len(samples) == 1
        assert "chosen" in samples[0].metadata
        assert "rejected" in samples[0].metadata

    def test_from_config(self):
        task = PreferenceTask.from_config({"task": {"domain": "writing", "criteria": "accuracy"}})
        assert task.domain == "writing"
        assert task.criteria == "accuracy"

    def test_build_messages_with_domain(self):
        task = PreferenceTask(domain="cooking", criteria="helpfulness")
        msgs = task.build_messages(batch_size=3)
        assert "cooking" in msgs[1]["content"]
        assert "helpfulness" in msgs[1]["content"]

    def test_build_messages_no_domain(self):
        task = PreferenceTask()
        msgs = task.build_messages(batch_size=2)
        assert len(msgs) == 2

    def test_parse_markdown_wrapped(self):
        task = PreferenceTask()
        response = '```json\n[{"prompt": "P", "chosen": "C", "rejected": "R"}]\n```'
        samples = task.parse_response(response)
        assert len(samples) == 1

    def test_parse_invalid_json(self):
        task = PreferenceTask()
        assert task.parse_response("not json") == []

    def test_parse_skips_missing_fields(self):
        task = PreferenceTask()
        response = json.dumps(
            [
                {"prompt": "P", "chosen": "C"},
                {"prompt": "P2", "chosen": "C2", "rejected": "R2"},
            ]
        )
        samples = task.parse_response(response)
        assert len(samples) == 1
        assert samples[0].text == "P2"

    def test_parse_skips_non_dict(self):
        task = PreferenceTask()
        response = json.dumps(["not a dict"])
        assert task.parse_response(response) == []


class TestSFTTask:
    def setup_method(self):
        self.task = SFTTask(domain="python", complexity="high", response_style="concise")

    def test_build_messages(self):
        msgs = self.task.build_messages(batch_size=3)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "3" in msgs[1]["content"]
        assert "python" in msgs[1]["content"].lower()

    def test_parse_valid_response(self):
        response = json.dumps(
            [
                {"instruction": "Explain decorators", "response": "Decorators wrap functions..."},
                {
                    "instruction": "What is a list comp?",
                    "response": "A concise way to create lists...",
                },
            ]
        )
        samples = self.task.parse_response(response)
        assert len(samples) == 2
        assert samples[0].text == "Explain decorators"
        assert samples[0].metadata["instruction"] == "Explain decorators"
        assert samples[0].metadata["response"] == "Decorators wrap functions..."

    def test_parse_invalid_response(self):
        # Missing required fields
        response = json.dumps([{"instruction": "only instruction, no response"}])
        samples = self.task.parse_response(response)
        assert len(samples) == 0

    def test_parse_invalid_json(self):
        samples = self.task.parse_response("not valid json")
        assert samples == []

    def test_from_config(self):
        task = SFTTask.from_config(
            {
                "task": {
                    "domain": "math",
                    "complexity": "easy",
                    "response_style": "brief",
                    "system_prompt": "Be helpful",
                }
            }
        )
        assert task.domain == "math"
        assert task.complexity == "easy"
        assert task.response_style == "brief"
        assert task.system_prompt == "Be helpful"


class TestConversationTask:
    def setup_method(self):
        self.task = ConversationTask(domain="tech support", min_turns=2, max_turns=4)

    def test_build_messages(self):
        msgs = self.task.build_messages(batch_size=2)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "2" in msgs[1]["content"]
        assert "tech support" in msgs[1]["content"].lower()

    def test_parse_valid_response(self):
        response = json.dumps(
            [
                {
                    "messages": [
                        {"role": "user", "content": "My laptop won't start"},
                        {"role": "assistant", "content": "Try holding the power button for 10s"},
                        {"role": "user", "content": "That worked!"},
                        {"role": "assistant", "content": "Glad to help!"},
                    ]
                }
            ]
        )
        samples = self.task.parse_response(response)
        assert len(samples) == 1
        assert samples[0].text == "My laptop won't start"
        assert len(samples[0].metadata["messages"]) == 4

    def test_validate_role_alternation_rejects_bad_roles(self):
        response = json.dumps(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "user", "content": "Are you there?"},
                    ]
                }
            ]
        )
        samples = self.task.parse_response(response)
        assert len(samples) == 0

    def test_rejects_conversation_starting_with_assistant(self):
        response = json.dumps(
            [
                {
                    "messages": [
                        {"role": "assistant", "content": "How can I help?"},
                        {"role": "user", "content": "Fix my code"},
                    ]
                }
            ]
        )
        samples = self.task.parse_response(response)
        assert len(samples) == 0

    def test_parse_markdown_wrapped(self):
        response = '```json\n[{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey"}]}]\n```'
        samples = self.task.parse_response(response)
        assert len(samples) == 1

    def test_parse_invalid_json(self):
        assert self.task.parse_response("not json") == []

    def test_parse_skips_missing_messages_key(self):
        response = json.dumps([{"no_messages": True}])
        assert self.task.parse_response(response) == []

    def test_parse_skips_single_message(self):
        response = json.dumps([{"messages": [{"role": "user", "content": "Hi"}]}])
        assert self.task.parse_response(response) == []

    def test_parse_skips_message_missing_content(self):
        response = json.dumps(
            [{"messages": [{"role": "user"}, {"role": "assistant", "content": "Hey"}]}]
        )
        assert self.task.parse_response(response) == []

    def test_from_config(self):
        task = ConversationTask.from_config(
            {
                "task": {
                    "domain": "cooking",
                    "min_turns": 4,
                    "max_turns": 10,
                    "system_prompt": "Be a chef",
                }
            }
        )
        assert task.domain == "cooking"
        assert task.min_turns == 4
        assert task.max_turns == 10
        assert task.system_prompt == "Be a chef"


class TestSummarizationTask:
    def setup_method(self):
        self.task = SummarizationTask(domain="news", summary_style="extractive")

    def test_build_messages(self):
        msgs = self.task.build_messages(batch_size=2)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "2" in msgs[1]["content"]
        assert "extractive" in msgs[1]["content"]

    def test_parse_valid_response(self):
        response = json.dumps(
            [
                {
                    "document": "A long news article about climate change...",
                    "summary": "Climate change is accelerating.",
                }
            ]
        )
        samples = self.task.parse_response(response)
        assert len(samples) == 1
        assert samples[0].text == "A long news article about climate change..."
        assert samples[0].label == "Climate change is accelerating."

    def test_parse_missing_fields(self):
        response = json.dumps([{"document": "only document, no summary"}])
        samples = self.task.parse_response(response)
        assert len(samples) == 0

    def test_parse_markdown_wrapped(self):
        response = '```json\n[{"document": "Text", "summary": "Short"}]\n```'
        samples = self.task.parse_response(response)
        assert len(samples) == 1

    def test_parse_invalid_json(self):
        assert self.task.parse_response("not json") == []

    def test_parse_skips_non_dict(self):
        response = json.dumps(["not a dict"])
        assert self.task.parse_response(response) == []


class TestDistillationTask:
    def setup_method(self):
        self.task = DistillationTask(domain="science", teacher_style="step_by_step")

    def test_build_messages(self):
        msgs = self.task.build_messages(batch_size=4)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "4" in msgs[1]["content"]
        assert "step_by_step" in msgs[1]["content"]

    def test_parse_valid_response(self):
        response = json.dumps(
            [
                {
                    "instruction": "Explain photosynthesis",
                    "teacher_response": "Photosynthesis is the process by which plants...",
                    "reasoning": "Let me break this down step by step...",
                }
            ]
        )
        samples = self.task.parse_response(response)
        assert len(samples) == 1
        assert samples[0].text == "Explain photosynthesis"
        assert (
            samples[0].metadata["teacher_response"]
            == "Photosynthesis is the process by which plants..."
        )
        assert samples[0].metadata["reasoning"] == "Let me break this down step by step..."

    def test_parse_missing_reasoning(self):
        response = json.dumps([{"instruction": "Explain X", "teacher_response": "X is..."}])
        samples = self.task.parse_response(response)
        assert len(samples) == 0

    def test_parse_markdown_wrapped(self):
        response = '```json\n[{"instruction": "Q", "teacher_response": "A", "reasoning": "R"}]\n```'
        samples = self.task.parse_response(response)
        assert len(samples) == 1

    def test_parse_invalid_json(self):
        assert self.task.parse_response("not json") == []

    def test_parse_skips_non_dict(self):
        response = json.dumps(["not a dict"])
        assert self.task.parse_response(response) == []


class TestRequiredKeys:
    def test_classification_keys(self):
        task = ClassificationTask(labels=["a", "b"])
        assert task.required_keys() == {"text", "label"}

    def test_ner_keys(self):
        task = NERTask(entity_types=["PERSON"])
        assert task.required_keys() == {"text"}

    def test_qa_keys(self):
        task = QATask()
        assert task.required_keys() == {"text", "label"}

    def test_sft_keys(self):
        task = SFTTask()
        assert "instruction" in task.required_keys()
        assert "response" in task.required_keys()

    def test_conversation_keys(self):
        task = ConversationTask()
        assert "messages" in task.required_keys()


class TestValidateSampleSchema:
    def test_valid(self):
        from dataset_generator.tasks.base import validate_sample_schema

        assert validate_sample_schema({"text": "hi", "label": "a"}, {"text", "label"})

    def test_missing_key(self):
        from dataset_generator.tasks.base import validate_sample_schema

        assert not validate_sample_schema({"text": "hi"}, {"text", "label"})

    def test_empty_value(self):
        from dataset_generator.tasks.base import validate_sample_schema

        assert not validate_sample_schema({"text": "", "label": "a"}, {"text", "label"})


class TestCreateTask:
    def test_creates_classification(self):
        task = create_task({"type": "classification", "task": {"labels": ["a", "b"]}})
        assert isinstance(task, ClassificationTask)

    def test_creates_sft(self):
        task = create_task({"type": "sft", "task": {"domain": "code"}})
        assert isinstance(task, SFTTask)

    def test_creates_conversation(self):
        task = create_task({"type": "conversation", "task": {"min_turns": 3}})
        assert isinstance(task, ConversationTask)

    def test_creates_summarization(self):
        task = create_task({"type": "summarization", "task": {"summary_style": "abstractive"}})
        assert isinstance(task, SummarizationTask)

    def test_creates_distillation(self):
        task = create_task({"type": "distillation", "task": {"teacher_style": "concise"}})
        assert isinstance(task, DistillationTask)

    def test_unknown_type_raises(self):
        import pytest

        with pytest.raises(ValueError):
            create_task({"type": "unknown"})
