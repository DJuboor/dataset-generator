"""Tests for output format serialization."""

import json
from unittest.mock import patch

import pytest

from dataset_generator.formats import (
    WRITERS,
    _sample_to_messages,
    read_jsonl,
    write_alpaca,
    write_csv,
    write_jsonl,
    write_openai_finetune,
    write_output,
    write_sharegpt,
)
from dataset_generator.tasks.base import Sample


class TestJSONL:
    def test_write_and_read(self, tmp_path):
        samples = [
            Sample(text="hello", label="a"),
            Sample(text="world", label="b", metadata={"key": "val"}),
        ]
        path = tmp_path / "out.jsonl"

        write_jsonl(samples, path)

        # Verify file content
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"text": "hello", "label": "a"}
        assert json.loads(lines[1]) == {"text": "world", "label": "b", "key": "val"}

        # Round-trip
        loaded = read_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0].text == "hello"
        assert loaded[1].metadata == {"key": "val"}

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "out.jsonl"
        write_jsonl([Sample(text="test", label="x")], path)
        assert path.exists()


class TestCSV:
    def test_write_csv(self, tmp_path):
        samples = [
            Sample(text="hello", label="a"),
            Sample(text="world", label="b"),
        ]
        path = tmp_path / "out.csv"
        write_csv(samples, path)

        content = path.read_text()
        assert "text,label" in content
        assert "hello,a" in content

    def test_write_csv_empty(self, tmp_path):
        path = tmp_path / "empty.csv"
        write_csv([], path)
        assert not path.exists()

    def test_write_csv_with_complex_metadata(self, tmp_path):
        samples = [
            Sample(text="test", label="a", metadata={"nested": {"x": 1}, "tags": ["a", "b"]}),
        ]
        path = tmp_path / "complex.csv"
        write_csv(samples, path)

        content = path.read_text()
        assert "test" in content
        # Dicts and lists should be JSON-serialized (CSV escapes inner quotes)
        assert "x" in content
        assert "1" in content


class TestOpenAIFinetune:
    def test_single_turn(self, tmp_path):
        samples = [
            Sample(text="What is Python?", metadata={"response": "A programming language."}),
        ]
        path = tmp_path / "ft.jsonl"
        write_openai_finetune(samples, path)

        lines = path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert "messages" in record
        msgs = record["messages"]
        assert msgs[0]["role"] == "user"
        assert "Python" in msgs[0]["content"]
        assert msgs[1]["role"] == "assistant"

    def test_with_system_prompt(self, tmp_path):
        samples = [
            Sample(
                text="Translate hello",
                metadata={"system_prompt": "You are a translator.", "response": "Hola"},
            ),
        ]
        path = tmp_path / "ft.jsonl"
        write_openai_finetune(samples, path)

        record = json.loads(path.read_text().strip())
        msgs = record["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_multi_turn_passthrough(self, tmp_path):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        samples = [Sample(text="Hi", metadata={"messages": messages})]
        path = tmp_path / "ft.jsonl"
        write_openai_finetune(samples, path)

        record = json.loads(path.read_text().strip())
        assert record["messages"] == messages


class TestAlpaca:
    def test_basic(self, tmp_path):
        samples = [
            Sample(
                text="Explain ML", metadata={"instruction": "Explain ML", "response": "ML is..."}
            ),
        ]
        path = tmp_path / "alpaca.json"
        write_alpaca(samples, path)

        records = json.loads(path.read_text())
        assert len(records) == 1
        assert records[0]["instruction"] == "Explain ML"
        assert records[0]["output"] == "ML is..."
        assert records[0]["input"] == ""

    def test_fallback_to_text_and_label(self, tmp_path):
        samples = [Sample(text="What is X?", label="A thing")]
        path = tmp_path / "alpaca.json"
        write_alpaca(samples, path)

        records = json.loads(path.read_text())
        assert records[0]["instruction"] == "What is X?"
        assert records[0]["output"] == "A thing"


class TestShareGPT:
    def test_single_turn(self, tmp_path):
        samples = [
            Sample(text="Hello", metadata={"instruction": "Hello", "response": "Hi!"}),
        ]
        path = tmp_path / "sharegpt.jsonl"
        write_sharegpt(samples, path)

        record = json.loads(path.read_text().strip())
        convs = record["conversations"]
        assert convs[0] == {"from": "human", "value": "Hello"}
        assert convs[1] == {"from": "gpt", "value": "Hi!"}

    def test_multi_turn(self, tmp_path):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Goodbye!"},
        ]
        samples = [Sample(text="Hi", metadata={"messages": messages})]
        path = tmp_path / "sharegpt.jsonl"
        write_sharegpt(samples, path)

        record = json.loads(path.read_text().strip())
        convs = record["conversations"]
        assert len(convs) == 4
        assert convs[0]["from"] == "human"
        assert convs[1]["from"] == "gpt"

    def test_single_turn_no_response(self, tmp_path):
        samples = [Sample(text="Just a question")]
        path = tmp_path / "sharegpt.jsonl"
        write_sharegpt(samples, path)

        record = json.loads(path.read_text().strip())
        assert len(record["conversations"]) == 1


class TestSampleToMessages:
    def test_basic_text(self):
        msgs = _sample_to_messages({"text": "Hello"})
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "Hello"}

    def test_with_input_field(self):
        msgs = _sample_to_messages({"text": "Translate", "input": "Hello"})
        assert "Hello" in msgs[0]["content"]
        assert "Translate" in msgs[0]["content"]

    def test_passthrough_messages(self):
        orig = [{"role": "user", "content": "Hi"}]
        msgs = _sample_to_messages({"messages": orig})
        assert msgs is orig


class TestWriteOutput:
    def test_dispatch_jsonl(self, tmp_path):
        path = tmp_path / "out.jsonl"
        write_output([Sample(text="test", label="a")], path, fmt="jsonl")
        assert path.exists()

    def test_unknown_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown format"):
            write_output([], tmp_path / "out.xyz", fmt="badformat")

    def test_all_writers_registered(self):
        expected = {"jsonl", "csv", "huggingface", "parquet", "openai", "alpaca", "sharegpt"}
        assert set(WRITERS.keys()) == expected


class TestHuggingFaceParquet:
    def test_huggingface_import_error(self, tmp_path):
        with (
            patch.dict("sys.modules", {"datasets": None}),
            pytest.raises(ImportError, match="huggingface"),
        ):
            from dataset_generator.formats import write_huggingface

            write_huggingface([Sample(text="test")], tmp_path / "hf")

    def test_parquet_import_error(self, tmp_path):
        with (
            patch.dict("sys.modules", {"datasets": None}),
            pytest.raises(ImportError, match="huggingface"),
        ):
            from dataset_generator.formats import write_parquet

            write_parquet([Sample(text="test")], tmp_path / "out.parquet")


class TestReadCSV:
    def test_read_csv(self, tmp_path):
        from dataset_generator.formats import read_csv

        path = tmp_path / "data.csv"
        path.write_text("text,label\nhello,a\nworld,b\n")

        samples = read_csv(path)
        assert len(samples) == 2
        assert samples[0].text == "hello"
        assert samples[0].label == "a"

    def test_read_csv_with_extra_fields(self, tmp_path):
        from dataset_generator.formats import read_csv

        path = tmp_path / "data.csv"
        path.write_text("text,label,source\nhello,a,web\n")

        samples = read_csv(path)
        assert samples[0].metadata == {"source": "web"}


class TestReadSamples:
    def test_dispatches_jsonl(self, tmp_path):
        from dataset_generator.formats import read_samples

        path = tmp_path / "data.jsonl"
        path.write_text('{"text": "hi", "label": "a"}\n')

        samples = read_samples(path)
        assert len(samples) == 1

    def test_dispatches_csv(self, tmp_path):
        from dataset_generator.formats import read_samples

        path = tmp_path / "data.csv"
        path.write_text("text,label\nhello,a\n")

        samples = read_samples(path)
        assert len(samples) == 1
        assert samples[0].text == "hello"


class TestReadJSONL:
    def test_handles_blank_lines(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text('{"text": "a", "label": "x"}\n\n{"text": "b"}\n')

        loaded = read_jsonl(path)
        assert len(loaded) == 2

    def test_extra_fields_become_metadata(self, tmp_path):
        path = tmp_path / "extra.jsonl"
        path.write_text('{"text": "hello", "label": "a", "source": "web", "score": 0.9}\n')

        loaded = read_jsonl(path)
        assert loaded[0].metadata == {"source": "web", "score": 0.9}


class TestSampleModel:
    def test_to_dict_excludes_none(self):
        s = Sample(text="hello")
        d = s.to_dict()
        assert "label" not in d
        assert "metadata" not in d

    def test_to_dict_includes_values(self):
        s = Sample(text="hello", label="a", metadata={"k": "v"})
        d = s.to_dict()
        assert d == {"text": "hello", "label": "a", "k": "v"}
