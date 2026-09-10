"""Tests for data utility functions."""

import json
from pathlib import Path

import pytest

from dte.utils.data_utils import (
    deduplicate_data,
    filter_by_length,
    load_jsonl,
    merge_datasets,
    sample_balanced,
    save_jsonl,
    split_dataset,
    validate_data_format,
)


class TestLoadSaveJsonl:
    """Tests for JSONL load/save."""

    def test_roundtrip(self, tmp_path):
        data = [
            {"question": "2+2?", "answer": "4"},
            {"question": "3+3?", "answer": "6"},
        ]
        path = str(tmp_path / "test.jsonl")
        save_jsonl(data, path)
        loaded = load_jsonl(path)
        assert loaded == data

    def test_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        save_jsonl([], path)
        loaded = load_jsonl(path)
        assert loaded == []

    def test_unicode(self, tmp_path):
        data = [{"text": "Hello, world!"}]
        path = str(tmp_path / "unicode.jsonl")
        save_jsonl(data, path)
        loaded = load_jsonl(path)
        assert loaded[0]["text"] == "Hello, world!"

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "file.jsonl")
        save_jsonl([{"a": 1}], path)
        assert Path(path).exists()


class TestSplitDataset:
    """Tests for dataset splitting."""

    def test_default_ratios(self):
        data = list(range(100))
        train, val, test = split_dataset(data)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_custom_ratios(self):
        data = list(range(100))
        train, val, test = split_dataset(data, 0.6, 0.2, 0.2)
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_bad_ratios(self):
        with pytest.raises(AssertionError):
            split_dataset(list(range(100)), 0.5, 0.3, 0.3)

    def test_deterministic_with_seed(self):
        data = list(range(100))
        train1, _, _ = split_dataset(data, seed=42)
        train2, _, _ = split_dataset(data, seed=42)
        assert train1 == train2

    def test_no_shuffle(self):
        data = list(range(10))
        train, val, test = split_dataset(data, shuffle=False)
        assert train == [0, 1, 2, 3, 4, 5, 6, 7]
        assert val == [8]
        assert test == [9]


class TestFilterByLength:
    """Tests for length-based filtering."""

    def test_filter_short(self):
        data = [
            {"text": "hi"},
            {"text": "a" * 50},
            {"text": "a" * 500},
        ]
        result = filter_by_length(data, min_length=10, max_length=100)
        assert len(result) == 1
        assert len(result[0]["text"]) == 50

    def test_missing_key(self):
        data = [{"other": "value"}]
        result = filter_by_length(data, text_key="text")
        assert len(result) == 0


class TestDeduplicateData:
    """Tests for deduplication."""

    def test_removes_duplicates(self):
        data = [
            {"text": "hello"},
            {"text": "world"},
            {"text": "hello"},
        ]
        result = deduplicate_data(data)
        assert len(result) == 2

    def test_keeps_first_occurrence(self):
        data = [
            {"text": "first", "id": 1},
            {"text": "first", "id": 2},
        ]
        result = deduplicate_data(data)
        assert result[0]["id"] == 1

    def test_no_duplicates(self):
        data = [{"text": "a"}, {"text": "b"}]
        result = deduplicate_data(data)
        assert len(result) == 2


class TestSampleBalanced:
    """Tests for balanced sampling."""

    def test_balanced_sampling(self):
        data = [{"label": "A", "val": i} for i in range(10)] + [{"label": "B", "val": i} for i in range(10)]
        result = sample_balanced(data, "label", 3)
        label_counts = {}
        for item in result:
            label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1
        assert label_counts["A"] == 3
        assert label_counts["B"] == 3

    def test_insufficient_samples(self):
        data = [{"label": "A"}, {"label": "B"}, {"label": "B"}]
        result = sample_balanced(data, "label", 5)
        assert len(result) == 3  # Only 1 A and 2 B available


class TestValidateDataFormat:
    """Tests for data format validation."""

    def test_valid_data(self):
        data = [{"question": "q", "answer": "a"}]
        errors = validate_data_format(data, ["question", "answer"])
        assert errors == []

    def test_missing_key(self):
        data = [{"question": "q"}]
        errors = validate_data_format(data, ["question", "answer"])
        assert len(errors) == 1
        assert "answer" in errors[0]

    def test_non_dict_item(self):
        data = ["not a dict"]
        errors = validate_data_format(data, ["key"])
        assert len(errors) == 1
        assert "not a dictionary" in errors[0]


class TestMergeDatasets:
    """Tests for dataset merging."""

    def test_merge(self):
        ds1 = [{"a": 1}]
        ds2 = [{"b": 2}]
        merged = merge_datasets([ds1, ds2])
        assert len(merged) == 2

    def test_source_info(self):
        ds1 = [{"a": 1}]
        ds2 = [{"b": 2}]
        merged = merge_datasets([ds1, ds2], add_source_info=True)
        assert merged[0]["source_dataset_idx"] == 0
        assert merged[1]["source_dataset_idx"] == 1

    def test_no_source_info(self):
        ds1 = [{"a": 1}]
        merged = merge_datasets([ds1], add_source_info=False)
        assert "source_dataset_idx" not in merged[0]
