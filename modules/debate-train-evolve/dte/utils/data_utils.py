"""Data processing utilities."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of JSON objects
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file.

    Args:
        data: List of JSON objects to save
        file_path: Path to save the file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def split_dataset(
    data: List[Any],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Split dataset into train, validation, and test sets.

    Args:
        data: List of data items
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        shuffle: Whether to shuffle data before splitting
        seed: Random seed for shuffling

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    if shuffle:
        random.seed(seed)
        data = data.copy()
        random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def filter_by_length(
    data: List[Dict[str, Any]], min_length: int = 10, max_length: int = 1000, text_key: str = "text"
) -> List[Dict[str, Any]]:
    """Filter data by text length.

    Args:
        data: List of data items with text
        min_length: Minimum text length (in characters)
        max_length: Maximum text length (in characters)
        text_key: Key containing the text to filter

    Returns:
        Filtered data list
    """
    filtered = []
    for item in data:
        if text_key in item:
            text_len = len(item[text_key])
            if min_length <= text_len <= max_length:
                filtered.append(item)
    return filtered


def deduplicate_data(data: List[Dict[str, Any]], key: str = "text") -> List[Dict[str, Any]]:
    """Remove duplicate entries based on a key.

    Args:
        data: List of data items
        key: Key to use for deduplication

    Returns:
        Deduplicated data list
    """
    seen = set()
    deduplicated = []

    for item in data:
        if key in item:
            text = item[key]
            if text not in seen:
                seen.add(text)
                deduplicated.append(item)
        else:
            # Keep items without the key
            deduplicated.append(item)

    return deduplicated


def sample_balanced(data: List[Dict[str, Any]], label_key: str, samples_per_class: int) -> List[Dict[str, Any]]:
    """Sample balanced data across different classes.

    Args:
        data: List of data items with labels
        label_key: Key containing the class labels
        samples_per_class: Number of samples per class

    Returns:
        Balanced sample of data
    """
    # Group by label
    label_groups = {}
    for item in data:
        if label_key in item:
            label = item[label_key]
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)

    # Sample from each group
    balanced_data = []
    for label, items in label_groups.items():
        if len(items) >= samples_per_class:
            sampled = random.sample(items, samples_per_class)
        else:
            # If not enough samples, take all available
            sampled = items
        balanced_data.extend(sampled)

    # Shuffle the final result
    random.shuffle(balanced_data)
    return balanced_data


def validate_data_format(data: List[Dict[str, Any]], required_keys: List[str]) -> List[str]:
    """Validate that data has required format.

    Args:
        data: List of data items to validate
        required_keys: List of required keys

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Item {i} is not a dictionary")
            continue

        for key in required_keys:
            if key not in item:
                errors.append(f"Item {i} missing required key: {key}")

    return errors


def merge_datasets(datasets: List[List[Dict[str, Any]]], add_source_info: bool = True) -> List[Dict[str, Any]]:
    """Merge multiple datasets into one.

    Args:
        datasets: List of datasets to merge
        add_source_info: Whether to add source dataset index

    Returns:
        Merged dataset
    """
    merged = []

    for i, dataset in enumerate(datasets):
        for item in dataset:
            if add_source_info:
                item = item.copy()  # Don't modify original
                item["source_dataset_idx"] = i
            merged.append(item)

    return merged
