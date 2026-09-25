"""Dataset management utilities for DTE pipeline."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset, load_dataset

from .generator import TrainingExample


class DatasetManager:
    """Manages dataset loading and preprocessing for DTE pipeline."""

    # Dataset configurations matching original DTE implementation.
    # Includes all 7 supported benchmarks (5 original + GPQA + CommonsenseQA).
    DATASET_CONFIGS = {
        "gsm8k": {
            "hf_name": "gsm8k",
            "hf_config": "main",
            "question_field": "question",
            "answer_field": "answer",
            "task_type": "math",
        },
        "gsm_plus": {
            "hf_name": "qintongli/GSM-Plus",
            "hf_config": None,
            "question_field": "question",
            "answer_field": "answer",
            "task_type": "math",
        },
        "math": {
            "hf_name": "hendrycks/competition_math",
            "hf_config": None,
            "question_field": "problem",
            "answer_field": "solution",
            "task_type": "math",
        },
        "arc_challenge": {
            "hf_name": "allenai/ai2_arc",
            "hf_config": "ARC-Challenge",
            "question_field": "question",
            "answer_field": "answerKey",
            "task_type": "arc",
            "choices_field": "choices",
        },
        "arc_easy": {
            "hf_name": "allenai/ai2_arc",
            "hf_config": "ARC-Easy",
            "question_field": "question",
            "answer_field": "answerKey",
            "task_type": "arc",
            "choices_field": "choices",
        },
        "gpqa": {
            "hf_name": "Idavidrein/gpqa",
            "hf_config": "gpqa_main",
            "question_field": "question",
            "answer_field": "answer",
            "task_type": "general",
        },
        "commonsense_qa": {
            "hf_name": "tau/commonsense_qa",
            "hf_config": None,
            "question_field": "question",
            "answer_field": "answerKey",
            "task_type": "arc",
            "choices_field": "choices",
        },
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize dataset manager.

        Args:
            cache_dir: Optional directory for caching datasets
        """
        self.loaded_datasets = {}
        self.cache_dir = cache_dir

    def load_dataset_by_name(
        self, name: str, split: str = "train", max_samples: Optional[int] = None, force_reload: bool = False
    ) -> Dataset:
        """Load a dataset by name using HuggingFace datasets.

        Args:
            name: Dataset name (gsm8k, gsm_plus, math, arc_challenge, arc_easy)
            split: Dataset split (train, test, validation)
            max_samples: Maximum number of samples to load
            force_reload: Force reload even if cached

        Returns:
            Loaded dataset

        Raises:
            ValueError: If dataset name is not supported
            RuntimeError: If dataset loading fails
        """
        # Check if dataset is supported
        if name not in self.DATASET_CONFIGS:
            available = list(self.DATASET_CONFIGS.keys())
            raise ValueError(f"Dataset '{name}' not supported. Available: {available}")

        # Check cache
        cache_key = f"{name}_{split}_{max_samples}"
        if not force_reload and cache_key in self.loaded_datasets:
            return self.loaded_datasets[cache_key]

        config = self.DATASET_CONFIGS[name]

        try:
            # Load from HuggingFace
            if config["hf_config"]:
                dataset = load_dataset(config["hf_name"], config["hf_config"], split=split, cache_dir=self.cache_dir)
            else:
                dataset = load_dataset(config["hf_name"], split=split, cache_dir=self.cache_dir)

            # Limit samples if requested
            if max_samples is not None and max_samples > 0:
                total_samples = len(dataset)
                actual_samples = min(max_samples, total_samples)
                dataset = dataset.select(range(actual_samples))

            # Cache the dataset
            self.loaded_datasets[cache_key] = dataset

            return dataset

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{name}': {str(e)}")

    def preprocess_dataset(self, dataset: Dataset, dataset_name: str) -> Dataset:
        """Preprocess dataset for training format.

        Args:
            dataset: Raw dataset from HuggingFace
            dataset_name: Name of the dataset for preprocessing rules

        Returns:
            Preprocessed dataset with standardized fields

        Raises:
            ValueError: If dataset name is not supported
        """
        if dataset_name not in self.DATASET_CONFIGS:
            available = list(self.DATASET_CONFIGS.keys())
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")

        config = self.DATASET_CONFIGS[dataset_name]

        def preprocess_sample(sample):
            """Preprocess a single sample."""
            # Extract basic fields
            processed = {
                "query": sample[config["question_field"]],
                "ground_truth": sample[config["answer_field"]],
                "task_type": config["task_type"],
                "dataset_name": dataset_name,
            }

            # Handle ARC-specific preprocessing (multiple choice)
            if config["task_type"] == "arc" and "choices_field" in config:
                choices = sample[config["choices_field"]]
                processed["choices"] = choices

                # Format the question with choices for better prompting
                if isinstance(choices, dict) and "text" in choices and "label" in choices:
                    choices_text = ""
                    for label, choice in zip(choices["label"], choices["text"]):
                        choices_text += f"{label}. {choice}\n"
                    processed["formatted_query"] = f"{processed['query']}\n\nChoices:\n{choices_text}"
                else:
                    processed["formatted_query"] = processed["query"]
            else:
                processed["formatted_query"] = processed["query"]

            # Handle math problems - clean up answer format
            if config["task_type"] == "math":
                # Clean up math answers (remove extra formatting)
                answer = processed["ground_truth"]
                if isinstance(answer, str):
                    # Extract numerical answer if it's in #### format
                    if "####" in answer:
                        processed["ground_truth"] = answer.split("####")[-1].strip()

            return processed

        # Apply preprocessing
        processed_dataset = dataset.map(preprocess_sample)

        return processed_dataset

    def load_from_file(self, file_path: str, format: str = "auto") -> Dataset:
        """Load dataset from local file.

        Args:
            file_path: Path to dataset file
            format: File format (json, jsonl, csv, auto)

        Returns:
            Loaded dataset

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Auto-detect format
        if format == "auto":
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".json":
                format = "json"
            elif ext == ".jsonl":
                format = "jsonl"
            elif ext == ".csv":
                format = "csv"
            else:
                raise ValueError(f"Cannot auto-detect format for file: {file_path}")

        # Load based on format
        if format == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)
        elif format == "jsonl":
            data = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            dataset = Dataset.from_list(data)
        elif format == "csv":
            dataset = Dataset.from_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return dataset

    def save_dataset(self, dataset: Dataset, file_path: str, format: str = "json"):
        """Save dataset to file.

        Args:
            dataset: Dataset to save
            file_path: Output file path
            format: Output format (json, jsonl, csv)

        Raises:
            ValueError: If format is not supported
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if format == "json":
            data = [sample for sample in dataset]
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == "jsonl":
            with open(file_path, "w", encoding="utf-8") as f:
                for sample in dataset:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        elif format == "csv":
            dataset.to_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get information about a supported dataset.

        Args:
            name: Dataset name

        Returns:
            Dataset configuration and info

        Raises:
            ValueError: If dataset name is not supported
        """
        if name not in self.DATASET_CONFIGS:
            available = list(self.DATASET_CONFIGS.keys())
            raise ValueError(f"Dataset '{name}' not supported. Available: {available}")

        config = self.DATASET_CONFIGS[name].copy()

        # Add additional info
        config["supported_splits"] = ["train", "test", "validation"]
        config["description"] = self._get_dataset_description(name)

        return config

    def _get_dataset_description(self, name: str) -> str:
        """Get human-readable description of a supported dataset.

        Args:
            name: Dataset identifier.

        Returns:
            Human-readable description string.
        """
        descriptions = {
            "gsm8k": "Grade School Math 8K: Math word problems for elementary school students",
            "gsm_plus": "GSM8K-Plus: Extended version of GSM8K with additional problems",
            "math": "MATH: Competition mathematics problems from high school competitions",
            "arc_challenge": "ARC Challenge: Science questions requiring complex reasoning",
            "arc_easy": "ARC Easy: Easier science questions from ARC dataset",
            "gpqa": "GPQA: Graduate-level questions requiring domain expertise",
            "commonsense_qa": "CommonsenseQA: Questions requiring commonsense reasoning",
        }
        return descriptions.get(name, "No description available")

    def list_supported_datasets(self) -> List[str]:
        """List all supported dataset names.

        Returns:
            List of supported dataset names
        """
        return list(self.DATASET_CONFIGS.keys())

    def clear_cache(self):
        """Clear cached datasets from memory."""
        self.loaded_datasets.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached datasets.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_datasets": list(self.loaded_datasets.keys()),
            "cache_size": len(self.loaded_datasets),
            "memory_usage": sum(len(ds) for ds in self.loaded_datasets.values()),
        }
