"""
Debate data generation system.

This module generates training data from multi-agent debates, implementing
the data collection and filtering pipeline described in the DTE paper.
"""

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset

from ..core.logger import DTELogger
from ..debate.manager import DebateManager, DebateResult


@dataclass
class TrainingExample:
    """Structured training example generated from debate."""

    query: str
    answer: str
    reasoning: str
    confidence: float
    source_dataset: str
    debate_rounds: int
    consensus_reached: bool
    metadata: Dict[str, Any]


class DebateDataGenerator:
    """Generates training data from multi-agent debates.

    This class orchestrates the debate data generation process, managing
    datasets, conducting debates, and filtering/processing the results
    into high-quality training examples.
    """

    def __init__(self, config_datasets, config_debate, config_model, logger: Optional[DTELogger] = None):
        """Initialize debate data generator.

        Args:
            config_datasets: Dataset configuration
            config_debate: Debate configuration
            config_model: Model configuration
            logger: Optional logger for tracking progress
        """
        self.config_datasets = config_datasets
        self.config_debate = config_debate
        self.config_model = config_model
        self.logger = logger

        # Initialize debate manager
        self.debate_manager = DebateManager(config_debate, config_model, logger)

        # Data storage
        self.generated_examples: List[TrainingExample] = []
        self.debate_results: List[DebateResult] = []

        # Quality filters
        self.quality_filters = {
            "min_consensus_confidence": 0.7,
            "min_reasoning_length": 50,
            "max_reasoning_length": 1000,
            "require_consensus": True,
            "filter_error_responses": True,
        }

    def generate_training_data(
        self, num_samples: int, evolution_round: int = 0, save_path: Optional[str] = None
    ) -> List[TrainingExample]:
        """Generate training data through multi-agent debates.

        Args:
            num_samples: Number of training examples to generate
            evolution_round: Current evolution round (affects temperature annealing)
            save_path: Optional path to save generated data

        Returns:
            List of generated training examples
        """
        if self.logger:
            with self.logger.component_context("data_generation"):
                self.logger.info(f"Starting data generation: {num_samples} samples, round {evolution_round}")

        # Update evolution round for temperature annealing
        self.debate_manager.update_evolution_round(evolution_round)

        # Load datasets
        datasets = self._load_datasets()
        if not datasets:
            raise ValueError("No datasets loaded")

        # Sample queries from datasets
        sampled_queries = self._sample_queries(datasets, num_samples)

        if self.logger:
            self.logger.info(f"Sampled {len(sampled_queries)} queries from {len(datasets)} datasets")
            self.logger.start_progress("Generating debate data", total=len(sampled_queries))

        generated_examples = []
        successful_debates = 0
        failed_debates = 0

        for i, (query, dataset_name, task_type) in enumerate(sampled_queries):
            try:
                # Conduct debate
                debate_result = self.debate_manager.conduct_debate(query, task_type)
                self.debate_results.append(debate_result)

                # Convert to training example
                training_example = self._debate_to_training_example(debate_result, dataset_name, task_type)

                # Apply quality filters
                if self._passes_quality_filters(training_example, debate_result):
                    generated_examples.append(training_example)
                    successful_debates += 1
                else:
                    failed_debates += 1
                    if self.logger:
                        self.logger.debug(f"Example filtered out: {query[:50]}...")

                if self.logger:
                    self.logger.update_progress("Generating debate data", advance=1)

                # Log progress every 50 examples
                if (i + 1) % 50 == 0 and self.logger:
                    self.logger.info(
                        f"Progress: {i + 1}/{len(sampled_queries)} "
                        f"(Success: {successful_debates}, Filtered: {failed_debates})"
                    )

            except Exception as e:
                failed_debates += 1
                if self.logger:
                    self.logger.error(f"Failed to generate example for query: {query[:50]}... Error: {e}")
                continue

        if self.logger:
            self.logger.finish_progress("Generating debate data")
            self.logger.info(
                f"Data generation completed. Generated: {len(generated_examples)}, "
                f"Success rate: {successful_debates}/{len(sampled_queries)} "
                f"({100 * successful_debates / len(sampled_queries):.1f}%)"
            )

        # Store generated examples
        self.generated_examples.extend(generated_examples)

        # Save if path provided
        if save_path:
            self._save_generated_data(generated_examples, save_path)

        return generated_examples

    def _load_datasets(self) -> List[Tuple[Dataset, str, str]]:
        """Load configured training datasets.

        Returns:
            List of (dataset, name, task_type) tuples
        """
        datasets = []

        for dataset_config in self.config_datasets.train_datasets:
            try:
                if self.logger:
                    self.logger.info(f"Loading dataset: {dataset_config.name}")

                # Load dataset
                if dataset_config.path.startswith("openai/gsm8k"):
                    dataset = load_dataset("openai/gsm8k", "main", split=dataset_config.split)
                    task_type = "math"
                elif "gsm" in dataset_config.name.lower():
                    dataset = load_dataset(dataset_config.path, split=dataset_config.split)
                    task_type = "math"
                elif "math" in dataset_config.name.lower():
                    dataset = load_dataset(dataset_config.path, split=dataset_config.split)
                    task_type = "math"
                elif "arc" in dataset_config.name.lower():
                    dataset = load_dataset(dataset_config.path, "ARC-Challenge", split="train")
                    task_type = "reasoning"
                else:
                    dataset = load_dataset(dataset_config.path, split=dataset_config.split)
                    task_type = "reasoning"

                # Limit samples if specified
                if dataset_config.max_samples > 0:
                    dataset = dataset.select(range(min(len(dataset), dataset_config.max_samples)))

                datasets.append((dataset, dataset_config.name, task_type))

                if self.logger:
                    self.logger.info(f"Loaded {len(dataset)} examples from {dataset_config.name}")

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to load dataset {dataset_config.name}: {e}")
                continue

        return datasets

    def _sample_queries(self, datasets: List[Tuple[Dataset, str, str]], num_samples: int) -> List[Tuple[str, str, str]]:
        """Sample queries from loaded datasets.

        Args:
            datasets: List of (dataset, name, task_type) tuples
            num_samples: Number of samples to generate

        Returns:
            List of (query, dataset_name, task_type) tuples
        """
        sampled_queries = []

        # Calculate samples per dataset
        total_available = sum(len(dataset) for dataset, _, _ in datasets)
        if total_available == 0:
            return []

        for dataset, dataset_name, task_type in datasets:
            if len(dataset) == 0:
                continue

            # Proportional sampling based on dataset size
            dataset_samples = int(num_samples * len(dataset) / total_available)
            dataset_samples = max(1, min(dataset_samples, len(dataset)))

            # Random sampling from dataset
            indices = random.sample(range(len(dataset)), dataset_samples)

            for idx in indices:
                example = dataset[idx]
                query = self._extract_query(example, dataset_name, task_type)
                if query:
                    sampled_queries.append((query, dataset_name, task_type))

        # Shuffle and limit to requested number
        random.shuffle(sampled_queries)
        return sampled_queries[:num_samples]

    def _extract_query(self, example: Dict[str, Any], dataset_name: str, task_type: str) -> Optional[str]:
        """Extract query from dataset example.

        Args:
            example: Dataset example
            dataset_name: Name of the dataset
            task_type: Type of task

        Returns:
            Extracted query string or None if extraction fails
        """
        try:
            if "gsm8k" in dataset_name.lower():
                return example.get("question", "")
            elif "gsm" in dataset_name.lower():
                return example.get("question", example.get("prompt", ""))
            elif "math" in dataset_name.lower():
                return example.get("problem", example.get("question", ""))
            elif "arc" in dataset_name.lower():
                question = example.get("question", "")
                choices = example.get("choices", {})
                if choices and "text" in choices:
                    choices_text = "\n".join([f"{i}: {choice}" for i, choice in enumerate(choices["text"])])
                    return f"{question}\n\nChoices:\n{choices_text}"
                return question
            else:
                # Generic extraction
                for key in ["question", "prompt", "input", "text"]:
                    if key in example and example[key]:
                        return example[key]
                return None

        except Exception as e:
            if self.logger:
                self.logger.debug(f"Failed to extract query from {dataset_name}: {e}")
            return None

    def _debate_to_training_example(
        self, debate_result: DebateResult, dataset_name: str, task_type: str
    ) -> TrainingExample:
        """Convert debate result to training example.

        Args:
            debate_result: Result from multi-agent debate
            dataset_name: Source dataset name
            task_type: Type of task

        Returns:
            Structured training example
        """
        # Calculate average confidence
        final_confidences = debate_result.confidence_progression[-1]
        avg_confidence = sum(final_confidences) / len(final_confidences)

        # Create metadata
        metadata = {
            "task_type": task_type,
            "total_time": debate_result.metrics.get("total_time", 0),
            "sycophancy_rate": debate_result.metrics.get("sycophancy_rate", 0),
            "answer_progression": debate_result.extracted_answers,
            "agent_count": len(debate_result.all_responses[0]) if debate_result.all_responses else 0,
        }

        return TrainingExample(
            query=debate_result.query,
            answer=debate_result.final_answer,
            reasoning=debate_result.consolidated_reasoning,
            confidence=avg_confidence,
            source_dataset=dataset_name,
            debate_rounds=debate_result.total_rounds,
            consensus_reached=debate_result.consensus_reached,
            metadata=metadata,
        )

    def _passes_quality_filters(self, example: TrainingExample, debate_result: DebateResult) -> bool:
        """Apply quality filters to training example.

        Args:
            example: Training example to filter
            debate_result: Original debate result

        Returns:
            True if example passes all filters
        """
        # Filter error responses
        if self.quality_filters["filter_error_responses"] and (
            "error" in example.answer.lower() or "failed" in example.answer.lower()
        ):
            return False

        # Require consensus if specified
        if self.quality_filters["require_consensus"] and not example.consensus_reached:
            return False

        # Check confidence threshold
        if example.confidence < self.quality_filters["min_consensus_confidence"]:
            return False

        # Check reasoning length
        reasoning_length = len(example.reasoning.split())
        if reasoning_length < self.quality_filters["min_reasoning_length"]:
            return False
        if reasoning_length > self.quality_filters["max_reasoning_length"]:
            return False

        # Additional quality checks
        if not example.answer.strip():
            return False

        if not example.reasoning.strip():
            return False

        return True

    def _save_generated_data(self, examples: List[TrainingExample], save_path: str) -> None:
        """Save generated training examples to file.

        Args:
            examples: List of training examples to save
            save_path: Path to save the data
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSONL
        with open(save_path, "w", encoding="utf-8") as f:
            for example in examples:
                json.dump(asdict(example), f, ensure_ascii=False)
                f.write("\n")

        if self.logger:
            self.logger.info(f"Saved {len(examples)} training examples to {save_path}")

    def load_generated_data(self, load_path: str) -> List[TrainingExample]:
        """Load previously generated training data.

        Args:
            load_path: Path to load data from

        Returns:
            List of loaded training examples
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Data file not found: {load_path}")

        examples = []
        with open(load_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    example_dict = json.loads(line)
                    example = TrainingExample(**example_dict)
                    examples.append(example)

        if self.logger:
            self.logger.info(f"Loaded {len(examples)} training examples from {load_path}")

        return examples

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated data.

        Returns:
            Dictionary with generation statistics
        """
        if not self.generated_examples:
            return {}

        total_examples = len(self.generated_examples)
        consensus_rate = sum(1 for ex in self.generated_examples if ex.consensus_reached) / total_examples
        avg_confidence = sum(ex.confidence for ex in self.generated_examples) / total_examples
        avg_reasoning_length = sum(len(ex.reasoning.split()) for ex in self.generated_examples) / total_examples

        # Dataset distribution
        dataset_counts = {}
        for example in self.generated_examples:
            dataset_counts[example.source_dataset] = dataset_counts.get(example.source_dataset, 0) + 1

        # Debate rounds distribution
        rounds_distribution = {}
        for example in self.generated_examples:
            rounds = example.debate_rounds
            rounds_distribution[rounds] = rounds_distribution.get(rounds, 0) + 1

        return {
            "total_examples": total_examples,
            "consensus_rate": consensus_rate,
            "average_confidence": avg_confidence,
            "average_reasoning_length": avg_reasoning_length,
            "dataset_distribution": dataset_counts,
            "rounds_distribution": rounds_distribution,
            "debate_statistics": self.debate_manager.get_debate_statistics(),
        }

    def update_quality_filters(self, new_filters: Dict[str, Any]) -> None:
        """Update quality filtering criteria.

        Args:
            new_filters: Dictionary with new filter values
        """
        self.quality_filters.update(new_filters)
        if self.logger:
            self.logger.info(f"Updated quality filters: {self.quality_filters}")

    def reset_generated_data(self) -> None:
        """Reset all generated data and debate history."""
        self.generated_examples.clear()
        self.debate_results.clear()
        if self.logger:
            self.logger.info("Reset all generated data")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.debate_manager.cleanup()

    def __del__(self) -> None:
        """Cleanup when generator is destroyed."""
        try:
            self.cleanup()
        except Exception:
            pass
