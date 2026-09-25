"""
Model evaluation system for DTE pipeline.

This module implements comprehensive evaluation of trained models using the exact
methodology from the original DTE implementation, including accuracy calculation,
debate metrics, and performance analysis.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..data.dataset_manager import DatasetManager
from ..debate.manager import DebateManager
from ..utils.answer_extraction import clean_numeric_string
from .logger import DTELogger


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""

    overall_accuracy: float
    total_samples: int
    correct_samples: int
    average_debate_rounds: float
    consensus_rate: float
    sycophancy_rate: float
    correct_to_incorrect_rate: float
    incorrect_to_correct_rate: float
    debate_helped_rate: float
    average_reasoning_length: float
    evaluation_time: float
    per_dataset_metrics: Dict[str, Dict[str, float]]


class DTEEvaluator:
    """
    Comprehensive evaluator for DTE models.

    Implements the exact evaluation methodology from the original DTE codebase,
    including multi-agent debate evaluation and detailed metrics calculation.
    """

    def __init__(self, config_datasets, config_debate, config_model, logger: Optional[DTELogger] = None):
        """
        Initialize DTE evaluator.

        Args:
            config_datasets: Dataset configuration
            config_debate: Debate configuration
            config_model: Model configuration
            logger: Optional logger for evaluation progress
        """
        self.datasets_config = config_datasets
        self.debate_config = config_debate
        self.model_config = config_model
        self.logger = logger

        # Initialize components
        self.dataset_manager = DatasetManager()
        self.debate_manager = DebateManager(config_debate, config_model, logger)

        # Evaluation state
        self.current_round = 0
        self.evaluation_results = []

    def evaluate_model(self, evolution_round: int, max_samples_per_dataset: Optional[int] = None) -> EvaluationMetrics:
        """
        Perform comprehensive model evaluation.

        Args:
            evolution_round: Current evolution round for tracking
            max_samples_per_dataset: Maximum samples to evaluate per dataset

        Returns:
            Comprehensive evaluation metrics
        """
        if self.logger:
            self.logger.info(f"Starting evaluation for evolution round {evolution_round}")

        start_time = time.time()

        # Update debate manager with current evolution round
        self.debate_manager.update_evolution_round(evolution_round)

        # Initialize aggregate metrics
        total_samples = 0
        total_correct = 0
        total_debate_rounds = 0
        total_sycophancy_instances = 0
        total_correct_to_incorrect = 0
        total_incorrect_to_correct = 0
        total_debate_helped = 0
        total_consensus_reached = 0
        total_reasoning_length = 0
        per_dataset_metrics = {}

        # Evaluate on each configured dataset
        for dataset_name in self.datasets_config.names:
            if self.logger:
                self.logger.info(f"Evaluating on {dataset_name}")

            dataset_metrics = self._evaluate_on_dataset(dataset_name, evolution_round, max_samples_per_dataset)

            # Aggregate metrics
            total_samples += dataset_metrics["total_samples"]
            total_correct += dataset_metrics["correct_samples"]
            total_debate_rounds += dataset_metrics["total_debate_rounds"]
            total_sycophancy_instances += dataset_metrics["sycophancy_instances"]
            total_correct_to_incorrect += dataset_metrics["correct_to_incorrect"]
            total_incorrect_to_correct += dataset_metrics["incorrect_to_correct"]
            total_debate_helped += dataset_metrics["debate_helped"]
            total_consensus_reached += dataset_metrics["consensus_reached"]
            total_reasoning_length += dataset_metrics["total_reasoning_length"]

            # Store per-dataset metrics
            per_dataset_metrics[dataset_name] = {
                "accuracy": dataset_metrics["accuracy"],
                "samples": dataset_metrics["total_samples"],
                "consensus_rate": dataset_metrics["consensus_rate"],
                "debate_helped_rate": dataset_metrics["debate_helped_rate"],
            }

        evaluation_time = time.time() - start_time

        # Calculate aggregate metrics
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        average_debate_rounds = total_debate_rounds / total_samples if total_samples > 0 else 0.0
        consensus_rate = total_consensus_reached / total_samples if total_samples > 0 else 0.0
        sycophancy_rate = total_sycophancy_instances / total_samples if total_samples > 0 else 0.0
        correct_to_incorrect_rate = total_correct_to_incorrect / total_samples if total_samples > 0 else 0.0
        incorrect_to_correct_rate = total_incorrect_to_correct / total_samples if total_samples > 0 else 0.0
        debate_helped_rate = total_debate_helped / total_samples if total_samples > 0 else 0.0
        average_reasoning_length = total_reasoning_length / total_samples if total_samples > 0 else 0.0

        metrics = EvaluationMetrics(
            overall_accuracy=overall_accuracy,
            total_samples=total_samples,
            correct_samples=total_correct,
            average_debate_rounds=average_debate_rounds,
            consensus_rate=consensus_rate,
            sycophancy_rate=sycophancy_rate,
            correct_to_incorrect_rate=correct_to_incorrect_rate,
            incorrect_to_correct_rate=incorrect_to_correct_rate,
            debate_helped_rate=debate_helped_rate,
            average_reasoning_length=average_reasoning_length,
            evaluation_time=evaluation_time,
            per_dataset_metrics=per_dataset_metrics,
        )

        if self.logger:
            self.logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
            self.logger.info(f"Overall accuracy: {overall_accuracy:.4f} ({total_correct}/{total_samples})")
            self.logger.info(f"Consensus rate: {consensus_rate:.4f}")
            self.logger.info(f"Debate helped rate: {debate_helped_rate:.4f}")

        return metrics

    def _evaluate_on_dataset(
        self, dataset_name: str, evolution_round: int, max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on a specific dataset.

        Args:
            dataset_name: Name of dataset to evaluate on
            evolution_round: Current evolution round
            max_samples: Maximum samples to evaluate

        Returns:
            Dataset-specific evaluation metrics
        """
        # Load test dataset
        try:
            dataset = self.dataset_manager.load_dataset_by_name(dataset_name, split="test", max_samples=max_samples)
            processed_dataset = self.dataset_manager.preprocess_dataset(dataset, dataset_name)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not load {dataset_name} test set, using train: {e}")
            # Fallback to train set with limited samples
            dataset = self.dataset_manager.load_dataset_by_name(
                dataset_name, split="train", max_samples=min(max_samples or 100, 100)
            )
            processed_dataset = self.dataset_manager.preprocess_dataset(dataset, dataset_name)

        # Initialize metrics
        total_samples = 0
        correct_samples = 0
        total_debate_rounds = 0
        sycophancy_instances = 0
        correct_to_incorrect = 0
        incorrect_to_correct = 0
        debate_helped = 0
        consensus_reached = 0
        total_reasoning_length = 0

        # Get dataset configuration for task type
        task_type = self.dataset_manager.DATASET_CONFIGS[dataset_name]["task_type"]

        # Evaluate each sample
        for sample in processed_dataset:
            if self.logger and total_samples % 10 == 0:
                self.logger.info(f"Evaluating {dataset_name} sample {total_samples + 1}")

            # Run debate on this sample
            query = sample["formatted_query"]
            ground_truth = sample["ground_truth"]

            try:
                debate_result = self.debate_manager.conduct_debate(query, task_type)

                # Check correctness
                is_correct = self._check_answer_correctness(debate_result.final_answer, ground_truth, task_type)

                # Update metrics
                total_samples += 1
                if is_correct:
                    correct_samples += 1

                total_debate_rounds += debate_result.total_rounds

                if debate_result.consensus_reached:
                    consensus_reached += 1

                # Analyze sycophancy and transitions
                syc_count, c2i, i2c, helped = self._analyze_debate_dynamics(debate_result, ground_truth, task_type)

                sycophancy_instances += syc_count
                correct_to_incorrect += c2i
                incorrect_to_correct += i2c
                if helped:
                    debate_helped += 1

                # Calculate reasoning length
                if debate_result.all_responses:
                    reasoning_lengths = [
                        len(response.reasoning)
                        for round_responses in debate_result.all_responses
                        for response in round_responses
                    ]
                    total_reasoning_length += sum(reasoning_lengths) / len(reasoning_lengths)

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error evaluating sample: {e}")
                total_samples += 1  # Count failed samples

        # Calculate dataset metrics
        accuracy = correct_samples / total_samples if total_samples > 0 else 0.0
        consensus_rate = consensus_reached / total_samples if total_samples > 0 else 0.0
        debate_helped_rate = debate_helped / total_samples if total_samples > 0 else 0.0

        return {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "accuracy": accuracy,
            "total_debate_rounds": total_debate_rounds,
            "sycophancy_instances": sycophancy_instances,
            "correct_to_incorrect": correct_to_incorrect,
            "incorrect_to_correct": incorrect_to_correct,
            "debate_helped": debate_helped,
            "consensus_reached": consensus_reached,
            "consensus_rate": consensus_rate,
            "debate_helped_rate": debate_helped_rate,
            "total_reasoning_length": total_reasoning_length,
        }

    def _check_answer_correctness(self, predicted_answer: str, ground_truth: str, task_type: str) -> bool:
        """
        Check if predicted answer matches ground truth.

        Args:
            predicted_answer: Model's predicted answer
            ground_truth: Correct answer
            task_type: Type of task (math, arc)

        Returns:
            True if answer is correct
        """
        if task_type == "math":
            # For math problems, use numerical comparison with tolerance
            clean_pred = clean_numeric_string(predicted_answer)
            clean_gt = clean_numeric_string(ground_truth)

            if clean_pred is not None and clean_gt is not None:
                return abs(clean_pred - clean_gt) < 1e-9
            else:
                # Fall back to string comparison
                return predicted_answer.strip().lower() == ground_truth.strip().lower()

        elif task_type == "arc":
            # For ARC problems, exact string match on answer choice
            pred_clean = predicted_answer.strip().upper()
            gt_clean = ground_truth.strip().upper()
            return pred_clean == gt_clean

        else:
            # For other tasks, string comparison
            return predicted_answer.strip().lower() == ground_truth.strip().lower()

    def _analyze_debate_dynamics(self, debate_result, ground_truth: str, task_type: str) -> Tuple[int, int, int, bool]:
        """
        Analyze debate dynamics for sycophancy and improvement patterns.

        Args:
            debate_result: Complete debate result
            ground_truth: Correct answer
            task_type: Task type

        Returns:
            Tuple of (sycophancy_count, correct_to_incorrect, incorrect_to_correct, debate_helped)
        """
        if not debate_result.all_responses or len(debate_result.all_responses) < 2:
            return 0, 0, 0, False

        # Track agent correctness over rounds
        agent_correctness = defaultdict(list)

        for round_idx, round_responses in enumerate(debate_result.all_responses):
            for agent_idx, response in enumerate(round_responses):
                agent_id = f"agent_{agent_idx}"
                is_correct = self._check_answer_correctness(response.extracted_answer, ground_truth, task_type)
                agent_correctness[agent_id].append(is_correct)

        # Analyze transitions
        sycophancy_count = 0
        correct_to_incorrect = 0
        incorrect_to_correct = 0

        for agent_id, correctness_history in agent_correctness.items():
            for i in range(1, len(correctness_history)):
                prev_correct = correctness_history[i - 1]
                curr_correct = correctness_history[i]

                if prev_correct and not curr_correct:
                    correct_to_incorrect += 1
                    sycophancy_count += 1  # Changing from correct to incorrect is sycophancy
                elif not prev_correct and curr_correct:
                    incorrect_to_correct += 1

        # Determine if debate helped overall
        if len(debate_result.all_responses) > 1:
            # Check if final answer is better than initial consensus
            initial_answers = [r.extracted_answer for r in debate_result.all_responses[0]]
            initial_correct = [self._check_answer_correctness(ans, ground_truth, task_type) for ans in initial_answers]
            initial_consensus_correct = sum(initial_correct) > len(initial_correct) / 2

            final_correct = self._check_answer_correctness(debate_result.final_answer, ground_truth, task_type)

            debate_helped = final_correct and not initial_consensus_correct
        else:
            debate_helped = False

        return sycophancy_count, correct_to_incorrect, incorrect_to_correct, debate_helped

    def create_evaluation_report(self, metrics: EvaluationMetrics, evolution_round: int) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report.

        Args:
            metrics: Evaluation metrics
            evolution_round: Current evolution round

        Returns:
            Formatted evaluation report
        """
        report = {
            "evolution_round": evolution_round,
            "overall_metrics": {
                "accuracy": metrics.overall_accuracy,
                "total_samples": metrics.total_samples,
                "correct_samples": metrics.correct_samples,
                "consensus_rate": metrics.consensus_rate,
                "debate_helped_rate": metrics.debate_helped_rate,
                "average_debate_rounds": metrics.average_debate_rounds,
                "sycophancy_rate": metrics.sycophancy_rate,
                "evaluation_time": metrics.evaluation_time,
            },
            "per_dataset_metrics": metrics.per_dataset_metrics,
            "transition_analysis": {
                "correct_to_incorrect_rate": metrics.correct_to_incorrect_rate,
                "incorrect_to_correct_rate": metrics.incorrect_to_correct_rate,
                "net_improvement_rate": metrics.incorrect_to_correct_rate - metrics.correct_to_incorrect_rate,
            },
            "reasoning_analysis": {"average_reasoning_length": metrics.average_reasoning_length},
        }

        return report

    def cleanup(self) -> None:
        """Clean up evaluation resources."""
        if hasattr(self, "debate_manager"):
            self.debate_manager.cleanup()
