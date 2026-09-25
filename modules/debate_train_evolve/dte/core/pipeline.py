"""
Main DTE Pipeline Orchestrator.

This module implements the complete end-to-end DTE (Debate, Train, Evolve)
pipeline, coordinating all components and managing the iterative evolution process.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..data.generator import DebateDataGenerator
from ..debate.manager import DebateManager
from ..training.grpo_trainer import GRPOTrainer
from .config import DTEConfig
from .evaluator import DTEEvaluator
from .logger import DTELogger


@dataclass
class EvolutionRoundResult:
    """Results from a single evolution round."""

    round_number: int
    data_generation_stats: Dict[str, Any]
    training_metrics: Dict[str, Any]
    evaluation_results: Dict[str, Any]
    performance_improvement: float
    total_time: float


class DTEPipeline:
    """Complete DTE (Debate, Train, Evolve) Pipeline.

    This class orchestrates the entire DTE process:
    1. Generate debate data using multi-agent RCR debates
    2. Train models using GRPO on the debate data
    3. Evaluate performance and iterate (evolve)
    """

    def __init__(self, config: DTEConfig):
        """Initialize DTE pipeline.

        Args:
            config: Complete DTE configuration
        """
        self.config = config

        # Setup environment
        config.setup_environment()

        # Initialize logger
        self.logger = DTELogger(config.logging, config.experiment.name)

        # Initialize components
        self.data_generator = DebateDataGenerator(config.datasets, config.debate, config.model, self.logger)
        self.grpo_trainer = GRPOTrainer(config.training, config.model, config.paths, self.logger)
        self.evaluator = DTEEvaluator(config.datasets, config.debate, config.model, self.logger)

        # Pipeline state
        self.current_round = 0
        self.evolution_results: List[EvolutionRoundResult] = []
        self.best_performance = 0.0
        self.patience_counter = 0

        # Base model for comparison
        self.base_model_performance = None

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete DTE pipeline from start to finish.

        Returns:
            Complete pipeline results including all evolution rounds
        """
        with self.logger.component_context("pipeline"):
            self.logger.info("=" * 60)
            self.logger.info("STARTING DTE PIPELINE")
            self.logger.info(f"Experiment: {self.config.experiment.name}")
            self.logger.info(f"Max Evolution Rounds: {self.config.evolution.max_rounds}")
            self.logger.info("=" * 60)

        pipeline_start_time = time.time()

        try:
            # Initialize W&B if enabled
            self._initialize_experiment_tracking()

            # Run evolution rounds
            for round_num in range(1, self.config.evolution.max_rounds + 1):
                with self.logger.round_context(round_num):
                    round_result = self._run_evolution_round(round_num)
                    self.evolution_results.append(round_result)

                    # Check convergence
                    if self._check_convergence(round_result):
                        self.logger.info(f"Convergence reached after {round_num} rounds")
                        break

            # Final evaluation and summary
            total_time = time.time() - pipeline_start_time
            final_results = self._generate_final_results(total_time)

            # Log experiment summary
            self.logger.log_experiment_summary(final_results)

            return final_results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self._cleanup()

    def _run_evolution_round(self, round_num: int) -> EvolutionRoundResult:
        """Run a single evolution round.

        Args:
            round_num: Current evolution round number

        Returns:
            Results from this evolution round
        """
        self.logger.info(f"Starting Evolution Round {round_num}")
        round_start_time = time.time()

        # Phase 1: Generate debate data
        with self.logger.component_context("data_generation"):
            self.logger.info("Phase 1: Generating debate data")
            training_examples = self.data_generator.generate_training_data(
                num_samples=self.config.evolution.samples_per_round,
                evolution_round=round_num - 1,  # 0-indexed for temperature annealing
                save_path=f"{self.config.paths.data_dir}/round_{round_num}_training_data.jsonl",
            )
            data_stats = self.data_generator.get_generation_statistics()

        # Phase 2: Train model using GRPO
        with self.logger.component_context("grpo_training"):
            self.logger.info("Phase 2: Training model with GRPO")
            training_metrics = self.grpo_trainer.train(training_examples)

        # Phase 3: Evaluate performance
        with self.logger.component_context("evaluation"):
            self.logger.info("Phase 3: Evaluating model performance")
            evaluation_results = self._evaluate_model(round_num)

        # Calculate improvement
        current_performance = evaluation_results.get("overall_accuracy", 0.0)
        if self.best_performance == 0:
            self.best_performance = current_performance
            improvement = 0.0
        else:
            improvement = current_performance - self.best_performance
            if improvement > 0:
                self.best_performance = current_performance
                self.patience_counter = 0
            else:
                self.patience_counter += 1

        round_time = time.time() - round_start_time

        # Create round result
        round_result = EvolutionRoundResult(
            round_number=round_num,
            data_generation_stats=data_stats,
            training_metrics=training_metrics,
            evaluation_results=evaluation_results,
            performance_improvement=improvement,
            total_time=round_time,
        )

        # Log round completion
        self.logger.log_evolution_round(round_num, evaluation_results, improvement)

        return round_result

    def _evaluate_model(self, round_num: int) -> Dict[str, Any]:
        """Evaluate current model performance using real metrics.

        Args:
            round_num: Current round number

        Returns:
            Comprehensive evaluation results
        """
        self.logger.info(f"Evaluating model performance for round {round_num}")

        # Run comprehensive evaluation
        max_samples = getattr(self.config.datasets, "max_samples_per_dataset", 100)
        evaluation_metrics = self.evaluator.evaluate_model(round_num, max_samples)

        # Create evaluation report
        evaluation_report = self.evaluator.create_evaluation_report(evaluation_metrics, round_num)

        # Convert to format compatible with existing pipeline
        evaluation_results = {
            "round": round_num,
            "overall_accuracy": evaluation_metrics.overall_accuracy,
            "total_samples": evaluation_metrics.total_samples,
            "correct_samples": evaluation_metrics.correct_samples,
            "consensus_rate": evaluation_metrics.consensus_rate,
            "debate_helped_rate": evaluation_metrics.debate_helped_rate,
            "average_debate_rounds": evaluation_metrics.average_debate_rounds,
            "sycophancy_rate": evaluation_metrics.sycophancy_rate,
            "correct_to_incorrect_rate": evaluation_metrics.correct_to_incorrect_rate,
            "incorrect_to_correct_rate": evaluation_metrics.incorrect_to_correct_rate,
            "average_reasoning_length": evaluation_metrics.average_reasoning_length,
            "evaluation_time": evaluation_metrics.evaluation_time,
            "per_dataset_metrics": evaluation_metrics.per_dataset_metrics,
            "full_report": evaluation_report,
        }

        # Log key metrics
        self.logger.info(f"Overall accuracy: {evaluation_metrics.overall_accuracy:.4f}")
        self.logger.info(f"Consensus rate: {evaluation_metrics.consensus_rate:.4f}")
        self.logger.info(f"Debate helped rate: {evaluation_metrics.debate_helped_rate:.4f}")
        self.logger.info(f"Sycophancy rate: {evaluation_metrics.sycophancy_rate:.4f}")

        return evaluation_results

    def _check_convergence(self, round_result: EvolutionRoundResult) -> bool:
        """Check if the pipeline should stop based on convergence criteria.

        Args:
            round_result: Latest round result

        Returns:
            True if convergence criteria are met
        """
        # Check improvement threshold
        if round_result.performance_improvement < self.config.evolution.convergence_threshold:
            if self.patience_counter >= self.config.evolution.patience:
                self.logger.info("Stopping: No improvement for multiple rounds")
                return True

        return False

    def _initialize_experiment_tracking(self) -> None:
        """Initialize experiment tracking (W&B, etc.)."""
        if self.config.experiment.wandb.enabled:
            try:
                import wandb

                wandb.init(
                    project=self.config.experiment.wandb.project,
                    entity=self.config.experiment.wandb.entity,
                    name=self.config.experiment.name,
                    config=self.config.to_dict(),
                    tags=self.config.experiment.tags,
                )
                self.logger.info("Initialized Weights & Biases tracking")
            except ImportError:
                self.logger.warning("wandb not installed, skipping experiment tracking")

    def _generate_final_results(self, total_time: float) -> Dict[str, Any]:
        """Generate final pipeline results.

        Args:
            total_time: Total pipeline execution time

        Returns:
            Complete results dictionary
        """
        final_results = {
            "experiment_name": self.config.experiment.name,
            "total_time_hours": total_time / 3600,
            "total_evolution_rounds": len(self.evolution_results),
            "best_performance": self.best_performance,
            "final_performance": self.evolution_results[-1].evaluation_results["overall_accuracy"]
            if self.evolution_results
            else 0.0,
            "total_improvement": self.best_performance - (self.base_model_performance or 0.0),
            "convergence_achieved": self.patience_counter >= self.config.evolution.patience,
            "evolution_rounds": [asdict(result) for result in self.evolution_results],
            "config": self.config.to_dict(),
        }

        # Save results
        results_path = Path(self.config.paths.output_dir) / f"{self.config.experiment.name}_results.json"
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        self.logger.info(f"Results saved to {results_path}")

        return final_results

    def _cleanup(self) -> None:
        """Clean up pipeline resources."""
        try:
            self.data_generator.cleanup()
            self.grpo_trainer.cleanup()

            if self.config.experiment.wandb.enabled:
                try:
                    import wandb

                    wandb.finish()
                except Exception:
                    pass

        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    def run_single_round(self, round_num: int) -> EvolutionRoundResult:
        """Run a single evolution round (for testing/debugging).

        Args:
            round_num: Round number to run

        Returns:
            Results from the evolution round
        """
        return self._run_evolution_round(round_num)

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status.

        Returns:
            Status information
        """
        return {
            "current_round": self.current_round,
            "best_performance": self.best_performance,
            "patience_counter": self.patience_counter,
            "total_rounds_completed": len(self.evolution_results),
            "last_round_improvement": self.evolution_results[-1].performance_improvement
            if self.evolution_results
            else 0.0,
        }

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save pipeline checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint_data = {
            "current_round": self.current_round,
            "evolution_results": [asdict(result) for result in self.evolution_results],
            "best_performance": self.best_performance,
            "patience_counter": self.patience_counter,
            "config": self.config.to_dict(),
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        self.logger.info(f"Pipeline checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load pipeline checkpoint.

        Args:
            checkpoint_path: Path to load checkpoint from
        """
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        self.current_round = checkpoint_data["current_round"]
        self.best_performance = checkpoint_data["best_performance"]
        self.patience_counter = checkpoint_data["patience_counter"]

        # Restore evolution results
        self.evolution_results = [EvolutionRoundResult(**result) for result in checkpoint_data["evolution_results"]]

        self.logger.info(f"Pipeline checkpoint loaded from {checkpoint_path}")

    def __del__(self) -> None:
        """Cleanup when pipeline is destroyed."""
        try:
            self._cleanup()
        except Exception:
            pass
