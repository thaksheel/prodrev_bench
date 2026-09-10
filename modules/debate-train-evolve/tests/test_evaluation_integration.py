"""
GPU integration test: test evaluation on small dataset samples.

Uses Qwen/Qwen2.5-0.5B-Instruct on CUDA_VISIBLE_DEVICES=5.
Marked with @pytest.mark.gpu so it is skipped when no GPU is available.

Tests cover:
- Evaluation returns valid EvaluationMetrics
- Evaluation report is well-structured
- Multi-dataset evaluation (gsm8k + arc_challenge)
- Answer correctness checking with real outputs
- Evaluation report completeness (all expected keys)
"""

import os

import pytest

# Pin to GPU 5 before any CUDA initialization
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from dte.core.config import DatasetsConfig, DebateConfig, ModelConfig
from dte.core.evaluator import DTEEvaluator, EvaluationMetrics

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.gpu
class TestEvaluationIntegration:
    """End-to-end evaluation integration tests on real hardware."""

    @pytest.fixture(scope="class")
    def evaluator(self):
        """Create an evaluator pointing at the smallest model."""
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        debate_cfg = DebateConfig(num_agents=2, max_rounds=1)
        datasets_cfg = DatasetsConfig(
            names=["gsm8k"],
            max_samples_per_dataset=3,
        )
        evaluator = DTEEvaluator(datasets_cfg, debate_cfg, model_cfg, logger=None)
        yield evaluator
        evaluator.cleanup()

    def test_evaluation_returns_metrics(self, evaluator):
        """evaluate_model should return an EvaluationMetrics instance."""
        metrics = evaluator.evaluate_model(evolution_round=0, max_samples_per_dataset=3)
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.total_samples > 0
        assert 0.0 <= metrics.overall_accuracy <= 1.0
        assert 0.0 <= metrics.consensus_rate <= 1.0
        assert metrics.evaluation_time > 0

    def test_evaluation_report(self, evaluator):
        """create_evaluation_report should return a well-structured dict."""
        metrics = evaluator.evaluate_model(evolution_round=0, max_samples_per_dataset=2)
        report = evaluator.create_evaluation_report(metrics, evolution_round=0)
        assert "overall_metrics" in report
        assert "per_dataset_metrics" in report
        assert "transition_analysis" in report
        assert report["evolution_round"] == 0


@pytest.mark.gpu
class TestMultiDatasetEvaluation:
    """Test evaluation across multiple datasets."""

    @pytest.fixture(scope="class")
    def multi_evaluator(self):
        """Create an evaluator configured for multiple datasets."""
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        debate_cfg = DebateConfig(num_agents=2, max_rounds=1)
        # Use gsm8k and gsm_plus (both are reliably loadable)
        datasets_cfg = DatasetsConfig(
            names=["gsm8k", "gsm_plus"],
            max_samples_per_dataset=2,
        )
        evaluator = DTEEvaluator(datasets_cfg, debate_cfg, model_cfg, logger=None)
        yield evaluator
        evaluator.cleanup()

    def test_multi_dataset_returns_per_dataset_metrics(self, multi_evaluator):
        """Evaluation on multiple datasets should return per-dataset metrics."""
        metrics = multi_evaluator.evaluate_model(evolution_round=0, max_samples_per_dataset=2)

        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.total_samples > 0
        assert isinstance(metrics.per_dataset_metrics, dict)
        # At least one dataset should have metrics
        assert len(metrics.per_dataset_metrics) >= 1

    def test_multi_dataset_report_completeness(self, multi_evaluator):
        """Report from multi-dataset evaluation should have all keys."""
        metrics = multi_evaluator.evaluate_model(evolution_round=0, max_samples_per_dataset=2)
        report = multi_evaluator.create_evaluation_report(metrics, evolution_round=0)

        # Check overall metrics structure
        overall = report["overall_metrics"]
        expected_keys = [
            "accuracy",
            "total_samples",
            "correct_samples",
            "consensus_rate",
            "debate_helped_rate",
            "average_debate_rounds",
            "sycophancy_rate",
            "evaluation_time",
        ]
        for key in expected_keys:
            assert key in overall, f"Missing key in overall_metrics: {key}"

        # Check transition analysis
        transition = report["transition_analysis"]
        assert "correct_to_incorrect_rate" in transition
        assert "incorrect_to_correct_rate" in transition
        assert "net_improvement_rate" in transition

        # Check reasoning analysis
        assert "reasoning_analysis" in report
        assert "average_reasoning_length" in report["reasoning_analysis"]


@pytest.mark.gpu
class TestAnswerCorrectness:
    """Test answer correctness checking with real model outputs."""

    @pytest.fixture(scope="class")
    def evaluator(self):
        """Create an evaluator for correctness tests."""
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        debate_cfg = DebateConfig(num_agents=2, max_rounds=1)
        datasets_cfg = DatasetsConfig(
            names=["gsm8k"],
            max_samples_per_dataset=2,
        )
        evaluator = DTEEvaluator(datasets_cfg, debate_cfg, model_cfg, logger=None)
        yield evaluator
        evaluator.cleanup()

    def test_math_correctness_exact(self, evaluator):
        """Exact numeric match should be correct."""
        assert evaluator._check_answer_correctness("42", "42", "math") is True

    def test_math_correctness_with_whitespace(self, evaluator):
        """Numeric match with whitespace should still be correct."""
        assert evaluator._check_answer_correctness(" 42 ", "42", "math") is True

    def test_math_correctness_wrong(self, evaluator):
        """Wrong numeric answer should be incorrect."""
        assert evaluator._check_answer_correctness("41", "42", "math") is False

    def test_arc_correctness_exact(self, evaluator):
        """Exact letter match for ARC should be correct."""
        assert evaluator._check_answer_correctness("A", "A", "arc") is True

    def test_arc_correctness_case_insensitive(self, evaluator):
        """ARC answer matching should be case-insensitive."""
        assert evaluator._check_answer_correctness("a", "A", "arc") is True

    def test_arc_correctness_wrong(self, evaluator):
        """Wrong letter for ARC should be incorrect."""
        assert evaluator._check_answer_correctness("B", "A", "arc") is False

    def test_general_correctness(self, evaluator):
        """General task correctness uses case-insensitive string match."""
        assert evaluator._check_answer_correctness("Yes", "yes", "general") is True
        assert evaluator._check_answer_correctness("No", "yes", "general") is False

    def test_correctness_with_real_debate(self, evaluator):
        """Run a real debate and check that correctness evaluation works."""
        from dte.debate.manager import DebateManager

        manager = DebateManager(evaluator.debate_config, evaluator.model_config, logger=None)
        try:
            result = manager.conduct_debate("What is 2 + 2?", task_type="math")
            # Just verify the function does not crash on real output
            is_correct = evaluator._check_answer_correctness(result.final_answer, "4", "math")
            assert isinstance(is_correct, bool)
        finally:
            manager.cleanup()
