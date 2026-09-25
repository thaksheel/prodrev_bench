#!/usr/bin/env python3
"""
Evaluation Example -- Evaluate a model on standard benchmarks using
multi-agent debate.

Usage:
    python examples/evaluation_example.py
"""

from dte.core.config import ModelConfig, DebateConfig, DatasetsConfig, LoggingConfig
from dte.core.logger import DTELogger
from dte.core.evaluator import DTEEvaluator


def main():
    # Configure the model (use smallest model for demo)
    model_config = ModelConfig(
        base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="auto",
        max_length=512,
        temperature=0.7,
    )

    # Configure the debate
    debate_config = DebateConfig(
        num_agents=3,
        max_rounds=2,
    )

    # Configure datasets to evaluate on
    datasets_config = DatasetsConfig(
        names=["gsm8k"],
        max_samples_per_dataset=10,  # Small sample for demo
    )

    # Set up logging
    log_config = LoggingConfig(level="INFO", log_dir="./logs")
    logger = DTELogger(log_config, "evaluation_demo")

    # Create evaluator and run
    print("Initializing evaluator ...")
    evaluator = DTEEvaluator(datasets_config, debate_config, model_config, logger)

    try:
        print("Running evaluation ...")
        metrics = evaluator.evaluate_model(
            evolution_round=0,
            max_samples_per_dataset=10,
        )

        # Create and display report
        report = evaluator.create_evaluation_report(metrics, evolution_round=0)

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Overall accuracy      : {metrics.overall_accuracy:.2%}")
        print(f"Total samples         : {metrics.total_samples}")
        print(f"Correct samples       : {metrics.correct_samples}")
        print(f"Consensus rate        : {metrics.consensus_rate:.2%}")
        print(f"Debate helped rate    : {metrics.debate_helped_rate:.2%}")
        print(f"Sycophancy rate       : {metrics.sycophancy_rate:.2%}")
        print(f"Avg debate rounds     : {metrics.average_debate_rounds:.1f}")
        print(f"Avg reasoning length  : {metrics.average_reasoning_length:.0f}")
        print(f"Evaluation time       : {metrics.evaluation_time:.1f}s")

        # Per-dataset breakdown
        print("\nPer-dataset metrics:")
        for ds_name, ds_metrics in metrics.per_dataset_metrics.items():
            print(f"  {ds_name}:")
            for key, value in ds_metrics.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")

    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main()
