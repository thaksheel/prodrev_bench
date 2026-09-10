#!/usr/bin/env python3
"""
Custom Debate -- Demonstrates how to configure and run debates
with non-default settings.

Usage:
    python examples/custom_debate.py
"""

from dte.core.config import ModelConfig, DebateConfig
from dte.debate.manager import DebateManager


def run_custom_debate():
    """Run a debate with a custom configuration."""

    # Configure the model
    model_config = ModelConfig(
        base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="auto",
        max_length=1024,
        temperature=0.5,       # Lower temperature for more focused answers
        top_p=0.95,
        top_k=40,
    )

    # Configure the debate
    debate_config = DebateConfig(
        num_agents=3,
        max_rounds=4,           # Allow more rounds
    )

    # Create the manager and run
    print("Loading model and initializing agents ...")
    manager = DebateManager(debate_config, model_config)

    queries = [
        ("What is 123 * 456?", "math"),
        ("What is the square root of 144?", "math"),
    ]

    for query, task_type in queries:
        print(f"\nQuery: {query}")
        result = manager.conduct_debate(query, task_type)

        print(f"  Final answer      : {result.final_answer}")
        print(f"  Consensus reached : {result.consensus_reached}")
        print(f"  Rounds used       : {result.total_rounds}")
        print(f"  Sycophancy rate   : {result.metrics.get('sycophancy_rate', 0):.2%}")

    # Show aggregate statistics
    stats = manager.get_debate_statistics()
    print("\nAggregate debate statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    manager.cleanup()


if __name__ == "__main__":
    run_custom_debate()
