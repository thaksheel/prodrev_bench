#!/usr/bin/env python3
"""
Quick Start -- Run a multi-agent debate with minimal setup.

Usage:
    python examples/quick_start.py
"""

import dte


def main():
    # Run a 3-agent debate on a simple math problem
    print("Running a quick 3-agent debate ...")
    result = dte.debate(
        query="What is 15 * 24?",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        num_agents=3,
        max_rounds=3,
        task_type="math",
    )

    # Display results
    print(f"\nFinal answer : {result.final_answer}")
    print(f"Consensus    : {result.consensus_reached}")
    print(f"Total rounds : {result.total_rounds}")
    print(f"Time         : {result.metrics.get('total_time', 0):.2f}s")

    # Show each agent's answer progression
    print("\nAnswer progression per round:")
    for round_idx, answers in enumerate(result.extracted_answers):
        print(f"  Round {round_idx}: {answers}")


if __name__ == "__main__":
    main()
