#!/usr/bin/env python3
"""
Reward Functions -- Demonstrates all 5 DTE reward functions and how
they combine into a training signal.

Usage:
    python examples/reward_functions.py
"""

from dte.training.reward_model import DTERewardModel


def main():
    model = DTERewardModel()

    # Example responses with varying quality
    responses = [
        # Perfect: correct answer + strict XML format
        "<reasoning>\n6 * 7 = 42\n</reasoning>\n<answer>\n42\n</answer>\n",
        # Correct but sloppy format
        "<reasoning>6 * 7 = 42</reasoning><answer>42</answer>",
        # Incorrect answer but good format
        "<reasoning>\n6 * 7 = 48\n</reasoning>\n<answer>\n48\n</answer>\n",
        # No XML format at all
        "The answer is 42.",
    ]

    ground_truth = "42"

    print("=" * 70)
    print("DTE REWARD FUNCTIONS DEMO")
    print("=" * 70)
    print(f"Ground truth: {ground_truth}")
    print()

    # Calculate all rewards
    rewards_dict = model.calculate_all_rewards(
        query="What is 6 * 7?",
        responses=responses,
        ground_truth=ground_truth,
    )

    # Display per-response breakdown
    labels = ["Perfect", "Sloppy format", "Wrong answer", "No XML"]
    for i, (label, response) in enumerate(zip(labels, responses)):
        print(f"--- Response {i + 1}: {label} ---")
        print(f"  Text: {response[:60]}...")
        for reward_name, reward_values in rewards_dict.items():
            print(f"  {reward_name:20s}: {reward_values[i]:.3f}")

        # Combined with DTE weights (weighted SUM)
        weights = {
            "correctness": 2.0,
            "int": 0.5,
            "strict_format": 0.5,
            "soft_format": 0.5,
            "xmlcount": 0.5,
        }
        combined = model.combine_rewards(
            {k: [v[i]] for k, v in rewards_dict.items()},
            weights,
        )
        print(f"  {'COMBINED (weighted sum)':20s}: {combined[0]:.3f}")
        print()

    # Show statistics
    stats = model.get_reward_statistics(rewards_dict)
    print("Reward Statistics:")
    for reward_name, stat in stats.items():
        print(f"  {reward_name}: mean={stat['mean']:.3f}, "
              f"min={stat['min']:.3f}, max={stat['max']:.3f}")


if __name__ == "__main__":
    main()
