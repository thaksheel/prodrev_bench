"""
Complete reward model implementation based on original DTE GRPO training.

This module implements all 5 reward functions used in the original DTE GRPO training:
1. correctness_reward_func: +2.0 for correct answers
2. int_reward_func: +0.5 for numeric answers
3. strict_format_reward_func: +0.5 for exact XML format
4. soft_format_reward_func: +0.5 for flexible XML format
5. xmlcount_reward_func: Granular XML format scoring
"""

import re
from typing import Any, Dict, List, Optional

from ..utils.answer_extraction import answers_match, extract_final_answer


class DTERewardModel:
    """
    Complete reward model implementing all 5 reward functions from original DTE.

    This exactly replicates the reward structure used in the original GRPO training,
    providing multiple complementary signals for training effectiveness.
    """

    def __init__(self):
        """Initialize the DTE reward model."""
        pass

    def calculate_all_rewards(
        self, query: str, responses: List[str], ground_truth: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Calculate all 5 reward functions for a batch of responses.

        Args:
            query: The input query/question
            responses: List of model responses
            ground_truth: Ground truth answer for correctness evaluation

        Returns:
            Dictionary mapping reward function names to lists of rewards
        """
        rewards = {}

        # 1. Correctness reward (requires ground truth)
        if ground_truth:
            rewards["correctness"] = self.correctness_reward_func(responses, ground_truth)
        else:
            rewards["correctness"] = [0.0] * len(responses)

        # 2. Integer/numeric reward
        rewards["int"] = self.int_reward_func(responses)

        # 3. Strict format reward
        rewards["strict_format"] = self.strict_format_reward_func(responses)

        # 4. Soft format reward
        rewards["soft_format"] = self.soft_format_reward_func(responses)

        # 5. XML count reward
        rewards["xmlcount"] = self.xmlcount_reward_func(responses)

        return rewards

    def correctness_reward_func(self, responses: List[str], ground_truth: str) -> List[float]:
        """
        Reward function for answer correctness.

        Gives +2.0 for correct answers, 0.0 for incorrect answers.
        This is the most important reward signal.

        Args:
            responses: List of model responses
            ground_truth: Correct answer

        Returns:
            List of correctness rewards
        """
        extracted_responses = [self._extract_xml_answer(r) for r in responses]
        return [2.0 if answers_match(r, ground_truth) else 0.0 for r in extracted_responses]

    def int_reward_func(self, responses: List[str]) -> List[float]:
        """
        Reward function for numeric answers.

        Gives +0.5 if extracted answer is a valid integer/number.
        Encourages models to provide numeric responses for math problems.

        Args:
            responses: List of model responses

        Returns:
            List of integer rewards
        """
        extracted_responses = [self._extract_xml_answer(r) for r in responses]
        rewards = []

        for response in extracted_responses:
            # Check if response is a valid number
            try:
                # Try to convert to float first, then check if it's effectively an integer
                num_val = float(response.replace(",", ""))
                if num_val.is_integer() or "." not in response:
                    rewards.append(0.5)
                else:
                    rewards.append(0.5)  # Still reward non-integer numbers
            except (ValueError, AttributeError):
                # Check if it's a digit string
                clean_response = response.replace(",", "").replace(".", "")
                if clean_response.isdigit():
                    rewards.append(0.5)
                else:
                    rewards.append(0.0)

        return rewards

    def strict_format_reward_func(self, responses: List[str]) -> List[float]:
        """
        Reward function for strict XML format compliance.

        Checks for exact format: ^<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n$
        Gives +0.5 for perfect format compliance.

        Args:
            responses: List of model responses

        Returns:
            List of strict format rewards
        """
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward_func(self, responses: List[str]) -> List[float]:
        """
        Reward function for flexible XML format compliance.

        Checks for flexible format: <reasoning>...</reasoning><answer>...</answer>
        More lenient than strict format, gives +0.5 for basic XML structure.

        Args:
            responses: List of model responses

        Returns:
            List of soft format rewards
        """
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def xmlcount_reward_func(self, responses: List[str]) -> List[float]:
        """
        Reward function for granular XML format scoring.

        Provides detailed scoring based on XML tag counts and structure:
        - +0.125 for exactly one <reasoning> tag
        - +0.125 for exactly one </reasoning> tag
        - +0.125 for exactly one <answer> tag
        - +0.125 for exactly one </answer> tag
        - -0.001 per character after </answer> (penalizes extra content)

        Args:
            responses: List of model responses

        Returns:
            List of granular XML rewards
        """
        return [self._count_xml(response) for response in responses]

    def _count_xml(self, text: str) -> float:
        """
        Count XML elements and calculate granular reward.

        This is the exact implementation from the original DTE codebase.

        Args:
            text: Response text to analyze

        Returns:
            Granular XML reward score
        """
        count = 0.0

        # Reward correct number of opening/closing tags
        if text.count("<reasoning>") == 1:
            count += 0.125
        if text.count("</reasoning>") == 1:
            count += 0.125
        if text.count("<answer>") == 1:
            count += 0.125
        if text.count("</answer>") == 1:
            count += 0.125
            # Penalize content after </answer>
            count -= len(text.split("</answer>")[-1]) * 0.001

        return count

    def _extract_xml_answer(self, response: str) -> str:
        """
        Extract answer from XML format response.

        Looks for <answer>...</answer> tags first, then falls back to
        standard answer extraction patterns.

        Args:
            response: Model response text

        Returns:
            Extracted answer or "Unable to Extract"
        """
        # First try to extract from XML format
        xml_pattern = r"<answer>\s*(.*?)\s*</answer>"
        match = re.search(xml_pattern, response, re.DOTALL)

        if match:
            answer = match.group(1).strip()
            # Clean up the answer
            answer = re.sub(r"\n+", " ", answer)  # Replace newlines with spaces
            answer = re.sub(r"\s+", " ", answer)  # Normalize whitespace
            return answer

        # Fall back to standard extraction
        return extract_final_answer(response)

    def combine_rewards(
        self, rewards_dict: Dict[str, List[float]], weights: Optional[Dict[str, float]] = None
    ) -> List[float]:
        """
        Combine multiple reward signals into a final reward using weighted SUM.

        The DTE paper specifies a weighted sum (not a weighted average). With
        the DTE weights (correctness=2.0, all others=0.5), a perfect response
        achieves a combined reward of 5.0 (2.0*2.0 + 0.5*0.5 + 0.5*0.5 +
        0.5*0.5 + 0.5*0.5). When no weights dict is provided, defaults to
        1.0 for every function.

        Args:
            rewards_dict: Dictionary mapping reward function names to lists
                of per-response reward values.
            weights: Optional dictionary mapping reward function names to
                scalar weights. Defaults to 1.0 for every function.

        Returns:
            List of combined (weighted-sum) rewards, one per response.
        """
        if not rewards_dict:
            return []

        # Default weights (equal for all reward functions)
        if weights is None:
            weights = {key: 1.0 for key in rewards_dict.keys()}

        # Get the number of responses
        num_responses = len(next(iter(rewards_dict.values())))

        # Combine rewards as weighted SUM (not average)
        combined_rewards = []
        for i in range(num_responses):
            total_reward = 0.0

            for reward_type, reward_list in rewards_dict.items():
                if i < len(reward_list):
                    weight = weights.get(reward_type, 1.0)
                    total_reward += reward_list[i] * weight

            combined_rewards.append(total_reward)

        return combined_rewards

    def get_reward_statistics(self, rewards_dict: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each reward function.

        Args:
            rewards_dict: Dictionary of reward lists

        Returns:
            Dictionary with statistics for each reward function
        """
        stats = {}

        for reward_type, rewards in rewards_dict.items():
            if rewards:
                stats[reward_type] = {
                    "mean": sum(rewards) / len(rewards),
                    "min": min(rewards),
                    "max": max(rewards),
                    "std": (sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)) ** 0.5,
                }
            else:
                stats[reward_type] = {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        return stats
