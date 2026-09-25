"""
GPU integration test: test reward functions on actual model outputs.

Uses Qwen/Qwen2.5-0.5B-Instruct on CUDA_VISIBLE_DEVICES=4.
Marked with @pytest.mark.gpu so it is skipped when no GPU is available.

Tests cover:
- All 5 reward functions on real model-generated output
- Combined reward is a weighted sum
- Reward statistics
- Reward breakdown analysis
- Edge cases: empty responses, very long responses, no XML tags
"""

import os

import pytest

# Pin to GPU 4 before any CUDA initialization
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from dte.core.config import DebateConfig, ModelConfig
from dte.debate.manager import DebateManager
from dte.training.reward_model import DTERewardModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.gpu
class TestRewardIntegration:
    """Test reward functions on actual model-generated debate output."""

    @pytest.fixture(scope="class")
    def debate_result(self):
        """Run a real debate and return the result for reward analysis."""
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        debate_cfg = DebateConfig(num_agents=2, max_rounds=1)
        manager = DebateManager(debate_cfg, model_cfg, logger=None)
        try:
            result = manager.conduct_debate("What is 6 * 7?", task_type="math")
            yield result
        finally:
            manager.cleanup()

    @pytest.fixture
    def reward_model(self):
        return DTERewardModel()

    def test_all_five_rewards_on_real_output(self, debate_result, reward_model):
        """All 5 reward functions should return valid floats for real output."""
        # Grab reasoning from the first agent's last response
        response_text = debate_result.all_responses[-1][0].reasoning
        rewards = reward_model.calculate_all_rewards(
            query="What is 6 * 7?",
            responses=[response_text],
            ground_truth="42",
        )
        assert set(rewards.keys()) == {"correctness", "int", "strict_format", "soft_format", "xmlcount"}
        for name, values in rewards.items():
            assert len(values) == 1
            assert isinstance(values[0], float), f"{name} reward is not float"

    def test_combined_reward_is_weighted_sum(self, debate_result, reward_model):
        """Combined reward should be a weighted sum, not an average."""
        response_text = debate_result.all_responses[-1][0].reasoning
        rewards = reward_model.calculate_all_rewards(
            query="What is 6 * 7?",
            responses=[response_text],
            ground_truth="42",
        )
        weights = {
            "correctness": 2.0,
            "int": 0.5,
            "strict_format": 0.5,
            "soft_format": 0.5,
            "xmlcount": 0.5,
        }
        combined = reward_model.combine_rewards(rewards, weights)
        assert len(combined) == 1

        # Manually compute expected weighted sum
        expected = sum(rewards[k][0] * weights[k] for k in rewards)
        assert combined[0] == pytest.approx(expected, abs=1e-6)

    def test_reward_statistics(self, debate_result, reward_model):
        """get_reward_statistics should return valid stats."""
        responses = [r.reasoning for r in debate_result.all_responses[-1]]
        rewards = reward_model.calculate_all_rewards(
            query="What is 6 * 7?",
            responses=responses,
            ground_truth="42",
        )
        stats = reward_model.get_reward_statistics(rewards)
        for name in rewards:
            assert name in stats
            assert "mean" in stats[name]
            assert "min" in stats[name]
            assert "max" in stats[name]


@pytest.mark.gpu
class TestRewardBreakdownAnalysis:
    """Detailed analysis of reward component behavior."""

    @pytest.fixture
    def reward_model(self):
        return DTERewardModel()

    def test_perfect_format_gets_all_format_rewards(self, reward_model):
        """A perfectly formatted response should get max format rewards."""
        perfect = "<reasoning>\nStep 1: 6*7=42\n</reasoning>\n<answer>\n42\n</answer>\n"
        rewards = reward_model.calculate_all_rewards(
            query="What is 6*7?",
            responses=[perfect],
            ground_truth="42",
        )
        # Correctness should be 2.0 (correct answer)
        assert rewards["correctness"][0] == 2.0
        # Int reward should be 0.5 (answer is numeric)
        assert rewards["int"][0] == 0.5
        # Strict format should be 0.5
        assert rewards["strict_format"][0] == 0.5
        # Soft format should be 0.5
        assert rewards["soft_format"][0] == 0.5
        # XML count should be positive (all tags present)
        assert rewards["xmlcount"][0] > 0.0

    def test_wrong_answer_gets_zero_correctness(self, reward_model):
        """Wrong answer should get 0.0 correctness but may get format rewards."""
        wrong = "<reasoning>\n6*7=43\n</reasoning>\n<answer>\n43\n</answer>\n"
        rewards = reward_model.calculate_all_rewards(
            query="What is 6*7?",
            responses=[wrong],
            ground_truth="42",
        )
        assert rewards["correctness"][0] == 0.0
        # Format rewards should still be given for proper formatting
        assert rewards["strict_format"][0] == 0.5
        assert rewards["soft_format"][0] == 0.5

    def test_no_xml_format_gets_zero_format_rewards(self, reward_model):
        """Response without XML tags should get zero format rewards."""
        plain = "The answer is 42."
        rewards = reward_model.calculate_all_rewards(
            query="What is 6*7?",
            responses=[plain],
            ground_truth="42",
        )
        assert rewards["strict_format"][0] == 0.0
        assert rewards["soft_format"][0] == 0.0
        assert rewards["xmlcount"][0] == 0.0

    def test_multiple_responses_batch(self, reward_model):
        """Reward calculation should work for batches of responses."""
        responses = [
            "<reasoning>\n42\n</reasoning>\n<answer>\n42\n</answer>\n",
            "The answer is 42",
            "<reasoning>\nwrong\n</reasoning>\n<answer>\n99\n</answer>\n",
        ]
        rewards = reward_model.calculate_all_rewards(
            query="What is 6*7?",
            responses=responses,
            ground_truth="42",
        )
        for name, values in rewards.items():
            assert len(values) == 3, f"{name} should have 3 values"


@pytest.mark.gpu
class TestRewardEdgeCases:
    """Test reward functions on edge case inputs."""

    @pytest.fixture
    def reward_model(self):
        return DTERewardModel()

    def test_empty_response(self, reward_model):
        """Empty response should not crash and should return valid floats."""
        rewards = reward_model.calculate_all_rewards(
            query="What is 1+1?",
            responses=[""],
            ground_truth="2",
        )
        for name, values in rewards.items():
            assert len(values) == 1
            assert isinstance(values[0], float)
            assert values[0] == values[0]  # not NaN

    def test_very_long_response(self, reward_model):
        """Very long response should not crash."""
        long_response = "<reasoning>\n" + "This is a long step. " * 500 + "\n</reasoning>\n<answer>\n42\n</answer>\n"
        rewards = reward_model.calculate_all_rewards(
            query="What is 6*7?",
            responses=[long_response],
            ground_truth="42",
        )
        for name, values in rewards.items():
            assert len(values) == 1
            assert isinstance(values[0], float)

    def test_response_with_special_characters(self, reward_model):
        """Response with special characters should not crash."""
        special = "<reasoning>\n$100 + $200 = $300 & 50% off\n</reasoning>\n<answer>\n300\n</answer>\n"
        rewards = reward_model.calculate_all_rewards(
            query="What is 100+200?",
            responses=[special],
            ground_truth="300",
        )
        assert rewards["correctness"][0] == 2.0

    def test_no_ground_truth(self, reward_model):
        """Without ground truth, correctness reward should be 0."""
        response = "<reasoning>\n42\n</reasoning>\n<answer>\n42\n</answer>\n"
        rewards = reward_model.calculate_all_rewards(
            query="What is 6*7?",
            responses=[response],
            ground_truth=None,
        )
        assert rewards["correctness"][0] == 0.0
        # Other rewards should still work
        assert rewards["strict_format"][0] == 0.5

    def test_duplicate_xml_tags(self, reward_model):
        """Duplicate XML tags should reduce xmlcount reward."""
        dup = "<reasoning>step1</reasoning>\n<reasoning>step2</reasoning>\n<answer>42</answer>\n"
        rewards = reward_model.calculate_all_rewards(
            query="What is 6*7?",
            responses=[dup],
            ground_truth="42",
        )
        # With duplicate <reasoning>, xmlcount should not give full credit
        # since count("<reasoning>") == 2 != 1
        assert rewards["xmlcount"][0] < 0.5

    def test_content_after_answer_tag_penalized(self, reward_model):
        """Content after </answer> should reduce xmlcount reward."""
        trailing = (
            "<reasoning>\nstep\n</reasoning>\n"
            "<answer>\n42\n</answer>\n"
            "Extra trailing content here that should be penalized."
        )
        rewards_trailing = reward_model.calculate_all_rewards(
            query="What is 6*7?",
            responses=[trailing],
            ground_truth="42",
        )

        clean = "<reasoning>\nstep\n</reasoning>\n<answer>\n42\n</answer>\n"
        rewards_clean = reward_model.calculate_all_rewards(
            query="What is 6*7?",
            responses=[clean],
            ground_truth="42",
        )

        # Trailing content should reduce xmlcount
        assert rewards_trailing["xmlcount"][0] < rewards_clean["xmlcount"][0]


@pytest.mark.gpu
class TestRewardOnRealModelOutput:
    """Run a real debate, then score every agent response with all rewards."""

    def test_score_all_agent_responses(self):
        """Score every response from a real debate with all 5 reward functions."""
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        debate_cfg = DebateConfig(num_agents=3, max_rounds=2)
        manager = DebateManager(debate_cfg, model_cfg, logger=None)
        reward_model = DTERewardModel()

        try:
            result = manager.conduct_debate("What is 8 * 9?", task_type="math")

            # Score every agent's response in every round
            for round_idx, round_responses in enumerate(result.all_responses):
                all_reasoning = [r.reasoning for r in round_responses]
                rewards = reward_model.calculate_all_rewards(
                    query="What is 8 * 9?",
                    responses=all_reasoning,
                    ground_truth="72",
                )

                assert len(rewards["correctness"]) == len(round_responses)
                for name, values in rewards.items():
                    for v in values:
                        assert isinstance(v, float)
                        assert v == v  # not NaN

                # Combined reward should work too
                weights = {
                    "correctness": 2.0,
                    "int": 0.5,
                    "strict_format": 0.5,
                    "soft_format": 0.5,
                    "xmlcount": 0.5,
                }
                combined = reward_model.combine_rewards(rewards, weights)
                assert len(combined) == len(round_responses)

        finally:
            manager.cleanup()
