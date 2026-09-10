"""Tests for all 5 DTE reward functions."""

import pytest

from dte.training.reward_model import DTERewardModel


@pytest.fixture
def model():
    return DTERewardModel()


# -----------------------------------------------------------------------
# 1. correctness_reward_func
# -----------------------------------------------------------------------


class TestCorrectnessReward:
    """Tests for the correctness reward function."""

    def test_correct_answer(self, model):
        response = "<reasoning>5*6=30</reasoning>\n<answer>\n30\n</answer>\n"
        rewards = model.correctness_reward_func([response], "30")
        assert rewards == [2.0]

    def test_incorrect_answer(self, model):
        response = "<reasoning>5*6=31</reasoning>\n<answer>\n31\n</answer>\n"
        rewards = model.correctness_reward_func([response], "30")
        assert rewards == [0.0]

    def test_batch(self, model):
        responses = [
            "<answer>42</answer>",
            "<answer>99</answer>",
            "<answer>42</answer>",
        ]
        rewards = model.correctness_reward_func(responses, "42")
        assert rewards == [2.0, 0.0, 2.0]


# -----------------------------------------------------------------------
# 2. int_reward_func
# -----------------------------------------------------------------------


class TestIntReward:
    """Tests for the integer/numeric reward function."""

    def test_integer_answer(self, model):
        response = "<answer>42</answer>"
        rewards = model.int_reward_func([response])
        assert rewards == [0.5]

    def test_non_numeric_answer(self, model):
        response = "<answer>hello world</answer>"
        rewards = model.int_reward_func([response])
        assert rewards == [0.0]

    def test_float_answer(self, model):
        response = "<answer>3.14</answer>"
        rewards = model.int_reward_func([response])
        assert rewards == [0.5]

    def test_batch(self, model):
        responses = [
            "<answer>10</answer>",
            "<answer>abc</answer>",
        ]
        rewards = model.int_reward_func(responses)
        assert rewards[0] == 0.5
        assert rewards[1] == 0.0


# -----------------------------------------------------------------------
# 3. strict_format_reward_func
# -----------------------------------------------------------------------


class TestStrictFormatReward:
    """Tests for strict XML format compliance."""

    def test_perfect_format(self, model):
        response = "<reasoning>\nStep-by-step reasoning here\n</reasoning>\n<answer>\n42\n</answer>\n"
        rewards = model.strict_format_reward_func([response])
        assert rewards == [0.5]

    def test_missing_newlines(self, model):
        response = "<reasoning>reasoning</reasoning><answer>42</answer>"
        rewards = model.strict_format_reward_func([response])
        assert rewards == [0.0]

    def test_no_tags(self, model):
        response = "The answer is 42."
        rewards = model.strict_format_reward_func([response])
        assert rewards == [0.0]


# -----------------------------------------------------------------------
# 4. soft_format_reward_func
# -----------------------------------------------------------------------


class TestSoftFormatReward:
    """Tests for flexible XML format compliance."""

    def test_basic_xml(self, model):
        response = "<reasoning>step by step</reasoning><answer>42</answer>"
        rewards = model.soft_format_reward_func([response])
        assert rewards == [0.5]

    def test_xml_with_spaces(self, model):
        response = "<reasoning>step by step</reasoning>  <answer>42</answer>"
        rewards = model.soft_format_reward_func([response])
        assert rewards == [0.5]

    def test_no_tags(self, model):
        response = "The answer is 42."
        rewards = model.soft_format_reward_func([response])
        assert rewards == [0.0]

    def test_multiline_content(self, model):
        response = "<reasoning>\nStep 1\nStep 2\n</reasoning>\n<answer>\n42\n</answer>"
        rewards = model.soft_format_reward_func([response])
        assert rewards == [0.5]


# -----------------------------------------------------------------------
# 5. xmlcount_reward_func
# -----------------------------------------------------------------------


class TestXMLCountReward:
    """Tests for granular XML scoring."""

    def test_perfect_xml(self, model):
        response = "<reasoning>reason</reasoning><answer>42</answer>"
        rewards = model.xmlcount_reward_func([response])
        # All 4 tags present: 4 * 0.125 = 0.5, minus any trailing chars
        assert rewards[0] == pytest.approx(0.5)

    def test_no_tags(self, model):
        response = "No XML tags here."
        rewards = model.xmlcount_reward_func([response])
        assert rewards[0] == 0.0

    def test_extra_content_after_answer(self, model):
        response = "<reasoning>r</reasoning><answer>42</answer>extra"
        rewards = model.xmlcount_reward_func([response])
        # 0.5 - 0.001 * len("extra") = 0.5 - 0.005 = 0.495
        assert rewards[0] == pytest.approx(0.495)

    def test_duplicate_tags_no_reward(self, model):
        response = "<reasoning>a</reasoning><reasoning>b</reasoning><answer>42</answer>"
        rewards = model.xmlcount_reward_func([response])
        # <reasoning> appears twice, so no +0.125 for it
        # </reasoning> appears twice, so no +0.125 for it
        # <answer> appears once -> +0.125
        # </answer> appears once -> +0.125 - penalty
        assert rewards[0] < 0.5


# -----------------------------------------------------------------------
# Combined reward calculations
# -----------------------------------------------------------------------


class TestCombineRewards:
    """Tests for combining multiple reward signals (weighted SUM)."""

    def test_combine_equal_weights(self, model):
        rewards_dict = {
            "correctness": [2.0],
            "int": [0.5],
        }
        combined = model.combine_rewards(rewards_dict)
        assert len(combined) == 1
        # Weighted SUM: 2.0 * 1.0 + 0.5 * 1.0 = 2.5
        assert combined[0] == pytest.approx(2.5)

    def test_combine_custom_weights(self, model):
        rewards_dict = {
            "correctness": [2.0],
            "int": [0.5],
        }
        weights = {"correctness": 2.0, "int": 0.5}
        combined = model.combine_rewards(rewards_dict, weights)
        # Weighted SUM: 2.0 * 2.0 + 0.5 * 0.5 = 4.25
        assert combined[0] == pytest.approx(4.25)

    def test_combine_empty(self, model):
        assert model.combine_rewards({}) == []

    def test_combine_max_reward(self, model):
        """With default DTE weights the max possible reward is 4.0."""
        rewards_dict = {
            "correctness": [2.0],
            "int": [0.5],
            "strict_format": [0.5],
            "soft_format": [0.5],
            "xmlcount": [0.5],
        }
        weights = {
            "correctness": 2.0,
            "int": 0.5,
            "strict_format": 0.5,
            "soft_format": 0.5,
            "xmlcount": 0.5,
        }
        combined = model.combine_rewards(rewards_dict, weights)
        # 2.0*2.0 + 0.5*0.5 + 0.5*0.5 + 0.5*0.5 + 0.5*0.5 = 4.0 + 1.0 = 5.0
        assert combined[0] == pytest.approx(5.0)


class TestCalculateAllRewards:
    """Tests for the full reward calculation pipeline."""

    def test_all_five_rewards_returned(self, model):
        response = "<reasoning>\nSolving step by step\n</reasoning>\n<answer>\n42\n</answer>\n"
        rewards = model.calculate_all_rewards("What is 6*7?", [response], "42")
        assert set(rewards.keys()) == {"correctness", "int", "strict_format", "soft_format", "xmlcount"}
        for v in rewards.values():
            assert len(v) == 1

    def test_no_ground_truth(self, model):
        response = "<answer>42</answer>"
        rewards = model.calculate_all_rewards("question", [response], ground_truth=None)
        assert rewards["correctness"] == [0.0]

    def test_reward_statistics(self, model):
        rewards_dict = {
            "correctness": [2.0, 0.0, 2.0],
            "int": [0.5, 0.5, 0.0],
        }
        stats = model.get_reward_statistics(rewards_dict)
        assert stats["correctness"]["mean"] == pytest.approx(4.0 / 3)
        assert stats["int"]["min"] == 0.0
        assert stats["int"]["max"] == 0.5
