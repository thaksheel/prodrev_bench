"""Tests for answer extraction utilities."""

import pytest

from dte.utils.answer_extraction import (
    answers_match,
    calculate_accuracy,
    check_consensus,
    clean_numeric_string,
    consolidate_reasoning_traces,
    detect_sycophancy,
    extract_arc_answer,
    extract_final_answer,
    extract_ground_truth,
)


class TestExtractFinalAnswer:
    """Tests for the extract_final_answer function."""

    def test_boxed_format(self):
        assert extract_final_answer("The answer is \\boxed{42}") == "42"

    def test_boxed_with_comma(self):
        assert extract_final_answer("\\boxed{1,234}") == "1234"

    def test_simple_number(self):
        result = extract_final_answer("The answer is 42")
        assert result == "42"

    def test_decimal_returns_integer_part(self):
        result = extract_final_answer("The result is 3.14")
        assert result == "3"

    def test_no_answer(self):
        assert extract_final_answer("I don't know") == "Unable to Extract"

    def test_multiple_numbers_last_used(self):
        result = extract_final_answer("First I got 10, then I got 20, so the answer is 30")
        # Should extract last occurrence
        assert result == "30"

    def test_currency_stripped(self):
        result = extract_final_answer("The total cost is $150")
        assert result == "150"


class TestExtractGroundTruth:
    """Tests for ground truth extraction."""

    def test_hash_format(self):
        assert extract_ground_truth("Long solution text #### 42") == "42"

    def test_no_hash(self):
        assert extract_ground_truth("42") == "42"

    def test_whitespace(self):
        assert extract_ground_truth("solution ####  55 ") == "55"


class TestCleanNumericString:
    """Tests for clean_numeric_string."""

    def test_integer_string(self):
        assert clean_numeric_string("42") == 42

    def test_float_string(self):
        assert clean_numeric_string("3.14") == 3.14

    def test_with_text(self):
        result = clean_numeric_string("the answer is 42")
        assert result == 42

    def test_negative_number(self):
        result = clean_numeric_string("-5")
        assert result == -5

    def test_none_on_no_number(self):
        assert clean_numeric_string("no numbers here") is None

    def test_int_input(self):
        assert clean_numeric_string(42) == 42

    def test_float_input(self):
        assert clean_numeric_string(3.14) == 3.14

    def test_none_input(self):
        assert clean_numeric_string(None) is None

    def test_empty_string(self):
        assert clean_numeric_string("") is None


class TestAnswersMatch:
    """Tests for answers_match."""

    def test_exact_match(self):
        assert answers_match("42", "42") is True

    def test_float_match(self):
        assert answers_match("42.0", "42") is True

    def test_no_match(self):
        assert answers_match("42", "43") is False

    def test_with_text(self):
        assert answers_match("answer is 42", "42") is True

    def test_non_numeric(self):
        assert answers_match("hello", "world") is False


class TestCheckConsensus:
    """Tests for check_consensus."""

    def test_all_agree(self):
        assert check_consensus(["42", "42", "42"]) is True

    def test_disagreement(self):
        assert check_consensus(["42", "43", "42"]) is False

    def test_unable_to_extract(self):
        assert check_consensus(["42", "Unable to Extract", "42"]) is False

    def test_single_agent(self):
        assert check_consensus(["42"]) is True

    def test_numeric_tolerance(self):
        assert check_consensus(["42", "42.0", "42"]) is True


class TestDetectSycophancy:
    """Tests for sycophancy detection."""

    def test_no_sycophancy(self):
        history = {
            "1": ["42", "42"],
            "2": ["43", "43"],
        }
        result = detect_sycophancy(history, 1)
        # Neither agent changed, so no sycophancy
        assert not any(result.values()) if result else True

    def test_sycophancy_detected(self):
        history = {
            "1": ["42", "42"],
            "2": ["43", "42"],  # Agent 2 switched to match agent 1
        }
        result = detect_sycophancy(history, 1)
        assert result.get("2") is True

    def test_round_zero(self):
        history = {"1": ["42"], "2": ["43"]}
        result = detect_sycophancy(history, 0)
        assert result == {}


class TestExtractArcAnswer:
    """Tests for ARC answer extraction."""

    def test_answer_colon_format(self):
        assert extract_arc_answer("The answer is: A") == "A"

    def test_answer_format(self):
        assert extract_arc_answer("Answer: B") == "B"

    def test_boxed_format(self):
        assert extract_arc_answer("\\boxed{C}") == "C"

    def test_no_answer(self):
        assert extract_arc_answer("I don't know") == "Unable to Extract"


class TestCalculateAccuracy:
    """Tests for accuracy calculation."""

    def test_perfect_score(self):
        preds = ["42", "43", "44"]
        truths = ["42", "43", "44"]
        assert calculate_accuracy(preds, truths) == 1.0

    def test_zero_score(self):
        preds = ["1", "2", "3"]
        truths = ["4", "5", "6"]
        assert calculate_accuracy(preds, truths) == 0.0

    def test_partial_score(self):
        preds = ["42", "99", "44"]
        truths = ["42", "43", "44"]
        assert calculate_accuracy(preds, truths) == pytest.approx(2 / 3)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            calculate_accuracy(["42"], ["42", "43"])

    def test_empty_lists(self):
        assert calculate_accuracy([], []) == 0.0

    def test_arc_mode(self):
        preds = ["a", "B", "c"]
        truths = ["A", "b", "C"]
        assert calculate_accuracy(preds, truths, task_type="arc") == 1.0


class TestConsolidateReasoningTraces:
    """Tests for consolidate_reasoning_traces."""

    def test_basic_consolidation(self):
        responses = [
            ["First I add 5 + 3 = 8 to get the intermediate result", "Adding 5 and 3 gives us 8 as a first step"],
            [
                "Then multiply 8 * 2 = 16 for the final answer",
                "The product of 8 and 2 is 16 therefore the answer is 16",
            ],
        ]
        result = consolidate_reasoning_traces(responses, "16")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_responses(self):
        result = consolidate_reasoning_traces([], "42")
        assert isinstance(result, str)

    def test_fallback_when_no_steps(self):
        result = consolidate_reasoning_traces([["ok"]], "42")
        assert "42" in result
