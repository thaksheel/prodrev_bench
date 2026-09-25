"""
Robust answer extraction utilities based on original DTE implementation.

This module provides exact implementations of answer extraction, consensus detection,
and sycophancy tracking as used in the original DTE codebase.
"""

import re
from typing import Any, Dict, List, Optional, Union


def extract_final_answer(response: str) -> str:
    """
    Extract final answer from model response using multiple patterns.

    This is the exact implementation from the original DTE codebase,
    implementing robust answer extraction with fallback patterns.

    Args:
        response: The model's response text

    Returns:
        Extracted answer as string, or "Unable to Extract" if no answer found
    """
    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"\*\*Final Answer\*\*:.*?([$\£€]?[\d,]+(?:\.\d+)?(?:[ million| thousand|%])?)",
        r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?)",
        r"(\d+\.?\d*)",
    ]

    for pattern in patterns:
        match = re.findall(pattern, response, re.IGNORECASE)
        if match:
            try:
                raw_answer = match[-1].strip()
                # Clean and standardize the answer
                cleaned = raw_answer.replace(",", "").replace("$", "").replace("£", "").replace("€", "")

                # Handle LaTeX formatting
                cleaned = re.sub(r"\\text\{[^}]*\}", "", cleaned).strip()

                if "%" in raw_answer:
                    cleaned = cleaned.replace("%", "")

                # Extract numeric value before "million" or "thousand"
                if "million" in raw_answer.lower():
                    num_match = re.search(r"(\d+\.?\d*)", cleaned)
                    if num_match:
                        cleaned = num_match.group(1)
                        cleaned = str(int(float(cleaned) * 1_000_000))
                elif "thousand" in raw_answer.lower():
                    num_match = re.search(r"(\d+\.?\d*)", cleaned)
                    if num_match:
                        cleaned = num_match.group(1)
                        cleaned = str(int(float(cleaned) * 1_000))

                return cleaned.split(".")[0]  # Return integer part
            except Exception:
                # If any error occurs during processing, continue to next pattern
                continue

    return "Unable to Extract"


def extract_ground_truth(answer: str) -> str:
    """
    Extract ground truth answer using #### split format.

    Args:
        answer: Ground truth answer string

    Returns:
        Cleaned ground truth answer
    """
    return answer.split("####")[-1].strip()


def clean_numeric_string(s: Union[str, int, float]) -> Optional[Union[int, float]]:
    """
    Extract and normalize numeric values from strings.

    Args:
        s: Input string, int, or float

    Returns:
        Normalized numeric value or None if extraction fails
    """
    if not isinstance(s, str):
        if isinstance(s, (int, float)):
            return s
        return None

    matches = re.findall(r"[-+]?\d*\.?\d+|\d+", s)
    if not matches:
        return None

    number_str = matches[-1].replace(",", "").strip()

    try:
        if "." in number_str:
            return round(float(number_str), 2)
        return int(number_str)
    except (ValueError, TypeError):
        return None


def answers_match(answer1: str, answer2: str) -> bool:
    """
    Check if two answers match numerically with 1e-9 tolerance.

    This implements the exact comparison logic from the original DTE codebase,
    using 1e-9 tolerance for floating point comparisons.

    Args:
        answer1: First answer to compare
        answer2: Second answer to compare

    Returns:
        True if answers match within tolerance
    """
    clean1 = clean_numeric_string(answer1)
    clean2 = clean_numeric_string(answer2)

    if clean1 is not None and clean2 is not None:
        return abs(clean1 - clean2) < 1e-9
    return False


def check_consensus(answers: List[str]) -> bool:
    """
    Check if all agents have reached a consensus.

    Args:
        answers: List of extracted answers from all agents

    Returns:
        True if consensus is reached (all answers match)
    """
    if "Unable to Extract" in answers:
        return False

    first_answer = answers[0]
    return all(answers_match(first_answer, answer) for answer in answers)


def detect_sycophancy(agent_answers: Dict[str, List[str]], round_num: int) -> Dict[str, bool]:
    """
    Detect if agents changed their answers to match other agents' previous answers.

    This implements the exact sycophancy detection algorithm from the original
    DTE codebase, identifying when agents abandon their previous answers to
    match peers.

    Args:
        agent_answers: Dictionary mapping agent IDs to their answer history
        round_num: Current round number

    Returns:
        Dictionary mapping agent IDs to boolean sycophancy indicators
    """
    if round_num <= 0:
        return {}

    sycophancy_detected = {}

    for agent_id, answers in agent_answers.items():
        if len(answers) < round_num + 1:
            continue

        # Current answer for this agent
        current_answer = answers[round_num]
        # Previous answer for this agent
        prev_answer = answers[round_num - 1]

        # If the agent changed their answer
        if not answers_match(current_answer, prev_answer):
            # Check if the new answer matches any other agent's previous answer
            sycophancy = False
            for other_id, other_answers in agent_answers.items():
                if other_id != agent_id and len(other_answers) >= round_num:
                    if answers_match(current_answer, other_answers[round_num - 1]):
                        sycophancy = True
                        break

            sycophancy_detected[agent_id] = sycophancy

    return sycophancy_detected


def extract_arc_answer(response: str) -> str:
    """
    Extract answer choice from ARC-Challenge responses.

    Args:
        response: Model response containing letter choice

    Returns:
        Extracted letter choice or "Unable to Extract"
    """
    # Look for patterns like "Answer: A" or "The answer is B"
    patterns = [
        r"(?:answer|choice)(?:\s+is)?\s*:?\s*([A-D])",
        r"(?:^|\s)([A-D])(?:\s|$|\.|,)",
        r"\\boxed\{([A-D])\}",
        r"\(([A-D])\)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    return "Unable to Extract"


def calculate_accuracy(predictions: List[str], ground_truths: List[str], task_type: str = "math") -> float:
    """
    Calculate accuracy for predictions vs ground truth.

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        task_type: Type of task ("math" or "arc")

    Returns:
        Accuracy as float between 0 and 1
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    correct = 0
    total = len(predictions)

    for pred, truth in zip(predictions, ground_truths):
        if task_type == "math":
            if answers_match(pred, truth):
                correct += 1
        else:  # ARC or other text-based tasks
            if pred.upper() == truth.upper():
                correct += 1

    return correct / total if total > 0 else 0.0


def consolidate_reasoning_traces(all_responses: List[List[str]], final_answer: str) -> str:
    """
    Extract consolidated reasoning from debate traces.

    Identifies reasoning steps that either appear in multiple agents' responses
    or introduce novel symbolic manipulations, as described in the paper.

    Args:
        all_responses: List of responses for each round [round][agent]
        final_answer: The consensus or majority-voted final answer

    Returns:
        Consolidated reasoning string
    """
    reasoning_steps = []
    step_counts = {}

    # Extract reasoning steps from all responses
    for round_responses in all_responses:
        for response in round_responses:
            steps = _extract_reasoning_steps(response)
            for step in steps:
                step_counts[step] = step_counts.get(step, 0) + 1

    # Select steps that appear multiple times or are particularly insightful
    for step, count in step_counts.items():
        if count > 1 or _is_novel_manipulation(step):
            reasoning_steps.append((step, count))

    # Sort by frequency and relevance
    reasoning_steps.sort(key=lambda x: x[1], reverse=True)

    # Take top reasoning steps and format them
    consolidated = []
    for step, _ in reasoning_steps[:10]:  # Top 10 steps
        if step.strip() and len(step) > 20:  # Filter short/empty steps
            consolidated.append(step.strip())

    return "\n".join(consolidated) if consolidated else f"Solution leading to {final_answer}"


def _extract_reasoning_steps(response: str) -> List[str]:
    """Extract individual reasoning steps from response text."""
    # Split by common sentence endings and step indicators
    sentences = re.split(r"[.!?]+|\n+|Step \d+[:.]?|\d+[\.)]\s*", response)
    steps = []

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 15:  # Filter very short sentences
            # Clean up common artifacts
            sentence = re.sub(r"^[•\-\*\s]+", "", sentence)
            sentence = re.sub(r"\s+", " ", sentence)
            if sentence:
                steps.append(sentence)

    return steps


def _is_novel_manipulation(step: str) -> bool:
    """Check if a reasoning step introduces novel symbolic manipulations."""
    # Look for mathematical operations, equations, symbolic reasoning
    novel_indicators = [
        r"\d+\s*[+\-\*/=]\s*\d+",  # Mathematical operations
        r"[a-zA-Z]\s*=\s*",  # Variable assignments
        r"\\frac\{[^}]+\}\{[^}]+\}",  # Fractions
        r"therefore|thus|hence|consequently|implies",  # Logical connectors
        r"let\s+[a-zA-Z]|assume|suppose",  # Assumptions
        r"\$.*\$",  # LaTeX math
    ]

    step_lower = step.lower()
    return any(re.search(pattern, step_lower) for pattern in novel_indicators)
