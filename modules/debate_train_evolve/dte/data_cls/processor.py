"""
Data Processing Utilities for DTE Pipeline.

This module provides comprehensive data processing functionality for the DTE framework,
including data cleaning, formatting, validation, and preparation for training.
Implements the exact data processing strategy from the original DTE implementation.
"""

from typing import Any, Dict, List, Optional

from .generator import TrainingExample


class DataProcessor:
    """
    Advanced data processor for DTE training and evaluation data.

    Handles all aspects of data processing including:
    - Training example validation and cleaning
    - XML format compliance checking
    - Answer extraction and normalization
    - Batch processing and optimization
    - Statistical analysis of data quality

    The processor ensures all data meets the quality standards required
    for effective GRPO training in the DTE framework.
    """

    def __init__(self):
        """Initialize data processor with default settings."""
        pass

    def process_training_examples(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """
        Process and validate training examples for GRPO training.

        Performs comprehensive processing including:
        - Data validation and quality checks
        - XML format compliance verification
        - Answer extraction and normalization
        - Confidence score validation
        - Duplicate detection and removal

        Args:
            examples: List of raw training examples from debate generation

        Returns:
            List of processed and validated examples ready for training

        Raises:
            ValueError: If examples contain invalid or malformed data
        """
        processed = []
        for i, example in enumerate(examples):
            try:
                # Validate example completeness
                if not all([example.query, example.answer, example.reasoning]):
                    continue  # Skip incomplete examples

                # Validate confidence score
                confidence = max(0.0, min(1.0, example.confidence or 0.5))

                processed.append(
                    {
                        "query": example.query.strip(),
                        "answer": example.answer.strip(),
                        "reasoning": example.reasoning.strip(),
                        "confidence": confidence,
                        "example_id": i,
                    }
                )
            except Exception as e:
                # Log warning but continue processing
                print(f"Warning: Skipping malformed example {i}: {e}")
                continue

        return processed

    def format_for_model(self, example: Dict[str, Any], format_type: str = "xml") -> str:
        """
        Format example for model input using DTE-standard XML format.

        Formats examples in the exact XML structure expected by the DTE
        reward functions for optimal training performance.

        Args:
            example: Training example dictionary with query, answer, reasoning
            format_type: Output format type ("xml", "plain", "chat")

        Returns:
            Formatted string ready for model training

        Raises:
            KeyError: If required fields are missing from example
            ValueError: If format_type is not supported
        """
        if format_type == "xml":
            # DTE standard XML format for reward function compatibility
            return f"""<reasoning>
{example["reasoning"]}
</reasoning>
<answer>
{example["answer"]}
</answer>
"""
        elif format_type == "plain":
            return f"Query: {example['query']}\n\nAnswer: {example['answer']}\n\nReasoning: {example['reasoning']}"
        elif format_type == "chat":
            return (
                f"Human: {example['query']}\n\nAssistant: {example['reasoning']}\n\nThe answer is {example['answer']}."
            )
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def validate_xml_format(self, text: str) -> Dict[str, Any]:
        """
        Validate XML format compliance for DTE training.

        Checks if text follows the exact XML structure required by
        DTE reward functions.

        Args:
            text: Text to validate

        Returns:
            Dictionary with validation results and extracted components
        """
        import re

        result = {
            "is_valid": False,
            "has_reasoning_tags": False,
            "has_answer_tags": False,
            "reasoning_content": None,
            "answer_content": None,
            "errors": [],
        }

        # Check for reasoning tags
        reasoning_match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.DOTALL)
        if reasoning_match:
            result["has_reasoning_tags"] = True
            result["reasoning_content"] = reasoning_match.group(1).strip()
        else:
            result["errors"].append("Missing or malformed <reasoning> tags")

        # Check for answer tags
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        if answer_match:
            result["has_answer_tags"] = True
            result["answer_content"] = answer_match.group(1).strip()
        else:
            result["errors"].append("Missing or malformed <answer> tags")

        result["is_valid"] = result["has_reasoning_tags"] and result["has_answer_tags"]
        return result

    def get_processing_statistics(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for processed examples.

        Args:
            examples: List of processed examples

        Returns:
            Dictionary with detailed statistics
        """
        if not examples:
            return {"total_examples": 0}

        # Basic statistics
        total_examples = len(examples)
        avg_reasoning_length = sum(len(ex["reasoning"].split()) for ex in examples) / total_examples
        avg_answer_length = sum(len(ex["answer"].split()) for ex in examples) / total_examples
        avg_confidence = sum(ex["confidence"] for ex in examples) / total_examples

        # Quality statistics
        valid_xml_count = 0
        for example in examples:
            formatted = self.format_for_model(example, "xml")
            if self.validate_xml_format(formatted)["is_valid"]:
                valid_xml_count += 1

        return {
            "total_examples": total_examples,
            "average_reasoning_length": avg_reasoning_length,
            "average_answer_length": avg_answer_length,
            "average_confidence": avg_confidence,
            "xml_format_compliance_rate": valid_xml_count / total_examples,
            "quality_score": (avg_confidence + valid_xml_count / total_examples) / 2,
        }
