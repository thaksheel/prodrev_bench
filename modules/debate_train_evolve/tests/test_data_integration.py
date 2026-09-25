"""
GPU integration test: test data loading, preprocessing, and processing.

Uses CUDA_VISIBLE_DEVICES=0 (shared with debate tests, but these tests
are lighter and can coexist).
Marked with @pytest.mark.gpu so it is skipped when no GPU is available.

Tests cover:
- Loading GSM8K dataset from HuggingFace
- Loading ARC dataset from HuggingFace
- Dataset preprocessing (field extraction, answer cleaning)
- DataProcessor XML validation
- DataProcessor format_for_model
- DataProcessor processing statistics
- DatasetManager info and listing
"""

import os

import pytest

# Pin to GPU 0 before any CUDA initialization
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dte.data.dataset_manager import DatasetManager
from dte.data.generator import TrainingExample
from dte.data.processor import DataProcessor


@pytest.mark.gpu
class TestGSM8KDataLoading:
    """Test loading and preprocessing the GSM8K dataset."""

    @pytest.fixture(scope="class")
    def dataset_manager(self):
        return DatasetManager()

    def test_load_gsm8k_train(self, dataset_manager):
        """Load a small slice of GSM8K train split from HuggingFace."""
        dataset = dataset_manager.load_dataset_by_name("gsm8k", split="train", max_samples=5)
        assert len(dataset) == 5
        # GSM8K should have 'question' and 'answer' fields
        assert "question" in dataset.column_names
        assert "answer" in dataset.column_names

    def test_load_gsm8k_test(self, dataset_manager):
        """Load a small slice of GSM8K test split."""
        dataset = dataset_manager.load_dataset_by_name("gsm8k", split="test", max_samples=3)
        assert len(dataset) == 3

    def test_preprocess_gsm8k(self, dataset_manager):
        """Preprocessing GSM8K should extract question and clean answer."""
        dataset = dataset_manager.load_dataset_by_name("gsm8k", split="train", max_samples=3)
        processed = dataset_manager.preprocess_dataset(dataset, "gsm8k")

        # Check that processed samples have standardized fields
        sample = processed[0]
        assert "query" in sample
        assert "ground_truth" in sample
        assert "formatted_query" in sample
        assert "task_type" in sample
        assert sample["task_type"] == "math"

        # GSM8K answers often contain #### format; after preprocessing
        # the ground_truth should be the cleaned numeric answer
        gt = sample["ground_truth"]
        assert isinstance(gt, str)
        assert len(gt.strip()) > 0

    def test_gsm8k_answer_cleaning(self, dataset_manager):
        """GSM8K answers in #### format should have the number extracted."""
        dataset = dataset_manager.load_dataset_by_name("gsm8k", split="train", max_samples=10)
        processed = dataset_manager.preprocess_dataset(dataset, "gsm8k")

        for sample in processed:
            gt = sample["ground_truth"]
            # After preprocessing, #### should be removed
            assert "####" not in gt, f"Answer still contains ####: {gt}"


@pytest.mark.gpu
class TestGSMPlusDataLoading:
    """Test loading and preprocessing GSM-Plus dataset (math reasoning)."""

    @pytest.fixture(scope="class")
    def dataset_manager(self):
        return DatasetManager()

    def test_load_gsm_plus(self, dataset_manager):
        """Load a small slice of GSM-Plus from HuggingFace."""
        dataset = dataset_manager.load_dataset_by_name("gsm_plus", split="testmini", max_samples=5)
        assert len(dataset) == 5

    def test_preprocess_gsm_plus(self, dataset_manager):
        """Preprocessing GSM-Plus should extract question and answer."""
        dataset = dataset_manager.load_dataset_by_name("gsm_plus", split="testmini", max_samples=3)
        processed = dataset_manager.preprocess_dataset(dataset, "gsm_plus")

        sample = processed[0]
        assert "query" in sample
        assert "ground_truth" in sample
        assert "formatted_query" in sample
        assert sample["task_type"] == "math"

    def test_gsm_plus_has_answer(self, dataset_manager):
        """GSM-Plus ground truth should be non-empty."""
        dataset = dataset_manager.load_dataset_by_name("gsm_plus", split="testmini", max_samples=5)
        processed = dataset_manager.preprocess_dataset(dataset, "gsm_plus")

        for sample in processed:
            gt = sample["ground_truth"].strip()
            assert len(gt) > 0, "GSM-Plus answer should not be empty"


@pytest.mark.gpu
class TestDatasetManagerUtilities:
    """Test DatasetManager utility methods."""

    @pytest.fixture(scope="class")
    def dataset_manager(self):
        return DatasetManager()

    def test_list_supported_datasets(self, dataset_manager):
        """Should list all 7 supported datasets."""
        supported = dataset_manager.list_supported_datasets()
        assert isinstance(supported, list)
        assert len(supported) >= 5  # At least the original 5
        assert "gsm8k" in supported
        assert "arc_challenge" in supported

    def test_get_dataset_info(self, dataset_manager):
        """Should return config info for a supported dataset."""
        info = dataset_manager.get_dataset_info("gsm8k")
        assert info["task_type"] == "math"
        assert info["question_field"] == "question"
        assert info["answer_field"] == "answer"
        assert "description" in info

    def test_unsupported_dataset_raises(self, dataset_manager):
        """Loading an unsupported dataset should raise ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            dataset_manager.load_dataset_by_name("nonexistent_dataset")

    def test_cache_works(self, dataset_manager):
        """Loading the same dataset twice should use cache."""
        ds1 = dataset_manager.load_dataset_by_name("gsm8k", split="train", max_samples=3)
        ds2 = dataset_manager.load_dataset_by_name("gsm8k", split="train", max_samples=3)
        # Same object from cache
        assert ds1 is ds2

    def test_cache_info(self, dataset_manager):
        """Cache info should reflect loaded datasets."""
        # Make sure at least one dataset is loaded
        dataset_manager.load_dataset_by_name("gsm8k", split="train", max_samples=2)
        cache_info = dataset_manager.get_cache_info()
        assert cache_info["cache_size"] >= 1


@pytest.mark.gpu
class TestDataProcessor:
    """Test DataProcessor XML validation and formatting."""

    @pytest.fixture
    def processor(self):
        return DataProcessor()

    def test_process_training_examples(self, processor):
        """Processing valid examples should produce cleaned dictionaries."""
        examples = [
            TrainingExample(
                query="What is 2+3?",
                answer="5",
                reasoning="2+3=5",
                confidence=0.9,
                source_dataset="test",
                debate_rounds=1,
                consensus_reached=True,
                metadata={},
            ),
            TrainingExample(
                query="What is 3+4?",
                answer="7",
                reasoning="3+4=7",
                confidence=0.85,
                source_dataset="test",
                debate_rounds=1,
                consensus_reached=True,
                metadata={},
            ),
        ]

        processed = processor.process_training_examples(examples)
        assert len(processed) == 2
        for p in processed:
            assert "query" in p
            assert "answer" in p
            assert "reasoning" in p
            assert "confidence" in p
            assert 0.0 <= p["confidence"] <= 1.0

    def test_process_skips_incomplete(self, processor):
        """Examples with empty fields should be filtered out."""
        examples = [
            TrainingExample(
                query="What is 2+3?",
                answer="",  # empty answer
                reasoning="step",
                confidence=0.9,
                source_dataset="test",
                debate_rounds=1,
                consensus_reached=True,
                metadata={},
            ),
        ]
        processed = processor.process_training_examples(examples)
        assert len(processed) == 0

    def test_format_for_model_xml(self, processor):
        """XML formatting should produce valid XML structure."""
        example = {"query": "Q", "answer": "42", "reasoning": "step 1"}
        formatted = processor.format_for_model(example, format_type="xml")

        assert "<reasoning>" in formatted
        assert "</reasoning>" in formatted
        assert "<answer>" in formatted
        assert "</answer>" in formatted
        assert "42" in formatted

    def test_format_for_model_plain(self, processor):
        """Plain formatting should include query, answer, reasoning."""
        example = {"query": "Q", "answer": "42", "reasoning": "step"}
        formatted = processor.format_for_model(example, format_type="plain")
        assert "42" in formatted
        assert "step" in formatted

    def test_format_for_model_invalid_type(self, processor):
        """Unsupported format type should raise ValueError."""
        example = {"query": "Q", "answer": "42", "reasoning": "step"}
        with pytest.raises(ValueError, match="Unsupported format type"):
            processor.format_for_model(example, format_type="invalid")

    def test_validate_xml_format_valid(self, processor):
        """Valid XML should pass validation."""
        text = "<reasoning>\nstep 1\n</reasoning>\n<answer>\n42\n</answer>"
        result = processor.validate_xml_format(text)
        assert result["is_valid"] is True
        assert result["has_reasoning_tags"] is True
        assert result["has_answer_tags"] is True
        assert result["reasoning_content"] == "step 1"
        assert result["answer_content"] == "42"

    def test_validate_xml_format_invalid(self, processor):
        """Text without XML tags should fail validation."""
        text = "The answer is 42."
        result = processor.validate_xml_format(text)
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_xml_partial_tags(self, processor):
        """Text with only reasoning tags should partially validate."""
        text = "<reasoning>step</reasoning>"
        result = processor.validate_xml_format(text)
        assert result["has_reasoning_tags"] is True
        assert result["has_answer_tags"] is False
        assert result["is_valid"] is False

    def test_processing_statistics(self, processor):
        """Statistics should be computed over processed examples."""
        examples = [
            TrainingExample(
                query="What is 2+3?",
                answer="5",
                reasoning="Two plus three equals five",
                confidence=0.9,
                source_dataset="test",
                debate_rounds=1,
                consensus_reached=True,
                metadata={},
            ),
            TrainingExample(
                query="What is 3+4?",
                answer="7",
                reasoning="Three plus four equals seven",
                confidence=0.8,
                source_dataset="test",
                debate_rounds=1,
                consensus_reached=True,
                metadata={},
            ),
        ]

        processed = processor.process_training_examples(examples)
        stats = processor.get_processing_statistics(processed)

        assert stats["total_examples"] == 2
        assert stats["average_reasoning_length"] > 0
        assert stats["average_confidence"] == pytest.approx(0.85, abs=0.01)
        assert 0.0 <= stats["xml_format_compliance_rate"] <= 1.0
        assert 0.0 <= stats["quality_score"] <= 1.0
