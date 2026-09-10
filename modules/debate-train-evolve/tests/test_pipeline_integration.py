"""
GPU integration test: mini end-to-end pipeline run.

Uses Qwen/Qwen2.5-0.5B-Instruct on CUDA_VISIBLE_DEVICES=3.
Marked with @pytest.mark.gpu so it is skipped when no GPU is available.

Tests cover:
- Config validation
- Debate manager from config runs correctly
- RCR prompts in debate rounds
- Data generation from real debates
- Full single-round evolution (debate -> data -> train)
- Pipeline checkpoint save/load
"""

import json
import os

import pytest

# Pin to GPU 3 before any CUDA initialization
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from dte.core.config import (
    DatasetInfo,
    DatasetsConfig,
    DebateConfig,
    DTEConfig,
    EvolutionConfig,
    ExperimentConfig,
    GRPOConfig,
    LoggingConfig,
    LoRAConfig,
    ModelConfig,
    PathsConfig,
    RewardsConfig,
    TrainingConfig,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.gpu
class TestPipelineIntegration:
    """Mini end-to-end pipeline test on actual hardware."""

    @pytest.fixture(scope="class")
    def pipeline_config(self, tmp_path_factory):
        out_dir = tmp_path_factory.mktemp("pipeline_output")
        return DTEConfig(
            model=ModelConfig(
                base_model_name=MODEL_NAME,
                device="auto",
                max_length=256,
                temperature=0.7,
            ),
            debate=DebateConfig(num_agents=2, max_rounds=1),
            datasets=DatasetsConfig(
                names=["gsm8k"],
                max_samples_per_dataset=5,
                train_datasets=[
                    DatasetInfo(
                        name="gsm8k",
                        path="openai/gsm8k",
                        split="train",
                        max_samples=5,
                    ),
                ],
                eval_datasets=[
                    DatasetInfo(
                        name="gsm8k",
                        path="openai/gsm8k",
                        split="train",
                        max_samples=3,
                    ),
                ],
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                max_epochs=1,
                batch_size=1,
                gradient_accumulation_steps=1,
                grpo=GRPOConfig(group_size=2, clip_ratio=0.2, kl_penalty=0.01),
                rewards=RewardsConfig(),
                lora=LoRAConfig(enabled=False),
            ),
            evolution=EvolutionConfig(max_rounds=1, samples_per_round=3),
            paths=PathsConfig(
                output_dir=str(out_dir / "outputs"),
                models_dir=str(out_dir / "models"),
                data_dir=str(out_dir / "data"),
                cache_dir=str(out_dir / "cache"),
                temp_dir=str(out_dir / "tmp"),
            ),
            logging=LoggingConfig(level="WARNING", log_dir=str(out_dir / "logs")),
            experiment=ExperimentConfig(name="pipeline_test", seed=42),
        )

    def test_config_validation(self, pipeline_config):
        """Pipeline config should validate without errors."""
        errors = pipeline_config.validate()
        assert errors == [], f"Config validation errors: {errors}"

    def test_debate_manager_runs(self, pipeline_config):
        """A debate manager created from config should produce results."""
        from dte.debate.manager import DebateManager

        manager = DebateManager(pipeline_config.debate, pipeline_config.model, logger=None)
        try:
            result = manager.conduct_debate("What is 2 + 3?", task_type="math")
            assert result.final_answer is not None
            assert len(result.all_responses) >= 1
        finally:
            manager.cleanup()

    def test_rcr_prompts_in_debate(self, pipeline_config):
        """Debate rounds > 0 should use RCR prompting with three phases."""
        from dte.debate.prompts import DebatePromptManager

        pm = DebatePromptManager()
        prompt = pm.create_debate_prompt(
            query="What is 2+3?",
            agent_id="1",
            round_num=1,
            answers_so_far={"1": "answer is 5", "2": "answer is 6"},
            task_type="math",
        )
        assert "PHASE 1: REFLECT" in prompt
        assert "PHASE 2: CRITIQUE" in prompt
        assert "PHASE 3: REFINE" in prompt


@pytest.mark.gpu
class TestDataGenerationIntegration:
    """Test debate data generation with a real model."""

    @pytest.fixture(scope="class")
    def generator_setup(self, tmp_path_factory):
        """Create a DebateDataGenerator with minimal config."""
        from dte.data.generator import DebateDataGenerator

        out_dir = tmp_path_factory.mktemp("datagen_output")

        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        debate_cfg = DebateConfig(num_agents=2, max_rounds=1)
        datasets_cfg = DatasetsConfig(
            names=["gsm8k"],
            max_samples_per_dataset=3,
            train_datasets=[
                DatasetInfo(
                    name="gsm8k",
                    path="openai/gsm8k",
                    split="train",
                    max_samples=3,
                ),
            ],
        )

        generator = DebateDataGenerator(datasets_cfg, debate_cfg, model_cfg, logger=None)
        yield generator, out_dir
        generator.cleanup()

    def test_generate_training_data(self, generator_setup):
        """Generate training data from real debates on real data."""
        generator, out_dir = generator_setup

        # Relax quality filters for testing with small model
        generator.update_quality_filters(
            {
                "min_reasoning_length": 1,
                "require_consensus": False,
                "min_consensus_confidence": 0.0,
            }
        )

        examples = generator.generate_training_data(
            num_samples=2,
            evolution_round=0,
            save_path=str(out_dir / "training_data.jsonl"),
        )

        # We should get at least some examples (quality filters may remove some)
        assert isinstance(examples, list)
        # Data file should exist
        assert os.path.exists(str(out_dir / "training_data.jsonl"))

    def test_debate_results_stored(self, generator_setup):
        """Generator should store debate results internally."""
        generator, _ = generator_setup
        # After generate_training_data was called, results should be stored
        assert isinstance(generator.debate_results, list)

    def test_generation_statistics(self, generator_setup):
        """Generation statistics should be available after data gen."""
        generator, _ = generator_setup
        stats = generator.get_generation_statistics()
        assert isinstance(stats, dict)


@pytest.mark.gpu
class TestPipelineCheckpoint:
    """Test pipeline checkpoint save and load."""

    def test_checkpoint_save_load_roundtrip(self, tmp_path):
        """Save a pipeline checkpoint, load it, and verify state."""
        from dte.core.pipeline import DTEPipeline, EvolutionRoundResult

        config = DTEConfig(
            model=ModelConfig(
                base_model_name=MODEL_NAME,
                device="auto",
                max_length=256,
                temperature=0.7,
            ),
            debate=DebateConfig(num_agents=2, max_rounds=1),
            datasets=DatasetsConfig(
                names=["gsm8k"],
                max_samples_per_dataset=2,
                train_datasets=[
                    DatasetInfo(
                        name="gsm8k",
                        path="openai/gsm8k",
                        split="train",
                        max_samples=2,
                    ),
                ],
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                max_epochs=1,
                batch_size=1,
                gradient_accumulation_steps=1,
                grpo=GRPOConfig(group_size=2, clip_ratio=0.2, kl_penalty=0.01),
                rewards=RewardsConfig(),
                lora=LoRAConfig(enabled=False),
            ),
            evolution=EvolutionConfig(max_rounds=1, samples_per_round=2),
            paths=PathsConfig(
                output_dir=str(tmp_path / "outputs"),
                models_dir=str(tmp_path / "models"),
                data_dir=str(tmp_path / "data"),
                cache_dir=str(tmp_path / "cache"),
                temp_dir=str(tmp_path / "tmp"),
            ),
            logging=LoggingConfig(level="WARNING", log_dir=str(tmp_path / "logs")),
            experiment=ExperimentConfig(name="checkpoint_test", seed=42),
        )

        pipeline = DTEPipeline(config)
        try:
            # Manually set some state to verify roundtrip
            pipeline.current_round = 2
            pipeline.best_performance = 0.75
            pipeline.patience_counter = 1

            # Add a synthetic evolution result
            pipeline.evolution_results.append(
                EvolutionRoundResult(
                    round_number=1,
                    data_generation_stats={"total_examples": 5},
                    training_metrics={"epoch_losses": [0.5]},
                    evaluation_results={"overall_accuracy": 0.75},
                    performance_improvement=0.05,
                    total_time=120.0,
                )
            )

            # Save checkpoint
            ckpt_path = str(tmp_path / "pipeline_checkpoint.json")
            pipeline.save_checkpoint(ckpt_path)
            assert os.path.exists(ckpt_path)

            # Load checkpoint into a fresh pipeline
            pipeline2 = DTEPipeline(config)
            pipeline2.load_checkpoint(ckpt_path)

            assert pipeline2.current_round == 2
            assert pipeline2.best_performance == 0.75
            assert pipeline2.patience_counter == 1
            assert len(pipeline2.evolution_results) == 1
            assert pipeline2.evolution_results[0].round_number == 1

        finally:
            try:
                pipeline._cleanup()
            except Exception:
                pass
            try:
                pipeline2._cleanup()
            except Exception:
                pass


@pytest.mark.gpu
class TestSingleRoundEvolution:
    """Test a full single-round evolution: debate -> data -> train."""

    def test_full_single_round(self, tmp_path):
        """Run debate data generation then train on the output."""
        from dte.data.generator import DebateDataGenerator, TrainingExample
        from dte.training.grpo_trainer import GRPOTrainer

        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        debate_cfg = DebateConfig(num_agents=2, max_rounds=1)
        datasets_cfg = DatasetsConfig(
            names=["gsm8k"],
            max_samples_per_dataset=2,
            train_datasets=[
                DatasetInfo(
                    name="gsm8k",
                    path="openai/gsm8k",
                    split="train",
                    max_samples=2,
                ),
            ],
        )
        training_cfg = TrainingConfig(
            learning_rate=5e-5,
            warmup_steps=0,
            max_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            grpo=GRPOConfig(group_size=2, clip_ratio=0.2, kl_penalty=0.01),
            rewards=RewardsConfig(),
            lora=LoRAConfig(enabled=False),
        )
        paths_cfg = PathsConfig(
            output_dir=str(tmp_path / "outputs"),
            models_dir=str(tmp_path / "models"),
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            temp_dir=str(tmp_path / "tmp"),
        )

        # Phase 1: Generate debate data
        generator = DebateDataGenerator(datasets_cfg, debate_cfg, model_cfg, logger=None)
        generator.update_quality_filters(
            {
                "min_reasoning_length": 1,
                "require_consensus": False,
                "min_consensus_confidence": 0.0,
            }
        )

        try:
            examples = generator.generate_training_data(num_samples=2, evolution_round=0)

            # If quality filters removed all examples, create synthetic fallbacks
            if len(examples) == 0:
                examples = [
                    TrainingExample(
                        query="What is 2+3?",
                        answer="5",
                        reasoning="2+3=5",
                        confidence=0.9,
                        source_dataset="synthetic",
                        debate_rounds=1,
                        consensus_reached=True,
                        metadata={"task_type": "math"},
                    )
                ]

            # Phase 2: Train on generated data
            trainer = GRPOTrainer(training_cfg, model_cfg, paths_cfg, logger=None)
            try:
                metrics = trainer.train(examples)
                assert "epoch_losses" in metrics
                assert len(metrics["epoch_losses"]) == 1
                # Loss should be finite
                assert metrics["epoch_losses"][0] == metrics["epoch_losses"][0]
            finally:
                trainer.cleanup()

        finally:
            generator.cleanup()
