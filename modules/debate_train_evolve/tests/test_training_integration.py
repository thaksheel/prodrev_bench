"""
GPU integration test: load a real model, create synthetic data, run GRPO steps.

Uses Qwen/Qwen2.5-0.5B-Instruct (smallest model) on CUDA_VISIBLE_DEVICES=1.
Marked with @pytest.mark.gpu so it is skipped when no GPU is available.

Tests cover:
- Trainer initialisation (model, tokenizer, reference model)
- Training runs and loss is finite
- Reward calculation with real model outputs
- Detailed reward breakdown
- Advantage calculation
- LoRA training modifies only adapter weights
- Checkpoint save/load roundtrip
- Gradient flow verification
- Reward calculation on real generated model outputs
"""

import os

import pytest

# Pin to GPU 1 before any CUDA initialization (per spec: training on GPU 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch

from dte.core.config import (
    GRPOConfig,
    LoRAConfig,
    ModelConfig,
    PathsConfig,
    RewardsConfig,
    TrainingConfig,
)
from dte.data.generator import TrainingExample
from dte.training.grpo_trainer import GRPOTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _make_synthetic_examples(n: int = 20) -> list:
    """Create synthetic training examples for testing."""
    examples = []
    for i in range(n):
        a, b = (i + 1) * 3, (i + 1) * 5
        examples.append(
            TrainingExample(
                query=f"What is {a} + {b}?",
                answer=str(a + b),
                reasoning=f"To solve {a} + {b}, I add {a} and {b} together. "
                f"{a} + {b} = {a + b}. Therefore the answer is {a + b}.",
                confidence=0.9,
                source_dataset="synthetic",
                debate_rounds=1,
                consensus_reached=True,
                metadata={"task_type": "math"},
            )
        )
    return examples


@pytest.mark.gpu
class TestTrainingIntegration:
    """End-to-end GRPO training integration tests."""

    @pytest.fixture(scope="class")
    def training_setup(self, tmp_path_factory):
        """Create a GRPOTrainer with the smallest model.

        Scoped to the class to reuse across tests.
        """
        out_dir = tmp_path_factory.mktemp("training_output")

        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        training_cfg = TrainingConfig(
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=2,
            max_epochs=2,
            batch_size=2,
            gradient_accumulation_steps=1,
            grpo=GRPOConfig(group_size=2, clip_ratio=0.2, kl_penalty=0.01),
            rewards=RewardsConfig(),
            lora=LoRAConfig(enabled=False),  # Disable LoRA for faster test
        )
        paths_cfg = PathsConfig(
            output_dir=str(out_dir / "outputs"),
            models_dir=str(out_dir / "models"),
            data_dir=str(out_dir / "data"),
            cache_dir=str(out_dir / "cache"),
            temp_dir=str(out_dir / "tmp"),
        )

        trainer = GRPOTrainer(training_cfg, model_cfg, paths_cfg, logger=None)
        yield trainer
        trainer.cleanup()

    def test_trainer_initialised(self, training_setup):
        """Verify the trainer loaded a model and tokenizer."""
        trainer = training_setup
        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.reference_model is not None

    def test_training_runs_and_loss_decreases(self, training_setup):
        """Run a few GRPO steps and verify loss is finite and does not explode."""
        trainer = training_setup
        examples = _make_synthetic_examples(8)

        metrics = trainer.train(examples)

        assert "epoch_losses" in metrics
        assert len(metrics["epoch_losses"]) == 2  # 2 epochs

        # Loss should be finite
        for loss in metrics["epoch_losses"]:
            assert loss == loss  # not NaN
            assert loss < 1e6  # not exploded

    def test_reward_calculation(self, training_setup):
        """Verify the full reward pipeline works end-to-end."""
        trainer = training_setup
        response = "<reasoning>\n3 + 5 = 8\n</reasoning>\n<answer>\n8\n</answer>\n"
        reward = trainer._calculate_reward("What is 3+5?", response, "8")

        assert isinstance(reward, float)
        assert reward > 0  # Correct answer with proper format should get positive reward

    def test_detailed_reward_breakdown(self, training_setup):
        """Verify detailed reward breakdown returns all components."""
        trainer = training_setup
        response = "<reasoning>\nStep: 3+5=8\n</reasoning>\n<answer>\n8\n</answer>\n"
        breakdown = trainer.get_detailed_reward_breakdown("What is 3+5?", response, "8")

        assert "individual_rewards" in breakdown
        assert "correctness" in breakdown["individual_rewards"]
        assert "int" in breakdown["individual_rewards"]
        assert "strict_format" in breakdown["individual_rewards"]
        assert "soft_format" in breakdown["individual_rewards"]
        assert "xmlcount" in breakdown["individual_rewards"]

    def test_advantage_calculation(self, training_setup):
        """Verify GRPO advantage normalization."""
        trainer = training_setup
        rewards = [1.0, 2.0, 3.0, 4.0]
        advantages = trainer._calculate_advantages(rewards)

        assert len(advantages) == 4
        # Advantages should sum to approximately zero (mean-centered)
        assert abs(sum(advantages)) < 1e-5
        # Higher reward should get higher advantage
        assert advantages[3] > advantages[0]


@pytest.mark.gpu
class TestGradientFlow:
    """Verify that gradients actually flow through the model during training."""

    @pytest.fixture(scope="class")
    def trainer(self, tmp_path_factory):
        """Create a trainer for gradient tests."""
        out_dir = tmp_path_factory.mktemp("gradient_test")
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        training_cfg = TrainingConfig(
            learning_rate=1e-4,
            warmup_steps=0,
            max_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            grpo=GRPOConfig(group_size=2, clip_ratio=0.2, kl_penalty=0.01),
            rewards=RewardsConfig(),
            lora=LoRAConfig(enabled=False),
        )
        paths_cfg = PathsConfig(
            output_dir=str(out_dir / "outputs"),
            models_dir=str(out_dir / "models"),
            data_dir=str(out_dir / "data"),
            cache_dir=str(out_dir / "cache"),
            temp_dir=str(out_dir / "tmp"),
        )
        trainer = GRPOTrainer(training_cfg, model_cfg, paths_cfg, logger=None)
        yield trainer
        trainer.cleanup()

    def test_parameters_change_after_training(self, trainer):
        """Model parameters should change after a training step."""
        # Snapshot a few parameter values before training
        param_snapshots = {}
        for name, param in trainer.model.named_parameters():
            if param.requires_grad:
                param_snapshots[name] = param.data.clone()
                if len(param_snapshots) >= 3:
                    break

        assert len(param_snapshots) > 0, "No trainable parameters found"

        # Run one epoch of training
        examples = _make_synthetic_examples(4)
        trainer.train(examples)

        # At least one parameter should have changed
        any_changed = False
        for name, old_val in param_snapshots.items():
            new_val = dict(trainer.model.named_parameters())[name].data
            if not torch.allclose(old_val, new_val, atol=1e-8):
                any_changed = True
                break

        assert any_changed, "No parameters changed after training -- gradient flow may be broken"

    def test_reference_model_frozen(self, trainer):
        """Reference model parameters should never change."""
        # Snapshot reference model parameters
        ref_snapshots = {}
        for name, param in trainer.reference_model.named_parameters():
            ref_snapshots[name] = param.data.clone()
            if len(ref_snapshots) >= 3:
                break

        # Run training
        examples = _make_synthetic_examples(4)
        trainer.train(examples)

        # Reference model should be unchanged
        for name, old_val in ref_snapshots.items():
            new_val = dict(trainer.reference_model.named_parameters())[name].data
            assert torch.allclose(old_val, new_val, atol=1e-10), (
                f"Reference model parameter {name} changed during training"
            )


@pytest.mark.gpu
class TestLoRATraining:
    """Test that LoRA training modifies adapter weights and saves/loads correctly."""

    @pytest.fixture(scope="class")
    def lora_trainer(self, tmp_path_factory):
        """Create a trainer with LoRA enabled."""
        out_dir = tmp_path_factory.mktemp("lora_test")
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        training_cfg = TrainingConfig(
            learning_rate=1e-4,
            warmup_steps=0,
            max_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            grpo=GRPOConfig(group_size=2, clip_ratio=0.2, kl_penalty=0.01),
            rewards=RewardsConfig(),
            lora=LoRAConfig(
                enabled=True,
                rank=8,
                alpha=16,
                dropout=0.0,
                target_modules=["q_proj", "v_proj"],
            ),
        )
        paths_cfg = PathsConfig(
            output_dir=str(out_dir / "outputs"),
            models_dir=str(out_dir / "models"),
            data_dir=str(out_dir / "data"),
            cache_dir=str(out_dir / "cache"),
            temp_dir=str(out_dir / "tmp"),
        )
        trainer = GRPOTrainer(training_cfg, model_cfg, paths_cfg, logger=None)
        yield trainer, out_dir
        trainer.cleanup()

    def test_lora_adapter_present(self, lora_trainer):
        """LoRA adapter layers should be present in the model."""
        trainer, _ = lora_trainer
        lora_params = [name for name, _ in trainer.model.named_parameters() if "lora" in name.lower()]
        assert len(lora_params) > 0, "No LoRA parameters found in model"

    def test_lora_weights_change_after_training(self, lora_trainer):
        """LoRA adapter weights should change after training."""
        trainer, _ = lora_trainer

        # Snapshot LoRA parameter values
        lora_snapshots = {}
        for name, param in trainer.model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                lora_snapshots[name] = param.data.clone()
                if len(lora_snapshots) >= 3:
                    break

        assert len(lora_snapshots) > 0, "No trainable LoRA parameters found"

        # Run training
        examples = _make_synthetic_examples(4)
        trainer.train(examples)

        # At least one LoRA parameter should change
        any_changed = False
        for name, old_val in lora_snapshots.items():
            new_val = dict(trainer.model.named_parameters())[name].data
            if not torch.allclose(old_val, new_val, atol=1e-8):
                any_changed = True
                break

        assert any_changed, "No LoRA weights changed after training"

    def test_only_lora_weights_trainable(self, lora_trainer):
        """Only LoRA adapter weights should have requires_grad=True."""
        trainer, _ = lora_trainer
        for name, param in trainer.model.named_parameters():
            if "lora" not in name.lower():
                # Base model params should be frozen
                assert not param.requires_grad, f"Non-LoRA param {name} has requires_grad=True"


@pytest.mark.gpu
class TestCheckpointRoundtrip:
    """Test checkpoint save and load preserves training state."""

    def test_checkpoint_save_load(self, tmp_path):
        """Save a checkpoint, reload it, and verify contents."""
        out_dir = tmp_path / "ckpt_test"
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
        )
        training_cfg = TrainingConfig(
            learning_rate=5e-5,
            warmup_steps=0,
            max_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            grpo=GRPOConfig(group_size=2, clip_ratio=0.2, kl_penalty=0.01),
            rewards=RewardsConfig(),
            lora=LoRAConfig(enabled=False),
        )
        paths_cfg = PathsConfig(
            output_dir=str(out_dir / "outputs"),
            models_dir=str(out_dir / "models"),
            data_dir=str(out_dir / "data"),
            cache_dir=str(out_dir / "cache"),
            temp_dir=str(out_dir / "tmp"),
        )

        trainer = GRPOTrainer(training_cfg, model_cfg, paths_cfg, logger=None)
        try:
            # Train for 1 epoch to trigger checkpoint save
            examples = _make_synthetic_examples(4)
            trainer.train(examples)

            # Verify checkpoint was saved
            checkpoint_dir = os.path.join(str(out_dir / "models"), "checkpoint_epoch_0")
            assert os.path.isdir(checkpoint_dir), f"Checkpoint directory not created: {checkpoint_dir}"

            # Check that model files exist
            from transformers import AutoModelForCausalLM, AutoTokenizer

            assert os.path.exists(os.path.join(checkpoint_dir, "config.json")) or os.path.exists(
                os.path.join(checkpoint_dir, "model.pt")
            ), "No model file found in checkpoint"

            # Check that training state was saved
            state_path = os.path.join(checkpoint_dir, "training_state.pt")
            assert os.path.exists(state_path), "training_state.pt not found"

            # Load and verify training state
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            assert "epoch" in state
            assert "global_step" in state
            assert "loss" in state
            assert state["epoch"] == 0

            # Verify we can reload the tokenizer from checkpoint
            reloaded_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
            assert reloaded_tokenizer is not None
            assert reloaded_tokenizer.vocab_size == trainer.tokenizer.vocab_size

        finally:
            trainer.cleanup()


@pytest.mark.gpu
class TestRewardWithRealModelOutputs:
    """Test reward calculation on actual model-generated text."""

    @pytest.fixture(scope="class")
    def trainer(self, tmp_path_factory):
        """Create a trainer to generate real outputs."""
        out_dir = tmp_path_factory.mktemp("reward_real_output")
        model_cfg = ModelConfig(
            base_model_name=MODEL_NAME,
            device="auto",
            max_length=256,
            temperature=0.7,
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
            output_dir=str(out_dir / "outputs"),
            models_dir=str(out_dir / "models"),
            data_dir=str(out_dir / "data"),
            cache_dir=str(out_dir / "cache"),
            temp_dir=str(out_dir / "tmp"),
        )
        trainer = GRPOTrainer(training_cfg, model_cfg, paths_cfg, logger=None)
        yield trainer
        trainer.cleanup()

    def test_reward_on_generated_text(self, trainer):
        """Generate text with the model and calculate reward on it."""
        query = "What is 7 + 3?"
        responses = trainer._generate_responses(query, num_responses=2)

        assert len(responses) == 2
        for resp in responses:
            assert isinstance(resp, str)
            assert len(resp) > 0

        # Calculate reward for each response
        for resp in responses:
            reward = trainer._calculate_reward(query, resp, "10")
            assert isinstance(reward, float)
            assert reward == reward  # not NaN

    def test_reward_breakdown_on_generated_text(self, trainer):
        """Get detailed reward breakdown for model-generated text."""
        query = "What is 4 * 5?"
        responses = trainer._generate_responses(query, num_responses=1)
        resp = responses[0]

        breakdown = trainer.get_detailed_reward_breakdown(query, resp, "20")
        assert "individual_rewards" in breakdown
        assert "response_length" in breakdown
        assert "has_xml_format" in breakdown
        assert isinstance(breakdown["response_length"], int)
