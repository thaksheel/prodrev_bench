"""
GRPO (Group Relative Policy Optimization) Training Implementation.

This module implements the GRPO training algorithm as described in the DTE paper
(EMNLP 2025). GRPO eliminates the need for a separate value function by
estimating advantages through group-wise comparisons of multiple sampled
responses for each query.

Key implementation details matching the paper:
- Per-token log-probability ratios (not averaged over vocab dimension)
- 8-bit AdamW optimizer with betas (0.9, 0.99) when bitsandbytes is available
- Clipped surrogate objective with KL regularization against a frozen reference
- All 5 DTE reward functions for training signal
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    get_linear_schedule_with_warmup,
)

# Conditional import for 8-bit AdamW
try:
    import bitsandbytes as bnb

    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False

from torch.optim import AdamW

# Conditional import of PEFT
try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from ..core.logger import DTELogger
from ..data.generator import TrainingExample
from .reward_model import DTERewardModel


@dataclass
class GRPOBatch:
    """A batch prepared for GRPO training.

    Attributes:
        queries: List of query strings.
        responses: Nested list of sampled responses ``[query_idx][response_idx]``.
        rewards: Nested list of scalar rewards ``[query_idx][response_idx]``.
        advantages: Group-relative advantages ``[query_idx][response_idx]``.
    """

    queries: List[str]
    responses: List[List[str]]
    rewards: List[List[float]]
    advantages: List[List[float]]


class TrainingDataset(Dataset):
    """Dataset wrapper for GRPO training.

    Wraps :class:`TrainingExample` instances into a format suitable for the
    GRPO training loop, formatting responses in the XML structure expected
    by DTE reward functions.

    Args:
        examples: Training examples from debate data generation.
        tokenizer: HuggingFace tokenizer for the model.
        max_length: Maximum sequence length for tokenization.
    """

    def __init__(
        self,
        examples: List[TrainingExample],
        tokenizer,
        max_length: int = 2048,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        query = example.query
        response = f"<reasoning>\n{example.reasoning}\n</reasoning>\n<answer>\n{example.answer}\n</answer>\n"
        answer = example.answer

        return {
            "query": query,
            "response": response,
            "answer": answer,
            "reward": example.confidence,
        }


class GRPOTrainer:
    """GRPO (Group Relative Policy Optimization) Trainer.

    Implements the GRPO algorithm from the DTE paper. Key design choices:

    * **Per-token log-ratios**: The policy ratio is computed per token by
      gathering the log-probability assigned to the *actual* next token,
      then taking the difference between the current policy and the frozen
      reference. This avoids the incorrect practice of averaging over the
      entire vocabulary dimension.
    * **8-bit AdamW**: When ``bitsandbytes`` is installed the optimizer uses
      8-bit AdamW with betas ``(0.9, 0.99)`` as specified in the paper.
    * **Group-relative advantages**: For each query, multiple responses are
      sampled and their rewards are z-score normalised within the group.
    * **Clipped surrogate + KL penalty**: Standard PPO-style clipping
      combined with a KL divergence penalty against the reference model.

    Args:
        config_training: Training configuration (:class:`TrainingConfig`).
        config_model: Model configuration (:class:`ModelConfig`).
        config_paths: Paths configuration (:class:`PathsConfig`).
        logger: Optional DTE logger for progress tracking.
    """

    def __init__(
        self,
        config_training,
        config_model,
        config_paths,
        logger: Optional[DTELogger] = None,
    ):
        self.config = config_training
        self.model_config = config_model
        self.paths_config = config_paths
        self.logger = logger

        # Training parameters
        self.learning_rate = config_training.learning_rate
        self.weight_decay = config_training.weight_decay
        self.warmup_steps = config_training.warmup_steps
        self.max_epochs = config_training.max_epochs
        self.batch_size = config_training.batch_size
        self.gradient_accumulation_steps = config_training.gradient_accumulation_steps

        # GRPO parameters
        self.group_size = config_training.grpo.group_size
        self.clip_ratio = config_training.grpo.clip_ratio
        self.kl_penalty = config_training.grpo.kl_penalty

        # Device setup
        self.device = self._setup_device()

        # Model components (populated in _load_model)
        self.tokenizer = None
        self.model = None
        self.reference_model = None
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")

        # Reward model
        self.reward_model = DTERewardModel()

        # Load model
        self._load_model()

    def _setup_device(self) -> torch.device:
        """Detect and return the best available computation device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_model(self) -> None:
        """Load model, tokenizer, reference model, and optionally apply LoRA."""
        model_name = self.model_config.base_model_name

        if self.logger:
            self.logger.info(f"Loading model: {model_name}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model kwargs
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            "device_map": "auto" if self.device.type == "cuda" else None,
            "trust_remote_code": True,
        }

        # Policy model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Apply LoRA if enabled and PEFT is available
        if self.config.lora.enabled and PEFT_AVAILABLE:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora.rank,
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                target_modules=self.config.lora.target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)
            if self.logger:
                self.logger.info(f"Applied LoRA: rank={self.config.lora.rank}")
        elif self.config.lora.enabled and not PEFT_AVAILABLE:
            if self.logger:
                self.logger.warning("LoRA requested but PEFT not available. Training without LoRA.")

        # Reference model (frozen)
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        # Move to device if not using device_map
        if model_kwargs.get("device_map") is None:
            self.model = self.model.to(self.device)
            self.reference_model = self.reference_model.to(self.device)

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def train(
        self,
        training_examples: List[TrainingExample],
        validation_examples: Optional[List[TrainingExample]] = None,
    ) -> Dict[str, Any]:
        """Train the model using the GRPO algorithm.

        Args:
            training_examples: Training examples generated from debates.
            validation_examples: Optional held-out examples for validation.

        Returns:
            Dictionary of training metrics including per-epoch losses,
            learning rates, KL divergences, and reward statistics.
        """
        if self.logger:
            with self.logger.component_context("grpo_training"):
                self.logger.info(f"Starting GRPO training with {len(training_examples)} examples")

        # Build dataset and dataloader
        train_dataset = TrainingDataset(training_examples, self.tokenizer, self.model_config.max_length)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Optimizer and scheduler
        self._setup_optimizer_and_scheduler(len(train_dataloader))

        # Run training loop
        training_metrics = self._training_loop(train_dataloader, validation_examples)

        if self.logger:
            self.logger.info("GRPO training completed")
            if "reward_stats" in training_metrics and training_metrics["reward_stats"]:
                final_rewards = training_metrics["reward_stats"][-1]
                self.logger.info(f"Final reward stats: {final_rewards}")

        return training_metrics

    # ------------------------------------------------------------------
    # Optimizer setup
    # ------------------------------------------------------------------

    def _setup_optimizer_and_scheduler(self, num_training_steps_per_epoch: int) -> None:
        """Create optimizer (8-bit AdamW if available) and LR scheduler.

        The DTE paper specifies 8-bit AdamW with betas (0.9, 0.99). When
        ``bitsandbytes`` is installed we use its 8-bit implementation;
        otherwise we fall back to standard AdamW with the same betas.
        """
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Paper: 8-bit AdamW with betas (0.9, 0.99)
        adam_betas = (0.9, 0.99)

        if _BNB_AVAILABLE:
            self.optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=adam_betas,
                eps=1e-8,
            )
            optimizer_label = "8-bit AdamW (bitsandbytes)"
        else:
            self.optimizer = AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=adam_betas,
                eps=1e-8,
            )
            optimizer_label = "AdamW (betas=0.9, 0.99)"

        total_steps = num_training_steps_per_epoch * self.max_epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        if self.logger:
            self.logger.info(f"Optimizer: {optimizer_label}, LR: {self.learning_rate}, Steps: {total_steps}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _training_loop(
        self,
        train_dataloader: DataLoader,
        validation_examples: Optional[List[TrainingExample]],
    ) -> Dict[str, Any]:
        """Main training loop implementing the GRPO algorithm."""
        training_metrics: Dict[str, Any] = {
            "epoch_losses": [],
            "step_losses": [],
            "learning_rates": [],
            "kl_divergences": [],
            "advantages_stats": [],
        }

        self.model.train()

        if self.logger:
            self.logger.start_progress("GRPO Training", total=self.max_epochs)

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_dataloader):
                # Generate multiple responses and compute rewards/advantages
                grpo_batch = self._prepare_grpo_batch(batch)

                # Compute per-token GRPO loss
                loss, metrics = self._compute_grpo_loss(grpo_batch)

                # Backward with gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Track metrics
                step_loss = loss.item() * self.gradient_accumulation_steps
                epoch_loss += step_loss
                num_batches += 1

                training_metrics["step_losses"].append(step_loss)
                training_metrics["learning_rates"].append(self.scheduler.get_last_lr()[0])
                training_metrics["kl_divergences"].append(metrics["avg_kl_div"])
                training_metrics["advantages_stats"].append(metrics["advantages_stats"])

                if "reward_stats" in metrics:
                    if "reward_stats" not in training_metrics:
                        training_metrics["reward_stats"] = []
                    training_metrics["reward_stats"].append(metrics["reward_stats"])

                # Periodic logging
                if self.global_step % 10 == 0 and self.logger:
                    self.logger.log_training_step(
                        step=self.global_step,
                        loss=step_loss,
                        metrics={
                            "lr": self.scheduler.get_last_lr()[0],
                            "kl_div": metrics["avg_kl_div"],
                            "advantages_mean": metrics["advantages_stats"]["mean"],
                            **metrics.get("reward_stats", {}),
                        },
                    )

            # Epoch summary
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            training_metrics["epoch_losses"].append(avg_epoch_loss)

            if self.logger:
                self.logger.update_progress("GRPO Training", advance=1)
                self.logger.info(f"Epoch {epoch + 1}/{self.max_epochs} - Loss: {avg_epoch_loss:.4f}")

            # Checkpoint on improvement
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self._save_checkpoint(epoch, avg_epoch_loss)

        if self.logger:
            self.logger.finish_progress("GRPO Training")

        return training_metrics

    # ------------------------------------------------------------------
    # GRPO batch preparation
    # ------------------------------------------------------------------

    def _prepare_grpo_batch(self, batch: Dict[str, Any]) -> GRPOBatch:
        """Sample multiple responses per query and compute rewards/advantages."""
        queries = batch["query"]

        all_responses: List[List[str]] = []
        all_rewards: List[List[float]] = []
        all_advantages: List[List[float]] = []

        with torch.no_grad():
            for i, query in enumerate(queries):
                responses = self._generate_responses(query, self.group_size)

                ground_truth = None
                if "ground_truth" in batch and i < len(batch["ground_truth"]):
                    ground_truth = batch["ground_truth"][i]
                elif "answer" in batch and i < len(batch["answer"]):
                    ground_truth = batch["answer"][i]

                rewards = [self._calculate_reward(query, response, ground_truth) for response in responses]
                advantages = self._calculate_advantages(rewards)

                all_responses.append(responses)
                all_rewards.append(rewards)
                all_advantages.append(advantages)

        return GRPOBatch(
            queries=queries,
            responses=all_responses,
            rewards=all_rewards,
            advantages=all_advantages,
        )

    def _generate_responses(self, query: str, num_responses: int) -> List[str]:
        """Generate *num_responses* sampled completions for a single query."""
        responses: List[str] = []

        query_tokens = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_length // 2,
            padding=False,
        ).to(self.device)

        generation_config = GenerationConfig(
            max_length=self.model_config.max_length,
            temperature=self.model_config.temperature,
            top_p=self.model_config.top_p,
            top_k=self.model_config.top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        for _ in range(num_responses):
            with torch.no_grad():
                outputs = self.model.generate(
                    **query_tokens,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                )
                gen_tokens = outputs.sequences[0][len(query_tokens["input_ids"][0]) :]
                response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                responses.append(response.strip())

        return responses

    # ------------------------------------------------------------------
    # Reward calculation
    # ------------------------------------------------------------------

    def _calculate_reward(self, query: str, response: str, ground_truth: Optional[str] = None) -> float:
        """Compute the combined DTE reward for a (query, response) pair.

        Uses all 5 DTE reward functions and returns a **weighted sum** (not
        a weighted average) as specified in the paper.

        Args:
            query: Input query/question.
            response: Model-generated response.
            ground_truth: Optional ground-truth answer for correctness reward.

        Returns:
            Combined scalar reward.
        """
        rewards_dict = self.reward_model.calculate_all_rewards(
            query=query,
            responses=[response],
            ground_truth=ground_truth,
        )

        if hasattr(self.config.rewards, "get_dte_weights"):
            reward_weights = self.config.rewards.get_dte_weights()
        else:
            reward_weights = {
                "correctness": 2.0,
                "int": 0.5,
                "strict_format": 0.5,
                "soft_format": 0.5,
                "xmlcount": 0.5,
            }

        combined = self.reward_model.combine_rewards(rewards_dict, reward_weights)
        return combined[0] if combined else 0.0

    def get_detailed_reward_breakdown(
        self, query: str, response: str, ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return a detailed breakdown of all reward components.

        Useful for analysis and debugging of the reward signals.

        Args:
            query: Input query/question.
            response: Model-generated response.
            ground_truth: Optional ground-truth answer.

        Returns:
            Dictionary with individual rewards, statistics, and format info.
        """
        rewards_dict = self.reward_model.calculate_all_rewards(
            query=query,
            responses=[response],
            ground_truth=ground_truth,
        )
        reward_stats = self.reward_model.get_reward_statistics(rewards_dict)

        return {
            "individual_rewards": {k: v[0] for k, v in rewards_dict.items()},
            "reward_statistics": reward_stats,
            "extracted_answer": self.reward_model._extract_xml_answer(response),
            "response_length": len(response.split()),
            "has_xml_format": "<reasoning>" in response and "<answer>" in response,
        }

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def _calculate_advantages(self, rewards: List[float]) -> List[float]:
        """Compute group-relative z-score normalized advantages.

        Args:
            rewards: Scalar rewards for each response in the group.

        Returns:
            Zero-mean, unit-variance advantages.
        """
        if len(rewards) <= 1:
            return [0.0] * len(rewards)

        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = math.sqrt(variance + 1e-8)

        return [(r - mean_reward) / (std_reward + 1e-8) for r in rewards]

    # ------------------------------------------------------------------
    # GRPO loss (per-token log-probability ratios)
    # ------------------------------------------------------------------

    def _compute_grpo_loss(self, grpo_batch: GRPOBatch) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute GRPO loss with proper per-token log-probability ratios.

        For each (query, response) pair the loss is:

            L = -min(r_t * A, clip(r_t, 1-eps, 1+eps) * A) + beta * KL

        where ``r_t`` is the per-token ratio ``exp(log_pi - log_pi_ref)``
        averaged over response tokens, and ``A`` is the group-relative
        advantage.

        This corrects the original implementation which incorrectly averaged
        log-probabilities over the vocabulary dimension.
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_kl_div = 0.0
        all_advantages: List[float] = []
        all_rewards: List[float] = []
        num_valid = 0

        for query, responses, rewards, advantages in zip(
            grpo_batch.queries,
            grpo_batch.responses,
            grpo_batch.rewards,
            grpo_batch.advantages,
        ):
            all_rewards.extend(rewards)

            for response, advantage in zip(responses, advantages):
                # Tokenize full sequence
                full_text = f"{query}\n{response}"
                tokens = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.model_config.max_length,
                    padding=False,
                ).to(self.device)

                input_ids = tokens["input_ids"]  # (1, seq_len)
                seq_len = input_ids.size(1)

                if seq_len < 2:
                    continue

                # ---- Current policy log-probs ----
                current_outputs = self.model(**tokens)
                # logits shape: (1, seq_len, vocab_size)
                current_logits = current_outputs.logits.float()
                current_log_probs = F.log_softmax(current_logits, dim=-1)

                # ---- Reference model log-probs ----
                with torch.no_grad():
                    ref_outputs = self.reference_model(**tokens)
                    ref_logits = ref_outputs.logits.float()
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

                # ---- Per-token log-prob of the ACTUAL next token ----
                # Shift so position i predicts token i+1
                shift_log_probs = current_log_probs[:, :-1, :]  # (1, seq_len-1, V)
                shift_ref_log_probs = ref_log_probs[:, :-1, :]  # (1, seq_len-1, V)
                target_ids = input_ids[:, 1:]  # (1, seq_len-1)

                # Gather the log-prob assigned to the actual next token
                # Result shape: (1, seq_len-1)
                current_token_lp = shift_log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                ref_token_lp = shift_ref_log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

                # Per-token log-ratio, then average over response tokens
                token_log_ratio = current_token_lp - ref_token_lp  # (1, seq_len-1)
                avg_log_ratio = token_log_ratio.mean()

                # Clamp for numerical stability, then exponentiate
                ratio = torch.exp(avg_log_ratio.clamp(-10.0, 10.0))

                # ---- Clipped surrogate loss ----
                advantage_t = torch.tensor(advantage, dtype=torch.float32, device=self.device)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                policy_loss = -torch.min(ratio * advantage_t, clipped_ratio * advantage_t)

                # ---- KL divergence (per-token, averaged) ----
                # KL(pi || pi_ref) approx = mean of (log_pi - log_pi_ref)
                # We already have per-token log-ratios
                kl_div = token_log_ratio.mean()

                # ---- Combined loss ----
                loss_i = policy_loss + self.kl_penalty * kl_div
                total_loss = total_loss + loss_i
                total_kl_div += kl_div.item()
                all_advantages.append(advantage)
                num_valid += 1

        # Average over all (query, response) pairs
        if num_valid > 0:
            total_loss = total_loss / num_valid
            avg_kl_div = total_kl_div / num_valid
        else:
            avg_kl_div = 0.0

        metrics = {
            "avg_kl_div": avg_kl_div,
            "advantages_stats": {
                "mean": float(np.mean(all_advantages)) if all_advantages else 0.0,
                "std": float(np.std(all_advantages)) if all_advantages else 0.0,
                "min": float(np.min(all_advantages)) if all_advantages else 0.0,
                "max": float(np.max(all_advantages)) if all_advantages else 0.0,
            },
            "reward_stats": {
                "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
                "std_reward": float(np.std(all_rewards)) if all_rewards else 0.0,
                "min_reward": float(np.min(all_rewards)) if all_rewards else 0.0,
                "max_reward": float(np.max(all_rewards)) if all_rewards else 0.0,
            },
        }

        return total_loss, metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save a training checkpoint (model + optimizer state)."""
        checkpoint_dir = f"{self.paths_config.models_dir}/checkpoint_epoch_{epoch}"

        import os

        os.makedirs(checkpoint_dir, exist_ok=True)

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), f"{checkpoint_dir}/model.pt")

        self.tokenizer.save_pretrained(checkpoint_dir)

        training_state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "loss": loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        torch.save(training_state, f"{checkpoint_dir}/training_state.pt")

        if self.logger:
            self.logger.log_model_checkpoint(checkpoint_dir, {"loss": loss, "epoch": epoch})

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release model memory and clear GPU caches."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if self.reference_model is not None:
                del self.reference_model
                self.reference_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError:
            pass

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass
