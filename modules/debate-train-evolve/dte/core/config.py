"""
Configuration management for the DTE Framework.

This module provides comprehensive configuration handling for all components
of the Debate, Train, Evolve pipeline, including validation and defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..utils.helpers import ConfigurationError, validate_file_path, validate_model_name


@dataclass
class ModelConfig:
    """Configuration for model settings."""

    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    base_model_path: Optional[str] = None
    device: str = "auto"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


@dataclass
class DebatePromptingConfig:
    """Configuration for RCR (Reflect-Critique-Refine) debate prompting.

    RCR is the core prompting innovation of DTE. Each debate round after
    round 0 follows three explicit phases: Reflect, Critique, Refine.

    Attributes:
        enabled: Whether to use RCR prompting. Defaults to ``True`` since
            RCR is the core innovation of the DTE framework.
        initial_prompt_type: Task type for initial prompts (math, arc, general).
        include_agent_context: Include agent's own previous response in prompts.
        include_peer_solutions: Include peer solutions in debate prompts.
        defend_previous_answer: Encourage agents to defend correct answers.
        require_novel_reasoning: Require agents to introduce novel reasoning.
        critique_pairs: Number of peer solutions each agent must critique.
            Defaults to 2 per the DTE paper specification.
    """

    enabled: bool = True  # RCR is the core DTE innovation; on by default
    initial_prompt_type: str = "math"  # math, arc, general
    include_agent_context: bool = True
    include_peer_solutions: bool = True
    defend_previous_answer: bool = True
    require_novel_reasoning: bool = True
    critique_pairs: int = 2


@dataclass
class TemperatureAnnealingConfig:
    """Configuration for temperature annealing in debates."""

    enabled: bool = True
    start_temp: float = 0.7
    end_temp: float = 0.3
    min_model_size: str = "3B"


@dataclass
class DebateConfig:
    """Configuration for multi-agent debate settings matching original DTE."""

    num_agents: int = 3
    max_rounds: int = 3

    # Consensus detection parameters (exact match with original)
    consensus_threshold: float = 1.0  # Legacy parameter
    consensus_tolerance: float = 1e-9  # Exact tolerance from original DTE

    # Prompting configuration
    debate_prompting: DebatePromptingConfig = field(default_factory=DebatePromptingConfig)
    # Legacy field for backward compatibility
    rcr_prompting: DebatePromptingConfig = field(default_factory=DebatePromptingConfig)

    # Agent configuration
    use_diverse_agents: bool = False
    agent_models: List[str] = field(default_factory=list)

    # Temperature annealing
    temperature_annealing: TemperatureAnnealingConfig = field(default_factory=TemperatureAnnealingConfig)

    # Sycophancy tracking
    track_sycophancy: bool = True
    consolidate_reasoning: bool = True


@dataclass
class DatasetInfo:
    """Configuration for a single dataset."""

    name: str
    path: str
    split: str
    max_samples: int


@dataclass
class DatasetsConfig:
    """Configuration for training and evaluation datasets."""

    names: List[str] = field(default_factory=lambda: ["gsm8k"])
    max_samples_per_dataset: int = 1000
    quality_threshold: float = 0.8
    train_datasets: List[DatasetInfo] = field(default_factory=list)
    eval_datasets: List[DatasetInfo] = field(default_factory=list)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training parameters."""

    group_size: int = 4
    advantage_normalization: bool = True
    clip_ratio: float = 0.2
    kl_penalty: float = 0.02


@dataclass
class RewardsConfig:
    """Configuration for DTE reward functions exactly matching original implementation."""

    # All 5 DTE reward function weights
    correctness_weight: float = 2.0  # Most important - correct answers
    int_weight: float = 0.5  # Numeric answer format
    strict_format_weight: float = 0.5  # Exact XML format compliance
    soft_format_weight: float = 0.5  # Flexible XML format
    xmlcount_weight: float = 0.5  # Granular XML scoring

    # Legacy weights for backward compatibility
    answer_weight: float = 2.0
    format_weight: float = 0.5
    length_weight: float = 0.5
    length_tau: int = 120

    def get_dte_weights(self) -> Dict[str, float]:
        """Get DTE-specific reward weights."""
        return {
            "correctness": self.correctness_weight,
            "int": self.int_weight,
            "strict_format": self.strict_format_weight,
            "soft_format": self.soft_format_weight,
            "xmlcount": self.xmlcount_weight,
        }


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    enabled: bool = True
    rank: int = 128
    alpha: int = 256
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class EvolutionConfig:
    """Configuration for evolution rounds."""

    max_rounds: int = 3
    convergence_threshold: float = 0.01
    patience: int = 2
    samples_per_round: int = 500
    validation_split: float = 0.2
    validation_freq: int = 1
    min_improvement: float = 0.01


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""

    enabled: bool = False
    project: str = "dte-framework"
    entity: Optional[str] = None


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""

    level: str = "INFO"
    log_dir: str = "./logs"
    experiment_name: str = "dte_experiment"
    save_checkpoints: bool = True
    checkpoint_freq: int = 100
    track_metrics: List[str] = field(
        default_factory=lambda: [
            "accuracy",
            "debate_consensus_rate",
            "sycophancy_rate",
            "average_reasoning_length",
            "training_loss",
            "kl_divergence",
        ]
    )


@dataclass
class HardwareConfig:
    """Configuration for hardware and optimization settings."""

    device: str = "auto"
    mixed_precision: bool = True
    max_memory_per_gpu: str = "20GB"
    gradient_checkpointing: bool = True
    num_workers: int = 4
    dataloader_pin_memory: bool = True


@dataclass
class PathsConfig:
    """Configuration for file paths and storage."""

    output_dir: str = "./outputs"
    models_dir: str = "./models"
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    temp_dir: str = "./tmp"


@dataclass
class ExperimentConfig:
    """Configuration for experiment metadata and tracking."""

    name: str = "dte_pipeline_v1"
    description: str = "End-to-end DTE pipeline with RCR prompting and GRPO training"
    tags: List[str] = field(default_factory=lambda: ["dte", "multi-agent", "grpo", "reasoning"])
    seed: int = 42
    deterministic: bool = True
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class SafetyConfig:
    """Configuration for safety and validation settings."""

    filter_toxic_content: bool = True
    max_reasoning_length: int = 1000
    validate_model_outputs: bool = True
    check_format_compliance: bool = True
    auto_backup: bool = True
    backup_frequency: str = "1h"


@dataclass
class DTEConfig:
    """Main configuration class for the DTE Framework."""

    model: ModelConfig = field(default_factory=ModelConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "DTEConfig":
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            DTEConfig instance with loaded configuration.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML parsing fails.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DTEConfig":
        """Create configuration from a dictionary.

        Args:
            config_dict: Dictionary containing configuration values.

        Returns:
            DTEConfig instance.
        """
        # Create nested configurations
        model = ModelConfig(**config_dict.get("model", {}))

        # Parse debate config with nested structures
        debate_dict = config_dict.get("debate", {})
        debate_prompting = DebatePromptingConfig(
            **debate_dict.get("debate_prompting", debate_dict.get("rcr_prompting", {}))
        )
        temperature_annealing = TemperatureAnnealingConfig(**debate_dict.get("temperature_annealing", {}))
        debate = DebateConfig(
            **{
                k: v
                for k, v in debate_dict.items()
                if k not in ["debate_prompting", "rcr_prompting", "temperature_annealing"]
            },
            debate_prompting=debate_prompting,
            rcr_prompting=debate_prompting,  # Backward compatibility
            temperature_annealing=temperature_annealing,
        )

        # Parse datasets config
        datasets_dict = config_dict.get("datasets", {})
        train_datasets = [DatasetInfo(**ds) for ds in datasets_dict.get("train_datasets", [])]
        eval_datasets = [DatasetInfo(**ds) for ds in datasets_dict.get("eval_datasets", [])]
        datasets = DatasetsConfig(train_datasets=train_datasets, eval_datasets=eval_datasets)

        # Parse training config with nested structures
        training_dict = config_dict.get("training", {})
        grpo = GRPOConfig(**training_dict.get("grpo", {}))
        rewards = RewardsConfig(**training_dict.get("rewards", {}))
        lora = LoRAConfig(**training_dict.get("lora", {}))
        training = TrainingConfig(
            **{k: v for k, v in training_dict.items() if k not in ["grpo", "rewards", "lora"]},
            grpo=grpo,
            rewards=rewards,
            lora=lora,
        )

        # Parse other configurations
        evolution = EvolutionConfig(**config_dict.get("evolution", {}))
        logging = LoggingConfig(**config_dict.get("logging", {}))
        hardware = HardwareConfig(**config_dict.get("hardware", {}))
        paths = PathsConfig(**config_dict.get("paths", {}))

        # Parse experiment config with wandb
        experiment_dict = config_dict.get("experiment", {})
        wandb = WandbConfig(**experiment_dict.get("wandb", {}))
        experiment = ExperimentConfig(**{k: v for k, v in experiment_dict.items() if k != "wandb"}, wandb=wandb)

        safety = SafetyConfig(**config_dict.get("safety", {}))

        return cls(
            model=model,
            debate=debate,
            datasets=datasets,
            training=training,
            evolution=evolution,
            logging=logging,
            hardware=hardware,
            paths=paths,
            experiment=experiment,
            safety=safety,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """

        def dataclass_to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            else:
                return obj

        return dataclass_to_dict(self)

    def save_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.

        Args:
            config_path: Path where to save the configuration.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def validate(self, strict: bool = False) -> List[str]:
        """Validate configuration settings comprehensively.

        Args:
            strict: If True, perform additional strict validations

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Validate model configuration
        errors.extend(self._validate_model_config(strict))
        errors.extend(self._validate_debate_config(strict))
        errors.extend(self._validate_training_config(strict))
        errors.extend(self._validate_evolution_config(strict))
        errors.extend(self._validate_datasets_config(strict))
        errors.extend(self._validate_paths_config(strict))

        return errors

    def _validate_model_config(self, strict: bool = False) -> List[str]:
        """Validate model configuration."""
        errors = []

        # Basic validation
        if not self.model.base_model_name:
            errors.append("model.base_model_name cannot be empty")
        else:
            try:
                validate_model_name(self.model.base_model_name)
            except Exception as e:
                errors.append(f"Invalid model.base_model_name: {e}")

        if self.model.max_length <= 0:
            errors.append("model.max_length must be positive")
        elif strict and self.model.max_length < 512:
            errors.append("model.max_length should be at least 512 for reasonable performance")

        if not (0 <= self.model.temperature <= 2):
            errors.append("model.temperature must be between 0 and 2")

        if not (0 <= self.model.top_p <= 1):
            errors.append("model.top_p must be between 0 and 1")

        if self.model.top_k <= 0:
            errors.append("model.top_k must be positive")

        # Validate base model path if specified
        if self.model.base_model_path:
            try:
                validate_file_path(Path(self.model.base_model_path), must_exist=True)
            except ConfigurationError as e:
                errors.append(f"model.base_model_path: {e}")

        return errors

    def _validate_debate_config(self, strict: bool = False) -> List[str]:
        """Validate debate configuration."""
        errors = []

        if self.debate.num_agents < 2:
            errors.append("debate.num_agents must be at least 2")
        elif strict and self.debate.num_agents > 10:
            errors.append("debate.num_agents > 10 may be computationally expensive")

        if self.debate.max_rounds <= 0:
            errors.append("debate.max_rounds must be positive")
        elif strict and self.debate.max_rounds > 5:
            errors.append("debate.max_rounds > 5 may lead to diminishing returns")

        if not (0 <= self.debate.consensus_threshold <= 1):
            errors.append("debate.consensus_threshold must be between 0 and 1")

        if hasattr(self.debate, "consensus_tolerance"):
            try:
                tolerance = float(self.debate.consensus_tolerance)
                if tolerance <= 0:
                    errors.append("debate.consensus_tolerance must be positive")
            except (ValueError, TypeError):
                errors.append("debate.consensus_tolerance must be a valid number")

        # Validate agent models if using diverse agents
        if self.debate.use_diverse_agents:
            if not self.debate.agent_models:
                errors.append("debate.agent_models cannot be empty when use_diverse_agents is True")
            elif len(self.debate.agent_models) != self.debate.num_agents:
                errors.append(
                    f"Number of agent_models ({len(self.debate.agent_models)}) must match num_agents ({self.debate.num_agents})"
                )
            else:
                for i, model_name in enumerate(self.debate.agent_models):
                    try:
                        validate_model_name(model_name)
                    except Exception as e:
                        errors.append(f"Invalid agent_models[{i}]: {e}")

        # Validate temperature annealing
        if self.debate.temperature_annealing.enabled:
            if not (0 <= self.debate.temperature_annealing.start_temp <= 2):
                errors.append("debate.temperature_annealing.start_temp must be between 0 and 2")
            if not (0 <= self.debate.temperature_annealing.end_temp <= 2):
                errors.append("debate.temperature_annealing.end_temp must be between 0 and 2")
            if self.debate.temperature_annealing.start_temp < self.debate.temperature_annealing.end_temp:
                errors.append("debate.temperature_annealing.start_temp should be >= end_temp")

        return errors

    def _validate_training_config(self, strict: bool = False) -> List[str]:
        """Validate training configuration."""
        errors = []

        try:
            lr = float(self.training.learning_rate)
            if lr <= 0:
                errors.append("training.learning_rate must be positive")
            elif strict and (lr > 1e-3 or lr < 1e-6):
                errors.append("training.learning_rate should typically be between 1e-6 and 1e-3")
        except (ValueError, TypeError):
            errors.append("training.learning_rate must be a valid number")

        if self.training.batch_size <= 0:
            errors.append("training.batch_size must be positive")
        elif strict and self.training.batch_size > 64:
            errors.append("training.batch_size > 64 may cause memory issues")

        if self.training.max_epochs <= 0:
            errors.append("training.max_epochs must be positive")

        if self.training.weight_decay < 0:
            errors.append("training.weight_decay must be non-negative")

        if self.training.warmup_steps < 0:
            errors.append("training.warmup_steps must be non-negative")

        if self.training.gradient_accumulation_steps <= 0:
            errors.append("training.gradient_accumulation_steps must be positive")

        # Validate GRPO settings
        if self.training.grpo.group_size <= 0:
            errors.append("training.grpo.group_size must be positive")
        elif strict and self.training.grpo.group_size < 4:
            errors.append("training.grpo.group_size < 4 may lead to unstable training")

        if not (0 <= self.training.grpo.clip_ratio <= 1):
            errors.append("training.grpo.clip_ratio must be between 0 and 1")

        if self.training.grpo.kl_penalty < 0:
            errors.append("training.grpo.kl_penalty must be non-negative")

        # Validate LoRA settings if enabled
        if self.training.lora.enabled:
            if self.training.lora.rank <= 0:
                errors.append("training.lora.rank must be positive")
            elif strict and self.training.lora.rank > 512:
                errors.append("training.lora.rank > 512 may be unnecessarily large")

            if self.training.lora.alpha <= 0:
                errors.append("training.lora.alpha must be positive")

            if not (0 <= self.training.lora.dropout <= 1):
                errors.append("training.lora.dropout must be between 0 and 1")

        return errors

    def _validate_evolution_config(self, strict: bool = False) -> List[str]:
        """Validate evolution configuration."""
        errors = []

        if self.evolution.max_rounds <= 0:
            errors.append("evolution.max_rounds must be positive")
        elif strict and self.evolution.max_rounds > 5:
            errors.append("evolution.max_rounds > 5 may have diminishing returns")

        if self.evolution.samples_per_round <= 0:
            errors.append("evolution.samples_per_round must be positive")
        elif strict and self.evolution.samples_per_round < 100:
            errors.append("evolution.samples_per_round < 100 may not provide enough training data")

        if self.evolution.patience <= 0:
            errors.append("evolution.patience must be positive")

        if not (0 <= self.evolution.min_improvement <= 1):
            errors.append("evolution.min_improvement must be between 0 and 1")

        return errors

    def _validate_datasets_config(self, strict: bool = False) -> List[str]:
        """Validate datasets configuration."""
        errors = []

        # Check that we have some datasets configured
        if not self.datasets.names:
            errors.append("datasets.names cannot be empty")

        # Validate each dataset name against the full set of 7 supported datasets
        valid_datasets = [
            "gsm8k",
            "gsm_plus",
            "math",
            "arc_challenge",
            "arc_easy",
            "gpqa",
            "commonsense_qa",
        ]
        for dataset_name in self.datasets.names:
            if dataset_name not in valid_datasets:
                errors.append(f"Unknown dataset: {dataset_name}. Valid options: {valid_datasets}")

        if self.datasets.max_samples_per_dataset <= 0:
            errors.append("datasets.max_samples_per_dataset must be positive")

        if self.datasets.quality_threshold < 0 or self.datasets.quality_threshold > 1:
            errors.append("datasets.quality_threshold must be between 0 and 1")

        return errors

    def _validate_paths_config(self, strict: bool = False) -> List[str]:
        """Validate paths configuration."""
        errors = []

        # Validate that paths can be created
        for path_name, path_value in self.paths.__dict__.items():
            try:
                validate_file_path(Path(path_value), create_parent=True)
            except ConfigurationError as e:
                errors.append(f"paths.{path_name}: {e}")

        return errors

    def validate_strict(self) -> List[str]:
        """Perform strict validation with additional checks."""
        return self.validate(strict=True)

    def validate_and_raise(self, strict: bool = False) -> None:
        """Validate configuration and raise exception if invalid.

        Args:
            strict: If True, perform strict validation

        Raises:
            ConfigurationError: If validation fails
        """
        errors = self.validate(strict)
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ConfigurationError(error_msg, "config_validation")

    def setup_environment(self) -> None:
        """Set up environment variables and directories based on configuration."""
        import torch

        # Set random seeds for reproducibility
        if self.experiment.deterministic:
            torch.manual_seed(self.experiment.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.experiment.seed)
            import random

            import numpy as np

            random.seed(self.experiment.seed)
            np.random.seed(self.experiment.seed)

        # Create necessary directories
        for path_value in self.paths.__dict__.values():
            Path(path_value).mkdir(parents=True, exist_ok=True)

        # Set cache directory for transformers
        os.environ["TRANSFORMERS_CACHE"] = self.paths.cache_dir
        os.environ["HF_HOME"] = self.paths.cache_dir
