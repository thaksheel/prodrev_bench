"""Tests for DTE configuration loading and validation."""

from pathlib import Path

import pytest
import yaml

from dte.core.config import (
    DatasetsConfig,
    DebateConfig,
    DebatePromptingConfig,
    DTEConfig,
    EvolutionConfig,
    ExperimentConfig,
    GRPOConfig,
    HardwareConfig,
    LoggingConfig,
    LoRAConfig,
    ModelConfig,
    PathsConfig,
    RewardsConfig,
    SafetyConfig,
    TemperatureAnnealingConfig,
    TrainingConfig,
    WandbConfig,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.base_model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.device == "auto"
        assert cfg.max_length == 2048
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.top_k == 50
        assert cfg.base_model_path is None

    def test_custom_values(self):
        cfg = ModelConfig(base_model_name="gpt2", device="cuda:0", max_length=1024)
        assert cfg.base_model_name == "gpt2"
        assert cfg.device == "cuda:0"
        assert cfg.max_length == 1024


class TestDebateConfig:
    """Tests for DebateConfig dataclass."""

    def test_defaults(self):
        cfg = DebateConfig()
        assert cfg.num_agents == 3
        assert cfg.max_rounds == 3
        assert cfg.consensus_threshold == 1.0
        assert cfg.consensus_tolerance == pytest.approx(1e-9)
        assert cfg.track_sycophancy is True

    def test_temperature_annealing(self):
        cfg = DebateConfig()
        assert cfg.temperature_annealing.enabled is True
        assert cfg.temperature_annealing.start_temp == 0.7
        assert cfg.temperature_annealing.end_temp == 0.3


class TestTrainingConfig:
    """Tests for TrainingConfig and nested configs."""

    def test_grpo_defaults(self):
        cfg = GRPOConfig()
        assert cfg.group_size == 4
        assert cfg.clip_ratio == 0.2
        assert cfg.kl_penalty == 0.02

    def test_rewards_dte_weights(self):
        cfg = RewardsConfig()
        weights = cfg.get_dte_weights()
        assert weights["correctness"] == 2.0
        assert weights["int"] == 0.5
        assert weights["strict_format"] == 0.5
        assert weights["soft_format"] == 0.5
        assert weights["xmlcount"] == 0.5

    def test_lora_defaults(self):
        cfg = LoRAConfig()
        assert cfg.enabled is True
        assert cfg.rank == 128
        assert cfg.alpha == 256
        assert "q_proj" in cfg.target_modules


class TestDTEConfig:
    """Tests for the top-level DTEConfig."""

    def test_default_construction(self, default_config):
        assert isinstance(default_config, DTEConfig)
        assert isinstance(default_config.model, ModelConfig)
        assert isinstance(default_config.debate, DebateConfig)
        assert isinstance(default_config.training, TrainingConfig)
        assert isinstance(default_config.hardware, HardwareConfig)

    def test_from_dict(self, config_dict):
        cfg = DTEConfig.from_dict(config_dict)
        assert cfg.model.base_model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert cfg.debate.num_agents == 3
        assert cfg.training.learning_rate == 2e-5

    def test_from_yaml(self, sample_yaml_config):
        cfg = DTEConfig.from_yaml(sample_yaml_config)
        assert cfg.model.base_model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert cfg.debate.num_agents == 3

    def test_from_yaml_missing_file(self):
        with pytest.raises(FileNotFoundError):
            DTEConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_to_dict(self, default_config):
        d = default_config.to_dict()
        assert isinstance(d, dict)
        assert "model" in d
        assert "debate" in d
        assert d["model"]["base_model_name"] == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_save_and_load_yaml(self, default_config, tmp_path):
        path = tmp_path / "roundtrip.yaml"
        default_config.save_yaml(path)
        loaded = DTEConfig.from_yaml(path)
        assert loaded.model.base_model_name == default_config.model.base_model_name
        assert loaded.debate.num_agents == default_config.debate.num_agents

    def test_validate_valid_config(self, default_config):
        errors = default_config.validate()
        assert isinstance(errors, list)
        # Default config should be valid
        assert len(errors) == 0

    def test_validate_bad_temperature(self):
        cfg = DTEConfig()
        cfg.model.temperature = 5.0
        errors = cfg.validate()
        assert any("temperature" in e for e in errors)

    def test_validate_bad_learning_rate(self):
        cfg = DTEConfig()
        cfg.training.learning_rate = -1.0
        errors = cfg.validate()
        assert any("learning_rate" in e for e in errors)

    def test_validate_bad_num_agents(self):
        cfg = DTEConfig()
        cfg.debate.num_agents = 0
        errors = cfg.validate()
        assert any("num_agents" in e for e in errors)

    def test_validate_strict(self, default_config):
        errors = default_config.validate_strict()
        assert isinstance(errors, list)

    def test_validate_and_raise(self):
        from dte.utils.helpers import ConfigurationError

        cfg = DTEConfig()
        cfg.model.temperature = 10.0
        with pytest.raises(ConfigurationError):
            cfg.validate_and_raise()

    def test_hardware_config_defaults(self):
        cfg = HardwareConfig()
        assert cfg.device == "auto"
        assert cfg.mixed_precision is True
