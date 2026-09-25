"""Shared fixtures and configuration for DTE test suite."""

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests that require a GPU")


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests when no GPU is available."""
    try:
        import torch

        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    if not gpu_available:
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    """Return a DTEConfig with default values."""
    from dte.core.config import DTEConfig

    return DTEConfig()


@pytest.fixture
def config_dict():
    """Return a minimal configuration dictionary."""
    return {
        "model": {
            "base_model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
        },
        "debate": {
            "num_agents": 3,
            "max_rounds": 2,
        },
        "training": {
            "learning_rate": 2e-5,
            "max_epochs": 1,
            "batch_size": 2,
        },
        "evolution": {
            "max_rounds": 1,
            "samples_per_round": 10,
        },
    }


@pytest.fixture
def sample_yaml_config(tmp_path):
    """Write a minimal config.yaml and return its path."""
    import yaml

    config = {
        "model": {
            "base_model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_length": 512,
        },
        "debate": {"num_agents": 3, "max_rounds": 2},
        "training": {"learning_rate": 2e-5, "max_epochs": 1, "batch_size": 2},
        "evolution": {"max_rounds": 1},
    }
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def reward_model():
    """Return an initialized DTERewardModel."""
    from dte.training.reward_model import DTERewardModel

    return DTERewardModel()


@pytest.fixture
def prompt_manager():
    """Return an initialized DebatePromptManager."""
    from dte.debate.prompts import DebatePromptManager

    return DebatePromptManager()
