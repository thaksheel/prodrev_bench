"""Core DTE Framework components."""

from .config import (
    DebateConfig,
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
    TrainingConfig,
)
from .logger import DTELogger

# Conditional import for pipeline (requires ML deps)
try:
    from .evaluator import DTEEvaluator
    from .pipeline import DTEPipeline

    _PIPELINE_AVAILABLE = True
except ImportError:
    DTEPipeline = None  # type: ignore[assignment,misc]
    DTEEvaluator = None  # type: ignore[assignment,misc]
    _PIPELINE_AVAILABLE = False

__all__ = [
    "DTEConfig",
    "ModelConfig",
    "DebateConfig",
    "TrainingConfig",
    "EvolutionConfig",
    "HardwareConfig",
    "PathsConfig",
    "RewardsConfig",
    "GRPOConfig",
    "LoRAConfig",
    "LoggingConfig",
    "ExperimentConfig",
    "SafetyConfig",
    "DTELogger",
]

if _PIPELINE_AVAILABLE:
    __all__.extend(["DTEPipeline", "DTEEvaluator"])
