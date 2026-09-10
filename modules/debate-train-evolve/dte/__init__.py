"""
DTE Framework: Debate, Train, Evolve

A complete implementation of the Debate, Train, Evolve framework for improving
language model reasoning through multi-agent debate and iterative training.

This package provides:
- Multi-agent debate with RCR (Reflect-Critique-Refine) prompting
- GRPO (Group Relative Policy Optimization) training
- End-to-end pipeline orchestration
- Comprehensive evaluation and monitoring

Quick start::

    import dte

    # Run a quick debate
    result = dte.debate("What is 15 * 24?", model="Qwen/Qwen2.5-1.5B-Instruct")

    # Full pipeline from config
    pipeline = dte.from_config("config.yaml")
    results = pipeline.run_complete_pipeline()

Author: DTE Research Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "DTE Research Team"

# Core imports - always available
from .core.config import (
    DebateConfig,
    DTEConfig,
    EvolutionConfig,
    GRPOConfig,
    HardwareConfig,
    LoRAConfig,
    ModelConfig,
    PathsConfig,
    RewardsConfig,
    TrainingConfig,
)

# Conditional imports for components that require external ML dependencies
try:
    from .core.pipeline import DTEPipeline
    from .data.dataset_manager import DatasetManager
    from .data.generator import DebateDataGenerator
    from .debate.agent import DebateAgent
    from .debate.manager import DebateManager, DebateResult
    from .debate.prompts import DebatePromptManager, DebateResponse
    from .training.grpo_trainer import GRPOTrainer
    from .training.reward_model import DTERewardModel

    _FULL_IMPORTS_AVAILABLE = True
except ImportError:
    DTEPipeline = None  # type: ignore[assignment,misc]
    DebateManager = None  # type: ignore[assignment,misc]
    DebateResult = None  # type: ignore[assignment,misc]
    DebateAgent = None  # type: ignore[assignment,misc]
    DebatePromptManager = None  # type: ignore[assignment,misc]
    DebateResponse = None  # type: ignore[assignment,misc]
    GRPOTrainer = None  # type: ignore[assignment,misc]
    DTERewardModel = None  # type: ignore[assignment,misc]
    DebateDataGenerator = None  # type: ignore[assignment,misc]
    DatasetManager = None  # type: ignore[assignment,misc]
    _FULL_IMPORTS_AVAILABLE = False

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Config classes
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
    # High-level API
    "debate",
    "from_config",
    "train",
    "evaluate",
]

# Add ML-dependent exports when available
if _FULL_IMPORTS_AVAILABLE:
    __all__.extend(
        [
            "DTEPipeline",
            "DebateManager",
            "DebateResult",
            "DebateAgent",
            "DebatePromptManager",
            "DebateResponse",
            "GRPOTrainer",
            "DTERewardModel",
            "DebateDataGenerator",
            "DatasetManager",
        ]
    )


def debate(
    query: str,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_agents: int = 3,
    max_rounds: int = 3,
    task_type: str = "math",
    device: str = "auto",
    temperature: float = 0.7,
    verbose: bool = False,
) -> "DebateResult":
    """Run a quick multi-agent debate on a single query.

    This is the simplest way to use the DTE framework. It creates a
    temporary debate configuration, initializes agents, and runs a
    complete debate.

    Args:
        query: The question or problem to debate.
        model: HuggingFace model name or local path.
        num_agents: Number of debate agents (default: 3).
        max_rounds: Maximum number of debate rounds (default: 3).
        task_type: Type of task -- "math", "arc", or "general".
        device: Computation device -- "auto", "cpu", "cuda", or "cuda:N".
        temperature: Sampling temperature for generation.
        verbose: If True, print progress to console.

    Returns:
        DebateResult with final_answer, consensus info, and full history.

    Raises:
        RuntimeError: If ML dependencies are not installed.

    Example::

        result = dte.debate("What is 15 * 24?")
        print(result.final_answer)
        print(result.consensus_reached)
    """
    if not _FULL_IMPORTS_AVAILABLE:
        raise RuntimeError(
            "ML dependencies (torch, transformers) are required for debate. "
            "Install with: pip install torch transformers"
        )

    # Build minimal config
    model_config = ModelConfig(
        base_model_name=model,
        device=device,
        temperature=temperature,
    )
    debate_config = DebateConfig(
        num_agents=num_agents,
        max_rounds=max_rounds,
    )

    logger = None
    if verbose:
        from .core.config import LoggingConfig
        from .core.logger import DTELogger

        log_cfg = LoggingConfig(level="INFO")
        logger = DTELogger(log_cfg, "quick_debate")

    manager = DebateManager(debate_config, model_config, logger)
    try:
        result = manager.conduct_debate(query, task_type)
    finally:
        manager.cleanup()

    return result


def from_config(config_path: str) -> "DTEPipeline":
    """Create a DTEPipeline from a YAML configuration file.

    This is a convenience shortcut for ``DTEPipeline(DTEConfig.from_yaml(path))``.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        Initialized DTEPipeline ready to run.

    Raises:
        RuntimeError: If ML dependencies are not installed.
        FileNotFoundError: If config file does not exist.

    Example::

        pipeline = dte.from_config("config.yaml")
        results = pipeline.run_complete_pipeline()
    """
    if not _FULL_IMPORTS_AVAILABLE:
        raise RuntimeError(
            "ML dependencies (torch, transformers) are required. Install with: pip install torch transformers"
        )

    config = DTEConfig.from_yaml(config_path)
    return DTEPipeline(config)


def train(
    data_path: str,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    output_dir: str = "./models",
    verbose: bool = False,
) -> dict:
    """Train a model using GRPO on previously generated debate data.

    This is a convenience function that sets up a minimal training
    configuration, loads debate data from *data_path*, and runs
    GRPO training.

    Args:
        data_path: Path to a JSONL file of debate training examples.
        model: HuggingFace model name or local path.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Peak learning rate.
        output_dir: Directory for model checkpoints.
        verbose: Print progress to console.

    Returns:
        Dictionary of training metrics.

    Raises:
        RuntimeError: If ML dependencies are not installed.
        FileNotFoundError: If *data_path* does not exist.

    Example::

        metrics = dte.train("debate_data.jsonl", epochs=2)
        print(metrics["epoch_losses"])
    """
    if not _FULL_IMPORTS_AVAILABLE:
        raise RuntimeError(
            "ML dependencies (torch, transformers) are required. Install with: pip install torch transformers"
        )

    model_config = ModelConfig(base_model_name=model)
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        max_epochs=epochs,
        batch_size=batch_size,
    )
    paths_config = PathsConfig(models_dir=output_dir)

    logger = None
    if verbose:
        from .core.config import LoggingConfig
        from .core.logger import DTELogger

        log_cfg = LoggingConfig(level="INFO")
        logger = DTELogger(log_cfg, "quick_train")

    # Load training data
    generator = DebateDataGenerator.__new__(DebateDataGenerator)
    generator.logger = logger
    examples = generator.load_generated_data(data_path)

    # Train
    trainer = GRPOTrainer(training_config, model_config, paths_config, logger)
    try:
        metrics = trainer.train(examples)
    finally:
        trainer.cleanup()

    return metrics


def evaluate(
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    datasets: list = None,
    num_agents: int = 3,
    max_rounds: int = 3,
    max_samples: int = 100,
    verbose: bool = False,
) -> dict:
    """Evaluate a model using multi-agent debate on standard benchmarks.

    This is a convenience function that runs debates on the specified
    datasets and returns accuracy and debate metrics.

    Args:
        model: HuggingFace model name or local path.
        datasets: List of dataset names (e.g. ``["gsm8k", "arc_challenge"]``).
            Defaults to ``["gsm8k"]``.
        num_agents: Number of debate agents.
        max_rounds: Maximum debate rounds.
        max_samples: Maximum samples per dataset.
        verbose: Print progress to console.

    Returns:
        Dictionary of evaluation metrics.

    Raises:
        RuntimeError: If ML dependencies are not installed.

    Example::

        report = dte.evaluate(datasets=["gsm8k"], max_samples=50)
        print(f"Accuracy: {report['overall_metrics']['accuracy']:.2%}")
    """
    if not _FULL_IMPORTS_AVAILABLE:
        raise RuntimeError(
            "ML dependencies (torch, transformers) are required. Install with: pip install torch transformers"
        )

    if datasets is None:
        datasets = ["gsm8k"]

    from .core.config import DatasetsConfig, LoggingConfig
    from .core.evaluator import DTEEvaluator
    from .core.logger import DTELogger

    model_config = ModelConfig(base_model_name=model)
    debate_config = DebateConfig(num_agents=num_agents, max_rounds=max_rounds)
    datasets_config = DatasetsConfig(names=datasets, max_samples_per_dataset=max_samples)

    logger = None
    if verbose:
        log_cfg = LoggingConfig(level="INFO")
        logger = DTELogger(log_cfg, "quick_eval")

    evaluator = DTEEvaluator(datasets_config, debate_config, model_config, logger)
    try:
        metrics = evaluator.evaluate_model(evolution_round=0, max_samples_per_dataset=max_samples)
        report = evaluator.create_evaluation_report(metrics, 0)
        return report
    finally:
        evaluator.cleanup()
