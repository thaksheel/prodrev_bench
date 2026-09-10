#!/usr/bin/env python3
"""
DTE Framework - Main CLI Interface

Command-line interface for the Debate, Train, Evolve framework.
Provides easy access to all pipeline components and functionality.
"""

import click
import sys
from pathlib import Path
from typing import Optional

from dte.core.config import DTEConfig
from dte.core.pipeline import DTEPipeline
from dte.core.logger import DTELogger


@click.group()
@click.version_option(version="0.1.0")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default="config.yaml",
    help="Path to configuration file"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging"
)
@click.pass_context
def cli(ctx: click.Context, config: Path, verbose: bool):
    """DTE Framework: Debate, Train, Evolve

    A complete pipeline for improving language model reasoning through
    multi-agent debate and iterative training.

    Examples:
        dte run --config config.yaml
        dte debate --query "What is 2+2?" --agents 3
        dte train --data debate_data.jsonl --epochs 3
    """
    # Ensure configuration file exists
    if not config.exists():
        click.echo(f"Error: Configuration file not found: {config}", err=True)
        click.echo("Run 'dte init' to create a default configuration.", err=True)
        sys.exit(1)

    # Load configuration
    try:
        dte_config = DTEConfig.from_yaml(config)

        # Validate configuration
        errors = dte_config.validate()
        if errors:
            click.echo("Configuration validation errors:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            sys.exit(1)

        # Set verbose logging if requested
        if verbose:
            dte_config.logging.level = "DEBUG"

    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)

    # Store config in context
    ctx.ensure_object(dict)
    ctx.obj['config'] = dte_config


@cli.command()
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Overwrite existing configuration file"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default="config.yaml",
    help="Output configuration file path"
)
def init(force: bool, output: Path):
    """Initialize a new DTE configuration file."""
    if output.exists() and not force:
        click.echo(f"Configuration file already exists: {output}")
        click.echo("Use --force to overwrite.")
        return

    # Create default configuration
    config = DTEConfig()
    config.save_yaml(output)

    click.echo(f"Created configuration file: {output}")
    click.echo("\nNext steps:")
    click.echo(f"1. Edit {output} to customize your settings")
    click.echo("2. Run 'dte run' to start the complete pipeline")
    click.echo("3. Or run individual components like 'dte debate' or 'dte train'")


@cli.command()
@click.option(
    "--resume",
    type=click.Path(exists=True, path_type=Path),
    help="Resume from checkpoint"
)
@click.option(
    "--save-checkpoint",
    type=click.Path(path_type=Path),
    help="Save checkpoint after each round"
)
@click.pass_context
def run(ctx: click.Context, resume: Optional[Path], save_checkpoint: Optional[Path]):
    """Run the complete DTE pipeline."""
    config = ctx.obj['config']

    click.echo("[*] Starting DTE Pipeline")
    click.echo(f"Experiment: {config.experiment.name}")
    click.echo(f"Max Rounds: {config.evolution.max_rounds}")
    click.echo()

    try:
        # Initialize pipeline
        pipeline = DTEPipeline(config)

        # Resume from checkpoint if provided
        if resume:
            pipeline.load_checkpoint(resume)
            click.echo(f"Resumed from checkpoint: {resume}")

        # Run complete pipeline
        results = pipeline.run_complete_pipeline()

        # Save checkpoint if requested
        if save_checkpoint:
            pipeline.save_checkpoint(save_checkpoint)

        # Display results
        click.echo()
        click.echo("[OK] Pipeline Completed Successfully!")
        click.echo(f"Total Time: {results['total_time_hours']:.2f} hours")
        click.echo(f"Evolution Rounds: {results['total_evolution_rounds']}")
        click.echo(f"Best Performance: {results['best_performance']:.4f}")
        click.echo(f"Total Improvement: {results['total_improvement']:.4f}")

    except KeyboardInterrupt:
        click.echo("\n[WARN] Pipeline interrupted by user")
        if save_checkpoint:
            pipeline.save_checkpoint(save_checkpoint)
            click.echo(f"Checkpoint saved to: {save_checkpoint}")
    except Exception as e:
        click.echo(f"\n[ERROR] Pipeline failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--query", "-q",
    help="Single query to debate"
)
@click.option(
    "--dataset",
    type=click.Choice(["gsm8k", "gsm_plus", "math", "arc_challenge", "arc_easy", "gpqa", "commonsense_qa"]),
    help="Dataset to debate on (alternative to single query)"
)
@click.option(
    "--samples", "-s",
    default=10,
    help="Number of samples from dataset to debate (if using dataset)"
)
@click.option(
    "--agents", "-a",
    default=3,
    help="Number of debate agents"
)
@click.option(
    "--rounds", "-r",
    default=3,
    help="Maximum debate rounds"
)
@click.option(
    "--task-type",
    default="auto",
    type=click.Choice(["math", "arc", "general", "auto"]),
    help="Type of task: math, arc, general, or auto (auto-detects from dataset)"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Save debate results to file"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed debate progression"
)
@click.option(
    "--models",
    help="Comma-separated list of model names to use as agents"
)
@click.pass_context
def debate(ctx: click.Context, query: Optional[str], dataset: Optional[str],
          samples: int, agents: int, rounds: int, task_type: str,
          output: Optional[Path], verbose: bool, models: Optional[str]):
    """Run multi-agent debates on single queries or datasets.

    Examples:
        # Single query debate
        dte debate -q "What is 15 * 24?" --agents 3 --rounds 5

        # Dataset evaluation
        dte debate --dataset gsm8k --samples 20 --agents 3

        # Use specific models
        dte debate -q "Solve x^2 + 5x + 6 = 0" --models "model1,model2,model3"
    """
    if not query and not dataset:
        click.echo("[ERROR] Error: Must specify either --query or --dataset", err=True)
        sys.exit(1)

    if query and dataset:
        click.echo("[ERROR] Error: Cannot specify both --query and --dataset", err=True)
        sys.exit(1)

    config = ctx.obj['config']

    # Override config with CLI parameters
    config.debate.num_agents = agents
    config.debate.max_rounds = rounds

    # Handle custom models if specified
    if models:
        model_list = [m.strip() for m in models.split(',')]
        if len(model_list) != agents:
            click.echo(f"[ERROR] Error: Number of models ({len(model_list)}) must match number of agents ({agents})", err=True)
            sys.exit(1)
        config.debate.agent_models = model_list
        config.debate.use_diverse_agents = True

    try:
        from dte.debate.manager import DebateManager
        from dte.data.dataset_manager import DatasetManager

        # Initialize logger
        logger = DTELogger(config.logging, "debate_session")

        # Create debate manager
        debate_manager = DebateManager(config.debate, config.model, logger)

        if query:
            # Single query debate
            _run_single_debate(query, task_type, debate_manager, verbose, output)
        else:
            # Dataset-based debate
            _run_dataset_debate(dataset, samples, task_type, debate_manager, verbose, output)

    except Exception as e:
        click.echo(f"[ERROR] Debate failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _run_single_debate(query: str, task_type: str, debate_manager, verbose: bool, output: Optional[Path]):
    """Run a single debate on a query."""
    # Auto-detect task type if needed
    if task_type == "auto":
        # Simple heuristics for task type detection
        if any(char.isdigit() for char in query) and any(op in query for op in ['+', '-', '*', '/', '=', 'solve']):
            task_type = "math"
        elif "choice" in query.lower() or any(letter in query for letter in ['A.', 'B.', 'C.', 'D.']):
            task_type = "arc"
        else:
            task_type = "general"

    click.echo(f"[*] Starting {debate_manager.num_agents}-agent debate")
    click.echo(f"[>] Query: {query}")
    click.echo(f"[=] Task Type: {task_type}")
    click.echo(f"[~] Max Rounds: {debate_manager.max_rounds}")
    click.echo()

    # Conduct debate
    result = debate_manager.conduct_debate(query, task_type)

    # Display results
    click.echo("=" * 60)
    click.echo("[=] DEBATE RESULTS")
    click.echo("=" * 60)
    click.echo(f"[OK] Final Answer: {result.final_answer}")
    click.echo(f"[=] Consensus Reached: {'Yes' if result.consensus_reached else 'No'}")
    click.echo(f"[~] Total Rounds: {result.total_rounds}")
    click.echo(f"[T] Time: {result.metrics.get('total_time', 0):.2f}s")
    click.echo(f"[+] Consensus Rate: {result.metrics.get('consensus_rate', 0):.2%}")

    if verbose:
        click.echo("\n[+] ANSWER PROGRESSION:")
        for round_idx, round_answers in enumerate(result.extracted_answers):
            click.echo(f"Round {round_idx}: {round_answers}")

        click.echo("\n[?] SYCOPHANCY ANALYSIS:")
        for round_idx, syc_data in enumerate(result.sycophancy_history):
            if any(syc_data.values()):
                agents_syc = [agent for agent, is_syc in syc_data.items() if is_syc]
                click.echo(f"Round {round_idx + 1}: Sycophancy detected in agents: {agents_syc}")

        click.echo("\n[R] REASONING PROGRESSION:")
        for round_idx, round_responses in enumerate(result.all_responses):
            click.echo(f"\nRound {round_idx}:")
            for agent_idx, response in enumerate(round_responses):
                click.echo(f"  Agent {agent_idx + 1}: {response.reasoning[:100]}...")

    # Save if requested
    if output:
        _save_debate_results([result], output)
        click.echo(f"\n[S] Results saved to: {output}")


def _run_dataset_debate(dataset_name: str, samples: int, task_type: str,
                       debate_manager, verbose: bool, output: Optional[Path]):
    """Run debates on a dataset."""
    from dte.data.dataset_manager import DatasetManager

    # Auto-detect task type from dataset
    if task_type == "auto":
        dataset_configs = DatasetManager.DATASET_CONFIGS
        if dataset_name in dataset_configs:
            task_type = dataset_configs[dataset_name]["task_type"]
        else:
            task_type = "general"

    click.echo(f"[*] Starting dataset debate evaluation")
    click.echo(f"[D] Dataset: {dataset_name}")
    click.echo(f"[=] Task Type: {task_type}")
    click.echo(f"[+] Samples: {samples}")
    click.echo(f"[A] Agents: {debate_manager.num_agents}")
    click.echo(f"[~] Max Rounds: {debate_manager.max_rounds}")
    click.echo()

    # Load dataset
    dataset_manager = DatasetManager()
    try:
        dataset = dataset_manager.load_dataset_by_name(dataset_name, split="test", max_samples=samples)
        processed_dataset = dataset_manager.preprocess_dataset(dataset, dataset_name)
    except:
        click.echo("[WARN] Test split not available, using train split with limited samples")
        dataset = dataset_manager.load_dataset_by_name(dataset_name, split="train", max_samples=samples)
        processed_dataset = dataset_manager.preprocess_dataset(dataset, dataset_name)

    # Run debates
    results = []
    correct_count = 0
    total_time = 0

    with click.progressbar(processed_dataset, label="Running debates") as dataset_iter:
        for idx, sample in enumerate(dataset_iter):
            query = sample["formatted_query"]
            ground_truth = sample["ground_truth"]

            if verbose:
                click.echo(f"\n[>] Sample {idx + 1}: {query[:100]}...")

            # Conduct debate
            result = debate_manager.conduct_debate(query, task_type)
            results.append(result)

            # Check correctness
            is_correct = _check_answer_correctness(result.final_answer, ground_truth, task_type)
            if is_correct:
                correct_count += 1

            total_time += result.metrics.get('total_time', 0)

            if verbose:
                status = "[OK]" if is_correct else "[ERROR]"
                click.echo(f"{status} Answer: {result.final_answer} (Expected: {ground_truth})")

    # Display summary
    accuracy = correct_count / len(results) if results else 0
    avg_rounds = sum(r.total_rounds for r in results) / len(results) if results else 0
    consensus_rate = sum(1 for r in results if r.consensus_reached) / len(results) if results else 0

    click.echo("\n" + "=" * 60)
    click.echo("[=] DATASET DEBATE RESULTS")
    click.echo("=" * 60)
    click.echo(f"[*] Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
    click.echo(f"[=] Consensus Rate: {consensus_rate:.2%}")
    click.echo(f"[~] Average Rounds: {avg_rounds:.1f}")
    click.echo(f"[T] Total Time: {total_time:.1f}s")
    click.echo(f"[=] Avg Time per Sample: {total_time/len(results):.1f}s")

    # Save if requested
    if output:
        _save_debate_results(results, output)
        click.echo(f"\n[S] Results saved to: {output}")


def _check_answer_correctness(predicted: str, ground_truth: str, task_type: str) -> bool:
    """Check if predicted answer matches ground truth."""
    if task_type == "math":
        from dte.utils.answer_extraction import clean_numeric_string
        pred_num = clean_numeric_string(predicted)
        gt_num = clean_numeric_string(ground_truth)
        if pred_num is not None and gt_num is not None:
            return abs(pred_num - gt_num) < 1e-9

    # Fallback to string comparison
    return predicted.strip().lower() == ground_truth.strip().lower()


def _save_debate_results(results, output_path: Path):
    """Save debate results to file."""
    import json
    from dataclasses import asdict

    # Convert results to serializable format
    serializable_results = []
    for result in results:
        result_dict = asdict(result)
        # Handle non-serializable objects
        for key, value in result_dict.items():
            if hasattr(value, '__dict__'):
                result_dict[key] = str(value)
        serializable_results.append(result_dict)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)


@cli.command()
@click.option(
    "--data", "-d",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Training data file (JSONL format)"
)
@click.option(
    "--epochs", "-e",
    default=3,
    help="Number of training epochs"
)
@click.option(
    "--batch-size", "-b",
    default=4,
    help="Training batch size"
)
@click.option(
    "--learning-rate", "-lr",
    default=2e-5,
    type=float,
    help="Learning rate"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    help="Output directory for trained model"
)
@click.pass_context
def train(ctx: click.Context, data: Path, epochs: int, batch_size: int,
         learning_rate: float, output_dir: Optional[Path]):
    """Train model using GRPO on debate data."""
    config = ctx.obj['config']

    # Override config with CLI parameters
    config.training.max_epochs = epochs
    config.training.batch_size = batch_size
    config.training.learning_rate = learning_rate

    if output_dir:
        config.paths.models_dir = str(output_dir)

    click.echo("[*] Starting GRPO Training")
    click.echo(f"Data: {data}")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Batch Size: {batch_size}")
    click.echo(f"Learning Rate: {learning_rate}")
    click.echo()

    try:
        from dte.data.generator import DebateDataGenerator
        from dte.training.grpo_trainer import GRPOTrainer

        # Initialize logger
        logger = DTELogger(config.logging, "training_session")

        # Load training data
        data_generator = DebateDataGenerator(config.datasets, config.debate, config.model, logger)
        training_examples = data_generator.load_generated_data(data)

        click.echo(f"[D] Loaded {len(training_examples)} training examples")

        # Initialize trainer
        trainer = GRPOTrainer(config.training, config.model, config.paths, logger)

        # Train model
        metrics = trainer.train(training_examples)

        # Display results
        click.echo("\n[OK] Training Completed!")
        click.echo(f"Final Loss: {metrics['epoch_losses'][-1]:.4f}")
        click.echo(f"Model saved to: {config.paths.models_dir}")

    except Exception as e:
        click.echo(f"[ERROR] Training failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--samples", "-n",
    default=100,
    help="Number of samples to generate"
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output file for generated data"
)
@click.option(
    "--round", "-r",
    default=0,
    help="Evolution round (affects temperature annealing)"
)
@click.pass_context
def generate(ctx: click.Context, samples: int, output: Path, round: int):
    """Generate training data from multi-agent debates."""
    config = ctx.obj['config']

    click.echo("[~] Generating Debate Data")
    click.echo(f"Samples: {samples}")
    click.echo(f"Evolution Round: {round}")
    click.echo(f"Output: {output}")
    click.echo()

    try:
        from dte.data.generator import DebateDataGenerator

        # Initialize logger
        logger = DTELogger(config.logging, "data_generation")

        # Initialize data generator
        data_generator = DebateDataGenerator(config.datasets, config.debate, config.model, logger)

        # Generate data
        examples = data_generator.generate_training_data(
            num_samples=samples,
            evolution_round=round,
            save_path=str(output)
        )

        # Display statistics
        stats = data_generator.get_generation_statistics()
        click.echo("\n[=] Generation Statistics:")
        click.echo(f"Total Examples: {stats['total_examples']}")
        click.echo(f"Consensus Rate: {stats['consensus_rate']:.2%}")
        click.echo(f"Avg Confidence: {stats['average_confidence']:.3f}")
        click.echo(f"Avg Reasoning Length: {stats['average_reasoning_length']:.1f} words")

    except Exception as e:
        click.echo(f"[ERROR] Data generation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
def validate(config_file: Path):
    """Validate a configuration file."""
    click.echo(f"[?] Validating configuration: {config_file}")

    try:
        config = DTEConfig.from_yaml(config_file)
        errors = config.validate()

        if errors:
            click.echo("[ERROR] Configuration validation failed:")
            for error in errors:
                click.echo(f"  - {error}")
            sys.exit(1)
        else:
            click.echo("[OK] Configuration is valid!")

    except Exception as e:
        click.echo(f"[ERROR] Failed to load configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx: click.Context):
    """Show configuration and system information."""
    config = ctx.obj['config']

    click.echo("DTE Framework Information")
    click.echo("=" * 40)
    click.echo(f"Experiment: {config.experiment.name}")
    click.echo(f"Base Model: {config.model.base_model_name}")
    click.echo(f"Debate Agents: {config.debate.num_agents}")
    click.echo(f"Max Debate Rounds: {config.debate.max_rounds}")
    click.echo(f"Evolution Rounds: {config.evolution.max_rounds}")
    click.echo(f"GRPO Group Size: {config.training.grpo.group_size}")
    click.echo(f"LoRA Enabled: {config.training.lora.enabled}")
    click.echo()

    # System information
    import torch
    click.echo("System Information")
    click.echo("=" * 40)
    click.echo(f"PyTorch Version: {torch.__version__}")
    click.echo(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"CUDA Devices: {torch.cuda.device_count()}")
        click.echo(f"Current Device: {torch.cuda.current_device()}")

    # Configuration paths
    click.echo()
    click.echo("Paths")
    click.echo("=" * 40)
    for name, path in config.paths.__dict__.items():
        click.echo(f"{name}: {path}")

    # Project information
    click.echo()
    click.echo("Project Links")
    click.echo("=" * 40)
    click.echo("Paper: EMNLP 2025 - Debate, Train, Evolve: Self-Evolution of Language Model Reasoning")
    click.echo("Website: https://ctrl-gaurav.github.io/debate-train-evolve.github.io/")
    click.echo("Repository: https://github.com/ctrl-gaurav/Debate-Train-Evolve")
    click.echo("Issues: https://github.com/ctrl-gaurav/Debate-Train-Evolve/issues")


if __name__ == "__main__":
    cli()