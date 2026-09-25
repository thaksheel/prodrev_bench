#!/usr/bin/env python3
"""
Full Pipeline -- Run the complete Debate, Train, Evolve pipeline.

This example shows how to create a DTEPipeline from a config file and run
the iterative evolution process end-to-end.

Usage:
    python examples/full_pipeline.py [config.yaml]
"""

import sys
from pathlib import Path

import dte
from dte.core.config import DTEConfig


def main():
    # Resolve config path from CLI or default
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Use the project default config
        project_root = Path(__file__).resolve().parent.parent
        config_path = str(project_root / "config.yaml")

    print(f"Loading configuration from {config_path} ...")
    config = DTEConfig.from_yaml(config_path)

    # Validate before running
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"Experiment : {config.experiment.name}")
    print(f"Model      : {config.model.base_model_name}")
    print(f"Agents     : {config.debate.num_agents}")
    print(f"Evo rounds : {config.evolution.max_rounds}")
    print()

    # Create and run the pipeline
    pipeline = dte.DTEPipeline(config)

    try:
        results = pipeline.run_complete_pipeline()

        # Display summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Total time        : {results['total_time_hours']:.2f} hours")
        print(f"Evolution rounds  : {results['total_evolution_rounds']}")
        print(f"Best performance  : {results['best_performance']:.4f}")
        print(f"Total improvement : {results['total_improvement']:.4f}")
        print(f"Converged         : {results['convergence_achieved']}")
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
