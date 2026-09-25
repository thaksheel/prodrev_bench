#!/usr/bin/env python3
"""
Multi-GPU Training -- Shows how to configure DTE for multi-GPU setups.

This example demonstrates:
- Setting CUDA_VISIBLE_DEVICES to control GPU allocation
- Configuring the DTE pipeline for distributed training
- Running the full pipeline with proper GPU management

Usage:
    CUDA_VISIBLE_DEVICES=0,1,3,4 python examples/multi_gpu_training.py

Note:
    The DTE framework uses HuggingFace ``device_map="auto"`` which
    automatically shards large models across available GPUs. For the
    debate phase, agents share a single model instance to save memory.
"""

import os
import sys
from pathlib import Path

from dte.core.config import (
    DTEConfig,
    ModelConfig,
    DebateConfig,
    TrainingConfig,
    GRPOConfig,
    LoRAConfig,
    RewardsConfig,
    EvolutionConfig,
    HardwareConfig,
    PathsConfig,
    LoggingConfig,
    ExperimentConfig,
    DatasetsConfig,
    DatasetInfo,
)


def main():
    # Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("No GPUs detected. This example requires CUDA GPUs.")
            sys.exit(1)
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPU(s):")
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")
    except ImportError:
        print("PyTorch not installed.")
        sys.exit(1)

    # Build configuration for multi-GPU training
    config = DTEConfig(
        model=ModelConfig(
            base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            device="auto",  # Will use device_map="auto" for multi-GPU
            max_length=2048,
            temperature=0.7,
        ),
        debate=DebateConfig(
            num_agents=3,
            max_rounds=3,
        ),
        datasets=DatasetsConfig(
            names=["gsm8k"],
            max_samples_per_dataset=100,
            train_datasets=[
                DatasetInfo(
                    name="gsm8k",
                    path="openai/gsm8k",
                    split="train",
                    max_samples=100,
                ),
            ],
            eval_datasets=[
                DatasetInfo(
                    name="gsm8k",
                    path="openai/gsm8k",
                    split="test",
                    max_samples=50,
                ),
            ],
        ),
        training=TrainingConfig(
            learning_rate=2e-5,
            max_epochs=1,
            batch_size=4,
            gradient_accumulation_steps=4,
            grpo=GRPOConfig(group_size=4, clip_ratio=0.2, kl_penalty=0.02),
            rewards=RewardsConfig(),
            lora=LoRAConfig(enabled=True, rank=64, alpha=128),
        ),
        evolution=EvolutionConfig(
            max_rounds=1,  # Just 1 round for demo
            samples_per_round=50,
        ),
        hardware=HardwareConfig(
            device="auto",
            mixed_precision=True,
            gradient_checkpointing=True,
        ),
        paths=PathsConfig(
            output_dir="./outputs/multi_gpu",
            models_dir="./models/multi_gpu",
            data_dir="./data/multi_gpu",
            cache_dir="./cache",
            temp_dir="./tmp",
        ),
        logging=LoggingConfig(
            level="INFO",
            log_dir="./logs/multi_gpu",
            experiment_name="multi_gpu_demo",
        ),
        experiment=ExperimentConfig(
            name="multi_gpu_demo",
            seed=42,
        ),
    )

    # Validate
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("\nConfiguration validated successfully.")
    print(f"Model       : {config.model.base_model_name}")
    print(f"LoRA rank   : {config.training.lora.rank}")
    print(f"Batch size  : {config.training.batch_size}")
    print(f"Grad accum  : {config.training.gradient_accumulation_steps}")
    print(f"Evo rounds  : {config.evolution.max_rounds}")

    # Save the config for reference
    config_path = Path("./outputs/multi_gpu/config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save_yaml(config_path)
    print(f"\nConfig saved to {config_path}")

    print("\nTo run the full pipeline:")
    print(f"  CUDA_VISIBLE_DEVICES=0,1,3,4 python -c \"")
    print(f"    import dte")
    print(f"    pipeline = dte.from_config('{config_path}')")
    print(f"    results = pipeline.run_complete_pipeline()\"")


if __name__ == "__main__":
    main()
