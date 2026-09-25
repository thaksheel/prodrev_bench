# DTE Framework Documentation

Welcome to the documentation for the **DTE (Debate, Train, Evolve)** framework
-- a complete implementation for improving language model reasoning through
multi-agent debate and iterative GRPO training.

**Paper**: [Debate, Train, Evolve: Self-Evolution of Language Model Reasoning](https://aclanthology.org/2025.emnlp-main.1666/) (EMNLP 2025 Main Conference)

---

## Framework at a Glance

```
                    ╔══════════════════════════════════════════════════════╗
                    ║            DTE  EVOLUTION  LOOP                     ║
                    ║                                                     ║
                    ║   ┌───────────┐   ┌──────────┐   ┌──────────────┐  ║
                    ║   │  DEBATE   │──▶│  TRAIN   │──▶│   EVALUATE   │  ║
                    ║   │  (RCR)    │   │  (GRPO)  │   │  (Benchmark) │  ║
                    ║   └───────────┘   └──────────┘   └──────┬───────┘  ║
                    ║        ▲                                 │          ║
                    ║        │         next evolution          │          ║
                    ║        └─────────── round ◀──────────────┘          ║
                    ║                                                     ║
                    ╚══════════════════════════════════════════════════════╝

    ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
    │   DEBATE PHASE      │  │   TRAINING PHASE    │  │   EVALUATION PHASE  │
    ├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
    │ Multi-agent RCR     │  │ GRPO fine-tuning    │  │ Benchmark accuracy  │
    │ debates generate    │  │ with 5 reward       │  │ + consensus rate    │
    │ high-quality        │  │ functions on        │  │ + sycophancy        │
    │ reasoning traces    │  │ debate traces       │  │ + debate dynamics   │
    └─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

1. **Debate**: Multiple agents solve problems through structured
   multi-round RCR (Reflect-Critique-Refine) prompting. High-quality
   debate traces are collected as training data.
2. **Train**: The base model is fine-tuned using GRPO (Group Relative
   Policy Optimization) on the debate-generated data, with 5 specialized
   reward functions providing the training signal.
3. **Evolve**: Benchmark evaluation determines whether to continue iterating.

---

## Quick Navigation

| Section | What you will find |
|---------|--------------------|
| [Getting Started](getting_started.md) | Installation, prerequisites, 5-minute quickstart, verify installation |
| [Architecture Guide](architecture.md) | Package structure, pipeline lifecycle, data flow, component deep dives |
| [Configuration Reference](configuration.md) | Every config option with type, default, description, and annotated YAML |
| [Training Guide](training_guide.md) | Step-by-step GRPO training walkthrough, reward functions, LoRA, recipes |
| [Evaluation Guide](evaluation.md) | Benchmark evaluation, supported datasets, metrics interpretation |
| [API Reference](api_reference.md) | Complete signatures for every public class, function, and dataclass |
| [Examples Walkthrough](examples.md) | Detailed explanation of every script in the `examples/` directory |
| [Troubleshooting](troubleshooting.md) | Common issues: installation, GPU OOM, config errors, training problems |

---

## Suggested Reading Order

| #  | Document | Description |
|----|----------|-------------|
| 1  | [Getting Started](getting_started.md) | Installation, prerequisites, your first debate, understanding output |
| 2  | [Architecture Guide](architecture.md) | Package structure, pipeline lifecycle, data flow, component deep dives |
| 3  | [Configuration Reference](configuration.md) | Every config option with type, default, description, and annotated YAML |
| 4  | [Training Guide](training_guide.md) | Step-by-step GRPO training walkthrough, reward functions, LoRA, recipes |
| 5  | [Evaluation Guide](evaluation.md) | Benchmark evaluation, supported datasets, metrics interpretation |
| 6  | [API Reference](api_reference.md) | Complete signatures for every public class, function, and dataclass |
| 7  | [Examples Walkthrough](examples.md) | Detailed explanation of every script in the `examples/` directory |
| 8  | [Troubleshooting](troubleshooting.md) | Common issues: installation, GPU OOM, config errors, training problems |

---

## Quick Links

- **One-line debate**: `dte.debate("What is 15 * 24?")` -- see [Getting Started](getting_started.md#quick-start)
- **Full pipeline**: `dte.from_config("config.yaml").run_complete_pipeline()` -- see [Training Guide](training_guide.md#step-4-run-full-evolution-loop)
- **CLI usage**: `python main.py debate --query "..."` -- see [README](../README.md)
- **All config fields**: [Configuration Reference](configuration.md)
- **Supported datasets**: gsm8k, gsm_plus, math, arc_challenge, arc_easy, gpqa, commonsense_qa -- see [Evaluation Guide](evaluation.md#supported-benchmarks)

---

## What's New in v0.1

**v0.1.0** is the first public release accompanying the EMNLP 2025 paper.

- **Complete DTE pipeline**: Full Debate-Train-Evolve loop with automatic convergence detection and multi-round evolution.
- **RCR prompting system**: Three-phase Reflect-Critique-Refine debate prompting that reduces sycophancy by 50% compared to naive multi-agent debate.
- **GRPO training**: Group Relative Policy Optimization with 5 complementary reward functions (correctness, integer, strict format, soft format, XML count).
- **7 benchmark datasets**: GSM8K, GSM-Plus, MATH, ARC-Challenge, ARC-Easy, GPQA, CommonsenseQA.
- **LoRA support**: Parameter-efficient fine-tuning with configurable rank, alpha, dropout, and target modules.
- **Model weight sharing**: Thread-safe registry ensures multiple debate agents share a single model copy in GPU memory.
- **Temperature annealing**: Automatic temperature scheduling across evolution rounds for smaller models.
- **Sycophancy detection**: Per-round tracking of agents that abandon correct answers to match peers.
- **Rich CLI**: Full command-line interface for debate, training, evaluation, data generation, and configuration management.
- **W&B integration**: Optional Weights & Biases experiment tracking.
- **Production-quality code**: Comprehensive error handling, typed configuration, structured logging, and 124 unit tests + 11 GPU integration tests.

---

## Package Overview

```
dte/
    __init__.py          # Top-level API: debate(), train(), evaluate(), from_config()
    core/
        config.py        # 12 configuration dataclasses + YAML I/O + validation
        pipeline.py      # DTEPipeline: end-to-end orchestrator
        evaluator.py     # DTEEvaluator: benchmark evaluation
        logger.py        # DTELogger: structured logging + Rich console
    debate/
        agent.py         # DebateAgent + shared model registry
        manager.py       # DebateManager: debate orchestration
        prompts.py       # RCR prompt generation + response parsing
    training/
        grpo_trainer.py  # GRPOTrainer: full GRPO implementation
        reward_model.py  # DTERewardModel: all 5 reward functions
    data/
        dataset_manager.py  # DatasetManager: 7 benchmark loaders
        generator.py        # DebateDataGenerator: debate -> training data
        processor.py        # DataProcessor: validation + formatting
    utils/
        answer_extraction.py  # Answer extraction, consensus, sycophancy
        data_utils.py         # JSONL I/O, splitting, filtering
        helpers.py            # Error classes, device utils, timing
```

---

## Version

Current version: **0.1.0**

Paper: [*Debate, Train, Evolve: Self-Evolution of Language Model Reasoning*](https://aclanthology.org/2025.emnlp-main.1666/)
(EMNLP 2025 Main Conference)

Authors: Gaurav Srivastava, Zhenyu Bi, Meng Lu, Xuan Wang
