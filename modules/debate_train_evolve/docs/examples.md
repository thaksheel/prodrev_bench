# Examples Walkthrough

Detailed explanation of every example script in the `examples/` directory.

## Table of Contents

- [Overview](#overview)
- [quick_start.py](#quick_startpy)
- [custom_debate.py](#custom_debatepy)
- [full_pipeline.py](#full_pipelinepy)
- [evaluation_example.py](#evaluation_examplepy)
- [reward_functions.py](#reward_functionspy)
- [multi_gpu_training.py](#multi_gpu_trainingpy)

---

## Overview

All examples can be run from the project root:

```bash
cd Debate-Train-Evolve
python examples/<script_name>.py
```

| Script                   | Requires GPU | Purpose |
|--------------------------|-------------|---------|
| `quick_start.py`         | Yes         | Minimal 3-agent debate |
| `custom_debate.py`       | Yes         | Custom debate configuration |
| `full_pipeline.py`       | Yes         | Full Debate-Train-Evolve pipeline |
| `evaluation_example.py`  | Yes         | Benchmark evaluation |
| `reward_functions.py`    | No          | Demonstrate all 5 reward functions |
| `multi_gpu_training.py`  | Yes         | Multi-GPU setup and configuration |

---

## quick_start.py

**File**: `examples/quick_start.py`

**Purpose**: The absolute simplest way to use DTE -- one function call.

**What it does**:

1. Calls `dte.debate()` with a math question ("What is 15 * 24?")
2. Uses the smallest Qwen model (`Qwen2.5-0.5B-Instruct`)
3. Runs a 3-agent, 3-round debate
4. Prints the final answer, consensus status, and timing

**Key code**:

```python
import dte

result = dte.debate(
    query="What is 15 * 24?",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    num_agents=3,
    max_rounds=3,
    task_type="math",
)

print(f"Final answer : {result.final_answer}")
print(f"Consensus    : {result.consensus_reached}")
```

**Expected output**:

```
Running a quick 3-agent debate ...

Final answer : 360
Consensus    : True
Total rounds : 1
Time         : 45.23s

Answer progression per round:
  Round 0: ['360', '360', '360']
```

**Notes**:
- The `dte.debate()` function handles all setup and teardown automatically
- GPU memory is freed when the function returns
- If agents agree in round 0, no debate rounds are needed

---

## custom_debate.py

**File**: `examples/custom_debate.py`

**Purpose**: Shows how to configure debates with non-default settings
and run multiple queries.

**What it does**:

1. Creates `ModelConfig` with lower temperature (0.5) and shorter max_length (1024)
2. Creates `DebateConfig` with 4 rounds instead of the default 3
3. Initializes a `DebateManager`
4. Runs debates on two different math queries
5. Shows per-debate results and aggregate statistics
6. Cleans up resources

**Key learnings**:
- `DebateManager` is reusable -- you can run multiple debates with the same agents
- The manager tracks statistics across all debates via `get_debate_statistics()`
- Always call `manager.cleanup()` to free GPU memory

**Key code**:

```python
manager = DebateManager(debate_config, model_config)

for query, task_type in queries:
    result = manager.conduct_debate(query, task_type)
    print(f"Answer: {result.final_answer}")

stats = manager.get_debate_statistics()
print(f"Consensus rate: {stats['consensus_rate']:.2%}")

manager.cleanup()
```

---

## full_pipeline.py

**File**: `examples/full_pipeline.py`

**Purpose**: Run the complete Debate-Train-Evolve pipeline from a config file.

**What it does**:

1. Loads a config file (from CLI argument or the project default `config.yaml`)
2. Validates the configuration
3. Creates a `DTEPipeline` and runs `run_complete_pipeline()`
4. Displays summary results (total time, evolution rounds, best performance)

**Key learnings**:
- The pipeline manages the full evolution loop automatically
- Configuration validation catches errors before expensive GPU operations
- `run_complete_pipeline()` returns a comprehensive results dictionary

**Usage**:

```bash
# Use default config.yaml
python examples/full_pipeline.py

# Use custom config
python examples/full_pipeline.py my_config.yaml
```

**Key code**:

```python
config = DTEConfig.from_yaml(config_path)
errors = config.validate()
if errors:
    print("Configuration errors:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)

pipeline = dte.DTEPipeline(config)
results = pipeline.run_complete_pipeline()

print(f"Total time: {results['total_time_hours']:.2f} hours")
print(f"Best performance: {results['best_performance']:.4f}")
```

---

## evaluation_example.py

**File**: `examples/evaluation_example.py`

**Purpose**: Evaluate a model on standard benchmarks with full metrics.

**What it does**:

1. Configures a small model for quick evaluation
2. Creates a `DTEEvaluator` with GSM8K dataset (10 samples for demo)
3. Runs evaluation and displays comprehensive metrics
4. Shows per-dataset breakdown

**Key learnings**:
- `DTEEvaluator` is separate from the pipeline -- you can use it standalone
- The `EvaluationMetrics` dataclass has many fields for detailed analysis
- The evaluator creates debate manager instances internally

**Key code**:

```python
from dte.core.config import ModelConfig, DebateConfig, DatasetsConfig, LoggingConfig
from dte.core.logger import DTELogger
from dte.core.evaluator import DTEEvaluator

model_config = ModelConfig(
    base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    device="auto",
    max_length=512,
)

datasets_config = DatasetsConfig(
    names=["gsm8k"],
    max_samples_per_dataset=10,
)

logger = DTELogger(LoggingConfig(level="INFO", log_dir="./logs"), "evaluation_demo")
evaluator = DTEEvaluator(datasets_config, debate_config, model_config, logger)

try:
    metrics = evaluator.evaluate_model(evolution_round=0, max_samples_per_dataset=10)

    print(f"Overall accuracy: {metrics.overall_accuracy:.2%}")
    print(f"Consensus rate:   {metrics.consensus_rate:.2%}")
    print(f"Debate helped:    {metrics.debate_helped_rate:.2%}")
    print(f"Sycophancy rate:  {metrics.sycophancy_rate:.2%}")

    report = evaluator.create_evaluation_report(metrics, evolution_round=0)
finally:
    evaluator.cleanup()
```

**Available metrics fields**:

The `EvaluationMetrics` dataclass returned by `evaluate_model()` includes:
`overall_accuracy`, `total_samples`, `correct_samples`, `consensus_rate`,
`debate_helped_rate`, `sycophancy_rate`, `correct_to_incorrect_rate`,
`incorrect_to_correct_rate`, `average_debate_rounds`,
`average_reasoning_length`, `evaluation_time`, and `per_dataset_metrics`.

---

## reward_functions.py

**File**: `examples/reward_functions.py`

**Purpose**: Demonstrate all 5 DTE reward functions and how they combine.
This is the only example that does NOT require a GPU.

**What it does**:

1. Creates a `DTERewardModel`
2. Defines 4 test responses with varying quality:
   - Perfect: correct answer + strict XML format
   - Sloppy format: correct but without newlines
   - Wrong answer: good format but incorrect
   - No XML: plain text answer
3. Calculates all 5 rewards for each response
4. Shows the combined weighted sum
5. Displays reward statistics

**Key learnings**:
- Rewards are combined as a **weighted sum**, not an average
- The correctness reward (+2.0) dominates the signal
- Format rewards incentivize structured output
- The XML count reward provides granular feedback even when format is imperfect

**Expected output** (abbreviated):

```
--- Response 1: Perfect ---
  correctness         : 2.000
  int                 : 0.500
  strict_format       : 0.500
  soft_format         : 0.500
  xmlcount            : 0.500
  COMBINED (weighted sum): 5.000

--- Response 2: Sloppy format ---
  correctness         : 2.000   <-- still correct
  int                 : 0.500
  strict_format       : 0.000   <-- missing newlines
  soft_format         : 0.500   <-- flexible format passes
  xmlcount            : 0.500
  COMBINED (weighted sum): 4.750

--- Response 3: Wrong answer ---
  correctness         : 0.000   <-- incorrect
  int                 : 0.500
  strict_format       : 0.500
  soft_format         : 0.500
  xmlcount            : 0.500
  COMBINED (weighted sum): 1.000

--- Response 4: No XML ---
  correctness         : 0.000
  int                 : 0.000
  strict_format       : 0.000
  soft_format         : 0.000
  xmlcount            : 0.000
  COMBINED (weighted sum): 0.000
```

This demonstrates the reward hierarchy: correctness dominates (4.0 of the
5.0 max), format rewards provide a secondary signal, and the absence of XML
structure results in zero across all format-related functions.

---

## multi_gpu_training.py

**File**: `examples/multi_gpu_training.py`

**Purpose**: Show how to configure DTE for multi-GPU setups.

**What it does**:

1. Checks GPU availability and prints device info
2. Builds a complete `DTEConfig` programmatically (without YAML)
3. Validates the configuration
4. Saves the config to YAML for reference
5. Prints instructions for running the full pipeline

**Key learnings**:
- DTE uses `device_map="auto"` for automatic multi-GPU sharding
- Control GPU selection with `CUDA_VISIBLE_DEVICES`
- All config components can be constructed in Python
- `DatasetInfo` objects specify train/eval dataset splits

**Usage**:

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 python examples/multi_gpu_training.py
```

**Key code**:

```python
config = DTEConfig(
    model=ModelConfig(
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="auto",
    ),
    datasets=DatasetsConfig(
        train_datasets=[
            DatasetInfo(name="gsm8k", path="openai/gsm8k",
                        split="train", max_samples=100),
        ],
    ),
    training=TrainingConfig(
        lora=LoRAConfig(enabled=True, rank=64, alpha=128),
    ),
    hardware=HardwareConfig(
        device="auto",
        mixed_precision=True,
        gradient_checkpointing=True,
    ),
)

errors = config.validate()
config.save_yaml("./outputs/multi_gpu/config.yaml")
```

**Notes**:
- This example does not actually run training -- it sets up and validates
  the configuration
- To run training, use the printed instructions or pass the saved config
  to `dte.from_config()`
