# Configuration Reference

Every configuration option in the DTE framework, with defaults, types,
and detailed descriptions.

## Table of Contents

- [Overview](#overview)
- [YAML Configuration File](#yaml-configuration-file)
- [Model Configuration](#model-configuration)
- [Debate Configuration](#debate-configuration)
- [Dataset Configuration](#dataset-configuration)
- [Training Configuration](#training-configuration)
- [Evolution Configuration](#evolution-configuration)
- [Logging Configuration](#logging-configuration)
- [Hardware Configuration](#hardware-configuration)
- [Paths Configuration](#paths-configuration)
- [Experiment Configuration](#experiment-configuration)
- [Safety Configuration](#safety-configuration)
- [Validation](#validation)

---

## Overview

DTE configuration can be specified in three ways:

1. **YAML file** (recommended for reproducible experiments):
   ```python
   config = DTEConfig.from_yaml("config.yaml")
   ```

2. **Python dictionary**:
   ```python
   config = DTEConfig.from_dict({"model": {"base_model_name": "gpt2"}, ...})
   ```

3. **Direct construction**:
   ```python
   config = DTEConfig(model=ModelConfig(base_model_name="gpt2"))
   ```

All three produce identical `DTEConfig` objects. Unspecified parameters
take their documented defaults.

---

## YAML Configuration File

Below is a complete annotated `config.yaml`:

```yaml
# === Model Configuration ===
model:
  base_model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  base_model_path: null
  device: "auto"
  max_length: 2048
  temperature: 0.7
  top_p: 0.9
  top_k: 50

# === Multi-Agent Debate Configuration ===
debate:
  num_agents: 3
  max_rounds: 3
  consensus_threshold: 1.0
  consensus_tolerance: 1.0e-9
  use_diverse_agents: false
  agent_models: []
  track_sycophancy: true
  consolidate_reasoning: true

  debate_prompting:        # (or "rcr_prompting" for backward compat)
    enabled: true
    initial_prompt_type: "math"
    include_agent_context: true
    include_peer_solutions: true
    defend_previous_answer: true
    require_novel_reasoning: true
    critique_pairs: 2

  temperature_annealing:
    enabled: true
    start_temp: 0.7
    end_temp: 0.3
    min_model_size: "3B"

# === Dataset Configuration ===
datasets:
  names: ["gsm8k"]
  max_samples_per_dataset: 1000
  quality_threshold: 0.8
  train_datasets:
    - name: "gsm8k"
      path: "openai/gsm8k"
      split: "train"
      max_samples: 1000
  eval_datasets:
    - name: "gsm8k"
      path: "openai/gsm8k"
      split: "test"
      max_samples: 500

# === GRPO Training Configuration ===
training:
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_steps: 50
  max_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4

  grpo:
    group_size: 4
    advantage_normalization: true
    clip_ratio: 0.2
    kl_penalty: 0.02

  rewards:
    correctness_weight: 2.0
    int_weight: 0.5
    strict_format_weight: 0.5
    soft_format_weight: 0.5
    xmlcount_weight: 0.5

  lora:
    enabled: true
    rank: 128
    alpha: 256
    dropout: 0.05
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"

# === Evolution Configuration ===
evolution:
  max_rounds: 3
  convergence_threshold: 0.01
  patience: 2
  samples_per_round: 500
  validation_split: 0.2
  validation_freq: 1
  min_improvement: 0.01

# === Logging Configuration ===
logging:
  level: "INFO"
  log_dir: "./logs"
  experiment_name: "dte_experiment"
  save_checkpoints: true
  checkpoint_freq: 100
  track_metrics:
    - "accuracy"
    - "debate_consensus_rate"
    - "sycophancy_rate"
    - "average_reasoning_length"
    - "training_loss"
    - "kl_divergence"

# === Hardware Configuration ===
hardware:
  device: "auto"
  mixed_precision: true
  max_memory_per_gpu: "20GB"
  gradient_checkpointing: true
  num_workers: 4
  dataloader_pin_memory: true

# === Paths Configuration ===
paths:
  output_dir: "./outputs"
  models_dir: "./models"
  data_dir: "./data"
  cache_dir: "./cache"
  temp_dir: "./tmp"

# === Experiment Configuration ===
experiment:
  name: "dte_pipeline_v1"
  description: "End-to-end DTE pipeline"
  tags: ["dte", "multi-agent", "grpo", "reasoning"]
  seed: 42
  deterministic: true
  wandb:
    enabled: false
    project: "dte-framework"
    entity: null

# === Safety Configuration ===
safety:
  filter_toxic_content: true
  max_reasoning_length: 1000
  validate_model_outputs: true
  check_format_compliance: true
  auto_backup: true
  backup_frequency: "1h"
```

---

## Model Configuration

| Parameter          | Type            | Default                           | Description |
|--------------------|-----------------|-----------------------------------|-------------|
| `base_model_name`  | `str`           | `"Qwen/Qwen2.5-1.5B-Instruct"`   | HuggingFace model identifier or local path. Must match the `org/model` pattern for HF models. |
| `base_model_path`  | `Optional[str]` | `None`                            | Explicit local path to model weights. If set, overrides `base_model_name` for loading. |
| `device`           | `str`           | `"auto"`                          | Computation device. `"auto"` detects CUDA > MPS > CPU. Also accepts `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`. |
| `max_length`       | `int`           | `2048`                            | Maximum sequence length for tokenization. Higher values allow longer reasoning but consume more memory. Minimum recommended: 512. |
| `temperature`      | `float`         | `0.7`                             | Sampling temperature. Range: 0.0-2.0. Lower values produce more deterministic outputs. |
| `top_p`            | `float`         | `0.9`                             | Nucleus sampling probability. Range: 0.0-1.0. |
| `top_k`            | `int`           | `50`                              | Top-k sampling. Must be positive. |

---

## Debate Configuration

| Parameter               | Type    | Default | Description |
|-------------------------|---------|---------|-------------|
| `num_agents`            | `int`   | `3`     | Number of agents participating in each debate. Minimum: 2. The paper uses 3. |
| `max_rounds`            | `int`   | `3`     | Maximum number of debate rounds (not counting round 0). |
| `consensus_threshold`   | `float` | `1.0`   | Legacy parameter. All agents must agree for consensus (threshold=1.0 means unanimous). |
| `consensus_tolerance`   | `float` | `1e-9`  | Numerical tolerance for comparing answers. Two numbers within this tolerance are considered equal. |
| `use_diverse_agents`    | `bool`  | `False` | If True, use different models for different agents (specified in `agent_models`). |
| `agent_models`          | `List[str]` | `[]` | List of model names when `use_diverse_agents` is True. Must have length equal to `num_agents`. |
| `track_sycophancy`      | `bool`  | `True`  | Whether to detect and track sycophantic behavior. |
| `consolidate_reasoning` | `bool`  | `True`  | Whether to produce a consolidated reasoning trace from debate. |

### Debate Prompting (RCR)

| Parameter                 | Type   | Default  | Description |
|---------------------------|--------|----------|-------------|
| `enabled`                 | `bool` | `True`   | Whether to use RCR prompting. This is the core DTE innovation and should almost always be enabled. |
| `initial_prompt_type`     | `str`  | `"math"` | Task type for initial prompts: `"math"`, `"arc"`, or `"general"`. |
| `include_agent_context`   | `bool` | `True`   | Include agent's own previous response in debate prompts. |
| `include_peer_solutions`  | `bool` | `True`   | Include peer solutions in debate prompts. |
| `defend_previous_answer`  | `bool` | `True`   | Encourage agents to defend correct answers rather than blindly switch. |
| `require_novel_reasoning` | `bool` | `True`   | Require agents to introduce novel reasoning when refining. |
| `critique_pairs`          | `int`  | `2`      | Number of peer solutions each agent must critique per round. Paper default: 2. |

### Temperature Annealing

| Parameter        | Type   | Default | Description |
|------------------|--------|---------|-------------|
| `enabled`        | `bool` | `True`  | Whether to anneal temperature across evolution rounds. |
| `start_temp`     | `float`| `0.7`   | Starting temperature (round 0). |
| `end_temp`       | `float`| `0.3`   | Final temperature (last round). |
| `min_model_size` | `str`  | `"3B"`  | Only apply annealing to models smaller than this size (in billions of params). |

---

## Dataset Configuration

| Parameter                  | Type        | Default     | Description |
|----------------------------|-------------|-------------|-------------|
| `names`                    | `List[str]` | `["gsm8k"]` | Dataset names for validation. Must be from the 7 supported: `gsm8k`, `gsm_plus`, `math`, `arc_challenge`, `arc_easy`, `gpqa`, `commonsense_qa`. |
| `max_samples_per_dataset`  | `int`       | `1000`      | Maximum samples to load per dataset. |
| `quality_threshold`        | `float`     | `0.8`       | Minimum quality score for generated data. Range: 0.0-1.0. |
| `train_datasets`           | `List[DatasetInfo]` | `[]` | List of training dataset configurations. |
| `eval_datasets`            | `List[DatasetInfo]` | `[]` | List of evaluation dataset configurations. |

### DatasetInfo

| Parameter     | Type  | Description |
|---------------|-------|-------------|
| `name`        | `str` | Identifier for the dataset |
| `path`        | `str` | HuggingFace dataset path or local path |
| `split`       | `str` | Dataset split (train, test, validation) |
| `max_samples` | `int` | Maximum number of samples to use |

---

## Training Configuration

| Parameter                     | Type    | Default | Description |
|-------------------------------|---------|---------|-------------|
| `learning_rate`               | `float` | `2e-5`  | Peak learning rate for the optimizer. Recommended range: 1e-6 to 1e-3. |
| `weight_decay`                | `float` | `0.01`  | L2 regularization weight. Must be non-negative. |
| `warmup_steps`                | `int`   | `50`    | Number of linear warmup steps for the learning rate scheduler. |
| `max_epochs`                  | `int`   | `3`     | Maximum training epochs per evolution round. |
| `batch_size`                  | `int`   | `4`     | Training batch size. Larger values are faster but use more memory. |
| `gradient_accumulation_steps` | `int`   | `4`     | Effective batch = batch_size * gradient_accumulation_steps = 16 by default. |

### GRPO Parameters

| Parameter                  | Type    | Default | Description |
|----------------------------|---------|---------|-------------|
| `group_size`               | `int`   | `4`     | Number of sampled responses per query for group comparison. Minimum recommended: 4 for stable training. |
| `advantage_normalization`  | `bool`  | `True`  | Whether to z-score normalize advantages within a group. |
| `clip_ratio`               | `float` | `0.2`   | PPO clip ratio for the surrogate objective. Range: 0.0-1.0. |
| `kl_penalty`               | `float` | `0.02`  | KL divergence penalty against the frozen reference model. Higher values encourage staying close to the reference. |

### Reward Function Weights

| Parameter              | Type    | Default | Max per-response | Description |
|------------------------|---------|---------|------------------|-------------|
| `correctness_weight`   | `float` | `2.0`   | `2.0 * 2.0 = 4.0`| Weight for correctness reward (+2.0 for correct answer). |
| `int_weight`           | `float` | `0.5`   | `0.5 * 0.5 = 0.25`| Weight for numeric answer reward (+0.5 for numeric). |
| `strict_format_weight` | `float` | `0.5`   | `0.5 * 0.5 = 0.25`| Weight for strict XML format (+0.5 for exact format). |
| `soft_format_weight`   | `float` | `0.5`   | `0.5 * 0.5 = 0.25`| Weight for flexible XML format (+0.5 for basic XML). |
| `xmlcount_weight`      | `float` | `0.5`   | `0.5 * 0.5 = 0.25`| Weight for granular XML tag scoring (up to +0.5). |

With default weights and a perfect response, the maximum combined reward
is `4.0 + 0.25 + 0.25 + 0.25 + 0.25 = 5.0`.

#### Legacy Reward Weights (backward compatibility)

These fields exist for backward compatibility and are **not** used by the
DTE-specific reward functions. They may be referenced by custom reward
implementations.

| Parameter        | Type    | Default | Description |
|------------------|---------|---------|-------------|
| `answer_weight`  | `float` | `2.0`   | Legacy: generic answer correctness weight. |
| `format_weight`  | `float` | `0.5`   | Legacy: generic format compliance weight. |
| `length_weight`  | `float` | `0.5`   | Legacy: response length reward weight. |
| `length_tau`     | `int`   | `120`   | Legacy: target response length in tokens. |

### LoRA Configuration

| Parameter        | Type        | Default                                                 | Description |
|------------------|-------------|---------------------------------------------------------|-------------|
| `enabled`        | `bool`      | `True`                                                  | Whether to use LoRA for parameter-efficient fine-tuning. |
| `rank`           | `int`       | `128`                                                   | LoRA rank. Higher = more capacity but more parameters. |
| `alpha`          | `int`       | `256`                                                   | LoRA alpha (scaling factor). Typical: 2 * rank. |
| `dropout`        | `float`     | `0.05`                                                  | Dropout probability for LoRA layers. Range: 0.0-1.0. |
| `target_modules` | `List[str]` | `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` | Which modules to apply LoRA to. |

---

## Evolution Configuration

| Parameter               | Type    | Default | Description |
|-------------------------|---------|---------|-------------|
| `max_rounds`            | `int`   | `3`     | Maximum number of evolution rounds. Each round runs debate + train + eval. |
| `convergence_threshold` | `float` | `0.01`  | Stop if improvement falls below this threshold. |
| `patience`              | `int`   | `2`     | Stop after this many rounds without improvement. |
| `samples_per_round`     | `int`   | `500`   | Number of debate samples to generate per round. |
| `validation_split`      | `float` | `0.2`   | Fraction of generated data reserved for validation. |
| `validation_freq`       | `int`   | `1`     | Validate every N epochs during training. |
| `min_improvement`       | `float` | `0.01`  | Minimum improvement to reset patience counter. |

---

## Logging Configuration

| Parameter          | Type        | Default            | Description |
|--------------------|-------------|--------------------|-------------|
| `level`            | `str`       | `"INFO"`           | Logging level: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`. |
| `log_dir`          | `str`       | `"./logs"`         | Directory for log files. Created automatically. |
| `experiment_name`  | `str`       | `"dte_experiment"` | Name used for log file naming. |
| `save_checkpoints` | `bool`      | `True`             | Whether to save model checkpoints during training. |
| `checkpoint_freq`  | `int`       | `100`              | Save checkpoint every N training steps. |
| `track_metrics`    | `List[str]` | see below          | Metrics to track during pipeline execution. |

Default tracked metrics: `accuracy`, `debate_consensus_rate`, `sycophancy_rate`,
`average_reasoning_length`, `training_loss`, `kl_divergence`.

---

## Hardware Configuration

| Parameter                | Type   | Default  | Description |
|--------------------------|--------|----------|-------------|
| `device`                 | `str`  | `"auto"` | Same device spec as model config. |
| `mixed_precision`        | `bool` | `True`   | Use bfloat16/float16 mixed precision on GPU. |
| `max_memory_per_gpu`     | `str`  | `"20GB"` | Maximum memory allocation per GPU. |
| `gradient_checkpointing` | `bool` | `True`   | Trade compute for memory during training. |
| `num_workers`            | `int`  | `4`      | DataLoader workers for parallel data loading. |
| `dataloader_pin_memory`  | `bool` | `True`   | Pin memory for faster CPU-to-GPU transfer. |

---

## Paths Configuration

| Parameter    | Type  | Default       | Description |
|--------------|-------|---------------|-------------|
| `output_dir` | `str` | `"./outputs"` | Directory for pipeline outputs and results. |
| `models_dir` | `str` | `"./models"`  | Directory for model checkpoints. |
| `data_dir`   | `str` | `"./data"`    | Directory for generated training data. |
| `cache_dir`  | `str` | `"./cache"`   | Directory for HuggingFace model/dataset caches. |
| `temp_dir`   | `str` | `"./tmp"`     | Temporary storage directory. |

---

## Experiment Configuration

| Parameter       | Type            | Default                                          | Description |
|-----------------|-----------------|--------------------------------------------------|-------------|
| `name`          | `str`           | `"dte_pipeline_v1"`                              | Experiment name for tracking and log files. |
| `description`   | `str`           | `"End-to-end DTE pipeline with RCR prompting and GRPO training"` | Human-readable description. |
| `tags`          | `List[str]`     | `["dte", "multi-agent", "grpo", "reasoning"]`    | Tags for organization and filtering. |
| `seed`          | `int`           | `42`                                             | Global random seed for reproducibility. |
| `deterministic` | `bool`          | `True`                                           | Enable deterministic mode (sets CUDA deterministic). |

### W&B Configuration

| Parameter  | Type            | Default           | Description |
|------------|-----------------|-------------------|-------------|
| `enabled`  | `bool`          | `False`           | Enable Weights & Biases logging. |
| `project`  | `str`           | `"dte-framework"` | W&B project name. |
| `entity`   | `Optional[str]` | `None`            | W&B entity (team or username). |

---

## Safety Configuration

| Parameter                | Type   | Default | Description |
|--------------------------|--------|---------|-------------|
| `filter_toxic_content`   | `bool` | `True`  | Filter out potentially toxic model outputs. |
| `max_reasoning_length`   | `int`  | `1000`  | Maximum word count for reasoning traces. |
| `validate_model_outputs` | `bool` | `True`  | Validate that model outputs conform to expected format. |
| `check_format_compliance`| `bool` | `True`  | Check XML format compliance of training data. |
| `auto_backup`            | `bool` | `True`  | Automatically backup checkpoints. |
| `backup_frequency`       | `str`  | `"1h"`  | How often to create backups. |

---

## Validation

The configuration system provides comprehensive validation:

```python
config = DTEConfig.from_yaml("config.yaml")

# Basic validation (returns list of error strings)
errors = config.validate()
if errors:
    for e in errors:
        print(f"  - {e}")

# Strict validation (additional performance warnings)
errors = config.validate_strict()

# Raise on error
config.validate_and_raise()  # raises ConfigurationError
```

Validation checks include:

- **Model**: name format, temperature range (0-2), top_p range (0-1), positive top_k
- **Debate**: num_agents >= 2, max_rounds > 0, consensus threshold (0-1)
- **Training**: positive learning rate (strict: 1e-6 to 1e-3), positive batch_size, clip_ratio (0-1)
- **Datasets**: names are in the 7 supported datasets
- **Evolution**: positive max_rounds, positive samples_per_round
- **Paths**: directories can be created
