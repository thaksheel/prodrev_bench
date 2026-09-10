# API Reference

Complete reference for all public classes and functions in the DTE framework.

## Table of Contents

- [Top-Level Functions](#top-level-functions)
- [Configuration Classes](#configuration-classes)
- [Debate Module](#debate-module)
- [Training Module](#training-module)
- [Data Module](#data-module)
- [Utility Functions](#utility-functions)

---

## Top-Level Functions

These functions are available directly on the `dte` module after `import dte`.

### `dte.debate()`

```python
dte.debate(
    query: str,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_agents: int = 3,
    max_rounds: int = 3,
    task_type: str = "math",
    device: str = "auto",
    temperature: float = 0.7,
    verbose: bool = False,
) -> DebateResult
```

Run a multi-agent debate on a single query. This is the simplest entry point
to the framework.

**Parameters**:

| Parameter     | Type   | Default                           | Description                          |
|---------------|--------|-----------------------------------|--------------------------------------|
| `query`       | `str`  | (required)                        | The question or problem to debate    |
| `model`       | `str`  | `"Qwen/Qwen2.5-1.5B-Instruct"`   | HuggingFace model name or path       |
| `num_agents`  | `int`  | `3`                               | Number of debate agents              |
| `max_rounds`  | `int`  | `3`                               | Maximum debate rounds                |
| `task_type`   | `str`  | `"math"`                          | `"math"`, `"arc"`, or `"general"`    |
| `device`      | `str`  | `"auto"`                          | `"auto"`, `"cpu"`, `"cuda"`, etc.    |
| `temperature` | `float`| `0.7`                             | Sampling temperature                 |
| `verbose`     | `bool` | `False`                           | Print progress to console            |

**Returns**: `DebateResult`

**Raises**: `RuntimeError` if ML dependencies are not installed.

---

### `dte.from_config()`

```python
dte.from_config(config_path: str) -> DTEPipeline
```

Create a `DTEPipeline` from a YAML configuration file.

**Parameters**:

| Parameter     | Type  | Description               |
|---------------|-------|---------------------------|
| `config_path` | `str` | Path to YAML config file  |

**Returns**: `DTEPipeline`

---

### `dte.train()`

```python
dte.train(
    data_path: str,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    output_dir: str = "./models",
    verbose: bool = False,
) -> dict
```

Train a model using GRPO on previously generated debate data.

**Parameters**:

| Parameter       | Type    | Default                         | Description                  |
|-----------------|---------|---------------------------------|------------------------------|
| `data_path`     | `str`   | (required)                      | Path to JSONL training data  |
| `model`         | `str`   | `"Qwen/Qwen2.5-1.5B-Instruct"` | Model to fine-tune           |
| `epochs`        | `int`   | `3`                             | Training epochs              |
| `batch_size`    | `int`   | `4`                             | Training batch size          |
| `learning_rate` | `float` | `2e-5`                          | Peak learning rate           |
| `output_dir`    | `str`   | `"./models"`                    | Checkpoint directory         |
| `verbose`       | `bool`  | `False`                         | Print progress               |

**Returns**: `dict` with keys `epoch_losses`, `step_losses`, `learning_rates`,
`kl_divergences`, `advantages_stats`, and optionally `reward_stats`.

---

### `dte.evaluate()`

```python
dte.evaluate(
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    datasets: list = None,
    num_agents: int = 3,
    max_rounds: int = 3,
    max_samples: int = 100,
    verbose: bool = False,
) -> dict
```

Evaluate a model on standard benchmarks using multi-agent debate.

**Parameters**:

| Parameter     | Type   | Default                         | Description                     |
|---------------|--------|---------------------------------|---------------------------------|
| `model`       | `str`  | `"Qwen/Qwen2.5-1.5B-Instruct"` | Model to evaluate               |
| `datasets`    | `list` | `["gsm8k"]`                     | Dataset names to evaluate on    |
| `num_agents`  | `int`  | `3`                             | Number of debate agents         |
| `max_rounds`  | `int`  | `3`                             | Maximum debate rounds           |
| `max_samples` | `int`  | `100`                           | Max samples per dataset         |
| `verbose`     | `bool` | `False`                         | Print progress                  |

**Returns**: `dict` -- evaluation report with `overall_metrics`,
`per_dataset_metrics`, `transition_analysis`, and `reasoning_analysis`.

---

## Configuration Classes

All configuration classes are Python dataclasses in `dte.core.config`.

### `DTEConfig`

Top-level configuration container.

```python
@dataclass
class DTEConfig:
    model: ModelConfig
    debate: DebateConfig
    datasets: DatasetsConfig
    training: TrainingConfig
    evolution: EvolutionConfig
    logging: LoggingConfig
    hardware: HardwareConfig
    paths: PathsConfig
    experiment: ExperimentConfig
    safety: SafetyConfig
```

**Methods**:

| Method                | Returns              | Description                                |
|-----------------------|----------------------|--------------------------------------------|
| `from_yaml(path)`     | `DTEConfig`          | Load from YAML file                        |
| `from_dict(d)`        | `DTEConfig`          | Create from dictionary                     |
| `to_dict()`           | `dict`               | Serialize to dictionary                    |
| `save_yaml(path)`     | `None`               | Write to YAML file                         |
| `validate(strict)`    | `List[str]`          | Validate, return error strings             |
| `validate_strict()`   | `List[str]`          | Stricter validation                        |
| `validate_and_raise()`| `None`               | Raise `ConfigurationError` on failure      |
| `setup_environment()` | `None`               | Set seeds, create dirs, set env vars       |

### `ModelConfig`

```python
@dataclass
class ModelConfig:
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    base_model_path: Optional[str] = None
    device: str = "auto"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
```

### `DebateConfig`

```python
@dataclass
class DebateConfig:
    num_agents: int = 3
    max_rounds: int = 3
    consensus_threshold: float = 1.0
    consensus_tolerance: float = 1e-9
    debate_prompting: DebatePromptingConfig
    rcr_prompting: DebatePromptingConfig     # backward compat alias
    use_diverse_agents: bool = False
    agent_models: List[str] = []
    temperature_annealing: TemperatureAnnealingConfig
    track_sycophancy: bool = True
    consolidate_reasoning: bool = True
```

### `DebatePromptingConfig`

```python
@dataclass
class DebatePromptingConfig:
    enabled: bool = True
    initial_prompt_type: str = "math"
    include_agent_context: bool = True
    include_peer_solutions: bool = True
    defend_previous_answer: bool = True
    require_novel_reasoning: bool = True
    critique_pairs: int = 2
```

### `TrainingConfig`

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    grpo: GRPOConfig
    rewards: RewardsConfig
    lora: LoRAConfig
```

### `GRPOConfig`

```python
@dataclass
class GRPOConfig:
    group_size: int = 4
    advantage_normalization: bool = True
    clip_ratio: float = 0.2
    kl_penalty: float = 0.02
```

### `RewardsConfig`

```python
@dataclass
class RewardsConfig:
    # DTE-specific reward function weights
    correctness_weight: float = 2.0
    int_weight: float = 0.5
    strict_format_weight: float = 0.5
    soft_format_weight: float = 0.5
    xmlcount_weight: float = 0.5

    # Legacy weights (backward compatibility)
    answer_weight: float = 2.0
    format_weight: float = 0.5
    length_weight: float = 0.5
    length_tau: int = 120
```

**Methods**:

| Method              | Returns            | Description                                      |
|---------------------|--------------------|--------------------------------------------------|
| `get_dte_weights()` | `Dict[str, float]` | Returns DTE-specific weight dict with keys `correctness`, `int`, `strict_format`, `soft_format`, `xmlcount` |

### `LoRAConfig`

```python
@dataclass
class LoRAConfig:
    enabled: bool = True
    rank: int = 128
    alpha: int = 256
    dropout: float = 0.05
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", ...]
```

### `EvolutionConfig`

```python
@dataclass
class EvolutionConfig:
    max_rounds: int = 3
    convergence_threshold: float = 0.01
    patience: int = 2
    samples_per_round: int = 500
    validation_split: float = 0.2
    validation_freq: int = 1
    min_improvement: float = 0.01
```

### Other Config Classes

- **`TemperatureAnnealingConfig`**: `enabled`, `start_temp`, `end_temp`, `min_model_size`
- **`DatasetsConfig`**: `names`, `max_samples_per_dataset`, `quality_threshold`, `train_datasets`, `eval_datasets`
- **`DatasetInfo`**: `name`, `path`, `split`, `max_samples`
- **`LoggingConfig`**: `level`, `log_dir`, `experiment_name`, `save_checkpoints`, `checkpoint_freq`, `track_metrics`
- **`HardwareConfig`**: `device`, `mixed_precision`, `max_memory_per_gpu`, `gradient_checkpointing`, `num_workers`, `dataloader_pin_memory`
- **`PathsConfig`**: `output_dir`, `models_dir`, `data_dir`, `cache_dir`, `temp_dir`
- **`ExperimentConfig`**: `name`, `description`, `tags`, `seed`, `deterministic`, `wandb`
- **`WandbConfig`**: `enabled`, `project`, `entity`
- **`SafetyConfig`**: `filter_toxic_content`, `max_reasoning_length`, `validate_model_outputs`, etc.

---

## Debate Module

### `DebateManager`

```python
class DebateManager:
    def __init__(self, config_debate, config_model, logger=None)
    def conduct_debate(self, query: str, task_type: str = "math") -> DebateResult
    def get_debate_statistics(self) -> Dict[str, Any]
    def update_evolution_round(self, evolution_round: int) -> None
    def cleanup(self) -> None
```

### `DebateAgent`

```python
class DebateAgent:
    def __init__(self, agent_id, model_name, device="auto",
                 model_config=None, generation_config=None)
    def generate_initial_response(self, query, task_type="math") -> DebateResponse
    def generate_debate_response(self, query, answers_so_far, round_num,
                                  task_type="math") -> DebateResponse
    def get_current_response(self) -> Optional[DebateResponse]
    def get_response_history(self) -> List[DebateResponse]
    def reset_history(self) -> None
    def get_performance_stats(self) -> Dict[str, Any]
    def update_generation_config(self, new_config: Dict) -> None
    def cleanup(self) -> None
```

### `DebatePromptManager`

```python
class DebatePromptManager:
    def __init__(self, critique_pairs: int = 2)
    def create_initial_prompt(self, query, task_type="math") -> str
    def create_debate_prompt(self, query, agent_id, round_num,
                              answers_so_far, task_type="math") -> str
    def parse_response(self, response_text, agent_id="", round_num=0,
                        task_type="math") -> DebateResponse
    def validate_response_format(self, response, task_type="math") -> List[str]
```

### `DebateResponse`

```python
@dataclass
class DebateResponse:
    answer: str
    reasoning: str
    extracted_answer: str
    confidence: Optional[float] = None
    round_number: int = 0
    agent_id: str = ""
```

### `DebateResult`

```python
@dataclass
class DebateResult:
    query: str
    final_answer: str
    consensus_reached: bool
    total_rounds: int
    all_responses: List[List[DebateResponse]]
    extracted_answers: List[List[str]]
    agent_answer_history: Dict[str, List[str]]
    sycophancy_history: List[Dict[str, bool]]
    consensus_progression: List[bool]
    confidence_progression: List[List[float]]
    metrics: Dict[str, Any]
    consolidated_reasoning: str
    task_type: str
```

---

## Training Module

### `GRPOTrainer`

```python
class GRPOTrainer:
    def __init__(self, config_training, config_model, config_paths, logger=None)
    def train(self, training_examples, validation_examples=None) -> Dict[str, Any]
    def get_detailed_reward_breakdown(self, query, response,
                                       ground_truth=None) -> Dict[str, Any]
    def cleanup(self) -> None
```

### `DTERewardModel`

```python
class DTERewardModel:
    def __init__(self)
    def calculate_all_rewards(self, query, responses, ground_truth=None)
        -> Dict[str, List[float]]
    def correctness_reward_func(self, responses, ground_truth) -> List[float]
    def int_reward_func(self, responses) -> List[float]
    def strict_format_reward_func(self, responses) -> List[float]
    def soft_format_reward_func(self, responses) -> List[float]
    def xmlcount_reward_func(self, responses) -> List[float]
    def combine_rewards(self, rewards_dict, weights=None) -> List[float]
    def get_reward_statistics(self, rewards_dict) -> Dict[str, Dict[str, float]]
```

### `TrainingExample`

```python
@dataclass
class TrainingExample:
    query: str
    answer: str
    reasoning: str
    confidence: float
    source_dataset: str
    debate_rounds: int
    consensus_reached: bool
    metadata: Dict[str, Any]
```

---

## Data Module

### `DatasetManager`

```python
class DatasetManager:
    DATASET_CONFIGS: Dict[str, Dict]   # 7 supported datasets

    def __init__(self, cache_dir=None)
    def load_dataset_by_name(self, name, split="train",
                              max_samples=None) -> Dataset
    def preprocess_dataset(self, dataset, dataset_name) -> Dataset
    def load_from_file(self, file_path, format="auto") -> Dataset
    def save_dataset(self, dataset, file_path, format="json")
    def get_dataset_info(self, name) -> Dict[str, Any]
    def list_supported_datasets(self) -> List[str]
    def clear_cache(self)
    def get_cache_info(self) -> Dict[str, Any]
```

### `DebateDataGenerator`

```python
class DebateDataGenerator:
    def __init__(self, config_datasets, config_debate, config_model, logger=None)
    def generate_training_data(self, num_samples, evolution_round=0,
                                save_path=None) -> List[TrainingExample]
    def load_generated_data(self, load_path) -> List[TrainingExample]
    def get_generation_statistics(self) -> Dict[str, Any]
    def update_quality_filters(self, new_filters) -> None
    def reset_generated_data(self) -> None
    def cleanup(self) -> None
```

---

## Utility Functions

### Answer Extraction (`dte.utils.answer_extraction`)

```python
def extract_final_answer(response: str) -> str
def extract_ground_truth(answer: str) -> str
def clean_numeric_string(s) -> Optional[Union[int, float]]
def answers_match(answer1: str, answer2: str) -> bool
def check_consensus(answers: List[str]) -> bool
def detect_sycophancy(agent_answers, round_num) -> Dict[str, bool]
def extract_arc_answer(response: str) -> str
def calculate_accuracy(predictions, ground_truths, task_type="math") -> float
def consolidate_reasoning_traces(all_responses, final_answer) -> str
```

### Data Utilities (`dte.utils.data_utils`)

```python
def load_jsonl(file_path: str) -> List[Dict]
def save_jsonl(data: List[Dict], file_path: str) -> None
def split_dataset(data, train_ratio=0.8, val_ratio=0.1,
                   test_ratio=0.1, shuffle=True, seed=42) -> Tuple
def filter_by_length(data, min_length=10, max_length=1000,
                      text_key="text") -> List[Dict]
def deduplicate_data(data, key="text") -> List[Dict]
def sample_balanced(data, label_key, samples_per_class) -> List[Dict]
def validate_data_format(data, required_keys) -> List[str]
def merge_datasets(datasets, add_source_info=True) -> List[Dict]
```

### Helpers (`dte.utils.helpers`)

```python
# Time and formatting
def format_time(seconds: float) -> str
class Timer:   # context manager

# Device management
def validate_device(device: str) -> torch.device
def get_gpu_memory_info() -> Optional[Tuple[int, int]]
def clear_gpu_cache() -> None
def get_model_parameter_count(model) -> int

# Model utilities
def calculate_model_size(model_name: str) -> Optional[float]
def setup_reproducibility(seed: int = 42) -> None

# Error handling
class DTEError(Exception)
class ConfigurationError(DTEError)
class DebateError(DTEError)
class TrainingError(DTEError)
class DataError(DTEError)
class ModelError(DTEError)
def safe_execute(func, *args, component="unknown", fallback_result=None,
                  reraise=True, **kwargs) -> Any
def robust_retry(max_retries=3, delay=1.0, backoff_factor=2.0,
                  exceptions=(Exception,))  # decorator

# Path and model validation
def validate_file_path(path, must_exist=False, must_be_file=False,
                        create_parent=False) -> Path
def validate_model_name(model_name, available_models=None) -> str

# System checks
def check_system_requirements(require_cuda=False,
                               min_memory_gb=None) -> Dict[str, Any]
def create_experiment_directory(base_path, experiment_name) -> Path
```

### `DTEPipeline` (`dte.core.pipeline`)

```python
class DTEPipeline:
    def __init__(self, config: DTEConfig)
    def run_complete_pipeline(self) -> Dict[str, Any]
    def run_single_round(self, round_num: int) -> EvolutionRoundResult
    def get_pipeline_status(self) -> Dict[str, Any]
    def save_checkpoint(self, checkpoint_path: str) -> None
    def load_checkpoint(self, checkpoint_path: str) -> None
```

### `DTEEvaluator` (`dte.core.evaluator`)

```python
class DTEEvaluator:
    def __init__(self, config_datasets, config_debate, config_model, logger=None)
    def evaluate_model(self, evolution_round, max_samples_per_dataset=None)
        -> EvaluationMetrics
    def create_evaluation_report(self, metrics, evolution_round) -> Dict[str, Any]
    def cleanup(self) -> None
```

### `EvaluationMetrics`

```python
@dataclass
class EvaluationMetrics:
    overall_accuracy: float
    total_samples: int
    correct_samples: int
    average_debate_rounds: float
    consensus_rate: float
    sycophancy_rate: float
    correct_to_incorrect_rate: float
    incorrect_to_correct_rate: float
    debate_helped_rate: float
    average_reasoning_length: float
    evaluation_time: float
    per_dataset_metrics: Dict[str, Dict[str, float]]
```

### `DTELogger` (`dte.core.logger`)

```python
class DTELogger:
    def __init__(self, config_logging, experiment_name="dte_experiment")
    def info(self, message, **kwargs) -> None
    def warning(self, message, **kwargs) -> None
    def error(self, message, **kwargs) -> None
    def debug(self, message, **kwargs) -> None
    def component_context(self, component)     # context manager
    def round_context(self, round_id)          # context manager
    def start_progress(self, description, total=None) -> Progress
    def update_progress(self, description, advance=1) -> None
    def finish_progress(self, description=None) -> None
    def log_metrics(self, metrics, step=None) -> None
    def log_debate_round(self, round_num, agents_responses,
                          consensus_reached, final_answer) -> None
    def log_training_step(self, step, loss, metrics) -> None
    def log_evolution_round(self, round_num, performance_metrics,
                             improvement) -> None
    def log_model_checkpoint(self, checkpoint_path, metrics) -> None
    def log_experiment_summary(self, final_metrics) -> None
    def get_metrics_history(self) -> List[Dict]
    def save_metrics_csv(self, filename=None) -> Path
    def create_component_logger(self, component: str) -> logging.Logger
```

### `LogEntry` (`dte.core.logger`)

```python
@dataclass
class LogEntry:
    """Structured log entry written to JSONL files."""
    timestamp: float
    level: str
    message: str
    component: str
    round_id: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
```

### `EvolutionRoundResult` (`dte.core.pipeline`)

```python
@dataclass
class EvolutionRoundResult:
    """Results from a single evolution round."""
    round_number: int
    data_generation_stats: Dict[str, Any]
    training_metrics: Dict[str, Any]
    evaluation_results: Dict[str, Any]
    performance_improvement: float
    total_time: float
```

---

## Training Internals

### `GRPOBatch` (`dte.training.grpo_trainer`)

```python
@dataclass
class GRPOBatch:
    """A batch prepared for GRPO training."""
    queries: List[str]
    responses: List[List[str]]       # responses[query_idx][response_idx]
    rewards: List[List[float]]       # rewards[query_idx][response_idx]
    advantages: List[List[float]]    # group-relative advantages
```

### `TrainingDataset` (`dte.training.grpo_trainer`)

```python
class TrainingDataset(torch.utils.data.Dataset):
    """Dataset wrapper for GRPO training.

    Wraps TrainingExample instances into the XML format expected by
    DTE reward functions.
    """
    def __init__(self, examples: List[TrainingExample], tokenizer, max_length: int = 2048)
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> Dict[str, Any]
        # Returns: {"query": str, "response": str, "answer": str, "reward": float}
```

---

## Data Processing Module

### `DataProcessor` (`dte.data.processor`)

```python
class DataProcessor:
    """Data processing and validation for DTE training data."""
    def __init__(self)
    def process_training_examples(self, examples: List[TrainingExample])
        -> List[Dict[str, Any]]
    def format_for_model(self, example: Dict[str, Any],
                          format_type: str = "xml") -> str
    def validate_xml_format(self, text: str) -> Dict[str, Any]
    def get_processing_statistics(self, examples: List[Dict[str, Any]])
        -> Dict[str, Any]
```

**`format_for_model` format types**:

| `format_type` | Output format |
|---------------|---------------|
| `"xml"` | `<reasoning>...</reasoning>\n<answer>...</answer>` (DTE standard) |
| `"plain"` | `Query: ...\n\nAnswer: ...\n\nReasoning: ...` |
| `"chat"` | `Human: ...\n\nAssistant: ...\n\nThe answer is ...` |

**`validate_xml_format` return dict**:

| Key | Type | Description |
|-----|------|-------------|
| `is_valid` | `bool` | True if both reasoning and answer tags found |
| `has_reasoning_tags` | `bool` | True if `<reasoning>...</reasoning>` present |
| `has_answer_tags` | `bool` | True if `<answer>...</answer>` present |
| `reasoning_content` | `Optional[str]` | Extracted reasoning text |
| `answer_content` | `Optional[str]` | Extracted answer text |
| `errors` | `List[str]` | List of validation error descriptions |

---

## Model Weight Sharing Registry

### `_ModelRegistry` (`dte.debate.agent`)

```python
class _ModelRegistry:
    """Thread-safe registry that caches loaded models for weight sharing.

    Models are keyed by (model_name, device_str) and reference-counted.
    This is a module-level singleton -- not part of the public API, but
    important for understanding memory management.
    """
    def acquire(self, model_name, device, model_config) -> Tuple[model, tokenizer]
    def release(self, model_name, device) -> None
```

When multiple `DebateAgent` instances share the same model name and device,
only one copy of the model is loaded. When all agents are cleaned up (via
`agent.cleanup()` or `manager.cleanup()`), the reference count drops to zero
and the model is freed from GPU memory.
