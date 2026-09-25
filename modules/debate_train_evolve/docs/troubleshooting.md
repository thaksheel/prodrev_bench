# Troubleshooting

Common issues and their solutions when working with the DTE framework.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Import Errors](#import-errors)
- [GPU and Memory Issues](#gpu-and-memory-issues)
- [Configuration Errors](#configuration-errors)
- [Debate Issues](#debate-issues)
- [Training Issues](#training-issues)
- [Evaluation Issues](#evaluation-issues)
- [Dataset Issues](#dataset-issues)
- [Performance Tips](#performance-tips)

---

## Installation Issues

### `pip install -e ".[dev]"` fails with build errors

**Symptom**: Build fails with errors about missing C compilers or libraries.

**Solution**: Make sure you have the system prerequisites:

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# macOS
xcode-select --install
```

### `ModuleNotFoundError: No module named 'torch'`

**Symptom**: PyTorch is not installed.

**Solution**: Install PyTorch for your CUDA version:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### bitsandbytes installation fails

**Symptom**: `pip install bitsandbytes` fails or the library is not found.

**Solution**: bitsandbytes requires CUDA. Make sure CUDA is properly
installed:

```bash
nvcc --version   # Should show CUDA version
pip install bitsandbytes
```

If you get `libbitsandbytes_cuda*.so` errors, check that your CUDA version
matches the bitsandbytes build.

---

## Import Errors

### `import dte` works but `dte.debate()` raises RuntimeError

**Symptom**:
```
RuntimeError: ML dependencies (torch, transformers) are required for debate.
```

**Cause**: Core config classes load fine, but the ML components (debate, training)
failed to import. This usually means torch or transformers is missing.

**Solution**:
```bash
pip install torch transformers accelerate
```

### `ImportError: cannot import name 'DebateManager'`

**Symptom**: Importing specific components fails.

**Cause**: A dependency of the component is missing.

**Solution**: Install all core dependencies:
```bash
pip install -e "."
```

---

## GPU and Memory Issues

### `CUDA out of memory`

**Symptom**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions** (try in order):

1. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 2
     gradient_accumulation_steps: 8  # keep effective batch = 16
   ```

2. **Reduce max_length**:
   ```yaml
   model:
     max_length: 1024  # instead of 2048
   ```

3. **Enable gradient checkpointing**:
   ```yaml
   hardware:
     gradient_checkpointing: true
   ```

4. **Use a smaller LoRA rank**:
   ```yaml
   training:
     lora:
       rank: 64   # instead of 128
       alpha: 128
   ```

5. **Use fewer agents during debate** (for inference OOM):
   ```yaml
   debate:
     num_agents: 2  # instead of 3
   ```

6. **Clear GPU cache before training**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### `RuntimeError: Expected all tensors to be on the same device`

**Symptom**: Tensors are on different devices (CPU vs CUDA).

**Solution**: Make sure `device` is set consistently:
```yaml
model:
  device: "auto"   # Let the framework handle device placement
hardware:
  device: "auto"
```

### Model does not use all available GPUs

**Symptom**: Only 1 GPU is utilized even though multiple are available.

**Solution**: DTE uses `device_map="auto"` which shards large models across
GPUs. For models smaller than a single GPU's memory, all computation
happens on one GPU. This is expected behavior.

To control which GPUs are used:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python your_script.py
```

---

## Configuration Errors

### `ConfigurationError: Configuration validation failed`

**Symptom**: `config.validate_and_raise()` throws an error.

**Solution**: Check the specific error messages:

```python
config = DTEConfig.from_yaml("config.yaml")
errors = config.validate()
for e in errors:
    print(f"  - {e}")
```

Common validation failures:

| Error | Fix |
|-------|-----|
| `model.temperature must be between 0 and 2` | Set temperature to 0.0-2.0 |
| `debate.num_agents must be at least 2` | Use at least 2 agents |
| `training.learning_rate must be positive` | Set a positive learning rate |
| `Unknown dataset: xyz` | Use one of the 7 supported datasets |

### `FileNotFoundError: Configuration file not found`

**Symptom**: YAML file does not exist at the specified path.

**Solution**: Check the path is correct and the file exists:
```python
from pathlib import Path
print(Path("config.yaml").resolve())
print(Path("config.yaml").exists())
```

### `yaml.YAMLError` when loading config

**Symptom**: YAML parsing fails.

**Solution**: Validate your YAML syntax. Common issues:
- Tabs instead of spaces (YAML requires spaces)
- Missing colons after keys
- Incorrect indentation

Use an online YAML validator or:
```python
import yaml
with open("config.yaml") as f:
    yaml.safe_load(f)  # Will show exact error location
```

---

## Debate Issues

### All agents give the same wrong answer

**Cause**: Agents share the same model and produce similar outputs.

**Solutions**:
- Increase `temperature` to add diversity
- Use `use_diverse_agents: true` with different models
- Increase `max_rounds` to give agents more chances to revise

### Debate takes too long

**Cause**: Each round requires model inference for all agents.

**Solutions**:
- Reduce `max_rounds` to 2
- Reduce `num_agents` to 2
- Use a smaller model
- Reduce `max_length`

### `consensus_reached` is always False

**Cause**: Agents never fully agree.

**Solutions**:
- Check if answer extraction is working (test with known examples)
- The `consensus_tolerance` of `1e-9` requires near-exact matches
- For non-numeric tasks, answers must match exactly after extraction

### Sycophancy rate is very high

**Cause**: Agents are blindly adopting peer answers.

**Solutions**:
- Review the prompt templates in `prompts.py`
- Consider lowering `temperature` in later rounds (temperature annealing)
- Ensure `defend_previous_answer` is enabled in the config

---

## Training Issues

### Training loss is NaN or explodes

**Cause**: Learning rate too high or numerical instability.

**Solutions**:
- Lower learning rate: `learning_rate: 1e-5`
- Increase warmup: `warmup_steps: 100`
- Check for NaN in data (ground truth answers)
- Ensure mixed precision is enabled: `mixed_precision: true`

### Training is very slow

**Cause**: Large group_size or small batch_size.

**Solutions**:
- Reduce `group_size` from 4 to 2 (fewer response samples per query)
- Increase `batch_size` (if memory allows)
- Enable LoRA (if not already enabled)
- Use gradient accumulation to maintain effective batch size with smaller
  actual batch size

### Reward is always 0

**Cause**: Ground truth answers are not matching.

**Solutions**:
- Check that training data has correct ground truth
- Test answer extraction:
  ```python
  from dte.utils.answer_extraction import extract_final_answer
  print(extract_final_answer("The answer is \\boxed{42}"))
  ```
- Check that responses include `<answer>...</answer>` tags

### LoRA not applied

**Symptom**: Log message says "LoRA requested but PEFT not available."

**Solution**: Install peft:
```bash
pip install peft
```

---

## Evaluation Issues

### Dataset fails to load

**Symptom**: `RuntimeError: Failed to load dataset 'xyz'`

**Solutions**:
- Check internet connectivity (datasets are downloaded from HuggingFace)
- Use a cache directory: `DatasetManager(cache_dir="./cache")`
- Verify the dataset name is one of the 7 supported options
- Some datasets require authentication -- log in with `huggingface-cli login`

### Evaluation accuracy is 0%

**Cause**: Answer extraction is failing or task_type mismatch.

**Solutions**:
- Verify the model is producing readable output
- Check that task_type matches the dataset (math datasets need `"math"`,
  ARC datasets need `"arc"`)
- Test with a simple known-good model first

---

## Dataset Issues

### `ValueError: Dataset 'xyz' not supported`

**Cause**: The dataset name is not in the list of 7 supported datasets.

**Solution**: Use one of:
```
gsm8k, gsm_plus, math, arc_challenge, arc_easy, gpqa, commonsense_qa
```

### CommonsenseQA choices format error

**Cause**: CommonsenseQA uses a different choices format than ARC.

**Solution**: The framework handles this automatically through the
`DatasetManager.preprocess_dataset()` method. Make sure you are using the
dataset manager rather than loading data manually.

---

## Performance Tips

### Faster debates

- Use a smaller model for initial experiments (e.g., 0.5B instead of 1.5B)
- Reduce `max_rounds` to 2 (most improvement happens in round 1)
- Reduce `max_length` to 1024 for shorter problems

### Lower memory usage

- Enable LoRA (reduces trainable parameters by ~95%)
- Use gradient checkpointing
- Use bfloat16 mixed precision
- Use 8-bit AdamW (install bitsandbytes)

### Better training results

- Generate more debate data (increase `samples_per_round`)
- Use quality filtering (consensus-only examples)
- Use multiple evolution rounds (3 is typical)
- Increase `group_size` for more stable advantage estimates

### Reproducibility

- Set `experiment.seed: 42` and `experiment.deterministic: true`
- Save your config file alongside results
- Use `config.save_yaml("experiment_config.yaml")` to snapshot configuration

---

## Dependency Conflicts

### `transformers` and `peft` version mismatch

**Symptom**: `ImportError` or `AttributeError` when LoRA is applied.

**Solution**: Ensure compatible versions:
```bash
pip install transformers>=4.36.0 peft>=0.7.0 accelerate>=0.25.0
```

### `datasets` library version issues

**Symptom**: `TypeError` or missing methods when loading HuggingFace datasets.

**Solution**: Use a recent version:
```bash
pip install datasets>=2.14.0
```

### `rich` import errors

**Symptom**: `ModuleNotFoundError: No module named 'rich'` when using the logger.

**Solution**: Install rich:
```bash
pip install rich
```

---

## Advanced Debugging

### Inspecting reward signals during training

If training is not converging, inspect the individual reward components:

```python
from dte.training.grpo_trainer import GRPOTrainer

# After creating the trainer:
breakdown = trainer.get_detailed_reward_breakdown(
    query="What is 6 * 7?",
    response="<reasoning>\n6 * 7 = 42\n</reasoning>\n<answer>\n42\n</answer>\n",
    ground_truth="42",
)

for key, val in breakdown["individual_rewards"].items():
    print(f"  {key}: {val:.3f}")
print(f"  has_xml: {breakdown['has_xml_format']}")
print(f"  extracted: {breakdown['extracted_answer']}")
```

### Checking model weight sharing

If you suspect agents are not sharing model weights (excess memory usage),
verify the registry:

```python
from dte.debate.agent import _model_registry

# After creating agents:
print(_model_registry._cache)
# Should show one entry with refcount = num_agents
```

### Debugging answer extraction

If accuracy is unexpectedly low, test answer extraction directly:

```python
from dte.utils.answer_extraction import extract_final_answer, answers_match

# Test extraction
answer = extract_final_answer("The answer is \\boxed{42}")
print(f"Extracted: {answer}")

# Test matching
print(answers_match("42", "42.0"))  # True (within 1e-9)
print(answers_match("42", "43"))    # False
```

### Enabling debug-level logging

For maximum verbosity, set the logging level to DEBUG:

```python
from dte.core.config import LoggingConfig
from dte.core.logger import DTELogger

logger = DTELogger(LoggingConfig(level="DEBUG"), "debug_session")
# All structured log entries are written to ./logs/debug_session.jsonl
```

Or via the CLI:
```bash
python main.py --verbose debate --query "What is 2+2?"
```
