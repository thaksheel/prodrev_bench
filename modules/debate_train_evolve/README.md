# Debate, Train, Evolve

<div align="center">

[![EMNLP 2025](https://img.shields.io/badge/EMNLP_2025-Main_Conference-brightgreen?style=for-the-badge)](https://aclanthology.org/2025.emnlp-main.1666/)
[![Paper](https://img.shields.io/badge/Paper-ACL_Anthology-blue?style=for-the-badge)](https://aclanthology.org/2025.emnlp-main.1666/)
[![Website](https://img.shields.io/badge/Website-Live-orange?style=for-the-badge)](https://ctrl-gaurav.github.io/debate-train-evolve.github.io/)
[![Python](https://img.shields.io/badge/Python-3.9--3.13-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Self-Evolution of Language Model Reasoning via Multi-Agent Debate Traces**

**[Gaurav Srivastava](mailto:gks@vt.edu)**\* &nbsp;&bull;&nbsp; **[Zhenyu Bi](mailto:zhenyub@vt.edu)** &nbsp;&bull;&nbsp; **[Meng Lu](mailto:menglu@vt.edu)** &nbsp;&bull;&nbsp; **[Xuan Wang](mailto:xuanw@vt.edu)**&dagger;

[![Virginia Tech](https://img.shields.io/badge/Virginia_Tech-CS_Department-861F41?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSJjdXJyZW50Q29sb3IiLz4KPC9zdmc+)](https://cs.vt.edu/)
&nbsp;
[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-2a4dff?style=flat-square&logo=academia)](https://2025.emnlp.org/)
&nbsp;
[![ACL Anthology](https://img.shields.io/badge/ACL_Anthology-2025.emnlp--main.1666-red?style=flat-square)](https://aclanthology.org/2025.emnlp-main.1666/)

<sub>\* Lead Author &nbsp;&nbsp; &dagger; Corresponding Author</sub>

[**Read the Paper**](https://aclanthology.org/2025.emnlp-main.1666/) &nbsp;|&nbsp; [**Website & Docs**](https://ctrl-gaurav.github.io/debate-train-evolve.github.io/) &nbsp;|&nbsp; [**GitHub**](https://github.com/ctrl-gaurav/Debate-Train-Evolve)

</div>

---

## Overview

**DTE (Debate, Train, Evolve)** is a ground-truth-free training framework that evolves language model reasoning through multi-agent debate traces. Multiple LLM copies debate using Reflect-Critique-Refine (RCR) prompting, generating high-quality training data without external supervision. The model is then fine-tuned via Group Relative Policy Optimization (GRPO) and the process repeats.

**Key results:**
- Up to **+13.92%** accuracy gain (Qwen-1.5B on GSM-Plus)
- **+5.8%** average cross-domain generalization to science tasks
- Reduces sycophancy by **50%** via RCR prompting
- Single-model inference after training (no multi-agent overhead)

## Performance

| Model | GSM8K | GSM-Plus | MATH | ARC-Challenge | Best Gain |
|-------|-------|----------|------|---------------|-----------|
| Qwen-2.5-1.5B | 62.77 &rarr; **73.09** | 42.00 &rarr; **55.92** | 45.08 &rarr; **52.20** | 69.21 &rarr; 68.36 | **+13.92%** |
| Qwen-2.5-3B | 84.08 &rarr; **86.05** | 61.75 &rarr; **69.50** | 61.36 &rarr; **67.10** | 83.53 &rarr; **83.95** | **+7.75%** |
| Qwen-2.5-7B | 90.67 &rarr; 88.32 | 68.62 &rarr; **74.71** | 73.08 &rarr; **77.20** | 87.22 &rarr; **90.89** | **+6.09%** |
| Qwen-2.5-14B | 92.80 &rarr; **93.74** | 71.79 &rarr; **78.88** | 76.18 &rarr; **80.10** | 90.27 &rarr; **93.13** | **+7.09%** |
| Llama-3.2-3B | 72.55 &rarr; **75.06** | 45.67 &rarr; **53.79** | 39.76 &rarr; **43.80** | 73.12 &rarr; **77.23** | **+8.12%** |
| Llama-3.1-8B | 81.73 &rarr; **86.81** | 55.62 &rarr; **66.17** | 46.66 &rarr; **49.40** | 77.65 &rarr; **86.53** | **+10.55%** |

*Values show Base &rarr; Evolved performance. Bold = improvement.*

## Installation

**Prerequisites:** Python 3.9+ and a CUDA GPU (for training). Debate-only mode works on CPU.

```bash
# Quick setup (conda)
git clone https://github.com/ctrl-gaurav/Debate-Train-Evolve.git
cd Debate-Train-Evolve
bash setup.sh

# Or manual install
python -m venv dte_env && source dte_env/bin/activate
pip install -r requirements.txt
pip install -e .

# Verify
python main.py info
```

## Quick Start

### Python API

```python
import dte

# One-liner debate
result = dte.debate(
    "What is 15 * 24?",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    num_agents=3,
    max_rounds=3,
    task_type="math",
)
print(result.final_answer)       # "360"
print(result.consensus_reached)  # True
```

### CLI

```bash
# Single query debate
python main.py debate --query "What is 15 * 24?" --agents 3 --rounds 3

# Dataset evaluation
python main.py debate --dataset gsm8k --samples 20 --verbose

# Full pipeline (debate -> train -> evolve)
python main.py run --config config.yaml
```

### Full Pipeline

```python
import dte

pipeline = dte.from_config("config.yaml")
results = pipeline.run_complete_pipeline()
print(f"Improvement: {results['total_improvement']:.2%}")
```

## Project Structure

```
Debate-Train-Evolve/
├── dte/                        # Main package
│   ├── __init__.py             # Public API: dte.debate(), dte.from_config()
│   ├── core/                   # Config, pipeline, evaluator, logger
│   ├── debate/                 # Multi-agent debate (agent, manager, prompts)
│   ├── training/               # GRPO trainer + reward model
│   ├── data/                   # Dataset management + data generation
│   └── utils/                  # Answer extraction, helpers
├── examples/                   # 6 usage examples
├── tests/                      # Unit + GPU integration tests
├── config.yaml                 # Default configuration
├── main.py                     # CLI entry point
└── pyproject.toml              # Package metadata
```

## Documentation

Full documentation is available on the [project website](https://ctrl-gaurav.github.io/debate-train-evolve.github.io/#/docs), including:

- **Installation & Setup** -- prerequisites, GPU support, development setup
- **Quick Start** -- Python API, CLI, component-level usage
- **API Reference** -- all public classes and functions
- **Configuration** -- complete YAML config reference
- **Training Guide** -- GRPO hyperparameters, multi-GPU, expected training times
- **Reward Functions** -- the 5 shaped reward functions (total max: 4.0)
- **Dataset Reference** -- 7 benchmarks (GSM8K, GSM-Plus, MATH, ARC, GPQA, CommonsenseQA)
- **CLI Reference** -- all commands and flags
- **Troubleshooting** -- OOM, model loading, consensus issues
- **FAQ** -- common questions answered

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py run` | Run the complete DTE pipeline |
| `python main.py debate` | Standalone multi-agent debate |
| `python main.py generate` | Generate training data from debates |
| `python main.py train` | Train model with GRPO |
| `python main.py validate` | Validate a configuration file |
| `python main.py init` | Generate default config |
| `python main.py info` | Show system & GPU information |

## Contributing

```bash
pip install -e ".[dev]"

# Tests
pytest -m "not gpu" -v              # Unit tests (no GPU)
pytest tests/test_debate_integration.py -v  # GPU tests

# Lint & format
ruff check dte/ tests/
ruff format dte/ tests/
```

## Acknowledgments

This work was supported by NSF NAIRR Pilot with PSC Neocortex and NCSA Delta; Amazon, Cisco Research, Commonwealth Cyber Initiative, Amazon-Virginia Tech Center for Efficient and Robust Machine Learning, and the Sanghani Center for AI and Data Analytics at Virginia Tech.

## Citation

```bibtex
@inproceedings{srivastava2025debate,
  title={Debate, Train, Evolve: Self-Evolution of Language Model Reasoning},
  author={Srivastava, Gaurav and Bi, Zhenyu and Lu, Meng and Wang, Xuan},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025},
  url={https://aclanthology.org/2025.emnlp-main.1666/}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

[**Read the Paper**](https://aclanthology.org/2025.emnlp-main.1666/) &nbsp;|&nbsp; [**Website & Docs**](https://ctrl-gaurav.github.io/debate-train-evolve.github.io/) &nbsp;|&nbsp; [**GitHub**](https://github.com/ctrl-gaurav/Debate-Train-Evolve)

Made with &#10084;&#65039; by the DTE Research Team

[![Virginia Tech](https://img.shields.io/badge/Virginia_Tech-CS_Department-861F41?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSJjdXJyZW50Q29sb3IiLz4KPC9zdmc+)](https://cs.vt.edu/)
[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-2a4dff?style=flat&logo=academia)](https://2025.emnlp.org/)
[![ACL Anthology](https://img.shields.io/badge/ACL_Anthology-2025.emnlp--main.1666-red?style=flat)](https://aclanthology.org/2025.emnlp-main.1666/)

</div>
