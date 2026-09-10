"""Setup script for DTE Framework."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "DTE Framework: Debate, Train, Evolve"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "peft>=0.7.0",
        "datasets>=2.14.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "tqdm>=4.66.0",
        "rich>=13.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "psutil>=5.9.0",
    ]

setup(
    name="dte-framework",
    version="0.1.0",
    author="Gaurav Srivastava, Zhenyu Bi, Meng Lu, Xuan Wang",
    author_email="gks@vt.edu",
    description="Debate, Train, Evolve: Self-Evolution of Language Model Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ctrl-gaurav/Debate-Train-Evolve",
    project_urls={
        "Homepage": "https://ctrl-gaurav.github.io/debate-train-evolve.github.io/",
        "Repository": "https://github.com/ctrl-gaurav/Debate-Train-Evolve",
        "Issues": "https://github.com/ctrl-gaurav/Debate-Train-Evolve/issues",
        "Paper": "https://aclanthology.org/2025.emnlp-main.1666/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "ruff>=0.4.0",
            "mypy>=1.5.0",
        ],
        "wandb": [
            "wandb>=0.16.0",
        ],
        "gpu": [
            "bitsandbytes>=0.41.0",
            "flash-attn>=2.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dte=dte.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "dte": [
            "py.typed",
            "*.yaml",
            "*.json",
        ],
    },
    zip_safe=False,
)