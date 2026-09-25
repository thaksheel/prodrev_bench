"""General utility functions and error handling for DTE Framework."""

import functools
import re
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_model_size(model_name: str) -> Optional[float]:
    """Extract model size from model name.

    Args:
        model_name: Name of the model

    Returns:
        Model size in billions of parameters, or None if not found
    """
    model_name_lower = model_name.lower()

    # Common patterns for model sizes
    patterns = [
        r"(\d+\.?\d*)b",  # e.g., "1.5b", "3b", "7b"
        r"(\d+)b",  # e.g., "1b", "3b"
        r"(\d+\.?\d*)-?billion",  # e.g., "7-billion"
    ]

    for pattern in patterns:
        match = re.search(pattern, model_name_lower)
        if match:
            return float(match.group(1))

    return None


def validate_device(device: str) -> torch.device:
    """Validate and return appropriate device.

    Args:
        device: Device specification

    Returns:
        PyTorch device object

    Raises:
        ValueError: If device is invalid
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device in ["cpu", "cuda", "mps"]:
        device_obj = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS requested but not available")
        return device_obj
    else:
        # Try parsing as specific device (e.g., "cuda:0")
        try:
            return torch.device(device)
        except RuntimeError:
            raise ValueError(f"Invalid device specification: {device}")


def setup_reproducibility(seed: int = 42) -> None:
    """Set up reproducible random seeds.

    Args:
        seed: Random seed to use
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Timer:
    """Context manager for timing operations."""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def elapsed_str(self) -> str:
        """Get formatted elapsed time."""
        return format_time(self.elapsed)


def get_gpu_memory_info() -> Optional[Tuple[int, int]]:
    """Get GPU memory information.

    Returns:
        Tuple of (used_memory, total_memory) in MB, or None if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return None

    used = torch.cuda.memory_allocated() // (1024**2)
    total = torch.cuda.get_device_properties(0).total_memory // (1024**2)

    return used, total


def clear_gpu_cache() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model_parameter_count(model) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DTEError(Exception):
    """Base exception for DTE Framework errors."""

    def __init__(self, message: str, component: str = "unknown", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.component = component
        self.details = details or {}
        super().__init__(f"[{component}] {message}")


class ConfigurationError(DTEError):
    """Error in configuration validation or setup."""

    pass


class DebateError(DTEError):
    """Error during multi-agent debate process."""

    pass


class TrainingError(DTEError):
    """Error during model training."""

    pass


class DataError(DTEError):
    """Error in data processing or loading."""

    pass


class ModelError(DTEError):
    """Error in model loading or inference."""

    pass


def safe_execute(
    func: Callable, *args, component: str = "unknown", fallback_result: Any = None, reraise: bool = True, **kwargs
) -> Any:
    """Safely execute a function with error handling and logging.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        component: Component name for error context
        fallback_result: Result to return if function fails
        reraise: Whether to reraise exceptions after logging
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or fallback_result if error occurs

    Raises:
        DTEError: If reraise=True and an error occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_msg = f"Error in {component}: {str(e)}"

        # Log the full traceback
        tb_str = traceback.format_exc()

        if reraise:
            # Wrap in DTEError for better error context
            if isinstance(e, DTEError):
                raise
            else:
                raise DTEError(error_msg, component, {"original_error": str(e), "traceback": tb_str}) from e
        else:
            # Log and return fallback
            warnings.warn(f"{error_msg}. Returning fallback result.", UserWarning)
            return fallback_result


def robust_retry(
    max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0, exceptions: Tuple = (Exception,)
):
    """Decorator to retry function execution with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Factor to multiply delay by after each retry
        exceptions: Tuple of exceptions that trigger retry

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            # If we get here, all retries failed
            raise DTEError(
                f"Function {func.__name__} failed after {max_retries} retries",
                "retry_handler",
                {"last_exception": str(last_exception), "attempts": max_retries + 1},
            ) from last_exception

        return wrapper

    return decorator


def validate_file_path(
    path: Path, must_exist: bool = False, must_be_file: bool = False, create_parent: bool = False
) -> Path:
    """Validate and optionally create file paths.

    Args:
        path: Path to validate
        must_exist: Whether the path must already exist
        must_be_file: Whether the path must be a file (not directory)
        create_parent: Whether to create parent directories

    Returns:
        Validated Path object

    Raises:
        ConfigurationError: If validation fails
    """
    try:
        path = Path(path)

        if must_exist and not path.exists():
            raise ConfigurationError(f"Path does not exist: {path}", "file_validation")

        if must_be_file and path.exists() and not path.is_file():
            raise ConfigurationError(f"Path is not a file: {path}", "file_validation")

        if create_parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        return path

    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(f"Invalid file path {path}: {str(e)}", "file_validation") from e


def validate_model_name(model_name: str, available_models: Optional[List[str]] = None) -> str:
    """Validate model name format and availability.

    Args:
        model_name: Name of the model to validate
        available_models: List of available model names (optional)

    Returns:
        Validated model name

    Raises:
        ModelError: If model name is invalid or unavailable
    """
    if not model_name or not isinstance(model_name, str):
        raise ModelError("Model name must be a non-empty string", "model_validation")

    # Basic format validation (org/model or just model)
    if "/" in model_name and not re.match(r"^[\w\-\.]+/[\w\-\.]+$", model_name):
        raise ModelError(f"Invalid model name format: {model_name}", "model_validation")

    if available_models and model_name not in available_models:
        raise ModelError(f"Model '{model_name}' not in available models: {available_models}", "model_validation")

    return model_name


def check_system_requirements(require_cuda: bool = False, min_memory_gb: Optional[float] = None) -> Dict[str, Any]:
    """Check system requirements for DTE Framework.

    Args:
        require_cuda: Whether CUDA is required
        min_memory_gb: Minimum required memory in GB

    Returns:
        Dictionary with system information

    Raises:
        DTEError: If requirements are not met
    """
    system_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "python_version": f"{torch.__version__}",
        "torch_version": torch.__version__,
    }

    # Check CUDA requirement
    if require_cuda and not torch.cuda.is_available():
        raise DTEError("CUDA is required but not available", "system_check")

    # Check memory requirement
    if min_memory_gb:
        try:
            import psutil

            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            system_info["available_memory_gb"] = available_memory_gb

            if available_memory_gb < min_memory_gb:
                raise DTEError(
                    f"Insufficient memory: {available_memory_gb:.1f}GB available, {min_memory_gb}GB required",
                    "system_check",
                )
        except ImportError:
            warnings.warn("psutil not available, cannot check memory requirements", UserWarning)

    return system_info


def create_experiment_directory(base_path: Path, experiment_name: str) -> Path:
    """Create a unique experiment directory with timestamp.

    Args:
        base_path: Base directory for experiments
        experiment_name: Name of the experiment

    Returns:
        Path to created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_path / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    for subdir in ["logs", "checkpoints", "results", "config"]:
        (exp_dir / subdir).mkdir(exist_ok=True)

    return exp_dir
