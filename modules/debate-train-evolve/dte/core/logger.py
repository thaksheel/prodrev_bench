"""
Comprehensive logging system for the DTE Framework.

This module provides centralized logging capabilities with support for
multiple outputs, structured logging, and integration with experiment tracking.
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TaskID
from rich.table import Table


@dataclass
class LogEntry:
    """Structured log entry for DTE operations."""

    timestamp: float
    level: str
    message: str
    component: str
    round_id: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class DTELogger:
    """Comprehensive logging system for DTE Framework operations.

    Features:
    - Structured logging with JSON output
    - Rich console output with progress tracking
    - Metrics collection and reporting
    - Integration with experiment tracking systems
    - Component-specific loggers
    """

    def __init__(self, config_logging, experiment_name: str = "dte_experiment"):
        """Initialize the DTE logging system.

        Args:
            config_logging: Logging configuration object
            experiment_name: Name of the current experiment
        """
        self.config = config_logging
        self.experiment_name = experiment_name
        self.start_time = time.time()

        # Create log directory
        self.log_dir = Path(config_logging.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize console and loggers
        self.console = Console()
        self._setup_loggers()

        # Progress tracking
        self.progress_bars: Dict[str, TaskID] = {}
        self.current_progress: Optional[Progress] = None

        # Metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
        self.current_metrics: Dict[str, Any] = {}

        # Component tracking
        self.current_component = "main"
        self.current_round = None

    def _setup_loggers(self) -> None:
        """Set up Python logging infrastructure."""
        # Main logger
        self.logger = logging.getLogger("dte")
        self.logger.setLevel(getattr(logging, self.config.level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler for structured logs
        log_file = self.log_dir / f"{self.experiment_name}.jsonl"
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Rich console handler
        console_handler = RichHandler(console=self.console, show_time=True, show_path=False, rich_tracebacks=True)
        console_handler.setLevel(getattr(logging, self.config.level.upper()))
        self.logger.addHandler(console_handler)

        # Component-specific loggers
        self.component_loggers = {}
        for component in ["debate", "training", "evaluation", "evolution"]:
            comp_logger = logging.getLogger(f"dte.{component}")
            comp_logger.setLevel(getattr(logging, self.config.level.upper()))
            self.component_loggers[component] = comp_logger

    def _log_structured(self, level: str, message: str, **kwargs) -> None:
        """Log a structured entry to JSON file."""
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            component=self.current_component,
            round_id=self.current_round,
            metrics=kwargs.get("metrics"),
            metadata=kwargs.get("metadata"),
        )

        # Log to JSON file
        json_line = json.dumps(asdict(entry), default=str)
        self.logger.debug(json_line)

    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        self._log_structured("INFO", message, **kwargs)
        self.logger.info(f"[{self.current_component}] {message}")

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        self._log_structured("WARNING", message, **kwargs)
        self.logger.warning(f"[{self.current_component}] {message}")

    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        self._log_structured("ERROR", message, **kwargs)
        self.logger.error(f"[{self.current_component}] {message}")

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        self._log_structured("DEBUG", message, **kwargs)
        self.logger.debug(f"[{self.current_component}] {message}")

    @contextmanager
    def component_context(self, component: str):
        """Context manager for component-specific logging."""
        old_component = self.current_component
        self.current_component = component
        try:
            yield
        finally:
            self.current_component = old_component

    @contextmanager
    def round_context(self, round_id: int):
        """Context manager for round-specific logging."""
        old_round = self.current_round
        self.current_round = round_id
        try:
            yield
        finally:
            self.current_round = old_round

    def start_progress(self, description: str, total: Optional[int] = None) -> Progress:
        """Start a new progress bar."""
        if self.current_progress:
            self.current_progress.stop()

        self.current_progress = Progress(console=self.console)
        self.current_progress.start()

        task_id = self.current_progress.add_task(description, total=total)
        self.progress_bars[description] = task_id

        return self.current_progress

    def update_progress(self, description: str, advance: int = 1, **kwargs) -> None:
        """Update progress bar."""
        if self.current_progress and description in self.progress_bars:
            task_id = self.progress_bars[description]
            self.current_progress.update(task_id, advance=advance, **kwargs)

    def finish_progress(self, description: str = None) -> None:
        """Finish and remove progress bar."""
        if self.current_progress:
            if description and description in self.progress_bars:
                task_id = self.progress_bars[description]
                self.current_progress.remove_task(task_id)
                del self.progress_bars[description]
            else:
                self.current_progress.stop()
                self.current_progress = None
                self.progress_bars.clear()

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics with optional step information."""
        metrics_entry = {
            "timestamp": time.time(),
            "step": step,
            "round": self.current_round,
            "component": self.current_component,
            **metrics,
        }

        self.metrics_history.append(metrics_entry)
        self.current_metrics.update(metrics)

        # Log structured metrics
        self._log_structured("METRICS", "Metrics update", metrics=metrics)

        # Display metrics table
        self._display_metrics_table(metrics)

    def _display_metrics_table(self, metrics: Dict[str, Any]) -> None:
        """Display metrics in a formatted table."""
        table = Table(title=f"Metrics - {self.current_component.title()}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, torch.Tensor):
                formatted_value = f"{value.item():.4f}"
            else:
                formatted_value = str(value)
            table.add_row(key, formatted_value)

        self.console.print(table)

    def log_debate_round(
        self, round_num: int, agents_responses: List[Dict[str, Any]], consensus_reached: bool, final_answer: str
    ) -> None:
        """Log details of a debate round."""
        with self.round_context(round_num):
            self.info(f"Debate Round {round_num} completed")

            metrics = {
                "consensus_reached": consensus_reached,
                "num_agents": len(agents_responses),
                "final_answer": final_answer,
            }

            metadata = {"agents_responses": agents_responses, "round_type": "debate"}

            self.log_metrics(metrics)
            self._log_structured("DEBATE", f"Round {round_num} results", metrics=metrics, metadata=metadata)

    def log_training_step(self, step: int, loss: float, metrics: Dict[str, Any]) -> None:
        """Log training step information."""
        training_metrics = {"step": step, "loss": loss, "learning_rate": metrics.get("lr", 0), **metrics}

        self.log_metrics(training_metrics, step=step)

        if step % 10 == 0:  # Log every 10 steps
            self.info(f"Training step {step}: loss={loss:.4f}")

    def log_evolution_round(self, round_num: int, performance_metrics: Dict[str, Any], improvement: float) -> None:
        """Log evolution round results."""
        with self.round_context(round_num):
            self.info(f"Evolution Round {round_num} completed - Improvement: {improvement:.4f}")

            evolution_metrics = {"round": round_num, "improvement": improvement, **performance_metrics}

            self.log_metrics(evolution_metrics)

    def log_model_checkpoint(self, checkpoint_path: str, metrics: Dict[str, Any]) -> None:
        """Log model checkpoint information."""
        self.info(f"Model checkpoint saved: {checkpoint_path}")

        checkpoint_metadata = {
            "checkpoint_path": checkpoint_path,
            "model_metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        self._log_structured("CHECKPOINT", "Model saved", metadata=checkpoint_metadata)

    def log_experiment_summary(self, final_metrics: Dict[str, Any]) -> None:
        """Log final experiment summary."""
        duration = time.time() - self.start_time

        summary = {
            "experiment_name": self.experiment_name,
            "total_duration_seconds": duration,
            "total_duration_hours": duration / 3600,
            "final_metrics": final_metrics,
        }

        self.info("=" * 60)
        self.info("EXPERIMENT COMPLETED")
        self.info(f"Total Duration: {duration / 3600:.2f} hours")
        self.info("Final Metrics:")

        for key, value in final_metrics.items():
            self.info(f"  {key}: {value}")

        self.info("=" * 60)

        self._log_structured("SUMMARY", "Experiment completed", metadata=summary)

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get complete metrics history."""
        return self.metrics_history.copy()

    def save_metrics_csv(self, filename: Optional[str] = None) -> Path:
        """Save metrics history to CSV file."""
        import pandas as pd

        if not filename:
            filename = f"{self.experiment_name}_metrics.csv"

        csv_path = self.log_dir / filename

        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(csv_path, index=False)
            self.info(f"Metrics saved to {csv_path}")

        return csv_path

    def create_component_logger(self, component: str) -> logging.Logger:
        """Create a logger for a specific component."""
        if component not in self.component_loggers:
            comp_logger = logging.getLogger(f"dte.{component}")
            comp_logger.setLevel(getattr(logging, self.config.level.upper()))
            self.component_loggers[component] = comp_logger

        return self.component_loggers[component]
