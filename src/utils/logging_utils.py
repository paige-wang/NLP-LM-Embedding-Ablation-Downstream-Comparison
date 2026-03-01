"""Logging utilities for run logs and experiment markdown records."""

from datetime import datetime
import logging
from pathlib import Path
from typing import Mapping


def get_logger(run_name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    """Create and return a logger writing to file and stdout."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{run_name}.log"

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_experiment_result(
    run_name: str,
    epoch: int,
    metrics: Mapping[str, float | int | str],
    filepath: str = "outputs/EXPERIMENTS.md",
) -> None:
    """Append one experiment row to a markdown table file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    values: list[str] = []
    for value in metrics.values():
        if isinstance(value, float):
            values.append(f"{value:.4f}")
        else:
            values.append(str(value))

    row = f"| {run_name} | {epoch} | " + " | ".join(values) + f" | {timestamp} |\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(row)
