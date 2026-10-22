"""
Dual-stream logging utilities for experiments.

Provides safe redirection of stdout/stderr to both console and a disk log file,
preventing infinite recursion and supporting clean notebook execution.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, TextIO

__all__ = ["DualLogger", "setup_logging", "close_logging"]


class DualLogger:
    """Redirects stdout/stderr to both console and a disk log file safely without recursion."""

    def __init__(self, terminal_stream: TextIO, filepath: str = "experiment_logs.txt"):
        self.terminal = terminal_stream
        self.filepath = filepath
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self.log_file = open(filepath, "a", encoding="utf-8")
        self._is_active = True

    def write(self, message: str) -> None:
        if not self._is_active:
            return
        try:
            self.terminal.write(message)
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log_file.write(message)
            self.log_file.flush()
        except Exception:
            pass

    def flush(self) -> None:
        try:
            self.terminal.flush()
            self.log_file.flush()
        except Exception:
            pass

    def close(self) -> None:
        self._is_active = False
        try:
            self.log_file.close()
        except Exception:
            pass


def setup_logging(
    log_filename: str = "experiment_logs.txt",
    configure_cuda: bool = True,
) -> str:
    """
    Initializes dual-stream logging to console and disk file.

    Args:
        log_filename:   Path to the log file (default: "experiment_logs.txt").
        configure_cuda: If True, sets PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True".

    Returns:
        Absolute path to the log file.
    """
    if configure_cuda:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    log_dir = os.path.dirname(log_filename)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    if isinstance(sys.stdout, DualLogger):
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal
    if isinstance(sys.stderr, DualLogger):
        sys.stderr.close()
        sys.stderr = sys.stderr.terminal

    sys.stdout = DualLogger(sys.__stdout__, log_filename)
    sys.stderr = DualLogger(sys.__stderr__, log_filename)
    print(f"[*] Dual logging established -> {os.path.abspath(log_filename)}")
    return os.path.abspath(log_filename)


def close_logging() -> None:
    """Restores standard sys.stdout and sys.stderr streams and closes log files."""
    if isinstance(sys.stdout, DualLogger):
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal
    if isinstance(sys.stderr, DualLogger):
        sys.stderr.close()
        sys.stderr = sys.stderr.terminal
