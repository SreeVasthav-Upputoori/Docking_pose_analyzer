"""Utilities: logging, progress tracking, and common helpers."""

import logging
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("docking_consistency")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_file = output_dir / "pose_consistency.log"
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("docking_consistency")


class ProgressTracker:
    def __init__(self, total: int, desc: str = "Processing"):
        self.pbar = tqdm(total=total, desc=desc, unit="mol", ncols=100)

    def update(self, n: int = 1):
        self.pbar.update(n)

    def close(self):
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
