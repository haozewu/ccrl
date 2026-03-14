from __future__ import annotations

import logging
from pathlib import Path

import coloredlogs

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


class NullWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def create_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    coloredlogs.install(
        level="INFO",
        logger=logger,
        fmt="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path / "app.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
    return logger


def create_writer(log_dir: str | Path | None):
    if log_dir is None or SummaryWriter is None:
        return NullWriter()
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(path))
