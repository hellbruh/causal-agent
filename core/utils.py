from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from core.config import LoggingConfig


def configure_logging(
    log_path: Path,
    logging_cfg: LoggingConfig,
    add_stdout: bool = False,
    stdout_format: str | None = None,
) -> None:
    """Configure loguru logging for app entrypoints."""
    logger.remove()
    logger.add(
        log_path,
        rotation=logging_cfg.rotation,
        retention=logging_cfg.retention,
        level=logging_cfg.level,
        format=logging_cfg.format,
    )
    if add_stdout:
        logger.add(
            sys.stdout,
            level=logging_cfg.level,
            format=stdout_format or logging_cfg.format,
        )
