"""
logger.py
---------
Single place that configures Python logging for the whole pipeline.
Every module calls get_logger(__name__) and gets a logger that writes
to both the console and a timestamped file under logs/.
"""

import logging
import os
import sys
from datetime import datetime

from src.config import config

_CONFIGURED = False
_LOG_FILE_PATH = None


def _configure_root_logger():
    global _CONFIGURED, _LOG_FILE_PATH
    if _CONFIGURED:
        return

    os.makedirs(config.LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE_PATH = os.path.join(config.LOGS_DIR, f"pipeline_{timestamp}.log")

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(_LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    return logging.getLogger(name)


def current_log_file() -> str:
    _configure_root_logger()
    return _LOG_FILE_PATH
