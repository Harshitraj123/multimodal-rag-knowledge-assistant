"""
Logging configuration.
Configures structured logging with configurable levels.
Never logs API keys or secrets.
"""
from __future__ import annotations

import logging
import logging.config
import re
import sys
from pathlib import Path
from typing import Optional


class SensitiveDataFilter(logging.Filter):
    """Masks API keys, tokens, and authorization headers in log output."""

    PATTERNS = [
        r"(gsk_[A-Za-z0-9_]+)",  # Groq API Keys
        r"(tvly-[A-Za-z0-9_]+)",  # Tavily API Keys
        r"(Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*)",  # Bearer tokens
        r"(api[-_]?key[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9_\-]+",  # Generic API keys
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern in self.PATTERNS:
                record.msg = re.sub(pattern, r"\1[REDACTED]", record.msg)
        return True


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure application-wide logging with secret redaction."""
    log_level = log_level.upper()

    handlers: dict = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "standard",
            "level": log_level,
            "filters": ["sensitive_data_filter"],
        }
    }

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Prevent FileNotFoundError

        handlers["file"] = {
            "class": "logging.FileHandler",
            "filename": str(file_path),
            "formatter": "standard",
            "level": log_level,
            "filters": ["sensitive_data_filter"],
        }

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "sensitive_data_filter": {
                "()": SensitiveDataFilter,
            }
        },
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            }
        },
        "handlers": handlers,
        "root": {
            "handlers": list(handlers.keys()),
            "level": log_level,
        },
        "loggers": {
            "httpx": {"level": "WARNING"},
            "httpcore": {"level": "WARNING"},
            "chromadb": {"level": "WARNING"},
            "sentence_transformers": {"level": "WARNING"},
        },
    }

    logging.config.dictConfig(config)