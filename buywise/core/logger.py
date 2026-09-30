"""
Logging configuration for the BUYWISE agent.
Call setup_logging() once at application startup.
"""

import logging
import sys

from buywise.core.config import LOG_LEVEL


def setup_logging(level: str | None = None) -> None:
    """Configure root logger with a clean, informative format."""
    lvl = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "groq"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
