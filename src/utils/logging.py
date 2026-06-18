"""Structured logging via structlog.

- Production (``dev_mode=False``): JSON lines, CloudWatch/Datadog friendly.
- Development (``dev_mode=True``): pretty, colourised console output.

Usage:
    from src.utils.logging import configure_logging, get_logger

    configure_logging(dev_mode=True)
    log = get_logger(__name__)
    log.info("model_loaded", model="ChurnModel", stage="Production")
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(dev_mode: bool = True) -> None:
    """Configure structlog + stdlib logging once at process startup.

    Idempotent enough for repeated calls (e.g. tests), but intended to be
    called a single time from the application entry point.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    # Native (non-stdlib) processors so they work with PrintLoggerFactory.
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if dev_mode:
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Route the stdlib root logger to stdout too, so libraries that use
    # `logging` (uvicorn, mlflow) share a consistent stream/level.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )


def get_logger(name: str | None = None):
    """Return a bound structlog logger, optionally named (use ``__name__``).

    The name is bound into the event dict as ``logger`` so it shows up in
    both JSON and console output.
    """
    logger = structlog.get_logger()
    if name is not None:
        logger = logger.bind(logger=name)
    return logger
