"""Centralized logging configuration for BetaFit."""

import json
import logging
from typing import Any


# Configure logger
logger = logging.getLogger("betafits")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_event(
    workflow_id: str,
    node_id: str,
    event: str,
    details: dict[str, Any] | None = None,
) -> None:
    """
    Log a structured workflow event.

    Args:
        workflow_id (str): Unique workflow identifier.
        node_id (str): Node/step identifier.
        event (str): Event description.
        details (dict[str, Any] | None): Additional event details.
    """
    payload = {
        "workflow_id": workflow_id,
        "node_id": node_id,
        "event": event,
        "details": details or {},
    }
    logger.info(json.dumps(payload))


def log_error(
    workflow_id: str,
    node_id: str,
    error: Exception,
    context: dict[str, Any] | None = None,
) -> None:
    """
    Log an error with context.

    Args:
        workflow_id (str): Unique workflow identifier.
        node_id (str): Node/step identifier where error occurred.
        error (Exception): The exception that was raised.
        context (dict[str, Any] | None): Additional context about the error.
    """
    payload = {
        "workflow_id": workflow_id,
        "node_id": node_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {},
    }
    logger.error(json.dumps(payload))


def set_log_level(level: str) -> None:
    """
    Set the logging level.

    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logger.setLevel(numeric_level)
