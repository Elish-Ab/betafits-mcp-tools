"""Validation helpers shared across services and entrypoints."""

from __future__ import annotations


def validate_user_message(message: str, *, max_length: int = 2000) -> str:
    """
    Validate a user-provided message before sending it to any LLM workflow.

    Args:
        message: Raw message from CLI, Slack, etc.
        max_length: Optional cap to guard against prompt-injection vectors.

    Returns:
        Sanitised message trimmed of surrounding whitespace.

    Raises:
        ValueError: If the message is empty or exceeds the configured limit.
    """
    if not isinstance(message, str):
        raise ValueError("Message must be a string.")

    cleaned = message.strip()
    if not cleaned:
        raise ValueError("Message cannot be empty.")

    if len(cleaned) > max_length:
        raise ValueError(f"Message exceeds {max_length} characters.")

    return cleaned
