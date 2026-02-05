"""Utility decorators for BetaFit workflows."""

import time
from functools import wraps
from typing import Any, Callable, TypeVar

from utils.logging import logger

T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator for handling transient failures.

    Args:
        max_attempts (int): Maximum number of retry attempts. Defaults to 3.
        delay (float): Initial delay between retries in seconds. Defaults to 1.0.
        backoff_multiplier (float): Multiplier for exponential backoff. Defaults to 2.0.
        exceptions (tuple): Tuple of exception types to catch. Defaults to (Exception,).

    Returns:
        Callable: Decorated function with retry logic.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff_multiplier**attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected error in retry logic for {func.__name__}")

        return wrapper

    return decorator


def log_execution(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to log function execution start and end.

    Args:
        func (Callable): Function to decorate.

    Returns:
        Callable: Decorated function with logging.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        logger.debug(f"Starting execution of {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Completed execution of {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise

    return wrapper
