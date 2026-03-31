"""Exponential backoff retry decorator."""
import functools
import logging
import random
import time
from typing import Type

logger = logging.getLogger(__name__)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    job_id: str = "",
):
    """Decorator: retry with exponential backoff + jitter.

    Usage:
        @with_retry(max_attempts=3, exceptions=(HttpError,))
        def call_api(...): ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts - 1:
                        break
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "[%s] %s attempt %d/%d failed: %s — retrying in %.1fs",
                        job_id or "global",
                        fn.__name__,
                        attempt + 1,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            logger.error(
                "[%s] %s failed after %d attempts: %s",
                job_id or "global",
                fn.__name__,
                max_attempts,
                last_exc,
            )
            raise last_exc
        return wrapper
    return decorator
