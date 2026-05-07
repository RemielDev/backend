"""Async retry helper for sync I/O calls (e.g. supabase-py).

Wraps a sync callable so transient connection failures (DNS hiccups during
Railway cold-start egress provisioning, idle TCP resets, brief gateway
outages) get retried with exponential backoff instead of bubbling up as
request errors. The callable runs on the default executor so it doesn't
block the event loop any more than the underlying sync client already does.
"""
import asyncio
import logging
from typing import Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TRANSIENT_MARKERS = (
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname provided",
    "connection",
    "timed out",
    "timeout",
    "eof",
    "broken pipe",
    "reset by peer",
)


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


async def with_retries(
    label: str,
    fn: Callable[[], T],
    retries: int = 0,
    backoff: float = 0.5,
) -> T:
    """Run `fn` in the default executor; retry transient failures with exponential backoff."""
    loop = asyncio.get_event_loop()
    for attempt in range(retries + 1):
        try:
            return await loop.run_in_executor(None, fn)
        except Exception as exc:
            if attempt == retries or not _is_transient(exc):
                raise
            sleep_for = backoff * (2 ** attempt)
            logger.warning(
                f"{label} failed (attempt {attempt + 1}/{retries + 1}): {exc}; "
                f"retrying in {sleep_for:.1f}s"
            )
            await asyncio.sleep(sleep_for)
    # unreachable — loop either returns or raises
    raise RuntimeError(f"{label}: retry loop exited unexpectedly")


async def try_with_retries(
    label: str,
    fn: Callable[[], T],
    retries: int = 0,
    backoff: float = 0.5,
) -> Optional[T]:
    """Like `with_retries` but logs and returns None on terminal failure instead of raising.

    Use this for non-critical I/O (e.g. supabase writes that should not break
    the request if Supabase is briefly unreachable).
    """
    try:
        return await with_retries(label, fn, retries=retries, backoff=backoff)
    except Exception as exc:
        logger.error(f"{label} skipped after retries exhausted: {exc}")
        return None
