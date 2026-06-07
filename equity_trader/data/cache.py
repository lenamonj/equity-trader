from contextvars import ContextVar
from functools import wraps
from typing import Callable, Any

_run_cache: ContextVar[dict | None] = ContextVar("run_cache", default=None)


def reset_run_cache() -> None:
    """Start a fresh per-run cache for the current ContextVar context."""
    _run_cache.set({})


def _get_cache() -> dict:
    cache = _run_cache.get()
    if cache is None:
        cache = {}
        _run_cache.set(cache)
    return cache


def _make_hashable(v: Any) -> Any:
    """Recursively convert lists/dicts to hashable equivalents."""
    if isinstance(v, list):
        return tuple(_make_hashable(i) for i in v)
    if isinstance(v, dict):
        return tuple(sorted((_make_hashable(k), _make_hashable(val)) for k, val in v.items()))
    return v


def cached_for_run(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (func.__name__,
               tuple(_make_hashable(a) for a in args),
               tuple(sorted((k, _make_hashable(v)) for k, v in kwargs.items())))
        cache = _get_cache()
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper
