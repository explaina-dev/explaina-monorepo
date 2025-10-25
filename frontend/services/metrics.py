from threading import Lock
from typing import Optional

_counters: dict[str, int] = {}
_lock = Lock()

def mark(key: str) -> None:
    with _lock:
        _counters[key] = _counters.get(key, 0) + 1

def dump() -> dict:
    with _lock:
        return {"counters": dict(_counters)}
