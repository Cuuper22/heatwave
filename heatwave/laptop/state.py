"""Small shared pieces: guarded threads, JSON cleaning, run state."""

from __future__ import annotations

import json
import math
import threading
import time
import traceback
from collections import deque
from typing import Any, Callable, Optional


def to_jsonable(value: Any) -> Any:
    """Make numpy values JSON-safe; NaN and inf become None."""
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return to_jsonable(value.tolist())
        if isinstance(value, np.generic):
            return to_jsonable(value.item())
    except ImportError:
        pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def json_text(value: Any, indent: Optional[int] = None) -> str:
    return json.JSONEncoder(allow_nan=False, indent=indent).encode(to_jsonable(value))


class GuardedThread(threading.Thread):
    """Thread that runs ``loop()`` and records any exception.

    On an exception the shared stop event is set so the whole run winds down
    (and the controller releases the GPU) instead of failing silently.
    """

    def __init__(self, name: str, stop_event: threading.Event) -> None:
        super().__init__(name=name, daemon=True)
        self.stop_event = stop_event
        self.error: Optional[str] = None

    def loop(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def run(self) -> None:
        try:
            self.loop()
        except BaseException:  # noqa: BLE001 - recorded and surfaced
            self.error = traceback.format_exc()
            print(f"[{self.name}] stopped with an error:\n{self.error}", flush=True)
            self.stop_event.set()


class RunState:
    """Latest forecast, controller status and run metadata, shared by threads."""

    def __init__(self, meta: dict) -> None:
        self.meta = dict(meta)
        self._lock = threading.Lock()
        self._forecast = None
        self._control: dict = {}
        self.tracker = None  # ScheduleTracker, set by run.py
        self.messages: deque = deque(maxlen=50)

    def set_forecast(self, rec) -> None:
        with self._lock:
            self._forecast = rec

    def get_forecast(self):
        with self._lock:
            return self._forecast

    def set_control(self, status: dict) -> None:
        with self._lock:
            self._control = dict(status)

    def get_control(self) -> dict:
        with self._lock:
            return dict(self._control)

    def say(self, text: str) -> None:
        with self._lock:
            self.messages.append((time.time(), text))
