from __future__ import annotations

import contextlib
import signal
import threading
from typing import Iterator, Optional

APPLY_SECONDS = 30.0
SCREEN_SECONDS = 120.0


class Overtime(Exception):
    pass


@contextlib.contextmanager
def deadline(seconds: Optional[float], what: str = "a step"
             ) -> Iterator[None]:
    if not seconds or seconds <= 0 \
            or threading.current_thread() is not threading.main_thread():
        yield
        return

    def expire(_signum, _frame):
        raise Overtime("%s took over %gs" % (what, seconds))

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


__all__ = ["APPLY_SECONDS", "Overtime", "SCREEN_SECONDS", "deadline"]
