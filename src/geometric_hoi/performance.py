"""Bounded wall-clock stage statistics owned by the calling thread."""

import logging
import math
import time
from collections import defaultdict, deque
from statistics import fmean

from .logging import PERF

LOGGER = logging.getLogger(__name__)
MAX_STAGE_SAMPLES = 4096


class PerformanceWindow:
    """Report first-call latency separately from periodic steady-call statistics."""

    def __init__(self, scope: str, interval: float) -> None:
        self.scope = scope
        self.interval = interval
        self.started = None
        self.completed = 0
        self.samples = defaultdict(lambda: deque(maxlen=MAX_STAGE_SAMPLES))

    def record(self, duration: dict[str, float]) -> None:
        """Record seconds per stage; FPS measures completed calls over wall time."""
        if not LOGGER.isEnabledFor(PERF):
            return
        now = time.perf_counter()
        if self.started is None:
            LOGGER.log(PERF, "performance scope=%s first_call_ms=%s", self.scope,
                        " ".join(f"{name}:{seconds * 1000:.2f}" for name, seconds in duration.items()))
            self.started = now
            return
        self.completed += 1
        for name, seconds in duration.items():
            self.samples[name].append(seconds * 1000)
        elapsed = now - self.started
        if elapsed < self.interval:
            return
        stage = []
        for name, samples in self.samples.items():
            values = sorted(samples)
            stage.append(f"{name}[n={len(values)},avg={fmean(values):.2f},"
                         f"p95={values[math.ceil(len(values) * 0.95) - 1]:.2f},max={values[-1]:.2f}]")
        LOGGER.log(PERF, "performance scope=%s frames=%d window_s=%.2f fps=%.2f stage_ms=%s",
                    self.scope, self.completed, elapsed, self.completed / elapsed, " ".join(stage))
        self.started = now
        self.completed = 0
        self.samples.clear()
