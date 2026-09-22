"""A single in-flight frame shared by two independently owned inference workers."""

import threading
from dataclasses import dataclass

import numpy as np

from .keypoint import KeypointObservation


@dataclass(frozen=True)
class CaptureFrame:
    """Carry frame identity and monotonic capture time through both branches."""

    sequence: int
    captured_at: float
    image: np.ndarray


class FrameBus:
    """Coordinate bounded tasks and frame-aligned outputs with short critical sections."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.frame = None
        self.prediction = {}
        self.failure = None
        self.closed = False

    def submit(self, frame: CaptureFrame) -> None:
        """Publish one frame after the preceding pair has been consumed."""
        with self.condition:
            self.frame = frame
            self.prediction = {}
            self.condition.notify_all()

    def wait_frame(self, previous: int) -> CaptureFrame | None:
        """Sleep until a new frame is available or shutdown wakes the worker."""
        with self.condition:
            self.condition.wait_for(lambda: self.closed or (
                self.frame is not None and self.frame.sequence > previous))
            return None if self.closed else self.frame

    def publish(self, branch: str, sequence: int, observation: KeypointObservation) -> None:
        """Publish only an output belonging to the active frame."""
        with self.condition:
            if not self.closed and self.frame.sequence == sequence:
                self.prediction[branch] = observation
                self.condition.notify_all()

    def wait_pair(self, timeout: float) -> dict | None:
        """Allow the coordinator to check cancellation while awaiting both branches."""
        with self.condition:
            self.condition.wait_for(
                lambda: self.failure is not None or self.closed or len(self.prediction) == 2,
                timeout,
            )
            if self.failure is not None:
                raise RuntimeError("Keypoint worker failed") from self.failure
            return dict(self.prediction) if len(self.prediction) == 2 else None

    def fail(self, exception: Exception) -> None:
        """Propagate a worker failure and release every waiter."""
        with self.condition:
            self.failure = exception
            self.closed = True
            self.condition.notify_all()

    def close(self) -> None:
        """Wake sleeping workers for cooperative shutdown."""
        with self.condition:
            self.closed = True
            self.condition.notify_all()
