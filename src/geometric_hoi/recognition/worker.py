"""Human and object workers that never execute business logic."""

import logging
import threading

import torch

from .bus import FrameBus

LOGGER = logging.getLogger(__name__)


class KeypointWorker(threading.Thread):
    """Exclusively own one estimator and execute inference outside the bus lock."""

    def __init__(self, branch: str, estimator, bus: FrameBus, device: str) -> None:
        super().__init__(name=f"{branch}-keypoint")
        self.branch = branch
        self.estimator = estimator
        self.bus = bus
        self.device = device

    def run(self) -> None:
        """Consume each dispatched frame once on an independent CUDA stream."""
        try:
            torch.cuda.set_device(self.device)
            stream = torch.cuda.Stream(device=self.device)
            previous = -1
            with torch.cuda.stream(stream):
                while (frame := self.bus.wait_frame(previous)) is not None:
                    observation = self.estimator.estimate(frame.image)
                    self.bus.publish(self.branch, frame.sequence, observation)
                    previous = frame.sequence
        except Exception as exc:
            LOGGER.exception("%s keypoint worker failed", self.branch)
            self.bus.fail(exc)
