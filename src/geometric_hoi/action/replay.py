"""Fixed-window CUDA replay for action recognition networks."""

import logging

import torch

from .model import GEOMETRY_SIZE

LOGGER = logging.getLogger(__name__)
WARMUP_ITERATIONS = 3


class ActionReplay:
    """Own persistent inputs and outputs for one batch-one action graph."""

    @torch.inference_mode()
    def __init__(self, network: torch.nn.Module, window: int, object_points: int,
                 device: torch.device) -> None:
        self.human = torch.zeros((1, window, 1, 2048 + GEOMETRY_SIZE), device=device)
        self.target = torch.zeros((1, window, 1, 2048 + object_points * 3), device=device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(WARMUP_ITERATIONS):
                network(self.human, self.target)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = network(self.human, self.target)
        torch.cuda.current_stream(device).wait_stream(stream)
        LOGGER.info("Action CUDA graph ready: window=%d dtype=%s", window, self.human.dtype)

    @torch.inference_mode()
    def __call__(self, human: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Copy the current clip and replay without Python per-step dispatch."""
        self.human.copy_(human)
        self.target.copy_(target)
        self.graph.replay()
        return self.output
