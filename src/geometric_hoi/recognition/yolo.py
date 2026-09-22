"""Single-frame Ultralytics inference without global profiling barriers."""

import logging
import time

import numpy as np
import torch
from ultralytics import YOLO

from .engine import INPUT_SIZE
from ..performance import PerformanceWindow

LOGGER = logging.getLogger(__name__)


class YoloFrame:
    """Own a warmed predictor and preserve its task-specific pre/postprocessing."""

    @torch.inference_mode()
    def __init__(self, model: YOLO, device: str, confidence: float, class_id: int) -> None:
        model.predict(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8),
                      device=device, conf=confidence, classes=[class_id],
                      imgsz=INPUT_SIZE, rect=False, verbose=False)
        self.predictor = model.predictor
        self.backend = self.predictor.model.backend
        if self.backend.dynamic or not self.backend.is_trt10:
            raise RuntimeError("YOLO frame execution requires a static TensorRT 10 engine")
        self.input = self.backend.bindings["images"].data
        self.outputs = [self.backend.bindings[name].data for name in sorted(self.backend.output_names)]
        for name, binding in self.backend.bindings.items():
            if not self.backend.context.set_tensor_address(name, binding.data.data_ptr()):
                raise RuntimeError(f"Cannot bind YOLO tensor {name}")
        self.gpu_started = torch.cuda.Event(enable_timing=True)
        self.gpu_finished = torch.cuda.Event(enable_timing=True)
        self.performance = PerformanceWindow(f"yolo_{self.predictor.args.task}", 1.0)
        LOGGER.info("YOLO frame predictor ready task=%s fp16=%s execution=execute_async_v3; "
                    "gpu_engine is CUDA event time, other stages are wall-clock",
                    self.predictor.args.task, self.predictor.model.fp16)

    @torch.inference_mode()
    def predict(self, frame: np.ndarray):
        """Execute on the worker's stream without stream_inference's device synchronization."""
        started = time.perf_counter()
        self.predictor.batch = (["camera"], [frame], [""])
        tensor = self.predictor.preprocess([frame])
        prepared_at = time.perf_counter()
        self.input.copy_(tensor)
        self.gpu_started.record()
        if not self.backend.context.execute_async_v3(torch.cuda.current_stream(tensor.device).cuda_stream):
            raise RuntimeError("YOLO TensorRT execution failed")
        self.gpu_finished.record()
        submitted_at = time.perf_counter()
        outputs = self.outputs[0] if len(self.outputs) == 1 else self.outputs
        prediction = self.predictor.postprocess(outputs, tensor, [frame])[0].cpu()
        finished = time.perf_counter()
        self.gpu_finished.synchronize()
        self.performance.record({"preprocess_submit": prepared_at - started,
                                 "input_copy_enqueue": submitted_at - prepared_at,
                                 "gpu_engine": self.gpu_started.elapsed_time(self.gpu_finished) / 1000,
                                 "postprocess_download_wait": finished - submitted_at,
                                 "total": time.perf_counter() - started})
        return prediction
