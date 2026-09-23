"""Cached TensorRT ResNet50 descriptors with the pretrained image transform."""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import functional as image_transform

from ..recognition.engine import engine_directory, load_runtime
from ..setting import ROOT

LOGGER = logging.getLogger(__name__)
CROP_SIZE = 224
RESIZE_SIZE = 232
BATCH_SIZE = 2
FEATURE_SIZE = 2048
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def appearance_engine(setting: dict) -> Path:
    """Export the frozen encoder and cache a static-batch FP16 TensorRT engine."""
    import tensorrt as trt

    torch.cuda.set_device(torch.device(setting["model"]["device"]))
    onnx = ROOT / "temp" / "export" / "resnet50" / "imagenet1k_v2_batch2_v1.onnx"
    if not onnx.is_file():
        onnx.parent.mkdir(parents=True, exist_ok=True)
        encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
        encoder.fc = torch.nn.Identity()
        pending = onnx.with_suffix(".partial.onnx")
        with torch.inference_mode():
            torch.onnx.export(encoder, torch.zeros(BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE),
                              str(pending), input_names=["crop"], output_names=["appearance"],
                              opset_version=17, dynamo=False)
        pending.replace(onnx)
    cache_key = engine_directory(onnx, setting).name
    target = ROOT / "data" / "model" / "resnet50" / "engine" / cache_key / "appearance.engine"
    if target.is_file():
        return target
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError("ResNet50 ONNX parse failed: " + "; ".join(
            str(parser.get_error(index)) for index in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                setting["recognition"]["human"]["workspace_mb"] * 1024 * 1024)
    LOGGER.info("Building ResNet50 TensorRT FP16 batch=%d engine=%s", BATCH_SIZE, target)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("ResNet50 TensorRT engine build failed")
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(".partial")
    pending.write_bytes(bytes(serialized))
    pending.replace(target)
    return target


class AppearanceEncoder:
    """Reuse pinned staging memory and device buffers for two independent crops."""

    def __init__(self, setting: dict) -> None:
        load_runtime()
        import tensorrt as trt

        self.device = torch.device(setting["model"]["device"])
        torch.cuda.set_device(self.device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        path = appearance_engine(setting)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize appearance engine: {path}")
        for name, shape in (("crop", (BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE)),
                            ("appearance", (BATCH_SIZE, FEATURE_SIZE))):
            if (tuple(self.engine.get_tensor_shape(name)) != shape
                    or self.engine.get_tensor_dtype(name) != trt.float32):
                raise RuntimeError(f"Unexpected appearance engine tensor contract: {name}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create appearance execution context")
        self.host = torch.zeros((BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE), pin_memory=True)
        self.crop = torch.empty_like(self.host, device=self.device)
        self.output = torch.empty((BATCH_SIZE, FEATURE_SIZE), device=self.device)
        self.pixels = self.host.numpy()
        self.context.set_tensor_address("crop", self.crop.data_ptr())
        self.context.set_tensor_address("appearance", self.output.data_ptr())
        self.encode()
        LOGGER.info("Appearance ready backend=TensorRT precision=FP16 batch=%d engine=%s",
                    BATCH_SIZE, path)

    def prepare_crop(self, crop: np.ndarray, slot: int) -> None:
        """Keep PIL bilinear resize/crop semantics and normalize into reusable NCHW memory."""
        image = Image.fromarray(np.ascontiguousarray(crop[:, :, ::-1]))
        image = image_transform.resize(image, RESIZE_SIZE)
        image = image_transform.center_crop(image, [CROP_SIZE, CROP_SIZE])
        np.divide(np.asarray(image).transpose(2, 0, 1), np.float32(255),
                  out=self.pixels[slot])
        self.pixels[slot] -= MEAN
        self.pixels[slot] /= STD

    @torch.inference_mode()
    def encode(self) -> np.ndarray:
        """Complete asynchronous inference before reusing the host staging buffer."""
        self.crop.copy_(self.host, non_blocking=True)
        if not self.context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream):
            raise RuntimeError("ResNet50 TensorRT execution failed")
        return self.output.cpu().numpy()
