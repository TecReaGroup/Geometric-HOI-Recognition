"""Prepare model-specific TensorRT artifacts and runtime libraries."""

import hashlib
import logging
import os
import shutil
import site
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from rtmlib import RTMPose
from ultralytics import YOLO

from ..setting import ROOT

LOGGER = logging.getLogger(__name__)
MODEL_ROOT = ROOT / "data" / "model"
RTMW_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
    "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip"
)
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt"
DLL_DIRECTORY = []
INPUT_SIZE = 640


def load_runtime() -> None:
    """Load the CUDA and TensorRT libraries before creating worker sessions."""
    if sys.platform == "win32" and not DLL_DIRECTORY:
        for directory in site.getsitepackages():
            library = Path(directory) / "tensorrt_libs"
            if library.is_dir():
                DLL_DIRECTORY.append(os.add_dll_directory(str(library)))
                os.environ["PATH"] = str(library) + os.pathsep + os.environ["PATH"]
    ort.preload_dlls(directory="")
    import tensorrt

    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime TensorRT EP unavailable; run make install")
    LOGGER.info("TensorRT=%s ONNX Runtime=%s", tensorrt.__version__, ort.__version__)


def fingerprint(path: Path) -> str:
    """Identify the exact source model for engine invalidation."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def engine_directory(weight: Path, setting: dict) -> Path:
    """Separate cached engines by model, GPU, runtime and build settings."""
    import tensorrt

    device = torch.device(setting["model"]["device"])
    signature = (
        fingerprint(weight), torch.cuda.get_device_name(device),
        torch.cuda.get_device_capability(device), tensorrt.__version__, ort.__version__,
        setting["recognition"]["human"]["workspace_mb"], INPUT_SIZE, "fp16-v1",
    )
    key = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
    directory = weight.parent / "engine" / key
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def download_weight(url: str, target: Path) -> None:
    """Download and atomically publish a model asset."""
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / "temp" / "download"
    temporary.mkdir(parents=True, exist_ok=True)
    pending = temporary / (target.name + ".part")
    LOGGER.info("Downloading %s", target)
    try:
        with urllib.request.urlopen(url, timeout=60) as source, pending.open("wb") as sink:
            shutil.copyfileobj(source, sink)
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)


def rtmw_weight() -> Path:
    """Prepare RTMW-X whole-body ONNX under its own model directory."""
    target = MODEL_ROOT / "rtmw-x" / "rtmw-x.onnx"
    if target.is_file():
        return target
    archive = ROOT / "temp" / "download" / "rtmw-x.zip"
    download_weight(RTMW_URL, archive)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = archive.with_suffix(".onnx.part")
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if name.endswith(".onnx")]
            if len(members) != 1:
                raise RuntimeError("RTMW archive must contain exactly one ONNX model")
            with bundle.open(members[0]) as source, pending.open("wb") as sink:
                shutil.copyfileobj(source, sink)
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)
        archive.unlink(missing_ok=True)
    return target


def yolo_engine(weight: Path, setting: dict) -> Path:
    """Export a static FP16 YOLO engine while keeping intermediate files in temp."""
    target = engine_directory(weight, setting) / f"{weight.stem}.engine"
    if target.is_file():
        return target
    temporary = ROOT / "temp" / "export" / weight.parent.name
    temporary.mkdir(parents=True, exist_ok=True)
    copied = temporary / weight.name
    shutil.copyfile(weight, copied)
    LOGGER.info("Building FP16 engine %s", target)
    exported = Path(YOLO(str(copied)).export(
        format="engine", imgsz=INPUT_SIZE, batch=1, dynamic=False, half=True,
        simplify=False, opset=17, device=setting["model"]["device"],
        workspace=setting["recognition"]["human"]["workspace_mb"] / 1024,
    ))
    exported.replace(target)
    LOGGER.info("Engine persisted: %s", target)
    return target


def person_detector_weight() -> Path:
    """Prepare the official YOLO26m person detector."""
    weight = MODEL_ROOT / "yolo26m" / "yolo26m.pt"
    download_weight(YOLO_URL, weight)
    return weight


def load_rtmw(setting: dict) -> RTMPose:
    """Build or load RTMW TensorRT partitions and warm up batch-one inference."""
    weight = rtmw_weight()
    cache = engine_directory(weight, setting)
    estimator = RTMPose(str(weight), model_input_size=(192, 256),
                        to_openpose=False, backend="onnxruntime", device="cpu")
    device_id = torch.device(setting["model"]["device"]).index or 0
    estimator.session.disable_fallback()
    LOGGER.info("Building/loading RTMW FP16 engine: %s", cache)
    estimator.session.set_providers([
        ("TensorrtExecutionProvider", {
            "device_id": device_id,
            "trt_fp16_enable": True,
            "trt_max_workspace_size": setting["recognition"]["human"]["workspace_mb"] * 1024 * 1024,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache),
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(cache),
        }),
        ("CUDAExecutionProvider", {"device_id": device_id}),
        "CPUExecutionProvider",
    ])
    if "TensorrtExecutionProvider" not in estimator.session.get_providers():
        raise RuntimeError("RTMW TensorRT EP initialization failed")
    estimator.inference(np.zeros((256, 192, 3), dtype=np.float32))
    if not list(cache.glob("*.engine")):
        raise RuntimeError(f"RTMW produced no TensorRT engine in {cache}")
    LOGGER.info("RTMW ready: providers=%s", estimator.session.get_providers())
    return estimator


def prepare_engine(setting: dict) -> None:
    """Build recognition and appearance engines without opening a camera."""
    from ..action.appearance import appearance_engine

    load_runtime()
    yolo_engine(person_detector_weight(), setting)
    yolo_engine(ROOT / setting["recognition"]["object"]["object_weight"], setting)
    load_rtmw(setting)
    appearance_engine(setting)
    LOGGER.info("YOLO26m, RTMW-X, YOLO26-pose and ResNet50 engines are ready")
