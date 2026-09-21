"""Camera inference with the training feature contract."""

import logging
import time
from collections import deque
from collections.abc import Callable, Iterator

import cv2
import numpy as np
import torch

from .dataset import weight_digest
from .feature import FeatureExtractor, pack_clip
from .model import ActionModel
from .setting import ROOT, checkpoint_path
from .train import CHECKPOINT_VERSION
from .upstream import PATCH_VERSION, REVISION

LOGGER = logging.getLogger(__name__)


def camera_prediction(setting: dict, stopped: Callable[[], bool]) -> Iterator[tuple]:
    """Yield camera images and probabilities without discarding slow observations."""
    path = checkpoint_path(setting)
    if not path.is_file():
        raise FileNotFoundError(f"Train {setting['model']['name']} first; missing {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    trained = saved["setting"]
    if (saved["version"] != CHECKPOINT_VERSION or saved["patch_version"] != PATCH_VERSION
            or saved["revision"] != REVISION[setting["model"]["name"]][1]):
        raise ValueError("Checkpoint architecture version changed; retrain")
    if trained["model"]["name"] != setting["model"]["name"]:
        raise ValueError("Checkpoint model does not match configuration")
    for key in ("object_class", "object_point_index", "confidence", "sample_fps", "pose_size"):
        if trained["feature"][key] != setting["feature"][key]:
            raise ValueError(f"feature.{key} differs from training; restore it or retrain")
    if saved["object_weight_digest"] != weight_digest(ROOT / setting["feature"]["object_weight"]):
        raise ValueError("Object pose weights changed since training; retrain")
    trained["model"]["device"] = setting["model"]["device"]
    trained["feature"]["pose_device"] = setting["feature"]["pose_device"]
    trained["feature"]["object_weight"] = setting["feature"]["object_weight"]
    device = torch.device(trained["model"]["device"])
    network = ActionModel(trained, saved["object_point_count"]).to(device)
    network.load_state_dict(saved["state_dict"], strict=True)
    network.eval()
    extractor = FeatureExtractor(trained)
    option = setting["run"]
    capture = cv2.VideoCapture(option["camera"])
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    window_frames = trained["train"]["window_frames"]
    frames = deque(maxlen=window_frames)
    interval = 1 / trained["feature"]["sample_fps"]
    next_sample, last_log = 0.0, 0.0
    LOGGER.info("Camera=%s model=%s window=%d; first observation pads initial history",
                option["camera"], setting["model"]["name"], frames.maxlen)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {option['camera']}")
        while not stopped():
            available, frame = capture.read()
            if not available:
                raise RuntimeError("Camera stopped delivering frames")
            now = time.monotonic()
            if now < next_sample:
                continue
            extracted = extractor.extract(frame)
            if not frames:
                frames.extend([extracted] * (window_frames - 1))
            frames.append(extracted)
            next_sample = now + interval
            observation = {key: np.stack([item[key] for item in frames]) for key in frames[0]}
            human, object_feature = pack_clip(observation, trained)
            with torch.inference_mode():
                probability = network(torch.from_numpy(human[None]).to(device),
                                      torch.from_numpy(object_feature[None]).to(device))
            confidence = float(probability.exp()[0, 1])
            if not np.isfinite(confidence):
                raise RuntimeError("Action model returned a non-finite probability")
            elapsed = time.monotonic() - now
            if now - last_log >= option["log_interval_seconds"]:
                LOGGER.info("action=%s confidence=%.4f detected=%s inference_ms=%.1f",
                            option["action_name"], confidence, confidence >= option["threshold"],
                            elapsed * 1000)
                last_log = now
            yield frame, confidence, elapsed
    finally:
        capture.release()
        LOGGER.info("Camera released")


def run_camera(setting: dict) -> None:
    """Run the responsive desktop recognition preview."""
    from .preview import run_preview

    run_preview(setting)
