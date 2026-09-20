"""Camera inference with the training feature contract."""

import logging
import time
from collections import deque

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


def run_camera(setting: dict) -> None:
    """Display and log target-action probabilities from recent camera frames."""
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
    frames = deque(maxlen=trained["train"]["window_frames"])
    interval = 1 / trained["feature"]["sample_fps"]
    next_sample, last_sample, last_log = 0.0, None, 0.0
    confidence = None
    LOGGER.info("Camera=%s model=%s window=%d; press Q or Escape to exit",
                option["camera"], setting["model"]["name"], frames.maxlen)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {option['camera']}")
        while True:
            available, frame = capture.read()
            if not available:
                raise RuntimeError("Camera stopped delivering frames")
            now = time.monotonic()
            if now >= next_sample:
                if last_sample is not None and now - last_sample > interval * 2:
                    frames.clear()
                    confidence = None
                    extractor.reset()
                    LOGGER.warning("Camera/inference too slow for %.1f FPS; resetting temporal window",
                                   trained["feature"]["sample_fps"])
                frames.append(extractor.extract(frame))
                last_sample = now
                next_sample = now + interval
                if len(frames) == frames.maxlen:
                    observation = {key: np.stack([item[key] for item in frames]) for key in frames[0]}
                    human, object_feature = pack_clip(observation, trained)
                    with torch.inference_mode():
                        probability = network(torch.from_numpy(human[None]).to(device),
                                              torch.from_numpy(object_feature[None]).to(device))
                    confidence = float(probability.exp()[0, 1])
                if now - last_log >= option["log_interval_seconds"]:
                    if confidence is None:
                        LOGGER.info("Warming up %d/%d frames", len(frames), frames.maxlen)
                    else:
                        LOGGER.info("action=%s confidence=%.4f detected=%s", option["action_name"],
                                    confidence, confidence >= option["threshold"])
                    last_log = now
            caption = (f"{option['action_name']}: {confidence:.1%}" if confidence is not None
                       else f"Warming up {len(frames)}/{frames.maxlen}")
            color = (0, 220, 0) if confidence is not None and confidence >= option["threshold"] else (0, 200, 255)
            cv2.putText(frame, caption, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Geometric HOI", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
