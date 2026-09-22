"""Appearance and temporal features shared by training and live recognition."""

import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from ..recognition.engine import load_runtime
from ..performance import PerformanceWindow
from ..recognition.keypoint import HumanKeypoint, KeypointObservation, ObjectKeypoint


class AppearanceFeature:
    """Encode the crops belonging to one aligned human/object observation."""

    def __init__(self, setting: dict) -> None:
        self.device = torch.device(setting["model"]["device"])
        weight = ResNet50_Weights.IMAGENET1K_V2
        self.encoder = resnet50(weights=weight).to(self.device).eval()
        self.encoder.fc = torch.nn.Identity()
        self.transform = weight.transforms()
        self.performance = PerformanceWindow("appearance", setting["run"]["log_interval_seconds"])

    @torch.inference_mode()
    def extract(self, frame: np.ndarray, human: KeypointObservation,
                target: KeypointObservation) -> dict[str, np.ndarray]:
        """Select COCO17 from RTMW and attach aligned frozen crop descriptors."""
        started = time.perf_counter()
        height, width = frame.shape[:2]
        appearance = np.zeros((2, 2048), dtype=np.float32)
        crops, slots = [], []
        for slot, observation in enumerate((human, target)):
            if observation.box is None:
                continue
            x1, y1, x2, y2 = observation.box.astype(int)
            crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            if crop.size:
                crops.append(self.transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
                slots.append(slot)
        prepared_at = time.perf_counter()
        if crops:
            appearance[slots] = self.encoder(torch.stack(crops).to(self.device)).cpu().numpy()
        finished = time.perf_counter()
        duration = {"crop_preprocess": prepared_at - started, "total": finished - started}
        if crops:
            duration[f"resnet_batch_{len(crops)}_transfer_inference"] = finished - prepared_at
        self.performance.record(duration)
        return {"human_point": human.point[:17], "object_point": target.point,
                "appearance": appearance}


class FeatureExtractor:
    """Extract offline training features using the same models as live inference."""

    def __init__(self, setting: dict) -> None:
        load_runtime()
        self.human = HumanKeypoint(setting)
        self.target = ObjectKeypoint(setting)
        self.appearance = AppearanceFeature(setting)
        self.object_point_count = self.target.point_count

    def reset(self) -> None:
        """Forget associations between independent training videos."""
        self.human.center = None
        self.target.center = None

    def extract(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Extract one complete training observation."""
        return self.appearance.extract(frame, self.human.estimate(frame), self.target.estimate(frame))


def pack_clip(observation: dict[str, np.ndarray], setting: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build official x/y/velocity geometry using past observations only."""
    human = observation["human_point"]
    selected = observation["object_point"][:, setting["feature"]["object_point_index"]]
    point = np.concatenate((human, selected), axis=1)
    coordinate = point[:, :, :2]
    velocity = np.zeros_like(coordinate)
    visible_pair = (point[1:, :, 2] > 0) & (point[:-1, :, 2] > 0)
    elapsed = np.diff(observation["timestamp"])[:, None, None]
    np.divide(coordinate[1:] - coordinate[:-1], elapsed,
              out=velocity[1:], where=elapsed > 0)
    velocity[1:] *= visible_pair[:, :, None]
    geometry = np.concatenate((coordinate, velocity), axis=-1).reshape(len(point), -1)
    human_feature = np.concatenate((observation["appearance"][:, 0], geometry), axis=-1)
    object_feature = np.concatenate((observation["appearance"][:, 1],
                                     observation["object_point"].reshape(len(point), -1)), axis=-1)
    return human_feature[:, None].astype(np.float32), object_feature[:, None].astype(np.float32)
