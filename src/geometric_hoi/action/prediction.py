"""Own checkpoint validation, temporal history and action decisions."""

import logging
from collections import deque

import numpy as np
import torch

from ..recognition.engine import fingerprint
from ..setting import ROOT, checkpoint_path
from .feature import AppearanceFeature, pack_clip
from .model import CHECKPOINT_VERSION, ActionModel
from .upstream import PATCH_VERSION, REVISION

LOGGER = logging.getLogger(__name__)


class ActionPrediction:
    """Consume aligned observations through the trained action feature contract."""

    def __init__(self, setting: dict) -> None:
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
        for key in ("object_class", "object_point_index", "confidence", "sample_fps",
                    "person_detector", "person_pose"):
            if trained["feature"].get(key) != setting["feature"][key]:
                raise ValueError(f"feature.{key} differs from training; restore it or retrain")
        if saved["object_weight_digest"] != fingerprint(ROOT / setting["feature"]["object_weight"]):
            raise ValueError("Object pose weights changed since training; retrain")
        trained["model"]["device"] = setting["model"]["device"]
        self.setting = trained
        self.device = torch.device(setting["model"]["device"])
        self.network = ActionModel(trained, saved["object_point_count"]).to(self.device)
        self.network.load_state_dict(saved["state_dict"], strict=True)
        self.network.eval()
        self.appearance = AppearanceFeature(setting)
        self.window_frames = trained["train"]["window_frames"]
        self.frames = deque(maxlen=self.window_frames)
        self.threshold = setting["run"]["threshold"]
        self.active = None

    @torch.inference_mode()
    def predict(self, frame, human, target) -> float:
        """Update action history only after both branches have completed the same frame."""
        extracted = self.appearance.extract(frame.image, human, target)
        if not self.frames:
            self.frames.extend([extracted] * (self.window_frames - 1))
        self.frames.append(extracted)
        observation = {key: np.stack([item[key] for item in self.frames]) for key in extracted}
        human_feature, object_feature = pack_clip(observation, self.setting)
        probability = self.network(torch.from_numpy(human_feature[None]).to(self.device),
                                   torch.from_numpy(object_feature[None]).to(self.device))
        confidence = float(probability.exp()[0, 1])
        if not np.isfinite(confidence):
            raise RuntimeError("Action model returned a non-finite probability")
        detected = confidence >= self.threshold
        if detected != self.active:
            LOGGER.info("Action state %s -> %s frame=%d", self.active, detected, frame.sequence)
            self.active = detected
        return confidence
