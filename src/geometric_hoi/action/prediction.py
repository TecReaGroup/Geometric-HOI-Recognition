"""Own checkpoint validation, temporal history and action decisions."""

import logging
import time
from collections import deque

import numpy as np
import torch

from ..recognition.engine import fingerprint
from ..setting import ROOT, checkpoint_path
from ..performance import PerformanceWindow
from ..logging import PERF
from .feature import AppearanceFeature, pack_clip
from .model import ActionModel
from .replay import ActionReplay

LOGGER = logging.getLogger(__name__)


def validate_checkpoint(saved: dict, setting: dict) -> None:
    """Reject obsolete checkpoints before constructing inference models."""
    retrain = "请使用当前配置执行 make train 重新训练动作识别模型。"
    try:
        trained = saved["setting"]
        model_name = trained["model"]["name"]
        trained["model"]["hidden_size"]
        trained["train"]["window_frames"]
        saved["object_point_count"]
        saved["state_dict"]
        digest = saved["object_weight_digest"]
        contracts = [
            (section, key, trained["recognition"][section][key])
            for section, keys in (("object", ("object_class", "object_point")),
                                  ("human", ("confidence", "person_detector", "person_pose")))
            for key in keys
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"动作识别检查点格式已过期或不完整（缺失或无效字段：{exc}）。{retrain}") from exc
    if model_name != setting["model"]["name"]:
        raise ValueError(f"动作识别检查点的模型类型与当前配置不一致。{retrain}")
    for section, key, trained_value in contracts:
        if trained_value != setting["recognition"][section][key]:
            raise ValueError(f"recognition.{section}.{key} 与训练时不一致。{retrain}")
    if digest != fingerprint(ROOT / setting["recognition"]["object"]["object_weight"]):
        raise ValueError(f"目标姿态 PT 文件与训练时不一致（权重或元数据已变化）。{retrain}")


class ActionPrediction:
    """Consume aligned observations through the trained action feature contract."""

    def __init__(self, setting: dict) -> None:
        path = checkpoint_path(setting)
        if not path.is_file():
            raise FileNotFoundError(f"缺少动作识别检查点 {path}，请使用当前配置执行 make train。")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint(saved, setting)
        trained = saved["setting"]
        trained["model"]["device"] = setting["model"]["device"]
        self.setting = trained
        self.device = torch.device(setting["model"]["device"])
        self.network = ActionModel(trained, saved["object_point_count"]).to(self.device)
        self.network.load_state_dict(saved["state_dict"], strict=True)
        self.network.eval()
        self.appearance = AppearanceFeature(setting)
        self.window_frames = trained["train"]["window_frames"]
        self.infer = self.network
        if self.device.type == "cuda":
            self.infer = ActionReplay(self.network, self.window_frames,
                                     saved["object_point_count"], self.device)
        self.frames = deque(maxlen=self.window_frames)
        self.threshold = setting["run"]["threshold"]
        self.active = None
        self.performance = PerformanceWindow("action", setting["run"]["log_interval_seconds"])
        LOGGER.log(PERF, "Performance runtime: appearance=TensorRT FP16 action=PyTorch dtype=%s "
                    "window_frames=%d; stage timings are wall-clock including CPU transfers; "
                    "parallel human/object durations must not be summed",
                    next(self.network.parameters()).dtype, self.window_frames)

    @torch.inference_mode()
    def predict(self, frame, human, target) -> float:
        """Update action history only after both branches have completed the same frame."""
        started = time.perf_counter()
        extracted = self.appearance.extract(frame.image, human, target)
        appearance_finished = time.perf_counter()
        extracted["timestamp"] = np.asarray(frame.captured_at)
        if not self.frames:
            self.frames.extend([extracted] * (self.window_frames - 1))
        self.frames.append(extracted)
        observation = {key: np.stack([item[key] for item in self.frames]) for key in extracted}
        human_feature, object_feature = pack_clip(observation, self.setting)
        packed_at = time.perf_counter()
        probability = self.infer(torch.from_numpy(human_feature[None]).to(self.device),
                                 torch.from_numpy(object_feature[None]).to(self.device))
        confidence = float(probability.exp()[0, 1])
        self.performance.record({"appearance": appearance_finished - started,
                                 "pack_clip": packed_at - appearance_finished,
                                 "network_transfer_inference": time.perf_counter() - packed_at,
                                 "total": time.perf_counter() - started})
        if not np.isfinite(confidence):
            raise RuntimeError("Action model returned a non-finite probability")
        detected = confidence >= self.threshold
        if detected != self.active:
            LOGGER.info("Action state %s -> %s frame=%d", self.active, detected, frame.sequence)
            self.active = detected
        return confidence
