"""Shared RTMPose, YOLO-pose and appearance extraction."""

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image
from rtmlib import Body, RTMPose, YOLOX
from torchvision.models import ResNet50_Weights, resnet50
from ultralytics import YOLO, settings

from .setting import ROOT

LOGGER = logging.getLogger(__name__)


def pose_weight(url: str) -> str:
    """Download the preset ONNX weight into the project model directory."""
    name = Path(urlparse(url).path).stem
    target = ROOT / "data" / "model" / f"{name}.onnx"
    if not target.exists():
        archive = ROOT / "temp" / f"{name}.zip"
        LOGGER.info("Downloading pose weight %s", url)
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as bundle:
            member = next(item for item in bundle.namelist() if item.endswith(".onnx"))
            partial = ROOT / "temp" / f"{name}.onnx"
            with bundle.open(member) as source, partial.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            partial.replace(target)
        archive.unlink()
    return str(target)


class FeatureExtractor:
    """Extract one stable person and one target object per RGB frame."""

    def __init__(self, setting: dict) -> None:
        """Load frozen detectors and pretrained appearance encoder."""
        self.option = setting["feature"]
        self.device = torch.device(setting["model"]["device"])
        settings.update({"weights_dir": str(ROOT / "data" / "model"),
                         "datasets_dir": str(ROOT / "temp" / "dataset"),
                         "runs_dir": str(ROOT / "temp" / "run")})
        weight = ROOT / self.option["object_weight"]
        if not weight.is_file():
            raise FileNotFoundError(f"Missing YOLO pose weight: {weight}")
        self.object_pose = YOLO(str(weight), task="pose")
        shape = getattr(self.object_pose.model.model[-1], "kpt_shape", None)
        if shape is None or max(self.option["object_point_index"]) >= shape[0]:
            raise ValueError("YOLO weight must be pose with configured object point indexes")
        self.object_point_count = int(shape[0])
        if self.option["object_class"] not in self.object_pose.names:
            raise ValueError("feature.object_class does not exist in YOLO weight")
        preset = Body.MODE[self.option["pose_size"]]
        self.person_detector = YOLOX(pose_weight(preset["det"]),
                                     model_input_size=tuple(preset["det_input_size"]),
                                     backend="onnxruntime", device=self.option["pose_device"])
        self.person_pose = RTMPose(pose_weight(preset["pose"]),
                                   model_input_size=tuple(preset["pose_input_size"]),
                                   to_openpose=False, backend="onnxruntime",
                                   device=self.option["pose_device"])
        if self.option["pose_device"] == "cuda":
            for estimator in (self.person_detector, self.person_pose):
                if "CUDAExecutionProvider" not in estimator.session.get_providers():
                    raise RuntimeError("CUDA pose requested but ONNX CUDA provider is unavailable")
        weight_set = ResNet50_Weights.IMAGENET1K_V2
        self.appearance = resnet50(weights=weight_set).to(self.device).eval()
        self.appearance.fc = torch.nn.Identity()
        self.transform = weight_set.transforms()
        self.reset()
        LOGGER.info("Feature extractor ready: RTMPose COCO17, object points=%d, device=%s",
                    self.object_point_count, self.device)

    def reset(self) -> None:
        """Forget associations when starting a different video."""
        self.person_center = None
        self.object_center = None

    @staticmethod
    def select_box(boxes: np.ndarray, previous: np.ndarray | None) -> int:
        """Associate by nearest center, initially selecting the largest box."""
        if previous is None:
            return int(np.argmax(np.prod(boxes[:, 2:4] - boxes[:, :2], axis=1)))
        return int(np.argmin(np.linalg.norm((boxes[:, :2] + boxes[:, 2:4]) / 2 - previous, axis=1)))

    @torch.inference_mode()
    def extract(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Return normalized keypoints, confidence and frozen crop descriptors."""
        height, width = frame.shape[:2]
        scale = np.array([width, height], dtype=np.float32)
        human_point = np.zeros((17, 3), dtype=np.float32)
        object_point = np.zeros((self.object_point_count, 3), dtype=np.float32)
        crop_box = []
        crop_slot = []
        boxes = np.asarray(self.person_detector(frame), dtype=np.float32).reshape(-1, 4)
        if len(boxes):
            index = self.select_box(boxes, self.person_center)
            box = boxes[index]
            self.person_center = (box[:2] + box[2:]) / 2
            coordinate, score = self.person_pose(frame, bboxes=box[None])
            human_point[:, :2] = coordinate[0] / scale
            human_point[:, 2] = score[0]
            crop_box.append(box)
            crop_slot.append(0)
        else:
            self.person_center = None
        prediction = self.object_pose.predict(frame, device=str(self.device), verbose=False,
                                               conf=self.option["confidence"],
                                               classes=[self.option["object_class"]])[0]
        if len(prediction.boxes):
            object_box = prediction.boxes.xyxy.cpu().numpy()
            index = self.select_box(object_box, self.object_center)
            box = object_box[index]
            self.object_center = (box[:2] + box[2:]) / 2
            object_point[:, :2] = prediction.keypoints.xy[index].cpu().numpy() / scale
            confidence = prediction.keypoints.conf
            object_point[:, 2] = (confidence[index].cpu().numpy() if confidence is not None
                                   else float(prediction.boxes.conf[index]))
            crop_box.append(box)
            crop_slot.append(1)
        else:
            self.object_center = None
        for point in (human_point, object_point):
            visible = np.isfinite(point).all(axis=1) & (point[:, 2] >= self.option["confidence"])
            point[~visible] = 0
        appearance = np.zeros((2, 2048), dtype=np.float32)
        crops, slots = [], []
        for box, slot in zip(crop_box, crop_slot):
            x1, y1, x2, y2 = box.astype(int)
            crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            if crop.size:
                crops.append(self.transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
                slots.append(slot)
        if crops:
            appearance[slots] = self.appearance(torch.stack(crops).to(self.device)).cpu().numpy()
        return {"human_point": human_point, "object_point": object_point,
                "appearance": appearance}


def pack_clip(observation: dict[str, np.ndarray], setting: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build official x/y/velocity geometry using past observations only."""
    human = observation["human_point"]
    selected = observation["object_point"][:, setting["feature"]["object_point_index"]]
    point = np.concatenate((human, selected), axis=1)
    coordinate = point[:, :, :2]
    velocity = np.zeros_like(coordinate)
    visible_pair = (point[1:, :, 2] > 0) & (point[:-1, :, 2] > 0)
    velocity[1:] = (coordinate[1:] - coordinate[:-1]) * setting["feature"]["sample_fps"]
    velocity[1:] *= visible_pair[:, :, None]
    geometry = np.concatenate((coordinate, velocity), axis=-1).reshape(len(point), -1)
    human_feature = np.concatenate((observation["appearance"][:, 0], geometry), axis=-1)
    object_feature = np.concatenate((observation["appearance"][:, 1],
                                     observation["object_point"].reshape(len(point), -1)), axis=-1)
    return human_feature[:, None].astype(np.float32), object_feature[:, None].astype(np.float32)
