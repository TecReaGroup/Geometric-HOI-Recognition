"""Independent human and object keypoint operations."""

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from ..setting import ROOT
from .engine import INPUT_SIZE, load_rtmw, person_detector_weight, yolo_engine


@dataclass(frozen=True)
class KeypointObservation:
    """Normalized keypoints and the matching image-space crop box."""

    point: np.ndarray
    box: np.ndarray | None


def select_box(boxes: np.ndarray, previous: np.ndarray | None) -> int:
    """Associate by nearest center, initially choosing the largest box."""
    if previous is None:
        return int(np.argmax(np.prod(boxes[:, 2:4] - boxes[:, :2], axis=1)))
    return int(np.argmin(np.linalg.norm((boxes[:, :2] + boxes[:, 2:4]) / 2 - previous, axis=1)))


class HumanKeypoint:
    """Own YOLO26m detection followed by RTMW's 133-point pose inference."""

    def __init__(self, setting: dict) -> None:
        self.device = setting["model"]["device"]
        self.threshold = setting["feature"]["confidence"]
        self.detector = YOLO(str(yolo_engine(person_detector_weight(), setting)), task="detect")
        self.pose = load_rtmw(setting)
        self.center = None

    @torch.inference_mode()
    def estimate(self, frame: np.ndarray) -> KeypointObservation:
        """Return only the selected person's whole-body keypoints and box."""
        prediction = self.detector.predict(frame, device=self.device, classes=[0],
                                           conf=self.threshold, imgsz=INPUT_SIZE,
                                           rect=False, verbose=False)[0]
        boxes = prediction.boxes.xyxy.cpu().numpy()
        point = np.zeros((133, 3), dtype=np.float32)
        if not len(boxes):
            self.center = None
            return KeypointObservation(point, None)
        box = boxes[select_box(boxes, self.center)]
        self.center = (box[:2] + box[2:]) / 2
        coordinate, score = self.pose(frame, bboxes=box[None])
        point[:, :2] = coordinate[0] / np.array(frame.shape[1::-1], dtype=np.float32)
        point[:, 2] = score[0]
        point[~np.isfinite(point).all(axis=1) | (point[:, 2] < self.threshold)] = 0
        return KeypointObservation(point, box)


class ObjectKeypoint:
    """Own the custom YOLO26-pose object model independently of human inference."""

    def __init__(self, setting: dict) -> None:
        self.device = setting["model"]["device"]
        self.option = setting["feature"]
        weight = ROOT / self.option["object_weight"]
        source = YOLO(str(weight), task="pose")
        shape = getattr(source.model.model[-1], "kpt_shape", None)
        if shape is None or max(self.option["object_point_index"]) >= shape[0]:
            raise ValueError("Object weight must contain the configured pose keypoints")
        if self.option["object_class"] not in source.names:
            raise ValueError("Object class is absent from the pose weight")
        self.point_count = int(shape[0])
        del source
        self.pose = YOLO(str(yolo_engine(weight, setting)), task="pose")
        self.center = None

    @torch.inference_mode()
    def estimate(self, frame: np.ndarray) -> KeypointObservation:
        """Return normalized object keypoints and their corresponding crop box."""
        prediction = self.pose.predict(
            frame, device=self.device, verbose=False, imgsz=INPUT_SIZE, rect=False,
            conf=self.option["confidence"], classes=[self.option["object_class"]],
        )[0]
        point = np.zeros((self.point_count, 3), dtype=np.float32)
        boxes = prediction.boxes.xyxy.cpu().numpy()
        if not len(boxes):
            self.center = None
            return KeypointObservation(point, None)
        index = select_box(boxes, self.center)
        box = boxes[index]
        self.center = (box[:2] + box[2:]) / 2
        point[:, :2] = prediction.keypoints.xy[index].cpu().numpy() / np.array(
            frame.shape[1::-1], dtype=np.float32,
        )
        confidence = prediction.keypoints.conf
        point[:, 2] = (confidence[index].cpu().numpy() if confidence is not None
                       else float(prediction.boxes.conf[index]))
        point[~np.isfinite(point).all(axis=1) | (point[:, 2] < self.option["confidence"])] = 0
        return KeypointObservation(point, box)
