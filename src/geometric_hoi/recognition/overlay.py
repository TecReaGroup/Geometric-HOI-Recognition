"""Frame-aligned overlay snapshots and action-gated tip history."""

import logging
from collections import deque

LOGGER = logging.getLogger(__name__)
TRAJECTORY_TIMEOUT_SECONDS = 0.5
TRAJECTORY_MAX_POINTS = 2048
HAND_START_INDEX = (91, 112)


class PoseOverlay:
    """Keep trajectory state before the preview transport can drop frames."""

    def __init__(self, setting: dict) -> None:
        self.option = setting["view"]
        self.trail = deque(maxlen=TRAJECTORY_MAX_POINTS)
        self.last_seen = None
        self.connected = False
        LOGGER.info("Preview overlay %s", self.option)

    def snapshot(self, human, target, active: bool, timestamp: float, shape: tuple) -> dict:
        """Return plain coordinates without importing inference libraries in Qt."""
        if self.last_seen is not None and timestamp - self.last_seen > TRAJECTORY_TIMEOUT_SECONDS:
            self.trail.clear()
            self.last_seen = None
            self.connected = False
            LOGGER.debug("Tip trajectory cleared after recognition timeout")
        if self.option["tip_trajectory"]:
            tip = target.point[self.option["tip_point_index"]]
            if active and target.box is not None and tip[2] > 0:
                if self.last_seen is None:
                    LOGGER.debug("Tip trajectory started")
                self.trail.append((float(tip[0]), float(tip[1]), self.connected))
                self.last_seen = timestamp
                self.connected = True
            else:
                self.connected = False
        height, width = shape[:2]
        box = target.box
        return {
            "hand": [human.point[start:start + 21].tolist() for start in HAND_START_INDEX]
            if self.option["hand_skeleton"] else [],
            "keypoint": target.point.tolist() if self.option["yolo_pose_keypoint"] else [],
            "box": [float(box[0] / width), float(box[1] / height),
                    float(box[2] / width), float(box[3] / height)]
            if self.option["yolo_pose_box"] and box is not None else None,
            "trail": tuple(self.trail),
            "last_seen": self.last_seen,
        }
