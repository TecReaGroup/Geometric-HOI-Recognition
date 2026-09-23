"""Project configuration and local runtime directories."""

import math
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_setting(path: Path) -> dict:
    """Validate user configuration at the application boundary."""
    with path.open("rb") as stream:
        setting = tomllib.load(stream)
    from .logging import LOG_LEVELS

    log_setting = setting.setdefault("logging", {})
    level = log_setting.setdefault("level", "INFO")
    if not isinstance(level, str) or level.upper() not in LOG_LEVELS:
        raise ValueError(f"logging.level must be one of {', '.join(LOG_LEVELS)}")
    log_setting["level"] = level.upper()
    view = setting.setdefault("view", {})
    for key in ("hand_skeleton", "yolo_pose_keypoint", "yolo_pose_box", "tip_trajectory"):
        if not isinstance(view.setdefault(key, True), bool):
            raise ValueError(f"view.{key} must be a boolean")
    tip_index = view.setdefault("tip_point_index", 0)
    if type(tip_index) is not int or tip_index < 0:
        raise ValueError("view.tip_point_index must be a nonnegative integer")
    model, train, run = (setting[key] for key in ("model", "train", "run"))
    human = setting["recognition"]["human"]
    target = setting["recognition"]["object"]
    if model["name"] not in {"2g-gcn", "geovis-gnn"}:
        raise ValueError("model.name must be 2g-gcn or geovis-gnn")
    if model["hidden_size"] < 8 or model["hidden_size"] % 4:
        raise ValueError("model.hidden_size must be a positive multiple of four, >= 8")
    if human["person_detector"] != "yolo26m" or human["person_pose"] != "rtmw-x":
        raise ValueError("Human recognition requires yolo26m and rtmw-x")
    if not model["device"].startswith("cuda") or human["workspace_mb"] <= 0:
        raise ValueError("TensorRT requires a CUDA device and positive workspace_mb")
    point_names = target["object_point"]
    if not isinstance(target["object_class"], str) or not target["object_class"].strip():
        raise ValueError("recognition.object.object_class must be a nonempty class label")
    if (not isinstance(point_names, list) or len(point_names) != 2
            or any(not isinstance(name, str) or not name.strip() for name in point_names)
            or point_names[0] == point_names[1]):
        raise ValueError("Two distinct nonempty object_point labels are required")
    if not 0 < human["confidence"] < 1:
        raise ValueError("recognition.human.confidence must be between zero and one")
    if not 0 < train["validation_fraction"] < 0.5:
        raise ValueError("train.validation_fraction must be between zero and 0.5")
    resample_fps = train.setdefault("resample_fps", 20.0)
    if (type(resample_fps) not in (int, float)
            or not math.isfinite(resample_fps) or resample_fps <= 0):
        raise ValueError("train.resample_fps must be finite and positive")
    if train["window_frames"] < 2 or min(train[k] for k in (
        "stride_frames", "epochs", "batch_size", "learning_rate"
    )) <= 0:
        raise ValueError("Invalid training window or optimization settings")
    if not 0 < run["threshold"] < 1 or run["log_interval_seconds"] <= 0:
        raise ValueError("Invalid inference threshold or log interval")
    camera = setting["camera"]
    if not math.isfinite(camera["fps"]) or camera["fps"] <= 0:
        raise ValueError("camera.fps must be finite and positive")
    if not isinstance(camera["driver"], str) or not camera["driver"].isidentifier():
        raise ValueError("camera.driver must be a driver module and class name in device")
    if not (ROOT / "device" / f"{camera['driver']}.py").is_file():
        raise ValueError(f"Camera driver not found: {camera['driver']}")
    if camera["timeout_seconds"] <= 0:
        raise ValueError("camera.timeout_seconds must be positive")
    return setting


def configure_directory() -> None:
    """Keep library caches and intermediate artifacts in the project."""
    for variable, name in (("TORCH_HOME", "torch"), ("YOLO_CONFIG_DIR", "ultralytics"),
                           ("MPLCONFIGDIR", "matplotlib"), ("HF_HOME", "huggingface")):
        directory = ROOT / "temp" / name
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)


def checkpoint_path(setting: dict) -> Path:
    """Return the selected model's independent checkpoint."""
    return ROOT / "data" / "model" / setting["model"]["name"] / "checkpoint.pt"
