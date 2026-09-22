"""Project configuration and local runtime directories."""

import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_setting(path: Path) -> dict:
    """Validate user configuration at the application boundary."""
    with path.open("rb") as stream:
        setting = tomllib.load(stream)
    model, feature, train, run = (setting[key] for key in ("model", "feature", "train", "run"))
    if model["name"] not in {"2g-gcn", "geovis-gnn"}:
        raise ValueError("model.name must be 2g-gcn or geovis-gnn")
    if model["hidden_size"] < 8 or model["hidden_size"] % 4:
        raise ValueError("model.hidden_size must be a positive multiple of four, >= 8")
    if feature["person_detector"] != "yolo26m" or feature["person_pose"] != "rtmw-x":
        raise ValueError("Human recognition requires yolo26m and rtmw-x")
    if not model["device"].startswith("cuda") or feature["workspace_mb"] <= 0:
        raise ValueError("TensorRT requires a CUDA device and positive workspace_mb")
    indexes = feature["object_point_index"]
    if len(indexes) != 2 or min(indexes) < 0 or indexes[0] == indexes[1]:
        raise ValueError("Two distinct nonnegative object_point_index values are required")
    if feature["sample_fps"] <= 0 or not 0 < feature["confidence"] < 1:
        raise ValueError("Invalid feature sampling rate or confidence")
    if not 0 < train["validation_fraction"] < 0.5:
        raise ValueError("train.validation_fraction must be between zero and 0.5")
    if train["window_frames"] < 2 or min(train[k] for k in (
        "stride_frames", "epochs", "batch_size", "learning_rate"
    )) <= 0:
        raise ValueError("Invalid training window or optimization settings")
    if not 0 < run["threshold"] < 1 or run["log_interval_seconds"] <= 0:
        raise ValueError("Invalid inference threshold or log interval")
    camera = setting["camera"]
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
