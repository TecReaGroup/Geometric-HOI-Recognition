"""Write YAML pose metadata into an existing Ultralytics checkpoint."""

import logging
import os
import tempfile
from pathlib import Path

from geometric_hoi.logging import configure_logging
from geometric_hoi.setting import ROOT, configure_directory

PT_PATH = ROOT / "data/model/yolo26-pose/best.pt"
METADATA_PATH = Path(__file__).with_name("yolo_metadata.yaml")
LOGGER = logging.getLogger("yolo_metadata")


def write_pose_metadata(weight: Path) -> None:
    """Replace checkpoint metadata after validating the existing pose head."""
    import torch
    import yaml

    with METADATA_PATH.open(encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("YOLO metadata must be a YAML mapping")
    class_names = metadata.get("names")
    keypoint_shape = metadata.get("kpt_shape")
    keypoint_names = metadata.get("kpt_names")
    if (not isinstance(class_names, dict) or not class_names
            or any(type(index) is not int for index in class_names)
            or set(class_names) != set(range(len(class_names)))
            or any(not isinstance(name, str) or not name.strip() for name in class_names.values())
            or len(set(class_names.values())) != len(class_names)):
        raise ValueError("names must map consecutive integer class IDs to unique nonempty labels")
    if (not isinstance(keypoint_shape, list) or len(keypoint_shape) != 2
            or any(type(size) is not int for size in keypoint_shape)
            or keypoint_shape[0] <= 0 or keypoint_shape[1] not in (2, 3)):
        raise ValueError("kpt_shape must be [positive keypoint count, 2 or 3]")
    if (not isinstance(keypoint_names, dict)
            or any(type(index) is not int for index in keypoint_names)
            or set(keypoint_names) != set(class_names)):
        raise ValueError("kpt_names must contain the same integer class IDs as names")
    for names in keypoint_names.values():
        if (not isinstance(names, list) or len(names) != keypoint_shape[0]
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Each kpt_names list must match kpt_shape and contain unique labels")

    weight = weight.resolve(strict=True)
    checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected an Ultralytics checkpoint dictionary")
    models = [checkpoint[key] for key in ("model", "ema") if checkpoint.get(key) is not None]
    if not models:
        raise ValueError("Checkpoint contains neither model nor ema")
    for model in models:
        if not isinstance(model, torch.nn.Module) or not hasattr(model, "model"):
            raise ValueError("Checkpoint model and ema must be Ultralytics model modules")
        head = model.model[-1]
        shape = getattr(head, "kpt_shape", None)
        if shape is None or list(shape) != keypoint_shape:
            raise ValueError(f"Expected pose head kpt_shape={keypoint_shape}, got {shape}")
        if getattr(head, "nc", None) != len(class_names):
            raise ValueError("Pose head class count does not match YAML names")

    for model in models:
        model.names = class_names.copy()
        model.kpt_shape = keypoint_shape.copy()
        model.kpt_names = {index: names.copy() for index, names in keypoint_names.items()}

    temporary_directory = ROOT / "temp"
    temporary_directory.mkdir(exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".pt", dir=temporary_directory)
    temporary_weight = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic replacement keeps the original intact if serialization fails.
        temporary_weight.replace(weight)
    finally:
        temporary_weight.unlink(missing_ok=True)
    LOGGER.info("Updated %s: names=%s, kpt_shape=%s, kpt_names=%s",
                weight, class_names, keypoint_shape, keypoint_names)


def main() -> None:
    """Persist the adjacent YAML metadata into the configured checkpoint."""
    configure_directory()
    configure_logging()
    try:
        write_pose_metadata(PT_PATH)
    except Exception:
        LOGGER.exception("Failed to update pose metadata: %s", PT_PATH)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
