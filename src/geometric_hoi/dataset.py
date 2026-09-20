"""Video sampling, reproducible splitting and cached feature windows."""

import hashlib
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .feature import FeatureExtractor, pack_clip
from .setting import ROOT

LOGGER = logging.getLogger(__name__)
VIDEO_SUFFIX = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
FEATURE_VERSION = 1


def weight_digest(path: Path) -> str:
    """Fingerprint detector weights to invalidate stale feature caches."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_video(path: Path, extractor: FeatureExtractor, setting: dict, fingerprint: str) -> Path:
    """Decode sampled frames and persist features with source timestamps."""
    stamp = path.stat()
    signature = json.dumps([str(path.resolve()), stamp.st_size, stamp.st_mtime_ns,
                            setting["feature"], fingerprint, FEATURE_VERSION], sort_keys=True)
    cache = ROOT / "temp" / "feature" / (hashlib.sha256(signature.encode()).hexdigest() + ".npz")
    if cache.exists():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    extractor.reset()
    observations, timestamps = [], []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot decode video or determine FPS: {path}")
        sample_fps = setting["feature"]["sample_fps"]
        if fps < sample_fps:
            raise ValueError(f"{path}: source FPS {fps:.3f} < configured sample_fps {sample_fps}")
        frame_index, next_sample = 0, 0.0
        while True:
            available, frame = capture.read()
            if not available:
                break
            timestamp = frame_index / fps
            frame_index += 1
            if timestamp + 1e-8 < next_sample:
                continue
            observations.append(extractor.extract(frame))
            timestamps.append(timestamp)
            next_sample += 1 / sample_fps
            if len(observations) % 100 == 0:
                LOGGER.info("Extracting %s sampled_frames=%d", path.name, len(observations))
    finally:
        capture.release()
    if not observations:
        raise ValueError(f"Video contains no decoded frames: {path}")
    arrays = {key: np.stack([frame[key] for frame in observations]) for key in observations[0]}
    temporary = cache.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, human_point=arrays["human_point"],
                            object_point=arrays["object_point"], appearance=arrays["appearance"],
                            timestamp=np.asarray(timestamps))
    temporary.replace(cache)
    LOGGER.info("Extracted %s frames=%d", path, len(observations))
    return cache


class ClipDataset(Dataset):
    """Load only the requested feature window from a source video."""

    def __init__(self, entries: list[dict], setting: dict) -> None:
        """Store split windows and their feature contract."""
        self.entries = entries
        self.setting = setting

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple:
        """Assemble a normalized official-model input window."""
        entry = self.entries[index]
        start = entry["start"]
        stop = start + self.setting["train"]["window_frames"]
        with np.load(entry["cache"], allow_pickle=False) as archive:
            observation = {key: archive[key][start:stop] for key in
                           ("human_point", "object_point", "appearance")}
        human, object_feature = pack_clip(observation, self.setting)
        return torch.from_numpy(human), torch.from_numpy(object_feature), entry["label"]


def build_dataset(setting: dict, extractor: FeatureExtractor, fingerprint: str) -> tuple:
    """Split whole videos, or isolated temporal blocks when only one exists."""
    option = setting["train"]
    root = ROOT / option["video_dir"]
    window, stride = option["window_frames"], option["stride_frames"]
    generator = random.Random(option["seed"])
    split = {"train": [], "validation": []}
    for label, folder in enumerate(("negative-sample", "positive-sample")):
        paths = sorted(path for path in (root / folder).rglob("*") if path.suffix.lower() in VIDEO_SUFFIX)
        if not paths:
            raise ValueError(f"No videos found in {root / folder}")
        generator.shuffle(paths)
        validation_count = max(1, round(len(paths) * option["validation_fraction"]))
        for index, path in enumerate(paths):
            cache = extract_video(path, extractor, setting, fingerprint)
            with np.load(cache, allow_pickle=False) as archive:
                count = len(archive["timestamp"])
            if len(paths) > 1:
                ranges = [("validation" if index < validation_count else "train", 0, count)]
            else:
                validation_size = max(window, round(count * option["validation_fraction"]))
                boundary = count - validation_size
                ranges = [("train", 0, boundary - window), ("validation", boundary, count)]
                LOGGER.warning("%s has one video: temporal holdout with %d-frame gap; not cross-video evaluation",
                               folder, window)
            for partition, start, stop in ranges:
                if stop - start < window:
                    raise ValueError(f"{path}: {partition} needs >= {window} sampled frames; add longer/more videos")
                for offset in range(start, stop - window + 1, stride):
                    split[partition].append({"video": str(path), "cache": str(cache),
                                             "start": offset, "label": label})
    manifest = ROOT / "temp" / "split.json"
    manifest.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Split train=%d validation=%d manifest=%s", len(split["train"]), len(split["validation"]), manifest)
    return ClipDataset(split["train"], setting), ClipDataset(split["validation"], setting)
