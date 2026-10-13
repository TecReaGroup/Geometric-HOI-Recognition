"""Render recorded videos with the live recognition pipeline and preview layers."""

import logging
import math
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from ..setting import ROOT

LOGGER = logging.getLogger(__name__)
VIDEO_DIRECTORY = ROOT / "data" / "video" / "render"
OUTPUT_DIRECTORY = ROOT / "data" / "output"
VIDEO_SUFFIX = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
OUTPUT_CODEC = "mp4v"
PROGRESS_INTERVAL_FRAMES = 100


def render_video(setting: dict) -> None:
    """Save annotated MP4 files while preserving relative paths and source frame rates."""
    videos = sorted(path for path in VIDEO_DIRECTORY.rglob("*")
                    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIX)
    if not videos:
        raise FileNotFoundError(f"No videos found in {VIDEO_DIRECTORY}")
    destinations = [OUTPUT_DIRECTORY / path.relative_to(VIDEO_DIRECTORY).with_suffix(".mp4")
                    for path in videos]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Render videos with the same relative stem would overwrite one output")

    # NOTE: Windows offscreen QPA has no system fonts; hidden widgets can use native QPA.
    os.environ["QT_QPA_PLATFORM"] = "windows" if sys.platform == "win32" else "offscreen"
    from .preview import CameraView
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    LOGGER.info("Render Qt platform=%s", application.platformName())
    view = CameraView(setting)
    view.setMinimumSize(0, 0)
    try:
        for source, destination in zip(videos, destinations):
            render_recording(setting, source, destination, view)
    finally:
        view.close()
    LOGGER.info("Rendered %d video(s) to %s", len(videos), OUTPUT_DIRECTORY)


def render_recording(setting: dict, source: Path, destination: Path, view) -> None:
    """Own video decoding, per-video recognition state and completed output publication."""
    import cv2
    import numpy as np
    from PySide6.QtGui import QImage

    from .coordinator import frame_prediction

    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video has an invalid frame rate: {source}")
        available, first_frame = capture.read()
        if not available:
            raise ValueError(f"Video contains no readable frames: {source}")
        height, width = first_frame.shape[:2]
        view.resize(width, height)
        view.fps = fps
        frame_count = 0

        def video_frame() -> Iterator[tuple]:
            """Use the recording clock for action features and trajectory expiration."""
            yield 0.0, first_frame
            index = 1
            while True:
                available, frame = capture.read()
                if not available:
                    return
                if frame.shape != first_frame.shape:
                    raise ValueError(f"Video frame dimensions changed: {source}")
                yield index / fps, frame
                index += 1

        LOGGER.info("Rendering %s -> %s size=%dx%d fps=%.3f", source, destination,
                    width, height, fps)
        with TemporaryDirectory(prefix="render_", dir=ROOT / "temp") as directory:
            temporary = Path(directory) / "render.mp4"
            writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*OUTPUT_CODEC),
                                     fps, (width + width % 2, height + height % 2))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create video writer with codec {OUTPUT_CODEC}")
            try:
                with closing(frame_prediction(setting, video_frame(), lambda: False,
                                              LOGGER.info)) as predictions:
                    for frame, probability, _elapsed, annotation in predictions:
                        view.image = QImage(frame.data, width, height, frame.strides[0],
                                            QImage.Format.Format_BGR888).copy()
                        view.probability = probability
                        # Offline trails already expire against video time in PoseOverlay.
                        view.annotation = annotation
                        image = QImage(width, height, QImage.Format.Format_RGB888)
                        view.render(image)
                        pixels = np.frombuffer(image.bits(), dtype=np.uint8).reshape(
                            height, image.bytesPerLine()
                        )[:, :width * 3].reshape(height, width, 3)
                        annotated = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                        annotated = cv2.copyMakeBorder(annotated, 0, height % 2, 0, width % 2,
                                                       cv2.BORDER_REPLICATE)
                        writer.write(annotated)
                        frame_count += 1
                        if frame_count % PROGRESS_INTERVAL_FRAMES == 0:
                            LOGGER.info("Rendering %s frame=%d video_seconds=%.2f", source.name,
                                        frame_count, frame_count / fps)
            finally:
                writer.release()
                writer = None
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temporary), str(destination))
        LOGGER.info("Saved %s frames=%d duration_seconds=%.2f", destination,
                    frame_count, frame_count / fps)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
