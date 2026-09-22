"""Schedule camera frames, align keypoint branches and invoke action recognition."""

import logging
import time
from collections.abc import Callable, Iterator
from importlib.util import module_from_spec, spec_from_file_location

from ..action.prediction import ActionPrediction
from ..setting import ROOT
from .bus import CaptureFrame, FrameBus
from .engine import load_runtime
from .keypoint import HumanKeypoint, ObjectKeypoint
from .worker import KeypointWorker

LOGGER = logging.getLogger(__name__)
CAMERA_POLL_SECONDS = 0.1


def camera_prediction(setting: dict, stopped: Callable[[], bool]) -> Iterator[tuple]:
    """Dispatch one shared frame and yield only its fully aligned action prediction."""
    action = ActionPrediction(setting)
    load_runtime()
    human_estimator = HumanKeypoint(setting)
    object_estimator = ObjectKeypoint(setting)
    bus = FrameBus()
    device = setting["model"]["device"]
    workers = [KeypointWorker("human", human_estimator, bus, device),
               KeypointWorker("object", object_estimator, bus, device)]
    option = setting["run"]
    camera_option = setting["camera"]
    driver_name = camera_option["driver"]
    driver_path = ROOT / "device" / f"{driver_name}.py"
    spec = spec_from_file_location(driver_name, driver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load camera driver {driver_path}")
    driver_module = module_from_spec(spec)
    spec.loader.exec_module(driver_module)
    camera_class = getattr(driver_module, driver_name)
    interval = 1 / setting["feature"]["sample_fps"]
    next_sample, last_log = 0.0, 0.0
    capture = camera_class(camera_option)
    try:
        for worker in workers:
            worker.start()
        capture.open()
        sequence = 0
        last_frame = time.monotonic()
        LOGGER.info("Camera driver=%s device=%s model=%s; human/object workers ready",
                    driver_name, camera_option["deviceId"], setting["model"]["name"])
        while not stopped():
            captured_at, frame = capture.getFrame(timeout=CAMERA_POLL_SECONDS)
            now = time.monotonic()
            if frame is None:
                if now - last_frame >= camera_option["timeout_seconds"]:
                    raise RuntimeError("Camera stopped delivering frames")
                continue
            last_frame = now
            if now < next_sample:
                continue
            sequence += 1
            packet = CaptureFrame(sequence, captured_at, frame)
            bus.submit(packet)
            pair = None
            while pair is None and not stopped():
                pair = bus.wait_pair(CAMERA_POLL_SECONDS)
            if pair is None:
                break
            confidence = action.predict(packet, pair["human"], pair["object"])
            next_sample = now + interval
            elapsed = time.monotonic() - now
            if now - last_log >= option["log_interval_seconds"]:
                LOGGER.info("action=%s confidence=%.4f detected=%s inference_ms=%.1f",
                            option["action_name"], confidence, action.active, elapsed * 1000)
                last_log = now
            yield frame, confidence, elapsed
    finally:
        bus.close()
        for worker in workers:
            if worker.ident is not None:
                worker.join()
        capture.stopThread()
        LOGGER.info("Camera released")


def run_camera(setting: dict) -> None:
    """Run the responsive desktop recognition preview."""
    from .preview import run_preview

    run_preview(setting)
