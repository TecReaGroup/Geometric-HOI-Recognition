"""Schedule camera frames, align keypoint branches and invoke action recognition."""

import logging
import time
from collections.abc import Callable, Iterator
from importlib.util import module_from_spec, spec_from_file_location

from ..setting import ROOT
from ..logging import PERF
from ..performance import PerformanceWindow

LOGGER = logging.getLogger(__name__)
CAMERA_POLL_SECONDS = 0.1


def camera_prediction(
    setting: dict, stopped: Callable[[], bool], report_status: Callable[[str], None]
) -> Iterator[tuple]:
    """Dispatch one shared frame and yield only its fully aligned action prediction."""
    from ..action.prediction import ActionPrediction
    from .bus import CaptureFrame, FrameBus
    from .engine import load_runtime
    from .keypoint import HumanKeypoint, ObjectKeypoint
    from .worker import KeypointWorker
    from .overlay import PoseOverlay

    if stopped():
        return
    report_status("正在加载动作识别模型与外观特征模型…")
    action = ActionPrediction(setting)
    if stopped():
        return
    report_status("正在加载 CUDA / TensorRT 运行库…")
    load_runtime()
    if stopped():
        return
    report_status("正在加载人体关键点模型…")
    human_estimator = HumanKeypoint(setting)
    if stopped():
        return
    report_status("正在加载物体关键点模型…")
    object_estimator = ObjectKeypoint(setting)
    if setting["view"]["tip_point_index"] >= object_estimator.point_count:
        raise ValueError("view.tip_point_index exceeds the object pose keypoint count")
    overlay = PoseOverlay(setting)
    if stopped():
        return
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
    last_log = 0.0
    capture = camera_class(camera_option)
    performance = PerformanceWindow("pipeline", option["log_interval_seconds"])
    try:
        for worker in workers:
            worker.start()
        report_status("正在打开摄像头…")
        capture.open()
        report_status("正在等待首帧推理…")
        sequence = 0
        last_frame = time.perf_counter()
        LOGGER.info("Camera driver=%s device=%s model=%s; human/object workers ready",
                    driver_name, camera_option["deviceId"], setting["model"]["name"])
        while not stopped():
            poll_started = time.perf_counter()
            captured_at, frame = capture.getFrame(timeout=CAMERA_POLL_SECONDS)
            now = time.perf_counter()
            if frame is None:
                if now - last_frame >= camera_option["timeout_seconds"]:
                    raise RuntimeError("Camera stopped delivering frames")
                continue
            last_frame = now
            sequence += 1
            packet = CaptureFrame(sequence, captured_at, frame)
            bus.submit(packet)
            pair = None
            while pair is None and not stopped():
                pair = bus.wait_pair(CAMERA_POLL_SECONDS)
            if pair is None:
                break
            paired_at = time.perf_counter()
            confidence = action.predict(packet, pair["human"], pair["object"])
            finished = time.perf_counter()
            elapsed = finished - now
            performance.record({"capture_wait": now - poll_started,
                                "frame_age_at_dispatch": now - captured_at,
                                "keypoint_wait": paired_at - now,
                                "action": finished - paired_at,
                                "processing": elapsed,
                                "capture_to_prediction": finished - captured_at})
            if now - last_log >= option["log_interval_seconds"]:
                LOGGER.log(PERF, "action=%s confidence=%.4f detected=%s inference_ms=%.1f",
                            option["action_name"], confidence, action.active, elapsed * 1000)
                last_log = now
            annotation = overlay.snapshot(pair["human"], pair["object"], confidence >= option["threshold"],
                                          finished, frame.shape)
            yield frame, confidence, elapsed, annotation
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
