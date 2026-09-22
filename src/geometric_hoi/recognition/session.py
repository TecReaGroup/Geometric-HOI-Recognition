"""Isolate native model initialization and inference from the desktop interpreter."""

import logging
import time
from queue import Full

from ..logging import configure_logging
from ..performance import PerformanceWindow

LOGGER = logging.getLogger(__name__)
MESSAGE_TIMEOUT_SECONDS = 0.1


def run_recognition(setting: dict, stopped, mailbox, image_slots, slot_locks) -> None:
    """Own the inference runtime in a spawned process and publish bounded messages."""
    configure_logging()

    def report_status(message: str) -> None:
        LOGGER.info(message)
        while not stopped.is_set():
            try:
                mailbox.put(("status", message), timeout=MESSAGE_TIMEOUT_SECONDS)
                return
            except Full:
                continue

    try:
        report_status("正在导入推理依赖…")
        import torch
        import numpy as np

        from .coordinator import camera_prediction

        if stopped.is_set():
            return
        report_status("正在初始化 CUDA…")
        device = torch.device(setting["model"]["device"])
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; run make install and check the NVIDIA driver")
            torch.cuda.set_device(device)
            LOGGER.info("PyTorch=%s CUDA=%s device=%s GPU=%s", torch.__version__,
                        torch.version.cuda, device, torch.cuda.get_device_name(device))
        performance = PerformanceWindow("preview_publish", setting["run"]["log_interval_seconds"])
        dropped = 0
        reported_at = time.perf_counter()
        for frame, probability, elapsed in camera_prediction(
            setting, stopped.is_set, report_status
        ):
            started = time.perf_counter()
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Preview requires a uint8 BGR frame")
            if frame.nbytes > len(image_slots[0]):
                raise ValueError("Actual camera frame exceeds configured shared preview capacity")
            slot = next((index for index, lock in enumerate(slot_locks) if lock.acquire(False)), None)
            packed_at = time.perf_counter()
            if slot is None:
                dropped += 1
            else:
                try:
                    destination = np.frombuffer(image_slots[slot], dtype=np.uint8,
                                                count=frame.size).reshape(frame.shape)
                    np.copyto(destination, frame)
                    packed_at = time.perf_counter()
                    mailbox.put_nowait(("frame", (slot, frame.shape[1], frame.shape[0],
                                                  frame.shape[1] * 3, probability, elapsed, packed_at)))
                except Full:
                    slot_locks[slot].release()
                    dropped += 1
                except BaseException:
                    slot_locks[slot].release()
                    raise
            finished = time.perf_counter()
            performance.record({"shared_copy": packed_at - started,
                                "queue_submit": finished - packed_at,
                                "total": finished - started})
            if finished - reported_at >= setting["run"]["log_interval_seconds"]:
                LOGGER.info("performance ipc_dropped=%d frame_bytes=%d window_s=%.2f",
                            dropped, frame.nbytes, finished - reported_at)
                dropped = 0
                reported_at = finished
    except Exception as exc:
        LOGGER.exception("Recognition process failed")
        while not stopped.is_set():
            try:
                mailbox.put(("failure", str(exc)), timeout=MESSAGE_TIMEOUT_SECONDS)
                break
            except Full:
                continue
    finally:
        # Shutdown must not wait for an unread frame in the queue feeder.
        if stopped.is_set():
            mailbox.cancel_join_thread()
        mailbox.close()
