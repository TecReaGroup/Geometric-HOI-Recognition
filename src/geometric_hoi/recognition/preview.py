"""Qt video preview with a bounded mailbox and probability card."""

import logging
import multiprocessing
import threading
import time
from queue import Empty

from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from .session import run_recognition
from ..performance import PerformanceWindow

LOGGER = logging.getLogger(__name__)
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700
FPS_UPDATE_SECONDS = 1.0
PROCESS_POLL_SECONDS = 0.1
SHUTDOWN_GRACE_SECONDS = 3.0


class RecognitionThread(QThread):
    """Bridge the inference process to Qt without blocking the UI event loop."""

    failed = Signal(str)
    status_changed = Signal(str)

    def __init__(self, setting: dict) -> None:
        super().__init__()
        self.setting = setting
        self.lock = threading.Lock()
        self.latest = None
        self.overwritten = 0

    def take_frame(self) -> tuple | None:
        """Consume the latest prediction without queuing obsolete frames."""
        with self.lock:
            latest, self.latest = self.latest, None
        return latest

    def run(self) -> None:
        """Receive predictions and own bounded, asynchronous process shutdown."""
        context = multiprocessing.get_context("spawn")
        stopped = context.Event()
        mailbox = context.Queue(maxsize=2)
        inference = context.Process(target=run_recognition,
                                    args=(self.setting, stopped, mailbox),
                                    name="hoi-recognition")
        started = False
        performance = PerformanceWindow("preview_receive", self.setting["run"]["log_interval_seconds"])
        try:
            if self.isInterruptionRequested():
                return
            inference.start()
            started = True
            LOGGER.info("Recognition process started pid=%s", inference.pid)
            while not self.isInterruptionRequested():
                try:
                    kind, payload = mailbox.get(timeout=PROCESS_POLL_SECONDS)
                except Empty:
                    if not inference.is_alive():
                        raise RuntimeError(
                            f"Recognition process exited unexpectedly (code={inference.exitcode})"
                        )
                    continue
                if kind == "failure":
                    self.failed.emit(payload)
                    break
                if kind == "status":
                    self.status_changed.emit(payload)
                    continue
                pixels, width, height, stride, probability, elapsed, published_at = payload
                received_at = time.perf_counter()
                image = QImage(pixels, width, height, stride, QImage.Format.Format_RGB888).copy()
                with self.lock:
                    self.overwritten += self.latest is not None
                    self.latest = image, probability, elapsed, time.perf_counter()
                performance.record({"ipc_latency": received_at - published_at,
                                    "image_copy": time.perf_counter() - received_at})
        except Exception as exc:
            LOGGER.exception("Camera preview failed")
            self.failed.emit(str(exc))
        finally:
            stopped.set()
            if started:
                inference.join(SHUTDOWN_GRACE_SECONDS)
                if inference.is_alive():
                    LOGGER.warning("Recognition shutdown exceeded %.1fs; terminating pid=%s",
                                   SHUTDOWN_GRACE_SECONDS, inference.pid)
                    inference.terminate()
                    inference.join()
                LOGGER.info("Recognition process stopped exitcode=%s", inference.exitcode)
            inference.close()
            mailbox.close()


class CameraView(QWidget):
    """Render letterboxed video and a resolution-independent confidence card."""

    def __init__(self, setting: dict) -> None:
        super().__init__()
        self.image = None
        self.probability = None
        self.fps = 0.0
        self.action = setting["run"]["action_name"]
        self.threshold = setting["run"]["threshold"]
        self.performance = PerformanceWindow("preview_paint", setting["run"]["log_interval_seconds"])
        self.setMinimumSize(640, 360)

    def paintEvent(self, event) -> None:
        """Draw an unclipped video and a high-contrast probability indicator."""
        started = time.perf_counter()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#0b1018"))
        if self.image is not None:
            size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            bounds = QRectF((self.width() - size.width()) / 2,
                            (self.height() - size.height()) / 2, size.width(), size.height())
            painter.drawImage(bounds, self.image)
        card = QRectF(20, 20, 320, 152)
        painter.setPen(QPen(QColor(255, 255, 255, 35), 1))
        painter.setBrush(QColor(17, 24, 34, 235))
        painter.drawRoundedRect(card, 14, 14)
        font = QFont("Microsoft YaHei")
        font.setPixelSize(15)
        painter.setFont(font)
        painter.setPen(QColor("#aab8c9"))
        title = painter.fontMetrics().elidedText(
            f"目标动作 · {self.action}", Qt.TextElideMode.ElideRight, 280
        )
        painter.drawText(QRectF(40, 34, 280, 25), Qt.AlignmentFlag.AlignVCenter, title)
        active = self.probability is not None and self.probability >= self.threshold
        color = QColor("#46d99b" if active else "#71b7ff")
        font.setPixelSize(32)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(color)
        label = f"{self.probability:.1%}" if self.probability is not None else "—"
        painter.drawText(QRectF(40, 64, 280, 44), Qt.AlignmentFlag.AlignVCenter, label)
        track = QRectF(40, 120, 280, 10)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#344152"))
        painter.drawRoundedRect(track, 5, 5)
        if self.probability is not None:
            painter.save()
            painter.setClipRect(QRectF(track.x(), track.y(), track.width() * self.probability, 10))
            painter.setBrush(color)
            painter.drawRoundedRect(track, 5, 5)
            painter.restore()
        painter.setPen(QPen(QColor("#e8edf5"), 2))
        threshold_x = track.x() + track.width() * self.threshold
        painter.drawLine(int(threshold_x), 117, int(threshold_x), 133)
        font.setPixelSize(11)
        font.setWeight(QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(QColor("#aab8c9"))
        painter.drawText(QRectF(40, 137, 280, 20), Qt.AlignmentFlag.AlignVCenter,
                         f"动作概率                         判定阈值 {self.threshold:.0%}")
        fps_card = QRectF(self.width() - 160, 20, 140, 44)
        painter.setPen(QPen(QColor(255, 255, 255, 35), 1))
        painter.setBrush(QColor(17, 24, 34, 235))
        painter.drawRoundedRect(fps_card, 10, 10)
        font.setPixelSize(18)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(QColor("#e8edf5"))
        painter.drawText(fps_card, Qt.AlignmentFlag.AlignCenter, f"FPS  {self.fps:.1f}")
        painter.end()
        if self.image is not None:
            self.performance.record({"paint_event": time.perf_counter() - started})


class PreviewWindow(QMainWindow):
    """Own preview updates and cooperative camera shutdown."""

    def __init__(self, setting: dict) -> None:
        super().__init__()
        self.model_name = setting["model"]["name"]
        self.setWindowTitle(f"HOI 动作识别 · {self.model_name}")
        self.setStyleSheet("QStatusBar { background: #111822; color: #aab8c9; "
                          "font-size: 12px; padding: 6px 12px; }")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(WINDOW_WIDTH, screen.width() - 60),
                    min(WINDOW_HEIGHT, screen.height() - 80))
        self.view = CameraView(setting)
        self.setCentralWidget(self.view)
        self.statusBar().showMessage("正在加载模型与摄像头，首帧完成后直接显示概率…")
        self.worker = RecognitionThread(setting)
        self.worker.failed.connect(self.show_failure)
        self.worker.status_changed.connect(self.show_status)
        self.worker.finished.connect(self.finish_close)
        self.closing = False
        self.failure = None
        self.fps_started = None
        self.fps_frame_count = 0
        self.performance = PerformanceWindow("preview_consume", setting["run"]["log_interval_seconds"])
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_frame)
        self.timer.start(max(1, int(1000 / setting["camera"]["fps"])))
        QTimer.singleShot(0, self.start_recognition)

    def start_recognition(self) -> None:
        """Start the background bridge after the window enters the event loop."""
        if not self.closing:
            self.worker.start()

    def show_status(self, message: str) -> None:
        """Display startup progress without overwriting shutdown or failure messages."""
        if not self.closing and self.failure is None:
            self.statusBar().showMessage(message)

    def refresh_frame(self) -> None:
        """Display the newest inference output at a bounded refresh rate."""
        latest = self.worker.take_frame()
        now = time.monotonic()
        if latest is not None:
            if self.fps_started is None:
                self.fps_started = now
            else:
                self.fps_frame_count += 1
        if self.fps_started is not None and now - self.fps_started >= FPS_UPDATE_SECONDS:
            self.view.fps = self.fps_frame_count / (now - self.fps_started)
            with self.worker.lock:
                overwritten, self.worker.overwritten = self.worker.overwritten, 0
            LOGGER.info("performance displayed_fps=%.2f preview_overwritten=%d window_s=%.2f",
                        self.view.fps, overwritten, now - self.fps_started)
            self.fps_started = now
            self.fps_frame_count = 0
            self.view.update()
        if latest is None:
            return
        self.view.image, self.view.probability, elapsed, received_at = latest
        self.performance.record({"mailbox_wait": time.perf_counter() - received_at})
        self.view.update()
        if self.failure is None:
            self.statusBar().showMessage(
                f"持续识别 · {self.model_name}  |  推理 {elapsed * 1000:.0f} ms  |  "
                f"处理能力 {1 / max(elapsed, 1e-6):.1f} FPS  |  Q / Esc 退出"
            )

    def show_failure(self, message: str) -> None:
        """Keep errors visible instead of presenting a stale probability as live."""
        self.failure = message
        self.view.probability = None
        self.view.fps = 0.0
        self.fps_started = None
        self.fps_frame_count = 0
        self.worker.take_frame()
        self.view.update()
        self.statusBar().showMessage(f"识别已停止 · {message}")

    def keyPressEvent(self, event) -> None:
        """Support the existing camera exit shortcuts."""
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        """Wait asynchronously for camera ownership to be released."""
        self.timer.stop()
        self.closing = True
        if self.worker.isRunning():
            self.worker.requestInterruption()
            self.statusBar().showMessage("正在停止识别并释放摄像头…")
            event.ignore()
        else:
            event.accept()

    def finish_close(self) -> None:
        """Close only after the inference thread has exited."""
        if self.closing:
            self.close()


def run_preview(setting: dict) -> None:
    """Run the desktop preview on the main thread."""
    application = QApplication.instance() or QApplication([])
    window = PreviewWindow(setting)
    window.show()
    application.exec()
    if window.failure is not None:
        raise RuntimeError(window.failure)
