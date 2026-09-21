"""Qt video preview with a bounded mailbox and probability card."""

import logging
import threading

import cv2
from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

LOGGER = logging.getLogger(__name__)
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700


class RecognitionThread(QThread):
    """Keep model execution off the UI thread and retain only the newest image."""

    failed = Signal(str)

    def __init__(self, setting: dict) -> None:
        super().__init__()
        self.setting = setting
        self.lock = threading.Lock()
        self.latest = None

    def take_frame(self) -> tuple | None:
        """Consume the latest prediction without queuing obsolete frames."""
        with self.lock:
            latest, self.latest = self.latest, None
        return latest

    def run(self) -> None:
        """Load models and continuously publish actual model probabilities."""
        from .runtime import camera_prediction

        try:
            for frame, probability, elapsed in camera_prediction(
                self.setting, self.isInterruptionRequested
            ):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                               QImage.Format.Format_RGB888).copy()
                with self.lock:
                    self.latest = image, probability, elapsed
        except Exception as exc:
            LOGGER.exception("Camera preview failed")
            self.failed.emit(str(exc))


class CameraView(QWidget):
    """Render letterboxed video and a resolution-independent confidence card."""

    def __init__(self, setting: dict) -> None:
        super().__init__()
        self.image = None
        self.probability = None
        self.action = setting["run"]["action_name"]
        self.threshold = setting["run"]["threshold"]
        self.setMinimumSize(640, 360)

    def paintEvent(self, event) -> None:
        """Draw an unclipped video and a high-contrast probability indicator."""
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
        painter.end()


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
        self.worker.finished.connect(self.finish_close)
        self.closing = False
        self.failure = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_frame)
        self.timer.start(33)
        self.worker.start()

    def refresh_frame(self) -> None:
        """Display the newest inference output at a bounded refresh rate."""
        latest = self.worker.take_frame()
        if latest is None:
            return
        self.view.image, self.view.probability, elapsed = latest
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
        if self.worker.isRunning():
            self.closing = True
            self.worker.requestInterruption()
            self.statusBar().showMessage("正在释放摄像头…")
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
