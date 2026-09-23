"""List camera names and OpenCV device indices for the selected backend."""

import logging

import cv2
from cv2_enumerate_cameras import enumerate_cameras

from geometric_hoi.logging import configure_logging

CAMERA_BACKEND = cv2.CAP_DSHOW
LOGGER = logging.getLogger("list_camera")


def main() -> None:
    """Print and persist the camera indices used by the USB camera driver."""
    configure_logging()
    try:
        backend_name = cv2.videoio_registry.getBackendName(CAMERA_BACKEND)
        cameras = enumerate_cameras(CAMERA_BACKEND)
        LOGGER.info("Found %d camera(s), backend=%s", len(cameras), backend_name)
        for camera in cameras:
            LOGGER.info("deviceId=%d | name=%s | path=%s",
                        camera.index, camera.name, camera.path)
        if cameras:
            LOGGER.info('Set camera.backend="%s" and camera.deviceId to the selected index',
                        backend_name.lower())
    except Exception:
        LOGGER.exception("Camera enumeration failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
