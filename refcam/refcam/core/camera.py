"""Frame sources: webcam index, video file or still image."""
import sys
from pathlib import Path

import cv2


class FrameSource:
    """Uniform read() interface for --camera, --video and --image."""

    def __init__(self, camera=0, video=None, image=None):
        if video and image:
            raise ValueError("choose only one of --video and --image")
        self.camera = camera
        self.video = video
        self.image = image
        self.source_label = ""
        self._capture = None
        self._still = None

    def open(self):
        if self.image:
            still = cv2.imread(str(self.image), cv2.IMREAD_COLOR)
            if still is None:
                raise RuntimeError(f"cannot read image: {self.image}")
            self._still = still
            self.source_label = f"image {Path(self.image).name}"
            return self

        if self.video:
            path = Path(self.video)
            if not path.is_file():
                raise RuntimeError(f"video file not found: {self.video}")
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"cannot open video: {self.video}")
            self._capture = capture
            self.source_label = f"video {path.name}"
            return self

        backend = cv2.CAP_ANY
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            backend = cv2.CAP_AVFOUNDATION
        capture = (
            cv2.VideoCapture(self.camera, backend)
            if backend != cv2.CAP_ANY
            else cv2.VideoCapture(self.camera)
        )
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"camera {self.camera} not found. Try another --camera INDEX, "
                "check macOS camera permission, or pass --video/--image."
            )
        self._capture = capture
        self.source_label = f"camera {self.camera}"
        return self

    @property
    def is_image(self):
        return self._still is not None

    @property
    def is_video(self):
        return self._capture is not None and self.video is not None

    def read(self):
        if self._still is not None:
            return True, self._still.copy()
        ok, frame = self._capture.read()
        return ok, frame

    def release(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._still = None
