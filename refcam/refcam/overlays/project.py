"""Perspective-project an attached PNG/JPG onto its detected reference."""
from pathlib import Path

import cv2
import numpy as np

from ..core.display import draw_label
from ..core.registry import register_overlay
from .outline import OutlineOverlay


@register_overlay("project")
class ProjectOverlay:
    def __init__(self):
        self._images = {}
        self._outline = OutlineOverlay()

    def _image_for(self, ref):
        path = ref.overlay_path
        if path is None:
            return None
        path = Path(path)
        try:
            modified = path.stat().st_mtime_ns
        except OSError:
            modified = None
        cached = self._images.get(ref.name)
        if cached and cached[0] == modified:
            return cached[1]

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if modified is not None else None
        if image is not None:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            elif image.shape[2] != 4:
                image = None
        self._images[ref.name] = (modified, image)
        if image is None:
            print(f"refcam: could not read overlay for '{ref.name}'; drawing its outline")
        return image

    def draw(self, frame, detection, ref):
        image = self._image_for(ref)
        if image is None:
            return self._outline.draw(frame, detection, ref)

        height, width = image.shape[:2]
        source = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        target = np.asarray(detection.quad, dtype=np.float32)
        transform = cv2.getPerspectiveTransform(source, target)
        frame_height, frame_width = frame.shape[:2]
        warped = cv2.warpPerspective(
            image,
            transform,
            (frame_width, frame_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        x0 = max(0, int(np.floor(target[:, 0].min())))
        y0 = max(0, int(np.floor(target[:, 1].min())))
        x1 = min(frame_width, int(np.ceil(target[:, 0].max())) + 1)
        y1 = min(frame_height, int(np.ceil(target[:, 1].max())) + 1)
        if x1 > x0 and y1 > y0:
            region = warped[y0:y1, x0:x1]
            alpha = region[:, :, 3:4].astype(np.float32) / 255.0
            base = frame[y0:y1, x0:x1].astype(np.float32)
            color = region[:, :, :3].astype(np.float32)
            frame[y0:y1, x0:x1] = np.clip(
                color * alpha + base * (1.0 - alpha), 0, 255
            ).astype(np.uint8)

        label_y = int(np.clip(np.min(target[:, 1]), 24, frame_height - 1))
        draw_label(frame, f"{ref.name}  {detection.confidence:.0%}", label_y, (150, 255, 160))
        return frame
