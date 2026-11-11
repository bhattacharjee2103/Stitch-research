"""Draw a clear outline and match information for each detected reference."""
import cv2
import numpy as np

from ..core.display import draw_label
from ..core.registry import register_overlay


@register_overlay("outline")
class OutlineOverlay:
    def draw(self, frame, detection, ref):
        quad = np.rint(detection.quad).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [quad], True, (70, 235, 90), 2, cv2.LINE_AA)
        y = int(np.clip(np.min(detection.quad[:, 1]), 24, frame.shape[0] - 1))
        label = f"{ref.name}  {detection.confidence:.0%}  ({detection.inliers} matches)"
        draw_label(frame, label, y, (150, 255, 160))
        return frame
