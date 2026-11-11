"""Render posted content with an explicit alpha channel and anchor polygon."""
from typing import Optional
import cv2
import numpy as np
from .geometry import corners, valid_anchor, valid_homography

def render(frame: np.ndarray, overlay: np.ndarray, H: Optional[np.ndarray],
           reference_shape: tuple, anchor: Optional[np.ndarray] = None,
           opacity: float = 0.8) -> np.ndarray:
    """Warp content coordinates into a reference anchor, then into the query."""
    if not valid_homography(H, reference_shape):
        return frame.copy()
    H = H / H[2,2]
    if not 0 <= opacity <= 1:
        raise ValueError('opacity must be in [0,1]')
    if overlay.ndim==2:
        overlay=cv2.cvtColor(overlay,cv2.COLOR_GRAY2BGR)
    if overlay.ndim!=3 or overlay.shape[2] not in (3,4) or min(overlay.shape[:2])<2:
        raise ValueError('Overlay must be a grayscale, BGR or BGRA image of at least 2x2 pixels')
    anchor = corners(reference_shape) if anchor is None else np.asarray(anchor)
    if not valid_anchor(anchor):
        raise ValueError('Anchor must be a finite convex quadrilateral with ordered corners')
    content_to_ref = cv2.getPerspectiveTransform(corners(overlay.shape).astype(np.float32),
                                                anchor.astype(np.float32))
    transform = H @ content_to_ref
    color = overlay[:,:,:3]
    alpha = overlay[:,:,3]/255.0 if overlay.shape[2] == 4 else np.ones(overlay.shape[:2])
    alpha = alpha.astype(np.float32) * opacity
    # Premultiplication avoids dark fringes and preserves valid black content.
    premult = color.astype(np.float32) * alpha[:,:,None]
    size = (frame.shape[1], frame.shape[0])
    warped = cv2.warpPerspective(premult, transform, size)
    mask = cv2.warpPerspective(alpha, transform, size)
    return np.clip(frame*(1-mask[:,:,None]) + warped, 0, 255).astype(np.uint8)


def render_projected_quad(frame: np.ndarray, overlay: np.ndarray, query_quad: np.ndarray,
                          opacity: float = 0.8) -> np.ndarray:
    """Project an overlay onto four already-projected 3D plane corners."""
    if not 0 <= opacity <= 1:
        raise ValueError('Opacity must be in [0,1]')
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    if overlay.ndim != 3 or overlay.shape[2] not in (3, 4) or min(overlay.shape[:2]) < 2:
        raise ValueError('Overlay must be a grayscale, BGR or BGRA image of at least 2x2 pixels')
    quad = np.asarray(query_quad, dtype=np.float32).reshape(4, 2)
    polygon = np.rint(quad).astype(np.int32).reshape(-1, 1, 2)
    if (not np.isfinite(quad).all() or not cv2.isContourConvex(polygon)
            or abs(cv2.contourArea(polygon)) < 4):
        return frame.copy()

    height, width = overlay.shape[:2]
    source = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    transform = cv2.getPerspectiveTransform(source, quad)
    color = overlay[:, :, :3].astype(np.float32)
    alpha = (overlay[:, :, 3].astype(np.float32) / 255.0
             if overlay.shape[2] == 4 else np.ones((height, width), dtype=np.float32))
    alpha *= float(opacity)
    premultiplied = color * alpha[:, :, None]
    size = (frame.shape[1], frame.shape[0])
    warped_color = cv2.warpPerspective(premultiplied, transform, size)
    warped_alpha = cv2.warpPerspective(alpha, transform, size)[:, :, None]
    base = frame.astype(np.float32)
    return np.clip(base * (1.0 - warped_alpha) + warped_color, 0, 255).astype(np.uint8)
