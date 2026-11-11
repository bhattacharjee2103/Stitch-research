"""OpenCV preview helpers shared by commands."""
import cv2


def fit_frame(frame, width, height):
    """Downscale a frame to fit a width x height box, preserving aspect ratio."""
    frame_h, frame_w = frame.shape[:2]
    scale = min(width / frame_w, height / frame_h, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, round(frame_w * scale)), max(1, round(frame_h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def draw_label(frame, text, y, color=(255, 255, 255)):
    """Draw readable text with a dark outline."""
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
