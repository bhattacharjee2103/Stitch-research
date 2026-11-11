"""Optional monocular metric-depth inference for 3D scene anchors.

The model and Transformers are loaded only when a 3D depth map is requested,
so the existing planar SIFT pipeline keeps its small runtime requirements.
"""
import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


class MonocularDepthEstimator:
    """Depth Anything V2 outdoor metric model, returned as float32 metres."""

    def __init__(self, model_id=DEFAULT_DEPTH_MODEL, device="cpu"):
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError(
                "3D placement needs Transformers. Install it with "
                "`.venv/bin/pip install -r requirements-depth.txt`."
            ) from exc

        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device).eval()
        self.model_id = model_id

    @torch.inference_mode()
    def estimate(self, bgr_image):
        """Return a metric depth map with the same height and width as the image."""
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("cannot estimate depth for an empty image")
        height, width = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        predicted = self.model(**inputs).predicted_depth
        restored = torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        depth = restored.clamp(min=0.05, max=80.0).cpu().numpy().astype(np.float32)
        if not np.isfinite(depth).all():
            raise ValueError("depth model returned non-finite values")
        return depth


def colorize_depth(depth):
    """Create a near-to-far color preview without changing metric values."""
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0)
    if not valid.any():
        raise ValueError("depth map has no valid pixels")
    low, high = np.percentile(values[valid], [2, 98])
    if high <= low:
        high = low + 1e-3
    normalized = np.zeros(values.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (high - values[valid]) * 255.0 / (high - low), 0, 255
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
