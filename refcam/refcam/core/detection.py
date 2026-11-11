"""Detection value type shared by detectors, overlays and the pipeline."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A verified reference found in a frame.

    quad: 4x2 float32 corners ordered top-left, top-right, bottom-right,
    bottom-left in frame pixel coordinates.
    """

    quad: np.ndarray
    confidence: float
    inliers: int
