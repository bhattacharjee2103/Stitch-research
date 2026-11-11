"""Robust planar geometry. All homographies map reference to query pixels."""
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

@dataclass
class Verification:
    H: Optional[np.ndarray]
    inlier_mask: np.ndarray
    inliers: int
    ratio: float
    accepted: bool

def project(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a projective map, returning NaNs for points at infinity."""
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    H = np.asarray(H,dtype=np.float64)
    # Homographies are defined only up to scale. Use a stable representative
    # before testing the denominator against an absolute pixel-domain epsilon.
    scale = H[2,2] if H[2,2] != 0 else np.max(np.abs(H))
    if not np.isfinite(H).all() or not np.isfinite(scale) or scale == 0:
        return np.full((len(xy),2),np.nan)
    with np.errstate(over='ignore',invalid='ignore'):
        H = H / scale
    if not np.isfinite(H).all():
        return np.full((len(xy),2),np.nan)
    homogeneous = np.column_stack((xy, np.ones(len(xy)))) @ H.T
    denom = homogeneous[:, 2:3]
    return np.divide(homogeneous[:, :2], denom,
                     out=np.full((len(xy), 2), np.nan), where=np.abs(denom)>1e-10)

def corners(shape: Tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    return np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=np.float64)

def valid_homography(H: Optional[np.ndarray], shape: Optional[Tuple[int,...]] = None) -> bool:
    """Reject singular, nonfinite, horizon-crossing and collapsed projective maps."""
    if H is None or H.shape != (3,3) or not np.isfinite(H).all():
        return False
    if H[2,2] == 0:
        return False
    with np.errstate(over='ignore',invalid='ignore'):
        H = H / H[2,2]
    if not np.isfinite(H).all():
        return False
    if np.linalg.matrix_rank(H) < 3 or np.linalg.cond(H) > 1e12:
        return False
    if shape is not None:
        c = corners(shape)
        den = np.column_stack([c, np.ones(4)]) @ H[2]
        if np.any(np.abs(den)<1e-8) or not (np.all(den>0) or np.all(den<0)):
            return False
        polygon = project(c, H).astype(np.float32)
        if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
            return False
        if abs(cv2.contourArea(polygon)) < 16:
            return False
    return True

def valid_anchor(anchor: np.ndarray) -> bool:
    """An ordered anchor must describe a finite, convex quadrilateral."""
    polygon=np.asarray(anchor,dtype=np.float64)
    if polygon.shape!=(4,2) or not np.isfinite(polygon).all():
        return False
    polygon=polygon.astype(np.float32)
    return bool(cv2.isContourConvex(polygon) and abs(cv2.contourArea(polygon))>1e-8)

def verify(src: np.ndarray, dst: np.ndarray, threshold: float = 5.0,
           min_inliers: int = 12, min_ratio: float = 0.20,
           shape: Optional[Tuple[int,...]] = None) -> Verification:
    """Estimate H with RANSAC; retain failure rows instead of hiding them."""
    empty = np.zeros(len(src), dtype=bool)
    if len(src) < 4 or len(src) != len(dst):
        return Verification(None, empty, 0, 0.0, False)
    try:
        H, mask = cv2.findHomography(np.float32(src).reshape(-1,1,2),
                                    np.float32(dst).reshape(-1,1,2), cv2.RANSAC,
                                    threshold, maxIters=3000, confidence=0.999)
    except cv2.error:
        return Verification(None, empty, 0, 0.0, False)
    mask = mask.ravel().astype(bool) if mask is not None else empty
    n = int(mask.sum())
    ratio = n / len(src)
    accepted = n >= min_inliers and ratio >= min_ratio and valid_homography(H, shape)
    return Verification(H, mask, n, ratio, bool(accepted))

def corner_error(H: Optional[np.ndarray], ground_truth: np.ndarray, shape: Tuple[int,...]) -> float:
    """Mean four-corner error in resized query pixels; infinity on failure."""
    if not valid_homography(H, shape):
        return float('inf')
    c = corners(shape)
    return float(np.linalg.norm(project(c,H)-project(c,ground_truth), axis=1).mean())

def pose_errors(R: np.ndarray, t: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray) -> Tuple[float,float]:
    """Camera-center distance in map units and SO(3) error in degrees."""
    center = -R.T @ t.reshape(3)
    center_gt = -R_gt.T @ t_gt.reshape(3)
    angle = np.degrees(np.arccos(np.clip((np.trace(R @ R_gt.T)-1)/2, -1, 1)))
    return float(np.linalg.norm(center-center_gt)), float(angle)
