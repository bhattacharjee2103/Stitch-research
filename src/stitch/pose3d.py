"""Camera geometry for depth-backed reference maps and 3D overlay boxes."""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class PoseEstimate:
    """Pose mapping reference-camera 3D points into query-camera coordinates."""

    rvec: Optional[np.ndarray]
    tvec: Optional[np.ndarray]
    inlier_mask: np.ndarray
    inliers: int
    ratio: float
    accepted: bool
    median_error_px: float


def camera_matrix(image_shape, horizontal_fov_deg):
    """Build a zero-distortion pinhole camera matrix from horizontal FOV."""
    height, width = image_shape[:2]
    fov = float(horizontal_fov_deg)
    if width < 2 or height < 2 or not 1.0 < fov < 179.0:
        raise ValueError("camera dimensions or horizontal field of view are invalid")
    focal = (width / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
    return np.asarray(
        [[focal, 0.0, (width - 1) / 2.0],
         [0.0, focal, (height - 1) / 2.0],
         [0.0, 0.0, 1.0]], dtype=np.float64)


def crop_camera_matrix(matrix, x, y):
    """Shift the principal point when a reference crop is cut from a frame."""
    result = np.asarray(matrix, dtype=np.float64).copy()
    result[0, 2] -= float(x)
    result[1, 2] -= float(y)
    return result


def unproject_pixels(points, depths, matrix):
    """Back-project pixel coordinates and positive camera-Z depth to 3D."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    if len(points) != len(depths):
        raise ValueError("pixel and depth counts differ")
    inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    rays = np.column_stack((points, np.ones(len(points)))) @ inverse.T
    return rays * depths[:, None]


def sample_depth(depth_map, points, radius=2):
    """Sample a small local median around each image point."""
    depth_map = np.asarray(depth_map, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    height, width = depth_map.shape[:2]
    sampled = np.full(len(points), np.nan, dtype=np.float32)
    for index, (x, y) in enumerate(points):
        px, py = int(round(float(x))), int(round(float(y)))
        if px < 0 or py < 0 or px >= width or py >= height:
            continue
        patch = depth_map[max(0, py-radius):min(height, py+radius+1),
                          max(0, px-radius):min(width, px+radius+1)]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size:
            sampled[index] = float(np.median(valid))
    return sampled


def unique_matches(reference_points, query_points, cell_px=1.0):
    """Keep one-to-one correspondences.

    Ratio-test matching can assign many reference keypoints to one query
    pixel, typically in blurred frames. Such duplicates let PnP fit a
    degenerate pose with near-zero error, so only the first correspondence per
    query pixel and per reference pixel is kept.
    """
    reference_points = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    query_points = np.asarray(query_points, dtype=np.float32).reshape(-1, 2)
    keep = np.zeros(len(query_points), dtype=bool)
    seen_query, seen_reference = set(), set()
    for index, (r, q) in enumerate(zip(reference_points, query_points)):
        query_key = tuple(np.floor(q / cell_px).astype(int))
        reference_key = tuple(np.floor(r / cell_px).astype(int))
        if query_key in seen_query or reference_key in seen_reference:
            continue
        seen_query.add(query_key)
        seen_reference.add(reference_key)
        keep[index] = True
    return keep


def support_fraction(points, image_size):
    """Convex-hull area of points as a fraction of the image area."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    width, height = image_size
    if len(points) < 3 or width <= 0 or height <= 0:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(points))) / float(width * height)


def estimate_pose(reference_points, query_points, depth_map, reference_matrix,
                  query_matrix, min_inliers=12, min_ratio=0.2,
                  reprojection_px=5.0, max_depth_m=80.0, min_support=0.01,
                  query_size=None):
    """Estimate a reference-to-query pose by PnP RANSAC over 3D/2D matches.

    Acceptance requires enough one-to-one inliers, a sufficient inlier ratio
    and reprojection accuracy, inliers spread over at least ``min_support`` of
    the query image (convex hull area), points in front of the camera and a
    camera position within ``max_depth_m`` of the reference camera.
    ``query_size`` is the query image ``(width, height)``; when omitted it is
    inferred from a centred principal point.
    """
    reference_points = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    query_points = np.asarray(query_points, dtype=np.float32).reshape(-1, 2)
    empty = np.zeros(len(reference_points), dtype=bool)
    if len(reference_points) != len(query_points):
        raise ValueError("reference and query match counts differ")
    if len(reference_points) < 6:
        return PoseEstimate(None, None, empty, 0, 0.0, False, float("inf"))

    # The caller maps reference feature coordinates to depth-map pixel units
    # before calling this function.
    depths = sample_depth(depth_map, reference_points)
    valid = (np.isfinite(depths) & (depths >= 0.1) & (depths <= max_depth_m)
             & unique_matches(reference_points, query_points))
    valid_indices = np.flatnonzero(valid)
    valid_count = len(valid_indices)
    if valid_count < max(6, int(min_inliers)):
        return PoseEstimate(None, None, empty, 0, 0.0, False, float("inf"))

    points_3d = unproject_pixels(reference_points[valid], depths[valid], reference_matrix)
    points_2d = query_points[valid].astype(np.float32)
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d.astype(np.float32), points_2d,
            np.asarray(query_matrix, dtype=np.float64), None,
            iterationsCount=2000, reprojectionError=float(reprojection_px),
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        success, rvec, tvec, inliers = False, None, None, None
    if not success or inliers is None:
        return PoseEstimate(None, None, empty, 0, 0.0, False, float("inf"))

    inlier_indices = inliers.reshape(-1)
    if len(inlier_indices) >= 6 and hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                points_3d[inlier_indices].astype(np.float32),
                points_2d[inlier_indices], query_matrix, None, rvec, tvec)
        except cv2.error:
            pass
    projected, _ = cv2.projectPoints(
        points_3d[inlier_indices].astype(np.float32), rvec, tvec, query_matrix, None)
    errors = np.linalg.norm(projected.reshape(-1, 2) - points_2d[inlier_indices], axis=1)
    count = len(inlier_indices)
    ratio = count / max(1, valid_count)
    median_error = float(np.median(errors)) if len(errors) else float("inf")
    all_inliers = np.zeros(len(reference_points), dtype=bool)
    all_inliers[valid_indices[inlier_indices]] = True
    rotation, _ = cv2.Rodrigues(rvec)
    inlier_query_z = (points_3d[inlier_indices] @ rotation.T + tvec.reshape(1, 3))[:, 2]
    query_matrix = np.asarray(query_matrix, dtype=np.float64)
    if query_size is None:
        query_size = (2.0 * query_matrix[0, 2] + 1.0, 2.0 * query_matrix[1, 2] + 1.0)
    support = support_fraction(points_2d[inlier_indices], query_size)
    camera_distance = float(np.linalg.norm(rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3)))
    accepted = bool(
        count >= int(min_inliers)
        and ratio >= float(min_ratio)
        and median_error <= float(reprojection_px)
        and np.isfinite(rvec).all()
        and np.isfinite(tvec).all()
        and np.all(inlier_query_z > 0.05)
        and support >= float(min_support)
        and camera_distance <= float(max_depth_m)
    )
    return PoseEstimate(rvec, tvec, all_inliers, count, ratio, accepted, median_error)


def oriented_box_from_roi(bbox, depth_map, matrix, yaw_pitch_roll_deg=(0, 0, 0)):
    """Build a camera-facing metric box from a selected rectangle in a depth map.

    The selected rectangle determines width and height. Its centre depth is the
    median over a small patch. Optional yaw, pitch and roll rotate the plane
    around its centre in reference-camera coordinates.
    """
    x, y, width, height = (float(value) for value in bbox)
    if width < 2 or height < 2:
        raise ValueError("the 3D placement box is too small")
    depth_map = np.asarray(depth_map, dtype=np.float32)
    center = np.asarray([[x + width / 2.0, y + height / 2.0]], dtype=np.float32)
    depth = sample_depth(depth_map, center, radius=4)[0]
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError("the selected placement has no valid depth")
    pixel_corners = np.asarray(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float64)
    center_3d = unproject_pixels(center, [depth], matrix)[0]
    corners_3d = unproject_pixels(pixel_corners, np.full(4, depth), matrix)
    yaw, pitch, roll = np.deg2rad(np.asarray(yaw_pitch_roll_deg, dtype=np.float64))
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rotate_y = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotate_x = np.asarray([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rotate_z = np.asarray([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    rotation = rotate_y @ rotate_x @ rotate_z
    corners_3d = (corners_3d - center_3d) @ rotation.T + center_3d
    return corners_3d.astype(np.float32), float(depth)


def project_points(points_3d, rvec, tvec, matrix):
    """Project a reference-camera 3D quadrilateral into the query image."""
    points_3d = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    rotation, _ = cv2.Rodrigues(rvec)
    query_z = (points_3d @ rotation.T + np.asarray(tvec).reshape(1, 3))[:, 2]
    if not np.isfinite(query_z).all() or np.any(query_z <= 0.05):
        return None
    pixels, _ = cv2.projectPoints(points_3d, rvec, tvec, matrix, None)
    pixels = pixels.reshape(-1, 2)
    if not np.isfinite(pixels).all():
        return None
    return pixels
