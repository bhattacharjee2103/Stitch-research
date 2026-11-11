"""Multi-view 3D placement: one shared anchor, measured capture angles, stable poses.

A floating overlay is drawn once, in the first (primary) reference view. Each
additional view is related to that primary view with depth-backed PnP, so the
same physical anchor is expressed in every view's camera frame. The relation
also measures how far the camera moved around the anchor, which drives the
capture plan and the angle feedback shown while rescanning.
"""
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .pose3d import PoseEstimate, estimate_pose, sample_depth, unproject_pixels


# ---------------------------------------------------------------- geometry

def camera_center(rvec, tvec):
    """Query-camera centre expressed in the reference camera frame."""
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return -rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3)


def orbit_angles(anchor_center, rvec, tvec) -> Tuple[float, float]:
    """Signed horizontal and vertical angles the camera moved around the anchor.

    Camera axes follow OpenCV: x right, y down, z forward. Positive yaw means
    the new camera stands to the right of the reference camera as seen from
    the reference camera; positive pitch means it stands higher.
    """
    anchor = np.asarray(anchor_center, dtype=np.float64).reshape(3)
    before = -anchor                                  # anchor -> reference camera
    after = camera_center(rvec, tvec) - anchor        # anchor -> new camera
    yaw = math.degrees(math.atan2(before[0] * after[2] - before[2] * after[0],
                                  before[0] * after[0] + before[2] * after[2]))

    def elevation(vector):
        return math.degrees(math.atan2(-vector[1], math.hypot(vector[0], vector[2])))

    return yaw, elevation(after) - elevation(before)


def match_views(engine, image_a, image_b, max_side=640):
    """Match two images at the pipeline working size; return original-pixel points."""
    def prepare(image):
        height, width = image.shape[:2]
        scale = min(1.0, max_side / max(height, width)) if max_side else 1.0
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = image if size == (width, height) else cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return resized, np.asarray([width / size[0], height / size[1]], dtype=np.float32)

    small_a, scale_a = prepare(image_a)
    small_b, scale_b = prepare(image_b)
    points_a, points_b, _ = engine.match(engine.extract(small_a), engine.extract(small_b))
    return points_a * scale_a, points_b * scale_b


@dataclass
class ViewRelation:
    """How an additional reference view relates to the primary 3D placement."""

    pose: PoseEstimate
    anchor_3d: Optional[np.ndarray]   # anchor corners in the new view's camera frame (metres)
    depth_scale: float                # new-view depth / transferred depth (1.0 = consistent)
    yaw_deg: float
    pitch_deg: float

    @property
    def accepted(self) -> bool:
        return self.anchor_3d is not None


def relate_view(primary_points, new_points, primary_depth, primary_matrix, new_matrix,
                anchor_primary, new_depth=None, min_inliers=12, min_ratio=0.2,
                reprojection_px=5.0, max_depth_m=80.0, scale_limits=(0.5, 2.0),
                min_support=0.01, new_size=None) -> ViewRelation:
    """Transfer the primary 3D anchor into a new reference view.

    ``primary_points`` must be in primary depth-map pixels and ``new_points``
    in new-view pixels. The pose maps primary-camera 3D points into the new
    camera. Monocular metric depth can disagree in scale between views, so the
    transferred anchor is rescaled to the new view's own depth map. Scaling
    about the camera centre leaves its image projection unchanged, so the
    overlay keeps its position in the new view while its 3D coordinates agree
    with the depth points used for PnP during later rescans.
    """
    anchor_primary = np.asarray(anchor_primary, dtype=np.float64).reshape(4, 3)
    primary_points = np.asarray(primary_points, dtype=np.float32).reshape(-1, 2)
    new_points = np.asarray(new_points, dtype=np.float32).reshape(-1, 2)
    if new_size is None and new_depth is not None:
        new_size = (new_depth.shape[1], new_depth.shape[0])
    pose = estimate_pose(primary_points, new_points, primary_depth, primary_matrix, new_matrix,
                         min_inliers, min_ratio, reprojection_px, max_depth_m, min_support, new_size)
    nan = float('nan')
    if not pose.accepted:
        return ViewRelation(pose, None, nan, nan, nan)
    rotation, _ = cv2.Rodrigues(pose.rvec)
    translation = np.asarray(pose.tvec, dtype=np.float64).reshape(1, 3)
    transferred = anchor_primary @ rotation.T + translation
    scale = 1.0
    if new_depth is not None:
        inliers = pose.inlier_mask
        depths = sample_depth(primary_depth, primary_points[inliers])
        valid = np.isfinite(depths) & (depths > 0)
        predicted_z = (unproject_pixels(primary_points[inliers][valid], depths[valid], primary_matrix)
                       @ rotation.T + translation)[:, 2]
        observed_z = sample_depth(new_depth, new_points[inliers][valid])
        usable = np.isfinite(observed_z) & (observed_z > 0) & (predicted_z > 0.05)
        if usable.sum() >= 6:
            scale = float(np.median(observed_z[usable] / predicted_z[usable]))
        if not (math.isfinite(scale) and scale_limits[0] <= scale <= scale_limits[1]):
            return ViewRelation(pose, None, scale, nan, nan)
    anchor_new = transferred * scale
    if not np.isfinite(anchor_new).all() or np.any(anchor_new[:, 2] <= 0.05):
        return ViewRelation(pose, None, scale, nan, nan)
    yaw, pitch = orbit_angles(anchor_primary.mean(axis=0), pose.rvec, pose.tvec)
    return ViewRelation(pose, anchor_new.astype(np.float32), scale, yaw, pitch)


# ---------------------------------------------------------------- capture planning

def capture_plan(sweep_deg: float, tolerance_deg: float) -> List[float]:
    """Yaw angles to capture so every angle in the sweep is within tolerance of a view.

    ``sweep_deg`` is the total horizontal range from which people should be
    able to rescan (centred on the first view). ``tolerance_deg`` is how far a
    rescan may deviate from a registered view and still match reliably.
    """
    sweep, tolerance = float(sweep_deg), float(tolerance_deg)
    if not (math.isfinite(sweep) and 0 <= sweep <= 180):
        raise ValueError('capture sweep must be between 0 and 180 degrees')
    if not (math.isfinite(tolerance) and 1 <= tolerance <= 90):
        raise ValueError('view tolerance must be between 1 and 90 degrees')
    half = sweep / 2.0
    if half <= tolerance:
        return [0.0]
    step = 2.0 * tolerance
    sides = math.ceil((half - tolerance) / step)
    angles = {0.0}
    for k in range(1, sides + 1):
        offset = round(min(k * step, half - tolerance), 1)
        angles.update((-offset, offset))
    return sorted(angles)


def coverage(captured_deg: Sequence[float], sweep_deg: float, tolerance_deg: float) -> Dict:
    """Which parts of the sweep the captured views cover, and what to capture next."""
    plan = capture_plan(sweep_deg, tolerance_deg)
    tolerance, half = float(tolerance_deg), float(sweep_deg) / 2.0
    captured = sorted(float(a) for a in captured_deg if a is not None and math.isfinite(float(a)))
    intervals = []
    for angle in captured:
        low, high = angle - tolerance, angle + tolerance
        if intervals and low <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], high)
        else:
            intervals.append([low, high])
    missing, cursor = [], -half
    for low, high in intervals:
        if high < cursor:
            continue
        if low > cursor:
            missing.append((round(cursor, 1), round(min(low, half), 1)))
        cursor = max(cursor, high)
        if cursor >= half:
            break
    if cursor < half:
        missing.append((round(cursor, 1), round(half, 1)))
    missing = [(a, b) for a, b in missing if b - a > 0.05]
    pending = [a for a in plan if not any(abs(a - c) <= tolerance / 2.0 for c in captured)]
    return dict(plan=plan, captured=captured, covered=[tuple(round(v, 1) for v in i) for i in intervals],
                missing=missing, pending=pending, complete=not missing)


def in_captured_range(angle_deg: float, captured_deg: Sequence[float], tolerance_deg: float) -> bool:
    return any(abs(float(angle_deg) - float(c)) <= float(tolerance_deg) for c in captured_deg)


# ---------------------------------------------------------------- temporal stability

def _quaternion(rvec):
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    w = math.sqrt(max(0.0, 1.0 + rotation[0, 0] + rotation[1, 1] + rotation[2, 2])) / 2.0
    if w > 1e-6:
        return np.asarray([w, (rotation[2, 1] - rotation[1, 2]) / (4 * w),
                           (rotation[0, 2] - rotation[2, 0]) / (4 * w),
                           (rotation[1, 0] - rotation[0, 1]) / (4 * w)])
    # 180-degree rotation: recover the axis from the rotation vector instead.
    vector = np.asarray(rvec, dtype=np.float64).reshape(3)
    angle = np.linalg.norm(vector)
    axis = vector / angle if angle > 0 else np.asarray([1.0, 0.0, 0.0])
    return np.concatenate(([math.cos(angle / 2)], axis * math.sin(angle / 2)))


def _rotation_vector(quaternion):
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    matrix = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    vector, _ = cv2.Rodrigues(matrix)
    return vector


def slerp(q0, q1, fraction):
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = math.acos(min(1.0, dot))
    return (math.sin((1 - fraction) * theta) * q0 + math.sin(fraction * theta) * q1) / math.sin(theta)


class PoseSmoother:
    """Smooth accepted poses and bridge a few failed frames without inventing motion.

    Rotation is interpolated on the unit quaternion sphere and translation is
    exponentially averaged; entrywise matrix averaging is avoided. A change of
    matched reference or a large jump resets the filter. Failed frames reuse
    the last accepted pose for at most ``hold_frames`` frames.
    """

    def __init__(self, smoothing=0.5, hold_frames=5, max_jump_deg=20.0, max_jump_ratio=0.25):
        if not 0 < smoothing <= 1:
            raise ValueError('pose smoothing must be in (0, 1]')
        if hold_frames < 0:
            raise ValueError('pose hold frames must be non-negative')
        self.smoothing, self.hold_frames = float(smoothing), int(hold_frames)
        self.max_jump_deg, self.max_jump_ratio = float(max_jump_deg), float(max_jump_ratio)
        self.reset()

    def reset(self):
        self.key, self.quaternion, self.translation, self.misses = None, None, None, 0

    def update(self, key, rvec=None, tvec=None):
        """Return ``(rvec, tvec, held)`` for display, or ``None`` when nothing should show."""
        if rvec is None or tvec is None:
            self.misses += 1
            if self.key is not None and self.misses <= self.hold_frames:
                return _rotation_vector(self.quaternion), self.translation.reshape(3, 1).copy(), True
            self.reset()
            return None
        quaternion = _quaternion(rvec)
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        restart = self.key != key or self.quaternion is None
        if not restart:
            angle = 2 * math.degrees(math.acos(min(1.0, abs(float(np.dot(
                quaternion / np.linalg.norm(quaternion), self.quaternion))))))
            jump = np.linalg.norm(translation - self.translation) / max(np.linalg.norm(self.translation), 1e-6)
            restart = angle > self.max_jump_deg or jump > self.max_jump_ratio
        if restart:
            self.quaternion, self.translation = quaternion / np.linalg.norm(quaternion), translation
        else:
            self.quaternion = slerp(self.quaternion, quaternion, self.smoothing)
            self.translation = (1 - self.smoothing) * self.translation + self.smoothing * translation
        self.key, self.misses = key, 0
        return _rotation_vector(self.quaternion), self.translation.reshape(3, 1).copy(), False


# ---------------------------------------------------------------- database helpers

def place_views(database, place_id: str) -> List[Dict]:
    """All registered 3D views for a place, primary (first) view first."""
    views = []
    for row in database.rows:
        if row['place_id'] != place_id:
            continue
        for content in row.get('content', []):
            if content.get('anchor_3d_camera_m') is None or not content.get('depth_map'):
                continue
            metadata = content.get('capture_metadata', {}) or {}
            views.append(dict(reference=row['reference'], content=content,
                              yaw_deg=float(metadata.get('capture_yaw_offset_deg', 0.0) or 0.0),
                              matrix=metadata.get('reference_camera_matrix'),
                              fov=metadata.get('camera_fov_x_deg')))
    return views


def load_primary_view(database, place_id: str) -> Optional[Dict]:
    """Image, depth map, camera matrix and anchor of a place's first 3D view."""
    views = place_views(database, place_id)
    if not views:
        return None
    view = views[0]
    image = cv2.imread(str(database.resolve(view['reference'])), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError('Primary reference image cannot be decoded: ' + view['reference'])
    depth = np.load(database.resolve(view['content']['depth_map']), allow_pickle=False).astype(np.float32)
    if view['matrix'] is None:
        raise ValueError('Primary 3D view has no saved camera matrix; re-register it')
    return dict(reference=view['reference'], image=image, depth=depth,
                matrix=np.asarray(view['matrix'], dtype=np.float64).reshape(3, 3),
                anchor=np.asarray(view['content']['anchor_3d_camera_m'], dtype=np.float64).reshape(4, 3),
                fov=view['fov'], captured=[v['yaw_deg'] for v in views])
