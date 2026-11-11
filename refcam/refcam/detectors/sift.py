"""SIFT detector with ratio-tested descriptor matching and homography checks.

Contract:
    fit(image, bbox)          -> model dict (keypoints, descriptors, detector)
    match(frame, model)       -> Detection | None

fit() analyses the selected crop only; keypoints are relative to the crop's
top-left corner.
"""
import cv2
import numpy as np

from ..core.detection import Detection  # noqa: F401  (part of the match contract)
from ..core.registry import register_detector


@register_detector("sift")
class SiftDetector:
    name = "sift"

    def __init__(self):
        self._sift = cv2.SIFT_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    def fit(self, image, bbox):
        if image is None or image.size == 0:
            raise ValueError("cannot analyse an empty image")
        height, width = image.shape[:2]
        x, y, w, h = (int(round(float(value))) for value in bbox)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ValueError("selected box is outside the image or too small")
        crop = image[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        if descriptors is None or not keypoints:
            points = np.empty((0, 2), np.float32)
            descriptors = np.empty((0, 128), np.float32)
        else:
            points = np.asarray([keypoint.pt for keypoint in keypoints], np.float32)
            descriptors = descriptors.astype(np.float32)
        return {"detector": self.name, "keypoints": points, "descriptors": descriptors}

    def match(self, frame, model):
        """Find the reference in ``frame`` and return its projected corners."""
        if frame is None or frame.size == 0:
            return None

        reference_points = np.asarray(model.get("keypoints", []), dtype=np.float32)
        reference_descriptors = np.asarray(
            model.get("descriptors", []), dtype=np.float32
        )
        if (
            reference_points.ndim != 2
            or reference_points.shape[1] != 2
            or reference_descriptors.ndim != 2
            or reference_descriptors.shape[0] != reference_points.shape[0]
            or reference_descriptors.shape[0] < 4
        ):
            return None

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors = self._sift.detectAndCompute(gray, None)
        if frame_descriptors is None or len(frame_keypoints) < 4:
            return None

        try:
            pairs = self._matcher.knnMatch(reference_descriptors, frame_descriptors, k=2)
        except cv2.error:
            return None

        ratio = float(model.get("match_ratio", 0.75))
        candidates = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
        ]
        candidates.sort(key=lambda match: match.distance)
        good = []
        used_frame_features = set()
        for match in candidates:
            if match.trainIdx not in used_frame_features:
                good.append(match)
                used_frame_features.add(match.trainIdx)
        minimum = max(4, int(model.get("min_inliers", 12)))
        if len(good) < minimum:
            return None

        source_points = np.float32([reference_points[m.queryIdx] for m in good])
        target_points = np.float32([frame_keypoints[m.trainIdx].pt for m in good])
        try:
            homography, inlier_mask = cv2.findHomography(
                source_points,
                target_points,
                cv2.RANSAC,
                4.0,
                maxIters=2000,
                confidence=0.995,
            )
        except cv2.error:
            return None
        if homography is None or inlier_mask is None or not np.isfinite(homography).all():
            return None

        inliers = int(inlier_mask.ravel().sum())
        if inliers < minimum:
            return None

        ref_width, ref_height = model.get("size", (0, 0))
        ref_width, ref_height = int(ref_width), int(ref_height)
        if ref_width < 2 or ref_height < 2:
            return None
        corners = np.float32(
            [[0, 0], [ref_width - 1, 0], [ref_width - 1, ref_height - 1], [0, ref_height - 1]]
        ).reshape(-1, 1, 2)
        try:
            quad = cv2.perspectiveTransform(corners, homography).reshape(4, 2)
        except cv2.error:
            return None
        if not np.isfinite(quad).all():
            return None
        if np.max(np.abs(quad)) > max(width, height) * 20:
            return None

        polygon = np.rint(quad).astype(np.int32).reshape(-1, 1, 2)
        area = abs(float(cv2.contourArea(polygon)))
        if (
            not cv2.isContourConvex(polygon)
            or area < max(25.0, width * height * 0.0005)
            or area > width * height * 16.0
        ):
            return None
        if (
            quad[:, 0].max() < 0
            or quad[:, 1].max() < 0
            or quad[:, 0].min() >= width
            or quad[:, 1].min() >= height
        ):
            return None

        return Detection(
            quad=quad.astype(np.float32),
            confidence=inliers / len(good),
            inliers=inliers,
        )
