"""Multi-view 3D placement: anchor transfer, measured angles, capture plan, smoothing."""
from pathlib import Path
import math

import cv2
import numpy as np
import pytest

from stitch import runenv
from stitch.features import LocalEngine
from stitch.multiview import (PoseSmoother, capture_plan, coverage, in_captured_range,
                              match_views, orbit_angles, relate_view)
from stitch.pipeline import Pipeline, Prediction
from stitch.pose3d import PoseEstimate, camera_matrix, oriented_box_from_roi, unproject_pixels

ROOT = Path(__file__).resolve().parents[1]


def rotation_y(degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def orbit_pose(anchor_center, degrees):
    """Camera moved around the anchor by ``degrees`` while keeping it in place."""
    rotation = rotation_y(degrees)
    translation = anchor_center - rotation @ anchor_center
    return rotation, translation


def project(points, rotation, translation, matrix):
    camera = points @ rotation.T + translation
    pixels = camera @ matrix.T
    return pixels[:, :2] / pixels[:, 2:3], camera[:, 2]


@pytest.mark.parametrize('sweep,tolerance,expected', [
    (30, 15, [0.0]), (60, 15, [-15.0, 0.0, 15.0]), (90, 15, [-30.0, 0.0, 30.0]),
    (120, 15, [-45.0, -30.0, 0.0, 30.0, 45.0])])
def test_capture_plan_covers_the_whole_range(sweep, tolerance, expected):
    plan = capture_plan(sweep, tolerance)
    assert plan == expected
    assert all(in_captured_range(a, plan, tolerance) for a in np.linspace(-sweep / 2, sweep / 2, 61))


def test_coverage_reports_gaps_and_next_view():
    report = coverage([0.0, 14.0], 60, 15)
    assert not report['complete'] and report['missing'] == [(-30.0, -15.0), (29.0, 30.0)]
    assert report['pending'] == [-15.0]
    assert coverage([-15, 0, 15], 60, 15)['complete']
    with pytest.raises(ValueError):
        capture_plan(60, 0)


def test_orbit_angle_sign_and_size():
    anchor = np.asarray([0.0, 0.0, 4.0])
    for degrees in (-25.0, 10.0, 30.0):
        rotation, translation = orbit_pose(anchor, degrees)
        yaw, pitch = orbit_angles(anchor, cv2.Rodrigues(rotation)[0], translation)
        assert yaw == pytest.approx(degrees, abs=1e-6) and pitch == pytest.approx(0, abs=1e-6)


def test_relate_view_transfers_anchor_and_corrects_depth_scale():
    rng = np.random.default_rng(3)
    matrix = camera_matrix((480, 640), 70)
    u, v = np.meshgrid(np.arange(640), np.arange(480))
    primary_depth = (4.0 + 0.002 * (u - 320) + 0.001 * (v - 240)).astype(np.float32)
    anchor, _ = oriented_box_from_roi((290, 210, 60, 60), primary_depth, matrix)
    anchor = anchor.astype(np.float64)
    pixels = rng.uniform([20, 20], [620, 460], size=(250, 2))
    points_3d = unproject_pixels(pixels, primary_depth[np.rint(pixels[:, 1]).astype(int),
                                                       np.rint(pixels[:, 0]).astype(int)], matrix)
    pixels = (points_3d @ matrix.T)[:, :2] / (points_3d @ matrix.T)[:, 2:3]   # exact depth-pixel positions
    rotation, translation = orbit_pose(anchor.mean(axis=0), 20.0)
    new_pixels, new_z = project(points_3d, rotation, translation, matrix)
    new_depth = np.full((480, 640), np.nan, np.float32)
    keep = (new_pixels[:, 0] >= 0) & (new_pixels[:, 0] < 639) & (new_pixels[:, 1] >= 0) & (new_pixels[:, 1] < 479)
    cols, rows = np.rint(new_pixels[keep]).astype(int).T
    new_depth[rows, cols] = 1.08 * new_z[keep]          # the new view's depth model reads 8% far
    relation = relate_view(pixels[keep], new_pixels[keep], primary_depth, matrix, matrix, anchor,
                           new_depth=new_depth)
    assert relation.accepted
    assert relation.yaw_deg == pytest.approx(20.0, abs=0.5)
    assert relation.depth_scale == pytest.approx(1.08, abs=0.02)
    truth = anchor @ rotation.T + translation
    assert relation.anchor_3d[:, 2] == pytest.approx(1.08 * truth[:, 2], rel=0.02)
    projected_truth, _ = project(anchor, rotation, translation, matrix)
    projected, _ = project(relation.anchor_3d.astype(np.float64), np.eye(3), np.zeros(3), matrix)
    assert np.abs(projected - projected_truth).max() < 1.0   # same image position despite rescaling


def test_relate_view_rejects_unrelated_views():
    matrix = camera_matrix((480, 640), 70)
    depth = np.full((480, 640), 5.0, np.float32)
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0, 600, (60, 2)), rng.uniform(0, 600, (60, 2))
    anchor, _ = oriented_box_from_roi((300, 200, 40, 40), depth, matrix)
    assert not relate_view(a, b, depth, matrix, matrix, anchor).accepted


def test_real_image_view_relation_with_sift():
    image = cv2.imread(str(ROOT / 'reference.jpg'))
    assert image is not None
    height, width = image.shape[:2]
    scale = 640 / max(height, width)
    image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    matrix = camera_matrix((height, width), 60)
    distance = 5.0
    depth = np.full((height, width), distance, np.float32)       # textured plane facing the camera
    anchor, _ = oriented_box_from_roi((width * .4, height * .4, width * .2, height * .2), depth, matrix)
    anchor = anchor.astype(np.float64)
    rotation, translation = orbit_pose(anchor.mean(axis=0), 20.0)
    normal = np.asarray([[0.0, 0.0, 1.0]])
    homography = matrix @ (rotation + translation.reshape(3, 1) @ normal / distance) @ np.linalg.inv(matrix)
    second_view = cv2.warpPerspective(image, homography, (width, height))
    primary_points, new_points = match_views(LocalEngine('sift', 1024), image, second_view)
    relation = relate_view(primary_points, new_points, depth, matrix, matrix, anchor)
    assert relation.accepted and relation.pose.inliers >= 30
    assert relation.yaw_deg == pytest.approx(20.0, abs=2.0)
    expected, _ = project(anchor, rotation, translation, matrix)
    projected, _ = project(relation.anchor_3d.astype(np.float64), np.eye(3), np.zeros(3), matrix)
    assert np.abs(projected - expected).max() < 4.0


def test_pose_smoother_interpolates_holds_and_resets():
    smoother = PoseSmoother(smoothing=0.5, hold_frames=2)
    t = np.asarray([0.0, 0.0, 4.0])
    r0, r1 = np.zeros(3), cv2.Rodrigues(rotation_y(10))[0].ravel()
    first = smoother.update('a', r0, t)
    assert not first[2] and np.allclose(first[0].ravel(), 0)
    middle = smoother.update('a', r1, t + [0.1, 0, 0])
    assert np.linalg.norm(middle[0]) == pytest.approx(math.radians(5), abs=1e-6)
    assert middle[1].ravel()[0] == pytest.approx(0.05)
    assert smoother.update('a')[2] and smoother.update('a')[2]
    assert smoother.update('a') is None                 # hold exhausted: nothing is shown
    restarted = smoother.update('b', r1, t)
    assert np.allclose(restarted[0].ravel(), r1)        # a new reference never blends with the old one


def test_pipeline_stabilise_bridges_short_failures_only():
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.method, pipeline.smoother, pipeline._last_stable = 'depth3d', PoseSmoother(0.5, 1), None
    pose = PoseEstimate(np.zeros((3, 1)), np.asarray([[0.0], [0.0], [4.0]]), np.ones(20, bool), 20, 1.0, True, 0.5)
    found = Prediction('p', 'ref.png', None, [], 5.0, [], pose, np.eye(3), 3.0, 0.0, True)
    lost = Prediction(None, None, None, [], 6.0, [])
    assert pipeline.stabilise(found).place_id == 'p'
    held = pipeline.stabilise(lost)
    assert held.place_id == 'p' and held.held and held.elapsed_ms == 6.0
    assert pipeline.stabilise(lost).place_id is None


def test_end_to_end_register_two_views_then_rescan(tmp_path):
    """Register a first view and an auto-related +20° view, then rescan from +10°."""
    import json
    import yaml
    from stitch.database import PlaceDatabase
    from stitch.multiview import load_primary_view

    image = cv2.imread(str(ROOT / 'reference.jpg'))
    height, width = image.shape[:2]
    scale = 640 / max(height, width)
    image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    matrix = camera_matrix((height, width), 60)
    distance = 5.0
    depth = np.full((height, width), distance, np.float32)
    anchor, _ = oriented_box_from_roi((width * .35, height * .35, width * .3, height * .3), depth, matrix)
    anchor = anchor.astype(np.float64)
    center = anchor.mean(axis=0)
    normal = np.asarray([[0.0, 0.0, 1.0]])

    def view(degrees):
        rotation, translation = orbit_pose(center, degrees)
        homography = matrix @ (rotation + translation.reshape(3, 1) @ normal / distance) @ np.linalg.inv(matrix)
        return cv2.warpPerspective(image, homography, (width, height)), rotation, translation

    overlay = np.zeros((40, 60, 4), np.uint8)
    overlay[..., 2], overlay[..., 3] = 255, 255                     # opaque red post
    cv2.imwrite(str(tmp_path / 'post.png'), overlay)
    document = tmp_path / 'places.json'
    document.write_text(json.dumps({'schema_version': 1, 'places': []}))
    db = PlaceDatabase(str(document), allow_empty=True)

    def register(picture, name, view_depth, corners, yaw):
        cv2.imwrite(str(tmp_path / name), picture)
        metadata = dict(reference_camera_matrix=matrix.tolist(), camera_fov_x_deg=60.0,
                        capture_yaw_offset_deg=yaw)
        db.register('field', 'Field', str(tmp_path / name), str(tmp_path / 'post.png'), 30.0, 78.0,
                    opacity=1.0, capture_metadata=metadata, depth_map=view_depth, anchor_3d_camera=corners)

    register(image, 'front.png', depth, anchor, 0.0)
    side, rotation, translation = view(20.0)
    primary = load_primary_view(PlaceDatabase(str(document)), 'field')
    primary_points, new_points = match_views(LocalEngine('sift', 1024), primary['image'], side)
    # The second view's own depth: the plane seen from the new camera.
    rays = unproject_pixels(np.stack(np.meshgrid(np.arange(width), np.arange(height)), -1).reshape(-1, 2),
                            np.ones(width * height), matrix)
    plane_normal_new = rotation @ np.asarray([0.0, 0.0, 1.0])
    plane_offset_new = distance + plane_normal_new @ translation
    side_depth = (plane_offset_new / (rays @ plane_normal_new)).reshape(height, width).astype(np.float32)
    relation = relate_view(primary_points, new_points, primary['depth'], primary['matrix'], matrix,
                           primary['anchor'], new_depth=side_depth)
    assert relation.accepted and relation.yaw_deg == pytest.approx(20.0, abs=2.0)
    register(side, 'side.png', side_depth, relation.anchor_3d, relation.yaw_deg)

    config = runenv.load_config()  # follows STITCH_CONFIG (CPU default, GPU via configs/gpu.yaml)
    config['camera_fov_x_deg'] = 60.0
    pipeline = Pipeline(PlaceDatabase(str(document)), config, 'depth3d')
    query, rotation, translation = view(10.0)
    prediction = pipeline.stabilise(pipeline.query(query))
    assert prediction.place_id == 'field' and prediction.pose is not None
    assert prediction.view_yaw_deg == pytest.approx(10.0, abs=3.0) and prediction.in_captured_range
    rendered = pipeline.render(query, prediction)
    expected, _ = project(anchor, rotation, translation, matrix)
    cx, cy = np.rint(expected.mean(axis=0)).astype(int)
    assert rendered[cy, cx, 2] > 200 and rendered[cy, cx, :2].max() < 60   # post drawn at the true spot
    far, _, _ = view(60.0)
    outside = pipeline.query(far)
    assert outside.place_id is None or not outside.in_captured_range


def test_pose_rejects_matches_collapsed_onto_one_pixel():
    """Many reference points matched to one blurred query pixel must not yield a pose."""
    matrix = camera_matrix((360, 640), 70)
    depth = np.full((360, 640), 4.0, np.float32)
    rng = np.random.default_rng(1)
    reference = rng.uniform([10, 10], [630, 350], (40, 2)).astype(np.float32)
    collapsed = np.tile([[320.0, 180.0]], (40, 1)).astype(np.float32) + rng.normal(0, .2, (40, 2)).astype(np.float32)
    from stitch.pose3d import estimate_pose
    assert not estimate_pose(reference, collapsed, depth, matrix, matrix, query_size=(640, 360)).accepted
    tight = np.asarray([320, 180], np.float32) + rng.uniform(-10, 10, (40, 2)).astype(np.float32)
    assert not estimate_pose(reference, tight, depth, matrix, matrix, query_size=(640, 360)).accepted


def test_dialog_box_coordinates_map_back_to_full_resolution():
    from stitch.ui_dialogs import display_scale, to_image_box
    scale = display_scale((1080, 1920), 1536, 756)          # 1920x1080 camera frame on a laptop screen
    assert scale == pytest.approx(0.7)
    assert to_image_box((280, 140), (700, 490), scale, (1080, 1920)) == (400, 200, 600, 500)
    assert to_image_box((700, 490), (280, 140), scale, (1080, 1920)) == (400, 200, 600, 500)   # any drag direction
    assert to_image_box((-50, -50), (5000, 5000), scale, (1080, 1920)) == (0, 0, 1920, 1080)  # clamped to the frame
