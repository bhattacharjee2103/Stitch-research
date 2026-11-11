"""Known transforms, degeneracy and rendering behavior, independent of models."""
import cv2
import numpy as np
import pytest
from stitch.geometry import project,verify,corner_error,valid_homography,pose_errors
from stitch.rendering import render
from stitch.features import LocalEngine

def test_homography_with_outliers():
    rng=np.random.default_rng(17)
    src=rng.uniform(0,300,(100,2))
    H=np.array([[1.1,.03,20],[-.02,.95,15],[.0001,.0002,1]])
    dst=project(src,H)+rng.normal(0,.1,(100,2))
    dst[-30:]=rng.uniform(0,300,(30,2))
    cv2.setRNGSeed(17)
    result=verify(src,dst,threshold=1,shape=(300,300))
    assert result.accepted and result.inliers>=65
    assert corner_error(result.H,H,(300,300))<1

def test_reject_empty_and_collinear():
    assert not verify(np.empty((0,2)),np.empty((0,2))).accepted
    points=np.column_stack([np.arange(20),np.zeros(20)])
    assert not verify(points,points,shape=(100,100)).accepted
    assert not valid_homography(np.zeros((3,3)))
    assert not valid_homography(np.full((3,3),np.nan))

def test_scale_invariance_and_horizon_rejection():
    H=np.eye(3)
    assert valid_homography(20*H,(100,100))
    H[2]=[.1,0,-1]
    assert not valid_homography(H,(100,100))

def test_render_black_pixels_and_alpha():
    frame=np.full((40,40,3),255,np.uint8)
    overlay=np.zeros((40,40,4),np.uint8);overlay[:,:,3]=255
    out=render(frame,overlay,np.eye(3),frame.shape,opacity=1)
    assert np.max(out)==0
    overlay[:,:,3]=0
    assert np.array_equal(render(frame,overlay,np.eye(3),frame.shape),frame)

def test_blank_features_and_short_match_list():
    engine=LocalEngine('sift')
    f=engine.extract(np.zeros((100,100,3),np.uint8))
    a,b,indices=engine.match(f,f)
    assert len(a)==len(b)==len(indices)==0

def test_pose_camera_center_not_translation_vector():
    R=np.eye(3);t=np.array([1,0,0])
    distance,angle=pose_errors(R,t,R,np.zeros(3))
    assert distance==1 and angle==0

def test_grayscale_post_is_rendered():
    frame=np.full((40,40,3),255,np.uint8)
    overlay=np.full((40,40),64,np.uint8)
    result=render(frame,overlay,np.eye(3),frame.shape,opacity=1)
    np.testing.assert_array_equal(result,np.full_like(frame,64))

@pytest.mark.parametrize('scale',[1e-200,1e-12,-1e-12,1e200])
def test_homography_scale_does_not_change_projection_or_rendering(scale):
    H=np.array([[1.,.02,4.],[.01,1.,3.],[.0001,.0002,1.]])
    points=np.array([[10.,20.],[30.,40.]])
    assert valid_homography(H*scale,(40,40))
    np.testing.assert_allclose(project(points,H*scale),project(points,H),atol=1e-10)
    assert corner_error(H*scale,H,(40,40))<1e-10
    frame=np.full((40,40,3),255,np.uint8)
    overlay=np.zeros_like(frame)
    np.testing.assert_array_equal(render(frame,overlay,H*scale,frame.shape),
                                  render(frame,overlay,H,frame.shape))

@pytest.mark.parametrize('anchor',[
    [[5,5],[35,35],[35,5],[5,35]],
    [[5,5],[5,5],[5,5],[5,5]],
    [[0,0],[1,1]],
    [[0,0],[35,0],[35,float('nan')],[0,35]],
])
def test_invalid_anchor_rejected(anchor):
    frame=np.full((40,40,3),255,np.uint8)
    with pytest.raises(ValueError,match='convex quadrilateral'):
        render(frame,np.zeros_like(frame),np.eye(3),frame.shape,np.asarray(anchor))

def test_unrepresentable_homography_rejected_without_svd_failure():
    H=np.eye(3);H[2,2]=1e-320
    assert not valid_homography(H,(40,40))
