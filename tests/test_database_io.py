"""Data and coordinate contracts that prevent stale or misplaced anchors."""
import json
import cv2
import numpy as np
import pytest
from stitch.database import PlaceDatabase
from stitch.geometry import project
from stitch.io import read_image

def test_resize_homography_coordinate_conjugation(tmp_path):
    path=tmp_path/'image.png';cv2.imwrite(str(path),np.zeros((777,1001,3),np.uint8))
    image,S=read_image(path,640)
    H=np.array([[1,.01,12],[.02,.9,7],[.0001,.0002,1.]])
    points=np.array([[20.,30.],[400.,500.]])
    expected=project(project(points,H),S)
    actual=project(project(points,S),S@H@np.linalg.inv(S))
    np.testing.assert_allclose(actual,expected,atol=1e-9)
    assert max(image.shape[:2])==640

def test_reference_change_invalidates_index(tmp_path):
    path=tmp_path/'reference.png';cv2.imwrite(str(path),np.zeros((40,40,3),np.uint8))
    document=tmp_path/'places.json';document.write_text(json.dumps(dict(schema_version=1,
        places=[dict(id='a',references=['reference.png'],content=[])])))
    class FakeEncoder:
        size=224;enhance=False
        def extract(self,image):return np.array([1.,0.],np.float32)
    engine=FakeEncoder();db=PlaceDatabase(str(document));db.build(engine)
    assert db.load_descriptors(engine).shape==(1,2)
    cv2.imwrite(str(path),np.ones((40,40,3),np.uint8)*255)
    with pytest.raises(ValueError,match='changed'):db.load_descriptors(engine)

def test_duplicate_place_ids_rejected(tmp_path):
    document=tmp_path/'places.json';document.write_text(json.dumps(dict(schema_version=1,
        places=[dict(id='same',references=['a.png']),dict(id='same',references=['b.png'])])))
    with pytest.raises(ValueError,match='unique'):PlaceDatabase(str(document))

@pytest.mark.parametrize('change',['weights','resize','version','fingerprints','descriptors'])
def test_incompatible_or_damaged_index_rejected(tmp_path,change):
    cv2.imwrite(str(tmp_path/'reference.png'),np.zeros((40,40,3),np.uint8))
    document=tmp_path/'places.json'
    document.write_text(json.dumps(dict(schema_version=1,places=[dict(id='a',references=['reference.png'])])))
    class FakeEncoder:
        size=224;enhance=False;weights_sha256='original-checkpoint'
        def extract(self,image):return np.array([1.,0.],np.float32)
    engine=FakeEncoder();db=PlaceDatabase(str(document));db.build(engine)
    max_side=640
    if change=='weights':engine.weights_sha256='different-checkpoint'
    elif change=='resize':max_side=320
    else:
        with np.load(db.index_path,allow_pickle=False) as saved:
            values={name:saved[name] for name in saved.files}
        if change=='version':values['schema_version']=np.array(1)
        elif change=='fingerprints':values['fingerprints']=np.array([])
        else:values['descriptors']=np.array([[np.nan,0]])
        np.savez_compressed(db.index_path,**values)
    with pytest.raises(ValueError,match='rebuild database'):
        db.load_descriptors(engine,max_side)

def test_content_anchor_stays_on_its_reference(tmp_path):
    document=tmp_path/'places.json'
    document.write_text(json.dumps(dict(schema_version=1,places=[dict(id='a',
        references=['front.png','side.png'],content=[dict(id='default'),
        dict(id='explicit-side',reference='side.png')])])) )
    db=PlaceDatabase(str(document))
    assert [c['id'] for c in db.rows[0]['content']]==['default']
    assert [c['id'] for c in db.rows[1]['content']]==['explicit-side']

def test_empty_database_rejected(tmp_path):
    document=tmp_path/'places.json'
    document.write_text(json.dumps(dict(schema_version=1,places=[])))
    with pytest.raises(ValueError,match='at least one place'):
        PlaceDatabase(str(document))

@pytest.mark.parametrize('anchor',[
    [[0,0],[1,1],[1,0],[0,1]],
    [[0,0],[2,0],[2,1],[0,1]],
])
def test_invalid_normalized_anchor_rejected_before_inference(tmp_path,anchor):
    document=tmp_path/'places.json'
    document.write_text(json.dumps(dict(schema_version=1,places=[dict(id='a',references=['ref.png'],
        content=[dict(anchor_normalized=anchor)])])))
    with pytest.raises(ValueError,match='Normalized anchor'):
        PlaceDatabase(str(document))
