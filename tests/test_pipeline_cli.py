"""Exercise reference selection and I/O failures without loading neural models."""
from types import SimpleNamespace
import json
import sys
import cv2
import numpy as np
import pytest
from stitch import cli
from stitch.database import PlaceDatabase
from stitch.geometry import Verification
from stitch.pipeline import Pipeline,Prediction

def test_render_uses_place_and_reference_together(tmp_path):
    document=tmp_path/'places.json'
    document.write_text(json.dumps(dict(schema_version=1,places=[
        dict(id='a',references=['shared.png']),dict(id='b',references=['shared.png'])])))
    cv2.imwrite(str(tmp_path/'post.png'),np.zeros((10,10,3),np.uint8))
    pipeline=Pipeline.__new__(Pipeline)
    pipeline.database=PlaceDatabase(str(document))
    pipeline.images=[np.zeros((20,20,3),np.uint8),np.zeros((40,40,3),np.uint8)]
    verified=Verification(np.eye(3),np.ones(12,bool),12,1.,True)
    prediction=Prediction('b','shared.png',verified,[],0.,[
        dict(asset='post.png',anchor_normalized=[[0,0],[1,0],[1,1],[0,1]],opacity=1)])
    result=pipeline.render(np.full((40,40,3),255,np.uint8),prediction)
    assert result[30,30].max()==0

@pytest.mark.parametrize('command',['query','video'])
def test_cli_reports_output_failure_or_empty_video(tmp_path,monkeypatch,command):
    class FakePipeline:
        def __init__(self,*args):pass
        def query(self,image,latitude=None,longitude=None):return SimpleNamespace()
        def render(self,image,prediction):return image
    monkeypatch.setattr(cli,'Pipeline',FakePipeline)
    monkeypatch.setattr(cli,'PlaceDatabase',lambda path:None)
    monkeypatch.setattr(cli,'read_image',lambda *args:(np.zeros((10,10,3),np.uint8),np.eye(3)))
    monkeypatch.setattr(cli.cv2,'imwrite',lambda *args:False)
    monkeypatch.setattr(cli,'frames',lambda *args:iter([]))
    monkeypatch.setattr(sys,'argv',['stitch',command,'unused.json','input',str(tmp_path/'output')])
    with pytest.raises(OSError,match='Cannot write image|no decodable frames'):
        cli.main()
