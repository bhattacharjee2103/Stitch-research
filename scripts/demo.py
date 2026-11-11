"""Generate day/night query clips from real photographs and compare actual pipelines."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import yaml
from stitch.database import PlaceDatabase
from stitch.io import read_image
from stitch import runenv
from stitch.pipeline import Pipeline
from stitch.retrieval import GlobalEngine

ROOT=Path(__file__).resolve().parents[1]

def main():
    config=runenv.load_config();out=runenv.results_dir()
    torch.set_num_threads(config['threads']);cv2.setNumThreads(config['threads']);torch.manual_seed(17);cv2.setRNGSeed(17)
    benchmark=json.loads((ROOT/'data/benchmark.json').read_text())
    proto=next(p for p in benchmark['protocols'] if p['name']=='tokyo_day_to_night')
    predictions=json.loads((out/'predictions.json').read_text())
    night=[r for r in predictions if r['protocol']=='tokyo_day_to_night']
    lookup={(r['id'],r['method']):r for r in night}
    selected=next((q for q in proto['queries'] if lookup[q['id'],'D_full_top3']['correct_verified'] and
                    not lookup[q['id'],'A_SIFT']['correct_verified']),proto['queries'][0])
    dbfile=ROOT/'data/demo_places.json'
    places=[dict(id=g['place_id'],references=[str(Path(g['path']).relative_to('data'))],
                 content=[dict(id='example-post',kind='image',asset='../overlay.png',opacity=.75,
                               anchor_normalized=[[.25,.25],[.75,.25],[.75,.65],[.25,.65]])]) for g in proto['gallery']]
    dbfile.write_text(json.dumps(dict(schema_version=1,places=places),indent=2))
    db=PlaceDatabase(str(dbfile))
    db.build(GlobalEngine(config['cosplace_weights'],config['device'],size=config['cosplace_size']),
             config['max_side'])
    pipelines={m:Pipeline(db,config,m) for m in ['classical','full']}
    directory=runenv.artifact_dir('demo')
    video=cv2.VideoWriter(str(directory/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(960,430))
    if not video.isOpened():raise OSError('Cannot open demo writer')
    rows=[];gifs=[]
    for condition,path in [('day',selected['ref']),('night',selected['query'])]:
        source,_=read_image(ROOT/path,config['max_side'])
        h,w=source.shape[:2]
        clip=cv2.VideoWriter(str(directory/f'{condition}_query.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(w,h))
        for frame_id in range(24):
            phase=2*np.pi*frame_id/24
            M=cv2.getRotationMatrix2D((w/2,h/2),np.sin(phase)*1.3,1.02)
            M[:,2]+=np.array([8*np.sin(phase),4*np.cos(phase)])
            query=cv2.warpAffine(source,M,(w,h));clip.write(query)
            panels=[]
            for method,pipeline in pipelines.items():
                prediction=pipeline.query(query);output=pipeline.render(query,prediction)
                panel=cv2.resize(output,(480,360))
                header=np.full((70,480,3),(24,31,38),np.uint8)
                name='SIFT + FLANN + RANSAC' if method=='classical' else 'CosPlace + SuperPoint + LightGlue'
                cv2.putText(header,name,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.53,(245,245,245),1,cv2.LINE_AA)
                status=('verified '+prediction.place_id) if prediction.place_id else 'no verified anchor'
                cv2.putText(header,status+f' | {prediction.elapsed_ms:.0f} ms',(12,49),cv2.FONT_HERSHEY_SIMPLEX,.48,
                            (100,225,145) if prediction.place_id else (100,150,255),1,cv2.LINE_AA)
                cv2.putText(panel,condition.upper()+' | generated 2D camera motion',(12,342),
                            cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
                panels.append(np.vstack([header,panel]))
                rows.append(dict(condition=condition,frame=frame_id,method=method,predicted=prediction.place_id,
                                 true_place=selected['place_id'],latency_ms=prediction.elapsed_ms,
                                 accepted=prediction.place_id is not None))
            combined=np.hstack(panels);video.write(combined)
            if frame_id%2==0:gifs.append(Image.fromarray(cv2.cvtColor(cv2.resize(combined,(768,344)),cv2.COLOR_BGR2RGB)))
        clip.release()
    video.release()
    gifs[0].save(directory/'comparison.gif',save_all=True,append_images=gifs[1:],duration=250,loop=0)
    (out/'demo.json').write_text(json.dumps(dict(source_images=[selected['ref'],selected['query']],
        selection_rule='First real night query where full top3 verifies the right place and SIFT does not; otherwise first query',
        simulated_camera_motion=True,real_temporal_video=False,rows=rows),indent=2))
    print('Created',directory/'comparison.mp4','and GIF; selected',selected['id'])

if __name__=='__main__':main()
