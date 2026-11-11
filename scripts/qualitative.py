"""Record actual correspondence coordinates for reproducible match visualisations."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from stitch.features import LocalEngine
from stitch.geometry import verify
from stitch.io import read_image
from stitch import runenv

ROOT=Path(__file__).resolve().parents[1]

def main():
    config=runenv.load_config();out=runenv.results_dir()
    torch.set_num_threads(config['threads']);cv2.setNumThreads(config['threads']);torch.manual_seed(17)
    pairs=json.loads((out/'pairs.json').read_text());rows=[]
    engines={m:LocalEngine(m,512,device=runenv.device(config)) for m in ['sift','superpoint']}
    for protocol in ['hpatches_illumination','hpatches_viewpoint','tokyo_day_to_night','tokyo_day_to_day']:
        ps=[p for p in pairs if p['protocol']==protocol];lookup={(p['id'],p['method']):p for p in ps}
        identifiers=list(dict.fromkeys(p['id'] for p in ps))
        identifier=next((i for i in identifiers if lookup[i,'superpoint']['accepted'] and not lookup[i,'sift']['accepted']),identifiers[-1])
        for method,engine in engines.items():
            previous=lookup[identifier,method]
            reference,_=read_image(ROOT/previous['ref']);query,_=read_image(ROOT/previous['query'])
            a,b=engine.extract(reference),engine.extract(query)
            cv2.setRNGSeed(int(hashlib.sha256((previous['ref']+previous['query']+method).encode()).hexdigest()[:7],16))
            src,dst,indices=engine.match(a,b);v=verify(src,dst,shape=reference.shape)
            rows.append(dict(protocol=protocol,id=identifier,method=method,ref=previous['ref'],query=previous['query'],
                             source_points=src.tolist(),target_points=dst.tolist(),inlier_mask=v.inlier_mask.tolist(),
                             inliers=v.inliers,accepted=v.accepted,H=v.H.tolist() if v.H is not None else None))
    (out/'qualitative_matches.json').write_text(json.dumps(rows,indent=2))
    print('Recorded',len(rows),'actual match visualisations')

if __name__=='__main__':main()
