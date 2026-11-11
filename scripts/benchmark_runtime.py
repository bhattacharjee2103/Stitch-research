"""Isolated CPU wall-clock profiling with cached gallery features and peak RSS."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import cv2
import numpy as np
import torch
import yaml
from stitch.database import PlaceDatabase
from stitch.io import read_image
from stitch import runenv
from stitch.pipeline import Pipeline
from stitch.retrieval import GlobalEngine,rank

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['all','classical','local','retrieval','full'],default='all')
    args=p.parse_args();out=runenv.results_dir()
    if args.method=='all':
        for method in ['classical','local','retrieval','full']:
            command=[sys.executable,'scripts/run_logged.py',sys.executable,'scripts/benchmark_runtime.py','--method',method]
            subprocess.run(command,cwd=ROOT,check=True)
        rows=[json.loads((out/f'profile_{m}.json').read_text()) for m in ['classical','local','retrieval','full']]
        (out/'profiles.json').write_text(json.dumps(rows,indent=2));return
    config=runenv.load_config();threads=config['threads']
    torch.set_num_threads(threads);cv2.setNumThreads(threads);torch.manual_seed(17);cv2.setRNGSeed(17)
    manifest=json.loads((ROOT/'data/benchmark.json').read_text())
    proto=next(p for p in manifest['protocols'] if p['name']=='tokyo_day_to_night')
    database_path=ROOT/'data/profile_places.json'
    record=dict(schema_version=1,places=[dict(id=r['place_id'],references=[str(Path(r['path']).relative_to('data'))],
                    content=[dict(id='profile-content',kind='image',asset='../overlay.png',
                                  anchor_normalized=[[.25,.25],[.75,.25],[.75,.6],[.25,.6]],opacity=.8)]) for r in proto['gallery']])
    database_path.write_text(json.dumps(record,indent=2));db=PlaceDatabase(str(database_path))
    # Fixed first six queries include hard examples; selection precedes inference.
    queries=[read_image(ROOT/q['query'],config['max_side'])[0] for q in proto['queries'][:6]]
    rss_before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start=time.perf_counter()
    if args.method=='retrieval':
        engine=GlobalEngine(config['cosplace_weights'],config['device'],size=config['cosplace_size'])
        index=np.stack([engine.extract(read_image(ROOT/g['path'],config['max_side'])[0]) for g in proto['gallery']])
        def call(image):
            result=rank(engine.extract(image),index);sync();return result
    else:
        if args.method=='full':
            db.build(GlobalEngine(config['cosplace_weights'],config['device'],size=config['cosplace_size']),
                     config['max_side'])
        pipeline=Pipeline(db,config,args.method)
        def call(image):
            prediction=pipeline.query(image)
            result=pipeline.render(image,prediction);sync();return result
    def sync():
        if str(config['device']).startswith('cuda'):torch.cuda.synchronize()
    sync();cold_s=time.perf_counter()-start
    call(queries[0]);times=[]
    for repetition in range(3):
        for image in queries:
            start=time.perf_counter();call(image);times.append((time.perf_counter()-start)*1000)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    factor=1 if sys.platform=='darwin' else 1024
    result=dict(method=args.method,device=str(config['device']),threads=threads,gallery_size=len(proto['gallery']),
                queries=6,repetitions=3,n=len(times),cold_model_and_index_s=cold_s,
                median_ms=float(np.median(times)),mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),
                process_peak_rss_bytes=int(peak*factor),pre_model_peak_rss_bytes=int(rss_before*factor),
                definition='Warm actual query and overlay rendering; query images predecoded; gallery cached; full includes content asset I/O; excludes camera decoding and display',
                samples_ms=times)
    (out/f'profile_{args.method}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
