"""Shared A--D evaluation, oracle-pair geometry, stage ablations and raw predictions."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import yaml
from stitch.features import LocalEngine
from stitch.geometry import verify,corner_error
from stitch.io import read_image
from stitch import runenv
from stitch.retrieval import GlobalEngine,rank

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(runenv.config_path()))
    p.add_argument('--protocol',default='all');p.add_argument('--output',default=str(runenv.results_dir()))
    p.add_argument('--learned-exhaustive',action='store_true',help='Evaluate B on every candidate; otherwise B is oracle-pair geometry only')
    args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(config['threads']);cv2.setNumThreads(config['threads'])
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);cv2.setRNGSeed(config['seed'])
    manifest=json.loads((ROOT/'data/benchmark.json').read_text())
    protocols=[x for x in manifest['protocols'] if args.protocol=='all' or x['name']==args.protocol]
    paths=sorted({r['path'] for x in protocols for r in x['gallery']} |
                 {r['query'] for x in protocols for r in x['queries']})
    images={};scales={}
    for path in paths:images[path],scales[path]=read_image(ROOT/path,config['max_side'])
    engines={m:LocalEngine(m,config['max_keypoints'],config['sift_ratio'],config['device'],
                          depth_confidence=config.get('lightglue_depth_confidence',.95),
                          width_confidence=config.get('lightglue_width_confidence',.99))
             for m in ['sift','superpoint']}
    global_engine=GlobalEngine(config['cosplace_weights'],config['device'],size=config['cosplace_size'])
    feature={m:{} for m in engines};timings={m:{} for m in engines};glob={};global_ms={}
    # Warm models separately from recorded per-image extraction.
    for e in engines.values():e.extract(next(iter(images.values())))
    global_engine.extract(next(iter(images.values())))
    for i,path in enumerate(paths):
        for method,e in engines.items():
            start=time.perf_counter();feature[method][path]=e.extract(images[path])
            timings[method][path]=(time.perf_counter()-start)*1000
        start=time.perf_counter();glob[path]=global_engine.extract(images[path]);global_ms[path]=(time.perf_counter()-start)*1000
        if i%30==0:print('features',i,'/',len(paths),flush=True)
    pair_cache={};pair_rows=[];query_rows=[]
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    def match(method,ref,query):
        key=(method,ref,query)
        if key in pair_cache:return pair_cache[key]
        seed=int(hashlib.sha256((ref+query+method).encode()).hexdigest()[:7],16)
        cv2.setRNGSeed(seed)
        start=time.perf_counter()
        a,b,indices=engines[method].match(feature[method][ref],feature[method][query])
        v=verify(a,b,config['ransac_px'],config['min_inliers'],config['min_inlier_ratio'],images[ref].shape)
        elapsed=(time.perf_counter()-start)*1000
        result=dict(matches=len(a),inliers=v.inliers,inlier_ratio=v.ratio,accepted=v.accepted,
                    match_geometry_ms=elapsed,H=v.H.tolist() if v.H is not None else None)
        pair_cache[key]=result
        return result
    for protocol in protocols:
        name=protocol['name'];gallery=protocol['gallery'];gpaths=[r['path'] for r in gallery]
        matrix=np.stack([glob[r] for r in gpaths])
        for qi,query in enumerate(protocol['queries']):
            q=query['query'];true=query['place_id'];order=rank(glob[q],matrix)
            if query.get('ref'):
                for method in engines:
                    result=match(method,query['ref'],q)
                    Hgt=None
                    if query.get('H_original') is not None:
                        Hgt=scales[q]@np.array(query['H_original'])@np.linalg.inv(scales[query['ref']])
                    err=corner_error(np.array(result['H']) if result['H'] is not None else None,Hgt,images[query['ref']].shape) if Hgt is not None else None
                    pair_rows.append(dict(protocol=name,id=query['id'],method=method,seed=query.get('seed',17),
                                          ref=query['ref'],query=q,place_id=true,**result,
                                          corner_error_px=err if err is not None and math.isfinite(err) else None,
                                          has_homography_gt=Hgt is not None,
                                          homography_success=bool(result['accepted'] and err is not None and err<=5),
                                          feature_ms=timings[method][q]))
            methods=['A_SIFT','C_CosPlace','D_full_top1','D_full_top3','D_full_top5','D_no_geometry']
            if args.learned_exhaustive:methods.insert(1,'B_SP_LightGlue')
            for method in methods:
                accepted=False;selected=None;verification=None
                component_ms=0.;rank_order=order
                if method in ['C_CosPlace','D_no_geometry']:
                    selected=int(order[0]);accepted=method=='D_no_geometry'
                    component_ms=global_ms[q]
                else:
                    local='sift' if method=='A_SIFT' else 'superpoint'
                    candidates=np.arange(len(gallery)) if method in ['A_SIFT','B_SP_LightGlue'] else order[:int(method[-1])]
                    local_results=[match(local,gpaths[i],q) for i in candidates]
                    # Exhaustive methods rank by RANSAC support; ties preserve candidate order.
                    sorting=sorted(range(len(candidates)),key=lambda j:(local_results[j]['inliers'],local_results[j]['inlier_ratio']),reverse=True)
                    ranked=[int(candidates[j]) for j in sorting]
                    rank_order=np.array(ranked+[int(i) for i in order if i not in ranked])
                    valid=[j for j in sorting if local_results[j]['accepted']]
                    if valid:
                        j=valid[0];selected=int(candidates[j]);verification=local_results[j];accepted=True
                    component_ms=timings[local][q]+sum(r['match_geometry_ms'] for r in local_results)
                    if method.startswith('D_'):component_ms+=global_ms[q]
                ranked_ids=[gallery[int(i)]['place_id'] for i in rank_order]
                correct=bool(accepted and true is not None and gallery[selected]['place_id']==true)
                anchoring_success=False
                anchor_error=None
                if correct and verification and query.get('H_original') is not None:
                    Hgt=scales[q]@np.array(query['H_original'])@np.linalg.inv(scales[query['ref']])
                    error=corner_error(np.array(verification['H']),Hgt,images[query['ref']].shape)
                    anchoring_success=bool(error<=5)
                    anchor_error=float(error) if math.isfinite(error) else None
                row=dict(protocol=name,id=query['id'],query=q,method=method,true_place=true,
                         predicted_place=gallery[selected]['place_id'] if selected is not None else None,
                         accepted=accepted,correct_verified=correct,
                         homography_success=anchoring_success,corner_error_px=anchor_error,
                         has_homography_gt=query.get('H_original') is not None,
                         recall1=bool(true is not None and true in ranked_ids[:1]),
                         recall5=bool(true is not None and true in ranked_ids[:5]),
                         recall10=bool(true is not None and true in ranked_ids[:10]),
                         stage_sum_ms=component_ms,ranked_places=ranked_ids,
                         H=verification['H'] if verification else None,
                         inliers=verification['inliers'] if verification else 0)
                query_rows.append(row)
            if qi%10==0:print(name,'query',qi,'/',len(protocol['queries']),'cached pairs',len(pair_cache),flush=True)
        # Save after each protocol so interrupted runs preserve completed work.
        (out/'pairs.json').write_text(json.dumps(pair_rows,indent=2,allow_nan=False))
        (out/'predictions.json').write_text(json.dumps(query_rows,indent=2,allow_nan=False))
    cache_rows=[dict(method=k[0],ref=k[1],query=k[2],**v) for k,v in pair_cache.items()]
    (out/'candidate_pairs.json').write_text(json.dumps(cache_rows,indent=2,allow_nan=False))
    (out/'feature_timings.json').write_text(json.dumps(dict(local=timings,global_ms=global_ms),indent=2))
    np.savez_compressed(out/'global_descriptors.npz',paths=np.array(paths),descriptors=np.stack([glob[p] for p in paths]))
    summary=[]
    for protocol in protocols:
        for method in sorted({r['method'] for r in query_rows}):
            rows=[r for r in query_rows if r['protocol']==protocol['name'] and r['method']==method]
            if not rows:continue
            summary.append(dict(protocol=protocol['name'],method=method,n=len(rows),
                recall1=np.mean([r['recall1'] for r in rows]).item(),recall5=np.mean([r['recall5'] for r in rows]).item(),
                recall10=np.mean([r['recall10'] for r in rows]).item(),
                verified_correct_rate=np.mean([r['correct_verified'] for r in rows]).item(),
                acceptance_rate=np.mean([r['accepted'] for r in rows]).item(),
                median_stage_sum_ms=float(np.median([r['stage_sum_ms'] for r in rows]))))
    with (out/'retrieval.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (out/'run_config.json').write_text(json.dumps(dict(config=config,learned_exhaustive=args.learned_exhaustive,
        timing_definition='Sum of measured warm query feature/global extraction and candidate matching+geometry; cached references, excludes I/O and ranking',
        benchmark_sha256=hashlib.sha256((ROOT/'data/benchmark.json').read_bytes()).hexdigest()),indent=2))
    print('Completed',len(pair_rows),'oracle pair rows,',len(query_rows),'prediction rows',flush=True)

if __name__=='__main__':main()
