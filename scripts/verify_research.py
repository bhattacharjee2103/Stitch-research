"""Audit frozen inputs and independently reconstruct saved evaluation decisions."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from stitch.geometry import corner_error,valid_homography
from stitch.io import read_image

ROOT=Path(__file__).resolve().parents[1]

def read(name):
    return json.loads((ROOT/'results'/name).read_text())

def main():
    checked=0
    def check_file(record):
        nonlocal checked
        path=ROOT/record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],str(path)
        if 'bytes' in record:assert path.stat().st_size==record['bytes'],str(path)
        checked+=1
    for dataset in read('data_manifest.json').values():
        for record in dataset['files']:check_file(record)
    preparation=read('preparation.json')
    for record in preparation['generated_files']:check_file(record)
    models=read('models.json')
    for record in models['files']:check_file(record)
    for name,revision in models['code_revisions'].items():
        actual=subprocess.check_output(['git','-C',str(ROOT/'vendor'/name),'rev-parse','HEAD'],text=True).strip()
        assert actual==revision,name+' revision mismatch'
    for source in ['preparation.json','run_config.json']:
        assert hashlib.sha256((ROOT/'data/benchmark.json').read_bytes()).hexdigest()==read(source)['benchmark_sha256']
    training=read('training.json')
    assert not set(training['train_places']) & set(training['test_places'])
    benchmark=json.loads((ROOT/'data/benchmark.json').read_text())
    run=read('run_config.json');config=run['config']
    with np.load(ROOT/'results/global_descriptors.npz',allow_pickle=False) as saved:
        global_features=dict(zip(saved['paths'],saved['descriptors']))
    pair_cache={(r['method'],r['ref'],r['query']):r for r in read('candidate_pairs.json')}
    reference_shapes={}
    for key,row in pair_cache.items():
        assert 0<=row['inliers']<=row['matches'],key
        ratio=row['inliers']/row['matches'] if row['matches'] else 0.
        assert abs(row['inlier_ratio']-ratio)<1e-12,key
        accepted=False
        if row['inliers']>=config['min_inliers'] and ratio>=config['min_inlier_ratio']:
            if row['ref'] not in reference_shapes:
                reference_shapes[row['ref']]=read_image(ROOT/row['ref'],config['max_side'])[0].shape
            accepted=valid_homography(np.asarray(row['H']) if row['H'] is not None else None,
                                      reference_shapes[row['ref']])
        assert row['accepted']==accepted,key
    predictions=read('predictions.json');derived=read('derived_predictions.json')
    derived={(r['protocol'],r['id'],r['method']):r for r in derived}
    expected=[]
    for protocol in benchmark['protocols']:
        gallery=protocol['gallery']
        matrix=np.stack([global_features[g['path']] for g in gallery])
        for query in protocol['queries']:
            order=np.argsort(-(matrix@global_features[query['query']]),kind='stable')
            for method in ['A_SIFT','B_SP_LightGlue','C_CosPlace','D_full_top1','D_full_top3','D_full_top5','D_no_geometry']:
                key=(protocol['name'],query['id'],method)
                expected.append(key)
                rows=[r for r in predictions if (r['protocol'],r['id'],r['method'])==key]
                assert len(rows)==1,key
                row=rows[0];selected=None;verification=None;rank_order=order
                accepted=method=='D_no_geometry'
                if method in ['C_CosPlace','D_no_geometry']:
                    selected=int(order[0])
                else:
                    local='sift' if method=='A_SIFT' else 'superpoint'
                    candidates=np.arange(len(gallery)) if method in ['A_SIFT','B_SP_LightGlue'] else order[:int(method[-1])]
                    evidence=[pair_cache[local,gallery[i]['path'],query['query']] for i in candidates]
                    sorting=sorted(range(len(candidates)),key=lambda j:(evidence[j]['inliers'],evidence[j]['inlier_ratio']),reverse=True)
                    ranked=[int(candidates[j]) for j in sorting]
                    rank_order=ranked+[int(i) for i in order if i not in ranked]
                    valid=[j for j in sorting if evidence[j]['accepted']]
                    accepted=bool(valid)
                    if valid:
                        j=valid[0];selected=int(candidates[j]);verification=evidence[j]
                ranked_ids=[gallery[i]['place_id'] for i in rank_order]
                predicted=gallery[selected]['place_id'] if selected is not None else None
                assert row['ranked_places']==ranked_ids,key
                assert row['accepted']==accepted and row['predicted_place']==predicted,key
                correct=bool(accepted and query['place_id'] is not None and predicted==query['place_id'])
                assert row['correct_verified']==correct,key
                for k in [1,5,10]:
                    assert row['recall'+str(k)]==bool(query['place_id'] is not None and query['place_id'] in ranked_ids[:k]),key
                assert row['H']==(verification['H'] if verification else None),key
                success=False
                if correct and verification and query.get('H_original') is not None:
                    reference,Sr=read_image(ROOT/query['ref'],config['max_side'])
                    _,Sq=read_image(ROOT/query['query'],config['max_side'])
                    Hgt=Sq@np.array(query['H_original'])@np.linalg.inv(Sr)
                    success=corner_error(np.array(row['H']),Hgt,reference.shape)<=5
                if 'homography_success' in row:
                    assert row['homography_success']==success,key
                assert derived[key]['homography_success']==success,key
    assert len(expected)==len(predictions)==len(derived)
    pairs=read('pairs.json')
    for row in pairs:
        evidence=pair_cache[row['method'],row['ref'],row['query']]
        for name in ['matches','inliers','inlier_ratio','accepted','H']:
            assert row[name]==evidence[name],(row['id'],name)
    with (ROOT/'results/retrieval.csv').open() as f:
        for summary in csv.DictReader(f):
            rows=[r for r in predictions if r['protocol']==summary['protocol'] and r['method']==summary['method']]
            assert len(rows)==int(summary['n'])
            for csv_key,row_key in [('recall1','recall1'),('recall5','recall5'),('recall10','recall10'),
                                    ('verified_correct_rate','correct_verified'),('acceptance_rate','accepted')]:
                assert abs(float(summary[csv_key])-np.mean([r[row_key] for r in rows]))<1e-12
    report=dict(status='passed',input_files_sha256_verified=checked,vendor_revisions_verified=True,
                benchmark_hash_verified=True,training_test_places_disjoint=True,
                prediction_decisions_independently_reconstructed=len(predictions),
                oracle_pairs_cross_checked=len(pairs),candidate_pairs=len(pair_cache),
                candidate_acceptance_decisions_recomputed=len(pair_cache),
                retrieval_csv_consistent=True,homography_success_recomputed=True,
                inference_rerun=False)
    (ROOT/'results/research_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
