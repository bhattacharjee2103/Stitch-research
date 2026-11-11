"""Replay saved adaptation and exported-model accuracy without retraining."""
import json
from pathlib import Path
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn import functional as F
from stitch.io import read_image
from stitch.retrieval import image_tensor

ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(2);cv2.setNumThreads(2)
    benchmark=json.loads((ROOT/'data/benchmark.json').read_text())
    protocols={p['name']:p for p in benchmark['protocols']}
    training=json.loads((ROOT/'results/training.json').read_text())
    exports=json.loads((ROOT/'results/export.json').read_text())
    with np.load(ROOT/'results/global_descriptors.npz',allow_pickle=False) as saved:
        descriptors=dict(zip(saved['paths'],saved['descriptors']))
    def recalls(protocol,values):
        matrix=np.stack([values[g['path']] for g in protocol['gallery']]);hits=[]
        for query in protocol['queries']:
            order=np.argsort(-(matrix@values[query['query']]),kind='stable')
            hits.append([any(protocol['gallery'][i]['place_id']==query['place_id'] for i in order[:k]) for k in [1,5,10]])
        return np.mean(hits,axis=0)
    heads={}
    with torch.inference_mode():
        for row in training['results']:
            protocol=protocols[row['protocol']]
            if row['protocol'].startswith('hpatches_'):
                protocol=dict(gallery=[g for g in protocol['gallery'] if g['place_id'] in training['test_places']],
                              queries=[q for q in protocol['queries'] if q['place_id'] in training['test_places']])
            values=descriptors
            if row['variant']=='projection_128':
                key=(row['augmentation'],row['seed'])
                if key not in heads:
                    name='projection_'+('aug' if key[0] else 'plain')+'_'+str(key[1])+'.pth'
                    weight=torch.load(ROOT/'models'/name,map_location='cpu',weights_only=True)['weight']
                    assert tuple(weight.shape)==(128,512)
                    heads[key]={name:F.normalize(F.linear(torch.from_numpy(value)[None],weight),dim=1)[0].numpy()
                                for name,value in descriptors.items()}
                values=heads[key]
            actual=recalls(protocol,values)
            assert len(protocol['queries'])==row['n']
            np.testing.assert_allclose(actual,[row['recall1'],row['recall5'],row['recall10']],atol=1e-12,rtol=0)
    selected=[p for p in protocols.values() if p['name'].startswith('tokyo_')]
    paths=sorted({g['path'] for p in selected for g in p['gallery']}|{q['query'] for p in selected for q in p['queries']})
    tensors={p:image_tensor(read_image(ROOT/p)[0]).numpy() for p in paths}
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    export_checks=[]
    for record in exports['models']:
        path=ROOT/record['path']
        assert path.stat().st_size==record['bytes']
        onnx.checker.check_model(str(path))
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        actual={p:session.run(None,{'image':tensor})[0][0] for p,tensor in tensors.items()}
        for row in record['retrieval']:
            protocol=protocols[row['protocol']]
            assert row['n']==len(protocol['queries'])
            np.testing.assert_allclose(recalls(protocol,actual),[row['recall1'],row['recall5'],row['recall10']],atol=1e-12,rtol=0)
        error=max(float(np.max(np.abs(actual[p]-descriptors[p]))) for p in paths)
        assert error<=record['max_abs_parity_error']+1e-5,(record['variant'],error)
        export_checks.append(dict(variant=record['variant'],images_replayed=len(paths),
                                  retrieval_scores_match=True,max_abs_error_to_native_cache=error))
    profiles=json.loads((ROOT/'results/profiles.json').read_text())
    for row in profiles:
        values=np.asarray(row['samples_ms'])
        assert len(values)==row['n']
        for name,actual in [('median_ms',np.median(values)),('mean_ms',np.mean(values)),('p95_ms',np.percentile(values,95))]:
            assert abs(row[name]-actual)<1e-9
    report=dict(status='passed',trained_heads_replayed=len(heads),training_result_rows_reproduced=len(training['results']),
                exported_models=export_checks,profile_aggregates_checked=len(profiles),
                retraining_performed=False,latency_rebenchmarked=False)
    (ROOT/'results/model_artifacts_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
