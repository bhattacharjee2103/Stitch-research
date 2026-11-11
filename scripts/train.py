"""Short frozen-backbone contrastive projection ablation; disjoint HPatches places."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stitch.augmentation import appearance,warp
from stitch.io import read_image
from stitch import runenv
from stitch.retrieval import GlobalEngine

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=120);args=p.parse_args()
    config=runenv.load_config();device=runenv.device(config);out=runenv.results_dir()
    torch.set_num_threads(config['threads']);cv2.setNumThreads(config['threads'])
    manifest=json.loads((ROOT/'data/benchmark.json').read_text())
    sequences=json.loads((ROOT/'results/data_manifest.json').read_text())['hpatches']['sequences']
    train_sequences=sequences[:4]+sequences[12:16]
    test_sequences=[s for s in sequences if s not in train_sequences]
    assert not set(train_sequences)&set(test_sequences)
    engine=GlobalEngine(config['cosplace_weights'],device)
    base_features=[];base_labels=[];aug_features=[];aug_labels=[]
    for label,seq in enumerate(train_sequences):
        for n in range(1,7):
            image,_=read_image(ROOT/f'data/hpatches-sequences-release/{seq}/{n}.ppm')
            base_features.append(engine.extract(image));base_labels.append(label)
        reference,_=read_image(ROOT/f'data/hpatches-sequences-release/{seq}/1.ppm')
        for seed in [101,103,107]:
            for condition in ['night','night_severe','warm_night']:
                rng=np.random.default_rng(seed);image,_=warp(reference,rng)
                aug_features.append(engine.extract(appearance(image,condition,rng)));aug_labels.append(label)
    # Test embeddings are evaluated only after head fitting; no test-driven selection.
    protocols=[]
    for proto in manifest['protocols']:
        if proto['name'].startswith('hpatches_'):
            gallery=[g for g in proto['gallery'] if g['place_id'] in test_sequences]
            queries=[q for q in proto['queries'] if q['place_id'] in test_sequences]
        elif proto['name'].startswith('tokyo_'):
            gallery=proto['gallery'];queries=proto['queries']
        else:continue
        if not queries:continue
        protocols.append(dict(name=proto['name'],gallery=gallery,queries=queries))
    paths=sorted({g['path'] for p in protocols for g in p['gallery']} |
                 {q['query'] for p in protocols for q in p['queries']})
    descriptors={path:engine.extract(read_image(ROOT/path)[0]) for path in paths}
    results=[];losses=[]
    for augment in [False,True]:
        data=np.stack(base_features+(aug_features if augment else []))
        labels=np.array(base_labels+(aug_labels if augment else []))
        x=torch.from_numpy(data).to(device);y=torch.from_numpy(labels)
        for seed in [17,23,42]:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            head=nn.Linear(512,128,bias=False).to(device)
            optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
            start=time.perf_counter()
            for step in range(args.steps):
                a=rng.integers(len(x),size=32)
                pos=np.array([rng.choice(np.flatnonzero((labels==labels[i]) & (np.arange(len(labels))!=i))) for i in a])
                neg=np.array([rng.choice(np.flatnonzero(labels!=labels[i])) for i in a])
                za,zp,zn=[F.normalize(head(x[idx]),dim=1) for idx in [a,pos,neg]]
                loss=F.relu(1-(za*zp).sum(1)-(1-(za*zn).sum(1))+.2).mean()
                optimizer.zero_grad();loss.backward();optimizer.step()
                losses.append(dict(seed=seed,augmentation=augment,step=step,loss=float(loss.detach())))
            elapsed=time.perf_counter()-start
            head.eval()
            target=runenv.artifact_dir('models')/f'projection_{"aug" if augment else "plain"}_{seed}.pth'
            torch.save(head.state_dict(),target)
            with torch.inference_mode():
                mapped={p:F.normalize(head(torch.from_numpy(d)[None].to(device)),dim=1)[0].cpu().numpy() for p,d in descriptors.items()}
            for protocol in protocols:
                gallery=protocol['gallery'];queries=protocol['queries']
                baseline=np.stack([descriptors[g['path']] for g in gallery])
                changed=np.stack([mapped[g['path']] for g in gallery])
                for variant,index,qd in [('pretrained_512',baseline,descriptors),('projection_128',changed,mapped)]:
                    hits=[]
                    for query in queries:
                        order=np.argsort(-(index@qd[query['query']]),kind='stable')
                        hits.append([any(gallery[i]['place_id']==query['place_id'] for i in order[:k]) for k in [1,5,10]])
                    recalls=np.asarray(hits).mean(0)
                    results.append(dict(seed=seed,augmentation=augment,protocol=protocol['name'],variant=variant,
                                        n=len(queries),recall1=float(recalls[0]),recall5=float(recalls[1]),recall10=float(recalls[2]),
                                        steps=args.steps,train_s=elapsed,final_loss=float(loss.detach())))
            print('training complete',augment,seed,'seconds',round(elapsed,3),flush=True)
    (out/'training.json').write_text(json.dumps(dict(results=results,losses=losses,device=device,
        train_places=train_sequences,test_places=test_sequences,backbone_frozen=True,
        design='Train only a 512->128 linear head; triplet margin 0.2; no test-based checkpoint selection',
        augmentation_seeds=[101,103,107]),indent=2))

if __name__=='__main__':main()
