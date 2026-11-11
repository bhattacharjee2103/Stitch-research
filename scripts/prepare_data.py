"""Freeze evaluation manifests before model inference; all query labels are explicit."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from stitch.augmentation import appearance,warp
from stitch.io import read_image

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'results/data_manifest.json').read_text())
    protocols=[];pairs=[];hp_gallery=[]
    for seq in manifest['hpatches']['sequences']:
        ref=f'data/hpatches-sequences-release/{seq}/1.ppm'
        hp_gallery.append(dict(path=ref,place_id=seq))
        for n in range(2,7):
            query=f'data/hpatches-sequences-release/{seq}/{n}.ppm'
            H=np.loadtxt(ROOT/f'data/hpatches-sequences-release/{seq}/H_1_{n}')
            pairs.append(dict(id=f'{seq}_{n}',dataset='hpatches',split='illumination' if seq.startswith('i_') else 'viewpoint',
                              ref=ref,query=query,place_id=seq,H_original=H.tolist(),seed=17))
    for split in ['illumination','viewpoint']:
        protocols.append(dict(name='hpatches_'+split,gallery=hp_gallery,
                              queries=[dict(p) for p in pairs if p['dataset']=='hpatches' and p['split']==split]))
    tokyo=json.loads((ROOT/'data/tokyo/selected.json').read_text())
    sites=sorted({r['place_id'] for r in tokyo})
    maps={r['place_id']:{} for r in tokyo}
    for r in tokyo:maps[r['place_id']][r['view'],r['condition']]=r
    for name,ref_condition,query_view,query_condition in [
        ('day_to_day','D','L','D'),('day_to_night','D','F','N'),
        ('night_to_day','N','F','D'),('day_to_sunset','D','F','S')]:
        gallery=[];queries=[]
        for site in sites:
            records=maps[site]
            if ('F',ref_condition) not in records or (query_view,query_condition) not in records:continue
            ref='data/tokyo/'+records['F',ref_condition]['image']
            query='data/tokyo/'+records[query_view,query_condition]['image']
            gallery.append(dict(path=ref,place_id=site))
            pair=dict(id=site+'_'+name,dataset='tokyo',split=name,ref=ref,query=query,
                      place_id=site,H_original=None,seed=17)
            pairs.append(pair);queries.append(pair)
        if queries:protocols.append(dict(name='tokyo_'+name,gallery=gallery,queries=queries))
    # The repository has no place identities: label these as source-image identity.
    repo_gallery=[dict(path=p,place_id='repo_'+Path(p).stem) for p in ['reference.jpg','test.jpg']]
    synth_gallery=hp_gallery+repo_gallery
    synth_bases=repo_gallery+hp_gallery[:2]+hp_gallery[12:14]
    synthetic=[]
    for item in synth_bases:
        image,_=read_image(ROOT/item['path'],640)
        for seed in [17,23,42]:
            for condition in ['day','night','night_severe','warm_night','fog']:
                rng=np.random.default_rng(seed)
                transformed,H=warp(image,rng)
                transformed=appearance(transformed,condition,rng)
                path=f'data/synthetic/{item["place_id"]}_{condition}_{seed}.png'
                (ROOT/path).parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(ROOT/path),transformed)
                # H maps resized reference pixels to the generated query; convert from original reference.
                _,S=read_image(ROOT/item['path'],640)
                pair=dict(id=Path(path).stem,dataset='synthetic',split=condition,ref=item['path'],
                          query=path,place_id=item['place_id'],H_original=(H@S).tolist(),seed=seed)
                pairs.append(pair);synthetic.append(pair)
    for condition in ['day','night','night_severe','warm_night','fog']:
        protocols.append(dict(name='synthetic_'+condition,gallery=synth_gallery,
                              queries=[p for p in synthetic if p['split']==condition]))
    # Open-set queries come from different collections, never from this gallery.
    protocols.append(dict(name='open_set',gallery=[dict(path='data/tokyo/'+maps[s]['F','D']['image'],place_id=s) for s in sites],
                          queries=[dict(id='unknown_'+r['place_id'],query=r['path'],place_id=None,ref=None,
                                        dataset='open_set',split='unknown',seed=17,H_original=None) for r in hp_gallery[:12]]))
    document=dict(schema_version=1,direction='reference condition -> query condition',
                  geometry='H_original maps original reference pixels to original query pixels',
                  protocols=protocols,pairs=pairs)
    (ROOT/'data/benchmark.json').write_text(json.dumps(document,indent=2))
    provenance=dict(pairs=len(pairs),protocols={p['name']:dict(gallery=len(p['gallery']),queries=len(p['queries'])) for p in protocols},
                    synthetic_conditions=['day','night','night_severe','warm_night','fog'],seeds=[17,23,42],
                    generated_files=[dict(path=p['query'],sha256=hashlib.sha256((ROOT/p['query']).read_bytes()).hexdigest()) for p in synthetic],
                    benchmark_sha256=hashlib.sha256((ROOT/'data/benchmark.json').read_bytes()).hexdigest())
    (ROOT/'results/preparation.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({k:v for k,v in provenance.items() if k!='generated_files'},indent=2))
    # A reviewable per-place content database, with normalized reference anchors.
    places=[]
    for r in synth_gallery:
        path=str(Path(r['path']).relative_to('data')) if r['path'].startswith('data/') else '../'+r['path']
        places.append(dict(id=r['place_id'],references=[path],content=[dict(id=r['place_id']+'_post',kind='image',
                          asset='../overlay.png',author='example-user',created_at='2026-11-05',
                          anchor_normalized=[[.2,.2],[.8,.2],[.8,.65],[.2,.65]],opacity=.8)]))
    (ROOT/'data/places.json').write_text(json.dumps(dict(schema_version=1,places=places),indent=2))

if __name__=='__main__':main()
