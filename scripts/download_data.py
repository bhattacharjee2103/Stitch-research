#!/usr/bin/env python3
"""Idempotent, CRC/SHA256-verified public subsets. No archive path extraction."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile
import zlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from remote_zip import RemoteZipIO

ROOT=Path(__file__).resolve().parents[1]
TOKYO='https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Tokyo247/queries/247query_subset_v2.zip'
HP='https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def get_range(url,start,end):
    s=requests.Session()
    s.headers['Connection']='close'
    s.mount('https://',HTTPAdapter(max_retries=Retry(total=5,backoff_factor=.5,
                                  status_forcelist=[429,500,502,503,504])))
    r=s.get(url,headers={'Range':f'bytes={start}-{end-1}'},timeout=120)
    r.raise_for_status()
    if r.status_code!=206 or len(r.content)!=end-start:
        raise IOError('Range response size/status mismatch')
    return r.content

def member(info,url,destination):
    """Download one compressed member, verify publisher ZIP CRC, then rename."""
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        content=destination.read_bytes()
        if len(content)==info['size'] and zlib.crc32(content)==info['crc']:
            return destination
    start=info['offset']
    header=get_range(url,start,start+30)
    fields=struct.unpack('<4s5H3I2H',header)
    if fields[0]!=b'PK\x03\x04':
        raise IOError('Invalid ZIP local header')
    payload=start+30+fields[-2]+fields[-1]
    compressed=get_range(url,payload,payload+info['compressed'])
    content=zlib.decompress(compressed,-15) if info['method']==8 else compressed
    if len(content)!=info['size'] or zlib.crc32(content)!=info['crc']:
        raise IOError('Archive CRC verification failed: '+info['name'])
    temporary=destination.with_suffix(destination.suffix+'.part')
    temporary.write_bytes(content)
    temporary.replace(destination)
    print('verified',destination.relative_to(ROOT),flush=True)
    return destination

def directory(url,cache):
    if cache.is_file():
        return json.loads(cache.read_text())
    remote=RemoteZipIO(url)
    with zipfile.ZipFile(remote) as z:
        records=[dict(name=i.filename,offset=i.header_offset,size=i.file_size,
                      compressed=i.compress_size,crc=i.CRC,method=i.compress_type) for i in z.infolist()]
    cache.write_text(json.dumps(records,indent=2))
    return records

def tokyo(sites):
    records=directory(TOKYO,ROOT/'data/tokyo_zip_directory.json')
    base=ROOT/'data/tokyo'
    csvs=[i for i in records if i['name'].endswith('.csv')]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda i:member(i,TOKYO,base/Path(i['name']).name),csvs))
    rows=[]
    for path in sorted(base.glob('*.csv')):
        f=path.read_text().strip().split(',')
        if len(f)>=9:
            rows.append(dict(image=f[0],site=f[1]+','+f[2],lat=float(f[1]),lon=float(f[2]),
                             view=f[3],condition=f[5],utm=[float(f[-2]),float(f[-1])]))
    grouped={}
    for row in rows:
        grouped.setdefault(row['site'],{}).setdefault((row['view'],row['condition']),row)
    # Fixed choice made using labels only, before inference.
    required=[('F','D'),('L','D'),('F','N'),('F','S')]
    valid=sorted(k for k,v in grouped.items() if all(r in v for r in required))[:sites]
    if len(valid)<sites:
        print('Available complete site groups:',len(valid),'conditions:',sorted({r['condition'] for r in rows}))
        # Some releases label sunset as E; handle only if metadata confirms it.
        if not valid:
            required=[('F','D'),('L','D'),('F','N')]
            valid=sorted(k for k,v in grouped.items() if all(r in v for r in required))[:sites]
    selected=[dict(grouped[k][r],place_id='tokyo_'+str(j).zfill(3))
              for j,k in enumerate(valid) for r in required]
    if not selected:
        raise RuntimeError('No complete labeled day/night groups; inspect CSV metadata')
    names={r['image'] for r in selected}
    images=[i for i in records if Path(i['name']).name in names]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i:member(i,TOKYO,base/Path(i['name']).name),images))
    (base/'selected.json').write_text(json.dumps(selected,indent=2))
    (base/'Readme.txt').write_text('Tokyo 24/7 queries, Torii et al., CVPR 2015. '
        'Use subject to original research dataset terms. No redistribution license inferred.\n')
    return dict(dataset='tokyo_query_cross_time_subset',source=TOKYO,sites=len(valid),
                image_count=len(images),metadata_count=len(csvs),selection=required,
                official_benchmark=False,files=manifest_files(base,[base/r['image'] for r in selected]+list(base.glob('*.csv'))))

def manifest_files(base,paths):
    return [dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(paths)]

def hpatches(per_type):
    archive=ROOT/'data/hpatches-sequences-release.zip'
    base=ROOT/'data/hpatches-sequences-release'
    if zipfile.is_zipfile(archive):
        z=zipfile.ZipFile(archive)
        names=z.namelist()
        records=None
    else:
        records=directory(HP,ROOT/'data/hpatches_zip_directory.json')
        names=[i['name'] for i in records]
        z=None
    sequences=sorted({Path(n).parts[-2] for n in names if n.endswith('/1.ppm')})
    chosen=[s for prefix in ['i_','v_'] for s in sequences if s.startswith(prefix)]
    chosen=[s for prefix in ['i_','v_'] for s in [x for x in chosen if x.startswith(prefix)][:per_type]]
    wanted=[n for n in names if len(Path(n).parts)>=2 and Path(n).parts[-2] in chosen
            and (n.endswith('.ppm') or Path(n).name.startswith('H_1_') or n.endswith('attribs.txt'))]
    if z:
        for name in wanted:
            content=z.read(name)  # zipfile verifies CRC on decompression
            out=base/Path(name).parts[-2]/Path(name).name
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_bytes(content)
        z.close()
    else:
        wanted_set=set(wanted)
        selected=[i for i in records if i['name'] in wanted_set]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i:member(i,HP,base/Path(i['name']).parts[-2]/Path(i['name']).name),selected))
    images=list(base.glob('*/*.ppm'))
    homographies=list(base.glob('*/H_1_*'))
    assert len(images)==len(chosen)*6 and len(homographies)==len(chosen)*5
    result=dict(dataset='hpatches_subset',source=HP,sequences=chosen,image_count=len(images),
                pair_count=len(homographies),selection='First N sequences per prefix in lexicographic order',
                files=manifest_files(base,images+homographies))
    if zipfile.is_zipfile(archive):
        result['archive_sha256']=digest(archive)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['all','hpatches','tokyo'],default='all')
    p.add_argument('--sites',type=int,default=24)
    p.add_argument('--per-type',type=int,default=12)
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    (ROOT/'data').mkdir(exist_ok=True)
    manifest=ROOT/'results/data_manifest.json'
    existing=json.loads(manifest.read_text()) if manifest.exists() else {}
    targets=['hpatches','tokyo'] if args.dataset=='all' else [args.dataset]
    if args.verify_only:
        for target in targets:
            for row in existing[target]['files']:
                path=ROOT/row['path']
                assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],row['path']
            print('Verified',target,len(existing[target]['files']),'files')
        return
    for target in targets:
        if target in existing:
            complete=True
            for row in existing[target]['files']:
                path=ROOT/row['path']
                if not path.is_file():complete=False
                elif digest(path)!=row['sha256']:
                    raise RuntimeError('Corrupt file; restore or remove it before retrying: '+str(path))
            if complete:
                print('Already downloaded and SHA256-verified:',target)
                continue
        previous=existing.get(target)
        result=hpatches(args.per_type) if target=='hpatches' else tokyo(args.sites)
        if previous:
            expected={r['path']:r['sha256'] for r in previous['files']}
            assert {r['path']:r['sha256'] for r in result['files']}==expected,'Re-downloaded data differs from pinned manifest'
        # Merge another dataset completed by a concurrent independent invocation.
        existing=json.loads(manifest.read_text()) if manifest.exists() else existing
        existing[target]=result
        manifest.write_text(json.dumps(existing,indent=2))
        print(target, 'complete',existing[target]['image_count'],'images',flush=True)

if __name__=='__main__':
    main()
