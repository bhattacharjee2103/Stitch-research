"""Snapshot the exact installed distribution set and immutable upstream sources."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    lines=[]
    for d in sorted(metadata.distributions(),key=lambda d:d.metadata['Name'].lower()):
        name=d.metadata['Name']
        if name.lower() in ('stitch-ar-research','lightglue'):continue
        lines.append(f'{name}=={d.version}')
    lines.extend(['-e .','-e vendor/LightGlue'])
    (ROOT/'requirements-lock.txt').write_text('\n'.join(lines)+'\n')
    source={}
    for filename in ['tep.py','tep2.py','tep3.py']:
        original=subprocess.check_output(['git','show','HEAD:'+filename],cwd=ROOT)
        current=(ROOT/filename).read_bytes()
        assert original==current,filename+' no longer matches inspected upstream source'
        source[filename]=hashlib.sha256(current).hexdigest()
    report=dict(interpreter=sys.version,platform=platform.platform(),
                upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                original_scripts_sha256=source,original_scripts_unchanged=True)
    (ROOT/'results/source_audit.json').write_text(json.dumps(report,indent=2))
    print('Saved dependency lock; original scripts match upstream exactly')

if __name__=='__main__':main()
