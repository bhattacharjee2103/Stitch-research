"""Download official pretrained weights, recording exact SHA256 and code revisions."""
import hashlib
import json
from pathlib import Path
import subprocess
import torch

ROOT=Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT/'models/torch/hub'))

def main():
    from stitch.features import LocalEngine
    from stitch.retrieval import GlobalEngine
    path=ROOT/'models/cosplace_r18_512.pth'
    path.parent.mkdir(exist_ok=True)
    earlier=ROOT/'models_cosplace_download.pth'
    if not path.exists() and earlier.exists():
        earlier.replace(path)
    if not path.exists():
        torch.hub.download_url_to_file('https://github.com/gmberton/CosPlace/releases/download/v1.0/ResNet18_512_cosplace.pth',str(path))
    engine=LocalEngine('superpoint')
    global_engine=GlobalEngine(str(path))
    files=[path]+list((ROOT/'models/torch/hub/checkpoints').glob('*.pth'))
    record={'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,
                     'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
            'code_revisions':{name:subprocess.check_output(['git','-C',str(ROOT/'vendor'/name),
                             'rev-parse','HEAD'],text=True).strip() for name in ['LightGlue','CosPlace']},
            'parameters':{'SuperPoint':sum(p.numel() for p in engine.extractor.parameters()),
                          'LightGlue':sum(p.numel() for p in engine.matcher.parameters()),
                          'CosPlace':sum(p.numel() for p in global_engine.model.parameters())}}
    (ROOT/'results/models.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':
    main()
