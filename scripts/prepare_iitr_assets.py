"""Copy the user's original IITR assets and retain a byte-level provenance manifest."""
import hashlib
import json
from pathlib import Path
import shutil
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT.parent

def main():
    report=ROOT/'report';slides=ROOT/'presentation';brand=ROOT/'branding'
    for path in [report/'assets',slides/'assets',slides/'figures',report/'code',brand]:
        path.mkdir(parents=True,exist_ok=True)
    previous=report/'previous';previous.mkdir(exist_ok=True)
    for filename in ['main.tex','preamble.tex','thesis.pdf']:
        target=previous/('generic_'+filename)
        if not target.exists():shutil.copy2(report/filename,target)
    manifest=[]
    def copy(source,target):
        if source.resolve()!=target.resolve():
            shutil.copy2(source,target)
        original=hashlib.sha256(source.read_bytes()).hexdigest()
        assert original==hashlib.sha256(target.read_bytes()).hexdigest()
        manifest.append(dict(source=str(source),destination=str(target.relative_to(ROOT)),sha256=original))
    for name in ['slide_bg.png','titleslide_bg.png','white_bg.png','iitr.png','IITR.jpg','logo_color.jpg']:
        source=REFERENCE/name
        copy(source if source.exists() else slides/'assets'/name,slides/'assets'/name)
    theme=Path('/Users/bhattacharjee/Downloads/IIT ROORKEE/25565017_seminar/beamerthemeIITR.sty')
    copy(theme if theme.exists() else slides/'beamerthemeIITR.sty',slides/'beamerthemeIITR.sty')
    # Extract the exact embedded crest from the thesis rather than substituting a redraw.
    source=REFERENCE/'PRONOB-BHATTACHARJEE-Overleaf/main.pdf'
    destination=report/'assets/iitr_crest.jpg'
    if source.exists():
        crest=PdfReader(source).pages[0].images[0]
        destination.write_bytes(crest.data)
        manifest.append(dict(source=str(source)+' page 1 / '+crest.name,
                             destination=str(destination.relative_to(ROOT)),sha256=hashlib.sha256(crest.data).hexdigest()))
    else:
        copy(destination,destination)
    for module in ['geometry.py','features.py','pipeline.py','rendering.py']:
        copy(ROOT/'src/stitch'/module,report/'code'/module)
    copy(ROOT/'configs/default.yaml',report/'code/default.yaml')
    for name in ['hpatches_success.pdf','corner_ecdf.pdf','tokyo_retrieval.pdf','tokyo_contact.pdf',
                 'synthetic_contact.pdf','stage_ablation.pdf','training_loss.pdf','quantization.pdf',
                 'efficiency.pdf','qualitative_overlays.pdf','matches_tokyo_day_to_night.pdf',
                 'matches_hpatches_illumination.pdf']:
        copy(ROOT/'figures'/name,slides/'figures'/name)
    if (brand/'project_metadata.tex').exists():
        copy(brand/'project_metadata.tex',report/'metadata.tex')
        copy(brand/'project_metadata.tex',slides/'metadata.tex')
    (ROOT/'results/iitr_assets.json').write_text(json.dumps(dict(
        reference_report=str(source),reference_slides=str(REFERENCE/'IITR_template.pdf'),
        source_documents_modified=False,assets=manifest),indent=2))
    print('Prepared',len(manifest),'original IITR assets and source listings')

if __name__=='__main__':main()
