"""Create self-contained PDF/PowerPoint deliverables and editable LaTeX ZIP projects."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def archive(path,files,extra):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for source,name in files:z.write(source,name)
        for name,text in extra.items():z.writestr(name,text)

def main():
    out=ROOT/'deliverables';out.mkdir(exist_ok=True)
    report=ROOT/'report';slides=ROOT/'presentation'
    for source,name in [(report/'thesis.pdf','Stitch_IITR_Report.pdf'),
                        (slides/'Stitch_IITR_Presentation.pdf','Stitch_IITR_Presentation.pdf'),
                        (slides/'Stitch_IITR_Presentation.pptx','Stitch_IITR_Presentation.pptx')]:
        shutil.copy2(source,out/name)
    report_files=[(report/name,name) for name in ['main.tex','preamble.tex','frontmatter.tex','metadata.tex',
                                                 'references.bib','.latexmkrc']]
    for folder in ['chapters','figures','tables','assets','code']:
        report_files.extend((p,str(p.relative_to(report))) for p in (report/folder).rglob('*') if p.is_file())
    report_readme='''STITCH — IIT ROORKEE REPORT: EDITABLE LATEX PROJECT

Author: PRONOB BHATTACHARJEE, enrollment 25565017.
Supervisor: Dr. R. Balasubramanian (confirmed by the author).

OVERLEAF: Upload this ZIP as a project; select main.tex and pdfLaTeX.
LOCAL: run latexmk -pdf -outdir=build main.tex, or make.
The .latexmkrc also generates the nomenclature through MakeIndex.

All crests, figures, tables and code listings are bundled. No parent folder,
external image path, Python environment, dataset or model download is required
to compile this fixed report snapshot. Numeric tables are generated from the
retained Stitch measurements; the full research project supplies regeneration.

The original reference report was not overwritten. The cover layout, margins,
Latin Modern typography, institutional wording and embedded crest follow the
supplied PRONOB BHATTACHARJEE thesis. The project title/content are Stitch.
Certification remains unsigned for supervisor/institutional review.
'''
    archive(out/'Stitch_IITR_Report_LaTeX.zip',report_files,{
        'README.txt':report_readme,
        'Makefile':'all:\n\tlatexmk -pdf -outdir=build main.tex\n\tcp build/main.pdf Stitch_IITR_Report.pdf\n'})
    slide_files=[(slides/name,name) for name in ['Stitch_IITR_Presentation.tex','metadata.tex',
        'results_numbers.tex','results_tables.tex','beamerthemeIITR.sty']]
    for folder in ['assets','figures']:
        slide_files.extend((p,str(p.relative_to(slides))) for p in (slides/folder).rglob('*') if p.is_file())
    slide_readme='''STITCH — IIT ROORKEE PRESENTATION: EDITABLE LATEX PROJECT

Author: PRONOB BHATTACHARJEE, enrollment 25565017.
Supervisor: Dr. R. Balasubramanian.

OVERLEAF: Upload this ZIP; select Stitch_IITR_Presentation.tex and pdfLaTeX.
LOCAL: latexmk -pdf -outdir=build Stitch_IITR_Presentation.tex, or make.

The original beamerthemeIITR.sty, institutional crests, slide_bg.png,
titleslide_bg.png campus photograph and white_bg.png are bundled.
Results are the actual retained Stitch measurements; no experiments are rerun
for this presentation. It uses the original 4:3 IITR theme.

POWERPOINT: The companion PPTX is a faithful full-slide-image export of this
LaTeX PDF at 1920px width, with extracted text in speaker notes. Edit the TeX
master to change slide text, then recompile/export; PPTX text boxes are not
separately editable. This preserves the exact math, fonts and artwork.
'''
    archive(out/'Stitch_IITR_Presentation_LaTeX.zip',slide_files,{
        'README.txt':slide_readme,
        'Makefile':'all:\n\tlatexmk -pdf -outdir=build Stitch_IITR_Presentation.tex\n\tcp build/Stitch_IITR_Presentation.pdf Stitch_IITR_Presentation.pdf\n'})
    (out/'README.txt').write_text(report_readme+'\n'+slide_readme)
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
