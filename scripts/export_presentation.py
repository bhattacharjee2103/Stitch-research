"""Export LaTeX slides to a visually identical PPTX, with extracted text in notes."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from pptx import Presentation
from pptx.util import Inches
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]

def main():
    directory=ROOT/'presentation';pdf=directory/'Stitch_IITR_Presentation.pdf'
    reader=PdfReader(pdf)
    raster=directory/'build/rendered';raster.mkdir(parents=True,exist_ok=True)
    for old in raster.glob('slide-*.png'):old.unlink()
    command=['pdftoppm','-scale-to-x','1920','-scale-to-y','-1','-png',str(pdf),str(raster/'slide')]
    subprocess.run(command,check=True)
    images=sorted(raster.glob('slide-*.png'),key=lambda p:int(p.stem.split('-')[-1]))
    assert len(images)==len(reader.pages)
    deck=Presentation();deck.slide_width=Inches(10);deck.slide_height=Inches(7.5)
    deck.core_properties.title='Stitch: Appearance-Invariant Visual Place Recognition'
    deck.core_properties.author='PRONOB BHATTACHARJEE'
    deck.core_properties.subject='IIT Roorkee research presentation; supervisor Dr. R. Balasubramanian'
    deck.core_properties.comments='Faithful export of LaTeX/Beamer PDF. Edit the supplied .tex source to change slide content.'
    for i,(page,path) in enumerate(zip(reader.pages,images),1):
        slide=deck.slides.add_slide(deck.slide_layouts[6])
        image=slide.shapes.add_picture(str(path),0,0,width=deck.slide_width,height=deck.slide_height)
        text=page.extract_text() or ''
        image.name=f'LaTeX slide {i}'
        image._element.nvPicPr.cNvPr.set('descr',text[:10000])
        slide.notes_slide.notes_text_frame.text='LaTeX/Beamer source is the editable master.\n\n'+text
    target=directory/'Stitch_IITR_Presentation.pptx';deck.save(target)
    check=Presentation(target);assert len(check.slides)==len(reader.pages)
    report=dict(pdf_pages=len(reader.pages),pptx_slides=len(check.slides),aspect_ratio='4:3',
        rendered_image_width_px=1920,export_type='Full-slide images preserving exact LaTeX layout; editable master is the supplied TeX',
        speaker_notes='Extracted slide text',render_command=command,
        pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        pptx_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (ROOT/'results/presentation_export.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
