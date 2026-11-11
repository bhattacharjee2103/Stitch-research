"""Check metadata, original artwork, compilation, PowerPoint and result provenance."""
import hashlib
import json
from pathlib import Path
import re
from pypdf import PdfReader
from pptx import Presentation

ROOT=Path(__file__).resolve().parents[1]

def main():
    paths=[ROOT/'report/thesis.pdf',ROOT/'presentation/Stitch_IITR_Presentation.pdf']
    logs=[ROOT/'report/build/main.log',ROOT/'presentation/build/Stitch_IITR_Presentation.log']
    for log in logs:
        content=log.read_text(errors='replace')
        for pattern in [r'^! ',r'LaTeX Error',r'undefined',r'Overfull',r'Float too large']:
            assert not re.search(pattern,content,re.I|re.M),str(log)+' contains '+pattern
    report,slides=[PdfReader(path) for path in paths]
    reference_path=ROOT.parent/'PRONOB-BHATTACHARJEE-Overleaf/main.pdf'
    reference_layout=json.loads((ROOT/'branding/reference_layout.json').read_text())
    def chapter_geometry(pdf):
        page=next(p for p in pdf.pages if (p.extract_text() or '').startswith('Chapter 1\nIntroduction'))
        rows={}
        def capture(text,cm,tm,font,size):
            if text.strip() in ['Chapter 1','Introduction']:
                rows[text.strip()]=[float(tm[4]),float(tm[5]),float(size)]
        page.extract_text(visitor_text=capture)
        return rows
    original_geometry=reference_layout['chapter_geometry']
    if reference_path.exists():
        assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==reference_layout['reference_sha256'],'Reference PDF changed'
        assert chapter_geometry(PdfReader(reference_path))==original_geometry,'Recorded reference layout differs'
    new_geometry=chapter_geometry(report)
    for label in original_geometry:
        assert all(abs(a-b)<.01 for a,b in zip(original_geometry[label],new_geometry[label])),label+' layout differs from reference'
    for pdf in [report,slides]:
        first='\n'.join(p.extract_text() or '' for p in pdf.pages[:4])
        for name in ['PRONOB','BHATTACHARJEE','25565017','ROORKEE']:
            assert name in first,name+' absent from front matter'
        assert 'Balasubramanian' in first,'Supervisor missing'
        assert '24565022' not in first,'Outdated enrollment copied'
        assert 'Sparsh Mittal' not in first,'Incorrect supervisor copied'
    deck=Presentation(ROOT/'presentation/Stitch_IITR_Presentation.pptx')
    assert len(deck.slides)==len(slides.pages)==30
    assert abs(deck.slide_width/deck.slide_height-4/3)<1e-6
    assert all(len(s.shapes)==1 and s.notes_slide.notes_text_frame.text for s in deck.slides)
    assets=json.loads((ROOT/'results/iitr_assets.json').read_text())
    checked=0
    for record in assets['assets']:
        actual=hashlib.sha256((ROOT/record['destination']).read_bytes()).hexdigest()
        assert actual==record['sha256'],record['destination']
        checked+=1
    provenance=json.loads((ROOT/'results/presentation_data.json').read_text())
    for name,expected in provenance['inputs'].items():
        assert hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest()==expected,name
    report_data=dict(report_pages=len(report.pages),slides=len(slides.pages),powerpoint_slides=len(deck.slides),
        metadata_verified=True,original_asset_copies_verified=checked,measured_result_sources_verified=True,
        latex_errors=0,undefined_references_or_citations=0,overfull_boxes=0,oversized_floats=0,
        supervisor='Dr. R. Balasubramanian',enrollment='25565017',
        original_reference_documents_modified=False,
        reference_chapter_geometry=original_geometry,stitch_chapter_geometry=new_geometry,
        reference_pdf_available=reference_path.exists(),
        chapter_heading_alignment_tolerance_pdf_points=.01,
        powerpoint_editing='Edit supplied LaTeX master; PPTX is a faithful rendered-slide export')
    (ROOT/'results/iitr_documents_verification.json').write_text(json.dumps(report_data,indent=2))
    print(json.dumps(report_data,indent=2))

if __name__=='__main__':main()
