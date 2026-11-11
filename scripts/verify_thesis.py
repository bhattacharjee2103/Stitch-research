"""Verify compiled report, references, result provenance, metrics, and demo decoding."""
import hashlib
import json
from pathlib import Path
import re
import cv2
import numpy as np
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]

def main():
    thesis=ROOT/'report';log=(thesis/'build/main.log').read_text(errors='replace')
    bad=[]
    for pattern in [r'^! ',r'LaTeX Error',r'Undefined control sequence',
                    r'(?:Reference|Citation) .+? undefined',r'There were undefined',
                    r'undefined references',r'undefined citations']:
        if re.search(pattern,log,re.M|re.I):bad.append(pattern)
    assert not bad,'LaTeX errors/unresolved references: '+str(bad)
    for extension in ['nls','lof','lot','loa']:
        path=thesis/'build'/('main.'+extension)
        assert path.is_file() and path.stat().st_size>0,'Missing front-matter index: '+extension
    pdf=PdfReader(str(thesis/'thesis.pdf'))
    def destinations(items):
        out={}
        for item in items:
            if isinstance(item,list):out.update(destinations(item))
            else:
                try:out[str(item.title)]=pdf.get_destination_page_number(item)
                except Exception:pass
        return out
    outline=destinations(pdf.outline)
    intro=next(v for k,v in outline.items() if k=='Introduction')
    bibliography=next(v for k,v in outline.items() if k=='Bibliography')
    body_pages=bibliography-intro
    full_edition='\\FullReporttrue' in (thesis/'main.tex').read_text().split('\\newif\\ifFullReport',1)[1].split('\\begin{document}',1)[0]
    page_range=(200,10**6) if full_edition else (80,90)
    assert page_range[0]<=len(pdf.pages)<=page_range[1],f'Thesis has {len(pdf.pages)} pages; expected {page_range} for this edition'
    assert body_pages>=60,'Fewer than 60 substantive chapter pages: '+str(body_pages)
    evidence=json.loads((ROOT/'results/extended_evidence.json').read_text())
    for name,expected in evidence['inputs'].items():
        assert hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest()==expected,'Changed evidence input: '+name
    for group in ['extra_inputs','generated_outputs']:
        for name,expected in evidence[group].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,'Changed evidence asset: '+name
    assert evidence['hpatches_sequences']==24 and evidence['tokyo_sites']==24
    assert evidence['oracle_pair_rows']==612
    assert len(list((thesis/'figures/evidence').glob('*.pdf')))==48
    all_text='\n'.join(page.extract_text() or '' for page in pdf.pages)
    assert '\ufffd' not in all_text and '??' not in all_text,'Unresolved or malformed PDF text'
    for sequence in ([] if not full_edition else json.loads((ROOT/'results/data_manifest.json').read_text())['hpatches']['sequences']):
        assert 'Sequence '+sequence in all_text,'Missing sequence atlas entry: '+sequence
    for site in ([] if not full_edition else sorted({row['place_id'] for row in json.loads((ROOT/'data/tokyo/selected.json').read_text())})):
        assert 'Site '+site in all_text,'Missing site atlas entry: '+site
    bibliography_source=(thesis/'references.bib').read_text()
    defined=set(re.findall(r'@\w+\s*\{\s*([^,]+),',bibliography_source))
    cited=set()
    for path in thesis.rglob('*.tex'):
        if 'build' in path.parts:continue
        for group in re.findall(r'\\cite\w*\s*(?:\[[^\]]*\])*\{([^}]+)\}',path.read_text()):
            cited.update(k.strip() for k in group.split(','))
    assert cited<=defined,'Missing bibliography entries: '+str(cited-defined)
    provenance=json.loads((ROOT/'results/artifact_provenance.json').read_text())
    for name,expected in provenance['inputs'].items():
        assert hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest()==expected,'Changed result input: '+name
    stats=json.loads((ROOT/'results/statistics.json').read_text())
    predictions=json.loads((ROOT/'results/predictions.json').read_text())
    for row in stats['retrieval']:
        source=[r for r in predictions if r['protocol']==row['protocol'] and r['method']==row['method']]
        assert len(source)==row['n']
        for metric in ['recall1','recall5','recall10']:
            assert abs(np.mean([r[metric] for r in source])-row[metric])<1e-12
        assert abs(np.mean([r['correct_verified'] for r in source])-row['verified'])<1e-12
    for name in ['comparison.mp4','day_query.mp4','night_query.mp4']:
        cap=cv2.VideoCapture(str(ROOT/'demo'/name));assert cap.isOpened(),name
        count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count>0,name+' contains no decodable frames'
    from PIL import Image
    gif=Image.open(ROOT/'demo/comparison.gif');assert gif.n_frames>1
    overfull=log.count('Overfull')
    report=dict(pdf_pages=len(pdf.pages),substantive_chapter_pages=body_pages,
                edition='full' if full_edition else 'short',required_pdf_page_range=list(page_range),
                evidence_scene_entries=48 if full_edition else 0,complete_oracle_register_rows=612 if full_edition else 0,
                intro_pdf_page=intro+1,bibliography_pdf_page=bibliography+1,
                latex_errors=0,undefined_references=0,undefined_citations=0,
                cited_entries=len(cited),bibliography_entries=len(defined),
                result_groups_checked=len(stats['retrieval']),demo_decoding='passed',
                overfull_box_warnings=overfull,
                pdf_sha256=hashlib.sha256((thesis/'thesis.pdf').read_bytes()).hexdigest())
    (ROOT/'results/final_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
