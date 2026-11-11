"""Compile delivered ZIP projects after extraction, without project-relative inputs."""
import json
import hashlib
import re
from pathlib import Path
import subprocess
import tempfile
import zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]

def main():
    rows=[]
    delivered=ROOT/'deliverables'
    manifest=json.loads((delivered/'manifest.json').read_text())
    for filename,record in manifest.items():
        path=delivered/filename
        assert path.stat().st_size==record['bytes'],filename+' size differs from manifest'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],filename+' hash differs from manifest'
    for local,filename in [('report/thesis.pdf','Stitch_IITR_Report.pdf'),
                           ('presentation/Stitch_IITR_Presentation.pdf','Stitch_IITR_Presentation.pdf'),
                           ('presentation/Stitch_IITR_Presentation.pptx','Stitch_IITR_Presentation.pptx')]:
        assert (ROOT/local).read_bytes()==(delivered/filename).read_bytes(),filename+' is stale'
    for name,mainfile in [('Stitch_IITR_Report','main.tex'),
                          ('Stitch_IITR_Presentation','Stitch_IITR_Presentation.tex')]:
        archive=ROOT/'deliverables'/(name+'_LaTeX.zip')
        cache=ROOT/'.cache/package-check';cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=name+'-',dir=cache) as temporary:
            destination=Path(temporary)
            with zipfile.ZipFile(archive) as z:
                source=ROOT/('report' if name=='Stitch_IITR_Report' else 'presentation')
                for member in z.namelist():
                    if member in ['README.txt','Makefile']:continue
                    assert z.read(member)==(source/member).read_bytes(),name+'/'+member+' is stale'
                z.extractall(destination)
            command=['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir=build',mainfile]
            process=subprocess.run(command,cwd=destination,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (ROOT/'results/logs').mkdir(parents=True,exist_ok=True)
            (ROOT/'results/logs'/(name+'_standalone.log')).write_bytes(process.stdout)
            assert process.returncode==0,name+' standalone build failed'
            final_log=(destination/'build'/(Path(mainfile).stem+'.log')).read_text(errors='replace')
            assert not re.search(r'^! |LaTeX Error|undefined|Overfull|Float too large',final_log,re.I|re.M),name+' final LaTeX log is not clean'
            pdf=destination/'build'/(Path(mainfile).stem+'.pdf')
            pages=len(PdfReader(pdf).pages)
            rows.append(dict(archive=str(archive.relative_to(ROOT)),command=command,
                             clean_temporary_directory=True,manifest_sha256_verified=True,
                             bundled_sources_match_current_project=True,
                             final_latex_log_clean=True,
                             returncode=process.returncode,pdf_pages=pages,
                             outside_project_data_required=False))
        print(name,'standalone compilation passed:',pages,'pages',flush=True)
    (ROOT/'results/standalone_packages_verification.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
