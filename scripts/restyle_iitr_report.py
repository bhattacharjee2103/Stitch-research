"""Switch the editable Stitch report to the supplied thesis's IITR layout."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    path=ROOT/'report/main.tex';text=path.read_text()
    if '\\input{frontmatter}' not in text:
        start=text.index('\\title{')
        end=text.index('\\tableofcontents')
        text=text[:start]+'\\input{metadata}\n\\title{\\StitchFullTitle}\n\\author{\\StudentName}\n\\date{\\ReportDate}\n\\begin{document}\n\\input{frontmatter}\n'+text[end:]
    text=text.replace('\\documentclass[a4paper,12pt,oneside]{report}',
                      '\\documentclass[a4paper,11pt,oneside]{report}')
    path.write_text(text)
    path=ROOT/'report/chapters/appendix_d_code.tex';text=path.read_text()
    for name in ['geometry','features','pipeline','rendering']:
        text=text.replace('../src/stitch/'+name+'.py','code/'+name+'.py')
    text=text.replace('../configs/default.yaml','code/default.yaml');path.write_text(text)
    path=ROOT/'report/chapters/10_ethics.tex';text=path.read_text()
    text=text.replace("Author, supervisor and institution fields remain placeholders because the experiment cannot establish those identities or institutional approval. The certificate is explicitly unsigned.",
        "Author and institution details follow the supplied IIT Roorkee reference documents, and the supervisor was explicitly selected by the author. Signatures and institutional acceptance are not manufactured; the certification remains unsigned for review.")
    path.write_text(text)
    print('Updated report main document, standalone code listings and metadata description')

if __name__=='__main__':main()
