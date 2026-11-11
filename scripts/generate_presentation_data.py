"""Create slide tables and numbers from the already measured Stitch outputs."""
import hashlib
import json
from pathlib import Path
import shutil
import cv2

ROOT=Path(__file__).resolve().parents[1]

def main():
    directory=ROOT/'presentation';directory.mkdir(exist_ok=True)
    stats=json.loads((ROOT/'results/statistics.json').read_text())
    profiles=json.loads((ROOT/'results/profiles.json').read_text())
    export=json.loads((ROOT/'results/export.json').read_text())
    training=json.loads((ROOT/'results/training.json').read_text())
    names={'A_SIFT':'SIFT + FLANN','B_SP_LightGlue':'SuperPoint + LightGlue',
           'C_CosPlace':'CosPlace only','D_full_top3':'Full pipeline ($k=3$)'}
    def pct(value):return f'{100*value:.1f}'
    def table(command,header,rows,alignment):
        return '\\newcommand{\\'+command+'}{%\n\\begin{tabular}{'+alignment+'}\n\\toprule\n'+\
            ' & '.join(header)+' \\\\\n\\midrule\n'+\
            '\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)+\
            '\n\\bottomrule\n\\end{tabular}}\n'
    blocks=['% Generated solely from recorded results; do not edit numerical cells.\n']
    for protocol,command in [('tokyo_day_to_night','NightTable'),('tokyo_day_to_day','ViewTable')]:
        rows=[]
        for method in names:
            r=next(r for r in stats['retrieval'] if r['protocol']==protocol and r['method']==method)
            rows.append([names[method],pct(r['recall1']),pct(r['recall5']),
                         pct(r['verified']) if method!='C_CosPlace' else '--'])
        blocks.append(table(command,['Method','R@1','R@5','Verified'],rows,'lrrr'))
    rows=[]
    for split,title in [('hpatches_illumination','Illumination'),('hpatches_viewpoint','Viewpoint')]:
        results=[next(r for r in stats['matching'] if r['protocol']==split and r['method']==m) for m in ['sift','superpoint']]
        rows.append([title,*[pct(r['success']) for r in results]])
    blocks.append(table('MatchingTable',['HPatches split','SIFT','SuperPoint + LightGlue'],rows,'lrr'))
    blocks.append(table('RuntimeTable',['Method','Median ms','Peak RSS MB'],[
        [dict(classical='Classical',local='Exhaustive learned',retrieval='Retrieval only',full='Full pipeline')[r['method']],
         f'{r["median_ms"]:.1f}',f'{r["process_peak_rss_bytes"]/1e6:.1f}'] for r in profiles],'lrr'))
    blocks.append(table('QuantizationTable',['Descriptor','MB','Median ms','View R@1'],[
        [r['variant'].upper(),f'{r["bytes"]/1e6:.2f}',f'{r["median_inference_ms"]:.2f}',
         pct(next(q['recall1'] for q in r['retrieval'] if q['protocol']=='tokyo_day_to_day'))] for r in export['models']],'lrrr'))
    rows=[]
    for protocol,label in [('hpatches_viewpoint','HP viewpoint'),('tokyo_day_to_day','Tokyo view'),('tokyo_day_to_night','Tokyo night')]:
        cells=[]
        for variant,aug in [('pretrained_512',False),('projection_128',False),('projection_128',True)]:
            r=next(r for r in stats['training'] if r['protocol']==protocol and r['variant']==variant and r['augmentation']==aug)
            cells.append(pct(r['mean']))
        rows.append([label,*cells])
    blocks.append(table('AdaptationTable',['Held-out split','Pretrained','Head','Head + aug.'],rows,'lrrr'))
    open_rows=[]
    prediction=json.loads((ROOT/'results/predictions.json').read_text())
    for method in ['A_SIFT','B_SP_LightGlue','D_full_top3','D_no_geometry']:
        rows=[r for r in prediction if r['protocol']=='open_set' and r['method']==method]
        label=names.get(method,'No geometry')
        open_rows.append([label,sum(r['accepted'] for r in rows),len(rows)])
    blocks.append(table('OpenSetTable',['Variant','False accepts','Unknowns'],open_rows,'lrr'))
    (directory/'results_tables.tex').write_text('\n'.join(blocks))
    shutil.copy2(ROOT/'report/tables/numbers.tex',directory/'results_numbers.tex')
    capture=cv2.VideoCapture(str(ROOT/'demo/comparison.mp4'));ok,frame=capture.read();capture.release()
    if ok:cv2.imwrite(str(directory/'assets/demo_frame.png'),frame)
    provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'results'/n for n in
        ['statistics.json','profiles.json','export.json','training.json','predictions.json']]}
    (ROOT/'results/presentation_data.json').write_text(json.dumps(dict(inputs=provenance,
        experiment_outputs_modified=False,tables_generated=['NightTable','ViewTable','MatchingTable',
        'RuntimeTable','QuantizationTable','AdaptationTable','OpenSetTable']),indent=2))
    print('Generated seven presentation tables and linked measured numerical macros')

if __name__=='__main__':main()
