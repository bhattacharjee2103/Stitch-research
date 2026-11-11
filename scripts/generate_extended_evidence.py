"""Generate complete scene-level evidence chapters from frozen measured outputs."""
import hashlib
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=['A_SIFT','B_SP_LightGlue','C_CosPlace','D_full_top3']
LABELS={'A_SIFT':'SIFT','B_SP_LightGlue':'SP+LG','C_CosPlace':'Global','D_full_top3':'Full'}

def read(name):return json.loads((ROOT/'results'/name).read_text())
def esc(text):return str(text).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def yes(value):return 'yes' if value else 'no'
def number(value):return '--' if value is None else f'{value:.2f}'
def count(n,singular,plural):return f'{n} '+(singular if n==1 else plural)
def table(headers,rows,caption,label,alignment):
    return ('\\begin{table}[H]\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n'
            '\\begin{tabular}{'+alignment+'}\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'+
            '\n'.join(' & '.join(str(v) for v in row)+r' \\' for row in rows)+
            '\n\\bottomrule\\end{tabular}\\end{table}\n')

def sheet(paths,titles,target,columns):
    rows=(len(paths)+columns-1)//columns
    fig,axes=plt.subplots(rows,columns,figsize=(10,3.1*rows),squeeze=False)
    for ax,path,title in zip(axes.flat,paths,titles):
        image=cv2.imread(str(ROOT/path))
        if image is None:raise FileNotFoundError(path)
        h,w=image.shape[:2];scale=min(1,720/max(h,w))
        image=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
        ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));ax.set_title(title,fontsize=10);ax.axis('off')
    for ax in list(axes.flat)[len(paths):]:ax.axis('off')
    fig.tight_layout();fig.savefig(target,bbox_inches='tight',dpi=120);plt.close(fig)

def write_manifest():
    inputs={name:hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest() for name in
            ['pairs.json','derived_predictions.json','data_manifest.json','training.json']}
    extra_inputs={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                  ['data/benchmark.json','data/tokyo/selected.json']}
    files=[ROOT/'report/chapters'/name for name in
           ['12_hpatches_evidence.tex','13_tokyo_evidence.tex','appendix_e_pair_register.tex']]
    figures=list((ROOT/'report/figures/evidence').glob('*.pdf'));files+=figures
    outputs={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    report=dict(inputs=inputs,extra_inputs=extra_inputs,generated_outputs=outputs,
                hpatches_sequences=len(read('data_manifest.json')['hpatches']['sequences']),
                tokyo_sites=len({r['place_id'] for r in json.loads((ROOT/'data/tokyo/selected.json').read_text())}),
                figures_generated=len(figures),oracle_pair_rows=len(read('pairs.json')),experiment_inference_rerun=False)
    (ROOT/'results/extended_evidence.json').write_text(json.dumps(report,indent=2))
    return report

def main():
    output=ROOT/'report/chapters';figures=ROOT/'report/figures/evidence';figures.mkdir(parents=True,exist_ok=True)
    pairs=read('pairs.json');predictions=read('derived_predictions.json')
    data=read('data_manifest.json');training=read('training.json')
    selected=json.loads((ROOT/'data/tokyo/selected.json').read_text())
    hp=[r'\chapter{Complete HPatches Sequence Evidence}\label{ch:hp-atlas}',
        r'\section{How to read this evidence atlas}',
        'This chapter covers every selected sequence rather than only a favourable qualitative example. '
        'The six photographs in each panel are the actual reference and five evaluated targets. '
        'The first table reports saved oracle-pair estimates, while the second reports end-to-end retrieval and accepted decisions. '
        'A correct place can still have an inaccurate homography, and an accurate oracle pair can be lost during candidate search. '
        'All counts retain rejected queries in their denominators. These are additional views of the same experiments, not independent new trials. '
        r'HPatches supplies the source homographies \citep{balntas2017}; the selected protocol is defined in \cref{ch:datasets}.',
        'In the pair tables, I is the saved RANSAC inlier count, E is mean four-corner error in working pixels, '
        'and H5 requires both accepted geometry and E at most five pixels. A dash denotes an unavailable finite error, '
        'not a zero error. End-to-end H5 additionally requires a correctly selected place. '
        'The SP+LG label abbreviates SuperPoint with LightGlue.']
    for index,sequence in enumerate(data['hpatches']['sequences']):
        protocol='hpatches_illumination' if sequence.startswith('i_') else 'hpatches_viewpoint'
        rows=[r for r in pairs if r['place_id']==sequence and r['protocol']==protocol]
        by={(r['id'],r['method']):r for r in rows}
        queries=sorted({r['id'] for r in rows},key=lambda s:int(s.rsplit('_',1)[-1]))
        target=figures/f'hp_{index:02d}.pdf'
        sheet([f'data/hpatches-sequences-release/{sequence}/{n}.ppm' for n in range(1,7)],
              ['Reference 1']+[f'Target {n}' for n in range(2,7)],target,3)
        hp+=['\\section{Sequence '+esc(sequence)+'}',
             ('This sequence belongs to the '+('illumination' if sequence.startswith('i_') else 'viewpoint')+
              ' subset and contributes five reference-to-target pairs. Its identity is '+
              ('reserved for the projection-head training split' if sequence in training['train_places'] else 'held out from projection-head training')+
              '; learned components in the main A--D comparison retain their original pretrained weights in either case.'),
             '\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth]{evidence/'+target.name+'}\n'
             '\\caption{Actual reference and all evaluated targets for '+esc(sequence)+'.}\\label{fig:hp-scene-'+str(index)+'}\\end{figure}']
        records=[]
        for identifier in queries:
            a=by[identifier,'sift'];b=by[identifier,'superpoint']
            records.append([identifier.rsplit('_',1)[-1],a['inliers'],number(a['corner_error_px']),yes(a['homography_success']),
                            b['inliers'],number(b['corner_error_px']),yes(b['homography_success'])])
        hp.append(table(['Target','SIFT I','SIFT E','SIFT H5','SP+LG I','SP+LG E','SP+LG H5'],records,
                        'Saved oracle-pair support and corner error for '+esc(sequence)+'.','tab:hp-pairs-'+str(index),'lrrrrrr'))
        a=[by[q,'sift'] for q in queries];b=[by[q,'superpoint'] for q in queries]
        wins=sum(not x['homography_success'] and y['homography_success'] for x,y in zip(a,b))
        losses=sum(x['homography_success'] and not y['homography_success'] for x,y in zip(a,b))
        hp.append('SIFT obtains '+str(sum(r['homography_success'] for r in a))+' of five successful oracle homographies; '
                  'SP+LG obtains '+str(sum(r['homography_success'] for r in b))+' of five. The paired comparison contains '+
                  count(wins,'learned-only success','learned-only successes')+' and '+
                  count(losses,'SIFT-only success','SIFT-only successes')+'. Median inlier counts are '+
                  f"{np.median([r['inliers'] for r in a]):.1f}"+' and '+f"{np.median([r['inliers'] for r in b]):.1f}"+
                  ', respectively. These support counts are computed from different tentative correspondence sets, so their magnitudes alone do not establish a more accurate mapping.')
        end=[]
        for method in ['A_SIFT','B_SP_LightGlue','D_full_top3']:
            rs=[r for r in predictions if r['protocol']==protocol and r['true_place']==sequence and r['method']==method]
            end.append([LABELS[method],str(sum(r['recall1'] for r in rs))+'/5',
                        str(sum(r['correct_verified'] for r in rs))+'/5',str(sum(r['homography_success'] for r in rs))+'/5'])
        hp.append(table(['Method','Rank-1 correct','Correctly verified','End-to-end H5'],end,
                        'Search and anchoring outcomes for the same five targets.','tab:hp-end-'+str(index),'lrrr'))
        full=[r for r in predictions if r['protocol']==protocol and r['true_place']==sequence and r['method']=='D_full_top3']
        recovered=sum(r['correct_verified'] for r in full)
        hp.append('The full hierarchy correctly verifies '+str(recovered)+' of these five queries after searching the mixed 24-reference gallery. '
                  'Its end-to-end column includes candidate selection and homography accuracy; the oracle table supplies the true reference directly. '
                  'Any difference between those columns is therefore a search or decision effect rather than an additional illumination observation. '
                  'The sequence remains one physical source of correlated targets, so five outcomes should not be treated as five independently sampled places.')
    (output/'12_hpatches_evidence.tex').write_text('\n\n'.join(hp)+'\n')
    tokyo=[r'\chapter{Complete Tokyo Site Evidence}\label{ch:tokyo-atlas}',
           r'\section{Site-level observations and decision states}',
           'Each section contains the four retained photographs for one selected physical site and every A--D decision for its four cross-time protocols. '
           'The panels expose reference coverage directly. Front/day, left/day, front/night and front/sunset metadata define the views; '
           'their identity labels do not supply an independently measured planar transform or metric camera pose. '
           r'The source imagery is from Tokyo 24/7 \citep{torii2015}, but these tables belong to the custom query-to-query subset.',
           'R is the rank of the correct identity in the saved complete ranking. Decision states are correct, wrong, rejected and ranking only. '
           'A wrong state means geometric acceptance of another place; it does not mean every correspondence is known to be false. '
           'Global retrieval has no acceptance or inlier-count claim. Protocol labels D--D, D--N, N--D and D--S mean day to changed-direction day, day to night, night to day and day to sunset.']
    protocols=[('tokyo_day_to_day','D--D'),('tokyo_day_to_night','D--N'),('tokyo_night_to_day','N--D'),('tokyo_day_to_sunset','D--S')]
    for index,site in enumerate(sorted({r['place_id'] for r in selected})):
        records=[r for r in selected if r['place_id']==site]
        target=figures/f'tokyo_{index:02d}.pdf'
        sheet(['data/tokyo/'+r['image'] for r in records],
              [dict(F='Front',L='Left')[r['view']]+' / '+dict(D='day',N='night',S='sunset')[r['condition']] for r in records],target,2)
        tokyo+=['\\section{Site '+esc(site)+'}',
                'This site contributes four real photographs and four labelled retrieval queries. '
                'Each protocol uses one of the photographs as its query and the condition-specific 24-site gallery defined in the frozen benchmark. '
                'Reusing a site across protocols produces paired condition observations, not four new geographic locations.',
                '\\begin{figure}[H]\\centering\\includegraphics[width=.92\\textwidth]{evidence/'+target.name+'}\n'
                '\\caption{All retained photographs for '+esc(site)+'.}\\label{fig:tokyo-scene-'+str(index)+'}\\end{figure}']
        rs=[r for r in predictions if r['protocol'].startswith('tokyo_') and r['true_place']==site and r['method'] in METHODS]
        rows=[]
        for protocol,short in protocols:
            for method in METHODS:
                row=next(r for r in rs if r['protocol']==protocol and r['method']==method)
                state='ranking only' if method=='C_CosPlace' else ('correct' if row['correct_verified'] else ('wrong' if row['accepted'] else 'rejected'))
                rank=row['ranked_places'].index(site)+1
                rows.append([short,LABELS[method],rank,state,'--' if method=='C_CosPlace' else row['inliers']])
        tokyo.append(table(['Protocol','Method','R','Decision','Selected inliers'],rows,
                           'All main-method decisions for '+esc(site)+'.','tab:tokyo-site-'+str(index),'llrlr'))
        full=[r for r in rs if r['method']=='D_full_top3'];a=[r for r in rs if r['method']=='A_SIFT'];b=[r for r in rs if r['method']=='B_SP_LightGlue']
        tokyo.append('Across these four protocols, SIFT correctly verifies '+str(sum(r['correct_verified'] for r in a))+', exhaustive SP+LG '+
                     str(sum(r['correct_verified'] for r in b))+', and the full hierarchy '+str(sum(r['correct_verified'] for r in full))+
                     ' of the four queries. The full hierarchy makes '+
                     count(sum(r['accepted'] and not r['correct_verified'] for r in full),'wrong-place acceptance','wrong-place acceptances')+
                     ' and '+count(sum(not r['accepted'] for r in full),'rejection','rejections')+'. '
                     'These counts separate a missing anchor from an incorrectly accepted identity; the latter is the more direct location-association failure.')
        daytime=next(r for r in full if r['protocol']=='tokyo_day_to_day')
        night=next(r for r in full if r['protocol']=='tokyo_day_to_night')
        tokyo.append('For this site, the full-pipeline changed-direction daytime query is '+
                     ('correctly verified' if daytime['correct_verified'] else ('accepted at the wrong place' if daytime['accepted'] else 'rejected'))+
                     ', while its front-facing day-to-night query is '+
                     ('correctly verified' if night['correct_verified'] else ('accepted at the wrong place' if night['accepted'] else 'rejected'))+
                     '. This paired observation should be read with the photographs and the rank column. '
                     'It does not identify the causal effect of illumination alone, because overlap and view direction also change between the protocols. '
                     'No corner-error or metric-pose value is inferred from a correct place label.')
    (output/'13_tokyo_evidence.tex').write_text('\n\n'.join(tokyo)+'\n')
    # Query-level evidence complements the scene atlas without repeating source code.
    register=[r'\chapter{Complete Oracle-Pair Result Register}\label{app:pair-register}',
              'This register retains every positive pair for both local methods, including rejections. It is generated from '
              r'\code{results/pairs.json}; the scene atlases provide the corresponding images. '
              'Error is mean corner error in working pixels. The Tokyo rows have no independent homography ground truth, so their error cells are unavailable. '
              'The register is an audit view of the existing experiment and adds no new test observations.']
    for protocol in dict.fromkeys(r['protocol'] for r in pairs):
        register+=['\\section{'+esc(protocol)+'}',r'\begingroup\scriptsize\setlength{\tabcolsep}{3pt}',
                   r'\begin{longtable}{llrrrrr}',
                   '\\caption{All oracle-pair results for '+esc(protocol)+'.}\\\\\n'+r'\toprule',
                   r'Query & Method & Matches & Inliers & Ratio & Error & Accept\\\midrule\endfirsthead',
                   r'\toprule Query & Method & Matches & Inliers & Ratio & Error & Accept\\\midrule\endhead',
                   r'\bottomrule\endfoot']
        for row in [r for r in pairs if r['protocol']==protocol]:
            identifier=row['id'].replace('tokyo_','t').replace('synthetic_','')
            register.append(' & '.join([esc(identifier),'SIFT' if row['method']=='sift' else 'SP+LG',str(row['matches']),
                str(row['inliers']),f"{row['inlier_ratio']:.2f}",number(row['corner_error_px']),yes(row['accepted'])])+r' \\')
        register += [r'\end{longtable}\endgroup']
    (output/'appendix_e_pair_register.tex').write_text('\n'.join(register)+'\n')
    report=write_manifest()
    print(json.dumps({k:v for k,v in report.items() if k not in ['inputs','extra_inputs','generated_outputs']},indent=2))

if __name__=='__main__':main()
