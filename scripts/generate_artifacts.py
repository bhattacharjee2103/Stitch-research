"""Derive every empirical table, macro and plot from preserved measured outputs."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
from stitch.geometry import corner_error
from stitch.io import read_image

ROOT=Path(__file__).resolve().parents[1]
TABLE=ROOT/'report/tables';FIG=ROOT/'figures'
METHODS=['A_SIFT','B_SP_LightGlue','C_CosPlace','D_full_top3']
LABELS={'A_SIFT':'A: SIFT','B_SP_LightGlue':'B: SP+LG','C_CosPlace':'C: CosPlace',
        'D_full_top3':'D: Full ($k=3$)','D_full_top1':'Full ($k=1$)',
        'D_full_top5':'Full ($k=5$)','D_no_geometry':'No geometry'}
COLORS=['#777777','#2972a3','#df9a31','#399471']

def read(name):return json.loads((ROOT/'results'/name).read_text())
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def pct(x):return f'{100*x:.1f}'
def pretty(s):
    labels={'hpatches_illumination':'HP illum.','hpatches_viewpoint':'HP view.',
            'tokyo_day_to_day':'Tokyo D--D','tokyo_day_to_night':'Tokyo D--N',
            'tokyo_night_to_day':'Tokyo N--D','tokyo_day_to_sunset':'Tokyo D--S',
            'synthetic_day':'Synth. day','synthetic_night':'Synth. night',
            'synthetic_night_severe':'Synth. severe','synthetic_warm_night':'Synth. warm',
            'synthetic_fog':'Synth. fog','open_set':'Open set'}
    return labels.get(s,esc(s))
def fraction(rows,key):return np.mean([r[key] for r in rows]).item() if rows else 0.

def table(name,header,rows,caption,label,alignment=None,long=False):
    alignment=alignment or ('l'+'r'*(len(header)-1))
    body=' & '.join(header)+r' \\'+'\n'+r'\midrule'+'\n'
    body+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    if long:
        heading=' & '.join(header)+r' \\'+'\n'+r'\midrule'+'\n'
        entries='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
        text=(r'\begingroup\small\setlength{\tabcolsep}{4pt}'+'\n'+r'\begin{longtable}{'+alignment+'}\n'+
              r'\caption{'+caption+r'}\label{'+label+r'}\\'+'\n'+r'\toprule'+'\n'+heading+
              r'\endfirsthead'+'\n'+r'\multicolumn{'+str(len(header))+r'}{l}{\tablename\ \thetable\ (continued)}\\'+'\n'+
              r'\toprule'+'\n'+heading+r'\endhead'+'\n'+r'\midrule'+
              '\n'+r'\multicolumn{'+str(len(header))+r'}{r}{Continued on next page}\\'+'\n'+r'\endfoot'+'\n'+
              r'\bottomrule\endlastfoot'+'\n'+entries+'\n'+r'\end{longtable}\endgroup'+'\n')
    else:
        text=(r'\begin{table}[htbp]\centering\small'+'\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+
              r'\begin{tabular}{'+alignment+'}\n'+r'\toprule'+'\n'+body+'\n'+r'\bottomrule'+'\n'+
              r'\end{tabular}\end{table}'+'\n')
    (TABLE/name).write_text(text)

def save(name):
    plt.tight_layout();plt.savefig(FIG/name,bbox_inches='tight');plt.close()

def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return center-half,center+half

def cluster_ci(rows,key,clusters,seed=17):
    rng=np.random.default_rng(seed);names=sorted(set(clusters))
    groups={c:[r[key] for r,g in zip(rows,clusters) if g==c] for c in names}
    samples=[np.mean([v for c in rng.choice(names,len(names),replace=True) for v in groups[c]]) for _ in range(2000)]
    return np.percentile(samples,[2.5,97.5]).tolist()

def main():
    TABLE.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    pairs=read('pairs.json');pred=read('predictions.json');benchmark=json.loads((ROOT/'data/benchmark.json').read_text())
    max_side=read('run_config.json')['config']['max_side']
    lookup={(p['name'],q['id']):q for p in benchmark['protocols'] for q in p['queries']}
    # Compute verified homography accuracy from saved transforms; no inference is rerun here.
    for row in pred:
        query=lookup[row['protocol'],row['id']]
        has_gt=query.get('H_original') is not None
        success=False;err=None
        if has_gt and row['correct_verified'] and row['H'] is not None:
            ref,Sr=read_image(ROOT/query['ref'],max_side);_,Sq=read_image(ROOT/query['query'],max_side)
            Hgt=Sq@np.array(query['H_original'])@np.linalg.inv(Sr)
            value=corner_error(np.array(row['H']),Hgt,ref.shape)
            success=value<=5;err=value if math.isfinite(value) else None
        row.update(has_homography_gt=has_gt,homography_success=bool(success),corner_error_px=err)
    (ROOT/'results/derived_predictions.json').write_text(json.dumps(pred,indent=2))
    data=read('data_manifest.json');prep=read('preparation.json');orig=read('original_baselines.json')
    exports=read('export.json');training=read('training.json');models=read('models.json')
    profiles=read('profiles.json') if (ROOT/'results/profiles.json').exists() else []
    stats={'retrieval':[],'matching':[],'confidence':[]}
    names=[p['name'] for p in benchmark['protocols']]
    for protocol in names:
        for method in sorted({r['method'] for r in pred}):
            rows=[r for r in pred if r['protocol']==protocol and r['method']==method]
            if not rows:continue
            stats['retrieval'].append(dict(protocol=protocol,method=method,n=len(rows),
                recall1=fraction(rows,'recall1'),recall5=fraction(rows,'recall5'),recall10=fraction(rows,'recall10'),
                verified=fraction(rows,'correct_verified'),accepted=fraction(rows,'accepted'),
                homography_success=fraction(rows,'homography_success')))
        for method in ['sift','superpoint']:
            rows=[r for r in pairs if r['protocol']==protocol and r['method']==method]
            if not rows:continue
            valid=[r['corner_error_px'] for r in rows if r['corner_error_px'] is not None]
            stats['matching'].append(dict(protocol=protocol,method=method,n=len(rows),
                median_inliers=float(np.median([r['inliers'] for r in rows])),
                mean_ratio=float(np.mean([r['inlier_ratio'] for r in rows])),accepted=fraction(rows,'accepted'),
                success=fraction(rows,'homography_success'),has_gt=rows[0]['has_homography_gt'],
                median_valid_corner_error=float(np.median(valid)) if valid else None))
            if protocol.startswith('hpatches'):
                lo,hi=cluster_ci(rows,'homography_success',[r['place_id'] for r in rows])
                stats['confidence'].append(dict(protocol=protocol,method=method,metric='H success',low=lo,high=hi))
    def lookup_stat(protocol,method):return next(r for r in stats['retrieval'] if r['protocol']==protocol and r['method']==method)
    macros={'HpImages':data['hpatches']['image_count'],'HpPairs':data['hpatches']['pair_count'],
            'TokyoSites':data['tokyo']['sites'],'TokyoImages':data['tokyo']['image_count'],
            'OraclePairs':len(pairs)//2,'PredictionRows':len(pred),'SyntheticQueries':len(prep['generated_files']),
            'DayNightSiftRecall':pct(lookup_stat('tokyo_day_to_night','A_SIFT')['recall1']),
            'DayNightLearnedRecall':pct(lookup_stat('tokyo_day_to_night','B_SP_LightGlue')['recall1']),
            'DayNightGlobalRecall':pct(lookup_stat('tokyo_day_to_night','C_CosPlace')['recall1']),
            'DayNightFullVerified':pct(lookup_stat('tokyo_day_to_night','D_full_top3')['verified']),
            'DayViewGlobalRecall':pct(lookup_stat('tokyo_day_to_day','C_CosPlace')['recall1']),
            'DayViewFullVerified':pct(lookup_stat('tokyo_day_to_day','D_full_top3')['verified']),
            'FpModelMB':f'{exports["models"][0]["bytes"]/1e6:.2f}',
            'IntModelMB':f'{exports["models"][1]["bytes"]/1e6:.2f}',
            'FpLatency':f'{exports["models"][0]["median_inference_ms"]:.2f}',
            'IntLatency':f'{exports["models"][1]["median_inference_ms"]:.2f}'}
    for protocol,prefix in [('hpatches_illumination','HpIllum'),('hpatches_viewpoint','HpView')]:
        for method,suffix in [('sift','Sift'),('superpoint','Learned')]:
            r=next(r for r in stats['matching'] if r['protocol']==protocol and r['method']==method)
            macros[prefix+suffix]=pct(r['success'])
    for protocol,prefix in [('tokyo_day_to_night','Night'),('tokyo_day_to_day','View')]:
        for method,suffix in [('A_SIFT','Sift'),('B_SP_LightGlue','Learned'),('D_full_top3','Full')]:
            macros[prefix+suffix+'Verified']=pct(lookup_stat(protocol,method)['verified'])
    for r in profiles:
        macros['Latency'+r['method'].capitalize()]=f'{r["median_ms"]:.1f}'
    (TABLE/'numbers.tex').write_text('% Generated by scripts/generate_artifacts.py; do not edit.\n'+
        '\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n')
    table('originals.tex',['Script','Frames','H estimates','Warps','ms/frame'],
        [[esc(r['script']),r['decoded_frames'],r['homography_nonnull'],r['warp_calls'],f'{r["ms_per_decoded_frame"]:.2f}'] for r in orig],
        'Measured upstream-script execution on the same bounded file input. Timing includes initialisation and file I/O.','tab:originals')
    table('datasets.tex',['Collection','Images','Pairs / queries','Gallery'],[
        ['HPatches subset',data['hpatches']['image_count'],data['hpatches']['pair_count'],24],
        ['Tokyo cross-time subset',data['tokyo']['image_count'],96,24],
        ['Synthetic transformations',len(prep['generated_files']),len(prep['generated_files']),26]],
        'Downloaded and generated evaluation collections. Pair counts refer to this artifact, not full public datasets.','tab:datasets')
    table('protocols.tex',['Protocol','Gallery','Queries'],[[esc(p['name']),len(p['gallery']),len(p['queries'])] for p in benchmark['protocols']],
          'Frozen evaluation protocols. Reference-to-query direction is used in condition names.','tab:protocols',long=True)
    matching=[]
    for r in stats['matching']:
        matching.append([pretty(r['protocol']), 'SIFT' if r['method']=='sift' else 'SP+LG',r['n'],
                         f'{r["median_inliers"]:.1f}',pct(r['mean_ratio']),pct(r['accepted']),pct(r['success']) if r['has_gt'] else r'\NA'])
    table('matching.tex',['Protocol','Method','$n$','Inliers','Ratio (\%)','Accept (\%)','$H_5$ (\%)'],matching,
          'Oracle-pair matching under shared geometry settings. Inliers are medians, ratios are means over all pairs; $H_5$ requires accepted geometry and mean corner error at most five resized pixels. Tokyo has no homography ground truth.','tab:matching',long=True)
    for group,prefix in [('tokyo','tokyo_'),('hpatches','hpatches_'),('synthetic','synthetic_')]:
        rows=[]
        for protocol in [n for n in names if n.startswith(prefix)]:
            for method in METHODS:
                r=lookup_stat(protocol,method)
                rows.append([esc(protocol.replace(prefix,'')),LABELS[method],pct(r['recall1']),pct(r['recall5']),pct(r['recall10']),
                             pct(r['verified']) if method!='C_CosPlace' else r'\NA'])
        table('retrieval_'+group+'.tex',['Split','Method','R@1','R@5','R@10','Verified'],rows,
              'Retrieval and correctly verified place rates (percent) on '+dict(tokyo='Tokyo',hpatches='HPatches',synthetic='synthetic')[group]+' protocols. Global-only retrieval does not produce an anchor. Full-pipeline recall uses its reranked complete list, with the unverified tail retained.','tab:retrieval-'+group,long=True)
    rows=[]
    for protocol in ['tokyo_day_to_day','tokyo_day_to_night','synthetic_night_severe']:
        for method in ['B_SP_LightGlue','D_full_top1','D_full_top3','D_full_top5','D_no_geometry']:
            r=lookup_stat(protocol,method)
            rows.append([pretty(protocol),LABELS[method],pct(r['recall1']),pct(r['verified']),pct(r['accepted'])])
    table('ablations.tex',['Protocol','Variant','R@1','Correct (\%)','Accept (\%)'],rows,
          'Stage ablations on identical queries. Exhaustive learned matching removes global retrieval; the no-geometry variant accepts the highest global result without an estimated transform.','tab:ablations',long=True)
    rows=[]
    for protocol in ['tokyo_day_to_day','tokyo_day_to_night','tokyo_night_to_day']:
        for method in ['A_SIFT','B_SP_LightGlue','D_full_top3']:
            selected=[r for r in pred if r['protocol']==protocol and r['method']==method]
            k=sum(r['correct_verified'] for r in selected);lo,hi=wilson(k,len(selected))
            rows.append([pretty(protocol),LABELS[method],f'{k}/{len(selected)}',f'{100*lo:.1f}--{100*hi:.1f}'])
    table('confidence.tex',['Protocol','Method','Correct','95\% Wilson interval (\%)'],rows,
          'Finite-sample uncertainty for correctly verified places. These intervals describe this selected set under an independent-site approximation; they do not compensate for geographic selection bias.','tab:confidence',long=True)
    rows=[]
    geometry_rows=[]
    for protocol in [n for n in names if n.startswith(('hpatches_','synthetic_'))]:
        for method in ['A_SIFT','B_SP_LightGlue','D_full_top3']:
            r=lookup_stat(protocol,method)
            geometry_rows.append([pretty(protocol),LABELS[method],r['n'],pct(r['verified']),pct(r['homography_success'])])
    table('pipeline-geometry.tex',['Protocol','Method','$n$','Correct place (\%)','$H_5$ (\%)'],geometry_rows,
          'End-to-end planar success: the selected place must be correct, its geometry accepted, and mean corner error at most five working pixels. This differs from the oracle-pair matching table.','tab:pipeline-geometry',long=True)
    table('open-set.tex',['Variant','Unknown queries','False acceptances','Rate (\%)'],[
        [LABELS[m],len(rs),sum(r['accepted'] for r in rs),pct(fraction(rs,'accepted'))]
        for m in ['A_SIFT','B_SP_LightGlue','D_full_top1','D_full_top3','D_full_top5','D_no_geometry']
        for rs in [[r for r in pred if r['protocol']=='open_set' and r['method']==m]]],
        'Measured open-set diagnostic against a disjoint-collection gallery. Global-only ranking is omitted because ranking alone makes no acceptance decision.','tab:open-set')
    trainstats=[]
    for protocol in sorted({r['protocol'] for r in training['results']}):
        for variant,aug in [('pretrained_512',False),('projection_128',False),('projection_128',True)]:
            rs=[r for r in training['results'] if r['protocol']==protocol and r['variant']==variant and r['augmentation']==aug]
            values=np.array([r['recall1'] for r in rs]);mean=values.mean();sd=values.std(ddof=1)
            half=float(student_t.ppf(.975,2)*sd/np.sqrt(3))
            name='Pretrained 512' if variant=='pretrained_512' else ('Head + night aug.' if aug else 'Head, no night aug.')
            rows.append([pretty(protocol),name,pct(mean),f'{100*sd:.1f}',f'{100*max(0,mean-half):.1f}--{100*min(1,mean+half):.1f}'])
            trainstats.append(dict(protocol=protocol,variant=variant,augmentation=aug,mean=float(mean),sd=float(sd),ci95_half=half))
    stats['training']=trainstats
    table('training.tex',['Protocol','Descriptor','Mean R@1','SD','95\% $t$ interval'],rows,
          'Short frozen-backbone projection training, three fixed seeds. Rates are percentages. HPatches rows use only held-out place identities. Intervals are clipped to the feasible percentage range and are unstable with only three seeds.','tab:training',long=True)
    table('models.tex',['Model','Parameters','Checkpoint MB'],[
        [name,models['parameters'][name],f'{models["files"][i]["bytes"]/1e6:.2f}'] for name,i in [('CosPlace',0),('SuperPoint',1),('LightGlue',2)]],
        'Measured parameter counts and downloaded checkpoint sizes. Checkpoints include serialisation metadata.','tab:models')
    table('exports.tex',['ONNX variant','MB','Median ms','95th pct. ms','Max abs. error'],[
        [r['variant'],f'{r["bytes"]/1e6:.2f}',f'{r["median_inference_ms"]:.2f}',f'{r["p95_inference_ms"]:.2f}',f'{r["max_abs_parity_error"]:.2g}'] for r in exports['models']],
        'Executed global-descriptor export and quantisation on the host CPU. Parity is measured against native PyTorch across downloaded Tokyo images. Timing excludes preprocessing and is not a phone measurement.','tab:exports')
    table('quant-retrieval.tex',['Protocol','Float R@1','INT8 R@1','Float R@5','INT8 R@5'],[
        [esc(a['protocol']),pct(a['recall1']),pct(b['recall1']),pct(a['recall5']),pct(b['recall5'])]
        for a,b in zip(exports['models'][0]['retrieval'],exports['models'][1]['retrieval'])],
        'Retrieval accuracy (percent) after actual ONNX inference. Calibration uses training places, not Tokyo queries.','tab:quant-retrieval',long=True)
    table('profiles.tex',['Pipeline','Median ms','95th pct. ms','Peak RSS MB'],[
        [r['method'],f'{r["median_ms"]:.1f}',f'{r["p95_ms"]:.1f}',f'{r["process_peak_rss_bytes"]/1e6:.1f}'] for r in profiles],
        'Isolated warm CPU query and rendering measurements with cached references. RSS is the complete Python process peak, including imports, models, gallery and allocator state.','tab:profiles')
    table('pose-unavailable.tex',['Requested pose criterion','Measured rate','Reason'],[
        [r'$(0.25\,\mathrm{m},2^\circ)$',r'\NA','No calibrated query pose/map'],
        [r'$(0.5\,\mathrm{m},5^\circ)$',r'\NA','No calibrated query pose/map'],
        [r'$(5\,\mathrm{m},10^\circ)$',r'\NA','No calibrated query pose/map']],
        'Metric localisation results are unavailable, rather than zero. Neither GPS site labels nor pixel homographies establish these errors.','tab:poses')
    # Every plotted bar is computed from the same rows used in the tables.
    fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    for ax,protocol in zip(axes,['hpatches_illumination','hpatches_viewpoint']):
        rs=[r for r in stats['matching'] if r['protocol']==protocol]
        ax.bar(['SIFT','SP + LightGlue'],[100*r['success'] for r in rs],color=COLORS[:2])
        ax.set_ylim(0,105);ax.set_ylabel('Homography success (%)');ax.set_title(protocol.replace('hpatches_','').title())
    save('hpatches_success.pdf')
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    for ax,protocol in zip(axes,['hpatches_illumination','hpatches_viewpoint']):
        for method,color in zip(['sift','superpoint'],COLORS[:2]):
            rs=[r for r in pairs if r['protocol']==protocol and r['method']==method]
            errors=np.array([r['corner_error_px'] if r['corner_error_px'] is not None and r['accepted'] else np.inf for r in rs])
            x=np.linspace(0,20,201);ax.plot(x,[100*np.mean(errors<=t) for t in x],label=method,color=color)
        ax.set_title(protocol.replace('hpatches_',''));ax.set_xlabel('Mean corner-error threshold (pixels)');ax.set_ylabel('Queries within threshold (%)');ax.legend()
    save('corner_ecdf.pdf')
    fig,ax=plt.subplots(figsize=(9,3.8));splits=['day_to_day','day_to_night','night_to_day','day_to_sunset'];x=np.arange(4)
    for j,m in enumerate(METHODS):
        ax.bar(x+(j-1.5)*.19,[100*lookup_stat('tokyo_'+s,m)['recall1'] for s in splits],width=.19,label=LABELS[m],color=COLORS[j])
    ax.set_xticks(x, [s.replace('_',' ') for s in splits]);ax.set_ylim(0,110);ax.set_ylabel('Recall@1 (%)');ax.legend(ncol=2,loc='lower right');save('tokyo_retrieval.pdf')
    fig,ax=plt.subplots(figsize=(8,3.6));variants=['B_SP_LightGlue','D_full_top1','D_full_top3','D_full_top5','D_no_geometry'];x=np.arange(len(variants))
    for j,protocol in enumerate(['tokyo_day_to_day','tokyo_day_to_night']):
        ax.bar(x+(j-.5)*.35,[100*lookup_stat(protocol,m)['verified'] for m in variants],width=.35,label=protocol.replace('tokyo_','').replace('_',' '))
    ax.set_xticks(x,[LABELS[m] for m in variants]);ax.set_ylabel('Correctly accepted place (%)');ax.set_ylim(0,110);ax.legend();save('stage_ablation.pdf')
    fig,ax=plt.subplots(figsize=(8,3.5))
    for augment,color in zip([False,True],['#2972a3','#399471']):
        vals=np.array([[r['loss'] for r in training['losses'] if r['seed']==seed and r['augmentation']==augment] for seed in [17,23,42]])
        x=np.arange(vals.shape[1]);ax.plot(x,vals.mean(0),label='Night augmentation' if augment else 'No night augmentation',color=color)
        ax.fill_between(x,vals.min(0),vals.max(0),alpha=.15,color=color)
    ax.set_xlabel('Optimizer step');ax.set_ylabel('Triplet loss');ax.legend();save('training_loss.pdf')
    fig,axes=plt.subplots(1,2,figsize=(8,3.3));rs=exports['models']
    for ax,key,factor,title in [(axes[0],'bytes',1e6,'Model size (MB)'),(axes[1],'median_inference_ms',1,'Median descriptor inference (ms)')]:
        ax.bar([r['variant'] for r in rs],[r[key]/factor for r in rs],color=['#2972a3','#399471']);ax.set_ylabel(title)
    save('quantization.pdf')
    if profiles:
        fig,axes=plt.subplots(1,2,figsize=(9,3.5))
        for ax,key,factor,label in [(axes[0],'median_ms',1,'Median query and render (ms)'),(axes[1],'process_peak_rss_bytes',1e6,'Peak process RSS (MB)')]:
            ax.bar([r['method'] for r in profiles],[r[key]/factor for r in profiles],color=COLORS);ax.set_ylabel(label);ax.tick_params(axis='x',rotation=15)
        save('efficiency.pdf')
    # Contact sheets are actual evaluated inputs, with no image synthesis at plotting time.
    tokyo=json.loads((ROOT/'data/tokyo/selected.json').read_text());site_ids=sorted({r['place_id'] for r in tokyo})[:3]
    fig,axes=plt.subplots(3,4,figsize=(10,7))
    for ri,site in enumerate(site_ids):
        records=[r for r in tokyo if r['place_id']==site]
        for ax,r in zip(axes[ri],records):
            ax.imshow(cv2.cvtColor(read_image(ROOT/'data/tokyo'/r['image'])[0],cv2.COLOR_BGR2RGB));ax.set_title(site+' '+r['view']+'/'+r['condition'],fontsize=9);ax.axis('off')
    save('tokyo_contact.pdf')
    synthetic=[q for p in benchmark['protocols'] if p['name'].startswith('synthetic_') for q in p['queries']]
    fig,axes=plt.subplots(2,5,figsize=(10,4.5))
    for ri,place in enumerate(['repo_reference','repo_test']):
        for ax,condition in zip(axes[ri],['day','night','night_severe','warm_night','fog']):
            q=next(q for q in synthetic if q['place_id']==place and q['split']==condition and q['seed']==17)
            ax.imshow(cv2.cvtColor(read_image(ROOT/q['query'])[0],cv2.COLOR_BGR2RGB));ax.set_title(condition.replace('_',' '),fontsize=8);ax.axis('off')
    save('synthetic_contact.pdf')
    # Qualitative geometry overlays use saved estimates, including explicit failures.
    overlay=cv2.imread(str(ROOT/'overlay.png'),cv2.IMREAD_UNCHANGED)
    from stitch.rendering import render
    qualitative=[]
    choices=[]
    for protocol in ['hpatches_illumination','hpatches_viewpoint','tokyo_day_to_night','tokyo_day_to_day']:
        ids=[q['id'] for p in benchmark['protocols'] if p['name']==protocol for q in p['queries']]
        by={(r['id'],r['method']):r for r in pairs if r['protocol']==protocol}
        chosen=next((i for i in ids if by[i,'superpoint']['accepted'] and not by[i,'sift']['accepted']),ids[-1])
        choices.append((protocol,chosen,by))
    fig,axes=plt.subplots(len(choices),4,figsize=(11,9))
    for ri,(protocol,identifier,by) in enumerate(choices):
        a=by[identifier,'sift'];b=by[identifier,'superpoint'];reference,_=read_image(ROOT/a['ref']);query,_=read_image(ROOT/a['query'])
        views=[reference,query]
        for row in [a,b]:
            H=np.array(row['H']) if row['H'] is not None and row['accepted'] else None
            views.append(render(query,overlay,H,reference.shape))
        for ax,view,title in zip(axes[ri],views,['Reference','Query','SIFT '+('accepted' if a['accepted'] else 'rejected'),'SP+LG '+('accepted' if b['accepted'] else 'rejected')]):
            ax.imshow(cv2.cvtColor(view,cv2.COLOR_BGR2RGB));ax.set_title(title,fontsize=9);ax.axis('off')
        qualitative.append(dict(protocol=protocol,id=identifier,sift=a['accepted'],learned=b['accepted']))
    save('qualitative_overlays.pdf')
    (ROOT/'results/qualitative_selection.json').write_text(json.dumps(qualitative,indent=2))
    if (ROOT/'results/qualitative_matches.json').exists():
        from matplotlib.collections import LineCollection
        actual=read('qualitative_matches.json')
        for protocol in ['hpatches_illumination','hpatches_viewpoint','tokyo_day_to_night','tokyo_day_to_day']:
            rs=[r for r in actual if r['protocol']==protocol]
            fig,axes=plt.subplots(2,1,figsize=(10,7))
            for ax,r in zip(axes,rs):
                ref,_=read_image(ROOT/r['ref']);query,_=read_image(ROOT/r['query'])
                h=max(ref.shape[0],query.shape[0]);w=ref.shape[1]+query.shape[1]
                canvas=np.ones((h,w,3),np.uint8)*245
                canvas[:ref.shape[0],:ref.shape[1]]=ref;canvas[:query.shape[0],ref.shape[1]:]=query
                ax.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
                source=np.array(r['source_points']).reshape(-1,2);target=np.array(r['target_points']).reshape(-1,2)
                mask=np.array(r['inlier_mask'],bool)
                indices=np.linspace(0,len(source)-1,min(80,len(source)),dtype=int) if len(source) else []
                if len(indices):
                    target=target+np.array([ref.shape[1],0]);segments=np.stack([source[indices],target[indices]],axis=1)
                    ax.add_collection(LineCollection(segments,colors=['#39c67e' if v else '#ef805d' for v in mask[indices]],linewidths=.55,alpha=.75))
                ax.set_title(r['method']+f': {len(source)} matches, {r["inliers"]} inliers; '+('accepted' if r['accepted'] else 'rejected'))
                ax.axis('off')
            save('matches_'+protocol+'.pdf')
    # Generated command appendix preserves exact argument vectors, outcomes and log paths.
    commands=[json.loads(line) for line in (ROOT/'results/commands.jsonl').read_text().splitlines() if line.strip()]
    ends={r['id']:r for r in commands if r.get('event')=='end'}
    blocks=[]
    bootstrap=read('bootstrap_commands.json') if (ROOT/'results/bootstrap_commands.json').exists() else {}
    for i,command in enumerate(bootstrap.get('commands',[]),1):
        blocks.append('\\noindent\\textbf{Initial setup/inspection '+str(i)+'}.\n'+
                      '\\begin{lstlisting}[numbers=none]\n'+command+'\n\\end{lstlisting}\n')
    for r in commands:
        if r.get('event')!='start':continue
        end=ends.get(r['id']);status='in progress at artifact generation' if end is None else f'exit {end["returncode"]}, {end["elapsed_s"]:.3f} seconds'
        blocks.append('\\noindent\\textbf{'+esc(r['id'])+'}: '+esc(status)+'.\n'+
                      '\\begin{lstlisting}[numbers=none]\n'+r['command']+'\n\\end{lstlisting}\n')
    (TABLE/'commands.tex').write_text('\n'.join(blocks))
    (TABLE/'sequence_names.tex').write_text('\n'.join(esc(s)+r'\par' for s in data['hpatches']['sequences']))
    stats['macros']=macros
    (ROOT/'results/statistics.json').write_text(json.dumps(stats,indent=2))
    inputs=['pairs.json','predictions.json','data_manifest.json','preparation.json','original_baselines.json','export.json','training.json','models.json']
    if profiles:inputs.append('profiles.json')
    provenance={name:hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest() for name in inputs}
    (ROOT/'results/artifact_provenance.json').write_text(json.dumps(dict(inputs=provenance,
        generated_tables=sorted(p.name for p in TABLE.glob('*.tex')),
        generated_figures=sorted(p.name for p in FIG.glob('*.pdf'))),indent=2))
    for fig in FIG.glob('*.pdf'):shutil.copy2(fig,ROOT/'report/figures'/fig.name)
    print('Generated',len(list(TABLE.glob('*.tex'))),'tables/macros and',len(list(FIG.glob('*.pdf'))),'figures from actual results')

if __name__=='__main__':main()
