"""Render actual experiment outputs or explicitly labeled historical references.

Examples: python -m paper.plot tables --input runs/results --output figures/tables
          python -m paper.plot sweep --reference --output figures/sweep
"""
import argparse
import collections
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

from ._plot_data import clean,digest,load_input,number,paper_rows

MODES=('tables','liegroups','sweep','highd','ablations','embeddings')
VALID={'complete','complete_five_seeds','complete_finite','historical','historical_rounded','paper_reported'}


def method_label(row):
    label=row.get('method','?').replace('_gamma',' γ=').replace('_',' ')
    if row.get('gamma') is not None and 'gamma' not in label:label+=f" γ={row['gamma']:g}"
    return label


def reference_rows(mode,root):
    if mode in ('tables','liegroups'):
        path=root/'paper_targets.json';data=json.loads(path.read_text())
        rows=paper_rows(data,mode)
        if mode=='liegroups':
            for gamma in (0.1,0.05,0.01,0.005,0.001):
                rows.append({'suite':mode,'manifold':'SE3','family':'SE3','method':'RCPM','gamma':gamma,
                    'label':'Paper-reported reference','host':'paper: AMD MI300X','status':'paper: numerically unstable',
                    'kl':float('nan'),'kl_se':float('nan'),'ess_ratio':float('nan'),'ess_ratio_se':float('nan'),
                    'group_id':f'paper-SE3-rcpm-{gamma}','cohort_id':'paper-reference','uncertainty':'No numerical target stated',
                    'provenance':{'table':5,'claim':'Numerically unstable'}})
        return rows,[{'input':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'note':'Published targets; not new runs.'}]
    name={'sweep':'dimension_logs.json','highd':'dimension_logs.json','ablations':'ablation_results.json','embeddings':'embedding_results.json'}[mode]
    return load_input(root/name)


def curve_identity(row):
    # Dimension-specific arrays/geometry are retained per point, not used to combine metrics.
    # Lines only connect independently reported points with a common recipe, host and source.
    i=row.get('identity')
    if not i:return row.get('group_id','reference')
    cfg=copy.deepcopy(i.get('configuration',{}))
    cfg.pop('manifold_name',None)
    for name in ('manifold','dimension','dim'):
        cfg.pop(name,None)
    if isinstance(cfg.get('model'),dict):cfg['model'].pop('max_dist',None)
    return digest({'source':i.get('source'),'runner':i.get('runner_sha256'),'helper':i.get('helper_sha256'),'paper_source':i.get('paper_source'),
        'environment':i.get('environment'),'label':i.get('label'),'configuration':cfg,
        'evaluation':i.get('evaluation_config',i.get('evaluation_recipe')),'method':row.get('method'),
        'gamma':row.get('gamma'),'family':row.get('family'),'suite':row.get('suite')})


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':10,
        'axes.labelsize':8,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.18,'axes.axisbelow':True,'figure.facecolor':'white',
        'savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42})
    return plt


def render_tables(rows,mode,plt):
    manifolds=sorted({r['manifold'] for r in rows},key=lambda m:(m.startswith('T') or m=='SE3',m))
    sizes=[sum(r['manifold']==m for r in rows) for m in manifolds]
    height=max(4.2,.34*max(sizes,default=1)+2.0)
    fig,axes=plt.subplots(len(manifolds),2,figsize=(11,height*len(manifolds)),squeeze=False)
    for index,mf in enumerate(manifolds):
        subset=[r for r in rows if r['manifold']==mf]
        subset.sort(key=lambda r:(not method_label(r).startswith('RNOT'),r.get('host',''),-(number(r.get('gamma')) if r.get('gamma') is not None else 0),method_label(r),r.get('group_id','')))
        labels=collections.Counter()
        for r in subset:
            text=method_label(r) if r.get('status')=='paper_reported' or str(r.get('status','')).startswith('paper:') else method_label(r)+' · '+str(r.get('host','unknown'))
            labels[text]+=1;r['display_label']=text
        for text,count in labels.items():
            if count>1:
                for variant,r in enumerate([r for r in subset if r['display_label']==text],1):r['display_label']=text+f' · variant {variant}'
        for col,metric in enumerate(('kl','ess_ratio')):
            ax=axes[index,col]
            for y,row in enumerate(subset):
                value=number(row.get(metric));err=number(row.get(metric+'_se'))
                if row.get('status') in VALID and math.isfinite(value):
                    ax.errorbar(value,y,xerr=err if math.isfinite(err) and err>=0 else None,
                        fmt='o',ms=5,color='#bd3d38' if method_label(row).startswith('RNOT') else '#235d83',capsize=3,lw=1.3)
                else:
                    ax.text(.02,y,row.get('status','missing'),transform=ax.get_yaxis_transform(),fontsize=7,color='#9d3131',va='center')
            ax.set_yticks(range(len(subset)),[r['display_label'] for r in subset] if col==0 else [])
            ax.set_ylim(len(subset)-.5,-.5);ax.set_title(mf+' · '+('KL divergence' if metric=='kl' else 'ESS ratio'))
            ax.set_xlabel('KL' if metric=='kl' else 'ESS / batch size')
            if metric=='ess_ratio':ax.set_xlim(-.025,1.045)
    return fig


def render_sweep(rows,mode,plt,notes):
    import numpy as np
    families=sorted({r.get('family',str(r['manifold'])[0]) for r in rows})
    fig,axes=plt.subplots(1,len(families),figsize=(3.7*len(families),2.9),squeeze=False)
    for ax,family in zip(axes[0],families):
        groups=collections.defaultdict(list)
        for row in rows:
            if row.get('family',str(row['manifold'])[0])==family:groups[curve_identity(row)].append(row)
        colors=plt.cm.viridis(np.linspace(.08,.9,max(len(groups),2)))
        for color,(group,items) in zip(colors,sorted(groups.items(),key=lambda x:method_label(x[1][0]))):
            points=collections.defaultdict(list)
            for row in items:points[row['dimension']].append(row)
            expected=([2,5,10,20,30,40] if mode=='highd' and items[0]['method'].startswith('RNOT') else list(range(2,11)))
            xs=sorted(set(expected)|set(points));ys=[];low=[];high=[]
            for x in xs:
                candidates=points.get(x,[])
                if len(candidates)!=1:
                    ys.append(np.nan);low.append(np.nan);high.append(np.nan)
                    notes.append({'curve':group,'dimension':x,'plot_status':'missing' if not candidates else 'ambiguous_duplicate_point'});continue
                row=candidates[0];value=number(row.get('kl'))+1;err=number(row.get('kl_se'))
                if row.get('status') not in VALID or not math.isfinite(value) or value<=0:
                    ys.append(np.nan);low.append(np.nan);high.append(np.nan)
                    notes.append({'curve':group,'dimension':x,'plot_status':'nonfinite_failed_or_nonpositive_KL_plus_1','status':row.get('status')});continue
                ys.append(value);low.append(value-err if math.isfinite(err) and value-err>0 else np.nan);high.append(value+err if math.isfinite(err) else np.nan)
            first=items[0];ours=first['method'].startswith('RNOT')
            label=method_label(first) if str(first.get('status','')).startswith('historical') else method_label(first)+' · '+first.get('host','')
            repeated=sum(method_label(v[0])==method_label(first) and v[0].get('host')==first.get('host') for v in groups.values())
            if repeated>1:label+=' · variant '+str(list(sorted(groups)).index(group)+1)
            for item in items:item['display_label']=label
            ax.semilogy(xs,ys,'s-' if ours else 'o-',color='#bd3d38' if ours else color,ms=3.5,lw=1.4 if ours else 1,label=label)
            ax.fill_between(xs,low,high,color='#bd3d38' if ours else color,alpha=.12)
        ax.set_title('Spheres' if family=='S' else ('Tori' if family=='T' else family));ax.set_xlabel('Dimension p');ax.set_ylabel('KL + 1 (log scale)')
        ax.set_xlim(1.5,40.5 if mode=='highd' else 10.5);ax.legend(frameon=False,fontsize=6.5,loc='upper left',bbox_to_anchor=(0,-.22))
    return fig


def render_ablations(rows,plt,notes):
    import numpy as np
    ablations=list(dict.fromkeys(r['ablation'] for r in rows))
    columns=sorted({(r['manifold'],r.get('cohort_id','unknown')) for r in rows}, key=lambda v: ({'S2':0,'T2':1,'S10':2,'T10':3}.get(v[0],4),v))
    data=np.full((len(ablations),len(columns)),np.nan)
    for i,ablation in enumerate(ablations):
        for j,(mf,cohort) in enumerate(columns):
            candidates=[r for r in rows if r['ablation']==ablation and r['manifold']==mf and r.get('cohort_id','unknown')==cohort]
            if len(candidates)==1 and candidates[0]['status'] in VALID:data[i,j]=number(candidates[0].get('kl'))
            elif candidates:notes.append({'ablation':ablation,'manifold':mf,'cohort':cohort,'plot_status':'failed_or_ambiguous','n':len(candidates)})
    fig,ax=plt.subplots(figsize=(max(7,1.3*len(columns)),max(6,.34*len(ablations)+2)))
    # Color transformation only; raw signed KL remains in cells and exported data.
    signed_log=np.sign(data)*np.log1p(np.abs(data))
    im=ax.imshow(np.ma.masked_invalid(signed_log),aspect='auto',cmap='RdYlBu_r')
    for i in range(len(ablations)):
        for j in range(len(columns)):
            ax.text(j,i,f'{data[i,j]:.3g}' if np.isfinite(data[i,j]) else '—',ha='center',va='center',fontsize=8)
    ax.set_xticks(range(len(columns)),[mf if cohort.startswith('historical') else mf+'\n'+cohort[:6] for mf,cohort in columns]);ax.set_yticks(range(len(ablations)),[x.replace('_',' ') for x in ablations]);ax.grid(False)
    fig.colorbar(im,ax=ax,shrink=.55,label='sign(KL) × log(1 + |KL|), color only');ax.set_title('Ablations · cell values are recorded KL')
    return fig


def render_embeddings(rows,plt):
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    groups=collections.defaultdict(list)
    for row in rows:groups[(row.get('method'),row.get('group_id'))].append(row)
    cohorts=sorted({r.get('cohort_id',r.get('group_id','unknown')) for r in rows})
    for (method,group),values in groups.items():
        values=sorted(values,key=lambda r:r['M']);xs=[r['M'] for r in values]
        label=method
        if len(cohorts)>1:
            cohort=values[0].get('cohort_id',group)
            host=values[0].get('host')
            label+=' · '+(str(host)+' · ' if host else '')+'variant '+str(cohorts.index(cohort)+1)
        for row in values:row['display_label']=label
        for ax,key in zip(axes,('coverage_radius','separation','near_collision_fraction')):
            ys=[number(r[key]) for r in values]
            if key=='near_collision_fraction':ys=[max(y,1e-8) if math.isfinite(y) and y>=0 else np.nan for y in ys]
            ax.plot(xs,ys,'s-' if method.lower()=='fps' else 'o-',ms=4,lw=1.3,label=label)
            ax.set_xscale('log',base=2);ax.set_xlabel('Landmarks M')
    for ax,label in zip(axes,('Coverage radius','Minimum separation','Near-collision fraction')):ax.set_ylabel(label);ax.legend(frameon=False,fontsize=7)
    eps=rows[0].get('epsilon',.001);axes[1].axhline(eps,color='#bd3d38',ls='--',lw=1,label=f'ε={eps}')
    axes[2].set_yscale('log');axes[2].axhline(1e-8,color='.5',ls=':',lw=1);fig.text(.5,-.025,'Zero near-collision rates are displayed at 10⁻⁸; raw values are retained.',ha='center',va='top',fontsize=7,color='.45')
    return fig


def write_outputs(fig,rows,notes,out,mode,reference):
    out.mkdir(parents=True,exist_ok=True)
    label='Recorded experiment outputs'
    if reference:label='Paper-reported values' if mode in ('tables','liegroups') else ('Historical log values' if mode in ('sweep','highd') else 'Historical recorded values')
    elif all(str(r.get('status','')).startswith('historical') for r in rows):label='Historical recorded values'
    title={'tables':'Main table results','liegroups':'Lie-group results','sweep':'Dimension sweep','highd':'High-dimensional transport','ablations':'Ablation results','embeddings':'Landmark embedding experiment'}[mode]
    if any('smoke' in str(r.get('label','')).lower() for r in rows):label='Smoke check; not paper results'
    fig.suptitle(label,fontsize=8,color='.45',y=1.015)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/f'{mode}.{suffix}',dpi=300,bbox_inches='tight',pad_inches=.15)
    for row in rows:
        row['fresh_experiment']=False if reference or str(row.get('status','')).startswith('historical') else None
    summary={'schema_version':1,'plot':mode,'reference_requested':reference,'fresh_experiment':False if reference else None,'label':label,'rows':rows,'input_notes':notes,
        'uncertainty_note':'Each row records its uncertainty unit. Evaluation batches are not independent training seeds. Paper reference plus/minus is descriptive.',
        'rendering_note':'No raw measured value is clipped or replaced. Dimension plots transform KL to KL+1. Embedding zero rates use a labeled display floor.',
        'status_counts':dict(collections.Counter(r.get('status','unknown') for r in rows))}
    (out/f'{mode}.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')
    columns=['suite','manifold','dimension','method','gamma','ablation','M','coverage_radius','separation','near_collision_fraction','kl','kl_se','ess_ratio','ess_ratio_se','status','n_training_runs','n_evaluation_batches','uncertainty','label','host','group_id','cohort_id','display_label','fresh_experiment']
    with (out/f'{mode}.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,columns);writer.writeheader()
        for row in rows:writer.writerow({key:clean(row.get(key)) for key in columns})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=MODES);parser.add_argument('--input',type=Path,help='Results directory, comparison JSON, or explicit plot-data JSON.')
    parser.add_argument('--output',type=Path,required=True,help='Output directory for PNG, PDF, CSV and provenance JSON.')
    parser.add_argument('--reference',action='store_true',help='Use labeled archived/paper-reported reference data; no model execution.')
    parser.add_argument('--reference-dir',type=Path,default=Path(__file__).resolve().parent.parent/'reference')
    parser.add_argument('--group',action='append',help='Optional group/cohort ID selector; repeat for several explicit groups.')
    args=parser.parse_args(argv)
    if args.reference==bool(args.input):parser.error('Choose exactly one of --input and --reference.')
    rows,notes=reference_rows(args.mode,args.reference_dir) if args.reference else load_input(args.input)
    rows=[r for r in rows if r.get('suite')==args.mode or (args.mode=='highd' and r.get('suite')=='sweep' and r.get('method','').startswith('RCPM'))]
    if args.group:rows=[r for r in rows if r.get('group_id') in args.group or r.get('cohort_id') in args.group]
    if not rows:parser.error('No matching rows; no figure was produced. Check the input protocol and group IDs.')
    plt=style()
    if args.mode in ('tables','liegroups'):fig=render_tables(rows,args.mode,plt)
    elif args.mode in ('sweep','highd'):fig=render_sweep(rows,args.mode,plt,notes)
    elif args.mode=='ablations':fig=render_ablations(rows,plt,notes)
    else:fig=render_embeddings(rows,plt)
    write_outputs(fig,rows,notes,args.output,args.mode,args.reference);plt.close(fig)
    print(json.dumps({'output':str(args.output),'rows':len(rows),'mode':args.mode,'status_counts':dict(collections.Counter(r.get('status','unknown') for r in rows))}))


if __name__=='__main__':main()
