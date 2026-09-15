"""Pure-standard-library input adapters. No model imports or metric substitution."""
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

SEEDS = [12345,23456,34567,45678,56789]
ALIASES = {'table':'tables','highD':'highd'}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()[:12]

def number(value):
    try: return float(value)
    except (TypeError,ValueError): return float('nan')

def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return 'NaN' if math.isnan(value) else ('+Infinity' if value>0 else '-Infinity')
    return value

def stats(values):
    vals=[number(v) for v in values]
    if not vals or not all(math.isfinite(v) for v in vals):return {'mean':float('nan'),'se':float('nan'),'raw':vals}
    return {'mean':statistics.mean(vals),'se':statistics.pstdev(vals)/math.sqrt(len(vals)),'raw':vals}

def environment_identity(env):
    env=copy.deepcopy(env); settings=env.get('environment',{})
    selected=settings.get('CUDA_VISIBLE_DEVICES'); lines=env.get('nvidia_smi','').splitlines()
    parsed=[tuple(x.strip() for x in line.split(',')) for line in lines]
    match=[(x[1],x[3],x[4]) for x in parsed if len(x)==5 and x[0]==selected]
    if env.get('hostname') and len(match)==1:
        env['selected_accelerator_types']=match
        env.pop('nvidia_smi',None);env.pop('jax_devices',None)
        settings.pop('CUDA_VISIBLE_DEVICES',None)
        if settings.get('HIP_VISIBLE_DEVICES')==selected:settings.pop('HIP_VISIBLE_DEVICES',None)
    return env

def identity(record):
    sel=copy.deepcopy(record.get('selection',{}));sel.pop('seed',None)
    cfg=copy.deepcopy(record.get('configuration',{}));cfg.pop('seed',None)
    for part in ('training','builder'):
        if isinstance(cfg.get(part),dict):cfg[part].pop('seed',None)
    ev=copy.deepcopy(record.get('evaluation_config',{}));ev.pop('seed',None)
    return {'selection':sel,'configuration':cfg,'evaluation_config':ev,'source':record.get('source',{}),
        'runner_sha256':record.get('runner_sha256'),'helper_sha256':record.get('helper_sha256'),
        'label':record.get('label'),'environment':environment_identity(record.get('environment',{})),
        'density_configuration':record.get('density_configuration'),'geometry_configuration':record.get('geometry_configuration'),
        'landmark_configuration':record.get('landmark_configuration'),'recipe':record.get('recipe'),'paper_source':record.get('paper_source')}

def row_for(record,path,sha):
    sel=record.get('selection',{});manifold=sel.get('manifold','unknown');dim=sel.get('dimension')
    if manifold in ('sphere','torus'):manifold=('S' if manifold=='sphere' else 'T')+str(dim)
    match=re.fullmatch(r'([ST])(\d+)',manifold)
    if match:family,dim=match[1],int(match[2])
    else:family=manifold
    suite=ALIASES.get(sel.get('suite'),sel.get('suite')) or ('liegroups' if manifold in ('SO3','SE3') else 'unknown')
    method=sel.get('method','unknown');gamma=sel.get('gamma')
    name=('RNOT '+str(sel.get('landmark_method') or '').upper()).strip() if method=='ours' else ('RCPM' if method=='rcpm' else method)
    ident=identity(record);group=digest(ident)
    # Missing provenance must not accidentally merge otherwise unrelated records.
    if not record.get('source') or not record.get('environment'):group=digest([ident,sha])
    cohort={k:ident[k] for k in ('source','runner_sha256','helper_sha256','environment','label','paper_source')}
    batches=record.get('evaluation_batches',[])
    status=record.get('status','unknown')
    if status=='complete' and len(batches)!=5:status='partial_evaluation'
    row={'suite':suite,'family':family,'manifold':manifold,'dimension':dim,'method':name,'gamma':gamma,
        'ablation':sel.get('ablation'),'seed':sel.get('seed'),'label':record.get('label','unlabeled'),
        'host':record.get('environment',{}).get('hostname','unknown'),'group_id':group,'cohort_id':digest(cohort),
        'status':status,'n_training_runs':1,'n_evaluation_batches':len(batches),
        'uncertainty':'native SE across evaluation batches of one trained model','identity':ident,
        'provenance':{'file':str(path),'sha256':sha},'failure':record.get('error'),'raw_evaluation_batches':batches}
    for metric in ('kl','ess_ratio'):
        values=[b.get('metrics',{}).get(metric) for b in batches]
        result=stats(values);row[metric]=result['mean'];row[metric+'_se']=result['se']
        if status=='complete' and not math.isfinite(result['mean']):row['status']='nonfinite'
    row['runtime_seconds']=record.get('timings_seconds',{}).get('training_native_return')
    return row

def aggregate_tables(rows):
    grouped={};other=[]
    for row in rows:
        if row['suite']=='tables':grouped.setdefault(row['group_id'],[]).append(row)
        else:other.append(row)
    for group,items in grouped.items():
        row=copy.deepcopy(items[0]);row['individual_runs']=items;row['seed']=None
        seeds=[x['seed'] for x in items];row['training_seeds']=seeds
        row['n_training_runs']=len(items);row['uncertainty']='population SE across independent training-seed means'
        complete=len(seeds)==5 and sorted(seeds)==SEEDS and all(x['status']=='complete' for x in items)
        row['status']='complete_five_seeds' if complete else ('ambiguous_duplicate_seed' if len(set(seeds))!=len(seeds) else 'partial_or_failed_seed_group')
        for metric in ('kl','ess_ratio'):
            value=stats([x[metric] for x in items]);row[metric]=value['mean'] if complete else float('nan');row[metric+'_se']=value['se'] if complete else float('nan')
        other.append(row)
    return other

def guard_duplicates(rows):
    buckets={}
    for row in rows:
        if row.get('suite')!='tables':buckets.setdefault((row.get('group_id'),row.get('seed'),row.get('dimension'),row.get('ablation'),row.get('M')),[]).append(row)
    for values in buckets.values():
        if len(values)>1:
            for row in values:row['status']='ambiguous_duplicate_execution'
    return rows

def from_main_comparison(data):
    rows=[]
    for group in data['groups']:
        i=group['identity'];sel=i['selection'];mf=sel['manifold'];dim=sel['dimension']
        mf=('S' if mf=='sphere' else 'T')+str(dim) if mf in ('sphere','torus') else mf
        row={'suite':'tables','family':mf[0],'manifold':mf,'dimension':dim,'method':('RNOT '+str(sel.get('landmark_method','')).upper()) if sel['method']=='ours' else 'RCPM','gamma':sel.get('gamma'),'ablation':None,'label':i['label'],'host':i['environment'].get('hostname','unknown'),'group_id':group['group_id'],'cohort_id':digest(i['environment']),'identity':i,'status':group['status'],'n_training_runs':group['independent_valid_seed_count'],'n_evaluation_batches':5,'uncertainty':'population SE across training-seed means','provenance':{'runs':group['runs']}}
        usable=group['exact_shipped_seed_set'] and not group['duplicate_seeds'] and group['independent_valid_seed_count']==5
        for metric in ('kl','ess_ratio'):
            s=group.get('across_seed_statistics',{}).get(metric) or {};row[metric]=number(s.get('mean')) if usable else float('nan');row[metric+'_se']=number(s.get('se_ddof0')) if usable else float('nan')
        if usable:row['status']='complete_five_seeds'
        rows.append(row)
    return rows

def load_input(path):
    path=Path(path);notes=[];rows=[];seen=set()
    paths=sorted(path.rglob('result.json')) if path.is_dir() else [path]
    for file in paths:
        payload=file.read_bytes();sha=hashlib.sha256(payload).hexdigest()
        if sha in seen:notes.append({'identical_copy_ignored':str(file),'sha256':sha});continue
        seen.add(sha)
        try:data=json.loads(payload)
        except (json.JSONDecodeError,UnicodeDecodeError) as exc:notes.append({'invalid_input':str(file),'error':str(exc)});continue
        if data.get('record_type') in ('historical_plot_reference','paper_embedding_run') or ('rows' in data and not 'selection' in data):
            rows.extend(copy.deepcopy(data['rows']));notes.append({'input':str(file),'sha256':sha,'note':data.get('note'),'sources':data.get('sources'), 'provenance':data.get('provenance')});continue
        if 'groups' in data and 'comparison_policy' in data:
            rows.extend(from_main_comparison(data));notes.append({'comparison':str(file),'sha256':sha});continue
        if 'cohorts' in data and 'records' in data:
            for item in data['records']:
                raw=item.get('raw_record')
                if raw:
                    row=row_for(raw,item.get('evidence_paths'),item['sha256'])
                    row['status']=item['outcome'];row['cohort_id']=item['cohort_id'];row['group_id']=item['configuration_group_id']
                    rows.append(row)
            notes.append({'comparison':str(file),'sha256':sha});continue
        if data.get('evaluation_only') or 'evaluation' in data.get('record_type','') or data.get('smoke'):
            notes.append({'excluded_nontraining_or_smoke':str(file),'record_type':data.get('record_type')});continue
        if not isinstance(data.get('selection'),dict):notes.append({'unrecognized':str(file)});continue
        rows.append(row_for(data,file,sha))
    # Comparison groups are already aggregated; native rows are aggregated only once.
    raw=[r for r in rows if r.get('suite')=='tables' and 'seed' in r]
    rest=[r for r in rows if not (r.get('suite')=='tables' and 'seed' in r)]
    return guard_duplicates(aggregate_tables(raw)+rest),notes

def paper_rows(targets,mode):
    rows=[]
    for table in targets['tables']:
        if mode=='tables' and table['number'] not in (1,2):continue
        if mode=='liegroups' and table['number']!=5:continue
        for item in table['rows']:
            mf=item.get('manifold',table.get('manifold'))
            row={'suite':mode,'manifold':mf,'family':mf,'dimension':2 if mode=='tables' else None,
                'method':item['method'],'gamma':item.get('gamma'),'ablation':None,'label':'Paper-reported reference',
                'host':'paper: AMD MI300X','status':'paper_reported','group_id':f"paper-{table['number']}-{mf}-{item['method']}-{item.get('gamma')}",
                'cohort_id':'paper-reference','uncertainty':table.get('uncertainty_label','published plus/minus; construction not inferred'),
                'n_training_runs':table.get('independent_training_runs'),'n_evaluation_batches':5,'provenance':{'table':table['number']}}
            for metric in ('kl','ess_ratio'):
                v=item.get(metric);row[metric]=number(v.get('mean')) if isinstance(v,dict) else number(v);row[metric+'_se']=number(v.get('reported_plus_minus')) if isinstance(v,dict) else float('nan')
            gamma_match=re.fullmatch(r'RCPM_gamma([0-9.]+)',item['method'])
            if gamma_match:row['method']='RCPM';row['gamma']=float(gamma_match[1])
            rows.append(row)
    return rows
