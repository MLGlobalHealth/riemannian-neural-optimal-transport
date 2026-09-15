"""Bounded, predeclared release checks; never imports numerical libraries."""
import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/data/am1118/mlgh-paper-release-check-20260908')
PYTHON='/data/am1118/mlgh-zip-jax0435-restored-20260908/venv/bin/python'
CUDA_ROOT='/data/am1118/mlgh-zip-jax0435-restored-20260908/venv/lib/python3.11/site-packages/nvidia/cuda_nvcc'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def capacity(gpu):
 result=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
 parts=[x.strip() for x in result.split(',')]
 return {'at':now(),'raw':result,'idle':int(parts[2])<500 and int(parts[3])>16000 and int(parts[4])<=5}
def main():
 assert not (ROOT/'provenance/release_launch_manifest.json').exists(),'This declaration must run only once.'
 plans=json.loads((ROOT/'provenance/dryrun_plans.json').read_text())
 common={'CUDA_ROOT':CUDA_ROOT,'JAX_PLATFORMS':'cuda','JAX_ENABLE_X64':'True','JAX_THREEFRY_PARTITIONABLE':'False','JAX_DEFAULT_PRNG_IMPL':'threefry2x32','JAX_RANDOM_SEED_OFFSET':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2','XLA_PYTHON_CLIENT_PREALLOCATE':'false','MPLBACKEND':'Agg','MPLCONFIGDIR':str(ROOT/'mplconfig'),'XDG_CACHE_HOME':str(ROOT/'cache')}
 jobs=[]
 for name,gpu,cpu in [('se3_native',0,'0-3'),('s2_ours_smoke',1,'4-7'),('s2_rcpm_smoke',3,'12-15'),('ablation_smoke',4,'16-19')]:
  plan=plans[name]['plan']
  args=[PYTHON,'-u','-m','paper.run',plan['suite'],'--gpus',str(gpu),'--output',str(ROOT/'results'/name),'--job',plans[name]['job_id']]
  if plan['smoke']:args.append('--smoke')
  jobs.append({'id':name,'gpu':gpu,'cpu_affinity':cpu,'command':['taskset','-c',cpu,*args],'depends_on':None})
 for name,gpu,cpu,args in [
 ('continental_smoke',5,'20-23',['paper.continental','--config','paper/configs/continental.json','--output',str(ROOT/'results/continental_smoke'),'--gpu','5','--smoke']),
 ('embeddings_full',6,'24-27',['paper.embeddings','--output',str(ROOT/'results/embeddings_full.json'),'--gpu','6']),
 ('s2_checkpoint_transport',1,'4-7',['paper.transport','--input',str(ROOT/'results/s2_ours_smoke/results/tables_S2_ours_random_seed12345/result.json'),'--gpu','1','--output',str(ROOT/'results/s2_checkpoint_transport'),'--n-samples','16'])]:
  jobs.append({'id':name,'gpu':gpu,'cpu_affinity':cpu,'command':['taskset','-c',cpu,PYTHON,'-u','-m',*args],'depends_on':'s2_ours_smoke' if name=='s2_checkpoint_transport' else None})
 for j in jobs:
  j['environment']={**common,'CUDA_VISIBLE_DEVICES':str(j['gpu'])}
  if j['id']=='continental_smoke':j['environment']['PYTHONPATH']=str(ROOT/'pandas-overlay')
  else:j['environment']['PYTHONPATH']=''
  j['log']=str(ROOT/'logs'/f'{j["id"]}.log')
 manifest={'created_at':now(),'supervisor_pid':os.getpid(),'source_manifest':json.loads((ROOT/'provenance/source_export_manifest.json').read_text()),'base_freeze_sha256':'59bf165a84bbf82fa6ddeafc4e0472fa25c74a501c4526084c6e8000cb37f559','authorization':'One native200-updateSE3 RNOT model/five evaluations; three explicitly tinyS2 checks; continental smoke; full default embedding computation; one16-sampleS2checkpoint restoration. No other training.','jobs':jobs}
 save(ROOT/'provenance/release_launch_manifest.json',manifest)
 statuses={j['id']:{'status':'pending'} for j in jobs};active={}
 save(ROOT/'provenance/release_status.json',{'supervisor_pid':os.getpid(),'status':'running','jobs':statuses})
 while any(s['status'] in ('pending','running') for s in statuses.values()):
  for ident,(proc,handle) in list(active.items()):
   rc=proc.poll()
   if rc is not None:
    handle.close();statuses[ident].update(status='complete' if rc==0 else 'failed',exit_code=rc,finished_at=now());del active[ident]
    print(json.dumps({'id':ident,**statuses[ident]}),flush=True)
  for j in jobs:
   status=statuses[j['id']]
   if status['status']!='pending':continue
   if j['depends_on']:
    dep=statuses[j['depends_on']]['status']
    if dep in ('failed','blocked','not_started'):
     status.update(status='not_started',reason='dependency_failed');continue
    if dep!='complete':continue
   if any(statuses[x]['gpu']==j['gpu'] for x in active):continue
   cap=capacity(j['gpu'])
   if not cap['idle']:
    status.update(status='blocked',capacity=cap,reason='capacity_gate');continue
   handle=open(j['log'],'x')
   proc=subprocess.Popen(j['command'],cwd=ROOT/'source',env={**os.environ,**j['environment']},stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
   active[j['id']]=(proc,handle);status.update(status='running',pid=proc.pid,gpu=j['gpu'],started_at=now(),capacity=cap)
   print(json.dumps({'id':j['id'],**status}),flush=True)
  save(ROOT/'provenance/release_status.json',{'supervisor_pid':os.getpid(),'status':'running','updated_at':now(),'jobs':statuses})
  if active:time.sleep(1)
 result={'supervisor_pid':os.getpid(),'status':'complete' if all(x['status']=='complete' for x in statuses.values()) else 'failed','finished_at':now(),'jobs':statuses}
 save(ROOT/'provenance/release_status.json',result)
 print(json.dumps(result),flush=True)
 raise SystemExit(0 if result['status']=='complete' else 1)
if __name__=='__main__':main()
