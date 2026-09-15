"""Export native saved-model samples, or render an existing sample bundle offline.

python -m paper.transport --input runs/results/JOB/result.json --gpu 0 --output figures/transport
python -m paper.transport --samples RUN/samples.npz --output figures/continental
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_samples(arrays,metadata,output_dir):
    """Plot actual source/target/transport arrays. Does not load or run a model."""
    import numpy as np
    from .plot import style
    plt=style();out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    mf=metadata.get('manifold',metadata.get('selection',{}).get('manifold'))
    if mf not in ('S2','T2'):raise ValueError('The qualitative renderer supports S2 and T2 only.')
    expected=3 if mf=='S2' else 4
    samples={name:np.asarray(arrays[name]) for name in ('source_samples','target_samples','transported_samples')}
    for name,value in samples.items():
        if value.ndim!=2 or value.shape[1]!=expected or not len(value) or not np.isfinite(value).all():
            raise ValueError(f'{name} must be nonempty, finite [N,{expected}] native coordinates.')
    if samples['source_samples'].shape!=samples['transported_samples'].shape:
        raise ValueError('Source and transported rows must correspond one-to-one.')
    # Display is a sample plot, not an inferred density or a numerical metric.
    def display(value):
        if mf=='S2':return value
        angles=np.arctan2(value[:,[1,3]],value[:,[0,2]])
        a,b=angles.T
        return np.stack([(2+.65*np.cos(b))*np.cos(a),(2+.65*np.cos(b))*np.sin(a),.65*np.sin(b)],axis=1)
    def surface(ax):
        a,b=np.meshgrid(np.linspace(0,2*np.pi,48),np.linspace(0,np.pi if mf=='S2' else 2*np.pi,25))
        if mf=='S2':x,y,z=np.cos(a)*np.sin(b),np.sin(a)*np.sin(b),np.cos(b)
        else:x,y,z=(2+.65*np.cos(b))*np.cos(a),(2+.65*np.cos(b))*np.sin(a),.65*np.sin(b)
        ax.plot_wireframe(x,y,z,color='#aab4be',lw=.25,alpha=.24,rstride=3,cstride=4)
        ax.set_box_aspect([1,1,1 if mf=='S2' else .42]);ax.set_axis_off();ax.view_init(elev=22,azim=35)
    fig=plt.figure(figsize=(14,3.8));axes=[fig.add_subplot(1,4,i+1,projection='3d') for i in range(4)]
    for ax in axes:surface(ax)
    source=samples['source_samples'];color=np.arctan2(source[:,1],source[:,0])
    for ax,key,title in ((axes[0],'source_samples','Source samples'),(axes[1],'target_samples','Target samples'),(axes[3],'transported_samples','Transported samples')):
        points=display(samples[key]);c=color if key!='target_samples' else np.arctan2(samples[key][:,1],samples[key][:,0])
        # Deterministic subsampling only for display; all saved arrays remain unchanged.
        idx=np.linspace(0,len(points)-1,min(2500,len(points)),dtype=int)
        ax.scatter(*points[idx].T,c=c[idx],cmap='viridis',s=2.2,alpha=.65,rasterized=True,vmin=-np.pi,vmax=np.pi)
        ax.set_title(title,fontsize=11)
    starts=np.asarray(arrays.get('trajectory_sources',source[:min(32,len(source))]))
    ends=np.asarray(arrays.get('trajectory_targets',samples['transported_samples'][:len(starts)]))
    if starts.shape!=ends.shape or starts.ndim!=2 or starts.shape[1]!=expected:raise ValueError('Trajectory endpoint arrays must have matching native shapes.')
    trajectory_note='Geodesic display between recorded source/transport endpoints; not an optimizer trajectory.'
    for start,end in zip(starts,ends):
        t=np.linspace(0,1,48)
        if mf=='S2':
            dot=np.clip(np.dot(start,end)/(np.linalg.norm(start)*np.linalg.norm(end)),-1,1);angle=np.arccos(dot)
            if angle<1e-8:path=np.repeat(start[None],len(t),axis=0)
            elif abs(np.sin(angle))<1e-7:continue  # Antipodal path is not uniquely determined.
            else:path=(np.sin((1-t)*angle)[:,None]*start+np.sin(t*angle)[:,None]*end)/np.sin(angle)
        else:
            a=np.arctan2(start[[1,3]],start[[0,2]]);b=np.arctan2(end[[1,3]],end[[0,2]])
            delta=(b-a+np.pi)%(2*np.pi)-np.pi;v=a+t[:,None]*delta
            path=np.stack([np.cos(v[:,0]),np.sin(v[:,0]),np.cos(v[:,1]),np.sin(v[:,1])],axis=1)
        xyz=display(path);axes[2].plot(*xyz.T,lw=.8,alpha=.7,color=plt.cm.viridis((np.arctan2(start[1],start[0])+np.pi)/(2*np.pi)))
    axes[2].set_title('Recorded transport endpoints\nwith geodesic paths',fontsize=10)
    label=metadata.get('label','Recorded sample bundle')
    if metadata.get('smoke'):label='SMOKE / pipeline check — not paper results'
    fig.suptitle(mf+' · '+label,fontsize=12,y=1.02)
    fig.text(.5,.01,'Colors track source longitude through transport. Target colors show target longitude.',ha='center',fontsize=8)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/f'transport.{suffix}',dpi=300,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    from ._plot_data import clean
    info={'record_type':'sample_render','evaluation_only':True,'fresh_training':False,'input_metadata':metadata,
        'rendering':{'kind':'sample scatter; no density estimate','maximum_points_displayed':2500,'trajectory_note':trajectory_note,
            'sphere_view':{'elevation':22,'azimuth':35},'torus_radii':[2,.65]},
        'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in samples.items()}}
    (out/'transport.json').write_text(json.dumps(clean(info),indent=2,allow_nan=False)+'\n')
    return info


def validate_saved(saved,checkpoint,source_root,runtime):
    if saved.get('record_type') not in ('paper_training_run','paper_experiment') or saved.get('status')!='complete':
        raise ValueError('Use a completed record produced by python -m paper.run.')
    selection=saved['selection']
    if selection.get('manifold') not in ('S2','T2') or selection.get('method')!='ours':
        raise ValueError('Checkpoint transport currently supports S2/T2 RNOT models.')
    if sha(checkpoint)!=saved['checkpoint']['sha256'] or checkpoint.stat().st_size!=saved['checkpoint']['bytes']:
        raise ValueError('Saved checkpoint hash/size differs from the input record.')
    if runtime.source_manifest(source_root)!=saved['source']:
        raise ValueError('Core native source differs from the trained model source.')
    runtime_path=Path(runtime.__file__)
    if saved.get('paper_source',{}).get('paper/_runtime.py') not in (None,sha(runtime_path)):
        raise ValueError('Restoration runtime differs from the recorded training runtime.')


def export_samples(args):
    from . import _runtime as runtime
    saved=json.loads(args.input.read_text());checkpoint=args.checkpoint or args.input.with_name(saved['checkpoint']['file'])
    root=args.source_root.resolve();validate_saved(saved,checkpoint,root,runtime)
    runtime.validate_environment()
    if args.gpu is None:raise ValueError('--gpu is required when exporting samples from a checkpoint.')
    os.environ.update(JAX_PLATFORMS='cuda',CUDA_VISIBLE_DEVICES=args.gpu,HIP_VISIBLE_DEVICES=args.gpu,
        JAX_ENABLE_X64=str(saved['recipe']['x64']),JAX_THREEFRY_PARTITIONABLE='False',JAX_DEFAULT_PRNG_IMPL='threefry2x32',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    record={'record_type':'paper_transport_samples','evaluation_only':True,'independent_training_run':False,
        'status':'building','started_at':runtime.utc_now(),'input_result_sha256':sha(args.input),'checkpoint_sha256':sha(checkpoint),
        'source':saved['source'],'configuration':saved['configuration'],'selection':saved['selection'],'manifold':saved['selection']['manifold'],
        'label':saved['label']+' · saved-model visualization','smoke':saved.get('smoke',False),
        'sampling':{'seed':args.seed,'n_samples':args.n_samples,'recipe':'split2 source/target, unchanged native solver.batch_solve with full target hint batch'},
        'provenance':{'input_result':str(args.input.resolve()),'checkpoint':str(checkpoint.resolve()),'renderer_sha256':sha(Path(__file__))}}
    runtime.save_json(out/'samples.json',record)
    try:
        m=runtime.load_native(root,saved['selection']['suite'],saved['selection']['manifold'])
        if m.jax.default_backend()!='gpu' or bool(m.jax.config.jax_enable_x64)!=saved['recipe']['x64'] or m.jax.config.jax_threefry_partitionable:
            raise ValueError('Native precision/PRNG/backend differs from the declared recipe.')
        from flax import serialization
        payload=serialization.msgpack_restore(checkpoint.read_bytes())
        exp=runtime.restore_experiment(m,saved['selection'],saved['recipe'],payload,saved['configuration'],smoke=saved.get('smoke',False))
        density={name:runtime.density_configuration(exp[name]) for name in ('base','target')}
        geom=exp['manifold'];geometry={'class':type(geom).__name__,'D':geom.D,**{name:runtime.json_safe(getattr(geom,name)) for name in ('alpha','jitter') if hasattr(geom,name)}}
        if runtime.json_safe(density)!=saved['density_configuration'] or runtime.json_safe(geometry)!=saved['geometry_configuration']:
            raise ValueError('Restored native data or geometry differs from training.')
        if 'state' in payload:
            a=m.jax.tree_util.tree_map(lambda x:runtime.array_fingerprint(x,m.np),payload['state']['params'])
            b=m.jax.tree_util.tree_map(lambda x:runtime.array_fingerprint(x,m.np),payload['params'])
            if a!=b:raise ValueError('Checkpoint state.params and top-level params disagree.')
        count=sum(int(a.size) for a in m.jax.tree_util.tree_leaves(exp['state'].params))
        dtypes=sorted({str(a.dtype) for a in m.jax.tree_util.tree_leaves(exp['state'].params)})
        if count!=saved['parameter_count'] or dtypes!=saved['parameter_dtypes']:
            raise ValueError('Restored parameter count/dtypes differ from the training record.')
        record['environment']=runtime.environment();record['status']='sampling';runtime.save_json(out/'samples.json',record)
        k1,k2=m.jax.random.split(m.jax.random.PRNGKey(args.seed))
        xs=exp['base'].sample(k1,args.n_samples);targets=exp['target'].sample(k2,args.n_samples)
        ys,_=exp['solver'].batch_solve(exp['state'].params,xs,targets)
        arrays={name:m.np.asarray(value) for name,value in [('source_samples',xs),('target_samples',targets),('transported_samples',ys)]}
        record['sampling'].update(source_key=m.np.asarray(k1).tolist(),target_key=m.np.asarray(k2).tolist())
        record['arrays']={name:runtime.array_fingerprint(value,m.np) for name,value in arrays.items()}
        m.np.savez_compressed(out/'samples.npz',**arrays)
        record['samples']={'file':'samples.npz','sha256':sha(out/'samples.npz')}
        validate_saved(saved,checkpoint,root,runtime)
        record['status']='complete';record['finished_at']=runtime.utc_now();runtime.save_json(out/'samples.json',record)
        render_samples(arrays,record,out)
    except BaseException as exc:
        record.update(status='failed',error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        runtime.save_json(out/'samples.json',record);raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',type=Path,help='Completed public worker result.json for checkpoint-derived sampling.')
    source.add_argument('--samples',type=Path,help='Existing numeric sample NPZ to render with no JAX import.')
    parser.add_argument('--metadata',type=Path,help='Defaults to samples.json beside --samples.')
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu')
    parser.add_argument('--seed',type=int,default=123);parser.add_argument('--n-samples',type=int,default=1024)
    args=parser.parse_args(argv)
    if args.n_samples<1 or args.n_samples>100000:parser.error('--n-samples must be between1 and100000.')
    if args.gpu is not None and not args.gpu.isdigit():parser.error('--gpu must be one numeric GPU index.')
    sys.dont_write_bytecode=True
    if args.samples:
        import numpy as np
        metadata_path=args.metadata or args.samples.with_suffix('.json');metadata=json.loads(metadata_path.read_text())
        known=metadata.get('samples',{}).get('sha256')
        if known and sha(args.samples)!=known:raise ValueError('Sample NPZ hash differs from its metadata.')
        with np.load(args.samples,allow_pickle=False) as arrays:render_samples(dict(arrays),metadata,args.output)
    else:export_samples(args)
    print(args.output)


if __name__=='__main__':main()
