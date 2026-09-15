"""Regenerate the landmark embedding experiment underlying Figure 4."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

NOTEBOOK_SHA256 = "9efe760b5bc6d8293075715eeb131b302edc794acc1acfb105eba67c0a9b2d37"

def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New JSON output file.')
    parser.add_argument('--platform',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--gpu',help='Physical GPU index, required for CUDA.')
    parser.add_argument('--landmarks',type=int,nargs='+',default=[8,16,32,64,128,256,512,1024,2048])
    parser.add_argument('--validation-size',type=int,default=2048)
    parser.add_argument('--candidates',type=int,default=4096)
    parser.add_argument('--subset-size',type=int,default=512)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--epsilon',type=float,default=1e-3)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--x64',action='store_true',help='Explicit precision variant; the notebook does not enable x64.')
    parser.add_argument('--smoke',action='store_true',help='Small execution check; not the full paper experiment.')
    args=parser.parse_args(argv)
    if args.platform=='cuda' and (args.gpu is None or not args.gpu.isdigit()):
        parser.error('CUDA requires a single numeric --gpu index.')
    if args.output.exists(): parser.error('The output file must be new.')
    if args.smoke:
        args.landmarks=[8,16];args.validation_size=128;args.candidates=128;args.subset_size=64;args.repeats=2
    if not args.landmarks or min(args.landmarks)<1 or len(set(args.landmarks))!=len(args.landmarks):
        parser.error('Landmark counts must be unique positive integers.')
    if max(args.landmarks)>args.candidates or min(args.validation_size,args.subset_size)<2 or args.repeats<1 or args.epsilon<=0:
        parser.error('Need enough FPS candidates, at least two samples and a positive repeat count/epsilon.')
    args.landmarks.sort()
    return args

def compute(args, record):
    import jax
    import jax.numpy as jnp
    from src.embeddings import sample_landmarks_random,farthest_point_sampling,dist_to_landmarks
    from src.manifolds import get as get_manifold
    import src.densities as densities

    def min_pairwise_separation(Z: jnp.ndarray, tol: float):
        """
        Compute non-collapse statistics in embedding space (L2 norm).
    
        s_M := min_{a≠b} ||z_a - z_b||_2
        ρ_M(ε) := fraction of pairs with ||z_a - z_b||_2 < ε
        """
        H = Z.shape[0]
        norm2 = jnp.sum(Z * Z, axis=1, keepdims=True)
        d2 = norm2 + norm2.T - 2.0 * (Z @ Z.T)
        d2 = jnp.maximum(d2, 0.0)

        eye = jnp.eye(H, dtype=bool)
        d2 = jnp.where(eye, jnp.inf, d2)

        min_sep = jnp.sqrt(jnp.min(d2))  # s_M
        frac_close = jnp.mean((d2 < (tol * tol)) & (~eye))  # ρ_M(ε)
        return min_sep, frac_close

    def sweep_M_diagnostics(manifold, X_val, landmarks_max, Ms, key,
                            subset_size=512, tol=1e-3, repeats=5):
        """
        Sweep over landmark counts M and compute paper diagnostics:
        s_M, ρ_M(ε), R_M
        """
        D_all = dist_to_landmarks(manifold, X_val, landmarks_max)
        N = X_val.shape[0]

        results = []
        for M in Ms:
            Dm = D_all[:, :M]
        
            # Coverage radius: R_M = max_x min_j d(x, ℓ_j)
            R_M = float(jnp.max(jnp.min(Dm, axis=1)))
        
            # Non-collapse statistics (repeated subsampling for variance reduction)
            min_seps, frac_closes = [], []
            for _ in range(repeats):
                key, ksub = jax.random.split(key)
                H = min(subset_size, N)
                idx = jax.random.choice(ksub, N, shape=(H,), replace=False)
                Z = Dm[idx]
                s_M, rho_M = min_pairwise_separation(Z, tol=tol)
                min_seps.append(s_M)
                frac_closes.append(rho_M)

            results.append({
                "M": int(M),
                "R_M": R_M,
                "s_M_mean": float(jnp.mean(jnp.stack(min_seps))),
                "s_M_std": float(jnp.std(jnp.stack(min_seps))),
                "rho_M_mean": float(jnp.mean(jnp.stack(frac_closes))),
                "separation_repeats": [float(x) for x in min_seps],
                "near_collision_repeats": [float(x) for x in frac_closes],
            })

        return results

    key=jax.random.PRNGKey(args.seed)
    manifold=get_manifold('S2')
    base=densities.get(manifold,'SphereUniform')
    key,kVal=jax.random.split(key)
    X_val=base.sample(kVal,args.validation_size)
    key,kRand=jax.random.split(key)
    random=sample_landmarks_random(base,kRand,max(args.landmarks))
    key,kCand,kFPS=jax.random.split(key,3)
    candidates=base.sample(kCand,args.candidates)
    fps=farthest_point_sampling(manifold,candidates,max(args.landmarks),kFPS)
    key,kDiag1,kDiag2=jax.random.split(key,3)
    values={name:sweep_M_diagnostics(manifold,X_val,landmarks,args.landmarks,k,
        subset_size=args.subset_size,tol=args.epsilon,repeats=args.repeats)
        for name,landmarks,k in [('Random',random,kDiag1),('FPS',fps,kDiag2)]}
    record['native_results']=values
    record['sampling_keys']={name:jax.device_get(k).tolist() for name,k in
        [('validation',kVal),('random_landmarks',kRand),('fps_candidates',kCand),('fps',kFPS),('diagnostics_random',kDiag1),('diagnostics_fps',kDiag2)]}
    record['environment'].update(jax_backend=jax.default_backend(),jax_devices=[str(d) for d in jax.devices()],
        jax_enable_x64=bool(jax.config.jax_enable_x64),jax_threefry_partitionable=bool(jax.config.jax_threefry_partitionable),
        validation_dtype=str(X_val.dtype))
    record['rows']=[{'suite':'embeddings','method':name,'M':r['M'],'coverage_radius':r['R_M'],
        'separation':r['s_M_mean'],'near_collision_fraction':r['rho_M_mean'],
        'separation_std':r['s_M_std'],'epsilon':args.epsilon,'status':'complete',
        'label':'Smoke embedding computation' if args.smoke else 'Fresh embedding computation',
        'group_id':record['cohort_id']+'/'+name,'cohort_id':record['cohort_id']}
        for name,rows in values.items() for r in rows]
    return record

def main(argv=None):
    args=parse_args(argv)
    os.environ.update(JAX_PLATFORMS=args.platform,CUDA_VISIBLE_DEVICES=args.gpu or '',
        JAX_ENABLE_X64=str(args.x64),JAX_THREEFRY_PARTITIONABLE='False',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',MPLBACKEND='Agg')
    from paper._runtime import save_json,source_manifest,utc_now,environment
    root=Path(__file__).resolve().parent.parent
    config={k:v for k,v in vars(args).items() if k not in ('output','gpu')}
    record={'schema_version':1,'record_type':'paper_embedding_run','fresh_experiment':True,
        'smoke':args.smoke,'status':'running','started_at':utc_now(),'configuration':config,
        'source':source_manifest(root),'environment':environment(),
        'provenance':{'notebook':'archive/original/Gromov Embeddings E1.ipynb',
            'notebook_sha256':NOTEBOOK_SHA256,'cell':5,'arithmetic':'Recovered two diagnostic functions; raw repetition values additionally retained.',
            'precision_note':'The notebook does not enable x64; default execution explicitly uses float32. The original notebook kernel environment is not reconstructed.'}}
    record['cohort_id']=hashlib.sha256(json.dumps({'config':config,'source':record['source'],
        'environment':record['environment']},sort_keys=True).encode()).hexdigest()[:16]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_json(args.output,record)
    try:
        compute(args,record)
        record.update(status='complete',finished_at=utc_now())
        save_json(args.output,record)
    except Exception as exc:
        record.update(status='failed',error=type(exc).__name__+': '+str(exc),finished_at=utc_now())
        save_json(args.output,record)
        raise
    print(args.output)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
