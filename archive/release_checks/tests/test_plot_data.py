"""Meaningful offline data/plot checks. No JAX or experiment execution."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import paper
from paper._plot_data import load_input,SEEDS
from paper.plot import reference_rows,curve_identity


def record(seed=12345,suite='tables',method='ours',host='host-a',source='source',gpu='0',value=1.):
    return {'record_type':'paper_training_run','status':'complete','label':'native','source':{'sha256':source},
        'runner_sha256':'runner','helper_sha256':'helper','configuration':{'seed':seed,'training':{'seed':seed,'steps':100},'model':{'n_landmarks':128}},
        'selection':{'suite':suite,'manifold':'S2','dimension':2,'method':method,'gamma':1. if method=='rcpm' else None,'seed':seed,'landmark_method':'fps' if method=='ours' else None,'ablation':None},
        'environment':{'hostname':host,'packages':{'jax':'.4.35'},'environment':{'CUDA_VISIBLE_DEVICES':gpu},'nvidia_smi':f'{gpu}, Model, UUID{gpu}, Driver, 100 MiB'},
        'evaluation_config':{'seed':seed+1000,'batch_size':1024,'n_batches':5},
        'evaluation_batches':[{'index':i,'prng_key':[i,1],'metrics':{'kl':value+i*.1,'ess_ratio':.8+i*.01}} for i in range(5)]}


def load(records):
    with tempfile.TemporaryDirectory() as directory:
        for i,r in enumerate(records):
            path=Path(directory)/str(i);path.mkdir();(path/'result.json').write_text(json.dumps(r))
        return load_input(directory)


class PlotInputTests(unittest.TestCase):
    def test_five_seed_aggregation_and_native_single_model_batch_se_are_distinct(self):
        for method in ('ours','rcpm'):
            rows,_=load([record(seed=s,method=method,gpu=str(i%2),value=float(i)) for i,s in enumerate(SEEDS)])
            self.assertEqual(len(rows),1);row=rows[0];self.assertEqual(row['status'],'complete_five_seeds')
            self.assertAlmostEqual(row['kl'],2.2);self.assertAlmostEqual(row['kl_se'],math.sqrt(2/5))
        r=record(suite='liegroups');r['selection']['manifold']='SO3'
        rows,_=load([r]);self.assertEqual(rows[0]['n_training_runs'],1)
        self.assertAlmostEqual(rows[0]['kl_se'],math.sqrt(.02/5))

    def test_partial_configs_hosts_sources_never_fill_a_seed_group(self):
        for change in ('host','source','config','wrapper'):
            records=[record(seed=s,value=float(i)) for i,s in enumerate(SEEDS)]
            for r in records[3:]:
                if change=='host':r['environment']['hostname']='host-b'
                if change=='source':r['source']['sha256']='other'
                if change=='config':r['configuration']['model']['n_landmarks']=256
                if change=='wrapper':r['paper_source']={'paper/worker.py':'other'}
            rows,_=load(records);self.assertEqual(len(rows),2)
            self.assertTrue(all(r['status']=='partial_or_failed_seed_group' for r in rows))
            self.assertTrue(all(math.isnan(r['kl']) for r in rows))

    def test_duplicate_copy_is_alias_but_distinct_execution_is_ambiguous(self):
        records=[record(seed=s) for s in SEEDS];rows,notes=load(records+[copy.deepcopy(records[0])])
        self.assertEqual(rows[0]['status'],'complete_five_seeds');self.assertTrue(notes)
        extra=copy.deepcopy(records[0]);extra['finished_at']='different execution'
        rows,_=load(records+[extra]);self.assertEqual(rows[0]['status'],'ambiguous_duplicate_seed')
        r=record(suite='sweep');rows,_=load([r,{**r,'finished_at':'another'}])
        self.assertTrue(all(row['status']=='ambiguous_duplicate_execution' for row in rows))

    def test_failure_nonfinite_smoke_and_replay_are_not_measured_zero(self):
        records=[record(seed=s) for s in SEEDS];records[0]['evaluation_batches'][0]['metrics']['kl']='NaN'
        rows,_=load(records);self.assertTrue(math.isnan(rows[0]['kl']))
        r=record(suite='sweep');r['status']='failed';r['error']={'message':'build failed'};r['evaluation_batches']=[]
        replay={**record(),'evaluation_only':True,'record_type':'liegroup_checkpoint_evaluation'}
        smoke={**record(),'smoke':True}
        rows,notes=load([r,replay,smoke]);self.assertEqual(len(rows),1);self.assertEqual(rows[0]['status'],'failed')
        self.assertTrue(math.isnan(rows[0]['kl']));self.assertEqual(len(notes),2)

    def test_references_have_full_grids_and_retain_signed_or_zero_measurements(self):
        root=Path(paper.__file__).resolve().parent.parent/'reference'
        rows,_=reference_rows('sweep',root);self.assertEqual(sum(r['suite']=='sweep' for r in rows),126)
        self.assertEqual(sum(r['suite']=='highd' for r in rows),12)
        rows,_=reference_rows('ablations',root);self.assertEqual(len(rows),80)
        blank=[r for r in rows if r['manifold']=='T10' and r['ablation']=='landmarks_256'][0]
        self.assertLess(blank['kl'],0);self.assertIsNone(blank['paper_kl'])
        rows,_=reference_rows('embeddings',root);self.assertEqual(len(rows),18)
        self.assertTrue(all(r['status']=='historical_rounded' for r in rows))
        self.assertTrue(any(r['near_collision_fraction']==0 for r in rows))
        rows,_=reference_rows('liegroups',root);self.assertEqual(len(rows),14)
        self.assertEqual(sum(r['status']=='paper_reported' for r in rows),9)


if __name__=='__main__':unittest.main()
