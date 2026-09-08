"""CPU-free input and restoration tests; no JAX imports or real evaluations."""
import copy
import dataclasses
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('liegroup_replay_under_test', Path(__file__).with_name('evaluate_liegroup_checkpoint.py'))
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def saved_record(method='ours', manifold='SE3'):
    runner = f'experiments/run_{manifold.lower()}_experiment.py'
    return {'record_type': 'native_liegroup_training_run', 'status': 'complete',
        'label': 'recovered_liegroup_native_cuda_jax435', 'explicit_configuration_overrides': {},
        'selection': {'manifold': manifold, 'method': method, 'gamma': 1.0 if method == 'rcpm' else None,
                      'landmark_method': 'fps' if method == 'ours' else None, 'seed': 12345},
        'source_unchanged_after_run': True, 'runner_sha256': e.WRAPPER_SHA, 'helper_sha256': e.HELPER_SHA,
        'source': {'sha256': 'source', 'files': {runner: 'native'}},
        'native_runner': {'file': runner, 'sha256': 'native'},
        'evaluation_config': {'seed': 12345, 'batch_size': 1024, 'n_batches': 5, 'seed_offset': None},
        'evaluation_batches': [{'index': i, 'prng_key': [i, i+1], 'metrics': {'kl': 1., 'ess': 512., 'ess_ratio': .5}} for i in range(5)],
        'environment': {'jax_enable_x64': True, 'jax_threefry_partitionable': False,
                        'jax_default_prng_impl': 'threefry2x32', 'packages': dict(e.CORE), 'all_package_versions': dict(e.CORE)}}


@dataclasses.dataclass
class Model:
    n_landmarks: int = 128


@dataclasses.dataclass
class Solver:
    inner_steps: int = 100
    line_search_steps: tuple = (.5, .1)


class State:
    def __init__(self, params): self.params = params
    def replace(self, **kwargs): return State(kwargs['params'])


class Psi:
    def __init__(self, phi): self.phi = phi
    def clone(self, **kwargs): return Psi(kwargs['phi'])


class Geometry:
    D = 7
    alpha = 1.


def fake_array(value, _np):
    is_lm = str(value).startswith('lm')
    return {'shape': [128, 7] if is_lm else [2], 'dtype': 'float64' if is_lm else 'float32',
            'size': 896 if is_lm else 2, 'sha256': str(value), 'nonfinite_count': 0}


def restoration_fixture(method, manifold):
    saved = saved_record(method, manifold)
    geometry = Geometry()
    cfg = NS(model=Model(), solver=Solver())
    exp = {'manifold': geometry, 'base': 'base', 'target': 'target', 'cfg': cfg,
           'model_cfg': cfg.model, 'solver_cfg': cfg.solver, 'psi': Psi(NS(landmarks='lm-initial')),
           'state': State({'weight': 'initial'}), 'params': {'params': {'weight': 'initial'}},
           'flow': object(), 'solver': object(), 'trainer': object()}
    wrapper = NS(evidence=NS(json_safe=lambda x: x), native_configuration=Mock(return_value={'native': True}),
                 density_configuration=lambda x: {'class': x})
    saved.update(configuration={'native': True}, density_configuration={'base': {'class': 'base'}, 'target': {'class': 'target'}},
        geometry_configuration={'class': 'Geometry', 'D': 7, 'alpha': 1.},
        landmark_configuration={'method': 'fps', 'count': 128, 'fps_candidates_per_density': 4096},
        parameter_count=2, parameter_dtypes=['float32'], parameter_nonfinite_count=0)
    payload = {'params': {'weight': 'trained'}, 'state': {'params': {'weight': 'trained'}}, 'landmarks': 'lm-saved'}
    if method == 'rcpm': payload = {'params': {'params': {'weight': 'trained'}}, 'key': [1, 2]}
    def tree_map(fn, x):
        return {k: tree_map(fn, v) for k, v in x.items()} if isinstance(x, dict) else fn(x)
    module = NS(np=None, jnp=NS(asarray=lambda x: x), jax=NS(tree_util=NS(tree_map=tree_map)),
                build_rcpm_experiment=Mock(return_value=exp), build_ours_experiment=Mock(return_value=exp),
                build_experiment=Mock(return_value=exp), LANDMARK_METHOD='fps', FPS_CANDIDATES=4096,
                GromovDistanceEmbedding=lambda **kw: NS(**kw), ArgminSolver=Mock(side_effect=lambda **kw: NS(**kw)))
    return module, wrapper, saved, payload, exp


class ReplayTests(unittest.TestCase):
    def test_record_guards_preserve_four_models_and_reject_replays_or_changed_protocol(self):
        for manifold in ('SO3', 'SE3'):
            for method in ('ours', 'rcpm'): e.validate_record(saved_record(method, manifold))
        cases = [('record_type', 'liegroup_checkpoint_evaluation'), ('status', 'failed'), ('runner_sha256', 'changed')]
        for key, value in cases:
            saved = saved_record(); saved[key] = value
            with self.assertRaises(ValueError): e.validate_record(saved)
        for mutator in (lambda s: s['evaluation_batches'].pop(),
                        lambda s: s['evaluation_batches'][0].update(prng_key=[True, 1]),
                        lambda s: s['selection'].update(gamma=.1),
                        lambda s: s['evaluation_config'].update(seed=13345),
                        lambda s: s['environment'].update(jax_threefry_partitionable=True)):
            saved = saved_record(); mutator(saved)
            with self.assertRaises(ValueError): e.validate_record(saved)

    def test_all_pins_and_both_original_and_replay_runtime_are_checked(self):
        saved = saved_record(); pins = {**e.CORE, 'ml-dtypes': '.6'}
        actual = {**e.CORE, 'ml_dtypes': '.6'}
        saved['environment']['all_package_versions'] = actual
        e.validate_packages(saved, actual, pins, (3, 11, 15))
        for bad, python in (({**actual, 'jax': '.6'}, (3, 11)), (actual, (3, 12))):
            with self.assertRaises(ValueError): e.validate_packages(saved, bad, pins, python)
        saved['environment']['all_package_versions'] = {**actual, 'ml_dtypes': '.7'}
        with self.assertRaises(ValueError): e.validate_packages(saved, actual, pins, (3, 11))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pins.txt'; path.write_text('jax==0.4.35\n')
            with self.assertRaises(ValueError): e.read_pins(path)

    def test_rnot_restores_landmarks_state_and_fresh_solver_for_both_native_builders(self):
        for manifold in ('SO3', 'SE3'):
            module, wrapper, saved, payload, exp = restoration_fixture('ours', manifold)
            old_solver, old_psi, old_trainer = exp['solver'], exp['psi'], exp['trainer']
            with patch.object(e, 'array_fingerprint', fake_array):
                rebuilt, fp = e.restore_experiment(module, wrapper, saved, payload)
            self.assertIsNot(rebuilt['solver'], old_solver)
            self.assertIsNot(rebuilt['psi'], old_psi)
            self.assertIs(rebuilt['solver'].psi_module, rebuilt['psi'])
            self.assertEqual(rebuilt['psi'].phi.landmarks, 'lm-saved')
            self.assertEqual(rebuilt['state'].params, payload['params'])
            self.assertEqual(rebuilt['solver'].line_search_steps, (.5, .1))
            self.assertIs(rebuilt['trainer'], old_trainer)
            self.assertEqual(fp['landmarks']['sha256'], 'lm-saved')
            (module.build_ours_experiment if manifold == 'SO3' else module.build_experiment).assert_called_once_with()

    def test_rcpm_restores_entire_variables_tree_without_replacing_flow(self):
        module, wrapper, saved, payload, exp = restoration_fixture('rcpm', 'SO3')
        flow = exp['flow']
        with patch.object(e, 'array_fingerprint', fake_array):
            rebuilt, fp = e.restore_experiment(module, wrapper, saved, payload)
        self.assertEqual(rebuilt['params'], {'params': {'weight': 'trained'}})
        self.assertIn('params/params/weight', fp['parameters'])
        self.assertIs(rebuilt['flow'], flow)
        module.ArgminSolver.assert_not_called()
        module.build_rcpm_experiment.assert_called_once_with(gamma=1., seed=12345)

    def test_restoration_rejects_config_payload_state_and_dtype_drift(self):
        for alteration in ('config', 'state', 'dtype', 'density'):
            module, wrapper, saved, payload, exp = restoration_fixture('ours', 'SE3')
            if alteration == 'config': wrapper.native_configuration.return_value = {'changed': True}
            if alteration == 'state': payload['state']['params']['weight'] = 'wrong'
            if alteration == 'dtype': saved['parameter_dtypes'] = ['float64']
            if alteration == 'density': saved['density_configuration']['base']['class'] = 'wrong'
            with patch.object(e, 'array_fingerprint', fake_array), self.assertRaises(ValueError):
                e.restore_experiment(module, wrapper, saved, payload)

    def test_hash_and_source_guards_reject_tampering_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint'; checkpoint.write_bytes(b'fixed')
            helper_file = root / 'helper'; helper_file.write_bytes(b'helper')
            wrapper_file = root / 'wrapper'; wrapper_file.write_bytes(b'wrapper')
            saved = saved_record(); saved['checkpoint'] = {'sha256': e.sha256(checkpoint), 'bytes': 5}
            helper = NS(source_manifest=Mock(return_value=saved['source']))
            with patch.object(e, 'HELPER_SHA', e.sha256(helper_file)), patch.object(e, 'WRAPPER_SHA', e.sha256(wrapper_file)):
                e.verify_inputs(saved, root, checkpoint, helper_file, wrapper_file, helper)
                with self.assertRaises(ValueError): e.verify_inputs(saved, root, checkpoint, helper_file, wrapper_file, helper, 'wrong')
                helper.source_manifest.return_value = {'changed': True}
                with self.assertRaises(ValueError): e.verify_inputs(saved, root, checkpoint, helper_file, wrapper_file, helper)
                helper.source_manifest.return_value = saved['source']; checkpoint.write_bytes(b'other')
                with self.assertRaises(ValueError): e.verify_inputs(saved, root, checkpoint, helper_file, wrapper_file, helper)


if __name__ == '__main__': unittest.main()
