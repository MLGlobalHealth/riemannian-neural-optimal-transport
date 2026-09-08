"""Planner and dispatch guards; no accelerator or scientific stack is imported."""
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import paper
from paper import _runtime as runtime
from paper import run


class PlannerTests(unittest.TestCase):
    def test_complete_prescribed_grids_and_deduplication(self):
        counts = {"tables": 30, "liegroups": 14, "sweep": 126, "highd": 12, "ablations": 84}
        for suite, expected in counts.items():
            with self.subTest(suite=suite):
                recipe, _ = run.load_recipe(suite)
                jobs = run.declare_jobs(recipe)
                self.assertEqual(len(jobs), expected)
                self.assertEqual(len({job["job_id"] for job in jobs}), expected)
        jobs = run.declare_jobs(run.load_recipe("tables")[0])
        self.assertEqual(sum(job["selection"]["method"] == "rcpm" for job in jobs), 10)
        self.assertEqual({job["selection"]["seed"] for job in jobs}, {12345, 23456, 34567, 45678, 56789})

    def test_single_native_se3_and_explicit_smoke_selection(self):
        plan = run.make_plan("liegroups", "/tmp/unused", ["2"], manifolds=["SE3"], methods=["ours"])
        self.assertEqual(len(plan["jobs"]), 1)
        self.assertEqual(plan["jobs"][0]["job_id"], "liegroups_SE3_ours_fps_seed12345")
        self.assertEqual(plan["recipe"]["rnot_overrides"], {})
        self.assertFalse(plan["smoke"])
        smoke = run.make_plan("tables", "/tmp/unused", ["0"], smoke=True, platform="cpu",
            manifolds=["S2"], methods=["ours"], landmarks=["random"], seeds=[12345])
        self.assertEqual(len(smoke["jobs"]), 1)
        self.assertTrue(smoke["smoke"])
        self.assertEqual(smoke["recipe"]["rnot_overrides"]["training"]["n_steps"], 1000)
        self.assertIn("not paper reproduction", smoke["interpretation"])

    def test_invalid_gpu_cpu_and_empty_selection_rejected(self):
        for gpus in ([], ["0", "0"], ["all"], ["-1"]):
            with self.assertRaises(ValueError):
                run.make_plan("tables", "/tmp/unused", gpus)
        with self.assertRaises(ValueError):
            run.make_plan("tables", "/tmp/unused", ["0"], platform="cpu")
        with self.assertRaises(ValueError):
            run.make_plan("tables", "/tmp/unused", ["0"], manifolds=["SO3"])

    def test_recipe_cannot_redefine_native_evaluation_or_liegroup_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recipe.json"
            recipe, _ = run.load_recipe("liegroups")
            for mutate in (
                lambda data: data["evaluation"].update(n_batches=2),
                lambda data: data.update(seeds=[23456]),
                lambda data: data.update(landmark_methods=["random"]),
                lambda data: data.update(rnot_overrides={"training": {"n_steps": 50}}),
            ):
                data = json.loads(json.dumps(recipe)); mutate(data)
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    run.load_recipe("liegroups", path)

    def test_dry_run_has_no_output_or_scientific_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "never-created"
            code = "from paper.run import main; import sys; main(sys.argv[1:]); assert 'jax' not in sys.modules"
            result = subprocess.run([sys.executable, "-B", "-c", code, "tables", "--gpus", "0", "1",
                "--output", str(output), "--dry-run"], text=True, capture_output=True,
                cwd=Path(paper.__file__).resolve().parent.parent, check=True)
            self.assertEqual(len(json.loads(result.stdout)["jobs"]), 30)
            self.assertFalse(output.exists())

    def test_fresh_output_required_before_interpreter_or_jobs(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(run, "inspect_interpreter") as inspect:
            with self.assertRaises(FileExistsError):
                run.run_plan({"output_dir": temporary})
            inspect.assert_not_called()

    def test_queue_failure_preserves_active_job_and_unstarted_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("src", "experiments", "rcpm", "paper"):
                (root / name).mkdir()
            output = root / "runs"
            jobs = []
            # First slot fails while the second runs. Third must remain unstarted.
            for index, delay, code in ((0, .02, 3), (1, .15, 0), (2, 0, 0)):
                program = "import json,time,sys; from pathlib import Path; " + f"assert len(json.loads(Path({str(output / 'manifest.json')!r}).read_text())['jobs'])==3; time.sleep({delay}); print('finished'); sys.exit({code})"
                jobs.append({"job_id": str(index), "log": str(output / "logs" / f"{index}.log"),
                    "command_without_gpu": [sys.executable, "-B", "-c", program]})
            plan = {"output_dir": str(output), "source_root": str(root), "python": sys.executable,
                "gpus": ["0", "1"], "environment_overrides": {}, "jobs": jobs}
            with mock.patch.object(run, "inspect_interpreter", return_value={"test": True}):
                self.assertEqual(run.run_plan(plan, poll_seconds=.01), 1)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual([job["status"] for job in status["jobs"]], ["failed", "complete", "not_started"])
            self.assertFalse((output / "logs" / "2.log").exists())


class RuntimeProtocolTests(unittest.TestCase):
    def test_source_manifest_ignores_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir(); (root / "src" / "actual.py").write_text("x=1")
            (root / "archive" / "src").mkdir(parents=True)
            (root / "archive" / "src" / "old.py").write_text("old=True")
            self.assertEqual(set(runtime.source_manifest(root)["files"]), {"src/actual.py"})

    def test_table_offset_and_ablation_fixed_evaluation_seed(self):
        for suite in ("tables", "ablations", "liegroups"):
            recipe = run.load_recipe(suite)[0]
            selection = {"suite": suite, "seed": 23456}
            cfg = runtime.evaluation_recipe(selection, recipe)
            self.assertEqual(cfg["seed"], 24456 if suite == "tables" else 12345)
            self.assertEqual((cfg["n_batches"], cfg["batch_size"]), (5, 1024))
            smoke = runtime.evaluation_recipe(selection, recipe, smoke=True)
            self.assertEqual((smoke["n_batches"], smoke["batch_size"]), (1, 8))

    def test_smoke_is_applied_after_large_config_and_preserves_optimizer(self):
        @dataclasses.dataclass
        class Training:
            n_steps: int = 1000
            batch_size: int = 512
            log_every: int = 1
            eval_every: object = None
            learning_rate: float = .001
            lr_decay: bool = True
            lr_decay_alpha: float = .05
        @dataclasses.dataclass
        class Solver:
            inner_steps: int = 2500
            min_steps: int = 1000
            use_adam: bool = True
            use_line_search: bool = False
        @dataclasses.dataclass
        class Config:
            training: Training
            solver: Solver
        state = NS(params="native_initial_params")
        module = NS(ArgminSolver=lambda **kw: NS(**kw), SemiDualLoss=lambda **kw: NS(**kw),
            SemiDualTrainer=lambda **kw: NS(**kw, init_state=lambda params: NS(params=params)))
        exp = {"cfg": Config(Training(), Solver()), "state": state, "manifold": "m", "psi": "p"}
        runtime._make_smoke(module, exp)
        self.assertEqual((exp["cfg"].training.n_steps, exp["cfg"].training.batch_size), (2, 8))
        self.assertEqual((exp["cfg"].solver.inner_steps, exp["cfg"].solver.min_steps), (2, 2))
        self.assertTrue(exp["solver"].use_adam)
        self.assertFalse(exp["solver"].use_line_search)
        self.assertEqual(exp["trainer"].learning_rate, .001)
        self.assertTrue(exp["trainer"].lr_decay)
        self.assertEqual(exp["state"].params, "native_initial_params")

    def test_native_split2_dispatch_and_retained_se3_metric(self):
        calls = []
        def evaluate(exp, key, batch_size):
            calls.append((key, batch_size)); return {"kl": 1., "ess_ratio": .5, "mean_residual": .01}
        m = NS(jax=NS(random=NS(PRNGKey=lambda seed: seed, split=lambda key, n=2: tuple(key * 10+i for i in range(1, n+1)))),
            np=NS(asarray=lambda key: NS(tolist=lambda: key)), evaluate=evaluate)
        batches = list(runtime.evaluation_batches(m, {}, {"suite": "liegroups", "method": "ours", "manifold": "SE3"},
            {"seed": 1, "n_batches": 2, "batch_size": 1024}))
        self.assertEqual(calls, [(12, 1024), (112, 1024)])
        self.assertEqual([b["prng_key"] for b in batches], [12, 112])
        self.assertIn("mean_residual", batches[0]["metrics"])

    def test_ablation_split3_and_native_loss_no_extra_diagnostic(self):
        solves, losses = [], []
        def solve(params, xs, targets):
            solves.append((xs, targets)); return "ys", None
        def loss(params, xs, targets):
            losses.append((xs, targets)); return .5
        m = NS(jax=NS(random=NS(PRNGKey=lambda seed: seed, split=lambda key, n=2: tuple(key * 10+i for i in range(1, n+1)))),
            np=NS(asarray=lambda key: NS(tolist=lambda: key)), compute_kl_divergence=lambda **kw: NS(kl=2., ess_ratio=.6))
        exp = {"base": NS(sample=lambda key, n: ("source", key, n)), "target": NS(sample=lambda key, n: ("target", key, n)),
            "solver": NS(batch_solve=solve), "trainer": NS(loss_fn=loss), "state": NS(params="p"), "manifold": "m", "psi": "p"}
        batches = list(runtime.evaluation_batches(m, exp, {"suite": "ablations"}, {"seed": 1, "n_batches": 2, "batch_size": 1024}))
        self.assertEqual([(b["prng_key"], b["source_key"], b["target_key"]) for b in batches], [(1, 12, 13), (11, 112, 113)])
        self.assertEqual(len(solves), 2)
        self.assertEqual(solves, losses)
        self.assertEqual(batches[0]["metrics"], {"kl": 2., "ess_ratio": .6, "loss": .5})


if __name__ == "__main__":
    unittest.main()
