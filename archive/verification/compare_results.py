#!/usr/bin/env python3
"""Aggregate recorded ZIP table experiments without pooling source or environments.

Example:
  python compare_results.py --results-root ../results --paper-targets ../paper_targets.json --output-dir ../comparison

Uses only Python's standard library. Smoke runs never enter numerical comparisons.
Printed paper rounding is a descriptive check, not a statistical equivalence test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

METRICS = ("kl", "ess", "ess_ratio")
SHIPPED_SEEDS = [12345, 23456, 34567, 45678, 56789]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def short_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:16]


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def environment_identity(record):
    env = deepcopy(record.get("environment", {}))
    variables = env.get("environment", {})
    visible = variables.pop("CUDA_VISIBLE_DEVICES", None)
    gpu_output = env.pop("nvidia_smi", None)
    devices = env.pop("jax_devices", None)
    gpu_rows = []
    if isinstance(gpu_output, str):
        for line in gpu_output.splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) == 5 and fields[0].isdigit():
                gpu_rows.append(fields)
    if gpu_rows:
        tokens = {x.strip() for x in str(visible).split(",")} if visible else None
        selected = [r for r in gpu_rows if tokens is None or r[0] in tokens or r[2] in tokens]
        # Retain an unknown selector verbatim so dissimilar accelerators are not pooled.
        if not selected:
            env["unresolved_gpu_selector"] = visible
            selected = gpu_rows
        env["selected_accelerator_types"] = sorted({(r[1], r[3], r[4]) for r in selected})
    else:
        env["accelerator_description"] = gpu_output
        env["jax_devices"] = devices
        if visible:
            env["unresolved_gpu_selector"] = visible
    return env


def group_identity(record):
    selection = {k: v for k, v in record.get("selection", {}).items() if k != "seed"}
    config = deepcopy(record.get("configuration"))
    if isinstance(config, dict):
        config.pop("seed", None)
        if isinstance(config.get("training"), dict):
            config["training"].pop("seed", None)
    evaluation = deepcopy(record.get("evaluation_config", {}))
    eval_seed = evaluation.pop("seed", None)
    overrides = record.get("explicit_overrides", {})
    if "seed" in overrides.get("evaluation", {}):
        evaluation["seed_rule"] = {"fixed_explicit_seed": eval_seed}
    elif selection.get("suite") == "table":
        seed = number(record.get("selection", {}).get("seed"))
        evaluation["seed_rule"] = {"training_seed_offset": number(eval_seed) - seed
                                   if number(eval_seed) is not None and seed is not None else None}
    else:
        evaluation["seed_rule"] = {"fixed_archive_seed": eval_seed}
    return {
        "selection": selection, "label": record.get("label"),
        "source_sha256": record.get("source", {}).get("sha256"),
        "runner_sha256": record.get("runner_sha256"), "builder": record.get("builder"),
        "configuration_without_seed": config, "explicit_overrides": overrides,
        "evaluation_recipe": evaluation, "environment": environment_identity(record),
        "density_configuration": record.get("density_configuration"),
        "geometry_configuration": record.get("geometry_configuration"),
        "landmark_configuration": record.get("landmark_configuration"),
        "parameter_dtypes": record.get("parameter_dtypes"),
    }


def validate_record(record):
    errors = []
    if record.get("status") != "complete":
        errors.append({"kind": "unfinished_or_failed", "status": record.get("status"),
                       "error": record.get("error")})
        return None, errors
    batches = record.get("evaluation_batches", [])
    expected_batches = record.get("evaluation_config", {}).get("n_batches")
    if not batches or len(batches) != expected_batches:
        errors.append({"kind": "evaluation_batch_count", "actual": len(batches), "expected": expected_batches})
    means = {}
    for metric in METRICS:
        values = [number(batch.get("metrics", {}).get(metric)) for batch in batches]
        if not values or any(v is None for v in values):
            errors.append({"kind": "nonfinite_metric", "metric": metric})
            continue
        mean = statistics.fmean(values)
        reported = number(record.get("summary", {}).get(metric, {}).get("mean"))
        if reported is None:
            errors.append({"kind": "nonfinite_summary", "metric": metric})
        elif not math.isclose(reported, mean, rel_tol=1e-12, abs_tol=1e-12):
            errors.append({"kind": "summary_inconsistency", "metric": metric,
                           "reported": reported, "recomputed": mean})
        means[metric] = mean
        if any(batch.get("metric_nonfinite", {}).get(metric, False) for batch in batches):
            errors.append({"kind": "nonfinite_metric_flag", "metric": metric})
    if record.get("parameter_nonfinite_count", 0):
        errors.append({"kind": "nonfinite_parameters", "count": record["parameter_nonfinite_count"]})
    for batch in batches:
        diagnostic = batch.get("residual_diagnostics", {})
        for field in ("nonfinite_count", "transport_nonfinite_count"):
            if diagnostic.get(field, 0):
                errors.append({"kind": "nonfinite_diagnostic", "batch": batch.get("index"),
                               "field": field, "count": diagnostic[field]})
    return means, errors


def aggregate(values):
    if not values:
        return {"n": 0, "mean": None, "se_ddof0": None, "se_ddof1": None, "raw_values": []}
    n = len(values)
    return {"n": n, "mean": statistics.fmean(values),
            "se_ddof0": statistics.pstdev(values) / math.sqrt(n),
            "se_ddof1": statistics.stdev(values) / math.sqrt(n) if n > 1 else None,
            "raw_values": values}


def target_for(identity, paper):
    s = identity["selection"]
    if s.get("suite") != "table" or s.get("dimension") != 2:
        return None, None
    manifold = "S2" if s.get("manifold") == "sphere" else "T2"
    if s.get("method") == "ours":
        method = {"fps": "RNOT_FPS", "random": "RNOT_RND"}.get(s.get("landmark_method"))
    else:
        method = "RCPM_gamma1" if number(s.get("gamma")) == 1.0 else None
    for table in paper["tables"]:
        if table["number"] in (1, 2) and table.get("manifold") == manifold:
            for row in table["rows"]:
                if row["method"] == method:
                    return table, row
    return None, None


def rounding_check(observed, reported, digits):
    half = Decimal(5).scaleb(-digits - 1)
    center = Decimal(str(reported))
    lower, upper = center - half, center + half
    check = None if observed is None else lower <= Decimal(str(observed)) < upper
    return {"printed_mean": reported, "decimal_places": digits,
            "rounding_interval": {"lower_inclusive": str(lower), "upper_exclusive": str(upper)},
            "observed_mean_in_rounding_interval": check,
            "boundary_note": "At exact half-unit ties, original rounding convention is unspecified.",
            "difference_from_printed_mean": observed - reported if observed is not None else None,
            "relative_difference_percent": 100 * (observed - reported) / abs(reported)
            if observed is not None and reported != 0 else None}


def summarize_group(identity, members, paper):
    by_seed = defaultdict(list)
    for path, record in members:
        by_seed[record.get("selection", {}).get("seed")].append((path, record))
    runs, valid = [], []
    duplicates = []
    for seed, seed_members in sorted(by_seed.items(), key=lambda x: str(x[0])):
        if len(seed_members) != 1:
            duplicates.append(seed)
        for path, record in seed_members:
            means, errors = validate_record(record)
            if len(seed_members) != 1:
                errors.append({"kind": "duplicate_seed", "copies": len(seed_members)})
            run = {"seed": seed, "task_id": record.get("task_id"), "path": str(path),
                   "status": record.get("status"), "valid_for_aggregation": not errors,
                   "errors": errors, "metrics": means, "timings_seconds": record.get("timings_seconds"),
                   "residual_diagnostics": [b.get("residual_diagnostics") for b in record.get("evaluation_batches", [])
                                            if b.get("residual_diagnostics") is not None]}
            runs.append(run)
            if not errors:
                valid.append(run)
    valid.sort(key=lambda r: r["seed"])
    stats = {metric: aggregate([run["metrics"][metric] for run in valid]) for metric in METRICS}
    timings = {}
    for name in ("training_native_return", "training_synchronized"):
        values = [number((r.get("timings_seconds") or {}).get(name)) for r in valid]
        timings[name] = aggregate([value for value in values if value is not None])
    failed = [run for run in runs if run["errors"]]
    n = len(valid)
    complete = n == 5 and not failed and not duplicates
    seeds = [r["seed"] for r in valid]
    comparison = None
    table, target = target_for(identity, paper)
    if target:
        checks = {}
        for metric in ("kl", "ess_ratio"):
            digits = (4 if metric == "kl" else 3) if table["number"] == 1 and target["method"] == "RCPM_gamma1" else 2
            checks[metric] = rounding_check(stats[metric]["mean"], target[metric]["mean"], digits)
            checks[metric]["paper_reported_plus_minus"] = target[metric]["reported_plus_minus"]
            checks[metric]["uncertainty_interpretation"] = "Confidence level/construction unspecified; neither observed SE is assumed equivalent to paper uncertainty."
        native_protocol = identity.get("builder") == "native_archive_builder" and not any(identity.get("explicit_overrides", {}).values())
        comparison = {"table": table["number"], "manifold": table["manifold"], "method": target["method"],
                      "scope": "archive_defaults" if native_protocol else "explicit_configuration_descriptive_comparison",
                      "complete_five_seed_comparison": complete,
                      "native_default_five_seed_comparison": complete and native_protocol,
                      "configured_five_seed_comparison": complete and not native_protocol,
                      "numerical_checks": checks,
                      "both_metric_means_round_as_printed": all(c["observed_mean_in_rounding_interval"] is True for c in checks.values()) if n else None,
                      "paper_training_seconds": target["training_seconds"],
                      "timing_comparison": "Descriptive only: paper hardware AMD MI300X 192GB; device/software and fresh-process compilation affect runtime.",
                      "interpretation": "Rounded-mean agreement is descriptive, not proof of statistical equivalence or complete paper reproduction."}
    nonfinite = any(e["kind"].startswith("nonfinite") for run in runs for e in run["errors"])
    status = ("nonfinite_failure" if nonfinite else "duplicate_seed_conflict" if duplicates
              else "partial_with_failed_or_unfinished_runs" if failed else "complete_five_seeds" if complete
              else "partial_fewer_than_five_seeds" if n < 5 else "different_protocol_more_than_five_seeds")
    return {"group_id": short_hash(identity), "identity": identity, "status": status,
            "independent_valid_seed_count": n, "valid_seeds": seeds, "exact_shipped_seed_set": seeds == SHIPPED_SEEDS,
            "missing_shipped_seeds": sorted(set(SHIPPED_SEEDS) - set(seeds)),
            "duplicate_seeds": duplicates, "runs": runs, "across_seed_statistics": stats,
            "training_timings_seconds": timings, "paper_comparison": comparison}


def build_report(roots, paper, primary_host=None):
    files = sorted({p.resolve() for root in roots for p in ([root] if root.is_file() else root.rglob("result.json"))})
    buckets, excluded, errors, copies, seen = defaultdict(list), [], [], [], {}
    for path in files:
        try:
            record = json.loads(path.read_text())
            fingerprint = short_hash(record)
            if fingerprint in seen:
                copies.append({"path": str(path), "same_record_as": str(seen[fingerprint])})
                continue
            seen[fingerprint] = path
            if record.get("smoke") or record.get("label") == "smoke":
                excluded.append({"path": str(path), "reason": "smoke_run", "status": record.get("status")})
                continue
            identity = group_identity(record)
            buckets[canonical(identity)].append((path, record))
        except Exception as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    groups = [summarize_group(json.loads(key), members, paper) for key, members in buckets.items()]
    groups.sort(key=lambda g: canonical(g["identity"]))
    for group in groups:
        group["is_primary_environment"] = group["identity"]["environment"].get("hostname") == primary_host if primary_host else None
    environments = {}
    for group in groups:
        identity = group["identity"]
        key = short_hash({"environment": identity["environment"], "source": identity["source_sha256"],
                          "runner": identity["runner_sha256"], "label": identity["label"]})
        entry = environments.setdefault(key, {"environment": identity["environment"], "source_sha256": identity["source_sha256"],
                    "runner_sha256": identity["runner_sha256"], "label": identity["label"],
                    "complete_native_table_rows": [], "complete_configured_table_rows": [], "group_ids": []})
        entry["group_ids"].append(group["group_id"])
        comparison = group["paper_comparison"]
        if comparison and comparison["native_default_five_seed_comparison"]:
            entry["complete_native_table_rows"].append(comparison["manifold"] + ":" + comparison["method"])
        if comparison and comparison["configured_five_seed_comparison"]:
            entry["complete_configured_table_rows"].append(comparison["manifold"] + ":" + comparison["method"])
    expected = [m + ":" + method for m in ("S2", "T2") for method in ("RNOT_FPS", "RNOT_RND", "RCPM_gamma1")]
    for entry in environments.values():
        entry["missing_native_table_rows"] = sorted(set(expected) - set(entry["complete_native_table_rows"]))
        entry["all_six_archive_table_rows_complete"] = not entry["missing_native_table_rows"]
        entry["complete_table_rows"] = sorted(set(entry["complete_native_table_rows"] + entry["complete_configured_table_rows"]))
        entry["missing_table_rows"] = sorted(set(expected) - set(entry["complete_table_rows"]))
        entry["all_six_table_rows_complete_in_this_environment"] = not entry["missing_table_rows"]
    coverage = table_row_coverage(groups, environments, expected)
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "paper_title": paper["paper_title"], "version_assessment": paper.get("version_assessment"),
            "primary_host": primary_host, "input_roots": [str(p.resolve()) for p in roots],
            "input_result_files": len(files), "excluded_runs": excluded, "duplicate_file_copies": copies,
            "read_errors": errors, "groups": groups, "environment_completeness": environments,
            "table_row_coverage": coverage,
            "missing_implementations_and_coverage": paper.get("archive_coverage"),
            "comparison_policy": {
                "grouping": "Same suite, manifold/dimension, method/landmarks/gamma, label, source hash, runner hash, effective config, evaluation recipe, host and software/hardware environment. GPU index/UUID can differ only on the same host with matching model/driver/memory.",
                "independence": "A seed contributes once; conflicting repeated seeds are excluded, never selected by performance. Identical copied JSON files are deduplicated.",
                "uncertainty": "Each run mean averages its recorded evaluation batches; ddof0 and ddof1 SE are computed over independent training seeds. Paper confidence level is not invented.",
                "rounding": "Compare observed means to half-unit intervals at the printed precision. These checks do not establish statistical equivalence.",
                "smoke": "All smoke runs excluded, even when the explicit label differs.",
                "default_vs_override": "Explicit overrides are shown descriptively and cannot satisfy native-default table completeness. Complete configured groups count separately toward overall row coverage.",
                "row_coverage": "Each covered row requires a complete five-seed group. Rows may use different hosts, labels, sources or environments, which remain explicit. Partial groups are never pooled. Coverage does not establish numerical agreement or full paper reproduction.",
                "publication_readiness": "No automatic publication approval; missing experiments, source-version gaps, solver reliability and timing differences remain explicit."}}


def table_row_coverage(groups, environments, expected):
    """Index complete groups by paper row without aggregating across groups."""
    group_environments = {group_id: key for key, entry in environments.items() for group_id in entry["group_ids"]}
    rows = {row: {"complete_native_group_ids": [], "complete_configured_group_ids": [], "groups": []}
            for row in expected}
    for group in groups:
        comparison = group["paper_comparison"]
        if not comparison:
            continue
        row = comparison["manifold"] + ":" + comparison["method"]
        identity = group["identity"]
        entry = rows[row]
        if comparison["native_default_five_seed_comparison"]:
            entry["complete_native_group_ids"].append(group["group_id"])
        if comparison["configured_five_seed_comparison"]:
            entry["complete_configured_group_ids"].append(group["group_id"])
        entry["groups"].append({"group_id": group["group_id"], "status": group["status"],
            "scope": comparison["scope"], "complete_five_seed_comparison": comparison["complete_five_seed_comparison"],
            "valid_seeds": group["valid_seeds"], "independent_valid_seed_count": group["independent_valid_seed_count"],
            "hostname": identity["environment"].get("hostname"), "label": identity["label"],
            "source_sha256": identity["source_sha256"], "runner_sha256": identity["runner_sha256"],
            "environment_id": group_environments[group["group_id"]], "environment": identity["environment"]})
    for entry in rows.values():
        entry["covered_by_complete_five_seed_group"] = bool(entry["complete_native_group_ids"] or entry["complete_configured_group_ids"])
    native = [row for row in expected if rows[row]["complete_native_group_ids"]]
    configured = [row for row in expected if rows[row]["complete_configured_group_ids"]]
    covered = [row for row in expected if rows[row]["covered_by_complete_five_seed_group"]]
    return {"scope": "Six main-table rows implemented by the archive; row coverage may span environments without pooling seeds.",
            "expected_rows": expected, "complete_native_table_rows": native,
            "complete_configured_table_rows": configured, "complete_table_rows": covered,
            "missing_table_rows": [row for row in expected if row not in covered],
            "all_six_rows_have_complete_five_seed_group": len(covered) == len(expected), "rows": rows}


def relative_evidence_paths(report, output_dir):
    """Make derived evidence references portable without altering run identities."""
    result = deepcopy(report)
    base = output_dir.resolve()

    def relative(path):
        return Path(os.path.relpath(Path(path).resolve(), base)).as_posix()

    result["input_roots"] = [relative(path) for path in result["input_roots"]]
    for group in result["groups"]:
        for run in group["runs"]:
            run["path"] = relative(run["path"])
    for key in ("excluded_runs", "duplicate_file_copies", "read_errors"):
        for entry in result[key]:
            entry["path"] = relative(entry["path"])
            if "same_record_as" in entry:
                entry["same_record_as"] = relative(entry["same_record_as"])
    result["evidence_path_base"] = "output_directory"
    return result


def fmt(value):
    return "—" if value is None else f"{value:.6g}"


def markdown(report):
    lines = ["# Experimental verification comparison", "", f"Generated: {report['generated_at']}", "",
             "Smoke runs are excluded. Each environment and effective configuration is aggregated separately. Numerical agreement below refers to printed rounding; it does not establish statistical equivalence.", "",
             "These NVIDIA runs do not isolate hardware effects relative to the paper's AMD setup. Source, configuration, dependency versions, precision and PRNG behavior must also be aligned before attributing differences to a backend. Original AMD reproduction remains unverified.", "",
             "| Group | Host | Setting | Valid seeds | Status | KL mean ± shipped SE | ESS mean ± shipped SE | Paper mean rounding |",
             "|---|---|---|---:|---|---|---|---|---|"]
    for group in report["groups"]:
        identity = group["identity"]; selection = identity["selection"]; comparison = group["paper_comparison"]
        stats = group["across_seed_statistics"]
        setting = f"{selection.get('suite')} {selection.get('manifold')}{selection.get('dimension')} {selection.get('method')} {selection.get('landmark_method') or selection.get('gamma')} [{identity.get('label')}]"
        if comparison:
            outcomes = [f"{metric}: {'matches' if check['observed_mean_in_rounding_interval'] else 'differs' if check['observed_mean_in_rounding_interval'] is False else 'unavailable'}" for metric, check in comparison["numerical_checks"].items()]
            outcome = "; ".join(outcomes)
            if not comparison["complete_five_seed_comparison"]:
                outcome += " (partial)"
            if comparison["scope"] != "archive_defaults":
                outcome += " (configured)"
        else:
            outcome = "No main-table target"
        lines.append(f"| {group['group_id']} | {identity['environment'].get('hostname', 'unknown')} | {setting} | {group['independent_valid_seed_count']} | {group['status']} | {fmt(stats['kl']['mean'])} ± {fmt(stats['kl']['se_ddof0'])} | {fmt(stats['ess_ratio']['mean'])} ± {fmt(stats['ess_ratio']['se_ddof0'])} | {outcome} |")
    if not report["groups"]:
        lines += ["", "No eligible experiment records found."]
    lines += ["", "## Completeness", "",
              f"Read {report['input_result_files']} result files; excluded {len(report['excluded_runs'])} smoke runs, deduplicated {len(report['duplicate_file_copies'])} identical copies, and encountered {len(report['read_errors'])} read errors."]
    if report.get("primary_host"):
        lines += ["", f"Primary host selected: `{report['primary_host']}`. Other hosts remain separate sensitivity checks."]
    coverage = report["table_row_coverage"]
    lines += ["", f"Overall row coverage: {len(coverage['complete_table_rows'])}/6 rows have a complete five-seed group.",
              f"Missing rows across the provided groups: {', '.join(coverage['missing_table_rows']) or 'none'}.", "",
              "Native and configured groups count separately. Rows may use different hosts, labels, sources and environments; partial groups are never combined. Coverage does not establish numerical agreement or full paper reproduction.", "",
              "| Row | Complete native groups | Complete configured groups |",
              "|---|---|---|"]
    for row, entry in coverage["rows"].items():
        lines.append(f"| {row} | {', '.join(entry['complete_native_group_ids']) or 'none'} | {', '.join(entry['complete_configured_group_ids']) or 'none'} |")
    lines += ["", "Group provenance for row coverage:", "",
              "| Row | Group | Host | Label | Source SHA256 | Environment | Valid seeds | Status |",
              "|---|---|---|---|---|---|---:|---|"]
    for row, entry in coverage["rows"].items():
        for group in entry["groups"]:
            lines.append(f"| {row} | {group['group_id']} | {group['hostname']} | {group['label']} | {group['source_sha256']} | {group['environment_id']} | {group['independent_valid_seed_count']} | {group['status']} |")
    for key, env in report["environment_completeness"].items():
        lines += ["", f"Environment `{key}`: host `{env['environment'].get('hostname')}`, label `{env['label']}`.",
                  f"Complete native table rows: {', '.join(env['complete_native_table_rows']) or 'none'}.",
                  f"Complete configured table rows: {', '.join(env['complete_configured_table_rows']) or 'none'}.",
                  f"Missing rows in this environment across native/configured groups: {', '.join(env['missing_table_rows']) or 'none'}.",
                  f"Missing native-default-only rows in this environment: {', '.join(env['missing_native_table_rows']) or 'none'}."]
    lines += ["", "## Scope and interpretation", "",
              "Paper Tables 1–3 agree across the arXiv preprint, the historical download from the requested OpenReview URL, and the local camera-ready PDF. The currently served OpenReview revision has not been retrieved.", "",
              "The paper labels ± quantities as confidence intervals without specifying their level or construction. Shipped SE uses population standard deviation divided by √5; sample SE is also preserved in the JSON. Rounded ±0.00 does not mean zero uncertainty.", "",
              "Training time is descriptive: the paper used AMD MI300X 192GB, while fresh-process compilation, device and software differences affect these reruns."]
    for name, status in (report.get("missing_implementations_and_coverage") or {}).items():
        lines.append(f"\n- **{name}**: {status}.")
    lines += ["", "## Per-seed evidence", ""]
    for group in report["groups"]:
        lines += [f"### {group['group_id']}", "", f"Seeds: {group['valid_seeds']}; missing shipped seeds: {group['missing_shipped_seeds']}.", "",
                  f"KL raw run means: {group['across_seed_statistics']['kl']['raw_values']}; sample SE: {fmt(group['across_seed_statistics']['kl']['se_ddof1'])}.",
                  f"ESS raw run means: {group['across_seed_statistics']['ess_ratio']['raw_values']}; sample SE: {fmt(group['across_seed_statistics']['ess_ratio']['se_ddof1'])}."]
        for run in group["runs"]:
            errors = ", ".join(error["kind"] for error in run["errors"])
            lines.append(f"\n- Seed {run['seed']}: [{run['status']}]({run['path']})" + (f" — {errors}" if errors else ""))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, action="append", required=True,
                        help="Directory recursively containing result.json, or one result file; may repeat.")
    parser.add_argument("--paper-targets", type=Path, default=Path(__file__).with_name("paper_targets.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-host", help="Marks a primary hostname without pooling other environments.")
    parser.add_argument("--relative-paths", action="store_true",
                        help="Write evidence paths relative to output-dir so reports can move with their results.")
    args = parser.parse_args()
    for path in args.results_root:
        if not path.exists():
            parser.error(f"Result path does not exist: {path}")
    paper = json.loads(args.paper_targets.read_text())
    report = build_report(args.results_root, paper, args.primary_host)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.relative_paths:
        report = relative_evidence_paths(report, args.output_dir)
    for name, contents in (("comparison.json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"),
                           ("comparison.md", markdown(report))):
        path = args.output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(contents)
        temporary.replace(path)
        print(path)


if __name__ == "__main__":
    main()
