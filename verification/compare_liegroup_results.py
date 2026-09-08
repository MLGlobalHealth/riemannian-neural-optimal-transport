#!/usr/bin/env python3
"""Compare native single-seed SO3/SE3 runs with camera-ready Table 5.

Each record stays separate. Its five held-out batches provide native batch SE,
not five training replicates. Only the standard library is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

GAMMAS = (1.0, 0.1, 0.05, 0.01, 0.005, 0.001)
METRICS = ('kl', 'ess', 'ess_ratio')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identifier(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:16]


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else '+Infinity' if value > 0 else '-Infinity'
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def numeric(value):
    if isinstance(value, bool) or value is None:
        raise ValueError('Expected a numeric metric value')
    return float(value)


def same_number(a, b):
    try:
        a, b = numeric(a), numeric(b)
    except (ValueError, TypeError, OverflowError):
        return False
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def row_key(manifold, method, gamma=None):
    if manifold not in ('SO3', 'SE3') or method not in ('ours', 'rcpm'):
        return None
    if method == 'ours':
        return f'{manifold}/RNOT' if gamma is None else None
    try:
        gamma = numeric(gamma)
    except (ValueError, TypeError):
        return None
    return f'{manifold}/RCPM/{gamma:g}' if gamma in GAMMAS else None


ROWS = [row_key(manifold, method, gamma) for manifold in ('SO3', 'SE3')
        for method, gamma in [('ours', None), *(('rcpm', gamma) for gamma in GAMMAS)]]


def load_targets(path):
    paper = json.loads(path.read_text())
    table = next(table for table in paper['tables'] if table['number'] == 5)
    targets = {}
    for row in table['rows']:
        name = row['method']
        method, gamma = ('ours', None) if name == 'RNOT' else ('rcpm', float(name.split('RCPM_gamma')[1]))
        key = row_key(row['manifold'], method, gamma)
        if key is None or key in targets:
            raise ValueError('Unexpected or duplicate Table 5 numeric row')
        targets[key] = {'kind': 'numeric', 'kl_mean': row['kl']['mean'],
                        'kl_reported_plus_minus': row['kl']['reported_plus_minus'],
                        'ess_ratio_mean': row['ess_ratio']['mean']}
    if len(targets) != 9 or not any(claim.get('method') == 'RCPM_gamma_below1'
                                  for claim in table.get('non_numeric_claims', [])):
        raise ValueError('Requires all nine numeric Table 5 rows and its SE3 instability claim')
    for gamma in GAMMAS[1:]:
        targets[row_key('SE3', 'rcpm', gamma)] = {'kind': 'qualitative',
            'paper_claim': 'Numerically unstable', 'paper_criterion': 'No threshold or exact failure type specified.'}
    if set(targets) != set(ROWS):
        raise ValueError('Table 5 targets do not cover the declared fourteen native configurations')
    return targets


def environment_identity(record):
    env = deepcopy(record.get('environment', {}))
    variables = env.get('environment', {})
    visible = variables.get('CUDA_VISIBLE_DEVICES')
    rows = []
    for line in str(env.get('nvidia_smi', '')).splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) == 5 and fields[0].isdigit():
            rows.append(fields)
    selected = [row for row in rows if str(visible) in (row[0], row[2])]
    # Only normalize the index when the recorded host/model/driver is known.
    if env.get('hostname') and selected:
        variables.pop('CUDA_VISIBLE_DEVICES', None)
        if variables.get('HIP_VISIBLE_DEVICES') in (None, visible):
            variables.pop('HIP_VISIBLE_DEVICES', None)
        env.pop('nvidia_smi', None)
        env.pop('jax_devices', None)
        env['selected_accelerator_types'] = sorted({(row[1], row[3], row[4]) for row in selected})
    return env


def cohort_identity(record):
    return {'source_sha256': record.get('source', {}).get('sha256'),
            'wrapper_sha256': record.get('runner_sha256'), 'helper_sha256': record.get('helper_sha256'),
            'label': record.get('label'), 'environment': environment_identity(record)}


def configuration_identity(record):
    fields = ('selection', 'native_runner', 'configuration', 'evaluation_config',
              'explicit_configuration_overrides', 'density_configuration', 'geometry_configuration',
              'landmark_configuration', 'parameter_count', 'parameter_dtypes')
    return {field: record.get(field) for field in fields}


def batch_statistics(values):
    floats = [numeric(value) for value in values]
    if not all(math.isfinite(value) for value in floats):
        return {'status': 'nonfinite', 'raw_batches': values, 'mean': None,
                'se_ddof0': None, 'se_ddof1': None}
    count = len(floats)
    return {'status': 'finite', 'raw_batches': values, 'mean': statistics.mean(floats),
            'se_ddof0': statistics.pstdev(floats) / math.sqrt(count),
            'se_ddof1': statistics.stdev(floats) / math.sqrt(count) if count > 1 else None}


def rounding(value, target, places):
    center = Decimal(str(target)); half = Decimal(10) ** -places / 2
    observed = Decimal(str(value))
    return {'observed': value, 'paper': target, 'decimal_places': places,
            'paper_rounding_interval': [str(center - half), str(center + half)],
            'matches_printed_mean': center - half <= observed < center + half,
            'difference_from_paper': value - target}


def inspect_record(record):
    errors, protocol_errors, nonfinite = [], [], []
    batches = record.get('evaluation_batches', [])
    if not isinstance(batches, list):
        errors.append('evaluation_batches is not a list'); batches = []
    stats = {}
    keys = set(METRICS)
    for batch in batches:
        if not isinstance(batch, dict) or not isinstance(batch.get('metrics'), dict):
            errors.append('Malformed evaluation batch'); continue
        keys.update(batch['metrics'])
        for name, value in batch['metrics'].items():
            try:
                finite = math.isfinite(numeric(value))
            except (ValueError, TypeError, OverflowError):
                errors.append(f'Batch {batch.get("index")} {name}: invalid numeric value'); continue
            if not finite:
                nonfinite.append({'kind': 'evaluation_metric', 'batch': batch.get('index'), 'metric': name, 'value': value})
            flag = batch.get('metric_nonfinite', {}).get(name)
            if flag is not None and flag != (not finite):
                errors.append(f'Batch {batch.get("index")} {name}: nonfinite flag disagrees with value')
    count = record.get('parameter_nonfinite_count', 0)
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        nonfinite.append({'kind': 'recorded_nonfinite_trained_parameters', 'count': count})
    for name in sorted(keys):
        if batches and all(isinstance(batch, dict) and name in batch.get('metrics', {}) for batch in batches):
            try:
                stats[name] = batch_statistics([batch['metrics'][name] for batch in batches])
            except (ValueError, TypeError, OverflowError):
                errors.append(f'{name}: cannot recompute batch statistics')
    complete = record.get('status') == 'complete'
    if complete:
        if record.get('record_type') != 'native_liegroup_training_run':
            protocol_errors.append('Wrong record type')
        if record.get('selection', {}).get('seed') != 12345:
            protocol_errors.append('Not the native training seed12345')
        if record.get('selection', {}).get('method') == 'ours' and record.get('selection', {}).get('landmark_method') != 'fps':
            protocol_errors.append('Not the supplied native FPS landmark setting')
        cfg = record.get('evaluation_config', {})
        if cfg.get('seed') != 12345 or cfg.get('batch_size') != 1024 or cfg.get('n_batches') != 5 or cfg.get('seed_offset') is not None:
            protocol_errors.append('Not the native five-batch1024/seed12345 evaluation recipe')
        if len(batches) != 5:
            protocol_errors.append('Expected five held-out batches from this one trained model')
        for index, batch in enumerate(batches):
            key = batch.get('prng_key') if isinstance(batch, dict) else None
            if not isinstance(batch, dict) or batch.get('index') != index or not isinstance(key, list) or len(key) != 2 or any(
                    not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32 for value in key):
                protocol_errors.append('Malformed or unordered native evaluation subkey')
                continue
            if not set(METRICS).issubset(batch.get('metrics', {})):
                errors.append('A core metric is missing')
            else:
                try:
                    ess, ratio = numeric(batch['metrics']['ess']), numeric(batch['metrics']['ess_ratio'])
                    if math.isfinite(ess) and math.isfinite(ratio) and not same_number(ess / 1024, ratio):
                        errors.append('ESS ratio is inconsistent with ESS/1024')
                except (ValueError, TypeError):
                    pass
        if record.get('source_unchanged_after_run') is not True:
            protocol_errors.append('Source unchanged-after-run confirmation absent or false')
        if not all(record.get(field) for field in ('source', 'runner_sha256', 'helper_sha256', 'native_runner', 'configuration', 'environment')):
            protocol_errors.append('Required source/configuration/environment evidence missing')
        summaries = record.get('summary', {})
        for name, computed in stats.items():
            reported = summaries.get(name, {})
            raw = reported.get('raw_batches')
            if not isinstance(raw, list) or len(raw) != len(computed['raw_batches']) or not all(
                    same_number(a, b) for a, b in zip(raw or [], computed['raw_batches'])):
                errors.append(f'{name}: saved summary raw batches disagree')
            if computed['status'] == 'finite':
                for field in ('mean', 'se_ddof0', 'se_ddof1'):
                    if computed[field] is not None and not same_number(computed[field], reported.get(field)):
                        errors.append(f'{name}: saved summary {field} disagrees with batch recomputation')
            else:
                for field in ('mean', 'se_ddof0', 'se_ddof1'):
                    try:
                        if math.isfinite(numeric(reported.get(field))):
                            errors.append(f'{name}: saved summary {field} hides nonfinite batch values')
                    except (ValueError, TypeError, OverflowError):
                        errors.append(f'{name}: invalid nonfinite summary {field}')
    # Error text containing "NaN" is not numerical evidence. Actual metric
    # values or the wrapper's measured nonfinite parameter count are required.
    if nonfinite:
        outcome = 'observed_numerical_nonfinite'
    elif record.get('status') == 'failed':
        outcome = 'execution_failure'
    elif errors or protocol_errors:
        outcome = 'invalid_or_other_protocol'
    elif complete:
        outcome = 'complete_finite'
    else:
        outcome = 'in_progress'
    return {'outcome': outcome, 'execution_status': record.get('status'),
            'failed_during': record.get('failed_during'), 'errors': errors,
            'protocol_differences': protocol_errors, 'observed_nonfinite_evidence': nonfinite,
            'recomputed_batch_statistics': stats, 'evaluation_batch_count': len(batches),
            'complete_native_record': complete and not errors and not protocol_errors}


def paper_comparison(inspection, target):
    if target['kind'] == 'qualitative':
        observed = bool(inspection['observed_nonfinite_evidence'])
        return {'kind': 'qualitative', 'paper_claim': 'Numerically unstable',
                'observed_nonfinite_instability': observed,
                'assessment': 'observed_nonfinite_instability' if observed else
                    'execution_failure_does_not_establish_instability' if inspection['outcome'] == 'execution_failure' else
                    'finite_output_does_not_establish_instability' if inspection['outcome'] == 'complete_finite' else 'unresolved',
                'note': 'The paper supplies no instability threshold; no numeric target or missing value is invented.'}
    if inspection['outcome'] != 'complete_finite':
        return {'kind': 'numeric', 'assessment': 'not_a_valid_complete_finite_native_result'}
    stats = inspection['recomputed_batch_statistics']
    return {'kind': 'numeric', 'kl_mean': rounding(stats['kl']['mean'], target['kl_mean'], 2),
            'ess_ratio_mean': rounding(stats['ess_ratio']['mean'], target['ess_ratio_mean'], 3),
            'kl_native_batch_se': rounding(stats['kl']['se_ddof0'], target['kl_reported_plus_minus'], 2),
            'uncertainty_note': 'Descriptive printed-number comparison only; native batch SE is not assigned the unspecified paper confidence level.'}


def compare(paths, targets, output_dir):
    entries, copies, ignored = [], {}, []
    for path in sorted(set(paths)):
        data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        relative = os.path.relpath(path.resolve(), output_dir.resolve())
        if digest in copies:
            copies[digest]['evidence_paths'].append(relative)
            continue
        try:
            record = json.loads(data)
            if not isinstance(record, dict):
                raise ValueError('Record is not an object')
        except (json.JSONDecodeError, ValueError) as error:
            entries.append({'record_id': digest[:16], 'evidence_paths': [relative], 'sha256': digest,
                            'raw_text': data.decode('utf8', 'replace'), 'outcome': 'invalid_json', 'error': str(error)})
            copies[digest] = entries[-1]; continue
        if record.get('record_type') != 'native_liegroup_training_run':
            ignored.append({'path': relative, 'sha256': digest, 'reason': 'Not a native Lie-group training record', 'raw_record': record}); continue
        selection = record.get('selection', {})
        row = row_key(selection.get('manifold'), selection.get('method'), selection.get('gamma'))
        cohort = cohort_identity(record); configuration = configuration_identity(record)
        inspection = inspect_record(record)
        entry = {'record_id': digest[:16], 'evidence_paths': [relative], 'sha256': digest,
                 'row': row, 'cohort_id': identifier(cohort), 'cohort_identity': cohort,
                 'configuration_group_id': identifier({'cohort': cohort, 'configuration': configuration}),
                 'configuration_identity': configuration, 'raw_record': record, **inspection}
        entry['paper_comparison'] = paper_comparison(inspection, targets[row]) if row in targets else {'assessment': 'outside_Table5'}
        entries.append(entry); copies[digest] = entry
    cohorts = {}
    configuration_groups = defaultdict(list)
    for entry in entries:
        if 'cohort_id' not in entry:
            continue
        cohort = cohorts.setdefault(entry['cohort_id'], {'identity': entry['cohort_identity'], 'rows': {row: [] for row in ROWS}})
        if entry['row'] in cohort['rows']:
            cohort['rows'][entry['row']].append(entry)
        configuration_groups[entry['configuration_group_id']].append(entry['record_id'])
    for cohort in cohorts.values():
        slots = {}
        for row, members in cohort['rows'].items():
            slots[row] = {'record_ids': [member['record_id'] for member in members],
                'configuration_group_ids': sorted({member['configuration_group_id'] for member in members}),
                'status': 'missing' if not members else 'duplicate_or_configuration_variant' if len(members) > 1 else members[0]['outcome'],
                'one_complete_native_record': len(members) == 1 and members[0]['complete_native_record']}
        cohort['rows'] = slots
        cohort['complete_native_row_count'] = sum(slot['one_complete_native_record'] for slot in slots.values())
        cohort['status'] = 'complete_14_native_runs' if cohort['complete_native_row_count'] == 14 else 'partial_or_failed'
        cohort['missing_rows'] = [row for row, slot in slots.items() if slot['status'] == 'missing']
        cohort['duplicate_or_variant_rows'] = [row for row, slot in slots.items() if slot['status'] == 'duplicate_or_configuration_variant']
    counts = Counter(entry['outcome'] for entry in entries)
    return {'schema_version': 1, 'scope': 'Table5 only', 'evidence_path_base': 'output_directory',
        'protocol': {'training_repetitions_per_record': 1, 'evaluation_batches_per_complete_record': 5,
                     'evaluation_batch_size': 1024, 'training_and_evaluation_seed': 12345,
                     'aggregation': 'Only batch means and batch SE inside each individual record; never across runs, seeds, hosts, or configurations.'},
        'paper_targets': targets, 'record_outcome_counts': dict(counts), 'records': entries,
        'configuration_groups': dict(configuration_groups), 'cohorts': cohorts, 'ignored_records': ignored,
        'record_file_count': sum(len(entry['evidence_paths']) for entry in entries),
        'unique_record_count': len(entries),
        'identical_record_copy_count': sum(len(entry['evidence_paths']) - 1 for entry in entries),
        'coverage_note': 'A cohort is complete only with one complete native record for each of14 slots. Distinct duplicate runs/configurations make their slot ambiguous. Complete coverage does not mean numerical agreement with the paper.',
        'limitations': 'Table5 does not specify exact Lie-group training settings or interval construction. Sources and configurations remain explicit. Import/build failures are execution failures, not evidence of the paper’s numerical instability claim. Timings are descriptive, with no AMD timing match asserted.'}


def fmt(value):
    return f'{value:.8g}' if isinstance(value, (int, float)) and math.isfinite(value) else '—'


def markdown(report):
    lines = ['# Table5 native Lie-group comparison', '',
        'Each record represents one training seed and five held-out evaluation batches. All means and standard errors below are calculated within that record. Training runs, hosts and configurations are never pooled.', '',
        f"Unique records: {report['unique_record_count']}; identical copied records: {report['identical_record_copy_count']}. Outcome counts: {report['record_outcome_counts']}.", '',
        'Native batch SE uses population standard deviation divided by √5. Supplemental sample SE and all raw records/batches are retained in [comparison.json](comparison.json). The paper’s interval construction is unspecified; printed-number agreement is descriptive.', '',
        '## Source and environment cohorts', '', '| Cohort | Host | Source / helper / wrapper SHA prefixes | Complete unique native rows | Status |', '| --- | --- | --- | ---: | --- |']
    for key, cohort in report['cohorts'].items():
        hashes = ' / '.join(str(cohort['identity'].get(field) or 'unknown')[:12]
                            for field in ('source_sha256', 'helper_sha256', 'wrapper_sha256'))
        lines.append(f"| {key} | {cohort['identity']['environment'].get('hostname', 'unknown')} | {hashes} | {cohort['complete_native_row_count']} / 14 | {cohort['status']} |")
    lines += ['', 'Each slot must have exactly one complete native run. Additional executions or configuration variants remain visible and do not silently satisfy completeness. The JSON records full source/helper/wrapper hashes, environment and configuration identities.', '',
              '## All fourteen Table5 configurations', '',
              '| Row | Paper KL ± quantity; ESS ratio | Recorded outcomes |', '| --- | --- | --- |']
    for row in ROWS:
        target = report['paper_targets'][row]
        paper = 'Numerically unstable; —' if target['kind'] == 'qualitative' else f"{target['kl_mean']:.2f} ± {target['kl_reported_plus_minus']:.2f}; {target['ess_ratio_mean']:.3f}"
        entries = [entry for entry in report['records'] if entry.get('row') == row]
        outcomes = '; '.join(f"[{entry['record_id']}](<{entry['evidence_paths'][0]}>) ({entry['cohort_id']}, {entry['outcome']})" for entry in entries) or 'Missing'
        lines.append(f'| {row} | {paper} | {outcomes} |')
    lines += ['', '## Individual native outputs', '',
              '| Record / cohort | Row | KL mean ± batch SE | ESS ratio mean | Printed comparison / outcome |', '| --- | --- | ---: | ---: | --- |']
    validations = []
    for entry in report['records']:
        stats = entry.get('recomputed_batch_statistics', {}); kl = stats.get('kl', {}); ess = stats.get('ess_ratio', {})
        comparison = entry.get('paper_comparison', {})
        if 'kl_mean' in comparison:
            status = ', '.join(f"{name}: {'matches' if comparison[field]['matches_printed_mean'] else 'differs'}" for name, field in [('KL mean', 'kl_mean'), ('ESS mean', 'ess_ratio_mean'), ('KL batch SE', 'kl_native_batch_se')])
        else:
            status = comparison.get('assessment', entry['outcome'])
        link = f"[{entry['record_id']}](<{entry['evidence_paths'][0]}>) / {entry.get('cohort_id', 'unassigned')}"
        lines.append(f"| {link} | {entry.get('row', 'invalid JSON')} | {fmt(kl.get('mean'))} ± {fmt(kl.get('se_ddof0'))} | {fmt(ess.get('mean'))} | {status} |")
        if entry.get('errors') or entry.get('protocol_differences'):
            validations.append(f"Record {entry['record_id']} validation: {'; '.join(entry.get('errors', []) + entry.get('protocol_differences', []))}")
    if validations:
        lines += ['', *validations]
    lines += ['', 'Actual nonfinite evaluation values or measured nonfinite trained parameters establish observed numerical instability. Error messages containing “NaN”, import failures and build failures alone do not. Finite results for the qualitative SE3 rows are retained without inventing an instability threshold.', '',
              'Full original records, failures, duplicate executions, identical-copy paths, nonfinite values, supplemental sample SE and per-row configuration identities are retained in the JSON. The original inputs are unchanged.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, action='append', required=True)
    parser.add_argument('--paper-targets', type=Path, default=Path(__file__).with_name('paper_targets.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    targets = load_targets(args.paper_targets)
    paths = [path for root in args.results_root for path in root.rglob('result.json')]
    report = compare(paths, targets, args.output_dir)
    report['paper_targets_path'] = os.path.relpath(args.paper_targets.resolve(), args.output_dir.resolve())
    report['paper_targets_sha256'] = hashlib.sha256(args.paper_targets.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'comparison.json').write_text(json.dumps(safe(report), indent=2, sort_keys=True, allow_nan=False) + '\n')
    (args.output_dir / 'comparison.md').write_text(markdown(report))
    print(json.dumps({'records': report['record_outcome_counts'], 'cohorts': len(report['cohorts']),
                      'complete_14_native_cohorts': sum(cohort['status'] == 'complete_14_native_runs' for cohort in report['cohorts'].values())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
