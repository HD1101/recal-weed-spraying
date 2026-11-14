"""Replays registered D008–D010 on immutable local inputs. Python 3.12 and NumPy."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
from fractions import Fraction
import gzip
from hashlib import file_digest, sha256
import itertools
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get('RECAL_WORKSPACE', REPO.parent)).resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'analysis'))
from aggregate import cross_seed
from posthoc_20261104.engine import load_records
from geometry import GRID, geometry_arrays, truth_reference

OUT = Path(os.environ.get('RECAL_POSTHOC_OUTPUT', REPO / 'results/posthoc_20261114')).resolve()
PLAN = REPO / 'plan/POSTHOC_PLAN_20261114_REV2.md'
PLAN_SHA = '9b9cc441fef404c2a2fe90e2385662d06a0de440674aaa5bbf58510c920a8730'
FROZEN = REPO / 'plan/ANALYSIS_PLAN_FROZEN_20261016.md'
FROZEN_SHA = 'e80795012ce4d4b492a0152a5e89acebe8246703c2fd579f7b4b98f64a7e81f9'
PRIVATE = WORKSPACE / 'work/posthoc_20261104'
BACKUP = WORKSPACE / 'outputs/FORMAL_RAW_BACKUP_20261019'
HISTORY = REPO / 'results/posthoc_20261104'
TARGETS = {'bbro': 1104, 'near30': 2999, 'walledgarden': 299, 'PhenoBench': 772, 'OD_SugarBeets': 964}
FOLDS = ('bbro', 'near30', 'walledgarden')
MODELS = ('asset_detr', 'rtdetr_r18', 'yolo26m')
IDENTITY = ('rotation', 'model', 'dataset', 'method', 'scheme', 'nozzles', 'K', 'alpha', 'coverage')
FIELDS = ('M', 'A', 'full_spray_fallback_count')
REPEATS = 200


def read(path):
    return json.loads(path.read_text())


def read_gz(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def digest(path):
    with path.open('rb') as file:
        return file_digest(file, 'sha256').hexdigest()


def json_out(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def csv_out(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def timestamp():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')


def validate_registration():
    line, stamp = PLAN.with_suffix('.sha256.txt').read_text().splitlines()
    assert line.split() == [PLAN_SHA, PLAN.name]
    assert digest(PLAN) == PLAN_SHA and digest(FROZEN) == FROZEN_SHA
    assert datetime.fromisoformat(stamp.split('=', 1)[1]) < datetime.now(ZoneInfo('Asia/Tokyo'))
    return stamp.split('=', 1)[1]


def preflight():
    verified = HISTORY / 'INPUT_VERIFICATION_20261104.json'
    evidence = read(verified)
    provenance = PRIVATE / 'TRUTH_PRIVATE_PROVENANCE.json'
    assert digest(provenance) == evidence['private_hash_manifest_sha256']
    assert evidence['status'] == 'ALL_REQUIRED_INPUTS_VERIFIED'
    labels = [row for row in read(provenance)['verified_files'] if row['kind'] == 'label']
    for row in labels:
        assert row['formal_hash_matched'] and digest(Path(row['local'])) == row['sha256']
    assert len(labels) == 6138
    manifest = read(BACKUP / 'audit/FILE_MANIFEST.json')
    files = {row['archive_path']:row for row in manifest['files']}
    for rotation in FOLDS:
        for name in ('val_GT_ONLY.json', 'target_GT_ONLY.json', 'SPLIT_BINDING.json'):
            key = 'splits/' + rotation + '/' + name
            assert digest(BACKUP / key) == files[key]['sha256'], key
    return len(labels)


def source_threshold(counts, hits, alpha):
    total = int(counts.sum())
    if not total:
        return None
    # Equivalent to exact signed per-weed loss <= 0.
    feasible = np.flatnonzero((total - hits.sum(axis=0)) * alpha.denominator <= total * alpha.numerator)
    return int(feasible[-1]) if feasible.size else None


def exact_rate(counts, missed):
    # All inputs to bincount are small integers, represented exactly as doubles.
    numerators = np.bincount(counts, weights=missed)
    return sum((Fraction(int(v), n) for n, v in enumerate(numerators) if n and v), Fraction())


def crc_threshold(counts, hits, alpha):
    kp = int(np.count_nonzero(counts))
    budget = alpha * (kp + 1) - 1
    if budget < 0:
        return None
    low, high, best = 0, len(GRID) - 1, None
    while low <= high:
        mid = (low + high) // 2
        if exact_rate(counts, counts - hits[:, mid]) <= budget:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def counts_and_area(records):
    counts = np.asarray([sum(b.class_id == 1 for b in r.ground_truth) for r in records])
    footprint = np.asarray([r.height * r.width for r in records])
    return counts, footprint


def prepare_splits(raw, counts, footprint):
    for split in raw:
        if 'error' in split:
            yield (split, None)
            continue
        cal, test = (np.asarray(split[k], dtype=np.intp) for k in ('calibration', 'test'))
        cp, tp = counts[cal], counts[test]
        kp = int(np.count_nonzero(cp))
        assert kp == split['K_prime'] and len(cal) == split['K'] and len(test)
        yield (split, (cal, test, cp, tp, kp, int(np.count_nonzero(tp)), int(tp.sum()), int(footprint[test].sum())))


def describe(values, key):
    valid = [x for x in values if x is not None]
    return {key: fmean(valid) if valid else None,
            key + '_MC_SE': stdev(valid) / math.sqrt(REPEATS) if len(valid) == REPEATS else None}


def evaluate(splits, hits, areas, alpha, method, selected_source=None, ref_area=None):
    cells = {}
    for split, prepared in splits:
        key = split['scheme'], split['K']
        row = cells.setdefault(key, [])
        if prepared is None:
            row.append((None, None, None, None, False, None, False))
            continue
        cal, test, cp, tp, kp, positives, weeds, image_area = prepared
        if method == 'perfect_detection_reference':
            row.append((None, float(np.sum(ref_area[test])) / image_area, None, kp, None, None, True))
            continue
        threshold = crc_threshold(cp, hits[cal], alpha) if method == 'crc' else selected_source
        fallback = method == 'crc' and threshold is None
        if fallback:
            row.append((0. if positives else None, 1., 1. if weeds else None, kp, True, None, True))
        elif threshold is None:
            row.append((None, None, None, kp, False, None, False))
        else:
            sprayed = hits[test, threshold]
            missed = exact_rate(tp, tp - sprayed)
            area = float(np.sum(areas[test, threshold])) / image_area
            row.append((float(missed / positives) if positives else None, area,
                        int(sprayed.sum()) / weeds if weeds else None, kp, False, GRID[threshold], True))
    summaries = []
    for (scheme, K), rows in cells.items():
        assert len(rows) == REPEATS
        m, a, wcr, kp, fallback, thresholds, valid = zip(*rows)
        fvals = [float(f) for f, v in zip(fallback, valid) if v and f is not None]
        kvals = [k for k in kp if k is not None]
        summary = {'scheme': scheme, 'K': K, 'valid_repeats': sum(valid),
                   'valid_M_repeats': sum(x is not None for x in m), **describe(a, 'A')}
        if method != 'perfect_detection_reference':
            summary.update({**describe(m, 'M'), **describe(wcr, 'WCR'),
                            **describe(fvals, 'fallback_fraction'), **describe(kp, 'K_prime_mean'),
                            'K_prime_min': min(kvals) if kvals else None,
                            'K_prime_max': max(kvals) if kvals else None,
                            'full_spray_fallback_count': int(sum(fvals)),
                            'mean_selected_threshold': describe(thresholds, 'threshold')['threshold'],
                            'mean_M_le_alpha': int(fmean(x for x in m if x is not None) <= float(alpha))
                            if any(x is not None for x in m) else None})
        summaries.append(summary)
    return summaries


def read_cache(job, dataset, prediction, manifest, records, split_archive):
    cache = PRIVATE / 'geometry_formal_grid_v2' / job / (dataset + '.npz')
    marker = read(cache.with_suffix('.binding.json'))
    b = marker['binding']
    assert digest(cache) == marker['cache_sha256']
    for key, path in {
        'prediction_sha256': prediction, 'manifest_sha256': manifest,
        'input_verification_sha256': HISTORY / 'INPUT_VERIFICATION_20261104.json',
        'original_geometry_code_sha256': REPO / 'analysis/wcr.py',
        'original_crop_code_sha256': REPO / 'analysis/additional.py',
        'engine_sha256': REPO / 'analysis/posthoc_20261104/engine.py',
    }.items():
        assert b[key] == digest(path), (job, dataset, key)
    assert b['grid'] == {'start': 0, 'end': 1, 'step': .001}
    with np.load(cache, allow_pickle=False) as f:
        assert list(f['ids']) == [r.image_id for r in records]
        hits, areas = f['hits_4'], f['areas_4']
        counts, footprint = counts_and_area(records)
        assert np.array_equal(counts, f['counts']) and np.array_equal(footprint, f['image_areas'])
    assert tuple(split_archive['config']['threshold_grid']) == GRID
    return hits, areas


def calculate_job(task):
    rotation, model, seed, step = task
    job = f'{rotation}__{model}__seed{seed}'
    wall = time.monotonic()
    cpu = time.process_time()
    config = read(PRIVATE / 'VERIFIED_INPUT_CONFIG.json')
    manifests = {k:Path(v) for k,v in config['GT_manifests'].items()}
    base = Path(config['predictions_root']) / 'runs' / job
    markers = tuple(base.glob('attempt*/predictions*/EXPORT_COMPLETE.json'))
    assert len(markers) == 1
    pred = markers[0].parent
    export = read(markers[0])
    expected = {rotation, 'PhenoBench', 'OD_SugarBeets', 'source_holdout'}
    assert set(export['files']) == {ds + '.json' for ds in expected}
    for name, has in export['files'].items():
        assert digest(pred / name) == has, (job, name)
    source, _ = load_records(pred / 'source_holdout.json', manifests['LB'])
    source_formal = read(BACKUP / 'splits' / rotation / 'val_GT_ONLY.json')['images']
    assert len(source) == len(source_formal) == 200
    assert [r.image_id for r in source] == [r['id'] for r in source_formal]
    assert not ({r.image_id for r in source} & {r['id'] for r in read(
        BACKUP / 'splits' / rotation / 'target_GT_ONLY.json')['images']})
    src_count, _ = counts_and_area(source)
    n_values = (4,) if step == 'regression' else (8, 16)
    src_default = {n:source_threshold(src_count, geometry_arrays(source, n)[0], Fraction(1,10)) for n in n_values}
    if step == 'compute':
        src_partial = source_threshold(src_count, geometry_arrays(source, 4, partial=True)[0], Fraction(1,10))
    archive_receipt = read(HISTORY / 'D007_ARCHIVE_VERIFICATION.json')
    rows = []
    for dataset in (rotation, 'PhenoBench', 'OD_SugarBeets'):
        manifest = manifests['LB' if dataset == rotation else dataset]
        prediction = pred / (dataset + '.json')
        records, _ = load_records(prediction, manifest)
        assert len(records) == TARGETS[dataset]
        original = read_gz(PRIVATE / 'original_crc' / job / (dataset + '.json.gz'))
        assert original['dataset'] == dataset and original['job'] == job
        assert len(records) == original['target_images']
        vals = original['config']
        assert vals['alpha'] == .1 and vals['seed'] == 20261013 and vals['repeats'] == REPEATS
        canonical = read_gz(PRIVATE / 'canonical_splits' / (dataset + '.json.gz'))
        canonical_hash = sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert canonical_hash == original['canonical_splits_sha256'] == archive_receipt['canonical_split_sha256_by_dataset'][dataset]
        assert len(canonical) == 2000
        counts, footprint = counts_and_area(records)
        splits = list(prepare_splits(canonical, counts, footprint))

        def add(analysis, n, alpha, coverage, method, hits, areas, source_choice=None, ref=None):
            identity = {'analysis': analysis, 'rotation': rotation, 'model': model, 'seed': seed,
                        'dataset': dataset, 'method': method, 'nozzles': n, 'alpha': float(alpha), 'coverage': coverage}
            rows.extend({**identity, **row} for row in
                        evaluate(splits, hits, areas, alpha, method, source_choice, ref))

        if step == 'regression':
            hits, areas = read_cache(job, dataset, prediction, manifest, records, original)
            add('baseline', 4, Fraction(1,10), 'complete', 'source', hits, areas, src_default[4])
            add('baseline', 4, Fraction(1,10), 'complete', 'crc', hits, areas)
        else:
            for n in (8, 16):
                hits, areas = geometry_arrays(records, n)
                add('D008', n, Fraction(1,10), 'complete', 'source', hits, areas, src_default[n])
                add('D008', n, Fraction(1,10), 'complete', 'crc', hits, areas)
                add('D008', n, Fraction(1,10), 'complete', 'perfect_detection_reference', None, None,
                    ref=truth_reference(records, n))
                del hits, areas
            hits, areas = read_cache(job, dataset, prediction, manifest, records, original)
            for alpha in (Fraction(1,20), Fraction(1,5)):
                add('D009', 4, alpha, 'complete', 'crc', hits, areas)
            del hits, areas
            hits, areas = geometry_arrays(records, 4, partial=True)
            add('D010', 4, Fraction(1,10), '0.9', 'source', hits, areas, src_partial)
            add('D010', 4, Fraction(1,10), '0.9', 'crc', hits, areas)
    return rows, {'job': job, 'worker_cpu_seconds': time.process_time() - cpu,
                  'worker_wall_seconds': time.monotonic() - wall}


def row_key(row, fields):
    return tuple(str(row[k]) for k in fields)


def compare(rows, filename, cross=False, main=False):
    keys = tuple(k for k in IDENTITY if k not in ('alpha', 'coverage'))
    if not cross:
        keys += ('seed',)
    expected = {}
    with (REPO/'results'/filename).open(newline='') as file:
        for line, item in enumerate(csv.DictReader(file), 2):
            if item['nozzles'] == '4' and item['method'] in ('source','crc'):
                expected[row_key(item, keys)] = (line, item)
    if main:
        rows = [r for r in rows if r['rotation'] in ('bbro','near30') and r['scheme']=='grouped' and r['K'] in (20,50)]
    assert len(rows) == len(expected)
    stats = []
    for r in rows:
        line, old = expected.pop(row_key(r, keys))
        out = {k:r[k] for k in keys}
        out.update(source_file=filename, source_line=line)
        for field, source_col in zip(FIELDS, ('mean_actual_weed_image_miss_rate','mean_spray_area_fraction','full_spray_fallback_count')):
            if field == 'full_spray_fallback_count' and not cross:
                suffixes = ('',)
            else:
                suffixes = ('_seed_mean','_seed_min','_seed_max') if cross else ('',)
            for suffix in suffixes:
                col = source_col + suffix
                before = float(old[col]) if old.get(col, '') != '' else None
                after = r.get(field + suffix)
                error = abs(before - after) if before is not None and after is not None else None
                ok = before is None and after is None or error is not None and error <= 1e-9
                out.update({col + '_old':before, col + '_new':after, col + '_abs_diff':error, col + '_match':ok})
        stats.append(out)
    assert not expected
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('regression', 'compute'))
    p.add_argument('--workers', type=int, default=4)
    opts = p.parse_args()
    stamp = validate_registration()
    if opts.step == 'compute':
        gate = read(OUT / 'REGRESSION_STATUS.json')
        assert gate['status'] == 'PASSED' and gate['plan_sha256'] == PLAN_SHA
        assert gate['comparison_sha256'] == digest(OUT / 'REGRESSION_COMPARISON.csv')
        assert gate['engine_sha256'] == digest(Path(__file__))
    stage_wall, parent_cpu = time.monotonic(), time.process_time()
    start = timestamp()
    jobs = list(itertools.product(FOLDS, MODELS, range(3), (opts.step,)))
    nlabels = preflight() if opts.step == 'regression' else None
    rows, timing = [], []
    with ProcessPoolExecutor(max_workers=opts.workers) as pool:
        for result, worker in pool.map(calculate_job, jobs):
            rows.extend(result)
            timing.append(worker)
            print(opts.step, len(timing), '/', len(jobs), worker['job'], flush=True)
    common = {'plan_sha256': PLAN_SHA, 'registered_at': stamp, 'started_at': start,
              'ended_at': timestamp(), 'wall_seconds': time.monotonic() - stage_wall,
              'cpu_seconds': sum(r['worker_cpu_seconds'] for r in timing) + time.process_time() - parent_cpu,
              'engine_sha256': digest(Path(__file__)), 'workers': opts.workers, 'timing_by_job': timing}
    if opts.step == 'regression':
        cross = cross_seed(rows, IDENTITY)
        audits = (compare(rows,'PER_SEED_ALL_RESULTS.csv') +
                  compare(cross,'CROSS_SEED_ALL_RESULTS.csv',cross=True) +
                  compare(cross,'MAIN_TABLE_N4_GROUPED_K20_K50.csv',cross=True,main=True))
        path = OUT / 'REGRESSION_COMPARISON.csv'
        csv_out(path, audits)
        bad = sum(any(v is False for k,v in r.items() if k.endswith('_match')) for r in audits)
        max_abs = max((v for r in audits for k,v in r.items() if k.endswith('_abs_diff') and v is not None), default=0.)
        json_out(OUT/'REGRESSION_STATUS.json', {**common,'status':'FAILED' if bad else 'PASSED',
                 'rows_checked':len(audits),'bad_rows':bad,'maximum_absolute_difference':max_abs,
                 'labels_verified':nlabels,'comparison_sha256':digest(path)})
        if bad:
            raise SystemExit('Regression failed. Inspect REGRESSION_COMPARISON.csv.')
    else:
        sizes = {}
        for name in ('D008','D009','D010'):
            per = [r for r in rows if r['analysis']==name]
            cross = cross_seed(per, IDENTITY)
            assert all(row['n_seeds']==3 for row in cross)
            sizes[name] = {'per_seed':len(per), 'cross_seed':len(cross)}
            csv_out(OUT / (name + '_FULL.csv'),
                    [{'aggregation':'per_seed', **r} for r in per] +
                    [{'aggregation':'cross_seed', 'analysis':name, **r} for r in cross])
        json_out(OUT/'COMPUTATION_STATUS.json',{**common,'status':'COMPLETED','rows':sizes})
    print(opts.step, 'COMPLETE', flush=True)


if __name__ == '__main__':
    main()
