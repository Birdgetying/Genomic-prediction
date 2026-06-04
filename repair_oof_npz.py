#!/usr/bin/env python
"""保守修复旧 OOF NPZ 顺序的诊断脚本。

当前 ``genomic_ensemble._save_oof_npz`` 的标准契约是：``_y_true`` 与所有
OOF prediction 都按原始样本顺序保存。早期产物可能把 prediction 保存为
KFold test-fold 拼接顺序（fold0 test → fold1 test → ...），这会导致散点图
和临时复算指标错位。本脚本会先比较“原始顺序”和“按旧 fold 顺序重排”两种
解释，只有证据充分时才写回修复；差异不明显时默认跳过，避免误改新格式。
"""
import argparse
import io
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

try:
    import torch
except ImportError:  # pragma: no cover - repair script itself does not require torch
    torch = None

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42
N_FOLDS = 5
MIN_MEDIAN_R_GAIN = 0.02
MIN_IMPROVED_FRACTION = 0.60
MIN_VALID_MODELS = 2
DATASETS = ('wheat', 'wheat2000', 'rice', 'maize', 'soybean', 'wheat_gabi')
NON_PREDICTION_KEYS = {
    'sample_id', 'sample_ids', 'samples', 'indices', 'sample_indices',
    'genotype_indices', 'fold', 'folds', 'fold_id', 'fold_ids'
}


def set_global_seed(seed=RANDOM_SEED):
    """Set deterministic seeds used by this script and imported libraries."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_global_seed(RANDOM_SEED)
# PYTHONHASHSEED must be set at process launch: export PYTHONHASHSEED=42


def _load_npz_dict(npz_path):
    with np.load(npz_path, allow_pickle=True) as loaded:
        return {k: loaded[k] for k in loaded.files}


def _fold_reverse_perm(n_samples):
    """Return indices that map old fold-concat predictions back to sample order."""
    if n_samples < 2:
        return np.arange(n_samples)
    n_splits = min(N_FOLDS, n_samples)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_test_indices = [te for _, te in kf.split(np.arange(n_samples))]
    inverse_perm = np.concatenate(fold_test_indices)
    return np.argsort(inverse_perm)


def _as_prediction_array(value, n_samples):
    arr = np.asarray(value)
    if arr.ndim != 1 or len(arr) != n_samples:
        return None
    try:
        arr = arr.astype(np.float64)
    except (TypeError, ValueError):
        return None
    return arr


def _prediction_keys(data, n_samples):
    keys = []
    for key, value in data.items():
        key_l = str(key).lower()
        if key == '_y_true' or key.startswith('_') or key_l in NON_PREDICTION_KEYS:
            continue
        if _as_prediction_array(value, n_samples) is not None:
            keys.append(key)
    return keys


def _safe_metrics(y, preds):
    y = np.asarray(y, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(preds)
    if int(mask.sum()) < 3:
        return None
    y_m = y[mask]
    p_m = preds[mask]
    if np.std(y_m) <= 1e-12 or np.std(p_m) <= 1e-12:
        return None
    r = float(pearsonr(y_m, p_m)[0])
    r2 = float(r2_score(y_m, p_m))
    if not (np.isfinite(r) and np.isfinite(r2)):
        return None
    return {'r': r, 'r2': r2}


def _alignment_summary(data, reorder_perm=None):
    y = np.asarray(data['_y_true'], dtype=np.float64)
    n_samples = len(y)
    scores = {}
    skipped = 0
    for key in _prediction_keys(data, n_samples):
        preds = _as_prediction_array(data[key], n_samples)
        if reorder_perm is not None:
            preds = preds[reorder_perm]
        metrics = _safe_metrics(y, preds)
        if metrics is None:
            skipped += 1
            continue
        scores[key] = metrics

    if scores:
        rs = np.asarray([v['r'] for v in scores.values()], dtype=np.float64)
        r2s = np.asarray([v['r2'] for v in scores.values()], dtype=np.float64)
        median_r = float(np.median(rs))
        median_r2 = float(np.median(r2s))
    else:
        median_r = float('nan')
        median_r2 = float('nan')
    return {'scores': scores, 'valid': len(scores), 'skipped': skipped,
            'median_r': median_r, 'median_r2': median_r2}


def _compare_alignments(data):
    n_samples = len(data['_y_true'])
    reverse_perm = _fold_reverse_perm(n_samples)
    original = _alignment_summary(data)
    reordered = _alignment_summary(data, reverse_perm)
    common = sorted(set(original['scores']) & set(reordered['scores']))
    gains = [reordered['scores'][k]['r'] - original['scores'][k]['r'] for k in common]
    if gains:
        improved_fraction = float(np.mean(np.asarray(gains) > 0.0))
        median_gain = float(np.median(gains))
    else:
        improved_fraction = 0.0
        median_gain = float('nan')
    median_r_gain = reordered['median_r'] - original['median_r']
    median_r2_gain = reordered['median_r2'] - original['median_r2']
    if not np.isfinite(median_r_gain):
        median_r_gain = float('nan')
    if not np.isfinite(median_r2_gain):
        median_r2_gain = float('nan')
    return {'original': original, 'reordered': reordered,
            'common_models': len(common), 'median_r_gain': median_r_gain,
            'median_r2_gain': median_r2_gain,
            'median_pairwise_r_gain': median_gain,
            'improved_fraction': improved_fraction,
            'reverse_perm': reverse_perm}


def _decide_action(comparison, force_fold_order=False, force_original=False):
    if force_original:
        return 'skip', 'forced-original'
    if force_fold_order:
        return 'repair', 'forced-fold-order'

    original = comparison['original']
    reordered = comparison['reordered']
    valid = min(original['valid'], reordered['valid'], comparison['common_models'])
    if valid < MIN_VALID_MODELS:
        return 'skip', f'ambiguous: only {valid} valid comparable models'

    gain = comparison['median_r_gain']
    r2_gain = comparison['median_r2_gain']
    improved = comparison['improved_fraction']
    if (np.isfinite(gain) and gain >= MIN_MEDIAN_R_GAIN
            and np.isfinite(r2_gain) and r2_gain > 0.0
            and improved >= MIN_IMPROVED_FRACTION):
        return 'repair', (f'fold-order likely: median r gain={gain:+.4f}, '
                          f'median R2 gain={r2_gain:+.4f}, improved={improved:.0%}')
    if np.isfinite(gain) and gain <= -MIN_MEDIAN_R_GAIN:
        return 'skip', (f'original-order likely: median r gain={gain:+.4f}, '
                        f'median R2 gain={r2_gain:+.4f}')
    return 'skip', (f'ambiguous/original by default: median r gain={gain:+.4f}, '
                    f'median R2 gain={r2_gain:+.4f}, improved={improved:.0%}')


def _format_summary(label, summary):
    if summary['valid'] == 0:
        return f"{label}: valid=0 skipped={summary['skipped']}"
    return (f"{label}: valid={summary['valid']} skipped={summary['skipped']} "
            f"median_r={summary['median_r']:+.4f} "
            f"median_R2={summary['median_r2']:+.4f}")


def repair_npz(npz_path, data=None, n_samples=None, reverse_perm=None):
    """Rewrite one old fold-concat NPZ into original sample order."""
    if data is None:
        data = _load_npz_dict(npz_path)
    if n_samples is None:
        n_samples = len(data['_y_true'])
    if reverse_perm is None:
        reverse_perm = _fold_reverse_perm(n_samples)

    print(f"  Repairing: {npz_path.name}")
    y_old = data['_y_true']
    if len(y_old) != n_samples:
        raise ValueError(f"_y_true length {len(y_old)} != n_samples {n_samples}")

    new_data = {}
    repaired = 0
    for key, value in data.items():
        key_l = str(key).lower()
        if key == '_y_true' or key.startswith('_') or key_l in NON_PREDICTION_KEYS:
            new_data[key] = value
            continue
        preds = _as_prediction_array(value, n_samples)
        if preds is None:
            new_data[key] = value
            continue
        new_data[key] = preds[reverse_perm].astype(np.float32)
        repaired += 1

    backup_path = npz_path.with_suffix('.npz.bak')
    if not backup_path.exists():
        with open(backup_path, 'wb') as f:
            np.savez_compressed(f, **data)
    np.savez_compressed(npz_path, **new_data)
    print(f"    -> Repaired {repaired} prediction arrays; backup={backup_path.name}")
    return new_data


def verify_npz(data):
    """Print aggregate metrics for all usable prediction arrays."""
    summary = _alignment_summary(data)
    print(f"    Verify {_format_summary('original-order', summary)}")
    if summary['scores']:
        ranked = sorted(summary['scores'].items(), key=lambda kv: kv[1]['r'], reverse=True)
        best_name, best = ranked[0]
        worst_name, worst = ranked[-1]
        print(f"      best={best_name}: r={best['r']:+.4f}, R²={best['r2']:+.4f}")
        if worst_name != best_name:
            print(f"      worst={worst_name}: r={worst['r']:+.4f}, R²={worst['r2']:+.4f}")


def _iter_npz_files(datasets):
    for sub in datasets:
        oof_dir = SCRIPT_DIR / 'results' / f'{sub}_ensemble' / 'oof_predictions'
        if not oof_dir.exists():
            print(f"  SKIP {sub}: no oof_predictions dir")
            continue
        npz_files = sorted(oof_dir.glob('*_oof.npz'))
        if not npz_files:
            print(f"  SKIP {sub}: no NPZ files")
            continue
        print(f"\n[{sub}] {len(npz_files)} NPZ files")
        for npz_path in npz_files:
            yield sub, npz_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Auto-detect and repair old fold-concat OOF NPZ files.")
    parser.add_argument('--dry-run', action='store_true',
                        help='Report decisions without writing NPZ files or backups')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--force-fold-order', action='store_true',
                      help='Treat every NPZ as old fold-concat order and repair')
    mode.add_argument('--force-original', action='store_true',
                      help='Treat every NPZ as already original-order and skip repair')
    parser.add_argument('--datasets', nargs='*', default=DATASETS,
                        help='Dataset tags to scan (default: all known ensemble outputs)')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("=" * 70)
    print("OOF NPZ alignment check / conservative repair")
    print("=" * 70)
    if args.dry_run:
        print("Mode: dry-run (no files will be modified)")
    if args.force_fold_order:
        print("Mode: force-fold-order (manual override)")
    if args.force_original:
        print("Mode: force-original (manual override)")

    total = repaired = skipped = errors = 0
    for _, npz_path in _iter_npz_files(args.datasets):
        total += 1
        print(f"\n  Checking: {npz_path.name}")
        try:
            data = _load_npz_dict(npz_path)
            if '_y_true' not in data:
                raise ValueError("missing _y_true")
            comparison = _compare_alignments(data)
            print(f"    {_format_summary('original', comparison['original'])}")
            print(f"    {_format_summary('fold-reordered', comparison['reordered'])}")
            action, reason = _decide_action(
                comparison,
                force_fold_order=args.force_fold_order,
                force_original=args.force_original)
            if action == 'repair':
                if args.dry_run:
                    print(f"    DRY-RUN: would repair ({reason})")
                    skipped += 1
                else:
                    new_data = repair_npz(
                        npz_path, data=data, n_samples=len(data['_y_true']),
                        reverse_perm=comparison['reverse_perm'])
                    verify_npz(new_data)
                    repaired += 1
            else:
                print(f"    SKIP: {reason}")
                skipped += 1
        except Exception as e:
            errors += 1
            print(f"    [ERROR] {npz_path}: {e}")

    print("\n" + "=" * 70)
    print(f"Done. checked={total}, repaired={repaired}, skipped={skipped}, errors={errors}")
    print("=" * 70)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
