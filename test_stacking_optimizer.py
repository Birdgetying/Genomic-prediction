#!/usr/bin/env python3
"""Stacking optimizer — local testing of improved stacking strategies.

Uses genomic_ensemble.py for data loading & model training. Caches OOF predictions
to disk for fast iteration.

Three key improvements tested against baseline Stacking (All) Ridge:
  1. R² pre-filter: remove models with R² < 0 before stacking
  2. ElasticNetCV meta-learner: L1 sparsity + L2 stability
  3. Greedy forward selection: auto-find optimal model subset via nested CV
"""
import sys, os, time, json, random, pickle, warnings
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# Fix Windows console encoding for Unicode characters (R², Δ, etc.)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import genomic_ensemble as ge

# ============================================================================
# Config
# ============================================================================
CACHE_DIR = "results/stacking_cache"
N_FOLDS = 3
RANDOM_SEED = 42
RIDGE_ALPHAS_FINE = np.logspace(-3, 5, 50)
ENET_ALPHAS = np.logspace(-4, 2, 20)
R2_FILTER_THRESHOLD = 0.0

os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================================
# Data loading & OOF caching
# ============================================================================

def load_rice_trait(trait_name):
    data = np.load("results/rice_data/genotype_matrix.npz", allow_pickle=True)
    G = data['G']
    with open("results/rice_data/trait_data.json") as f:
        trait_info = json.load(f)
    td = trait_info[trait_name]
    idxs = td['genotype_indices']
    y = np.array(td['values']).astype(np.float32)
    X_t = G[idxs]
    mask = ~np.isnan(y)
    return X_t[mask], y[mask]


def get_cache_path(trait_name):
    return os.path.join(CACHE_DIR, f"oof_{trait_name}.pkl")


def train_all_models_cached(trait_name, force_retrain=False):
    """Train all models with 3-fold CV, cache OOF predictions."""
    cache_path = get_cache_path(trait_name)
    if not force_retrain and os.path.exists(cache_path):
        print(f"  [CACHE HIT] Loading OOF from {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data['oof'], data['y'], data['metrics']

    print(f"\n  [TRAINING] Full model training for {trait_name}...")
    X_all, y_all = load_rice_trait(trait_name)
    n_snps = min(ge.GWAS_TOP_K, max(50, X_all.shape[1] - 50))
    n_samples = len(y_all)
    print(f"  {n_samples} samples, {n_snps} SNPs")

    trad_models = ge.TRAD_NAMES
    dl_models = ge.DL_NAMES
    all_models = trad_models + dl_models

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=ge.RANDOM_SEED)
    oof = {m: np.zeros(n_samples) for m in all_models}
    fold_r2 = {m: [] for m in all_models}

    for fi, (tr, te) in enumerate(kf.split(X_all)):
        print(f"\n  --- Fold {fi+1}/{N_FOLDS} ---")
        Xtr_raw, Xte_raw = X_all[tr], X_all[te]
        ytr, yte = y_all[tr], y_all[te]

        maf_idx = ge.maf_filter(Xtr_raw)
        if len(maf_idx) >= n_snps:
            Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]

        gidx = ge.gwas_select(Xtr_raw, ytr, n_snps)
        Xtr = Xtr_raw[:, gidx]; Xte = Xte_raw[:, gidx]
        Xtr_s = Xtr.astype(np.float32); Xte_s = Xte.astype(np.float32)

        # Traditional models
        G_train = Xtr_s @ Xtr_s.T / n_snps
        G_te_tr = Xte_s @ Xtr_s.T / n_snps
        trad_configs = ge._make_trad_configs(G_train, G_te_tr, n_snps, len(tr))
        for tname, build_fn, fit_fn, pred_fn, param_count in trad_configs:
            tmodel = build_fn()
            fit_fn(tmodel, Xtr_s, ytr)
            preds = pred_fn(tmodel, Xte_s)
            oof[tname][te] = preds
            fold_r2[tname].append(float(r2_score(yte, preds)))

        # DL models
        for mname in dl_models:
            try:
                overrides = {'hidden': 96} if mname in ('FGN v4', 'FGN v7') else {}
                model = ge.create_model(mname, n_snps, overrides=overrides)
                bs = 32 if mname.startswith('FGN') or mname in ('GenomicFM', 'FusionNet') else 64
                model = ge.train_torch_model(model, Xtr_s, ytr, epochs=300,
                                              batch_size=bs, lr=2e-3, weight_decay=1e-3,
                                              patience=35, use_swa=True)
                preds = ge.predict_torch_model(model, Xte_s)
                oof[mname][te] = preds
                fold_r2[mname].append(float(r2_score(yte, preds)))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"    {mname:<22s} FAILED: {e}")
                oof[mname][te] = np.mean(ytr)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute per-model metrics
    metrics = {}
    for mname in all_models:
        r2_v = float(r2_score(y_all, oof[mname]))
        corr_v = float(pearsonr(y_all, oof[mname])[0])
        metrics[mname] = {'R2': r2_v, 'r': corr_v}

    # Cache
    with open(cache_path, 'wb') as f:
        pickle.dump({'oof': oof, 'y': y_all, 'metrics': metrics}, f)
    print(f"  [CACHED] OOF saved to {cache_path}")

    return oof, y_all, metrics


# ============================================================================
# Improvement 1: R² pre-filter
# ============================================================================

def filter_by_r2(oof_dict, y, threshold=0.0):
    """Remove models with R² < threshold. Returns filtered dict + removed list."""
    kept = {}
    removed = []
    for mname, preds in oof_dict.items():
        if float(r2_score(y, preds)) >= threshold:
            kept[mname] = preds
        else:
            removed.append(mname)
    return kept, removed


# ============================================================================
# Improvement 2+3: ElasticNetCV stacking + Greedy forward selection
# ============================================================================

def build_X_stack(model_list, oof):
    return np.column_stack([oof[m] for m in model_list])


def greedy_forward_select(oof_dict, y, meta_type='ElasticNet',
                          min_gain=0.0005, max_models=10):
    """Greedy forward model selection via nested 3-fold CV on meta-features."""
    names = list(oof_dict.keys())
    if len(names) <= 1:
        return names

    scores = {m: float(r2_score(y, oof_dict[m])) for m in names}
    ranked = sorted(names, key=lambda m: scores[m], reverse=True)
    selected = [ranked[0]]
    pool = ranked[1:]

    inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)

    def _eval_subset(sel):
        X = np.column_stack([oof_dict[m] for m in sel])
        preds = np.zeros(len(y))
        for itr, ite in inner_kf.split(X):
            if meta_type == 'ElasticNet':
                m = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                                 alphas=ENET_ALPHAS, cv=3, max_iter=10000, random_state=42)
            elif meta_type == 'Lasso':
                m = LassoCV(alphas=np.logspace(-4, 2, 30), cv=3, max_iter=10000, random_state=42)
            else:
                m = RidgeCV(alphas=RIDGE_ALPHAS_FINE, fit_intercept=True, cv=3)
            m.fit(X[itr], y[itr])
            preds[ite] = m.predict(X[ite])
        return float(r2_score(y, preds))

    best_r2 = _eval_subset(selected)

    while pool and len(selected) < max_models:
        gains = []
        for cand in pool[:min(12, len(pool))]:
            trial_r2 = _eval_subset(selected + [cand])
            gains.append((cand, trial_r2 - best_r2, trial_r2))
        gains.sort(key=lambda x: -x[2])
        best_cand, best_gain, best_trial_r2 = gains[0]
        if best_gain < min_gain:
            break
        selected.append(best_cand)
        pool.remove(best_cand)
        best_r2 = best_trial_r2

    return selected


def eval_stacking(model_list, oof_dict, y, meta_type='ElasticNet'):
    """Evaluate stacking with nested 3-fold CV. Returns {R2, Correlation, n_active, weights, models}."""
    X = np.column_stack([oof_dict[m] for m in model_list])
    n_models = X.shape[1]
    if n_models < 1:
        return None
    if n_models == 1:
        r2_v = float(r2_score(y, X[:, 0]))
        corr_v = float(pearsonr(y, X[:, 0])[0])
        return {'R2': r2_v, 'Correlation': corr_v, 'n_active': 1,
                'weights': [1.0], 'models': model_list, 'meta': meta_type}

    inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
    preds = np.zeros(len(y))
    all_weights = []

    for itr, ite in inner_kf.split(X):
        X_tr, X_te = X[itr], X[ite]
        if meta_type == 'ElasticNet':
            m = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                             alphas=ENET_ALPHAS, cv=3, max_iter=10000, random_state=42)
        elif meta_type == 'Lasso':
            m = LassoCV(alphas=np.logspace(-4, 2, 30), cv=3, max_iter=10000, random_state=42)
        elif meta_type == 'Ridge':
            m = RidgeCV(alphas=RIDGE_ALPHAS_FINE, fit_intercept=True, cv=3)
        else:
            m = RidgeCV(alphas=RIDGE_ALPHAS_FINE, fit_intercept=True, cv=3)
        m.fit(X_tr, y[itr])
        preds[ite] = m.predict(X_te)
        all_weights.append(m.coef_.copy())

    r2_v = float(r2_score(y, preds))
    corr_v = float(pearsonr(y, preds)[0])
    avg_weights = np.mean(all_weights, axis=0)
    n_nonzero = int(np.sum(np.abs(avg_weights) > 1e-6))
    return {'R2': r2_v, 'Correlation': corr_v, 'n_active': n_nonzero,
            'weights': avg_weights.tolist(), 'models': model_list, 'meta': meta_type}


# ============================================================================
# Main comparison
# ============================================================================

def run_comparison(trait_name="Plant_height", force_retrain=False):
    print(f"\n{'='*70}")
    print(f"  Stacking Comparison — {trait_name}")
    print(f"{'='*70}")

    oof, y, metrics = train_all_models_cached(trait_name, force_retrain=force_retrain)
    all_models = list(oof.keys())

    # Per-model summary
    print(f"\n  Per-Model OOF Performance:")
    for m in sorted(all_models, key=lambda m: metrics[m]['R2'], reverse=True):
        r2_v = metrics[m]['R2']
        marker = " [R²<0 FILTERED]" if r2_v < R2_FILTER_THRESHOLD else ""
        print(f"    {m:<22s} R²={r2_v:+.4f}  r={metrics[m]['r']:+.4f}{marker}")

    best_single = max(metrics.items(), key=lambda x: x[1]['R2'])
    best_single_r2 = best_single[1]['R2']

    # =========================================================================
    # Baseline: current genemic_ensemble.py Stacking (All) with Ridge, no filter
    # =========================================================================
    print(f"\n  {'='*60}")
    print(f"  BASELINE vs IMPROVED")
    print(f"  {'='*60}")

    results = {}

    # A: Baseline Stacking (All) — Ridge, all models (current default)
    bl = eval_stacking(all_models, oof, y, meta_type='Ridge')
    results['Baseline: Stacking(All) Ridge'] = bl
    print(f"\n  [Baseline] Stacking(All) Ridge:")
    print(f"    R²={bl['R2']:+.4f}, n_models={len(all_models)}, n_active={bl['n_active']}")

    # B: R² filter + ElasticNetCV
    oof_filtered, removed = filter_by_r2(oof, y, threshold=R2_FILTER_THRESHOLD)
    if len(oof_filtered) >= 2:
        fe = eval_stacking(list(oof_filtered.keys()), oof_filtered, y, meta_type='ElasticNet')
        results['Improved: R²-filter + ElasticNet'] = fe
        print(f"\n  [Improved] R²-filter + ElasticNetCV:")
        print(f"    Filtered: {len(removed)} models removed (R²<{R2_FILTER_THRESHOLD}): {removed}")
        print(f"    R²={fe['R2']:+.4f}, n_models={len(oof_filtered)}, n_active={fe['n_active']}")
        if fe['n_active'] < len(oof_filtered):
            active = [fe['models'][i] for i, w in enumerate(fe['weights']) if abs(w) > 1e-6]
            print(f"    ElasticNet zeroed out: {len(oof_filtered) - fe['n_active']} models")
            print(f"    Active: {active}")
        weights_str = dict(zip(fe['models'], [f'{w:.4f}' for w in fe['weights']]))
        print(f"    Weights: {weights_str}")
    else:
        print(f"\n  [Improved] R²-filter + ElasticNet: SKIP (only {len(oof_filtered)} models left)")

    # C: R² filter + Greedy forward selection + ElasticNet
    if len(oof_filtered) >= 3:
        selected = greedy_forward_select(oof_filtered, y, meta_type='ElasticNet', min_gain=0.0005)
        fg = eval_stacking(selected, oof_filtered, y, meta_type='ElasticNet')
        results['Improved: R²-filter + Greedy + ElasticNet'] = fg
        print(f"\n  [Improved] R²-filter + Greedy(ElasticNet):")
        print(f"    Greedy selected {len(selected)}/{len(oof_filtered)}: {selected}")
        print(f"    R²={fg['R2']:+.4f}, n_active={fg['n_active']}")
        weights_str = dict(zip(fg['models'], [f'{w:.4f}' for w in fg['weights']]))
        print(f"    Weights: {weights_str}")
    else:
        print(f"\n  [Improved] Greedy: SKIP (need >= 3 models after filter, have {len(oof_filtered)})")

    # D: Best single model (lower bound)
    results['Best Single Model'] = {
        'R2': best_single_r2, 'Correlation': best_single[1]['r'],
        'n_active': 1, 'weights': [1.0], 'models': [best_single[0]], 'meta': 'Single'
    }

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n  {'='*60}")
    print(f"  FINAL RANKING — {trait_name}")
    print(f"  {'='*60}")
    print(f"  {'Rank':<5s} {'Strategy':<45s} {'R²':>8s} {'Δ vs Baseline':>12s} {'n_mod':>6s}")
    print(f"  {'-'*75}")

    baseline_r2 = results['Baseline: Stacking(All) Ridge']['R2']
    sorted_items = sorted(results.items(), key=lambda x: -x[1]['R2'])

    for i, (name, res) in enumerate(sorted_items):
        delta = res['R2'] - baseline_r2
        n_mod = len(res.get('models', []))
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1:<5d} {name:<45s} {res['R2']:+.4f}  {delta:+.4f}        {n_mod:>5d}{marker}")

    # Save
    os.makedirs("results", exist_ok=True)
    output = {
        'trait': trait_name,
        'baseline_r2': float(baseline_r2),
        'best_single_r2': float(best_single_r2),
        'results': {k: {'R2': float(v['R2']), 'n_active': v['n_active'],
                         'n_models': len(v.get('models', [])),
                         'models': v.get('models', []),
                         'weights': v.get('weights', [])}
                     for k, v in results.items()},
        'ranking': [(name, float(res['R2'])) for name, res in sorted_items],
    }
    outpath = f"results/stacking_comp_{trait_name}.json"
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {outpath}")

    return results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stacking optimizer comparison')
    parser.add_argument('--trait', type=str, default='Plant_height',
                        help='Trait name (default: Plant_height)')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain all models (ignore cache)')
    args = parser.parse_args()

    run_comparison(args.trait, force_retrain=args.retrain)
