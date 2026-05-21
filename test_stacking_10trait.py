#!/usr/bin/env python3
"""10-trait stacking test with OOF prediction caching.

Only trains models whose cached OOF predictions don't exist.
Delete results/oof_cache/ to force full re-run.
"""
import sys, os, time, json, random, hashlib
import numpy as np

random.seed(42)
np.random.seed(42)

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr

import genomic_ensemble as ge

CACHE_DIR = "results/oof_cache"


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


def get_cache_path(trait_name, model_name):
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    d = os.path.join(CACHE_DIR, trait_name.replace(' ', '_'))
    return os.path.join(d, f"{safe_name}.npy")


def load_cached_oof(trait_name, models_list):
    """Load cached OOF predictions. Returns dict of model->array or None if missing."""
    result = {}
    for m in models_list:
        path = get_cache_path(trait_name, m)
        if os.path.exists(path):
            result[m] = np.load(path)
        else:
            return None  # missing cache for at least one model
    return result


def compute_config_hash(cfg_dict):
    """Simple hash for cache validation."""
    s = json.dumps(cfg_dict, sort_keys=True).encode()
    return hashlib.md5(s).hexdigest()[:8]


def run_trait(trait_name, trad_names, dl_names, N_FOLDS=3):
    print(f"\n{'='*60}")
    print(f"  {trait_name}")
    print(f"{'='*60}")

    all_names = trad_names + dl_names
    X_all, y_all = load_rice_trait(trait_name)
    n_samples = len(y_all)

    # Check cache first
    cached = load_cached_oof(trait_name, all_names)
    if cached is not None:
        print(f"  [CACHE HIT] All {len(all_names)} models cached")
        oof = cached
    else:
        n_snps = min(ge.GWAS_TOP_K, max(50, X_all.shape[1] - 50))
        print(f"  Samples: {n_samples}, Markers: {n_snps}")

        # Check per-model cache
        oof = {}
        models_to_train = []
        for m in all_names:
            cache_path = get_cache_path(trait_name, m)
            if os.path.exists(cache_path):
                oof[m] = np.load(cache_path)
                # Verify shape
                if len(oof[m]) != n_samples:
                    print(f"  [STALE] {m:<22s} cache shape mismatch, retraining")
                    models_to_train.append(m)
                    oof[m] = np.zeros(n_samples)
                else:
                    print(f"  [CACHE] {m:<22s} loaded")
            else:
                oof[m] = np.zeros(n_samples)
                models_to_train.append(m)

        if models_to_train:
            print(f"  Training {len(models_to_train)}/{len(all_names)} models: {models_to_train}")
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=ge.RANDOM_SEED)

            for fi, (tr, te) in enumerate(kf.split(X_all)):
                print(f"\n  --- Fold {fi+1}/{N_FOLDS} ---")
                Xtr_raw, Xte_raw = X_all[tr], X_all[te]
                ytr, yte = y_all[tr], y_all[te]

                maf_idx = ge.maf_filter(Xtr_raw)
                if len(maf_idx) >= n_snps:
                    Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]

                gidx = ge.gwas_select(Xtr_raw, ytr, n_snps)
                Xtr = Xtr_raw[:, gidx]
                Xte = Xte_raw[:, gidx]
                Xtr_s = Xtr.astype(np.float32)
                Xte_s = Xte.astype(np.float32)

                # Traditional models
                if any(m in models_to_train for m in trad_names):
                    G_train = Xtr_s @ Xtr_s.T / n_snps
                    G_te_tr = Xte_s @ Xtr_s.T / n_snps
                    trad_configs = ge._make_trad_configs(G_train, G_te_tr, n_snps, len(tr))
                    for tname, build_fn, fit_fn, pred_fn, param_count in trad_configs:
                        if tname not in models_to_train:
                            continue
                        t0 = time.time()
                        tmodel = build_fn()
                        fit_fn(tmodel, Xtr_s, ytr)
                        preds = pred_fn(tmodel, Xte_s)
                        oof[tname][te] = preds
                        print(f"    {tname:<16s} R2={r2_score(yte, preds):+.4f} ({time.time()-t0:.1f}s)")

                # DL models
                for mname in dl_names:
                    if mname not in models_to_train:
                        continue
                    t0 = time.time()
                    try:
                        overrides = {'hidden': 96} if mname in ('FGN v4', 'FGN v7') else {}
                        model = ge.create_model(mname, n_snps, overrides=overrides)
                        model = ge.train_torch_model(model, Xtr_s, ytr, epochs=300,
                                                     batch_size=32, lr=2e-3, weight_decay=1e-3,
                                                     patience=35, use_swa=True)
                        preds = ge.predict_torch_model(model, Xte_s)
                        oof[mname][te] = preds
                        print(f"    {mname:<22s} R2={r2_score(yte, preds):+.4f} ({time.time()-t0:.1f}s)")
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"    {mname:<22s} FAILED: {e}")
                        oof[mname][te] = np.mean(ytr)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Save newly trained models to cache
            for m in models_to_train:
                cache_path = get_cache_path(trait_name, m)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, oof[m])

    # --- Stacking evaluation (fast, always run) ---
    X_stack = np.column_stack([oof[m] for m in all_names])

    # Per-model metrics
    model_metrics = {}
    for m in all_names:
        r2_v = float(r2_score(y_all, oof[m]))
        corr_v = float(pearsonr(y_all, oof[m])[0])
        model_metrics[m] = {'R2': r2_v, 'r': corr_v}

    # Stacking: Ridge meta-learner with inner CV
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
    meta_preds = np.zeros(n_samples)
    for itr, ite in inner_kf.split(X_stack):
        X_mtr, X_mte = X_stack[itr], X_stack[ite]
        y_mtr = y_all[itr]
        meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        meta.fit(X_mtr, y_mtr)
        meta_preds[ite] = meta.predict(X_mte)

    stack_r2 = float(r2_score(y_all, meta_preds))
    stack_corr = float(pearsonr(y_all, meta_preds)[0])
    xgb_r2 = model_metrics['XGBoost']['R2']

    print(f"\n  Summary for {trait_name}:")
    print(f"  {'Model':<22s} {'R2':>8s}")
    for m in all_names:
        print(f"  {m:<22s} {model_metrics[m]['R2']:+.4f}")
    print(f"  {'Stacking (Ridge)':<22s} {stack_r2:+.4f}")
    print(f"  => Delta over XGBoost: {stack_r2 - xgb_r2:+.4f}")

    return {
        'trait': trait_name, 'n_samples': n_samples,
        'models': model_metrics, 'stacking': {'R2': stack_r2, 'r': stack_corr, 'meta': 'Ridge'},
        'xgb_baseline': xgb_r2, 'improvement': stack_r2 - xgb_r2
    }


def main():
    traits = ['Heading_date', 'Plant_height', 'Num_panicles', 'Num_effective_panicles',
              'Yield', 'Grain_weight', 'Spikelet_length', 'Grain_length',
              'Grain_width', 'Grain_thickness']

    trad_names = ['RRBLUP', 'GBLUP', 'XGBoost', 'ElasticNet', 'GWAS_RRBLUP']
    dl_names = ['FGN v11', 'FGN v7', 'FGN v4']

    print("=" * 70)
    print("  10-Trait Stacking Test (with OOF caching)")
    print(f"  Traditional: {trad_names}")
    print(f"  DL: {dl_names}")
    print("=" * 70)

    all_results = {}
    improvements, xgb_scores, stack_scores = [], [], []

    for trait_name in traits:
        res = run_trait(trait_name, trad_names, dl_names)
        all_results[trait_name] = res
        improvements.append(res['improvement'])
        xgb_scores.append(res['xgb_baseline'])
        stack_scores.append(res['stacking']['R2'])

    # Summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Trait':<25s} {'XGBoost':>8s} {'Stacking':>8s} {'Delta':>8s}")
    print(f"  {'-'*50}")
    for t in traits:
        r = all_results[t]
        print(f"  {t:<25s} {r['xgb_baseline']:+.4f}     {r['stacking']['R2']:+.4f}     {r['improvement']:+.4f}")
    print(f"  {'-'*50}")
    print(f"  {'AVERAGE':<25s} {np.mean(xgb_scores):+.4f}     {np.mean(stack_scores):+.4f}     {np.mean(improvements):+.4f}")

    win_count = sum(1 for d in improvements if d > 0)
    print(f"\n  Stacking beats XGBoost on {win_count}/{len(traits)} traits")
    print(f"  Average improvement over XGBoost: {np.mean(improvements):+.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/stacking_10trait.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/stacking_10trait.json")


if __name__ == '__main__':
    main()
