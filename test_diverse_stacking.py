#!/usr/bin/env python3
"""Feature-diverse stacking: each base model sees a different random SNP subset.

Core insight: current stacking fails because all models use identical GWAS top-5000
SNPs, producing correlated predictions. This script gives each model a different
random subset to force genuine diversity.
"""
import sys, os, time, json, random
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
import xgboost as xgb

import genomic_ensemble as ge


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


def train_xgb(Xtr, ytr, Xte, seed=42):
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=0
    )
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def train_dl(mname, Xtr, ytr, Xte, n_snps, seed=42):
    overrides = {'hidden': 96} if mname in ('FGN v4', 'FGN v7') else {}
    model = ge.create_model(mname, n_snps, overrides=overrides)
    model = ge.train_torch_model(model, Xtr, ytr, epochs=300,
                                 batch_size=32, lr=2e-3, weight_decay=1e-3,
                                 patience=35, use_swa=True)
    preds = ge.predict_torch_model(model, Xte)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds


def get_subset_indices(available_snps, shared_k, unique_k, model_seed):
    """Select SNP indices: shared core + unique random subset."""
    rng = random.Random(model_seed)
    n_available = len(available_snps)
    shared = available_snps[:shared_k]  # first K = strongest GWAS hits (shared)
    pool = available_snps[shared_k:min(shared_k + 3 * unique_k, n_available)]
    unique = sorted(rng.sample(list(pool), min(unique_k, len(pool))))
    return np.array(list(shared) + list(unique))


def run_feature_diverse_stacking(trait_name="Plant_height"):
    print(f"\n{'='*60}")
    print(f"  Feature-Diverse Stacking — {trait_name}")
    print(f"{'='*60}")

    X_all, y_all = load_rice_trait(trait_name)
    n_samples = len(y_all)
    print(f"  Samples: {n_samples}")

    N_FOLDS = 3
    N_TOTAL = 5000
    N_SHARED = 2500  # shared core SNPs
    N_UNIQUE = 2500  # unique per model
    GWAS_POOL = 20000

    trad_models = ['RRBLUP', 'GBLUP', 'XGBoost', 'ElasticNet', 'GWAS_RRBLUP']
    dl_models = ['FGN v11', 'FGN v7', 'FGN v4']
    all_models = trad_models + dl_models

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=ge.RANDOM_SEED)
    oof = {m: np.zeros(n_samples) for m in all_models}

    for fi, (tr, te) in enumerate(kf.split(X_all)):
        print(f"\n  --- Fold {fi+1}/{N_FOLDS} ---")
        Xtr_raw, Xte_raw = X_all[tr], X_all[te]
        ytr, yte = y_all[tr], y_all[te]

        # MAF filter
        maf_idx = ge.maf_filter(Xtr_raw)
        if len(maf_idx) >= GWAS_POOL:
            Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]

        # GWAS to get a LARGE pool of candidate SNPs
        gwas_pool_idx = ge.gwas_select(Xtr_raw, ytr, GWAS_POOL)

        for mi, mname in enumerate(all_models):
            t0 = time.time()
            # Each model gets a DIFFERENT subset
            subset = get_subset_indices(gwas_pool_idx, N_SHARED, N_UNIQUE, model_seed=100*fi + mi)
            n_actual = len(subset)

            Xtr_s = Xtr_raw[:, subset].astype(np.float32)
            Xte_s = Xte_raw[:, subset].astype(np.float32)

            if mname == 'XGBoost':
                preds = train_xgb(Xtr_s, ytr, Xte_s, seed=42+mi)
            elif mname == 'ElasticNet':
                # Use ge's trad_configs for ElasticNet in feature-diverse mode
                n_snps_actual = n_actual
                G_train = Xtr_s @ Xtr_s.T / n_snps_actual
                G_te_tr = Xte_s @ Xtr_s.T / n_snps_actual
                configs = ge._make_trad_configs(G_train, G_te_tr, n_snps_actual, len(tr))
                for tname, build_fn, fit_fn, pred_fn, pc in configs:
                    if tname == mname:
                        tmodel = build_fn()
                        fit_fn(tmodel, Xtr_s, ytr)
                        preds = pred_fn(tmodel, Xte_s)
                        break
            elif mname in ('RRBLUP', 'GBLUP', 'GWAS_RRBLUP'):
                n_snps_actual = n_actual
                G_train = Xtr_s @ Xtr_s.T / n_snps_actual
                G_te_tr = Xte_s @ Xtr_s.T / n_snps_actual
                configs = ge._make_trad_configs(G_train, G_te_tr, n_snps_actual, len(tr))
                for tname, build_fn, fit_fn, pred_fn, pc in configs:
                    if tname == mname:
                        tmodel = build_fn()
                        fit_fn(tmodel, Xtr_s, ytr)
                        preds = pred_fn(tmodel, Xte_s)
                        break
            else:
                # DL models — different SNP subset per model
                preds = train_dl(mname, Xtr_s, ytr, Xte_s, n_actual, seed=42+mi)

            oof[mname][te] = preds
            r2_f = r2_score(yte, preds)
            marker = "*" if n_actual != N_TOTAL else " "
            print(f"    {mname:<16s} R2={r2_f:+.4f} ({n_actual} SNPs) ({time.time()-t0:.1f}s)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Per-model OOF metrics
    print(f"\n  Per-Model OOF Performance:")
    model_metrics = {}
    for m in all_models:
        r2_v = float(r2_score(y_all, oof[m]))
        corr_v = float(pearsonr(y_all, oof[m])[0])
        model_metrics[m] = {'R2': r2_v, 'r': corr_v}
        print(f"  {m:<16s} R2={r2_v:+.4f}")

    # Stacking
    X_stack = np.column_stack([oof[m] for m in all_models])
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
    meta_preds = np.zeros(n_samples)
    for itr, ite in inner_kf.split(X_stack):
        X_mtr, X_mte = X_stack[itr], X_stack[ite]
        meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        meta.fit(X_mtr, y_all[itr])
        meta_preds[ite] = meta.predict(X_mte)

    stack_r2 = float(r2_score(y_all, meta_preds))
    xgb_r2 = model_metrics['XGBoost']['R2']

    print(f"\n  Stacking (Ridge): {stack_r2:+.4f}")
    print(f"  XGBoost:          {xgb_r2:+.4f}")
    print(f"  Delta:            {stack_r2 - xgb_r2:+.4f}")

    # Also check model prediction correlation
    print(f"\n  Prediction Correlations:")
    for i, m1 in enumerate(all_models):
        for m2 in all_models[i+1:]:
            corr = np.corrcoef(oof[m1], oof[m2])[0, 1]
            if corr < 0.98:
                print(f"    {m1} vs {m2}: r={corr:.4f}")

    return stack_r2 - xgb_r2


if __name__ == '__main__':
    delta = run_feature_diverse_stacking("Plant_height")
    print(f"\nFinal: Stacking - XGBoost = {delta:+.4f}")
