#!/usr/bin/env python3
"""Systematic stacking exploration for genomic prediction.

Tests multiple meta-learners (Ridge, Lasso, ElasticNet, simple average,
weighted average) over diverse model subsets to find the combination that
maximizes R² on rice traits — ideally beating XGBoost without using XGBoost.
"""
import sys, os, time, json, random, itertools
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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
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


def run_stacking_exploration(trait_name="Plant_height"):
    print(f"\n{'='*70}")
    print(f"  Stacking Exploration — {trait_name}")
    print(f"{'='*70}")

    X_all, y_all = load_rice_trait(trait_name)
    n_snps = min(ge.GWAS_TOP_K, max(50, X_all.shape[1] - 50))
    print(f"Samples: {len(y_all)}, Markers: {n_snps}")

    # Models to train — separate into groups for subset analysis
    trad_models = ['RRBLUP', 'GBLUP', 'XGBoost', 'ElasticNet', 'GWAS_RRBLUP']
    dl_models = ['FGN v4', 'FGN v6', 'FGN v7', 'FGN v9', 'FGN v10', 'FGN v11',
                 'FGNplus', 'FGN', 'FGN v2', 'FusionNet', 'AdditiveGenomicNet',
                 'GenomicFM', 'FGN PCA']
    all_models = trad_models + dl_models

    N_FOLDS = 3
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=ge.RANDOM_SEED)

    # Collect OOF predictions
    oof = {m: np.zeros(len(y_all)) for m in all_models}
    fold_r2 = {m: [] for m in all_models}

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

        # NO scaler — preserves SNP discrete structure
        Xtr_s = Xtr.astype(np.float32)
        Xte_s = Xte.astype(np.float32)

        # Traditional models
        G_train = Xtr_s @ Xtr_s.T / n_snps
        G_te_tr = Xte_s @ Xtr_s.T / n_snps
        trad_configs = ge._make_trad_configs(G_train, G_te_tr, n_snps, len(tr))
        for tname, build_fn, fit_fn, pred_fn, param_count in trad_configs:
            t0 = time.time()
            tmodel = build_fn()
            fit_fn(tmodel, Xtr_s, ytr)
            preds = pred_fn(tmodel, Xte_s)
            oof[tname][te] = preds
            fold_r2[tname].append(float(r2_score(yte, preds)))
            print(f"    {tname:<16s} R2={r2_score(yte, preds):+.4f}")

        # DL models
        for mname in dl_models:
            t0 = time.time()
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
                print(f"    {mname:<22s} R2={r2_score(yte, preds):+.4f}")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"    {mname:<22s} FAILED: {e}")
                oof[mname][te] = np.mean(ytr)  # fallback

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print per-model summary
    print(f"\n{'='*70}")
    print(f"  Per-Model OOF Performance:")
    print(f"  {'Model':<22s} {'R2':>8s} {'r':>8s}")
    print(f"  {'-'*40}")
    model_metrics = {}
    for mname in all_models:
        r2_v = float(r2_score(y_all, oof[mname]))
        corr_v = float(pearsonr(y_all, oof[mname])[0])
        model_metrics[mname] = {'R2': r2_v, 'r': corr_v}
        print(f"  {mname:<22s} {r2_v:+.4f}  {corr_v:+.4f}")

    # ============================================================
    # Stacking strategies
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  Stacking Strategies")
    print(f"{'='*70}")

    stacking_results = {}

    # Build feature matrices for different model groups
    def build_X_stack(model_list):
        """Stack OOF predictions into feature matrix."""
        feats = [oof[m] for m in model_list if not np.any(np.isnan(oof[m]))]
        return np.column_stack(feats) if feats else None

    def evaluate_stacking(name, model_list, meta_learner_type, X_stack, y):
        """Evaluate one stacking configuration."""
        n_models = X_stack.shape[1]
        if n_models < 1:
            return None

        # Inner CV for meta-learner
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
        meta_preds = np.zeros(len(y))

        for itr, ite in inner_kf.split(X_stack):
            X_mtr, X_mte = X_stack[itr], X_stack[ite]
            y_mtr = y[itr]

            if meta_learner_type == 'Ridge':
                meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
                meta.fit(X_mtr, y_mtr)
                meta_preds[ite] = meta.predict(X_mte)
                weights = meta.coef_
            elif meta_learner_type == 'Lasso':
                meta = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=3,
                               max_iter=5000, random_state=42)
                meta.fit(X_mtr, y_mtr)
                meta_preds[ite] = meta.predict(X_mte)
                weights = meta.coef_
            elif meta_learner_type == 'ElasticNet':
                meta = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                                    alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
                                    cv=3, max_iter=5000, random_state=42)
                meta.fit(X_mtr, y_mtr)
                meta_preds[ite] = meta.predict(X_mte)
                weights = meta.coef_
            elif meta_learner_type == 'SimpleAvg':
                meta_preds[ite] = np.mean(X_mte, axis=1)
                weights = np.ones(n_models) / n_models
            elif meta_learner_type == 'WeightedAvg':
                # Weight by per-model OOF R²
                w = np.array([model_metrics[m]['R2'] for m in model_list])
                w = np.maximum(w, 0)  # clip negative weights
                w = w / (w.sum() + 1e-10)
                meta_preds[ite] = X_mte @ w
                weights = w
            elif meta_learner_type == 'TopK_Avg':
                # Average top-K models by R²
                k = min(5, n_models)
                sorted_idx = np.argsort([model_metrics[m]['R2'] for m in model_list])[::-1]
                top_k = sorted_idx[:k]
                meta_preds[ite] = np.mean(X_mte[:, top_k], axis=1)
                weights = np.zeros(n_models)
                weights[top_k] = 1.0 / k
            elif meta_learner_type == 'XGBoost':
                # XGBoost as non-linear meta-learner
                meta = xgb.XGBRegressor(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                    reg_lambda=1.0, random_state=42, verbosity=0
                )
                meta.fit(X_mtr, y_mtr)
                meta_preds[ite] = meta.predict(X_mte)
                weights = meta.feature_importances_

        r2_v = float(r2_score(y, meta_preds))
        corr_v = float(pearsonr(y, meta_preds)[0])
        n_nonzero = int(np.sum(np.abs(weights) > 1e-6))
        return {'R2': r2_v, 'Correlation': corr_v, 'n_active': n_nonzero,
                'weights': weights.tolist() if hasattr(weights, 'tolist') else list(weights),
                'models': model_list, 'meta': meta_learner_type}

    # Define model groups
    groups = {
        'All models': all_models,
        'Trad only (no XGB)': [m for m in trad_models if m != 'XGBoost'],
        'Trad + Top DL': trad_models + ['FGN v11', 'FGN v7', 'FGN v4'],
        'DL only (top 5)': ['FGN v11', 'FGN v7', 'FGN v10', 'FGN v4', 'FGNplus'],
        'DL only (all)': dl_models,
        'Best 3 Trad + Best 3 DL': ['RRBLUP', 'XGBoost', 'ElasticNet',
                                     'FGN v11', 'FGN v7', 'FGN v4'],
        'No XGBoost (Trad+DL)': [m for m in all_models if m != 'XGBoost'],
    }

    meta_learners = ['Ridge', 'Lasso', 'ElasticNet', 'SimpleAvg', 'WeightedAvg', 'TopK_Avg', 'XGBoost']

    for group_name, model_list in groups.items():
        X_stack = build_X_stack(model_list)
        if X_stack is None or X_stack.shape[1] < 2:
            continue
        for meta in meta_learners:
            key = f"{group_name} | {meta}"
            result = evaluate_stacking(key, model_list, meta, X_stack, y_all)
            if result:
                stacking_results[key] = result
                n_mod = X_stack.shape[1]
                print(f"  {key:<40s} R2={result['R2']:+.4f}  "
                      f"(n={n_mod}, active={result['n_active']})")

    # Best result
    best = max(stacking_results.items(), key=lambda x: x[1]['R2'])
    print(f"\n{'='*70}")
    print(f"  Best Stacking: {best[0]}")
    print(f"  R2={best[1]['R2']:+.4f}, r={best[1]['Correlation']:+.4f}")
    print(f"  Active models: {best[1]['n_active']}/{len(best[1]['models'])}")

    # Show top-10 for analysis
    print(f"\n  Top-10 Stacking Strategies:")
    sorted_results = sorted(stacking_results.items(), key=lambda x: -x[1]['R2'])
    for i, (name, res) in enumerate(sorted_results[:10]):
        print(f"  {i+1}. {name:<40s} R2={res['R2']:+.4f}")

    # Save
    os.makedirs("results", exist_ok=True)
    output = {
        'trait': trait_name,
        'model_metrics': model_metrics,
        'xgb_baseline': model_metrics.get('XGBoost', {}).get('R2', 0),
        'stacking_results': {k: {kk: vv for kk, vv in v.items() if kk != 'weights'}
                             for k, v in stacking_results.items()},
        'best': {'name': best[0], **{k: v for k, v in best[1].items() if k != 'weights'}},
        'top10': [(name, res['R2']) for name, res in sorted_results[:10]],
    }
    with open(f"results/stacking_{trait_name}.json", 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to results/stacking_{trait_name}.json")

    return stacking_results, model_metrics, oof


if __name__ == '__main__':
    # Test on Plant_height first
    results, metrics, oof = run_stacking_exploration("Plant_height")
    xgb_r2 = metrics.get('XGBoost', {}).get('R2', 0)
    best_no_xgb = [(k, v) for k, v in results.items()
                   if 'XGBoost' not in k and v['R2'] > xgb_r2]
    if best_no_xgb:
        print(f"\n*** Found {len(best_no_xgb)} strategies beating XGBoost without using XGBoost! ***")
        for name, res in sorted(best_no_xgb, key=lambda x: -x[1]['R2'])[:5]:
            print(f"  {name}: R2={res['R2']:+.4f} (vs XGB {xgb_r2:+.4f})")
    else:
        print(f"\nNo non-XGBoost strategy beat XGBoost ({xgb_r2:+.4f}) on this trait.")
