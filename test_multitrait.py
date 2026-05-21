#!/usr/bin/env python3
"""Multi-trait test: compare FGN v4+SWA vs XGBoost across 10 rice traits."""
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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

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


def run_multitrait():
    traits = ['Heading_date', 'Plant_height', 'Num_panicles', 'Num_effective_panicles',
              'Yield', 'Grain_weight', 'Spikelet_length', 'Grain_length',
              'Grain_width', 'Grain_thickness']
    models = ['XGBoost', 'FGN v4']
    N_FOLDS = 3

    all_results = {t: {} for t in traits}

    for trait_name in traits:
        print(f"\n{'='*60}")
        print(f"  {trait_name}")
        print(f"{'='*60}")
        X_all, y_all = load_rice_trait(trait_name)
        n_snps = min(ge.GWAS_TOP_K, max(50, X_all.shape[1] - 50))
        print(f"  Samples: {len(y_all)}, Markers: {X_all.shape[1]} -> {n_snps}")

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=ge.RANDOM_SEED)
        results = {m: {'preds': [], 'targets': []} for m in models}

        for fi, (tr, te) in enumerate(kf.split(X_all)):
            Xtr_raw, Xte_raw = X_all[tr], X_all[te]
            ytr, yte = y_all[tr], y_all[te]

            maf_idx = ge.maf_filter(Xtr_raw)
            if len(maf_idx) >= n_snps:
                Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]

            gidx = ge.gwas_select(Xtr_raw, ytr, n_snps)
            Xtr = Xtr_raw[:, gidx]
            Xte = Xte_raw[:, gidx]
            # NO StandardScaler — preserves discrete SNP structure
            Xtr_s = Xtr.astype(np.float32)
            Xte_s = Xte.astype(np.float32)

            # XGBoost
            xgb_model = ge.XGBoostModel(n_estimators=300)
            xgb_model.fit(Xtr_s, ytr)
            xgb_preds = xgb_model.predict(Xte_s)
            results['XGBoost']['preds'].extend(xgb_preds.tolist())
            results['XGBoost']['targets'].extend(yte.tolist())
            r2_xgb = r2_score(yte, xgb_preds)
            print(f"  Fold {fi+1}: XGBoost R2={r2_xgb:+.4f}")

            # FGN v4 + SWA (best config)
            model = ge.create_model('FGN v4', n_snps, overrides={'hidden': 96})
            t0 = time.time()
            model = ge.train_torch_model(model, Xtr_s, ytr, epochs=400, batch_size=32,
                                         lr=2e-3, weight_decay=1e-3, patience=40,
                                         use_swa=True)
            preds = ge.predict_torch_model(model, Xte_s)
            elapsed = time.time() - t0
            results['FGN v4']['preds'].extend(preds.tolist())
            results['FGN v4']['targets'].extend(yte.tolist())
            r2_fgn = r2_score(yte, preds)
            print(f"  Fold {fi+1}: FGN v4 R2={r2_fgn:+.4f} ({elapsed:.1f}s)")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for mname in models:
            p = np.array(results[mname]['preds'])
            t = np.array(results[mname]['targets'])
            r2_v = float(r2_score(t, p))
            corr_v = float(pearsonr(t, p)[0])
            all_results[trait_name][mname] = {'R2': r2_v, 'Correlation': corr_v}
        xgb_r2 = all_results[trait_name]['XGBoost']['R2']
        v4_r2 = all_results[trait_name]['FGN v4']['R2']
        print(f"  => XGB: {xgb_r2:+.4f}, v4: {v4_r2:+.4f}, "
              f"v4-XGB: {v4_r2-xgb_r2:+.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  {'Trait':<25s} {'XGBoost':>8s} {'FGN v4':>8s} {'v4-XGB':>8s}")
    print(f"  {'-'*50}")
    xgb_avg, v4_avg = [], []
    for t in traits:
        x = all_results[t]['XGBoost']['R2']
        v4 = all_results[t]['FGN v4']['R2']
        xgb_avg.append(x)
        v4_avg.append(v4)
        print(f"  {t:<25s} {x:+.4f}     {v4:+.4f}     {v4-x:+.4f}")
    print(f"  {'-'*50}")
    print(f"  {'AVERAGE':<25s} {np.mean(xgb_avg):+.4f}     {np.mean(v4_avg):+.4f}     {np.mean(v4_avg)-np.mean(xgb_avg):+.4f}")

    with open("results/multitrait_swa.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/multitrait_swa.json")


if __name__ == '__main__':
    run_multitrait()
