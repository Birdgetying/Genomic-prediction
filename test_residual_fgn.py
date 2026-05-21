#!/usr/bin/env python3
"""Test: RRBLUP + FGN residual learning vs XGBoost on 10 rice traits.

RRBLUP captures additive effects, FGN v4 learns non-additive residuals.
Final prediction = RRBLUP(X) + FGN(X).
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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
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


def run_residual_test():
    traits = ['Heading_date', 'Plant_height', 'Num_panicles', 'Num_effective_panicles',
              'Yield', 'Grain_weight', 'Spikelet_length', 'Grain_length',
              'Grain_width', 'Grain_thickness']
    N_FOLDS = 3

    all_results = {}
    for trait_name in traits:
        print(f"\n{'='*60}")
        print(f"  {trait_name}")
        print(f"{'='*60}")
        X_all, y_all = load_rice_trait(trait_name)
        n_snps = min(ge.GWAS_TOP_K, max(50, X_all.shape[1] - 50))
        print(f"  Samples: {len(y_all)}, Markers: {n_snps}")

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=ge.RANDOM_SEED)
        results = {m: {'preds': [], 'targets': []}
                   for m in ['XGBoost', 'FGN v4', 'RRBLUP+FGN']}

        for fi, (tr, te) in enumerate(kf.split(X_all)):
            Xtr_raw, Xte_raw = X_all[tr], X_all[te]
            ytr, yte = y_all[tr], y_all[te]

            maf_idx = ge.maf_filter(Xtr_raw)
            if len(maf_idx) >= n_snps:
                Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]

            gidx = ge.gwas_select(Xtr_raw, ytr, n_snps)
            Xtr = Xtr_raw[:, gidx]
            Xte = Xte_raw[:, gidx]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
            Xte_s = sc.transform(Xte).astype(np.float32)

            # XGBoost baseline
            xgb_model = ge.XGBoostModel(n_estimators=300)
            xgb_model.fit(Xtr_s, ytr)
            xgb_preds = xgb_model.predict(Xte_s)
            results['XGBoost']['preds'].extend(xgb_preds.tolist())
            results['XGBoost']['targets'].extend(yte.tolist())
            print(f"  Fold {fi+1}: XGBoost           R2={r2_score(yte, xgb_preds):+.4f}")

            # FGN v4 + SWA (baseline)
            model = ge.create_model('FGN v4', n_snps, overrides={'hidden': 96})
            t0 = time.time()
            model = ge.train_torch_model(model, Xtr_s, ytr, epochs=300, batch_size=32,
                                         lr=2e-3, weight_decay=1e-3, patience=35, use_swa=True)
            fgn_preds = ge.predict_torch_model(model, Xte_s)
            results['FGN v4']['preds'].extend(fgn_preds.tolist())
            results['FGN v4']['targets'].extend(yte.tolist())
            print(f"  Fold {fi+1}: FGN v4            R2={r2_score(yte, fgn_preds):+.4f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # RRBLUP + FGN residual
            # Step 1: Fit RRBLUP on training data
            rr = RidgeCV(alphas=ge.RIDGE_ALPHAS)
            rr.fit(Xtr_s, ytr)
            rr_preds_train = rr.predict(Xtr_s)
            rr_preds_test = rr.predict(Xte_s)
            # Step 2: Compute residuals
            residual_train = ytr - rr_preds_train
            # Step 3: Train FGN on residuals
            model2 = ge.create_model('FGN v4', n_snps, overrides={'hidden': 64})
            model2 = ge.train_torch_model(model2, Xtr_s, residual_train, epochs=200,
                                          batch_size=32, lr=2e-3, weight_decay=1e-3,
                                          patience=25, use_swa=True)
            # Step 4: Combine predictions
            res_preds = ge.predict_torch_model(model2, Xte_s)
            combined_preds = rr_preds_test + res_preds
            results['RRBLUP+FGN']['preds'].extend(combined_preds.tolist())
            results['RRBLUP+FGN']['targets'].extend(yte.tolist())
            r2_comb = r2_score(yte, combined_preds)
            r2_rr = r2_score(yte, rr_preds_test)
            print(f"  Fold {fi+1}: RRBLUP+FGN        R2={r2_comb:+.4f}  (RRBLUP={r2_rr:+.4f})")
            del model2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        trait_res = {}
        for mname in ['XGBoost', 'FGN v4', 'RRBLUP+FGN']:
            p = np.array(results[mname]['preds'])
            t = np.array(results[mname]['targets'])
            trait_res[mname] = {'R2': float(r2_score(t, p)),
                               'Correlation': float(pearsonr(t, p)[0])}
        all_results[trait_name] = trait_res

        xgb_r2 = trait_res['XGBoost']['R2']
        fgn_r2 = trait_res['FGN v4']['R2']
        rfgn_r2 = trait_res['RRBLUP+FGN']['R2']
        print(f"  => XGB: {xgb_r2:+.4f}, FGN v4: {fgn_r2:+.4f}, "
              f"RRBLUP+FGN: {rfgn_r2:+.4f}, RF-XGB: {rfgn_r2-xgb_r2:+.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  {'Trait':<25s} {'XGBoost':>8s} {'FGN v4':>8s} {'RRBLUP+FGN':>11s} {'RF-XGB':>8s}")
    print(f"  {'-'*60}")
    xgb_avg, fgn_avg, rfgn_avg = [], [], []
    for t in traits:
        x = all_results[t]['XGBoost']['R2']
        f = all_results[t]['FGN v4']['R2']
        r = all_results[t]['RRBLUP+FGN']['R2']
        xgb_avg.append(x); fgn_avg.append(f); rfgn_avg.append(r)
        print(f"  {t:<25s} {x:+.4f}     {f:+.4f}     {r:+.4f}        {r-x:+.4f}")
    print(f"  {'-'*60}")
    print(f"  {'AVERAGE':<25s} {np.mean(xgb_avg):+.4f}     {np.mean(fgn_avg):+.4f}     {np.mean(rfgn_avg):+.4f}        {np.mean(rfgn_avg)-np.mean(xgb_avg):+.4f}")

    with open("results/residual_fgn.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/residual_fgn.json")


if __name__ == '__main__':
    run_residual_test()
