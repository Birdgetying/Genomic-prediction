#!/usr/bin/env python3
"""Quick test: run genomic_ensemble pipeline on Rice SNP data (529 samples, 360K markers)."""
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


def load_rice_trait(trait_name="Grain_length"):
    """Load a single rice trait from pre-processed data."""
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


def run_rice_quicktest():
    print("=" * 70)
    print("  Rice Data Quick Test — Genomic Ensemble Pipeline")
    print("=" * 70)

    X_all, y_all = load_rice_trait("Plant_height")
    n_snps = min(ge.GWAS_TOP_K, max(50, X_all.shape[1] - 50))
    print(f"\nSamples: {len(y_all)}, Markers: {X_all.shape[1]} -> {n_snps} GWAS-selected")
    print(f"Genotype range: [{X_all.min():.0f}, {X_all.max():.0f}]")
    print(f"Phenotype range: [{y_all.min():.4f}, {y_all.max():.4f}]")
    print(f"Device: {ge.DEVICE}")

    N_FOLDS = 3
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=ge.RANDOM_SEED)
    results = {m: {'preds': [], 'targets': [], 'time': 0.0} for m in ge.ALL_NAMES}
    oof_trad = {m: np.zeros(len(y_all)) for m in ge.TRAD_NAMES}
    oof_dl = {m: np.zeros(len(y_all)) for m in ge.DL_BASE_NAMES}

    for fi, (tr, te) in enumerate(kf.split(X_all)):
        print(f"\n  --- Fold {fi+1}/{N_FOLDS} ---")
        Xtr_raw, Xte_raw = X_all[tr], X_all[te]
        ytr, yte = y_all[tr], y_all[te]

        # MAF filter
        maf_idx = ge.maf_filter(Xtr_raw)
        if len(maf_idx) >= n_snps:
            Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]

        # GWAS marker selection
        gidx = ge.gwas_select(Xtr_raw, ytr, n_snps)
        Xtr = Xtr_raw[:, gidx]
        Xte = Xte_raw[:, gidx]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
        Xte_s = sc.transform(Xte).astype(np.float32)

        # Traditional models
        G_fold_train = Xtr_s @ Xtr_s.T / n_snps
        G_fold_te_tr = Xte_s @ Xtr_s.T / n_snps
        trad_configs = ge._make_trad_configs(G_fold_train, G_fold_te_tr, n_snps, len(tr))
        for tname, build_fn, fit_fn, pred_fn, param_count in trad_configs:
            t0 = time.time()
            tmodel = build_fn()
            fit_fn(tmodel, Xtr_s, ytr)
            preds = pred_fn(tmodel, Xte_s)
            results[tname]['preds'].extend(preds.tolist())
            results[tname]['targets'].extend(yte.tolist())
            results[tname]['time'] += time.time() - t0
            oof_trad[tname][te] = preds
            print(f"    {tname:<16s} R2={r2_score(yte, preds):+.4f}")

        # DL models
        for mname in ge.DL_NAMES:
            overrides = {}
            if mname == 'FGN':
                overrides = {'n_spec': 64, 'hidden': 80}
            elif mname == 'FGN v6':
                overrides = {'hidden': 64, 'max_freq': 384}
            elif mname in ('FGN v4', 'FGN v7'):
                overrides = {'hidden': 96}
            model = ge.create_model(mname, n_snps, overrides=overrides)
            t0 = time.time()
            bs = 64 if mname in ('FusionNet', 'AdditiveGenomicNet') else 128
            bs = 32 if mname.startswith('FGN') or mname == 'GenomicFM' else bs
            wd = 5e-3 if mname == 'AdditiveGenomicNet' else 1e-3
            model = ge.train_torch_model(model, Xtr_s, ytr, epochs=300, batch_size=bs,
                                         lr=2e-3, weight_decay=wd, patience=35, use_swa=True)
            preds = ge.predict_torch_model(model, Xte_s)
            elapsed = time.time() - t0
            results[mname]['preds'].extend(preds.tolist())
            results[mname]['targets'].extend(yte.tolist())
            results[mname]['time'] += elapsed
            print(f"    {mname:<22s} R2={r2_score(yte, preds):+.4f}  ({elapsed:.1f}s)")
            if mname in ge.DL_BASE_NAMES:
                oof_dl[mname][te] = preds
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"  Final Results (2-fold CV, {len(y_all)} samples):")
    print(f"  {'Model':<22s} {'R2':>8s} {'Corr':>8s} {'RMSE':>8s} {'Time':>8s}")
    print(f"  {'-'*60}")

    trait_res = {}
    for mname in ge.ALL_NAMES:
        p = np.array(results[mname]['preds'])
        t = np.array(results[mname]['targets'])
        r2_v = float(r2_score(t, p))
        corr_v = float(pearsonr(t, p)[0])
        rmse_v = float(np.sqrt(np.mean((p - t)**2)))
        trait_res[mname] = {'R2': r2_v, 'Correlation': corr_v, 'RMSE': rmse_v,
                            'Type': ge._model_type(mname),
                            'Time': results[mname]['time'] / N_FOLDS}
        tag = ' [Trad]' if ge._model_type(mname) == ge.TYPE_TRAD else ' [DL]'
        print(f"  {mname+tag:<22s} {r2_v:+.4f}  {corr_v:+.4f}  {rmse_v:.4f}  {results[mname]['time']:.1f}s")

    # Stacking
    ge._add_stacking_to_results(oof_dl, oof_trad, y_all, trait_res, N_FOLDS)
    for stack_name in ('Stacking (DL)', 'Stacking (All)', 'Trad Ensemble'):
        if stack_name in trait_res:
            sr = trait_res[stack_name]
            print(f"  {stack_name:<22s} {sr['R2']:+.4f}  {sr['Correlation']:+.4f}  {sr['RMSE']:.4f}")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/rice_quicktest.json", 'w') as f:
        json.dump(trait_res, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to results/rice_quicktest.json")


if __name__ == '__main__':
    run_rice_quicktest()
