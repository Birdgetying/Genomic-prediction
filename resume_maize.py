#!/usr/bin/env python
"""Resume maize ensemble from Drought_dtm (Drought_dth already done)."""
import json, time, os, sys, random
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Fix Windows GBK
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

random.seed(42); np.random.seed(42)

# Import everything from genomic_ensemble
import genomic_ensemble as ge

DEVICE = ge.DEVICE
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

_script_dir = ge._script_dir if hasattr(ge, '_script_dir') else ge.Path(__file__).resolve().parent
output_dir = _script_dir / "results" / "maize_ensemble"

# Load existing results
intermediate_path = output_dir / "ensemble_intermediate.json"
with open(intermediate_path, 'r', encoding='utf-8') as f:
    all_results = json.load(f)

done_traits = set(all_results.keys())
print(f"Already completed: {sorted(done_traits)}")

# Load data
print("\nLoading Iranian maize data...")
X_all, y_dict = ge.load_iranian_data(max_markers=None)

var_thresh = 0.005
vars_per_marker = np.var(X_all, axis=0)
keep = vars_per_marker >= var_thresh
if keep.sum() < X_all.shape[1]:
    X_all = X_all[:, keep]
print(f"  Low-variance filter: {X_all.shape[1]} markers kept")

traits = sorted(y_dict.keys())
remaining = [t for t in traits if t not in done_traits]
print(f"Traits: {traits}")
print(f"Remaining: {remaining}")

if not remaining:
    print("All traits done! Exiting.")
    sys.exit(0)

folds_run = ge.N_FOLDS
total_t0 = time.time()

for trait in remaining:
    print(f"\n{'='*60}\nTrait: {trait}\n{'='*60}")
    y = y_dict[trait].astype(np.float32)
    n_snps = min(ge.GWAS_TOP_K, max(50, X_all.shape[1] - 50))
    print(f"  {len(y)} samples, {X_all.shape[1]} markers -> {n_snps} GWAS-selected")

    kf = KFold(n_splits=folds_run, shuffle=True, random_state=ge.RANDOM_SEED)
    results = {m: {'preds': [], 'targets': [], 'params': 0, 'time': 0.0} for m in ge.ALL_NAMES}
    oof_trad = {m: np.zeros(len(y)) for m in ge.TRAD_NAMES}
    oof_dl = {m: np.zeros(len(y)) for m in ge.DL_BASE_NAMES}

    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X_all)):
        print(f"\n  --- Fold {fold_i+1}/{folds_run} ---")
        Xtr_raw, Xte_raw = X_all[tr_idx], X_all[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        maf_idx = ge.maf_filter(Xtr_raw, ge.MAF_THRESHOLD)
        if len(maf_idx) >= n_snps:
            gidx = ge.gwas_select(Xtr_raw[:, maf_idx], ytr, n_snps)
            gidx = maf_idx[gidx]
        else:
            gidx = ge.gwas_select(Xtr_raw, ytr, n_snps)
        Xtr = Xtr_raw[:, gidx]; Xte = Xte_raw[:, gidx]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
        Xte_s = sc.transform(Xte).astype(np.float32)

        G_fold_train = Xtr_s @ Xtr_s.T / n_snps
        G_fold_te_tr = Xte_s @ Xtr_s.T / n_snps

        # Traditional models
        trad_configs = ge._make_trad_configs(G_fold_train, G_fold_te_tr, n_snps, len(tr_idx))
        for tname, build_fn, fit_fn, pred_fn, param_count in trad_configs:
            t0 = time.time()
            tmodel = build_fn()
            fit_fn(tmodel, Xtr_s, ytr)
            preds = pred_fn(tmodel, Xte_s)
            results[tname]['preds'].extend(preds.tolist())
            results[tname]['targets'].extend(yte.tolist())
            results[tname]['time'] += time.time() - t0
            if fold_i == 0:
                results[tname]['params'] = param_count
            oof_trad[tname][te_idx] = preds
            print(f"    {tname:<16s} R2={r2_score(yte, preds):+.4f}")

        # DL models
        for mi, mname in enumerate(ge.DL_NAMES):
            model = ge.create_model(mname, n_snps)
            t0 = time.time()
            if fold_i == 0:
                results[mname]['params'] = sum(p.numel() for p in model.parameters())
            bs = 64 if mname in ('FusionNet', 'AdditiveGenomicNet') else 128
            bs = 32 if mname.startswith('FGN') or mname == 'GenomicFM' else bs
            wd = 5e-3 if mname == 'AdditiveGenomicNet' else 1e-3
            model = ge.train_torch_model(model, Xtr_s, ytr, epochs=300,
                                         batch_size=bs, lr=2e-3, weight_decay=wd, patience=30)
            preds = ge.predict_torch_model(model, Xte_s)
            elapsed = time.time() - t0
            results[mname]['preds'].extend(preds.tolist())
            results[mname]['targets'].extend(yte.tolist())
            results[mname]['time'] += elapsed
            print(f"    {mname:<22s} R2={r2_score(yte, preds):+.4f}  ({elapsed:.1f}s)")
            if mname in ge.DL_BASE_NAMES:
                oof_dl[mname][te_idx] = preds
        torch.cuda.empty_cache()

    # Trait summary
    print(f"\n  {'-'*70}\n  {trait} Final Results:\n  {'Model':<16s} {'R2':>8s} {'Corr':>8s} {'RMSE':>8s}\n  {'-'*70}")
    trait_res = {}
    for mname in ge.ALL_NAMES:
        p = np.array(results[mname]['preds']); t = np.array(results[mname]['targets'])
        r2_v = float(r2_score(t, p))
        corr_v = float(pearsonr(t, p)[0])
        rmse_v = float(np.sqrt(np.mean((p - t) ** 2)))
        mtype = ge._model_type(mname)
        trait_res[mname] = {'R2': r2_v, 'Correlation': corr_v, 'RMSE': rmse_v,
                            'Type': mtype, 'Time': results[mname]['time'] / folds_run}
        print(f"  {mname:<20s} R2={r2_v:+.4f}  Corr={corr_v:+.4f}  RMSE={rmse_v:.4f}")

    ge._add_stacking_to_results(oof_dl, oof_trad, y, trait_res, folds_run)
    all_results[trait] = trait_res
    with open(intermediate_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    ge.deploy_models(X_all, y, n_snps, trait, output_dir, None, quick_test=False)

# Final summary
ge._print_final_summary(all_results, traits, output_dir, total_t0, 'Maize')
print("\nResume complete!")
