#!/usr/bin/env python
"""Rice stacking supplement: compute only missing stacking variants (Greedy/Pruned/R²+Greedy)
by re-running 5-fold CV to get OOF predictions, then adding to existing results.
Skips AutoML tuning and model deployment to save time."""
import json, time, os, sys, random
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

import genomic_ensemble as ge

DEVICE = ge.DEVICE
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load existing results
INTER_PATH = ge.Path('results/rice_ensemble/ensemble_intermediate.json')
with open(INTER_PATH, 'r', encoding='utf-8') as f:
    all_results = json.load(f)

traits = sorted(all_results.keys())
print(f"Traits: {traits}")

# Check which traits already have stacking variants
existing_stacking = ['Stacking (Greedy)', 'Stacking (Pruned)', "Stacking (R²+Greedy)"]
first_trait = traits[0]
has_greedy = all([all(s in all_results[t] for s in existing_stacking) for t in traits])
if has_greedy:
    print("All stacking variants already exist! Exiting.")
    sys.exit(0)

missing_traits = [t for t in traits if 'Stacking (Greedy)' not in all_results[t]]
print(f"Traits needing stacking: {len(missing_traits)}/{len(traits)}: {missing_traits}")

# Load data
print("\nLoading rice data...")
trait_data_raw = ge.load_rice_data()

N_FOLDS = ge.N_FOLDS
GWAS_TOP_K = ge.GWAS_TOP_K
total_t0 = time.time()

for trait in traits:
    print(f"\n{'='*60}\nTrait: {trait}\n{'='*60}")

    # Skip if already has stacking greedy
    if 'Stacking (Greedy)' in all_results[trait]:
        print("  Stacking variants already present, skipping.")
        continue

    X_all, y = trait_data_raw[trait]  # rice: 2-tuple, no vt_all
    y = y.astype(np.float32)
    n_snps = min(GWAS_TOP_K, max(50, X_all.shape[1] - 50))
    print(f"  {len(y)} samples, {X_all.shape[1]} markers -> {n_snps} GWAS-selected")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=ge.RANDOM_SEED)
    oof_trad = {m: np.zeros(len(y)) for m in ge.TRAD_NAMES}
    oof_dl = {m: np.zeros(len(y)) for m in ge.DL_BASE_NAMES}

    for fold_i, (tr, te) in enumerate(kf.split(X_all)):
        print(f"  --- Fold {fold_i+1}/{N_FOLDS} ---")
        Xtr_raw, Xte_raw = X_all[tr], X_all[te]
        ytr, yte = y[tr], y[te]

        maf_idx = ge.maf_filter(Xtr_raw, ge.MAF_THRESHOLD)
        if len(maf_idx) >= n_snps:
            Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]

        # Traditional models: ALWAYS use GWAS markers
        gidx_gwas = ge.gwas_select(Xtr_raw, ytr, n_snps)
        Xtr_trad = Xtr_raw[:, gidx_gwas]; Xte_trad = Xte_raw[:, gidx_gwas]
        sc_trad = StandardScaler()
        Xtr_trad_s = sc_trad.fit_transform(Xtr_trad).astype(np.float32)
        Xte_trad_s = sc_trad.transform(Xte_trad).astype(np.float32)
        G_fold_train = Xtr_trad_s @ Xtr_trad_s.T / n_snps
        G_fold_te_tr = Xte_trad_s @ Xtr_trad_s.T / n_snps

        trad_configs = ge._make_trad_configs(G_fold_train, G_fold_te_tr, n_snps, len(tr))
        for tname, build_fn, fit_fn, pred_fn, param_count in trad_configs:
            t0 = time.time()
            tmodel = build_fn()
            fit_fn(tmodel, Xtr_trad_s, ytr)
            preds = pred_fn(tmodel, Xte_trad_s)
            oof_trad[tname][te] = preds
            print(f"    {tname:<16s} R2={r2_score(yte, preds):+.4f}  ({time.time()-t0:.1f}s)")

        # DL models (rice: no vt_maf)
        gidx_dl, _, Xtr_dl_s, Xte_dl_s = ge._select_dl_markers(
            Xtr_raw, Xte_raw, ytr, gidx_gwas, None, n_snps)

        for mi, mname in enumerate(ge.DL_NAMES):
            model = ge.create_model(mname, n_snps)
            t0 = time.time()
            bs = ge._get_batch_size(mname)
            wd = 5e-3 if mname == 'AdditiveGenomicNet' else 1e-3
            lr = 1e-3 if mname == 'FusionNet' else 2e-3
            model = ge.train_torch_model(model, Xtr_dl_s, ytr, epochs=300,
                                         batch_size=bs, lr=lr, weight_decay=wd, patience=30)
            preds = ge.predict_torch_model(model, Xte_dl_s)
            elapsed = time.time() - t0
            print(f"    {mname:<22s} R2={r2_score(yte, preds):+.4f}  ({elapsed:.1f}s)")
            if mname in ge.DL_BASE_NAMES:
                oof_dl[mname][te] = preds
        torch.cuda.empty_cache()

    # Compute stacking variants and add to results
    print(f"\n  Computing stacking variants...")
    ge._add_stacking_to_results(oof_dl, oof_trad, y, all_results[trait], N_FOLDS)

    # Print new stacking results
    for sname in existing_stacking:
        if sname in all_results[trait]:
            s = all_results[trait][sname]
            print(f"    {sname:<24s} R2={s['R2']:+.4f}  Corr={s['Correlation']:+.4f}")

    # Save intermediate
    with open(INTER_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved.")

elapsed = (time.time() - total_t0) / 60
print(f"\n{'='*60}")
print(f"Rice stacking supplement complete! ({elapsed:.1f} min)")
print(f"Results: {INTER_PATH}")

# Quick summary of new stacking vs baselines
print(f"\n{'Trait':<28s} {'XGBoost':>8s} {'Stk(Greedy)':>12s} {'Gain':>8s}")
print('-'*60)
for t in traits:
    xgb = all_results[t]['XGBoost']['R2']
    sg = all_results[t].get('Stacking (Greedy)', {}).get('R2', float('nan'))
    print(f'{t:<28s} {xgb:+8.4f} {sg:+12.4f} {sg-xgb:+8.4f}')
