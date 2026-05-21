#!/usr/bin/env python3
"""Quick ablation: test mixup on/off, SWA on/off for FGN v4/v7 on Plant_height."""
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


def run_ablation():
    print("=" * 70)
    print("  Training Ablation: Mixup & SWA on Plant_height")
    print("=" * 70)

    X_all, y_all = load_rice_trait("Plant_height")
    n_snps = min(ge.GWAS_TOP_K, max(50, X_all.shape[1] - 50))
    print(f"\nSamples: {len(y_all)}, Markers: {X_all.shape[1]} -> {n_snps}")

    models_to_test = ['FGN v4', 'FGN v7']
    configs = [
        ('default (mixup, no SWA)',  True,  False),
        ('no mixup',                 False, False),
        ('SWA',                      True,  True),
    ]

    all_results = {}
    kf = KFold(n_splits=3, shuffle=True, random_state=ge.RANDOM_SEED)

    for mname in models_to_test:
        for cfg_name, use_mixup, use_swa in configs:
            key = f"{mname} | {cfg_name}"
            print(f"\n  >>> {key}")
            preds_all, targets_all = [], []

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

                model = ge.create_model(mname, n_snps)
                t0 = time.time()
                model = ge.train_torch_model(model, Xtr_s, ytr, epochs=300,
                                             batch_size=32, lr=2e-3, weight_decay=1e-3,
                                             patience=35, use_mixup=use_mixup, use_swa=use_swa)
                preds = ge.predict_torch_model(model, Xte_s)
                elapsed = time.time() - t0
                preds_all.extend(preds.tolist())
                targets_all.extend(yte.tolist())
                r2_fold = r2_score(yte, preds)
                print(f"    Fold {fi+1}: R2={r2_fold:+.4f} ({elapsed:.1f}s)")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            r2_v = float(r2_score(targets_all, preds_all))
            corr_v = float(pearsonr(targets_all, preds_all)[0])
            all_results[key] = {'R2': r2_v, 'Correlation': corr_v}
            print(f"    => R2={r2_v:+.4f}, r={corr_v:+.4f}")

    print(f"\n{'='*70}")
    print("  Summary:")
    for k, v in sorted(all_results.items(), key=lambda x: -x[1]['R2']):
        print(f"  {k:<40s} R2={v['R2']:+.4f}  r={v['Correlation']:+.4f}")

    with open("results/ablation_training.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/ablation_training.json")


if __name__ == '__main__':
    run_ablation()
