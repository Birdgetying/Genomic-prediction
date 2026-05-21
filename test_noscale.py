#!/usr/bin/env python3
"""Quick test: FGN v4 with vs without StandardScaler on Plant_height."""
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


def run_test():
    X_all, y_all = load_rice_trait("Plant_height")
    n_snps = min(ge.GWAS_TOP_K, max(50, X_all.shape[1] - 50))
    kf = KFold(n_splits=3, shuffle=True, random_state=ge.RANDOM_SEED)
    configs = [('with scaler', True), ('no scaler', False)]

    for cfg_name, use_scaler in configs:
        preds_all, targets_all = [], []
        print(f"\n  >>> FGN v4 {cfg_name}")
        for fi, (tr, te) in enumerate(kf.split(X_all)):
            Xtr_raw, Xte_raw = X_all[tr], X_all[te]
            ytr, yte = y_all[tr], y_all[te]
            maf_idx = ge.maf_filter(Xtr_raw)
            if len(maf_idx) >= n_snps:
                Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]
            gidx = ge.gwas_select(Xtr_raw, ytr, n_snps)
            Xtr = Xtr_raw[:, gidx]
            Xte = Xte_raw[:, gidx]
            if use_scaler:
                sc = StandardScaler()
                Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
                Xte_s = sc.transform(Xte).astype(np.float32)
            else:
                Xtr_s = Xtr.astype(np.float32)
                Xte_s = Xte.astype(np.float32)

            model = ge.create_model('FGN v4', n_snps, overrides={'hidden': 96})
            t0 = time.time()
            model = ge.train_torch_model(model, Xtr_s, ytr, epochs=300, batch_size=32,
                                         lr=2e-3, weight_decay=1e-3, patience=35, use_swa=True)
            preds = ge.predict_torch_model(model, Xte_s)
            preds_all.extend(preds.tolist())
            targets_all.extend(yte.tolist())
            r2 = r2_score(yte, preds)
            print(f"    Fold {fi+1}: R2={r2:+.4f} ({time.time()-t0:.1f}s)")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        final_r2 = r2_score(targets_all, preds_all)
        print(f"    => {cfg_name}: R2={final_r2:+.4f}")


if __name__ == '__main__':
    run_test()
