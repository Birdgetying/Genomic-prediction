#!/usr/bin/env python
"""Generate wheat scatter plots only: XGBoost + Stacking (Greedy) predicted vs true.
Run on HPC GPU node via submit_wheat_figures.jsub."""
import json, time, sys, random
from collections import Counter
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNetCV

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

import genomic_ensemble as ge

DEVICE = ge.DEVICE
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
def get_single_model_oof(X_all_t, y, n_snps, model_name, vt_all=None):
    """5-fold OOF for a single model."""
    y = y.astype(np.float32)
    oof = np.zeros(len(y))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fi, (tr, te) in enumerate(kf.split(X_all_t)):
        Xtr_raw, Xte_raw = X_all_t[tr], X_all_t[te]
        ytr, yte = y[tr], y[te]
        vt_cur = vt_all
        maf_idx = ge.maf_filter(Xtr_raw, ge.MAF_THRESHOLD)
        if len(maf_idx) >= n_snps:
            Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]
            if vt_cur is not None: vt_cur = vt_cur[maf_idx]

        gidx_gwas = ge.gwas_select(Xtr_raw, ytr, n_snps)
        Xtr = Xtr_raw[:, gidx_gwas]; Xte = Xte_raw[:, gidx_gwas]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr).astype(np.float32); Xte_s = sc.transform(Xte).astype(np.float32)

        if model_name in ge.TRAD_NAMES:
            G_train = Xtr_s @ Xtr_s.T / n_snps
            G_te = Xte_s @ Xtr_s.T / n_snps
            for tname, bf, fi_fn, pf, _ in ge._make_trad_configs(G_train, G_te, n_snps, len(tr)):
                if tname == model_name:
                    tm = bf(); fi_fn(tm, Xtr_s, ytr); oof[te] = pf(tm, Xte_s)
                    break
        else:
            gidx_dl, _, Xtr_dl, Xte_dl = ge._select_dl_markers(Xtr_raw, Xte_raw, ytr, gidx_gwas, vt_cur, n_snps)
            model = ge.create_model(model_name, n_snps)
            bs = 32 if model_name.startswith('FGN') or model_name == 'GenomicFM' else 64
            wd = 5e-3 if model_name == 'AdditiveGenomicNet' else 1e-3
            model = ge.train_torch_model(model, Xtr_dl, ytr, epochs=300, batch_size=bs, lr=2e-3, weight_decay=wd, patience=30)
            oof[te] = ge.predict_torch_model(model, Xte_dl)
        torch.cuda.empty_cache()
    return oof


def stacking_greedy_oof(X_all, y, n_snps, vt_all=None):
    """Nested 5-fold CV: outer eval, inner base OOF + greedy select + meta train."""
    y = y.astype(np.float32)
    oof = np.zeros(len(y))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_selected = []

    for fold_i, (tr, te) in enumerate(kf.split(X_all)):
        t0 = time.time()
        Xtr_raw, Xte_raw = X_all[tr], X_all[te]
        ytr, yte = y[tr], y[te]

        maf_idx = ge.maf_filter(Xtr_raw, ge.MAF_THRESHOLD)
        if vt_all is not None:
            Xtr_f = Xtr_raw[:, maf_idx] if len(maf_idx) >= n_snps else Xtr_raw
            Xte_f = Xte_raw[:, maf_idx] if len(maf_idx) >= n_snps else Xte_raw
            vt_maf = vt_all[maf_idx] if len(maf_idx) >= n_snps else vt_all
        else:
            Xtr_f = Xtr_raw[:, maf_idx] if len(maf_idx) >= n_snps else Xtr_raw
            Xte_f = Xte_raw[:, maf_idx] if len(maf_idx) >= n_snps else Xte_raw
            vt_maf = None

        gidx_gwas = ge.gwas_select(Xtr_f, ytr, n_snps)

        # Base model predictions on test fold
        base_test = {}
        Xtr_trad = Xtr_f[:, gidx_gwas]; Xte_trad = Xte_f[:, gidx_gwas]
        sc_trad = StandardScaler()
        Xtr_trad_s = sc_trad.fit_transform(Xtr_trad).astype(np.float32)
        Xte_trad_s = sc_trad.transform(Xte_trad).astype(np.float32)
        G_tr = Xtr_trad_s @ Xtr_trad_s.T / n_snps; G_te = Xte_trad_s @ Xtr_trad_s.T / n_snps
        for tname, bf, fi_fn, pf, _ in ge._make_trad_configs(G_tr, G_te, n_snps, len(tr)):
            tm = bf(); fi_fn(tm, Xtr_trad_s, ytr); base_test[tname] = pf(tm, Xte_trad_s)

        gidx_dl, _, Xtr_dl, Xte_dl = ge._select_dl_markers(Xtr_f, Xte_f, ytr, gidx_gwas, vt_maf, n_snps)
        for mname in ge.DL_NAMES:
            m = ge.create_model(mname, n_snps)
            bs = 32 if mname.startswith('FGN') or mname == 'GenomicFM' else 64
            wd = 5e-3 if mname == 'AdditiveGenomicNet' else 1e-3
            m = ge.train_torch_model(m, Xtr_dl, ytr, epochs=300, batch_size=bs, lr=2e-3, weight_decay=wd, patience=30)
            base_test[mname] = ge.predict_torch_model(m, Xte_dl)

        # Base model OOF on train via inner 3-fold
        base_train = {m: np.zeros(len(tr)) for m in base_test}
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for itr_idx, ite_idx in inner_kf.split(Xtr_f):
            Xitr = Xtr_f[itr_idx]; Xite = Xtr_f[ite_idx]; yitr = ytr[itr_idx]
            sub_gidx = ge.gwas_select(Xitr, yitr, n_snps)
            Xitr_t = Xitr[:, sub_gidx]; Xite_t = Xite[:, sub_gidx]
            sc_i = StandardScaler()
            Xitr_ts = sc_i.fit_transform(Xitr_t).astype(np.float32)
            Xite_ts = sc_i.transform(Xite_t).astype(np.float32)
            Gi_tr = Xitr_ts @ Xitr_ts.T / n_snps; Gi_te = Xite_ts @ Xitr_ts.T / n_snps
            for tname, bf, fi_fn, pf, _ in ge._make_trad_configs(Gi_tr, Gi_te, n_snps, len(itr_idx)):
                tm = bf(); fi_fn(tm, Xitr_ts, yitr); base_train[tname][ite_idx] = pf(tm, Xite_ts)

            sub_gidx_dl, _, Xitr_dl, Xite_dl = ge._select_dl_markers(
                Xitr, Xite, yitr, sub_gidx, vt_maf[sub_gidx] if vt_maf is not None else None, n_snps)
            for mname in ge.DL_NAMES:
                m2 = ge.create_model(mname, n_snps)
                bs2 = 32 if mname.startswith('FGN') or mname == 'GenomicFM' else 64
                wd2 = 5e-3 if mname == 'AdditiveGenomicNet' else 1e-3
                m2 = ge.train_torch_model(m2, Xitr_dl, yitr, epochs=300, batch_size=bs2, lr=2e-3, weight_decay=wd2, patience=30)
                base_train[mname][ite_idx] = ge.predict_torch_model(m2, Xite_dl)

        torch.cuda.empty_cache()

        selected = ge._greedy_forward_select(base_train, ytr, meta_type='ElasticNet')
        all_selected.append(selected)

        X_meta_train = np.column_stack([base_train[m] for m in selected])
        X_meta_test = np.column_stack([base_test[m] for m in selected])
        meta = ElasticNetCV(l1_ratio=[.1,.5,.7,.9,.95,1], alphas=ge.ENET_ALPHAS, cv=3, max_iter=10000, random_state=42)
        meta.fit(X_meta_train, ytr)
        oof[te] = meta.predict(X_meta_test)
        print(f"    Fold {fold_i+1} R²={r2_score(yte, oof[te]):+.4f}  [{','.join(selected)}]  ({time.time()-t0:.0f}s)")

    return oof, all_selected


# ============================================================================
print("\nLoading wheat data...")
wheat_trait_data = ge.load_wheat_data()
W_TRAITS = list(wheat_trait_data.keys())
print(f"Traits: {W_TRAITS}")

total_t0 = time.time()

for w_trait in W_TRAITS:
    X_w, y_w, vt_w = wheat_trait_data[w_trait]
    y_w = y_w.astype(np.float32)
    n_snps_w = min(ge.GWAS_TOP_K, max(50, X_w.shape[1] - 50))
    print(f"\n{w_trait}: n={len(y_w)}, markers={X_w.shape[1]} -> {n_snps_w} GWAS")

    print("  XGBoost OOF ...")
    w_xgb_oof = get_single_model_oof(X_w, y_w, n_snps_w, 'XGBoost', vt_all=vt_w)

    print("  Stacking (Greedy) OOF (nested CV) ...")
    w_stk_oof, w_stk_sel = stacking_greedy_oof(X_w, y_w, n_snps_w, vt_all=vt_w)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Wheat {w_trait}: Predicted vs True (5-fold OOF)', fontsize=15, fontweight='bold')

    r2_xgb = r2_score(y_w, w_xgb_oof)
    ax = axes[0]
    ax.scatter(y_w, w_xgb_oof, alpha=0.5, s=25, c='#1565C0', edgecolors='none', zorder=3)
    mn = min(y_w.min(), w_xgb_oof.min()); mx = max(y_w.max(), w_xgb_oof.max())
    pad = (mx - mn) * 0.08
    ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], '--', color='#E53935', alpha=0.5, lw=1.2)
    ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    ax.set_title(f'XGBoost\nR²={r2_xgb:.4f}', fontsize=12, fontweight='bold'); ax.grid(alpha=0.2)

    r2_stk = r2_score(y_w, w_stk_oof)
    ax = axes[1]
    ax.scatter(y_w, w_stk_oof, alpha=0.5, s=25, c='#1B5E20', edgecolors='none', zorder=3)
    mn = min(y_w.min(), w_stk_oof.min()); mx = max(y_w.max(), w_stk_oof.max())
    pad = (mx - mn) * 0.08
    ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], '--', color='#E53935', alpha=0.5, lw=1.2)
    ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    top = Counter([m for fs in w_stk_sel for m in fs]).most_common(3)
    ax.set_title(f'Stacking (Greedy)\nR²={r2_stk:.4f} | top: {",".join(t[0] for t in top)}',
                 fontsize=11, fontweight='bold'); ax.grid(alpha=0.2)

    fig.tight_layout()
    out = FIG_DIR / f'10_wheat_{w_trait}_scatter.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")

elapsed = (time.time() - total_t0) / 60
print(f"\nDone! Total time: {elapsed:.1f} min")
