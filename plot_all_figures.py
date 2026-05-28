#!/usr/bin/env python
"""Generate ALL final figures for three-dataset genomic prediction ensemble.
To be run on HPC GPU node via submit_plot_figures.jsub.

Produces 11 figures:
  01-04: Bar charts (from existing JSON, no GPU needed)
  05-06: XGBoost/GBLUP scatter (light CV)
  07: Combined ranking (from JSON)
  08-09: Stacking Greedy scatter (heavy: nested CV for all base models)
  10: Wheat XGBoost scatter
  11: Wheat Stacking Greedy scatter
"""
import json, time, sys, random, os
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
OOF_DIR = SCRIPT_DIR / "results" / "oof_cache"
OOF_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# Color scheme
# ============================================================================
TRAD_COLORS = {'RRBLUP': '#90CAF9', 'GBLUP': '#64B5F6', 'XGBoost': '#1565C0',
               'ElasticNet': '#42A5F5', 'GWAS_RRBLUP': '#1E88E5'}
DL_COLORS = {
    'FGN': '#FFB74D', 'FGN v2': '#FF9800', 'FGN v4': '#F57C00',
    'FGN v5': '#E65100', 'FGN v6': '#BF360C', 'FGN v7': '#FFD54F',
    'FGN v9': '#FFCC80', 'FGN v10': '#FFE082', 'FGN v11': '#FFECB3',
    'FGNplus': '#A1887F', 'FGN PCA': '#BCAAA4', 'GenomicFM': '#D7CCC8',
    'FusionNet': '#E91E63', 'AdditiveGenomicNet': '#F48FB1',
    'DeepKernelGP': '#CE93D8', 'EFM v3': '#BA68C8', 'FGN v3': '#AB47BC',
    'MICNN': '#9C27B0', 'MICNN v2': '#6A1B9A', 'PreFGN': '#8E24AA',
    'ResFGN': '#4A148C',
}
ENS_COLORS = {
    'Stacking (DL)': '#66BB6A', 'Stacking (All)': '#2E7D32',
    'Trad Ensemble': '#0D47A1', 'Stacking (Pruned)': '#43A047',
    'Stacking (Greedy)': '#1B5E20', 'Stacking (R²+Greedy)': '#388E3C',
}

def model_color(name):
    if name in TRAD_COLORS: return TRAD_COLORS[name]
    if name in DL_COLORS: return DL_COLORS[name]
    if name in ENS_COLORS: return ENS_COLORS[name]
    return '#BDBDBD'

def avg_r2(data, m):
    r2s = [data[t][m]['R2'] for t in data if m in data[t]]
    return np.mean(r2s) if r2s else float('nan')

# ============================================================================
# Load all results
# ============================================================================
print("\nLoading results...")

def _load_json(path):
    if path.exists():
        return json.load(open(path, 'r', encoding='utf-8'))
    print(f"  WARNING: {path} not found — skipping this dataset")
    return None

wheat = _load_json(SCRIPT_DIR/"results/wheat_ensemble/ensemble_intermediate.json")
rice = _load_json(SCRIPT_DIR/"results/rice_ensemble/ensemble_intermediate.json")
maize = _load_json(SCRIPT_DIR/"results/maize_ensemble/ensemble_intermediate.json")

if wheat is None and rice is None and maize is None:
    print("ERROR: No result files found. Upload results/*/ensemble_intermediate.json first.")
    sys.exit(1)

# Build DATASETS list, skipping missing
DATASETS = []
if wheat is not None:
    W_TRAITS = list(wheat.keys())
    DATASETS.append(('Wheat', wheat, W_TRAITS))
else: W_TRAITS = []
if rice is not None:
    R_TRAITS = sorted(rice.keys())
    DATASETS.append(('Rice', rice, R_TRAITS))
else: R_TRAITS = []
if maize is not None:
    M_TRAITS = sorted(maize.keys())
    DATASETS.append(('Maize', maize, M_TRAITS))
else: M_TRAITS = []

if not DATASETS:
    print("ERROR: No datasets loaded. Check intermediate JSON files.")
    sys.exit(1)

print(f"Loaded {len(DATASETS)} dataset(s): {[d[0] for d in DATASETS]}")

# ============================================================================
# Figure 01: Per-dataset bar charts
# ============================================================================
print("\n[01/11] Per-dataset bar charts...")
fig, axes = plt.subplots(1, 3, figsize=(36, 14))
fig.suptitle('Genomic Prediction Ensemble — Per-Dataset Model Comparison (5-fold CV R²)',
             fontsize=22, fontweight='bold', y=1.01)

for ax_idx, (dname, data, traits) in enumerate(DATASETS):
    ax = axes[ax_idx]
    all_models = list(data[traits[0]].keys())
    means = {m: np.mean([data[t][m]['R2'] for t in traits if m in data[t]]) for m in all_models}
    sorted_m = sorted([m for m in all_models if means[m] > -5], key=lambda m: means[m], reverse=True)[:20]
    vals = [means[m] for m in sorted_m]
    colors = [model_color(m) for m in sorted_m]

    x = np.arange(len(sorted_m))
    bars = ax.bar(x, vals, 0.7, color=colors, edgecolor='white', linewidth=0.8, zorder=3)
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor('#C62828'); bars[best_idx].set_linewidth(3.0)
    for i, (m, v) in enumerate(zip(sorted_m, vals)):
        if i < 8 or v == max(vals):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom',
                    fontsize=8 if v != max(vals) else 10,
                    fontweight='bold' if v == max(vals) else 'normal',
                    color='#C62828' if v == max(vals) else '#555')
    ax.axhline(y=0, color='#666', linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(sorted_m, rotation=55, ha='right', fontsize=7.5)
    ax.set_ylabel('R²', fontsize=13)
    ax.set_title(f'{dname} ({len(traits)} trait{"s" if len(traits)>1 else ""})  Best: {sorted_m[0]} ({vals[best_idx]:.3f})',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(min(-0.5, min(vals)-0.15), max(vals)+0.15)

fig.tight_layout()
fig.savefig(FIG_DIR/'01_per_dataset_bar_charts.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# Figure 02: Cross-dataset comparison (common models)
# ============================================================================
print("[02/11] Cross-dataset comparison...")
w_models = set(wheat[W_TRAITS[0]].keys())
r_models = set(rice[R_TRAITS[0]].keys())
m_models = set(maize[M_TRAITS[0]].keys())
common = w_models & r_models & m_models
common_sorted = sorted(common, key=lambda m: (avg_r2(wheat,m)+avg_r2(rice,m)+avg_r2(maize,m))/3, reverse=True)

fig, ax = plt.subplots(figsize=(16, 10))
x = np.arange(len(common_sorted)); bar_w = 0.25
w_vals = [avg_r2(wheat, m) for m in common_sorted]
r_vals = [avg_r2(rice, m) for m in common_sorted]
m_vals = [avg_r2(maize, m) for m in common_sorted]
ax.bar(x - bar_w, w_vals, bar_w, color='#2196F3', edgecolor='white', label='Wheat (1 trait)', zorder=3)
ax.bar(x, r_vals, bar_w, color='#FF9800', edgecolor='white', label='Rice (10 traits)', zorder=3)
ax.bar(x + bar_w, m_vals, bar_w, color='#4CAF50', edgecolor='white', label='Maize (4 traits)', zorder=3)
ax.axhline(y=0, color='#666', linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(common_sorted, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('R²', fontsize=13)
ax.set_title('Cross-Dataset Comparison — Models Common to All Three Datasets', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper right'); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR/'02_cross_dataset_comparison.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# Figure 03: Stacking gain scatter
# ============================================================================
print("[03/11] Stacking gain scatter...")
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Stacking (Greedy) vs Best Single Model — Per Trait', fontsize=16, fontweight='bold')

for ax_idx, (dname, data, traits) in enumerate(DATASETS):
    ax = axes[ax_idx]
    singles = [m for m in data[traits[0]].keys() if 'Stacking' not in m and 'Ensemble' not in m]
    stk_key = 'Stacking (Greedy)' if 'Stacking (Greedy)' in data[traits[0]] else 'Stacking (All)'
    xs, ys = [], []
    for t in traits:
        bx = max(data[t][m]['R2'] for m in singles if m in data[t])
        by = data[t].get(stk_key, {}).get('R2', bx)
        xs.append(bx); ys.append(by)

    mn = min(min(xs), min(ys)) - 0.03; mx = max(max(xs), max(ys)) + 0.05
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.3, lw=1.5, label='y=x')
    for i, t in enumerate(traits):
        ax.scatter(xs[i], ys[i], s=180, edgecolors='#333', linewidth=1.2, zorder=4)
        ax.annotate(t.replace('_','\n')[:20], (xs[i], ys[i]), textcoords="offset points", xytext=(8, 10), fontsize=7)
    wins = sum(1 for yv, xv in zip(ys, xs) if yv > xv)
    ax.set_xlabel('Best Single Model R²'); ax.set_ylabel(f'{stk_key} R²')
    ax.set_title(f'{dname}: Stacking wins {wins}/{len(traits)}', fontsize=12, fontweight='bold')
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx); ax.set_aspect('equal'); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR/'03_stacking_gain_scatter.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# Figure 04: Rice per-trait detail
# ============================================================================
if rice is None:
    print("[04/11] Rice data not available — skipping")
else:
 print("[04/11] Rice per-trait detail...")
 all_rice_models = list(rice[R_TRAITS[0]].keys())
rice_means = {m: np.mean([rice[t][m]['R2'] for t in R_TRAITS if m in rice[t]]) for m in all_rice_models}
top_rice = sorted([m for m in all_rice_models if rice_means[m] > -1], key=lambda m: rice_means[m], reverse=True)[:12]

fig, axes = plt.subplots(5, 2, figsize=(28, 32))
fig.suptitle('Rice: Per-Trait Model Comparison (5-fold CV R²)', fontsize=18, fontweight='bold')
for idx, trait in enumerate(R_TRAITS):
    ax = axes[idx//2][idx%2]
    vals = [rice[trait][m]['R2'] if m in rice[trait] else 0 for m in top_rice]
    colors = [model_color(m) for m in top_rice]
    x = np.arange(len(top_rice))
    bars = ax.bar(x, vals, 0.65, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor('#C62828'); bars[best_idx].set_linewidth(2.5)
    ax.text(best_idx, vals[best_idx]+0.03, f'{vals[best_idx]:.3f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#C62828')
    ax.axhline(y=0, color='#666', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(top_rice, rotation=60, ha='right', fontsize=6)
    ax.set_ylabel('R²'); ax.set_title(f'{trait}  (best: {top_rice[best_idx]} {vals[best_idx]:.3f})', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.2); ax.set_ylim(min(-0.5, min(vals)-0.1), max(vals)+0.12)
fig.tight_layout()
fig.savefig(FIG_DIR/'04_rice_per_trait_detail.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# Figure 07: Combined ranking
# ============================================================================
print("[07/11] Combined ranking...")
combined = {}
for m in common:
    w = avg_r2(wheat, m); r = avg_r2(rice, m); mz = avg_r2(maize, m)
    combined[m] = (w + r + mz) / 3
sorted_all = sorted(combined.items(), key=lambda x: x[1], reverse=True)

fig, ax = plt.subplots(figsize=(14, 10))
y_pos = range(len(sorted_all))
models_r = [s[0] for s in sorted_all]; vals_r = [s[1] for s in sorted_all]
bars = ax.barh(y_pos, vals_r, 0.7, color=[model_color(m) for m in models_r], edgecolor='white', linewidth=1, zorder=3)
for i in range(min(3, len(sorted_all))): bars[i].set_edgecolor('#C62828'); bars[i].set_linewidth(2.5)
for i, (m, v) in enumerate(zip(models_r, vals_r)):
    ax.text(v+0.005, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
    tag = 'Traditional' if m in TRAD_COLORS else ('Ensemble' if m in ENS_COLORS else 'DL')
    ax.text(-0.35, i, f'{m} [{tag}]', va='center', ha='right', fontsize=9,
            fontweight='bold' if 'Stacking' in m or 'Ensemble' in m else 'normal')
ax.set_yticks([]); ax.set_xlabel('Mean R² (Wheat + Rice + Maize avg)', fontsize=12)
ax.set_title('Three-Dataset Combined Ranking', fontsize=15, fontweight='bold')
ax.grid(axis='x', alpha=0.25); ax.set_xlim(-0.75, max(vals_r)+0.08); ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIG_DIR/'07_combined_ranking.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# Helper: Get base model OOF (single model, light CV)
# ============================================================================
def get_single_model_oof(X_all_t, y, n_snps, model_name, vt_all=None):
    """5-fold OOF for a single model. Fast path for traditional models."""
    y = y.astype(np.float32)
    oof = np.zeros(len(y))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fi, (tr, te) in enumerate(kf.split(X_all_t)):
        Xtr_raw, Xte_raw = X_all_t[tr], X_all_t[te]
        ytr, yte = y[tr], y[te]
        vt_current = vt_all  # don't mutate input
        maf_idx = ge.maf_filter(Xtr_raw, ge.MAF_THRESHOLD)
        if len(maf_idx) >= n_snps:
            Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]
            if vt_current is not None: vt_current = vt_current[maf_idx]

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
            gidx_dl, _, Xtr_dl, Xte_dl = ge._select_dl_markers(Xtr_raw, Xte_raw, ytr, gidx_gwas, vt_current, n_snps)
            model = ge.create_model(model_name, n_snps)
            bs = 32 if model_name.startswith('FGN') or model_name == 'GenomicFM' else 64
            wd = 5e-3 if model_name == 'AdditiveGenomicNet' else 1e-3
            model = ge.train_torch_model(model, Xtr_dl, ytr, epochs=300, batch_size=bs, lr=2e-3, weight_decay=wd, patience=30)
            oof[te] = ge.predict_torch_model(model, Xte_dl)
        torch.cuda.empty_cache()
    return oof

# ============================================================================
# Helper: Stacking Greedy OOF (nested CV, heavy)
# ============================================================================
def stacking_greedy_oof(X_all, y, n_snps, vt_all=None):
    """Nested 5-fold CV: outer=eval, inner=base OOF + greedy select + meta train."""
    y = y.astype(np.float32)
    oof = np.zeros(len(y))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_selected = []
    fold_r2s = []

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

        # Base model test predictions
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

        # Base model train OOF via inner 3-fold
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
                Xitr, Xite, yitr, sub_gidx, vt_maf, n_snps)
            for mname in ge.DL_NAMES:
                m2 = ge.create_model(mname, n_snps)
                bs2 = 32 if mname.startswith('FGN') or mname == 'GenomicFM' else 64
                wd2 = 5e-3 if mname == 'AdditiveGenomicNet' else 1e-3
                m2 = ge.train_torch_model(m2, Xitr_dl, yitr, epochs=300, batch_size=bs2, lr=2e-3, weight_decay=wd2, patience=30)
                base_train[mname][ite_idx] = ge.predict_torch_model(m2, Xite_dl)

        torch.cuda.empty_cache()

        # Greedy forward select on train OOF
        selected = ge._greedy_forward_select(base_train, ytr, meta_type='ElasticNet')
        all_selected.append(selected)

        # Meta-learner: train on train-OOF, predict on test-OOF
        X_meta_train = np.column_stack([base_train[m] for m in selected])
        X_meta_test = np.column_stack([base_test[m] for m in selected])
        meta = ElasticNetCV(l1_ratio=[.1,.5,.7,.9,.95,1], alphas=ge.ENET_ALPHAS, cv=3, max_iter=10000, random_state=42)
        meta.fit(X_meta_train, ytr)
        oof[te] = meta.predict(X_meta_test)
        r2_f = r2_score(yte, oof[te])
        fold_r2s.append(r2_f)
        print(f"    Fold {fold_i+1} R²={r2_f:+.4f}  [{','.join(selected)}]  ({time.time()-t0:.0f}s)")

    return oof, all_selected, fold_r2s

# ============================================================================
# Rice: XGBoost scatter (Fig 05) + Stacking Greedy scatter (Fig 08)
# ============================================================================
print("\n[05+08] Rice scatter plots...")
rice_data = ge.load_rice_data()
rice_traits_alpha = sorted(rice_data.keys())

# XGBoost OOF
print("  XGBoost OOF...")
rice_xgb_oof = {}
for trait in rice_traits_alpha:
    X_t, y = rice_data[trait]
    n_snps = min(ge.GWAS_TOP_K, max(50, X_t.shape[1] - 50))
    print(f"    {trait}...")
    rice_xgb_oof[trait] = get_single_model_oof(X_t, y, n_snps, 'XGBoost')

# Stacking Greedy OOF
print("  Stacking (Greedy) OOF (nested CV)...")
rice_stk_oof = {}
rice_stk_selected = {}
rice_stk_r2 = {}
for trait in rice_traits_alpha:
    X_t, y = rice_data[trait]
    n_snps = min(ge.GWAS_TOP_K, max(50, X_t.shape[1] - 50))
    print(f"  {trait}:")
    oof, sel, r2s = stacking_greedy_oof(X_t, y, n_snps)
    rice_stk_oof[trait] = oof; rice_stk_selected[trait] = sel; rice_stk_r2[trait] = r2s
    np.savez(OOF_DIR/f'rice_stacking_{trait}.npz', oof=oof, y=y, r2_folds=r2s,
             selected_folds=str(sel))

# Rice XGBoost scatter (Fig 05)
n = len(rice_traits_alpha); cols = 4; rows = (n+cols-1)//cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4.5))
fig.suptitle('Rice: Predicted vs True — XGBoost (5-fold OOF)', fontsize=16, fontweight='bold')
for idx, trait in enumerate(rice_traits_alpha):
    ax = axes[idx//cols][idx%cols]
    oof = rice_xgb_oof[trait]; y = rice_data[trait][1].astype(np.float32)
    r2 = r2_score(y, oof)
    ax.scatter(y, oof, alpha=0.5, s=25, c='#1565C0', edgecolors='none', zorder=3)
    mn = min(y.min(), oof.min()); mx = max(y.max(), oof.max())
    pad = (mx-mn)*0.08
    ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], '--', color='#E53935', alpha=0.5, lw=1.2)
    ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    ax.set_title(f'{trait}\nR²={r2:.4f}', fontsize=10, fontweight='bold'); ax.grid(alpha=0.2)
for idx in range(n, rows*cols): axes[idx//cols][idx%cols].axis('off')
fig.tight_layout()
fig.savefig(FIG_DIR/'05_rice_xgboost_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  -> 05_rice_xgboost_scatter.png")

# Rice Stacking Greedy scatter (Fig 08)
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4.5))
fig.suptitle('Rice: Predicted vs True — Stacking (Greedy) [5-fold nested CV]', fontsize=16, fontweight='bold')
for idx, trait in enumerate(rice_traits_alpha):
    ax = axes[idx//cols][idx%cols]
    oof = rice_stk_oof[trait]; y = rice_data[trait][1].astype(np.float32)
    r2 = r2_score(y, oof)
    ax.scatter(y, oof, alpha=0.5, s=25, c='#1B5E20', edgecolors='none', zorder=3)
    mn = min(y.min(), oof.min()); mx = max(y.max(), oof.max())
    pad = (mx-mn)*0.08
    ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], '--', color='#E53935', alpha=0.5, lw=1.2)
    ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    ax.set_title(f'{trait}\nR²={r2:.4f}', fontsize=10, fontweight='bold'); ax.grid(alpha=0.2)
for idx in range(n, rows*cols): axes[idx//cols][idx%cols].axis('off')
fig.tight_layout()
fig.savefig(FIG_DIR/'08_rice_stacking_greedy_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  -> 08_rice_stacking_greedy_scatter.png")

# ============================================================================
# Maize: GBLUP scatter (Fig 06) + Stacking Greedy scatter (Fig 09)
# ============================================================================
print("\n[06+09] Maize scatter plots...")
X_m_all, y_m_dict = ge.load_iranian_data(max_markers=None)
var_thresh = 0.005; keep_idx = np.var(X_m_all, axis=0) >= var_thresh
if keep_idx.sum() < X_m_all.shape[1]: X_m_all = X_m_all[:, keep_idx]
maize_traits_m = sorted(y_m_dict.keys())

# GBLUP OOF
print("  GBLUP OOF...")
maize_gblup_oof = {}
for trait in maize_traits_m:
    n_snps = min(ge.GWAS_TOP_K, max(50, X_m_all.shape[1]-50))
    print(f"    {trait}...")
    maize_gblup_oof[trait] = get_single_model_oof(X_m_all, y_m_dict[trait], n_snps, 'GBLUP')

# Stacking Greedy OOF
print("  Stacking (Greedy) OOF (nested CV)...")
maize_stk_oof = {}
maize_stk_selected = {}
for trait in maize_traits_m:
    n_snps = min(ge.GWAS_TOP_K, max(50, X_m_all.shape[1]-50))
    print(f"  {trait}:")
    oof, sel, r2s = stacking_greedy_oof(X_m_all, y_m_dict[trait], n_snps)
    maize_stk_oof[trait] = oof; maize_stk_selected[trait] = sel
    np.savez(OOF_DIR/f'maize_stacking_{trait}.npz', oof=oof, y=y_m_dict[trait], r2_folds=r2s, selected_folds=str(sel))

# Maize GBLUP scatter (Fig 06)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Maize: Predicted vs True — GBLUP (5-fold OOF)', fontsize=15, fontweight='bold')
for idx, trait in enumerate(maize_traits_m):
    ax = axes[idx//2][idx%2]
    oof = maize_gblup_oof[trait]; y = y_m_dict[trait].astype(np.float32)
    r2 = r2_score(y, oof)
    ax.scatter(y, oof, alpha=0.5, s=20, c='#64B5F6', edgecolors='none', zorder=3)
    mn = min(y.min(), oof.min()); mx_ = max(y.max(), oof.max())
    pad = (mx_-mn)*0.08
    ax.plot([mn-pad, mx_+pad], [mn-pad, mx_+pad], '--', color='#E53935', alpha=0.5, lw=1.2)
    ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    ax.set_title(f'{trait}\nR²={r2:.4f}', fontsize=11, fontweight='bold'); ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FIG_DIR/'06_maize_gblup_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  -> 06_maize_gblup_scatter.png")

# Maize Stacking Greedy scatter (Fig 09)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Maize: Predicted vs True — Stacking (Greedy) [5-fold nested CV]', fontsize=15, fontweight='bold')
for idx, trait in enumerate(maize_traits_m):
    ax = axes[idx//2][idx%2]
    oof = maize_stk_oof[trait]; y = y_m_dict[trait].astype(np.float32)
    r2 = r2_score(y, oof)
    ax.scatter(y, oof, alpha=0.5, s=20, c='#1B5E20', edgecolors='none', zorder=3)
    mn = min(y.min(), oof.min()); mx_ = max(y.max(), oof.max())
    pad = (mx_-mn)*0.08
    ax.plot([mn-pad, mx_+pad], [mn-pad, mx_+pad], '--', color='#E53935', alpha=0.5, lw=1.2)
    ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    ax.set_title(f'{trait}\nR²={r2:.4f}', fontsize=11, fontweight='bold'); ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FIG_DIR/'09_maize_stacking_greedy_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  -> 09_maize_stacking_greedy_scatter.png")

# ============================================================================
# Wheat: XGBoost scatter (Fig 10) + Stacking Greedy scatter (Fig 11)
# ============================================================================
print("\n[10+11] Wheat scatter plots...")
print("  Loading wheat data...")
wheat_trait_data = ge.load_wheat_data()

for w_trait in W_TRAITS:
    X_w, y_w, vt_w = wheat_trait_data[w_trait]
    y_w = y_w.astype(np.float32)
    n_snps_w = min(ge.GWAS_TOP_K, max(50, X_w.shape[1]-50))
    print(f"  {w_trait}: n={len(y_w)}, markers={X_w.shape[1]} -> {n_snps_w} GWAS")

    # XGBoost OOF
    print(f"    XGBoost OOF...")
    w_xgb_oof = get_single_model_oof(X_w, y_w, n_snps_w, 'XGBoost', vt_all=vt_w)

    # Stacking Greedy OOF
    print(f"    Stacking Greedy OOF (nested CV)...")
    w_stk_oof, w_stk_sel, w_stk_r2 = stacking_greedy_oof(X_w, y_w, n_snps_w, vt_all=vt_w)

    # Wheat scatter: two panels side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Wheat {w_trait}: Predicted vs True (5-fold OOF)', fontsize=15, fontweight='bold')

    # XGBoost
    r2_xgb = r2_score(y_w, w_xgb_oof)
    ax = axes[0]
    ax.scatter(y_w, w_xgb_oof, alpha=0.5, s=25, c='#1565C0', edgecolors='none', zorder=3)
    mn = min(y_w.min(), w_xgb_oof.min()); mx = max(y_w.max(), w_xgb_oof.max())
    pad = (mx-mn)*0.08
    ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], '--', color='#E53935', alpha=0.5, lw=1.2)
    ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    ax.set_title(f'XGBoost\nR²={r2_xgb:.4f}', fontsize=12, fontweight='bold'); ax.grid(alpha=0.2)

    # Stacking Greedy
    r2_stk = r2_score(y_w, w_stk_oof)
    ax = axes[1]
    ax.scatter(y_w, w_stk_oof, alpha=0.5, s=25, c='#1B5E20', edgecolors='none', zorder=3)
    mn = min(y_w.min(), w_stk_oof.min()); mx = max(y_w.max(), w_stk_oof.max())
    pad = (mx-mn)*0.08
    ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], '--', color='#E53935', alpha=0.5, lw=1.2)
    ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    top_models = [m for fold_sel in w_stk_sel for m in fold_sel]
    top = Counter(top_models).most_common(3)
    ax.set_title(f'Stacking (Greedy)\nR²={r2_stk:.4f} | top: {",".join([t[0] for t in top])}',
                 fontsize=12, fontweight='bold'); ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR/f'10_wheat_{w_trait}_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> 10_wheat_{w_trait}_scatter.png")

    np.savez(OOF_DIR/f'wheat_{w_trait}_oof.npz', xgb_oof=w_xgb_oof, stk_oof=w_stk_oof, y=y_w,
             stk_r2_folds=w_stk_r2, stk_selected=str(w_stk_sel))

# ============================================================================
# Done!
# ============================================================================
print(f"\n{'='*60}")
print(f"ALL 11 FIGURES SAVED TO: {FIG_DIR}")
for f in sorted(FIG_DIR.glob('*.png')):
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
print(f"\nOOF cache: {OOF_DIR}")
print("Done!")
