#!/usr/bin/env python
"""Final visualization: bar charts + scatter plots for all three datasets."""
import json, time, sys, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

random.seed(42); np.random.seed(42)

import genomic_ensemble as ge
import torch

# Only import torch after seed
DEVICE = ge.DEVICE
SCRIPT_DIR = Path(__file__).resolve().parent
RES_DIR = SCRIPT_DIR / "results"
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# Load data
# ============================================================================
wheat = json.load(open(RES_DIR/"wheat_ensemble"/"ensemble_intermediate.json", 'r', encoding='utf-8'))
rice = json.load(open(RES_DIR/"rice_ensemble"/"ensemble_intermediate.json", 'r', encoding='utf-8'))
maize = json.load(open(RES_DIR/"maize_ensemble"/"ensemble_intermediate.json", 'r', encoding='utf-8'))

DATASETS = {'Wheat': (wheat, ['TFW_DSI']),
            'Rice': (rice, sorted(rice.keys())),
            'Maize': (maize, sorted(maize.keys()))}

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

def model_type(name):
    if name in TRAD_COLORS or name in ['GWAS_RRBLUP','ElasticNet','GBLUP','RRBLUP','XGBoost']:
        return 'Traditional'
    if name in ENS_COLORS or 'Stacking' in name or 'Ensemble' in name:
        return 'Ensemble'
    return 'DL'

# ============================================================================
# Figure 1: Per-dataset bar charts (3 panels)
# ============================================================================
print("\n--- Figure 1: Per-dataset bar charts ---")
fig, axes = plt.subplots(1, 3, figsize=(36, 14))
fig.suptitle('Genomic Prediction Ensemble — Per-Dataset Model Comparison (5-fold CV R²)',
             fontsize=22, fontweight='bold', y=1.01)

for ax_idx, (dname, (data, traits)) in enumerate(DATASETS.items()):
    ax = axes[ax_idx]
    trait0 = traits[0]
    # Get all models and sort by mean R²
    all_models = list(data[trait0].keys())
    means = {}
    for m in all_models:
        r2s = [data[t][m]['R2'] for t in traits if m in data[t]]
        means[m] = np.mean(r2s) if r2s else -999

    # Sort descending, but filter out utterly broken models (mean < -1)
    models_sorted = sorted([m for m in all_models if means[m] > -5], key=lambda m: means[m], reverse=True)
    # Cap at top 20 for readability
    models_sorted = models_sorted[:20]
    vals = [means[m] for m in models_sorted]
    colors = [model_color(m) for m in models_sorted]

    x = np.arange(len(models_sorted))
    bars = ax.bar(x, vals, 0.7, color=colors, edgecolor='white', linewidth=0.8, zorder=3)

    # Highlight best model
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor('#C62828')
    bars[best_idx].set_linewidth(3.0)

    # Value labels
    for i, (m, v) in enumerate(zip(models_sorted, vals)):
        if v == max(vals):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='#C62828')
        elif i < 8:
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.axhline(y=0, color='#666', linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models_sorted, rotation=55, ha='right', fontsize=7.5)
    ax.set_ylabel('R²', fontsize=13)
    ax.set_title(f'{dname} ({len(traits)} trait{"s" if len(traits)>1 else ""}): Best = {models_sorted[0]} ({vals[best_idx]:.3f})',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ymin = min(-0.5, min(vals) - 0.15)
    ymax = max(vals) + 0.15
    ax.set_ylim(ymin, ymax)

fig.tight_layout()
out = FIG_DIR / "01_per_dataset_bar_charts.png"
fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

# ============================================================================
# Figure 2: Cross-dataset comparison (common models only)
# ============================================================================
print("\n--- Figure 2: Cross-dataset comparison ---")
# Models present in ALL three datasets
w_models = set(wheat['TFW_DSI'].keys())
r_models = set(rice[list(rice.keys())[0]].keys())
m_models = set(maize[list(maize.keys())[0]].keys())
common = w_models & r_models & m_models

def avg_r2(data, m):
    r2s = [data[t][m]['R2'] for t in data if m in data[t]]
    return np.mean(r2s) if r2s else float('nan')

common_sorted = sorted(common, key=lambda m: (avg_r2(wheat,m)+avg_r2(rice,m)+avg_r2(maize,m))/3, reverse=True)

fig, ax = plt.subplots(figsize=(16, 10))
x = np.arange(len(common_sorted))
bar_w = 0.25

w_vals = [avg_r2(wheat, m) for m in common_sorted]
r_vals = [avg_r2(rice, m) for m in common_sorted]
m_vals = [avg_r2(maize, m) for m in common_sorted]

bars1 = ax.bar(x - bar_w, w_vals, bar_w, color='#2196F3', edgecolor='white', label='Wheat (1 trait)', zorder=3)
bars2 = ax.bar(x, r_vals, bar_w, color='#FF9800', edgecolor='white', label='Rice (10 traits)', zorder=3)
bars3 = ax.bar(x + bar_w, m_vals, bar_w, color='#4CAF50', edgecolor='white', label='Maize (4 traits)', zorder=3)

ax.axhline(y=0, color='#666', linewidth=1, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(common_sorted, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('R²', fontsize=13)
ax.set_title('Cross-Dataset Comparison — Models Common to All Three Datasets', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3, zorder=0)

fig.tight_layout()
out = FIG_DIR / "02_cross_dataset_comparison.png"
fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

# ============================================================================
# Figure 3: Stacking vs Best Single model per trait (scatter)
# ============================================================================
print("\n--- Figure 3: Stacking gain scatter ---")
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Stacking (Greedy) vs Best Single Model — Per Trait', fontsize=16, fontweight='bold')

for ax_idx, (dname, (data, traits)) in enumerate(DATASETS.items()):
    ax = axes[ax_idx]
    # Best single model = best non-ensemble, non-stacking model
    all_models = list(data[traits[0]].keys())
    singles = [m for m in all_models if 'Stacking' not in m and 'Ensemble' not in m and m not in ['Trad Ensemble']]
    stk_greedy_available = 'Stacking (Greedy)' in data[traits[0]]

    xs, ys = [], []
    for t in traits:
        best_single = max(singles, key=lambda m: data[t][m]['R2'] if m in data[t] else -999)
        bx = data[t][best_single]['R2']
        by = data[t].get('Stacking (Greedy)', data[t].get('Stacking (All)', {})).get('R2', bx)
        xs.append(bx); ys.append(by)

    mn = min(min(xs), min(ys)) - 0.03
    mx = max(max(xs), max(ys)) + 0.05
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.3, lw=1.5, label='y=x')

    for i, t in enumerate(traits):
        ax.scatter(xs[i], ys[i], s=180, edgecolors='#333', linewidth=1.2, zorder=4)
        ax.annotate(t.replace('_','\n')[:20], (xs[i], ys[i]), textcoords="offset points",
                    xytext=(8, 10), fontsize=7)

    wins = sum(1 for yv, xv in zip(ys, xs) if yv > xv)
    ax.set_xlabel('Best Single Model R²', fontsize=11)
    ax.set_ylabel('Stacking (Greedy) R²', fontsize=11)
    ax.set_title(f'{dname}: Stacking wins {wins}/{len(traits)} traits', fontsize=12, fontweight='bold')
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)

fig.tight_layout()
out = FIG_DIR / "03_stacking_gain_scatter.png"
fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

# ============================================================================
# Figure 4: Rice 10-trait detailed bar chart
# ============================================================================
print("\n--- Figure 4: Rice detailed per-trait ---")
fig, axes = plt.subplots(5, 2, figsize=(28, 32))
fig.suptitle('Rice: Per-Trait Model Comparison (5-fold CV R²)', fontsize=18, fontweight='bold')
rice_traits = sorted(rice.keys())

# Select top models for clarity
all_rice_models = list(rice[rice_traits[0]].keys())
rice_means = {m: np.mean([rice[t][m]['R2'] for t in rice_traits if m in rice[t]]) for m in all_rice_models}
top_rice = sorted([m for m in all_rice_models if rice_means[m] > -1],
                  key=lambda m: rice_means[m], reverse=True)[:12]

for idx, trait in enumerate(rice_traits):
    ax = axes[idx // 2][idx % 2]
    vals = [rice[trait][m]['R2'] if m in rice[trait] else 0 for m in top_rice]
    colors = [model_color(m) for m in top_rice]
    x = np.arange(len(top_rice))
    bars = ax.bar(x, vals, 0.65, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor('#C62828'); bars[best_idx].set_linewidth(2.5)
    ax.text(best_idx, vals[best_idx] + 0.03, f'{vals[best_idx]:.3f}', ha='center',
            va='bottom', fontsize=9, fontweight='bold', color='#C62828')
    ax.axhline(y=0, color='#666', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top_rice, rotation=60, ha='right', fontsize=6)
    ax.set_ylabel('R²', fontsize=9)
    ax.set_title(f'{trait}  (best: {top_rice[best_idx]} {vals[best_idx]:.3f})', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)
    ax.set_ylim(min(-0.5, min(vals)-0.1), max(vals)+0.12)

fig.tight_layout()
out = FIG_DIR / "04_rice_per_trait_detail.png"
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

# ============================================================================
# Figure 5-7: Predicted vs True scatter plots
# ============================================================================
print("\n--- Figure 5-7: Predicted vs True scatter plots ---")
print("Running 5-fold CV for OOF predictions...")

# For each dataset, get OOF predictions for the best model on each trait
def get_oof_for_model(data_dict, traits, model_name, crop='rice'):
    """Run 5-fold CV to get OOF predictions for a specific model."""
    if crop == 'rice':
        trait_data_raw = ge.load_rice_data()
    elif crop == 'maize':
        X_all, y_dict = ge.load_iranian_data(max_markers=None)
        var_thresh = 0.005
        vars_per_marker = np.var(X_all, axis=0)
        keep = vars_per_marker >= var_thresh
        if keep.sum() < X_all.shape[1]:
            X_all = X_all[:, keep]

    results = {}
    for trait in traits:
        print(f"  {crop}/{trait} — {model_name} ...")
        if crop == 'rice':
            X_all_t, y = trait_data_raw[trait]
        elif crop == 'maize':
            y = y_dict[trait]
            X_all_t = X_all

        y = y.astype(np.float32)
        n_snps = min(ge.GWAS_TOP_K, max(50, X_all_t.shape[1] - 50))
        oof = np.zeros(len(y))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fi, (tr, te) in enumerate(kf.split(X_all_t)):
            Xtr_raw, Xte_raw = X_all_t[tr], X_all_t[te]
            ytr, yte = y[tr], y[te]
            maf_idx = ge.maf_filter(Xtr_raw, ge.MAF_THRESHOLD)
            if len(maf_idx) >= n_snps:
                Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]

            gidx_gwas = ge.gwas_select(Xtr_raw, ytr, n_snps)
            Xtr = Xtr_raw[:, gidx_gwas]; Xte = Xte_raw[:, gidx_gwas]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
            Xte_s = sc.transform(Xte).astype(np.float32)

            if model_name in ge.TRAD_NAMES:
                G_train = Xtr_s @ Xtr_s.T / n_snps
                G_te = Xte_s @ Xtr_s.T / n_snps
                for tname, build_fn, fit_fn, pred_fn, _ in ge._make_trad_configs(G_train, G_te, n_snps, len(tr)):
                    if tname == model_name:
                        tmodel = build_fn(); fit_fn(tmodel, Xtr_s, ytr)
                        oof[te] = pred_fn(tmodel, Xte_s)
                        break
            else:
                gidx_dl, _, Xtr_dl, Xte_dl = ge._select_dl_markers(Xtr_raw, Xte_raw, ytr, gidx_gwas, None, n_snps)
                model = ge.create_model(model_name, n_snps)
                bs = 32 if model_name.startswith('FGN') or model_name == 'GenomicFM' else 64
                wd = 5e-3 if model_name == 'AdditiveGenomicNet' else 1e-3
                model = ge.train_torch_model(model, Xtr_dl, ytr, epochs=300, batch_size=bs, lr=2e-3, weight_decay=wd, patience=30)
                oof[te] = ge.predict_torch_model(model, Xte_dl)
            torch.cuda.empty_cache()

        results[trait] = oof
    return results

# --- Rice scatter plots (use best single model: XGBoost) ---
rice_best_model = 'XGBoost'
print("\nRice OOF predictions (XGBoost)...")
rice_oof = get_oof_for_model(rice, sorted(rice.keys()), rice_best_model, crop='rice')

n_traits = len(rice_oof)
cols = 4; rows = (n_traits + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4.5))
fig.suptitle(f'Rice: Predicted vs True Values — {rice_best_model} (5-fold OOF)',
             fontsize=16, fontweight='bold')

for idx, trait in enumerate(sorted(rice_oof.keys())):
    ax = axes[idx // cols][idx % cols] if rows > 1 else axes[idx]
    oof = rice_oof[trait]

    # Get true y values
    trait_data_raw = ge.load_rice_data()
    _, y = trait_data_raw[trait]
    y = y.astype(np.float32)
    # y and oof should match since data loading is deterministic

    r2 = r2_score(y, oof)
    ax.scatter(y, oof, alpha=0.5, s=25, c='#1565C0', edgecolors='none', zorder=3)
    mn = min(y.min(), oof.min()); mx = max(y.max(), oof.max())
    pad = (mx - mn) * 0.08
    ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], '--', color='#E53935', alpha=0.5, lw=1.2)
    ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    ax.set_title(f'{trait}\nR²={r2:.4f}', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2)

# Hide extra subplots
for idx in range(n_traits, rows*cols):
    ax = axes[idx // cols][idx % cols] if rows > 1 else axes[idx]
    ax.axis('off')

fig.tight_layout()
out = FIG_DIR / "05_rice_pred_vs_true.png"
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

# --- Maize scatter plots (use best single model: GBLUP) ---
maize_best_model = 'GBLUP'
print("\nMaize OOF predictions (GBLUP)...")
maize_oof = get_oof_for_model(maize, sorted(maize.keys()), maize_best_model, crop='maize')

n_traits = len(maize_oof)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Maize: Predicted vs True Values — {maize_best_model} (5-fold OOF)',
             fontsize=15, fontweight='bold')

maize_data = ge.load_iranian_data(max_markers=None)[1]

for idx, trait in enumerate(sorted(maize_oof.keys())):
    ax = axes[idx // 2][idx % 2]
    oof = maize_oof[trait]
    y = maize_data[trait].astype(np.float32)

    r2 = r2_score(y, oof)
    ax.scatter(y, oof, alpha=0.5, s=20, c='#2E7D32', edgecolors='none', zorder=3)
    mn = min(y.min(), oof.min()); mx_ = max(y.max(), oof.max())
    pad = (mx_ - mn) * 0.08
    ax.plot([mn-pad, mx_+pad], [mn-pad, mx_+pad], '--', color='#E53935', alpha=0.5, lw=1.2)
    ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    ax.set_title(f'{trait}\nR²={r2:.4f}', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.2)

fig.tight_layout()
out = FIG_DIR / "06_maize_pred_vs_true.png"
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

# --- Wheat scatter plot (1 trait) ---
# Wheat data is only on HPC, so we generate a comprehensive summary figure instead
print("\nWheat scatter data unavailable locally (HPC only). Generating summary figure instead.")

# ============================================================================
# Figure 7: Final summary ranking (all three datasets combined)
# ============================================================================
print("\n--- Figure 8: Final summary ranking ---")
fig, ax = plt.subplots(figsize=(14, 10))

# Combine means across all three datasets for common models
combined = {}
for m in common:
    w = avg_r2(wheat, m); r = avg_r2(rice, m); mz = avg_r2(maize, m)
    combined[m] = (w + r + mz) / 3

sorted_all = sorted(combined.items(), key=lambda x: x[1], reverse=True)
models_ranked = [s[0] for s in sorted_all]
vals_ranked = [s[1] for s in sorted_all]
colors_ranked = [model_color(m) for m in models_ranked]

y_pos = range(len(models_ranked))
bars = ax.barh(y_pos, vals_ranked, 0.7, color=colors_ranked, edgecolor='white', linewidth=1, zorder=3)

# Highlight top 3
for i in range(min(3, len(models_ranked))):
    bars[i].set_edgecolor('#C62828')
    bars[i].set_linewidth(2.5)

for i, (m, v) in enumerate(zip(models_ranked, vals_ranked)):
    ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
    mtype = model_type(m)
    ax.text(-0.35, i, f'{m} [{mtype}]', va='center', ha='right', fontsize=9,
            fontweight='bold' if 'Stacking' in m or 'Ensemble' in m else 'normal')

ax.set_yticks([])
ax.set_xlabel('Mean R² (avg across Wheat + Rice + Maize)', fontsize=12)
ax.set_title('Three-Dataset Combined Ranking', fontsize=15, fontweight='bold')
ax.grid(axis='x', alpha=0.25)
ax.set_xlim(-0.75, max(vals_ranked) + 0.08)
ax.invert_yaxis()

fig.tight_layout()
out = FIG_DIR / "07_combined_ranking.png"
fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

print(f"\n{'='*60}")
print(f"All figures saved to: {FIG_DIR}")
print(f"Files: {[f.name for f in sorted(FIG_DIR.glob('*.png'))]}")
print(f"\nDone!")
