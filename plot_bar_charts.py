#!/usr/bin/env python
"""Regenerate bar chart figures (01-04, 07) from updated intermediate JSON — no GPU needed."""
import json, sys, os, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Fix Windows GBK encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme (same as plot_all_figures.py)
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

# Load results
print("Loading results...")
def _load_json(path):
    if path.exists():
        return json.load(open(path, 'r', encoding='utf-8'))
    print(f"  WARNING: {path} not found")
    return None

wheat = _load_json(SCRIPT_DIR/"results/wheat_ensemble/ensemble_intermediate.json")
wheat2000 = _load_json(SCRIPT_DIR/"results/wheat2000_ensemble/ensemble_intermediate.json")
rice = _load_json(SCRIPT_DIR/"results/rice_ensemble/ensemble_intermediate.json")
maize = _load_json(SCRIPT_DIR/"results/maize_ensemble/ensemble_intermediate.json")

DATASETS = []
if wheat is not None:
    W_TRAITS = list(wheat.keys())
    DATASETS.append(('Wheat', wheat, W_TRAITS))
else: W_TRAITS = []
if wheat2000 is not None:
    W2_TRAITS = sorted(wheat2000.keys())
    DATASETS.append(('Wheat2000', wheat2000, W2_TRAITS))
else: W2_TRAITS = []
if rice is not None:
    R_TRAITS = sorted(rice.keys())
    DATASETS.append(('Rice', rice, R_TRAITS))
else: R_TRAITS = []
if maize is not None:
    M_TRAITS = sorted(maize.keys())
    DATASETS.append(('Maize', maize, M_TRAITS))
else: M_TRAITS = []

print(f"Loaded {len(DATASETS)} dataset(s): {[d[0] for d in DATASETS]}")

# ============================================================================
# Figure 01: Per-dataset bar charts
# ============================================================================
print("[01] Per-dataset bar charts...")
n_datasets = len(DATASETS)
fig, axes = plt.subplots(1, n_datasets, figsize=(12 * n_datasets, 14), squeeze=False)
fig.suptitle('Genomic Prediction Ensemble — Per-Dataset Model Comparison (5-fold CV R²)',
             fontsize=22, fontweight='bold', y=1.01)

for ax_idx, (dname, data, traits) in enumerate(DATASETS):
    ax = axes[0, ax_idx]
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
print("  -> 01_per_dataset_bar_charts.png")

# ============================================================================
# Figure 02: Cross-dataset comparison
# ============================================================================
print("[02] Cross-dataset comparison...")
model_sets = [set(data[traits[0]].keys()) for _, data, traits in DATASETS]
common = model_sets[0]
for s in model_sets[1:]: common = common & s
common_sorted = sorted(common, key=lambda m: np.mean([avg_r2(d, m) for _, d, _ in DATASETS]), reverse=True)

fig, ax = plt.subplots(figsize=(18, 10))
x = np.arange(len(common_sorted)); bar_w = 0.25
dataset_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']
for bi, (dname, data, _) in enumerate(DATASETS):
    vals = [avg_r2(data, m) for m in common_sorted]
    ax.bar(x + (bi - (len(DATASETS)-1)/2) * bar_w, vals, bar_w, color=dataset_colors[bi % len(dataset_colors)],
           edgecolor='white', label=f'{dname} ({len(DATASETS[bi][2])} trait{"s" if len(DATASETS[bi][2])>1 else ""})', zorder=3)
ax.axhline(y=0, color='#666', linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(common_sorted, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('R²', fontsize=13)
ax.set_title('Cross-Dataset Comparison — Models Common to All Three Datasets', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper right'); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR/'02_cross_dataset_comparison.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("  -> 02_cross_dataset_comparison.png")

# ============================================================================
# Figure 03: Stacking gain scatter
# ============================================================================
print("[03] Stacking gain scatter...")
fig, axes = plt.subplots(1, n_datasets, figsize=(8 * n_datasets, 8), squeeze=False)
fig.suptitle('Stacking (Greedy) vs Best Single Model — Per Trait', fontsize=16, fontweight='bold')

for ax_idx, (dname, data, traits) in enumerate(DATASETS):
    ax = axes[0, ax_idx]
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
print("  -> 03_stacking_gain_scatter.png")

# ============================================================================
# Figure 04: Rice per-trait detail
# ============================================================================
if rice is None:
    print("[04] Rice data not available — skipping")
else:
    print("[04] Rice per-trait detail...")
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
    print("  -> 04_rice_per_trait_detail.png")

# ============================================================================
# Figure 07: Combined ranking
# ============================================================================
print("[07] Combined ranking...")
combined = {}
for m in common:
    combined[m] = np.mean([avg_r2(d, m) for _, d, _ in DATASETS])
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
ax.set_yticks([]); ax.set_xlabel(f'Mean R² ({"+".join(d[0] for d in DATASETS)} avg)', fontsize=12)
ax.set_title(f'{len(DATASETS)}-Dataset Combined Ranking', fontsize=15, fontweight='bold')
ax.grid(axis='x', alpha=0.25); ax.set_xlim(-0.75, max(vals_r)+0.08); ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIG_DIR/'07_combined_ranking.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("  -> 07_combined_ranking.png")

# ============================================================================
print(f"\nAll 5 bar chart figures saved to: {FIG_DIR}")
for f in sorted(FIG_DIR.glob('0*.png')):
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
print("Done!")
