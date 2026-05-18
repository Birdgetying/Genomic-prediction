"""
Rice Genomic Prediction — 全模型对比可视化
传统模型 (5) + DL/集成 (6 DL + 3 集成)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import glob

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "results" / "rice_ensemble"

# ── 加载最新结果 ──
json_files = sorted(glob.glob(str(OUTPUT_DIR / "ensemble_final_*.json")))
if not json_files:
    raise FileNotFoundError("No ensemble_final_*.json found in results/rice_ensemble/")
latest_json = json_files[-1]
print(f"Loading: {latest_json}")
with open(latest_json) as f:
    all_data = json.load(f)

TRAITS = [
    'Heading_date', 'Plant_height', 'Num_panicles',
    'Num_effective_panicles', 'Yield', 'Grain_weight',
    'Spikelet_length', 'Grain_length', 'Grain_width', 'Grain_thickness'
]

# ── 按类型分组模型 ──
trait0 = all_data[TRAITS[0]]
TRAD_MODELS = [m for m in trait0 if trait0[m].get('Type') == 'Traditional']
DL_MODELS = [m for m in trait0 if trait0[m].get('Type') == 'DL']
ENS_MODELS = [m for m in trait0 if trait0[m].get('Type') == 'Ensemble']
ALL_MODELS = TRAD_MODELS + DL_MODELS + ENS_MODELS

print(f"Traditional ({len(TRAD_MODELS)}): {TRAD_MODELS}")
print(f"DL ({len(DL_MODELS)}): {DL_MODELS}")
print(f"Ensemble ({len(ENS_MODELS)}): {ENS_MODELS}")

# ── 颜色方案 ──
TRAD_COLORS = {
    'RRBLUP':      '#90CAF9',
    'GBLUP':       '#64B5F6',
    'XGBoost':     '#42A5F5',
    'ElasticNet':  '#2196F3',
    'GWAS_RRBLUP': '#1E88E5',
}
DL_COLORS = {
    'FGN':       '#FFB74D',
    'EFM':       '#FFD54F',
    'MICNN':     '#FF8A65',
    'FGN v2':    '#F57C00',
    'EFM v2':    '#FBC02D',
    'MICNN v2':  '#E64A19',
    'FusionNet': '#E91E63',
}
ENS_COLORS = {
    'Stacking (DL)':  '#4CAF50',
    'Stacking (All)': '#2E7D32',
    'Trad Ensemble':  '#1565C0',
}
ALL_COLORS = {**TRAD_COLORS, **DL_COLORS, **ENS_COLORS}

# ── 提取数据 ──
data = {}
for trait in TRAITS:
    data[trait] = {}
    for model in ALL_MODELS:
        if model in all_data[trait]:
            data[trait][model] = all_data[trait][model]['R2']

means = {}
for model in ALL_MODELS:
    vals = [data[t][model] for t in TRAITS if model in data[t]]
    means[model] = np.mean(vals) if vals else 0

# ── 绘图 ──
fig = plt.figure(figsize=(26, 34))
fig.suptitle('Rice Genomic Prediction — All Models Comparison (5-fold CV R²)',
             fontsize=20, fontweight='bold', y=0.995)

n_traits = len(TRAITS)
n_cols = 3
n_rows = (n_traits + 2 + n_cols - 1) // n_cols

# ── 子图 1-10: 各性状 R² ──
for idx, trait in enumerate(TRAITS):
    ax = plt.subplot(n_rows, n_cols, idx + 1)

    x = np.arange(len(ALL_MODELS))
    vals = [data[trait].get(m, 0) for m in ALL_MODELS]
    colors = [ALL_COLORS[m] for m in ALL_MODELS]

    bars = ax.bar(x, vals, 0.72, color=colors, edgecolor='white', linewidth=0.5, zorder=3)

    best_idx = np.argmax(vals)
    best_val = vals[best_idx]
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2.0)

    ax.text(best_idx, best_val + 0.04, f'{best_val:.3f}', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color='#C62828')

    # 分割线: Traditional | DL | Ensemble
    if TRAD_MODELS:
        ax.axvline(x=len(TRAD_MODELS) - 0.5, color='#333', linewidth=1.2, linestyle='--', alpha=0.6, zorder=2)
    if DL_MODELS:
        ax.axvline(x=len(TRAD_MODELS) + len(DL_MODELS) - 0.5, color='#333', linewidth=1.2, linestyle='--', alpha=0.6, zorder=2)

    ax.axhline(y=0, color='#999', linewidth=0.8, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_MODELS, rotation=60, ha='right', fontsize=5.5)
    ax.set_ylabel('R²', fontsize=9)

    ax.set_title(f'{trait}\nbest: {ALL_MODELS[best_idx]} ({best_val:.3f})',
                 fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.25, zorder=0)
    ax.set_ylim(min(-0.5, min(vals) - 0.15), max(vals) + 0.18)

# ── 子图 11: 总体平均排名 ──
ax_rank = plt.subplot(n_rows, n_cols, n_traits + 1)

sorted_models = sorted(ALL_MODELS, key=lambda m: means[m], reverse=True)
sorted_vals = [means[m] for m in sorted_models]
sorted_colors = [ALL_COLORS[m] for m in sorted_models]

bars = ax_rank.barh(range(len(sorted_models)), sorted_vals, 0.7, color=sorted_colors,
                     edgecolor='white', linewidth=0.5, zorder=3)

for i, (m, v) in enumerate(zip(sorted_models, sorted_vals)):
    ax_rank.text(v + 0.008, i, f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
    mtype = trait0[m].get('Type', 'DL')
    tag = f' [{mtype}]'
    ax_rank.text(-0.30, i, m + tag, va='center', ha='right', fontsize=7.5,
                 fontweight='bold' if m in ENS_MODELS else 'normal')

ax_rank.set_yticks([])
ax_rank.set_xlabel('Mean R² across 10 traits', fontsize=10)
ax_rank.set_title('Overall Ranking', fontsize=11, fontweight='bold')
ax_rank.grid(axis='x', alpha=0.25, zorder=0)
ax_rank.set_xlim(-0.65, max(sorted_vals) + 0.08)
ax_rank.invert_yaxis()

# ── 子图 12: DL vs Traditional 散点 ──
ax_scatter = plt.subplot(n_rows, n_cols, n_traits + 2)

trad_best_per_trait = []
dl_best_per_trait = []
for trait in TRAITS:
    trad_vals = [data[trait][m] for m in TRAD_MODELS if m in data[trait]]
    dl_vals = [data[trait][m] for m in DL_MODELS if m in data[trait]]
    trad_best = max(trad_vals) if trad_vals else 0
    dl_best = max(dl_vals) if dl_vals else 0
    trad_best_per_trait.append(trad_best)
    dl_best_per_trait.append(dl_best)

mx = max(max(trad_best_per_trait), max(dl_best_per_trait)) + 0.05
mn = min(min(trad_best_per_trait), min(dl_best_per_trait)) - 0.05
ax_scatter.plot([mn, mx], [mn, mx], 'k--', alpha=0.3, lw=1, zorder=1, label='y=x')

sc_colors = plt.cm.RdYlGn([0.15 + 0.7 * (i / (len(TRAITS) - 1)) for i in range(len(TRAITS))])
for i, trait in enumerate(TRAITS):
    ax_scatter.scatter(trad_best_per_trait[i], dl_best_per_trait[i],
                       c=[sc_colors[i]], s=120, edgecolors='#333', linewidth=0.8, zorder=3)
    offset_y = 0.03 if dl_best_per_trait[i] >= trad_best_per_trait[i] else -0.05
    ax_scatter.annotate(trait.replace('_', '\n')[:18],
                        (trad_best_per_trait[i], dl_best_per_trait[i]),
                        textcoords="offset points", xytext=(6, offset_y * 40),
                        fontsize=6.5, alpha=0.9)

ax_scatter.set_xlabel('Best Traditional R²', fontsize=10)
ax_scatter.set_ylabel('Best DL R²', fontsize=10)
ax_scatter.set_title('Best per Trait: DL vs Traditional', fontsize=11, fontweight='bold')
ax_scatter.set_xlim(mn, mx)
ax_scatter.set_ylim(mn, mx)
ax_scatter.set_aspect('equal')
ax_scatter.grid(alpha=0.25, zorder=0)

dl_wins = sum(1 for t, d in zip(trad_best_per_trait, dl_best_per_trait) if d > t)
ax_scatter.text(0.95, 0.08, f'DL wins {dl_wins}/{len(TRAITS)} traits',
                transform=ax_scatter.transAxes, fontsize=10, fontweight='bold',
                ha='right', color='#C62828',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, ec='#ddd'))

# ── 图例: 三行 ──
legend_ax = fig.add_axes([0.02, 0.003, 0.96, 0.03])
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)
legend_ax.axis('off')

trad_patches = [plt.Rectangle((0, 0), 1, 1, fc=TRAD_COLORS[m], ec='white', lw=0.5) for m in TRAD_MODELS]
dl_patches = [plt.Rectangle((0, 0), 1, 1, fc=DL_COLORS[m], ec='white', lw=0.5) for m in DL_MODELS]
ens_patches = [plt.Rectangle((0, 0), 1, 1, fc=ENS_COLORS[m], ec='white', lw=0.5) for m in ENS_MODELS]

leg1 = legend_ax.legend(trad_patches, TRAD_MODELS, loc='upper left', ncol=5, fontsize=7,
                        title='Traditional', title_fontsize=8, handlelength=1.2, handleheight=0.8)
legend_ax.add_artist(leg1)
leg2 = legend_ax.legend(dl_patches, DL_MODELS, loc='upper center', ncol=7, fontsize=7,
                        title='Deep Learning', title_fontsize=8, handlelength=1.2, handleheight=0.8)
legend_ax.add_artist(leg2)
leg3 = legend_ax.legend(ens_patches, ENS_MODELS, loc='upper right', ncol=3, fontsize=7,
                        title='Ensemble', title_fontsize=8, handlelength=1.2, handleheight=0.8)

fig.text(0.5, 0.005, '529 rice accessions × 360K SNPs → MAF≥5% + GWAS top-3000 × 5-fold CV  |  GPU: NVIDIA A800 80GB',
         ha='center', fontsize=8.5, color='#555', style='italic')

out_path = OUTPUT_DIR / "all_models_comparison.png"
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_path}")
print(f"DL wins {dl_wins}/{len(TRAITS)} traits")
print(f"Best overall: {sorted_models[0]} (mean R²={sorted_vals[0]:.4f})")
