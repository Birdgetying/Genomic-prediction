"""
Rice Genomic Prediction — 全模型对比可视化
传统模型 (6) + FGN/EFM/MICNN 集成系统 (8)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ── 加载数据 ──────────────────────────────────────────────
with open(SCRIPT_DIR / "results/rice_comparison/final_results_20260515_155733.json") as f:
    traditional = json.load(f)

with open(SCRIPT_DIR / "results/rice_ensemble/ensemble_final_20260517_085235.json") as f:
    ensemble = json.load(f)

TRAITS = [
    'Heading_date', 'Plant_height', 'Num_panicles',
    'Num_effective_panicles', 'Yield', 'Grain_weight',
    'Spikelet_length', 'Grain_length', 'Grain_width', 'Grain_thickness'
]

TRADITIONAL_MODELS = ['RRBLUP', 'GBLUP', 'XGBoost', 'ElasticNet', 'GWAS_RRBLUP', 'Ensemble']
DL_MODELS = ['FGN', 'EFM', 'MICNN', 'FGN v2', 'EFM v2', 'MICNN v2', 'FusionNet', 'Stacking']
ALL_MODELS = TRADITIONAL_MODELS + DL_MODELS

# ── 颜色方案 ──────────────────────────────────────────────
# 传统模型: 蓝色系
TRAD_COLORS = {
    'RRBLUP':      '#90CAF9',  # light blue
    'GBLUP':       '#64B5F6',
    'XGBoost':     '#42A5F5',
    'ElasticNet':  '#2196F3',
    'GWAS_RRBLUP': '#1E88E5',
    'Ensemble':    '#1565C0',  # dark blue
}
# 深度学习: 暖色系
DL_COLORS = {
    'FGN':       '#FFB74D',  # orange
    'EFM':       '#FFD54F',  # yellow
    'MICNN':     '#FF8A65',  # deep orange
    'FGN v2':    '#F57C00',  # dark orange
    'EFM v2':    '#FBC02D',  # dark yellow
    'MICNN v2':  '#E64A19',  # deep orange
    'FusionNet': '#E91E63',  # pink
    'Stacking':  '#C62828',  # red — best
}
ALL_COLORS = {**TRAD_COLORS, **DL_COLORS}

# ── 提取数据 ──────────────────────────────────────────────
data = {}
for trait in TRAITS:
    data[trait] = {}
    for model in TRADITIONAL_MODELS:
        data[trait][model] = traditional[trait][model]['R2']
    for model in DL_MODELS:
        if model in ensemble[trait]:
            data[trait][model] = ensemble[trait][model]['R2']

# 计算均值
means = {}
for model in ALL_MODELS:
    vals = [data[t][model] for t in TRAITS]
    means[model] = np.mean(vals)

# ── 绘图 ──────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 32))
fig.suptitle('Rice Genomic Prediction — All Models Comparison (5-fold CV R²)',
             fontsize=20, fontweight='bold', y=0.995)

# 颜色条带分割
n_traits = len(TRAITS)
n_cols = 3
n_rows = (n_traits + 2) // n_cols  # +2 for overall + ranking panels

# ── 子图 1-10: 各性状 R² ──
for idx, trait in enumerate(TRAITS):
    ax = plt.subplot(n_rows, n_cols, idx + 1)

    x = np.arange(len(ALL_MODELS))
    vals = [data[trait][m] for m in ALL_MODELS]
    colors = [ALL_COLORS[m] for m in ALL_MODELS]

    bars = ax.bar(x, vals, 0.72, color=colors, edgecolor='white', linewidth=0.5, zorder=3)

    # 标注最佳值
    best_idx = np.argmax(vals)
    best_val = vals[best_idx]
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2.0)

    # 在最佳 bar 上标值
    ax.text(best_idx, best_val + 0.04, f'{best_val:.3f}', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color='#C62828')

    # 传统/深度学习 分隔线
    ax.axvline(x=5.5, color='#333', linewidth=1.2, linestyle='--', alpha=0.6, zorder=2)

    # 零线
    ax.axhline(y=0, color='#999', linewidth=0.8, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_MODELS, rotation=60, ha='right', fontsize=6.5)
    ax.set_ylabel('R²', fontsize=9)

    # 标题: 性状名 + 最佳模型
    short_name = trait.replace('_', '\n')[:22]
    ax.set_title(f'{trait}\nbest: {ALL_MODELS[best_idx]} ({best_val:.3f})',
                 fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.25, zorder=0)
    ax.set_ylim(min(-0.4, min(vals) - 0.15), max(vals) + 0.18)

# ── 子图 11: 总体平均排名 ──
ax_rank = plt.subplot(n_rows, n_cols, n_traits + 1)

sorted_models = sorted(ALL_MODELS, key=lambda m: means[m], reverse=True)
sorted_vals = [means[m] for m in sorted_models]
sorted_colors = [ALL_COLORS[m] for m in sorted_models]

bars = ax_rank.barh(range(len(sorted_models)), sorted_vals, 0.7, color=sorted_colors,
                     edgecolor='white', linewidth=0.5, zorder=3)

# 标值
for i, (m, v) in enumerate(zip(sorted_models, sorted_vals)):
    ax_rank.text(v + 0.008, i, f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
    # 标记类型
    tag = ' [DL]' if m in DL_MODELS else ' [Trad]'
    ax_rank.text(-0.25, i, m + tag, va='center', ha='right', fontsize=8,
                 fontweight='bold' if m in ('Stacking', 'FusionNet') else 'normal')

# 分隔线
trad_pos = sum(1 for m in sorted_models if m in TRADITIONAL_MODELS) - 0.5
dl_pos = trad_pos
ax_rank.axhline(y=len(ALL_MODELS) - trad_pos - 1 + 0.5, color='#333', linewidth=1.2,
                linestyle='--', alpha=0.5, zorder=2)

ax_rank.set_yticks([])
ax_rank.set_xlabel('Mean R² across 10 traits', fontsize=10)
ax_rank.set_title('Overall Ranking', fontsize=11, fontweight='bold')
ax_rank.grid(axis='x', alpha=0.25, zorder=0)
ax_rank.set_xlim(-0.55, max(sorted_vals) + 0.08)
ax_rank.invert_yaxis()

# ── 子图 12: 深度学习 vs 传统 散点 ──
ax_scatter = plt.subplot(n_rows, n_cols, n_traits + 2)

# 对每个性状取传统最佳 vs DL最佳
trad_best_per_trait = []
dl_best_per_trait = []
for trait in TRAITS:
    trad_best = max(data[trait][m] for m in TRADITIONAL_MODELS)
    dl_best = max(data[trait][m] for m in DL_MODELS)
    trad_best_per_trait.append(trad_best)
    dl_best_per_trait.append(dl_best)

# 对角线
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

# 计数标注
dl_wins = sum(1 for t, d in zip(trad_best_per_trait, dl_best_per_trait) if d > t)
ax_scatter.text(0.95, 0.08, f'DL wins {dl_wins}/{len(TRAITS)} traits',
                transform=ax_scatter.transAxes, fontsize=10, fontweight='bold',
                ha='right', color='#C62828',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, ec='#ddd'))

# ── 图例 ──
legend_ax = fig.add_axes([0.02, 0.005, 0.96, 0.025])
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)
legend_ax.axis('off')

# 传统模型图例行
trad_patches = [plt.Rectangle((0, 0), 1, 1, fc=TRAD_COLORS[m], ec='white', lw=0.5)
                for m in TRADITIONAL_MODELS]
# DL图例行
dl_patches = [plt.Rectangle((0, 0), 1, 1, fc=DL_COLORS[m], ec='white', lw=0.5)
              for m in DL_MODELS]

legend1 = legend_ax.legend(trad_patches, TRADITIONAL_MODELS,
                           loc='upper left', ncol=6, fontsize=7.5,
                           title='Traditional', title_fontsize=8,
                           handlelength=1.2, handleheight=0.8)
legend_ax.add_artist(legend1)
legend2 = legend_ax.legend(dl_patches, DL_MODELS,
                           loc='upper right', ncol=8, fontsize=7.5,
                           title='Deep Learning (this study)', title_fontsize=8,
                           handlelength=1.2, handleheight=0.8)

# ── 底部注释 ──
fig.text(0.5, 0.008, '529 rice accessions × 360K SNPs → GWAS top-3000 × 5-fold CV  |  GPU: NVIDIA A800 80GB  |  Total DL training time: 49.7 min',
         ha='center', fontsize=8.5, color='#555', style='italic')

# ── 保存 ──
output_dir = SCRIPT_DIR / "results" / "rice_ensemble"
output_dir.mkdir(parents=True, exist_ok=True)
out_path = output_dir / "all_models_comparison.png"
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_path}")
print(f"DL wins {dl_wins}/{len(TRAITS)} traits")
print(f"Best overall: {sorted_models[0]} (mean R2={sorted_vals[0]:.4f})")
