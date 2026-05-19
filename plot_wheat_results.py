"""
小麦结果可视化 — 从 ensemble_intermediate.json 生成多模型对比图
亦可被 wheat_models_ensemble.py 的 main() 自动调用
"""

import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# 设置中文字体 (Windows)
for font_name in ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC']:
    for f in fm.fontManager.ttflist:
        if font_name in f.name:
            plt.rcParams['font.family'] = f.name
            break
    else:
        continue
    break
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "wheat_ensemble"

MODEL_COLORS = {
    'RRBLUP': '#64B5F6', 'GBLUP': '#42A5F5', 'XGBoost': '#1E88E5',
    'ElasticNet': '#1976D2', 'GWAS_RRBLUP': '#0D47A1',
    'FGN': '#FFB74D', 'MICNN': '#FF8A65',
    'FGN v2': '#F57C00', 'MICNN v2': '#E64A19',
    'FGN v3': '#BF360C',
    'PreFGN': '#00BCD4', 'DeepKernelGP': '#4CAF50',
    'FusionNet': '#E91E63',
    'Stacking (DL)': '#D32F2F', 'Stacking (All)': '#B71C1C',
    'Trad Ensemble': '#0D47A1', 'ResFGN': '#2E7D32',
}


def plot_single_trait(results, trait_name, save_path):
    """单性状: 水平柱状图 + R² 标注"""
    models_data = results[trait_name]
    items = sorted(models_data.items(), key=lambda x: x[1].get('R2', -99), reverse=True)

    names = [m for m, _ in items]
    r2s = [v.get('R2', 0) for _, v in items]
    corrs = [v.get('Correlation', 0) for _, v in items]
    type_abbr = {'Traditional': '传', 'DL': '深', 'Hybrid': '混', 'Ensemble': '集'}
    mtypes = [v.get('Type', 'DL') for _, v in items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12),
                                    gridspec_kw={'width_ratios': [1.4, 1]})

    # --- 左图: R² 水平柱状图 ---
    colors = [MODEL_COLORS.get(n, '#999999') for n in names]
    bars = ax1.barh(names, r2s, color=colors, alpha=0.88, edgecolor='white', linewidth=0.8,
                    height=0.7)
    ax1.set_xlabel('R² (5折交叉验证)', fontsize=13)
    ax1.set_title(f'小麦 — {trait_name} — 所有模型 R² 对比', fontsize=15, fontweight='bold')
    ax1.axvline(0, c='k', lw=0.5)
    ax1.grid(axis='x', alpha=0.2)

    # 标注数值
    for b, r2, c, mt in zip(bars, r2s, corrs, mtypes):
        sign = '+' if r2 >= 0 else ''
        label = f'{sign}{r2:.3f}  (r={c:.3f}) [{type_abbr.get(mt, mt[0])}]'
        x_pos = b.get_width() + 0.01 if b.get_width() >= 0 else b.get_width() - 0.12
        ha = 'left' if b.get_width() >= 0 else 'right'
        ax1.text(x_pos, b.get_y() + b.get_height()/2, label,
                 va='center', ha=ha, fontsize=8, fontweight='bold')

    # 图例: 类型色块
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1E88E5', label='传统模型 (GWAS筛选)'),
        Patch(facecolor='#FF8A65', label='深度学习 (HaploScore / Hybrid)'),
        Patch(facecolor='#D32F2F', label='集成模型 (Stacking)'),
        Patch(facecolor='#2E7D32', label='混合模型 (ResFGN)'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)

    ax1.invert_yaxis()

    # --- 右图: 类型分组 + 相关性 ---
    # 按类型分组: Traditional, DL, Ensemble, Hybrid
    type_order = {'Traditional': 0, 'DL': 1, 'Hybrid': 2, 'Ensemble': 3}
    type_colors = {'Traditional': '#1E88E5', 'DL': '#FF8A65',
                   'Hybrid': '#2E7D32', 'Ensemble': '#D32F2F'}

    groups = {}
    for n, r2, mt in zip(names, r2s, mtypes):
        groups.setdefault(mt, ([], []))[0].append(n)
        groups[mt][1].append(r2)

    ax2.set_xlim(-0.5, 2.0)
    ax2.set_ylim(-1, max(r2s) + 0.1 if r2s else 1)
    ax2.set_title('不同模型类型 R² 分布', fontsize=15, fontweight='bold')
    ax2.set_ylabel('R²', fontsize=13)
    ax2.grid(axis='y', alpha=0.2)
    ax2.axhline(0, c='k', lw=0.5)

    for mt in ['Traditional', 'DL', 'Hybrid', 'Ensemble']:
        if mt in groups:
            gnames, gr2s = groups[mt]
            x_pos = type_order[mt]
            jitter = np.linspace(-0.35, 0.35, len(gnames)) if len(gnames) > 1 else [0]
            for jx, (gn, gr2) in zip(jitter, zip(gnames, gr2s)):
                ax2.scatter(x_pos + jx, gr2, c=type_colors[mt], s=120, alpha=0.8,
                           edgecolors='white', linewidth=1.0, zorder=5)
                # 标注 top 模型
                if gr2 > 0.55:
                    ax2.annotate(gn, (x_pos + jx, gr2), textcoords="offset points",
                                xytext=(8, 4), fontsize=6.5, alpha=0.9)
        # 组均值
        if mt in groups:
            mean_r2 = np.mean(groups[mt][1])
            ax2.axhline(mean_r2, xmin=(type_order[mt]-0.4+2.5)/3.5,
                       xmax=(type_order[mt]+0.4+2.5)/3.5,
                       color=type_colors[mt], lw=1.5, ls='--', alpha=0.5)

    ax2.set_xticks(list(type_order.values()))
    type_labels_cn = {'Traditional': '传统模型', 'DL': '深度学习', 'Hybrid': '混合模型', 'Ensemble': '集成模型'}
    ax2.set_xticklabels([type_labels_cn[k] for k in type_order.keys()], fontsize=11)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [saved] {save_path}")


def plot_multi_trait(results, traits, save_path):
    """多性状: 分组柱状图 + 排名"""
    eval_models = list(results[traits[0]].keys())
    summ = {m: [] for m in eval_models}
    for t in traits:
        for m in eval_models:
            if m in results[t]:
                summ[m].append(results[t][m]['R2'])
    ranked = sorted([(m, np.mean(summ[m])) for m in eval_models if summ[m]],
                    key=lambda x: x[1], reverse=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 15))

    # 子图1: 各性状柱状图
    x = np.arange(len(traits))
    n_models = len(eval_models)
    w = 0.8 / n_models
    for i, m in enumerate(eval_models):
        rs = [results[t][m]['R2'] for t in traits if m in results[t]]
        offset = (i - n_models/2 + 0.5) * w
        ax1.bar(x + offset, rs, w, label=m, color=MODEL_COLORS.get(m, '#999'),
                alpha=0.88, edgecolor='white', linewidth=0.3)
    ax1.set_ylabel('R²', fontsize=12)
    ax1.set_title('小麦基因组预测 — 所有模型对比 (5折交叉验证)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([t[:25] for t in traits], rotation=45, ha='right', fontsize=9)
    ax1.legend(ncol=6, fontsize=6, loc='lower left')
    ax1.axhline(0, c='k', lw=0.5)
    ax1.grid(axis='y', alpha=0.2)

    # 子图2: 排名
    ns = [m for m, _ in ranked]
    vals = [np.mean(summ[m]) for m in ns]
    bar_colors = [MODEL_COLORS.get(m, '#999') for m in ns]
    bars = ax2.barh(ns, vals, color=bar_colors, alpha=0.88, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('平均 R²', fontsize=12)
    ax2.set_title('综合排名', fontsize=14, fontweight='bold')
    type_abbr = {'Traditional': '传', 'DL': '深', 'Hybrid': '混', 'Ensemble': '集'}
    for b, v, m in zip(bars, vals, ns):
        mtype = results[traits[0]][m].get('Type', 'DL')
        label = f'{v:.4f} [{type_abbr.get(mtype, mtype[0])}]'
        ax2.text(b.get_width() + 0.005, b.get_y() + b.get_height()/2,
                 label, va='center', fontsize=7, fontweight='bold')
    ax2.grid(axis='x', alpha=0.2)
    ax2.invert_yaxis()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [saved] {save_path}")


def generate_visualization(json_path=None):
    """主入口: 从 JSON 生成可视化图"""
    if json_path is None:
        json_path = OUTPUT_DIR / "ensemble_intermediate.json"
    else:
        json_path = Path(json_path)

    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        return

    results = json.load(open(json_path))
    traits = sorted(results.keys())
    ts = json_path.stem

    if len(traits) == 1:
        out = json_path.parent / f"wheat_models_comparison_{ts}.png"
        plot_single_trait(results, traits[0], out)
    else:
        out = json_path.parent / f"ensemble_comparison_{ts}.png"
        plot_multi_trait(results, traits, out)
    return out


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_visualization(path)
