"""
单倍型启发的标记打分 — 替代纯 GWAS p-value 的标记筛选

核心思路 (源自 haplotype_phenotype_analysis.py 的 HaplotypeScorer):
  不是孤立看待每个 SNP, 而是综合:
    1. 稀有度权重 (rare variants 可能效应更大)
    2. 标记间 LD 去冗余 (避免连锁标记重复计数)

与 GWAS 的关键区别:
  - GWAS: 边际线性回归 p-value → 偏向加性效应、常见变异
  - HaploScore: 稀有度+效应联合 → 保留更多信息多样性

评分公式 (变异类型不参与打分):
  score(pos) = rarity_weight × (1 + |effect_est|)
  其中 effect_est 来自快速单变量扫描 (与 GWAS 同源但只取效应大小)

变异类型的角色 — 后验富集分析:
  打分后才看选中位点的变异类型分布, 与全基因组背景对比, 计算富集倍数。
  作用: "为什么这些位点被选中?" → 增强可解释性, 而非影响筛选结果。

LD 剪枝: 滑动窗口内保留得分最高的标记, 窗口大小由 r² 阈值控制

被 wheat_models_ensemble.py / rice_models_ensemble.py 导入使用。
"""

import numpy as np

# 变异类型标签 (用于后验富集分析, 不参与打分)
VARIANT_TYPE_NAMES = {0: 'SNP', 1: 'INDEL', 2: 'SV'}

DEFAULT_WINDOW = 50      # LD 剪枝滑动窗口 (标记数)
DEFAULT_R2_THRESH = 0.6  # LD 剪枝 r² 阈值
MIN_MAF = 1e-4           # 防止 log(0)


def _compute_univariate_effects(X, y):
    """快速单变量效应量扫描 (仅效应大小, 不需要 p-value)

    对于二值基因型 (0/1/2), 效应量 = cov(x_j, y) / var(x_j)
    等价于单变量线性回归的斜率。

    Returns:
        effects: (p,) array of |β| per marker
    """
    y_c = y - y.mean()
    X_c = X - X.mean(axis=0, keepdims=True)
    var_x = X_c.var(axis=0)
    # 避免除零 (monomorphic markers)
    var_x = np.maximum(var_x, 1e-8)
    cov_xy = X_c.T @ y_c / len(y_c)
    effects = np.abs(cov_xy / var_x)
    return effects


def compute_haplotype_scores(X, y, variant_types=None, maf=None):
    """计算每个标记的单倍型启发得分 (变异类型不参与打分)

    公式: score = rarity_weight × (1 + |effect_est|)

    Args:
        X: (n, p) 基因型矩阵
        y: (n,) 表型向量
        variant_types: (p,) 每个标记的类型 — 0=SNP, 1=INDEL, 2=SV。
                       不参与打分, 仅传递给 components 供后验富集分析。
        maf: (p,) minor allele frequency。None 则从 X 计算

    Returns:
        scores: (p,) 每个标记的得分 (越高越 "好")
        components: dict with 'rarity', 'effect', 'variant_types' 各组分
    """
    p = X.shape[1]

    if maf is None:
        af = X.mean(axis=0) / 2.0
        maf = np.minimum(af, 1.0 - af)
    rarity_w = np.maximum(-np.log10(np.maximum(maf, MIN_MAF)), 1.0)

    effects = _compute_univariate_effects(X, y)

    # 变异类型不参与打分
    scores = rarity_w * (1.0 + effects)

    return scores, {'rarity': rarity_w, 'effect': effects,
                    'variant_types': variant_types}


def variant_type_enrichment(selected_indices, variant_types, top_k=None):
    """后验富集分析: 对比选中位点 vs 全基因组的变异类型分布

    Args:
        selected_indices: (k,) 被选中的标记索引 (在 variant_types 中的位置)
        variant_types: (p,) 全基因组变异类型数组 — 0=SNP, 1=INDEL, 2=SV
        top_k: 仅分析前 top_k 个 (默认全部 selected_indices)

    Returns:
        dict: {
            'background': {type_name: (count, fraction)},
            'selected':   {type_name: (count, fraction)},
            'enrichment': {type_name: fold_enrichment},
            'summary': 一行中文总结
        }
    """
    if variant_types is None:
        return {'summary': '(无变异类型信息, 跳过富集分析)',
                'background': {}, 'selected': {}, 'enrichment': {}}

    if top_k is not None and top_k < len(selected_indices):
        idx = selected_indices[:top_k]
    else:
        idx = selected_indices

    vt = np.asarray(variant_types, dtype=int)

    # 全基因组背景
    total_bg = len(vt)
    bg_counts = {}
    for tid, tname in VARIANT_TYPE_NAMES.items():
        bg_counts[tname] = int(np.sum(vt == tid))

    # 选中位点
    vt_sel = vt[idx]
    total_sel = len(vt_sel)
    sel_counts = {}
    for tid, tname in VARIANT_TYPE_NAMES.items():
        sel_counts[tname] = int(np.sum(vt_sel == tid))

    # 富集倍数 = (选中比例 / 背景比例)
    enrichment = {}
    summary_parts = []
    for tname in VARIANT_TYPE_NAMES.values():
        bg_frac = bg_counts[tname] / total_bg if total_bg > 0 else 0
        sel_frac = sel_counts[tname] / total_sel if total_sel > 0 else 0
        fold = sel_frac / bg_frac if bg_frac > 0 else float('inf')
        enrichment[tname] = round(fold, 2)

        if sel_counts[tname] > 0:
            summary_parts.append(
                f"{tname}: {sel_counts[tname]}/{total_sel} ({sel_frac:.1%}) "
                f"vs 全基因组 {bg_counts[tname]}/{total_bg} ({bg_frac:.1%}), "
                f"富集 {fold:.1f}×"
            )

    summary = (
        f"选中 {total_sel} 个位点的变异类型分布:\n  " +
        "\n  ".join(summary_parts) if summary_parts else
        "(仅 SNP)"
    )

    return {
        'background': {t: (bg_counts[t], bg_counts[t] / total_bg)
                       for t in VARIANT_TYPE_NAMES.values()},
        'selected': {t: (sel_counts[t], sel_counts[t] / total_sel)
                     for t in VARIANT_TYPE_NAMES.values()},
        'enrichment': enrichment,
        'summary': summary,
    }


def ld_prune_markers(X, scores, window=DEFAULT_WINDOW, r2_thresh=DEFAULT_R2_THRESH):
    """LD 剪枝: 滑动窗口内保留得分最高的标记"""
    n, p = X.shape
    # Pre-center + normalize columns for fast r = dot(x_j, x_k)
    X_c = X - X.mean(axis=0, keepdims=True)
    X_n = X_c / (np.linalg.norm(X_c, axis=0, keepdims=True) + 1e-12)

    order = np.argsort(-scores)
    kept = []
    kept_positions = []

    for idx in order:
        redundant = False
        xj = X_n[:, idx]
        for kp in kept_positions:
            if abs(idx - kp) <= window:
                r = np.dot(xj, X_n[:, kp])
                if r * r > r2_thresh:
                    redundant = True
                    break
        if not redundant:
            kept.append(idx)
            kept_positions.append(idx)

    return np.array(kept, dtype=int)


def haplotype_select(X, y, k, variant_types=None, window=DEFAULT_WINDOW,
                     r2_thresh=DEFAULT_R2_THRESH, show_enrichment=False):
    """单倍型启发标记筛选: 打分 → 预选候选集 → LD 剪枝 → 取 top k

    变异类型不参与打分, 仅可选地在筛选后打印富集分析。
    """
    af = X.mean(axis=0) / 2.0
    maf = np.minimum(af, 1.0 - af)

    scores, components = compute_haplotype_scores(X, y, variant_types, maf)

    # Pre-select top candidates to keep LD pruning feasible (original p can be 100K+)
    n_candidates = min(3 * k, X.shape[1])
    cand_idx = np.argsort(-scores)[:n_candidates]
    X_cand = X[:, cand_idx]
    scores_cand = scores[cand_idx]

    pruned_sub = ld_prune_markers(X_cand, scores_cand, window, r2_thresh)
    pruned = cand_idx[pruned_sub]

    if len(pruned) < k:
        remaining = np.setdiff1d(np.argsort(-scores), pruned)
        need = k - len(pruned)
        pruned = np.concatenate([pruned, remaining[:need]])

    top_k = pruned[np.argsort(-scores[pruned])[:k]]
    result = np.sort(top_k)

    # 后验富集分析 (仅在提供了 variant_types 且有非 SNP 类型时打印)
    if show_enrichment and variant_types is not None:
        enrich = variant_type_enrichment(result, variant_types)
        has_non_snp = any(vt not in (0,) for vt in np.unique(variant_types))
        if has_non_snp:
            print(f"  [富集分析] {enrich['summary']}")

    return result


def hybrid_select(X, y, k, variant_types=None, gwas_frac=0.6,
                  window=DEFAULT_WINDOW, r2_thresh=DEFAULT_R2_THRESH):
    """GWAS top (gwas_frac*k) + HaploScore ((1-gwas_frac)*k), deduped."""
    n_gwas = int(k * gwas_frac)
    n_hap = k - n_gwas

    y_c = y - y.mean()
    X_c = X - X.mean(axis=0)
    num = np.dot(y_c, X_c)
    denom = np.std(y_c) * len(y) * np.sqrt(np.sum(X_c ** 2, axis=0) + 1e-12)
    gwas_top = np.argsort(np.abs(num / denom))[-n_gwas * 2:]

    hap_top = haplotype_select(X, y, n_hap * 2, variant_types, window, r2_thresh)

    combined = list(gwas_top[-n_gwas:])
    combined_set = set(combined)
    for idx in hap_top:
        if idx not in combined_set and len(combined) < k:
            combined.append(idx)
            combined_set.add(idx)

    if len(combined) < k:
        for idx in gwas_top:
            if idx not in combined_set and len(combined) < k:
                combined.append(idx)
                combined_set.add(idx)

    return np.sort(np.array(combined[:k], dtype=int))
