"""
单倍型启发的标记打分 — 替代纯 GWAS p-value 的标记筛选

核心思路 (源自 haplotype_phenotype_analysis.py 的 HaplotypeScorer):
  不是孤立看待每个 SNP, 而是综合:
    1. 变异功能严重度 (SNP < INDEL < SV)
    2. 稀有度权重 (rare variants 可能效应更大)
    3. 标记间 LD 去冗余 (避免连锁标记重复计数)

与 GWAS 的关键区别:
  - GWAS: 边际线性回归 p-value → 偏向加性效应、常见变异
  - HaploScore: 功能+稀有度+效应联合 → 保留更多信息多样性

评分公式:
  score(pos) = func_weight × rarity_weight × (1 + |effect_est|)
  其中 effect_est 来自快速单变量扫描 (与 GWAS 同源但只取效应大小)

LD 剪枝: 滑动窗口内保留得分最高的标记, 窗口大小由 r² 阈值控制

被 wheat_models_ensemble.py / rice_models_ensemble.py 导入使用。
"""

import numpy as np


VARIANT_TYPE_WEIGHTS = {
    0: 1.0,   # SNP
    1: 3.5,   # INDEL
    2: 3.0,   # SV
}

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
    """计算每个标记的单倍型启发得分

    Args:
        X: (n, p) 基因型矩阵
        y: (n,) 表型向量
        variant_types: (p,) 每个标记的类型 — 0=SNP, 1=INDEL, 2=SV。None 则全为 SNP
        maf: (p,) minor allele frequency。None 则从 X 计算

    Returns:
        scores: (p,) 每个标记的得分 (越高越 "好")
        components: dict with 'func', 'rarity', 'effect' 各组分
    """
    p = X.shape[1]

    if variant_types is not None:
        func_w = np.array([VARIANT_TYPE_WEIGHTS.get(int(vt), 1.0) for vt in variant_types])
    else:
        func_w = np.ones(p)

    if maf is None:
        af = X.mean(axis=0) / 2.0
        maf = np.minimum(af, 1.0 - af)
    rarity_w = np.maximum(-np.log10(np.maximum(maf, MIN_MAF)), 1.0)

    effects = _compute_univariate_effects(X, y)

    scores = func_w * rarity_w * (1.0 + effects)

    return scores, {'func': func_w, 'rarity': rarity_w, 'effect': effects}


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
                     r2_thresh=DEFAULT_R2_THRESH):
    """单倍型启发标记筛选: 打分 → 预选候选集 → LD 剪枝 → 取 top k"""
    af = X.mean(axis=0) / 2.0
    maf = np.minimum(af, 1.0 - af)

    scores, _ = compute_haplotype_scores(X, y, variant_types, maf)

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
    return np.sort(top_k)


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
