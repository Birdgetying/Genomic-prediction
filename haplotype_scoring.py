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
from scipy.stats import pearsonr


# 变异类型功能权重 (与 HaplotypeScorer.FUNCTIONAL_WEIGHTS 对齐)
VARIANT_TYPE_WEIGHTS = {
    0: 1.0,   # SNP — 基准权重
    1: 3.5,   # INDEL — 可能破坏阅读框
    2: 3.0,   # SV — 结构变异, 影响大
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

    # 1. 功能权重
    if variant_types is not None:
        func_w = np.array([VARIANT_TYPE_WEIGHTS.get(int(vt), 1.0) for vt in variant_types])
    else:
        func_w = np.ones(p)

    # 2. 稀有度权重: -log10(maf) for rare variants, 下限 1.0 for common
    if maf is None:
        af = X.mean(axis=0) / 2.0
        maf = np.minimum(af, 1.0 - af)
    rarity_w = np.maximum(-np.log10(np.maximum(maf, MIN_MAF)), 1.0)

    # 3. 效应量 (绝对值, 来自快速单变量扫描)
    effects = _compute_univariate_effects(X, y)

    # 综合得分
    scores = func_w * rarity_w * (1.0 + effects)

    return scores, {'func': func_w, 'rarity': rarity_w, 'effect': effects}


def ld_prune_markers(X, scores, window=DEFAULT_WINDOW, r2_thresh=DEFAULT_R2_THRESH):
    """LD 剪枝: 滑动窗口内保留得分最高的标记

    遍历标记, 在窗口内检查与已保留标记的 r²:
    - 若 r² > r2_thresh → 跳过 (该 LD block 已被更高分标记代表)
    - 否则 → 保留

    Args:
        X: (n, p) 基因型矩阵
        scores: (p,) 每个标记的得分
        window: 滑动窗口大小 (标记数)
        r2_thresh: 剔除阈值

    Returns:
        keep_idx: 保留的标记索引 (按得分排序)
    """
    p = X.shape[1]
    # 按得分降序排列 → 高分标记优先保留
    order = np.argsort(-scores)
    kept = []
    # 跟踪每个已保留标记的位置, 用于快速窗口查询
    kept_positions = []

    for idx in order:
        pos = idx
        # 检查与窗口内已保留标记的 LD
        redundant = False
        for kp in kept_positions:
            if abs(pos - kp) <= window:
                r = pearsonr(X[:, pos], X[:, kp])[0]
                if r ** 2 > r2_thresh:
                    redundant = True
                    break
        if not redundant:
            kept.append(idx)
            kept_positions.append(pos)

    return np.array(kept, dtype=int)


def haplotype_select(X, y, k, variant_types=None, window=DEFAULT_WINDOW,
                     r2_thresh=DEFAULT_R2_THRESH):
    """单倍型启发标记筛选: 打分 → LD 剪枝 → 取 top k

    Args:
        X: (n, p) 基因型矩阵
        y: (n,) 表型向量
        k: 保留的标记数
        variant_types: (p,) 标记类型, None 则全为 SNP
        window: LD 剪枝窗口
        r2_thresh: LD 剪枝 r² 阈值

    Returns:
        selected: (k,) 选中标记的列索引
    """
    af = X.mean(axis=0) / 2.0
    maf = np.minimum(af, 1.0 - af)

    scores, _ = compute_haplotype_scores(X, y, variant_types, maf)
    pruned = ld_prune_markers(X, scores, window, r2_thresh)

    # 剪枝后若不足 k 个, 从剩余标记中补足
    if len(pruned) < k:
        remaining = np.setdiff1d(np.argsort(-scores), pruned)
        need = k - len(pruned)
        pruned = np.concatenate([pruned, remaining[:need]])

    # 按得分取 top k
    top_k = pruned[np.argsort(-scores[pruned])[:k]]
    return np.sort(top_k)


def hybrid_select(X, y, k, variant_types=None, gwas_frac=0.6,
                  window=DEFAULT_WINDOW, r2_thresh=DEFAULT_R2_THRESH):
    """混合筛选: GWAS top (gwas_frac * k) + HaploScore top ((1-gwas_frac) * k)

    GWAS 捕获线性加性信号, HaploScore 捕获功能/稀有度信号。
    合并后去重, 不足 k 则从各自剩余中补足。

    Args:
        X: (n, p) 基因型矩阵
        y: (n,) 表型向量
        k: 目标标记数
        variant_types: (p,) 标记类型
        gwas_frac: GWAS 标记占比 (0.0-1.0)

    Returns:
        selected: (k,) 选中标记的列索引
    """
    n_gwas = int(k * gwas_frac)
    n_hap = k - n_gwas

    # GWAS 部分 (标准 p-value 筛选)
    from scipy.stats import pearsonr as pr
    pvals = np.ones(X.shape[1])
    y_c = y - y.mean()
    for j in range(X.shape[1]):
        if X[:, j].std() > 0:
            r, pv = pr(X[:, j], y_c)
            pvals[j] = pv
    gwas_top = np.argsort(pvals)[:n_gwas * 2]  # 取 2× 做缓冲

    # HaploScore 部分
    hap_top = haplotype_select(X, y, n_hap * 2, variant_types, window, r2_thresh)

    # 合并: 先取 GWAS top n_gwas, 再取 HaploScore 中不重复的
    combined = list(gwas_top[:n_gwas])
    for idx in hap_top:
        if idx not in combined and len(combined) < k:
            combined.append(idx)

    # 不足则从 GWAS 剩余补
    if len(combined) < k:
        for idx in gwas_top:
            if idx not in combined and len(combined) < k:
                combined.append(idx)

    return np.sort(np.array(combined[:k], dtype=int))
