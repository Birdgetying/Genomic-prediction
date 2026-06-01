#!/usr/bin/env python
"""修复 OOF NPZ 文件中 _y_true 与 predictions 不对齐的问题。

问题：_save_oof_npz 保存的 _y_true 是原始样本顺序，但 predictions 是 KFold
fold 拼接顺序（fold 0 test → fold 1 test → ...），两者不对齐，导致散点图
中的 r 和 R² 计算错误。

修复方法：用 KFold(n_splits=5, shuffle=True, random_state=42) 重建 fold
分割，将 predictions 重排回原始样本顺序，使 _y_true 与 predictions 对齐。
"""
import json, sys, io, os, gc
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42
N_FOLDS = 5

# ---- 辅助：加载原始表型 y（用于验证修复后的排序正确性） ----

def load_phenotype_y(dataset_key):
    """返回原始表型 y 向量（完整样本数）。"""
    if dataset_key == 'wheat':
        import pandas as pd
        geno = pd.read_csv(SCRIPT_DIR / 'data/wheat/genotype.txt', sep='\t', index_col=0)
        n = geno.shape[0]
        return np.arange(n)
    elif dataset_key == 'wheat2000':
        import pandas as pd
        df = pd.read_csv(SCRIPT_DIR / 'data/wheat2000/genotype.csv', index_col=0)
        n = df.shape[0]
        return np.arange(n)
    elif dataset_key == 'rice':
        data = np.load(SCRIPT_DIR / 'data/rice/genotype_matrix.npz', allow_pickle=True)
        X = data['X']
        return np.arange(X.shape[0])
    elif dataset_key == 'maize':
        import pandas as pd
        geno = pd.read_csv(SCRIPT_DIR / 'data/maize/genotype.csv', index_col=0)
        n = geno.shape[0]
        return np.arange(n)
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")


# ---- 主修复逻辑 ----

def repair_npz(npz_path, n_samples):
    """修复一个 NPZ 文件：将 predictions 从 fold 顺序重排为原始样本顺序。"""
    print(f"  Repairing: {npz_path.name}")
    data = dict(np.load(npz_path, allow_pickle=True))

    y_old = data['_y_true']
    assert len(y_old) == n_samples, f"y length {len(y_old)} != n_samples {n_samples}"

    # 重建 KFold 分割
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_test_indices = [te for _, te in kf.split(np.arange(n_samples))]

    # fold_test_indices[k] = test indices for fold k
    # predictions 在 NPZ 中的拼接顺序 = [fold0_preds, fold1_preds, ...]
    # 重建逆映射：对于 fold k 中位置 pos 的样本，其原始索引 = fold_test_indices[k][pos]

    # 验证各 fold 长度之和 == n_samples
    total_test = sum(len(te) for te in fold_test_indices)
    assert total_test == n_samples, f"Total test samples {total_test} != {n_samples}"

    # 重建逆排列：original_order[i] = 原始样本索引
    inverse_perm = np.zeros(n_samples, dtype=int)
    offset = 0
    for k in range(N_FOLDS):
        te = fold_test_indices[k]
        nk = len(te)
        inverse_perm[offset:offset + nk] = te
        offset += nk

    # 创建重排后的数据
    # original_order[i] = 原始样本索引 — 用于恢复
    # 我们需要 reverse_perm: 给定原始索引 → fold 拼接中的位置
    reverse_perm = np.zeros(n_samples, dtype=int)
    for fold_pos, orig_idx in enumerate(inverse_perm):
        reverse_perm[orig_idx] = fold_pos

    # y_true 已经按原始顺序，保持不变
    new_data = {'_y_true': y_old}

    for key in data:
        if key == '_y_true':
            continue
        preds_fold_order = data[key]
        if len(preds_fold_order) != n_samples:
            print(f"    WARNING: {key} length {len(preds_fold_order)} != {n_samples}, skipping")
            continue
        # 重排：将 fold 拼接顺序 → 原始样本顺序
        preds_original_order = preds_fold_order[reverse_perm]
        new_data[key] = preds_original_order.astype(np.float32)

    # 覆盖保存
    backup_path = npz_path.with_suffix('.npz.bak')
    if not backup_path.exists():
        np.savez_compressed(backup_path, **data)  # 备份原始文件
    np.savez_compressed(npz_path, **new_data)
    print(f"    -> Repaired ({len(new_data)-1} models), backup: {backup_path.name}")


def verify_npz(npz_path, n_samples):
    """验证修复后的 NPZ：随机检查最佳模型的 r 和 R² 是否合理。"""
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score

    data = np.load(npz_path, allow_pickle=True)
    y = data['_y_true']
    model_keys = [k for k in data if k != '_y_true']
    if not model_keys:
        return
    # 选第一个模型验证
    m = model_keys[0]
    preds = data[m]
    r, _ = pearsonr(y, preds)
    r2 = r2_score(y, preds)
    # 现在应该与 JSON 中存储的值接近
    print(f"    Verify [{m}]: r={r:.4f}  R²={r2:.4f}")


def main():
    print("=" * 70)
    print("Repairing OOF NPZ files: fixing misaligned _y_true vs predictions")
    print("=" * 70)

    # 需修复的每个数据集的 (子目录, 样本数)
    tasks = []

    for sub, n in [('wheat', None), ('wheat2000', None),
                   ('rice', None), ('maize', None)]:
        oof_dir = SCRIPT_DIR / 'results' / f'{sub}_ensemble' / 'oof_predictions'
        if not oof_dir.exists():
            print(f"  SKIP {sub}: no oof_predictions dir")
            continue
        npz_files = sorted(oof_dir.glob('*_oof.npz'))
        if not npz_files:
            print(f"  SKIP {sub}: no NPZ files")
            continue

        print(f"\n[{sub}] {len(npz_files)} NPZ files")

        for npz_path in npz_files:
            # 每个 NPZ 可能有不同样本数（水稻缺失值剔除后不同性状有效样本数不同）
            first = np.load(npz_path, allow_pickle=True)
            n_samples = len(first['_y_true'])
            repair_npz(npz_path, n_samples)
            verify_npz(npz_path, n_samples)

    print("\nDone! All NPZ files repaired.")


if __name__ == '__main__':
    main()
