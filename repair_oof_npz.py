#!/usr/bin/env python
"""修复 OOF NPZ 文件中 _y_true 与 predictions 不对齐的问题。

问题：_save_oof_npz 保存的 _y_true 是原始样本顺序，但 predictions 是 KFold
fold 拼接顺序（fold 0 test → fold 1 test → ...），两者不对齐，导致散点图
中的 r 和 R² 计算错误。

修复方法：用 KFold(n_splits=5, shuffle=True, random_state=42) 重建 fold
分割，将 predictions 重排回原始样本顺序，使 _y_true 与 predictions 对齐。
"""
import sys, io
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42
N_FOLDS = 5


def repair_npz(npz_path, data=None, n_samples=None):
    """修复一个 NPZ 文件：将 predictions 从 fold 顺序重排为原始样本顺序。

    参数:
        npz_path: NPZ 文件路径
        data: 预加载的 NPZ 数据（避免重复 I/O），为 None 则自行加载
        n_samples: 样本数（从 data 推断或调用方提供）
    """
    if data is None:
        data = dict(np.load(npz_path, allow_pickle=True))
    if n_samples is None:
        n_samples = len(data['_y_true'])

    print(f"  Repairing: {npz_path.name}")
    y_old = data['_y_true']
    assert len(y_old) == n_samples, f"y length {len(y_old)} != n_samples {n_samples}"

    # 重建 KFold 分割
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_test_indices = [te for _, te in kf.split(np.arange(n_samples))]

    assert sum(len(te) for te in fold_test_indices) == n_samples

    # 构建 fold_pos → original_idx 映射，取 argsort 即得逆映射
    inverse_perm = np.concatenate(fold_test_indices)
    reverse_perm = np.argsort(inverse_perm)

    # 重排 predictions → 原始样本顺序
    new_data = {'_y_true': y_old}
    for key in data:
        if key == '_y_true':
            continue
        preds_fold_order = data[key]
        if len(preds_fold_order) != n_samples:
            print(f"    WARNING: {key} length {len(preds_fold_order)} != {n_samples}, skipping")
            continue
        new_data[key] = preds_fold_order[reverse_perm].astype(np.float32)

    # 备份（仅首次）
    backup_path = npz_path.with_suffix('.npz.bak')
    if not backup_path.exists():
        np.savez_compressed(backup_path, **data)
    np.savez_compressed(npz_path, **new_data)
    print(f"    -> Repaired ({len(new_data)-1} models), backup: {backup_path.name}")
    return new_data


def verify_npz(data):
    """验证修复后的 NPZ 数据：检查第一个模型的 r 和 R² 是否合理。"""
    y = data['_y_true']
    model_keys = [k for k in data if k != '_y_true']
    if not model_keys:
        return
    m = model_keys[0]
    r, _ = pearsonr(y, data[m])
    r2 = r2_score(y, data[m])
    print(f"    Verify [{m}]: r={r:.4f}  R²={r2:.4f}")


def main():
    print("=" * 70)
    print("Repairing OOF NPZ files: fixing misaligned _y_true vs predictions")
    print("=" * 70)

    for sub in ('wheat', 'wheat2000', 'rice', 'maize'):
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
            data = dict(np.load(npz_path, allow_pickle=True))
            new_data = repair_npz(npz_path, data=data, n_samples=len(data['_y_true']))
            verify_npz(new_data)

    print("\nDone! All NPZ files repaired.")


if __name__ == '__main__':
    main()
