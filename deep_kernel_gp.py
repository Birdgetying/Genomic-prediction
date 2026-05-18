"""
Deep Kernel Learning for Genomic Prediction
============================================
神经网络学习基因型 → 特征空间, Gaussian Process 在特征空间上建模表型。

核心思想:
  - 传统 GBLUP: y ~ N(0, σ²_g * XX^T/p)  — 线性核, 仅加性效应
  - Deep Kernel GP: y ~ N(0, K_θ) where K_θ = k(f_θ(x_i), f_θ(x_j))
    f_θ 是 NN 编码器, k 是 RBF 核 — 学习非线性特征空间中的相似度

推理: 精确 GP (n ~ 800, O(n³) 完全可承受)

被 wheat_models_ensemble.py / rice_models_ensemble.py 导入使用。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# 由调用方设置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Genomic Encoder — 将 SNP 编码到紧凑特征空间
# ============================================================================

class GenomicEncoder(nn.Module):
    """轻量 SNP 编码器: 输入 (batch, n_snps) → 输出 (batch, latent_dim)

    设计原则:
    - 3 层残差 MLP, 每层维度递减: n_snps → 256 → 128 → latent_dim
    - BatchNorm + GELU + Dropout 防止过拟合
    - 残差连接帮助梯度在 n 小时也能训练
    - 最终输出 layer_norm 保证特征尺度稳定 (GP 核对此敏感)
    """
    def __init__(self, n_snps, latent_dim=24, hidden1=256, hidden2=128, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.BatchNorm1d(n_snps),
            nn.Linear(n_snps, hidden1),
            nn.GELU(),
            nn.Dropout(dropout * 0.5))

        # 残差块1: hidden1 → hidden1 (维数不变, 安全残差)
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(hidden1),
            nn.Linear(hidden1, hidden1),
            nn.GELU(),
            nn.Dropout(dropout))

        # 维度压缩: hidden1 → hidden2 → latent_dim
        self.down_proj = nn.Sequential(
            nn.BatchNorm1d(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.BatchNorm1d(hidden2),
            nn.Linear(hidden2, latent_dim))

        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        h = self.input_proj(x)
        h = h + self.block1(h)  # 残差连接
        h = self.down_proj(h)
        return self.layer_norm(h)


# ============================================================================
# Deep Kernel GP — 精确 GP + 可学习编码器
# ============================================================================

class DeepKernelGP(nn.Module):
    """精确 GP 推理 + 深度核

    模型: y = f(x) + ε,  f ~ GP(0, K_θ),  ε ~ N(0, σ²_n)

    核: K(x_i, x_j) = σ²_f * exp(-0.5 * ||e(x_i) - e(x_j)||² / l²)
        其中 e 是 GenomicEncoder

    训练: 最小化负对数边缘似然 (NLL)
          NLL = ½ y^T (K + σ²_n I)^(-1) y + ½ log|K + σ²_n I| + (n/2)·log(2π)
    """

    def __init__(self, encoder, outputscale=1.0, lengthscale=1.0, noise=0.1):
        super().__init__()
        self.encoder = encoder

        # GP 超参数 (log 空间, 保证正值)
        self.log_outputscale = nn.Parameter(torch.tensor(np.log(outputscale)))
        self.log_lengthscale = nn.Parameter(torch.tensor(np.log(lengthscale)))
        self.log_noise = nn.Parameter(torch.tensor(np.log(noise)))

        self._train_features = None
        self._train_y = None
        self._K_inv = None   # (K + σ²_n I)^(-1)

    def _build_kernel(self, X1, X2=None):
        """RBF 核: 在编码器特征空间上计算 (输入=原始基因型)

        K(x_i, x_j) = σ²_f * exp(-½||f(x_i) - f(x_j)||² / l²)
        """
        lsq = torch.exp(self.log_lengthscale) ** 2
        outscale = torch.exp(self.log_outputscale)

        f1 = self.encoder(X1)
        if X2 is None:
            sq_dist = torch.cdist(f1, f1, p=2).pow(2)
            K = outscale * torch.exp(-0.5 * sq_dist / lsq)
            return K
        else:
            f2 = self.encoder(X2)
            sq_dist = torch.cdist(f1, f2, p=2).pow(2)
            K = outscale * torch.exp(-0.5 * sq_dist / lsq)
            return K

    def _kernel_on_features(self, F1, F2):
        """RBF 核: 直接在已编码特征上计算 (不调用 encoder)"""
        lsq = torch.exp(self.log_lengthscale) ** 2
        outscale = torch.exp(self.log_outputscale)
        sq_dist = torch.cdist(F1, F2, p=2).pow(2)
        return outscale * torch.exp(-0.5 * sq_dist / lsq)

    def marginal_nll(self, X, y):
        """负对数边缘似然 — 训练目标"""
        n = X.shape[0]
        K = self._build_kernel(X)
        noise = torch.exp(self.log_noise)
        Ky = K + noise * torch.eye(n, device=X.device)

        # Cholesky: Ky = L @ L^T
        L = torch.linalg.cholesky(Ky)

        # α = Ky^{-1} y = L^{-T} @ L^{-1} @ y
        alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)

        # NLL = ½ y^T α + sum(log(diag(L))) + ½ n log(2π)
        nll = 0.5 * (y * alpha).sum()
        nll += L.diag().log().sum()
        nll += 0.5 * n * np.log(2 * np.pi)
        return nll / n  # 归一化到 per-sample

    def fit(self, X, y):
        """缓存训练特征以加速预测 (在训练循环外调用)"""
        self.eval()
        with torch.no_grad():
            self._train_features = self.encoder(X).detach()
            self._train_y = y.detach()

            n = X.shape[0]
            K = self._build_kernel(X)
            noise = torch.exp(self.log_noise)
            Ky = K + noise * torch.eye(n, device=X.device)
            L = torch.linalg.cholesky(Ky)
            self._K_inv = torch.cholesky_inverse(L)

    def predict(self, X_test):
        """GP 后验预测

        Returns:
            mean: 预测均值 (n_test,)
            variance: 预测方差 (n_test,)
        """
        if self._K_inv is None:
            raise RuntimeError("call fit() before predict()")

        with torch.no_grad():
            f_test = self.encoder(X_test)
            K_star = self._kernel_on_features(f_test, self._train_features)
            # μ_* = K(x_test, X_train) @ K^{-1} @ y
            alpha = self._K_inv @ self._train_y.unsqueeze(1)
            mean = (K_star @ alpha).squeeze()
            # σ²_* = k(x_*, x_*) - diag(K_* @ K^{-1} @ K_*^T)
            K_ss = torch.exp(self.log_outputscale) * torch.ones(
                X_test.shape[0], device=X_test.device)
            diag_terms = torch.diag(K_star @ self._K_inv @ K_star.T)
            variance = K_ss - diag_terms
            variance = torch.clamp(variance, min=0)
        return mean, variance


# ============================================================================
# 训练循环
# ============================================================================

def train_dkgp(model, X_train, y_train, epochs=200, lr=5e-3, patience=15,
               batch_size=None, verbose=True):
    """训练 Deep Kernel GP

    Args:
        model: DeepKernelGP 实例
        X_train: (n, p) numpy array
        y_train: (n,) numpy array
        epochs: 最大轮数
        lr: 学习率 (DKL 通常需要比 NN 更大的 lr)
        patience: 早停耐心值
        batch_size: 批量大小, None 表示全量 (推荐, GP 核需要完整矩阵)
        verbose: 是否打印训练信息

    返回训练好的 model (cpu)
    """
    model = model.to(DEVICE)
    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=8)

    best_state, best_loss, wait = None, float('inf'), 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()

        if batch_size and batch_size < len(X_train):
            idx = torch.randperm(len(X_train))[:batch_size]
            loss = model.marginal_nll(X_t[idx], y_t[idx])
        else:
            loss = model.marginal_nll(X_t, y_t)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        with torch.no_grad():
            total_loss = model.marginal_nll(X_t, y_t).item()
        scheduler.step(total_loss)

        if total_loss < best_loss:
            best_loss = total_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"    DKL early stop @ epoch {epoch+1}, NLL={best_loss:.4f}")
                break

        if verbose and (epoch + 1) % 50 == 0:
            ls = model.log_lengthscale.exp().item()
            os = model.log_outputscale.exp().item()
            ns = model.log_noise.exp().item()
            print(f"    DKL epoch {epoch+1:3d}: NLL={total_loss:.4f}, "
                  f"l={ls:.3f}, σ²_f={os:.3f}, σ²_n={ns:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model = model.cpu()
    return model


def cross_val_dkgp(X, y, cv_splits, epochs=200, lr=5e-3, patience=15,
                   latent_dim=24, encoder_kwargs=None, verbose=True):
    """Deep Kernel GP 交叉验证

    Returns:
        preds: (n,) 所有样本的预测值 (按原始顺序)
        targets: (n,) 目标值
        final_model: 在全量数据上训练的模型
    """
    n = len(y)
    preds = np.zeros(n)
    targets = np.zeros(n)

    for fi, (tr_idx, te_idx) in enumerate(cv_splits):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
        Xte_s = scaler.transform(Xte).astype(np.float32)

        # 创建模型
        ek = encoder_kwargs or {}
        encoder = GenomicEncoder(n_snps=Xtr_s.shape[1], latent_dim=latent_dim, **ek)
        model = DeepKernelGP(encoder)

        model = train_dkgp(model, Xtr_s, ytr, epochs=epochs, lr=lr,
                           patience=patience, verbose=verbose)
        model.fit(torch.FloatTensor(Xtr_s), torch.FloatTensor(ytr))
        mean, _ = model.predict(torch.FloatTensor(Xte_s))
        preds[te_idx] = mean.cpu().numpy()
        targets[te_idx] = yte

        if verbose:
            fold_r2 = 1 - np.sum((yte - preds[te_idx]) ** 2) / np.sum((yte - yte.mean()) ** 2)
            print(f"    DKL fold {fi+1}: R²={fold_r2:.4f}")

    # 在全量数据上训练最终模型
    scaler_full = StandardScaler()
    X_s = scaler_full.fit_transform(X).astype(np.float32)
    ek = encoder_kwargs or {}
    encoder_final = GenomicEncoder(n_snps=X_s.shape[1], latent_dim=latent_dim, **ek)
    final_model = DeepKernelGP(encoder_final)
    final_model = train_dkgp(final_model, X_s, y, epochs=epochs, lr=lr,
                             patience=patience, verbose=verbose)
    final_model.fit(torch.FloatTensor(X_s), torch.FloatTensor(y))

    return preds, targets, final_model
