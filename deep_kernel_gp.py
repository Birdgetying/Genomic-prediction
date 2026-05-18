"""
Deep Kernel Learning for Genomic Prediction
============================================
NN encoder → learned feature space → GP with RBF kernel.

GBLUP uses the fixed linear kernel K = σ²_g * XX^T/p.
DKL learns a nonlinear kernel K_θ(x_i, x_j) = σ²_f * exp(-½||e(x_i) - e(x_j)||² / l²)
where e is a learned neural encoder. Exact GP inference (n ~ 800, O(n³) ≈ 0.001s).

Imported by wheat_models_ensemble.py / rice_models_ensemble.py.
"""

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

JITTER = 1e-5  # 数值稳定性, 加在 Cholesky 分解前的对角线上


# ============================================================================
# Genomic Encoder — SNP → compact feature space
# ============================================================================

class GenomicEncoder(nn.Module):
    """Residual MLP: n_snps → 256 → 256 → 128 → latent_dim, LayerNorm output."""

    def __init__(self, n_snps, latent_dim=24, hidden1=256, hidden2=128, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.BatchNorm1d(n_snps),
            nn.Linear(n_snps, hidden1),
            nn.GELU(),
            nn.Dropout(dropout * 0.5))

        self.block1 = nn.Sequential(
            nn.BatchNorm1d(hidden1),
            nn.Linear(hidden1, hidden1),
            nn.GELU(),
            nn.Dropout(dropout))

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
        h = h + self.block1(h)
        h = self.down_proj(h)
        return self.layer_norm(h)


# ============================================================================
# Deep Kernel GP — exact GP + learnable encoder
# ============================================================================

class DeepKernelGP(nn.Module):
    """y = f(x) + ε, f ~ GP(0, K_θ), ε ~ N(0, σ²_n).

    K(x_i, x_j) = σ²_f * exp(-½||e(x_i) - e(x_j)||² / l²)

    Trained by minimizing negative log marginal likelihood.
    """

    def __init__(self, encoder, outputscale=1.0, lengthscale=1.0, noise=0.1):
        super().__init__()
        self.encoder = encoder

        self.log_outputscale = nn.Parameter(torch.tensor(np.log(outputscale)))
        self.log_lengthscale = nn.Parameter(torch.tensor(np.log(lengthscale)))
        self.log_noise = nn.Parameter(torch.tensor(np.log(noise)))

        self._train_features = None
        self._train_y = None
        self._K_inv = None  # (K + σ²_n I)^(-1), set by fit()

    def _build_kernel(self, X1, X2=None):
        lsq = torch.exp(self.log_lengthscale) ** 2
        outscale = torch.exp(self.log_outputscale)

        f1 = self.encoder(X1)
        if X2 is None:
            sq_dist = torch.cdist(f1, f1, p=2).pow(2)
        else:
            f2 = self.encoder(X2)
            sq_dist = torch.cdist(f1, f2, p=2).pow(2)
        return outscale * torch.exp(-0.5 * sq_dist / lsq)

    def _kernel_on_features(self, F1, F2):
        lsq = torch.exp(self.log_lengthscale) ** 2
        outscale = torch.exp(self.log_outputscale)
        sq_dist = torch.cdist(F1, F2, p=2).pow(2)
        return outscale * torch.exp(-0.5 * sq_dist / lsq)

    def marginal_nll(self, X, y):
        n = X.shape[0]
        K = self._build_kernel(X)
        noise = torch.exp(self.log_noise)
        Ky = K + noise * torch.eye(n, device=X.device)
        Ky = Ky + JITTER * torch.eye(n, device=X.device)

        L = torch.linalg.cholesky(Ky)
        alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)

        nll = 0.5 * (y * alpha).sum()
        nll += L.diag().log().sum()
        nll += 0.5 * n * np.log(2 * np.pi)
        return nll / n

    def fit(self, X, y):
        self.eval()
        with torch.no_grad():
            self._train_features = self.encoder(X).detach()
            self._train_y = y.detach()

            n = X.shape[0]
            K = self._build_kernel(X)
            noise = torch.exp(self.log_noise)
            Ky = K + noise * torch.eye(n, device=X.device)
            Ky = Ky + JITTER * torch.eye(n, device=X.device)
            L = torch.linalg.cholesky(Ky)
            self._K_inv = torch.cholesky_inverse(L)

    def predict(self, X_test):
        if self._K_inv is None:
            raise RuntimeError("call fit() before predict()")

        with torch.no_grad():
            f_test = self.encoder(X_test)
            K_star = self._kernel_on_features(f_test, self._train_features)
            alpha = self._K_inv @ self._train_y.unsqueeze(1)
            mean = (K_star @ alpha).squeeze()

            K_ss = torch.exp(self.log_outputscale) * torch.ones(
                X_test.shape[0], device=X_test.device)
            diag_terms = torch.diag(K_star @ self._K_inv @ K_star.T)
            variance = torch.clamp(K_ss - diag_terms, min=0)
        return mean, variance


# ============================================================================
# 训练循环
# ============================================================================

def train_dkgp(model, X_train, y_train, epochs=200, lr=5e-3, patience=15,
               batch_size=None, verbose=True):
    model = model.to(DEVICE)
    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=8)

    best_state, best_loss, wait = None, float('inf'), 0
    # Deterministic eval pass for monitoring — only every N epochs to avoid
    # computing a second Cholesky (O(n³)) every step.
    eval_every = 50

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

        current_loss = loss.item()
        scheduler.step(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"    DKL early stop @ epoch {epoch+1}, NLL={best_loss:.4f}")
                break

        if verbose and (epoch + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_nll = model.marginal_nll(X_t, y_t).item()
            print(f"    DKL epoch {epoch+1:3d}: NLL={eval_nll:.4f}, "
                  f"l={model.log_lengthscale.exp():.3f}, "
                  f"sf={model.log_outputscale.exp():.3f}, "
                  f"sn={model.log_noise.exp():.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model = model.cpu()
    return model
