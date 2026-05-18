"""
共享基因组神经网络模型 — PreFGN / LD-GCN

被 wheat_models_ensemble.py 和 rice_models_ensemble.py 导入使用。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# 由调用方设置 (wheat/rice 各自有 DEVICE)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 共享早停训练工具
# ============================================================================

def early_stop_restore(model, opt, scheduler_fn, loss_fn, forward_fn,
                       data_loader, epochs, patience, device):
    """通用早停训练循环。训练循环内聚到闭包中。

    scheduler_fn(opt) -> lr_scheduler, 可为 None。
    forward_fn(batch_data, model) -> loss。
    """
    sch = scheduler_fn(opt) if scheduler_fn else None
    best_state, best_loss, wait = None, float('inf'), 0

    for _ in range(epochs):
        model.train()
        for batch in data_loader:
            opt.zero_grad()
            loss = forward_fn(batch, model)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            count = 0
            for batch in data_loader:
                val_loss += forward_fn(batch, model).item() * len(batch[0])
                count += len(batch[0])
            val_loss = val_loss / max(count, 1)

        if sch is not None:
            sch.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================================
# FGNEncoder — 频域基因组编码器 (被 FGNv3 和 PreFGN 共用)
# ============================================================================

class FGNEncoder(nn.Module):
    """BN → FFT → FreqSE → FreqConv + SNP gate → TimeConv → concat output"""
    def __init__(self, n_snps, hidden=64, dropout=0.35):
        super().__init__()
        self.n_freq = n_snps // 2 + 1
        self.bn = nn.BatchNorm1d(n_snps)
        se_hidden = max(4, self.n_freq // 8)
        self.freq_se = nn.Sequential(
            nn.Linear(self.n_freq, se_hidden), nn.GELU(),
            nn.Linear(se_hidden, self.n_freq), nn.Sigmoid())
        self.freq_conv = nn.Sequential(
            nn.Conv1d(1, hidden, 7, padding=3), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.snp_gate = nn.Sequential(nn.Linear(n_snps, 1), nn.Sigmoid())
        self.time_conv = nn.Sequential(
            nn.Conv1d(1, hidden // 2, 21, padding=10), nn.BatchNorm1d(hidden // 2),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = hidden + hidden // 2

    def forward(self, x):
        x = self.bn(x)
        xc = torch.fft.rfft(x, dim=1)
        mag = xc.abs()
        se_w = self.freq_se(mag.mean(dim=0, keepdim=True))
        mag_w = mag * se_w
        fp = self.pool(self.freq_conv(mag_w.unsqueeze(1))).squeeze(-1)
        g = self.snp_gate(x)
        tp = self.pool(self.time_conv((x * g).unsqueeze(1))).squeeze(-1)
        return torch.cat([fp, tp], dim=1)


# ============================================================================
# PreFGN — 自监督预训练频域基因组网络
# ============================================================================

class PreFGN(nn.Module):
    """阶段1: 掩码标记重建 → 阶段2: 表型微调"""
    def __init__(self, n_snps, hidden=64, dropout=0.35):
        super().__init__()
        enc_out = hidden + hidden // 2
        self.encoder = FGNEncoder(n_snps, hidden, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(enc_out, hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, n_snps))
        self.head = nn.Sequential(
            nn.Linear(enc_out, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.head(self.encoder(x))

    def reconstruct(self, x_masked):
        return self.decoder(self.encoder(x_masked))

    def pretrainable_params(self):
        """返回预训练阶段的可训练参数 (encoder + decoder)"""
        return list(self.encoder.parameters()) + list(self.decoder.parameters())


def pretrain_prefgn(model, X, epochs=50, lr=1e-3, mask_ratio=0.2, patience=10):
    """阶段1: 掩码重建预训练。复用 early_stop_restore。"""
    model = model.to(DEVICE)
    X_t = torch.FloatTensor(X)
    dl = DataLoader(TensorDataset(X_t),
                    batch_size=min(128, len(X)), shuffle=True)

    opt = torch.optim.AdamW(model.pretrainable_params(), lr=lr)

    def forward_masked(batch, m):
        xb = batch[0].to(DEVICE)
        mask = torch.rand_like(xb) > mask_ratio
        x_masked = xb.clone()
        x_masked[~mask] = 0.0
        recon = m.reconstruct(x_masked)
        return F.mse_loss(recon[~mask], xb[~mask])

    def sched(opt_):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_, mode='min', factor=0.5, patience=5)

    return early_stop_restore(model, opt, sched, None, forward_masked,
                              dl, epochs, patience, DEVICE).cpu()


# ============================================================================
# LD-GCN — LD 图卷积网络 (稀疏邻接矩阵)
# ============================================================================

class GCNLayer(nn.Module):
    """稀疏图卷积: H' = A_sparse @ H @ W + residual"""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.do = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, h, adj):
        """h: (B, M, dim), adj: (M, M) sparse COO tensor"""
        B, M, D = h.shape
        h_2d = h.permute(1, 0, 2).reshape(M, B * D)
        h_agg_2d = torch.sparse.mm(adj, h_2d)
        h_agg = h_agg_2d.reshape(M, B, D).permute(1, 0, 2)
        return self.do(self.act(self.norm(self.linear(h_agg))))


class LDGCN(nn.Module):
    """LD 图卷积网络: LD 图结构作为归纳偏置"""

    def __init__(self, n_snps, hidden=64, dropout=0.35, n_layers=2, k_neighbors=15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(1, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.gcn_layers = nn.ModuleList(
            [GCNLayer(hidden, dropout * 0.5) for _ in range(n_layers)])
        self.pool_bn = nn.BatchNorm1d(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
        self.register_buffer('adj', torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0), (n_snps, n_snps)).coalesce())
        self._k_neighbors = k_neighbors

    @staticmethod
    def build_adjacency(X, k=15):
        """从训练基因型构建 LD 归一化稀疏邻接矩阵 (float32 COO)"""
        X_f = X.astype(np.float32)
        # 内联标准化计算相关: corr = (X_std.T @ X_std) / (N-1)
        X_c = X_f - X_f.mean(axis=0, keepdims=True)
        std = np.sqrt(np.maximum((X_c ** 2).sum(axis=0), 1e-12))
        X_s = X_c / std
        corr = (X_s.T @ X_s) / max(X_f.shape[0] - 1, 1)
        corr_abs = np.abs(corr)
        np.fill_diagonal(corr_abs, 0)

        rows, cols, vals = [], [], []
        for i in range(corr_abs.shape[0]):
            top = np.argpartition(corr_abs[i], -(k + 1))[-(k + 1):]
            for j in top:
                if corr_abs[i, j] > 0:
                    rows.append(i); cols.append(j)
                    vals.append(corr_abs[i, j])

        n = corr_abs.shape[0]
        idx = torch.tensor([rows, cols], dtype=torch.long)
        val = torch.tensor(vals, dtype=torch.float32)
        adj_raw = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()

        # 对称化: adj = max(A, A^T)
        adj_sym = adj_raw + adj_raw.transpose(0, 1)
        adj_sym = adj_sym.coalesce()
        # 去重 (max 变为 sum 后减半):
        adj_sym.values().div_(2.0)

        # 添加自环 + 对称归一化: D^{-1/2} (A+I) D^{-1/2}
        deg = torch.sparse.sum(adj_sym, dim=1).to_dense() + 1.0
        d_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(deg, min=1e-12))

        new_idx = adj_sym.indices().clone()
        new_val = adj_sym.values().clone()
        # 归一化: val_{ij} *= d_inv_sqrt[i] * d_inv_sqrt[j]
        new_val *= d_inv_sqrt[new_idx[0]] * d_inv_sqrt[new_idx[1]]

        # 添加自环 (eye normalized to d_inv_sqrt^2 中对角贡献)
        diag_idx = torch.arange(n).unsqueeze(0).repeat(2, 1)
        diag_val = d_inv_sqrt ** 2
        all_idx = torch.cat([new_idx, diag_idx], dim=1)
        all_val = torch.cat([new_val, diag_val], dim=0)

        return torch.sparse_coo_tensor(all_idx, all_val, (n, n)).coalesce()

    def forward(self, x):
        h = self.input_proj(x.unsqueeze(-1))
        for gcn in self.gcn_layers:
            h = gcn(h, self.adj) + h
        h = h.mean(dim=1)
        h = self.pool_bn(h.unsqueeze(-1)).squeeze(-1)
        return self.head(h)
