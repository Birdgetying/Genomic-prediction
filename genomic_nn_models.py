"""
共享基因组神经网络模型 — PreFGN / FGNEncoder

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
    """BN → FFT → FreqSE → FreqConv + SNP gate → TimeConv → concat output

    marker_types: optional (n_snps,) int tensor [0=SNP, 1=INDEL, 2=SV].
                  Adds a learnable per-type bias before BN.
    """
    def __init__(self, n_snps, hidden=64, dropout=0.35, marker_types=None):
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

        # 变异类型编码 (SNP/INDEL/SV), None 表示不使用
        if marker_types is not None:
            n_types = int(max(marker_types) + 1) if hasattr(marker_types, '__len__') else 3
            self.type_embed = nn.Parameter(torch.zeros(n_types))
            self.register_buffer('_marker_type_idx',
                                 torch.as_tensor(marker_types, dtype=torch.long))
        else:
            self.type_embed = None
            self._marker_type_idx = None

    def set_marker_types(self, marker_types):
        """更新变异类型索引 (per-fold GWAS 筛选后调用)."""
        if self.type_embed is None:
            raise RuntimeError("set_marker_types() called but type_embed is None — "
                               "construct with marker_types first")
        self._marker_type_idx = torch.as_tensor(marker_types, dtype=torch.long)

    def forward(self, x):
        # 注入变异类型偏置
        if self.type_embed is not None and self._marker_type_idx is not None:
            x = x + self.type_embed[self._marker_type_idx]
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
    def __init__(self, n_snps, hidden=64, dropout=0.35, marker_types=None):
        super().__init__()
        enc_out = hidden + hidden // 2
        self.encoder = FGNEncoder(n_snps, hidden, dropout, marker_types)
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
    """阶段1: 掩码重建预训练 (独立随机掩码, 原始版本 — 保留兼容)"""
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


def pretrain_prefgn_v2(model, X, epochs=50, lr=1e-3, patience=10,
                       mask_block=(3, 15), mask_frac=0.2, noise_std=0.05):
    """阶段1 v2: 块掩码 + 高斯噪声去噪。

    - 随机选择起始位置, 掩码连续 mask_block 个 SNP (模拟 LD 块缺失)
    - 对未掩码位置加高斯噪声
    - 损失 = 0.7 × MSE(掩码位置) + 0.3 × MSE(所有位置)
    """
    model = model.to(DEVICE)
    X_t = torch.FloatTensor(X)
    dl = DataLoader(TensorDataset(X_t),
                    batch_size=min(128, len(X)), shuffle=True)

    opt = torch.optim.AdamW(model.pretrainable_params(), lr=lr)
    n_snps = X.shape[1]
    bl, bh = mask_block

    def forward_block_masked(batch, m):
        xb = batch[0].to(DEVICE)
        # 创建块掩码
        mask = torch.ones_like(xb, dtype=torch.bool)
        n_mask_target = int(n_snps * mask_frac)
        n_masked = 0
        while n_masked < n_mask_target:
            start = torch.randint(0, max(1, n_snps - bl), (1,)).item()
            end = min(start + torch.randint(bl, bh + 1, (1,)).item(), n_snps)
            mask[:, start:end] = False
            n_masked += (end - start)

        x_masked = xb.clone()
        x_masked[~mask] = 0.0
        # 对未掩码位置加高斯噪声
        noise = torch.randn_like(xb) * noise_std
        x_masked[mask] = x_masked[mask] + noise[mask]

        recon = m.reconstruct(x_masked)
        loss_masked = F.mse_loss(recon[~mask], xb[~mask])
        loss_all = F.mse_loss(recon, xb)
        return 0.7 * loss_masked + 0.3 * loss_all

    def sched(opt_):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_, mode='min', factor=0.5, patience=5)

    return early_stop_restore(model, opt, sched, None, forward_block_masked,
                              dl, epochs, patience, DEVICE).cpu()


# ============================================================================
# LD-GCN 已移除 (LD 图卷积极不适应基因组预测任务)
# ============================================================================
