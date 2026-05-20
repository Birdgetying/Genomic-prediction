#!/usr/bin/env python
"""
Genomic Prediction Ensemble — Consolidated Self-Contained Script
==================================================================
Wheat + Rice + Maize pipelines bundled into one file for HPC deployment.
No local imports — all model definitions, training utilities, and shared
modules are inlined.

Usage:
  python genomic_ensemble.py wheat          # Quick test: 1 trait x 2 folds
  python genomic_ensemble.py wheat --full   # Full: all traits x 5 folds
  python genomic_ensemble.py rice
  python genomic_ensemble.py rice --full
  python genomic_ensemble.py maize
  python genomic_ensemble.py maize --full
  python genomic_ensemble.py all --full     # Run all three crops
"""

import json, time, os, sys, random
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from scipy.stats import pearsonr
import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Global Config
# ============================================================================
RANDOM_SEED = 42
N_FOLDS = 5
GWAS_TOP_K = 5000
MAF_THRESHOLD = 0.05
MARKER_SELECTOR = 'gwas'  # 'gwas' | 'haplotype' | 'hybrid'
HAPLO_GWAS_FRAC = 0.6

# Project root on HPC
PROJECT_DIR = "/storage/public/home/2024110093/genomic_prediction"

# Wheat data paths
WHEAT_DATA_BASE = "/storage/public/home/2024110093/data/Variation/CSIAAS/"
WHEAT_SNP_VCF   = WHEAT_DATA_BASE + "Core819Samples_snp.filter.final.id_gt.813m.vcf.gz"
WHEAT_INDEL_VCF = WHEAT_DATA_BASE + "Core819Samples_indel.filter.final.id_gt.813m.vcf.gz"
WHEAT_SV_VCF    = WHEAT_DATA_BASE + "SV.new.vcf.gz"
WHEAT_PHENO     = WHEAT_DATA_BASE + "Phe.txt"
WHEAT_VCF_ID    = WHEAT_DATA_BASE + "VCFID.txt"
WHEAT_MAX_VARIANTS_PER_TYPE = 15000

# Rice data paths
RICE_DATA_DIR = PROJECT_DIR + "/results/rice_data"

# Maize data paths
MAIZE_DATA_DIR = PROJECT_DIR + "/data2"

# Seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# PYTHONHASHSEED must be set at process launch: export PYTHONHASHSEED=42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TYPE_TRAD = 'Traditional'
TYPE_DL = 'DL'
TYPE_ENS = 'Ensemble'
TYPE_HYBRID = 'Hybrid'

def _model_type(mname):
    if mname == 'ResFGN': return TYPE_HYBRID
    if mname in ('Stacking (DL)', 'Stacking (All)', 'Trad Ensemble'): return TYPE_ENS
    if mname in ('RRBLUP', 'GBLUP', 'XGBoost', 'ElasticNet', 'GWAS_RRBLUP'): return TYPE_TRAD
    return TYPE_DL

# ============================================================================
# Section A: Shared NN Utilities (from genomic_nn_models.py)
# ============================================================================

def early_stop_restore(model, opt, scheduler_fn, loss_fn, forward_fn,
                       data_loader, epochs, patience, device):
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
            val_loss = 0.0; count = 0
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


class FGNEncoder(nn.Module):
    """BN -> FFT -> FreqSE -> FreqConv + SNP gate -> TimeConv -> concat output"""
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
        if marker_types is not None:
            n_types = int(max(marker_types) + 1) if hasattr(marker_types, '__len__') else 3
            self.type_embed = nn.Parameter(torch.zeros(n_types))
            self.register_buffer('_marker_type_idx',
                                 torch.as_tensor(marker_types, dtype=torch.long))
        else:
            self.type_embed = None
            self._marker_type_idx = None

    def set_marker_types(self, marker_types):
        if self.type_embed is None:
            raise RuntimeError("set_marker_types(): type_embed is None")
        self._marker_type_idx = torch.as_tensor(marker_types, dtype=torch.long)

    def forward(self, x):
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


class PreFGN(nn.Module):
    """Self-supervised pretraining FGN: stage1=masked reconstruction, stage2=phenotype finetune"""
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
        return list(self.encoder.parameters()) + list(self.decoder.parameters())


def pretrain_prefgn(model, X, epochs=50, lr=1e-3, mask_ratio=0.2, patience=10):
    model = model.to(DEVICE)
    X_t = torch.FloatTensor(X)
    dl = DataLoader(TensorDataset(X_t), batch_size=min(128, len(X)), shuffle=True)
    opt = torch.optim.AdamW(model.pretrainable_params(), lr=lr)
    def forward_masked(batch, m):
        xb = batch[0].to(DEVICE)
        mask = torch.rand_like(xb) > mask_ratio
        x_masked = xb.clone(); x_masked[~mask] = 0.0
        recon = m.reconstruct(x_masked)
        return F.mse_loss(recon[~mask], xb[~mask])
    def sched(opt_):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt_, mode='min', factor=0.5, patience=5)
    return early_stop_restore(model, opt, sched, None, forward_masked, dl, epochs, patience, DEVICE).cpu()


def pretrain_prefgn_v2(model, X, epochs=50, lr=1e-3, patience=10,
                       mask_block=(3, 15), mask_frac=0.2, noise_std=0.05):
    """Block masking + Gaussian denoising pretraining"""
    model = model.to(DEVICE)
    X_t = torch.FloatTensor(X)
    dl = DataLoader(TensorDataset(X_t), batch_size=min(128, len(X)), shuffle=True)
    opt = torch.optim.AdamW(model.pretrainable_params(), lr=lr)
    n_snps = X.shape[1]; bl, bh = mask_block
    def forward_block_masked(batch, m):
        xb = batch[0].to(DEVICE)
        mask = torch.ones_like(xb, dtype=torch.bool)
        n_mask_target = int(n_snps * mask_frac); n_masked = 0
        while n_masked < n_mask_target:
            start = torch.randint(0, max(1, n_snps - bl), (1,)).item()
            end = min(start + torch.randint(bl, bh + 1, (1,)).item(), n_snps)
            mask[:, start:end] = False
            n_masked += (end - start)
        x_masked = xb.clone(); x_masked[~mask] = 0.0
        noise = torch.randn_like(xb) * noise_std
        x_masked[mask] = x_masked[mask] + noise[mask]
        recon = m.reconstruct(x_masked)
        loss_masked = F.mse_loss(recon[~mask], xb[~mask])
        loss_all = F.mse_loss(recon, xb)
        return 0.7 * loss_masked + 0.3 * loss_all
    def sched(opt_):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt_, mode='min', factor=0.5, patience=5)
    return early_stop_restore(model, opt, sched, None, forward_block_masked, dl, epochs, patience, DEVICE).cpu()


# ============================================================================
# Section B: Deep Kernel GP (from deep_kernel_gp.py)
# ============================================================================
JITTER = 1e-5


class GenomicEncoder(nn.Module):
    """Residual MLP: n_snps -> 256 -> 256 -> 128 -> latent_dim"""
    def __init__(self, n_snps, latent_dim=24, hidden1=256, hidden2=128, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.BatchNorm1d(n_snps), nn.Linear(n_snps, hidden1), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(hidden1), nn.Linear(hidden1, hidden1), nn.GELU(),
            nn.Dropout(dropout))
        self.down_proj = nn.Sequential(
            nn.BatchNorm1d(hidden1), nn.Linear(hidden1, hidden2), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.BatchNorm1d(hidden2), nn.Linear(hidden2, latent_dim))
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        h = self.input_proj(x); h = h + self.block1(h)
        h = self.down_proj(h); return self.layer_norm(h)


class DeepKernelGP(nn.Module):
    """Exact GP with learned NN encoder"""
    def __init__(self, encoder, outputscale=1.0, lengthscale=1.0, noise=0.1):
        super().__init__()
        self.encoder = encoder
        self.log_outputscale = nn.Parameter(torch.tensor(np.log(outputscale)))
        self.log_lengthscale = nn.Parameter(torch.tensor(np.log(lengthscale)))
        self.log_noise = nn.Parameter(torch.tensor(np.log(noise)))
        self._train_features = None; self._train_y = None; self._K_inv = None

    def _build_kernel(self, X1, X2=None):
        lsq = torch.exp(self.log_lengthscale) ** 2
        outscale = torch.exp(self.log_outputscale)
        f1 = self.encoder(X1)
        if X2 is None:
            sq_dist = torch.cdist(f1, f1, p=2).pow(2)
        else:
            sq_dist = torch.cdist(f1, self.encoder(X2), p=2).pow(2)
        return outscale * torch.exp(-0.5 * sq_dist / lsq)

    def _kernel_on_features(self, F1, F2):
        lsq = torch.exp(self.log_lengthscale) ** 2
        outscale = torch.exp(self.log_outputscale)
        sq_dist = torch.cdist(F1, F2, p=2).pow(2)
        return outscale * torch.exp(-0.5 * sq_dist / lsq)

    def marginal_nll(self, X, y):
        n = X.shape[0]; K = self._build_kernel(X)
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
            n = X.shape[0]; K = self._build_kernel(X)
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
            K_ss = torch.exp(self.log_outputscale) * torch.ones(X_test.shape[0], device=X_test.device)
            diag_terms = torch.diag(K_star @ self._K_inv @ K_star.T)
            variance = torch.clamp(K_ss - diag_terms, min=0)
        return mean, variance


def train_dkgp(model, X_train, y_train, epochs=200, lr=5e-3, patience=15,
               batch_size=None, verbose=True):
    model = model.to(DEVICE)
    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8)
    best_state, best_loss, wait = None, float('inf'), 0
    eval_every = 50
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        if batch_size and batch_size < len(X_train):
            idx = torch.randperm(len(X_train))[:batch_size]
            loss = model.marginal_nll(X_t[idx], y_t[idx])
        else:
            loss = model.marginal_nll(X_t, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        current_loss = loss.item(); scheduler.step(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose: print(f"    DKL early stop @ epoch {epoch+1}, NLL={best_loss:.4f}")
                break
        if verbose and (epoch + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_nll = model.marginal_nll(X_t, y_t).item()
            print(f"    DKL epoch {epoch+1:3d}: NLL={eval_nll:.4f}, "
                  f"l={model.log_lengthscale.exp():.3f}, "
                  f"sf={model.log_outputscale.exp():.3f}, "
                  f"sn={model.log_noise.exp():.3f}")
    if best_state is not None: model.load_state_dict(best_state)
    model.eval(); model = model.cpu(); return model


# ============================================================================
# Section C: Haplotype Scoring (from haplotype_scoring.py)
# ============================================================================
VARIANT_TYPE_WEIGHTS = {0: 1.0, 1: 3.5, 2: 3.0}
DEFAULT_WINDOW = 50
DEFAULT_R2_THRESH = 0.6
MIN_MAF = 1e-4


def _compute_univariate_effects(X, y):
    y_c = y - y.mean(); X_c = X - X.mean(axis=0, keepdims=True)
    var_x = np.maximum(X_c.var(axis=0), 1e-8)
    cov_xy = X_c.T @ y_c / len(y_c)
    return np.abs(cov_xy / var_x)


def compute_haplotype_scores(X, y, variant_types=None, maf=None):
    p = X.shape[1]
    func_w = np.array([VARIANT_TYPE_WEIGHTS.get(int(vt), 1.0) for vt in variant_types]) if variant_types is not None else np.ones(p)
    if maf is None:
        af = X.mean(axis=0) / 2.0
        maf = np.minimum(af, 1.0 - af)
    rarity_w = np.maximum(-np.log10(np.maximum(maf, MIN_MAF)), 1.0)
    effects = _compute_univariate_effects(X, y)
    scores = func_w * rarity_w * (1.0 + effects)
    return scores, {'func': func_w, 'rarity': rarity_w, 'effect': effects}


def ld_prune_markers(X, scores, window=DEFAULT_WINDOW, r2_thresh=DEFAULT_R2_THRESH):
    n, p = X.shape
    X_c = X - X.mean(axis=0, keepdims=True)
    X_n = X_c / (np.linalg.norm(X_c, axis=0, keepdims=True) + 1e-12)
    order = np.argsort(-scores); kept = []; kept_positions = []
    for idx in order:
        redundant = False; xj = X_n[:, idx]
        for kp in kept_positions:
            if abs(idx - kp) <= window:
                r = np.dot(xj, X_n[:, kp])
                if r * r > r2_thresh: redundant = True; break
        if not redundant: kept.append(idx); kept_positions.append(idx)
    return np.array(kept, dtype=int)


def haplotype_select(X, y, k, variant_types=None, window=DEFAULT_WINDOW, r2_thresh=DEFAULT_R2_THRESH):
    af = X.mean(axis=0) / 2.0; maf = np.minimum(af, 1.0 - af)
    scores, _ = compute_haplotype_scores(X, y, variant_types, maf)
    n_candidates = min(3 * k, X.shape[1])
    cand_idx = np.argsort(-scores)[:n_candidates]
    X_cand = X[:, cand_idx]; scores_cand = scores[cand_idx]
    pruned_sub = ld_prune_markers(X_cand, scores_cand, window, r2_thresh)
    pruned = cand_idx[pruned_sub]
    if len(pruned) < k:
        remaining = np.setdiff1d(np.argsort(-scores), pruned)
        need = k - len(pruned)
        pruned = np.concatenate([pruned, remaining[:need]])
    top_k = pruned[np.argsort(-scores[pruned])[:k]]
    return np.sort(top_k)


def hybrid_select(X, y, k, variant_types=None, gwas_frac=0.6, window=DEFAULT_WINDOW, r2_thresh=DEFAULT_R2_THRESH):
    n_gwas = int(k * gwas_frac); n_hap = k - n_gwas
    y_c = y - y.mean(); X_c = X - X.mean(axis=0)
    num = np.dot(y_c, X_c)
    denom = np.std(y_c) * len(y) * np.sqrt(np.sum(X_c ** 2, axis=0) + 1e-12)
    gwas_top = np.argsort(np.abs(num / denom))[-n_gwas * 2:]
    hap_top = haplotype_select(X, y, n_hap * 2, variant_types, window, r2_thresh)
    combined = list(gwas_top[-n_gwas:]); combined_set = set(combined)
    for idx in hap_top:
        if idx not in combined_set and len(combined) < k: combined.append(idx); combined_set.add(idx)
    if len(combined) < k:
        for idx in gwas_top:
            if idx not in combined_set and len(combined) < k: combined.append(idx); combined_set.add(idx)
    return np.sort(np.array(combined[:k], dtype=int))


# ============================================================================
# Section D: Common DL Model Classes (canonical wheat/rice versions)
# ============================================================================

class FourierGenomicNet(nn.Module):
    """FGN — FFT spectral + time-domain dual path"""
    def __init__(self, n_snps, hidden=64, dropout=0.35):
        super().__init__()
        self.n_freq = n_snps // 2 + 1
        self.spec_r = nn.Parameter(torch.randn(1, 32, self.n_freq) * 0.02)
        self.spec_i = nn.Parameter(torch.randn(1, 32, self.n_freq) * 0.02)
        self.freq_conv = nn.Sequential(
            nn.Conv1d(32, hidden, 7, padding=3), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout*0.5))
        self.time_conv = nn.Sequential(
            nn.Conv1d(1, hidden, 21, padding=10), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout*0.5))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden*2, hidden*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        xc = torch.fft.rfft(x, dim=1)
        xr = xc.real.unsqueeze(1).expand(-1, 32, -1) * self.spec_r
        xi = xc.imag.unsqueeze(1).expand(-1, 32, -1) * self.spec_i
        fp = self.pool(self.freq_conv(xr + xi)).squeeze(-1)
        tp = self.pool(self.time_conv(x.unsqueeze(1))).squeeze(-1)
        return self.head(torch.cat([fp, tp], dim=1))


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        e = out_ch // 4; r = out_ch - e*4
        self.b3 = nn.Sequential(nn.Conv1d(in_ch, e, 3, padding=1), nn.BatchNorm1d(e), nn.GELU())
        self.b7 = nn.Sequential(nn.Conv1d(in_ch, e, 7, padding=3), nn.BatchNorm1d(e), nn.GELU())
        self.b15 = nn.Sequential(nn.Conv1d(in_ch, e, 15, padding=7), nn.BatchNorm1d(e), nn.GELU())
        self.b31 = nn.Sequential(nn.Conv1d(in_ch, e+r, 31, padding=15), nn.BatchNorm1d(e+r), nn.GELU())
        self.do = nn.Dropout(dropout*0.5)

    def forward(self, x):
        return self.do(torch.cat([self.b3(x), self.b7(x), self.b15(x), self.b31(x)], dim=1))


class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(ch, ch//r, 1),
                                nn.GELU(), nn.Conv1d(ch//r, ch, 1), nn.Sigmoid())
    def forward(self, x):
        return x * self.se(x)


class MultiScaleInceptionCNN(nn.Module):
    """MICNN — multi-scale Inception + SE"""
    def __init__(self, n_snps, hidden=48, dropout=0.35):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, hidden, 7, padding=3), nn.BatchNorm1d(hidden), nn.GELU())
        self.b1 = InceptionBlock(hidden, hidden*2, dropout); self.s1 = SEBlock(hidden*2)
        self.b2 = InceptionBlock(hidden*2, hidden*3, dropout); self.s2 = SEBlock(hidden*3)
        self.b3 = InceptionBlock(hidden*3, hidden*4, dropout); self.s3 = SEBlock(hidden*4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden*4, hidden*3), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden*3, hidden*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden*2, 1))

    def forward(self, x):
        x = self.stem(x.unsqueeze(1))
        x = self.s1(self.b1(x)); x = self.s2(self.b2(x)); x = self.s3(self.b3(x))
        return self.head(self.pool(x).squeeze(-1))


class HaarWaveletDecomp(nn.Module):
    """Learnable Haar wavelet decomposition"""
    def __init__(self):
        super().__init__()
        s2 = np.sqrt(2)
        self.lo = nn.Parameter(torch.tensor([[[1., 1.]]]) / s2)
        self.hi = nn.Parameter(torch.tensor([[[1., -1.]]]) / s2)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return F.conv1d(x, self.lo, stride=2), F.conv1d(x, self.hi, stride=2)


class FGNv2(nn.Module):
    """FGN v2: FFT + Wavelet dual-spectral + time residual"""
    def __init__(self, n_snps, hidden=64, dropout=0.35):
        super().__init__()
        self.n_freq = n_snps // 2 + 1
        self.spec_r = nn.Parameter(torch.randn(1, 24, self.n_freq) * 0.02)
        self.spec_i = nn.Parameter(torch.randn(1, 24, self.n_freq) * 0.02)
        self.freq_conv = nn.Sequential(
            nn.Conv1d(24, hidden, 7, padding=3), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout*0.5))
        self.wavelet = HaarWaveletDecomp()
        self.wavelet_conv = nn.Sequential(
            nn.Conv1d(2, hidden//2, 7, padding=3), nn.BatchNorm1d(hidden//2), nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Conv1d(hidden//2, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout*0.5))
        self.time_conv = nn.Sequential(
            nn.Conv1d(1, hidden//2, 21, padding=10), nn.BatchNorm1d(hidden//2), nn.GELU(),
            nn.Dropout(dropout*0.5))
        self.pool = nn.AdaptiveAvgPool1d(1)
        total = hidden + hidden + hidden//2
        self.head = nn.Sequential(
            nn.Linear(total, hidden*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        xc = torch.fft.rfft(x, dim=1)
        xr = xc.real.unsqueeze(1).expand(-1, 24, -1) * self.spec_r
        xi = xc.imag.unsqueeze(1).expand(-1, 24, -1) * self.spec_i
        fp = self.pool(self.freq_conv(xr + xi)).squeeze(-1)
        cA, cD = self.wavelet(x)
        wp = self.pool(self.wavelet_conv(torch.cat([cA, cD], dim=1))).squeeze(-1)
        tp = self.pool(self.time_conv(x.unsqueeze(1))).squeeze(-1)
        return self.head(torch.cat([fp, wp, tp], dim=1))


class FGNv3(nn.Module):
    """FGN v3: Freq SE + SNP gate + BN — composes FGNEncoder + regression head"""
    def __init__(self, n_snps, hidden=48, dropout=0.35, marker_types=None):
        super().__init__()
        self.encoder = FGNEncoder(n_snps, hidden, dropout, marker_types)
        total = self.encoder.out_dim
        self.head = nn.Sequential(
            nn.Linear(total, hidden*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.head(self.encoder(x))


class DilatedInceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        e = out_ch // 4; r = out_ch - e*4
        self.b7 = nn.Sequential(nn.Conv1d(in_ch, e, 7, padding=3), nn.BatchNorm1d(e), nn.GELU())
        self.b15 = nn.Sequential(nn.Conv1d(in_ch, e, 15, padding=7), nn.BatchNorm1d(e), nn.GELU())
        self.bd2 = nn.Sequential(nn.Conv1d(in_ch, e, 7, padding=6, dilation=2), nn.BatchNorm1d(e), nn.GELU())
        self.bd4 = nn.Sequential(nn.Conv1d(in_ch, e+r, 7, padding=12, dilation=4), nn.BatchNorm1d(e+r), nn.GELU())
        self.do = nn.Dropout(dropout*0.5)

    def forward(self, x):
        return self.do(torch.cat([self.b7(x), self.b15(x), self.bd2(x), self.bd4(x)], dim=1))


class SpatialPyramidPool1D(nn.Module):
    def __init__(self, bins=(1, 2, 4, 8)):
        super().__init__(); self.bins = bins
    def forward(self, x):
        B = x.size(0)
        return torch.cat([F.adaptive_avg_pool1d(x, b).view(B, -1) for b in self.bins], dim=1)


class MICNNv2(nn.Module):
    """MICNN v2: +DilatedConv + SPP"""
    def __init__(self, n_snps, hidden=40, dropout=0.35, spp_bins=(1, 2, 4)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, hidden, 7, padding=3), nn.BatchNorm1d(hidden), nn.GELU())
        self.b1 = DilatedInceptionBlock(hidden, hidden*2, dropout); self.s1 = SEBlock(hidden*2)
        self.b2 = DilatedInceptionBlock(hidden*2, hidden*3, dropout); self.s2 = SEBlock(hidden*3)
        self.b3 = DilatedInceptionBlock(hidden*3, hidden*4, dropout); self.s3 = SEBlock(hidden*4)
        self.spp = SpatialPyramidPool1D(spp_bins)
        spp_dim = (hidden*4) * sum(spp_bins)
        self.head = nn.Sequential(
            nn.Linear(spp_dim, hidden*8), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden*8, hidden*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden*4, 1))

    def forward(self, x):
        x = self.stem(x.unsqueeze(1))
        x = self.s1(self.b1(x)); x = self.s2(self.b2(x)); x = self.s3(self.b3(x))
        return self.head(self.spp(x))


class FusionNet(nn.Module):
    """Three-branch fusion: FGN(spectral) + EFM(statistical) + MICNN(local)"""
    def __init__(self, n_snps, hidden_dim=48, dropout=0.35):
        super().__init__()
        self.n_freq = n_snps // 2 + 1
        self.spec_r = nn.Parameter(torch.randn(1, 16, self.n_freq) * 0.02)
        self.spec_i = nn.Parameter(torch.randn(1, 16, self.n_freq) * 0.02)
        self.fgn_conv = nn.Sequential(
            nn.Conv1d(16, hidden_dim, 7, padding=3), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(dropout*0.3))
        self.fgn_pool = nn.AdaptiveAvgPool1d(1)
        self.efm_linear = nn.Linear(n_snps, 1)
        self.V_f = nn.Parameter(torch.randn(n_snps, 4) * 0.01)
        self.efm_deep = nn.Sequential(
            nn.Linear(n_snps, hidden_dim*2), nn.BatchNorm1d(hidden_dim*2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.micnn_stem = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 7, padding=3), nn.BatchNorm1d(hidden_dim), nn.GELU())
        self.micnn_block = DilatedInceptionBlock(hidden_dim, hidden_dim, dropout)
        self.micnn_se = SEBlock(hidden_dim)
        self.micnn_spp = SpatialPyramidPool1D(bins=(1, 2, 4))
        micnn_dim = hidden_dim * 7
        fusion_in = hidden_dim + (hidden_dim + 2) + micnn_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim*3), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*3, hidden_dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, 1))

    def forward(self, x):
        xc = torch.fft.rfft(x, dim=1)
        xr = xc.real.unsqueeze(1).expand(-1, 16, -1) * self.spec_r
        xi = xc.imag.unsqueeze(1).expand(-1, 16, -1) * self.spec_i
        fgn_f = self.fgn_pool(self.fgn_conv(xr+xi)).squeeze(-1)
        lo = self.efm_linear(x)
        Vd = F.dropout(self.V_f, p=0.1, training=self.training)
        xv = x.unsqueeze(2) * Vd.unsqueeze(0)
        fm = 0.5 * (xv.sum(1).pow(2) - (xv.pow(2)).sum(1)).sum(1, keepdim=True)
        efm_f = torch.cat([lo, fm, self.efm_deep(x)], dim=1)
        m = self.micnn_stem(x.unsqueeze(1))
        m = self.micnn_se(self.micnn_block(m))
        micnn_f = self.micnn_spp(m)
        return self.fusion_head(torch.cat([fgn_f, efm_f, micnn_f], dim=1))


# ============================================================================
# Section E: Common Training Functions
# ============================================================================

def train_torch_model(model, X_train, y_train,
                      epochs=300, batch_size=128, lr=1e-3, weight_decay=1e-4,
                      patience=30, val_ratio=0.15):
    model = model.to(DEVICE)
    n_total = len(X_train)
    n_val = max(1, int(n_total * val_ratio))
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.permutation(n_total)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xt = torch.FloatTensor(X_train[tr_idx]).to(DEVICE)
    yt = torch.FloatTensor(y_train[tr_idx]).to(DEVICE)
    Xv = torch.FloatTensor(X_train[val_idx]).to(DEVICE)
    yv = torch.FloatTensor(y_train[val_idx]).to(DEVICE)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=min(batch_size, len(tr_idx)), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=15)
    crit = nn.MSELoss()
    best_state, best_loss, wait = None, float('inf'), 0
    for _ in range(epochs):
        model.train()
        for bx, by in dl:
            opt.zero_grad()
            loss = crit(model(bx).squeeze(), by)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv).squeeze(), yv).item()
        sch.step(vl)
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    model.eval()
    return model.cpu()


def predict_torch_model(model, X):
    model = model.to(DEVICE)
    Xt = torch.FloatTensor(X).to(DEVICE)
    model.eval()
    with torch.no_grad():
        preds = model(Xt).squeeze().cpu().numpy()
    model.cpu()
    return preds


# ============================================================================
# Section F: Traditional Models
# ============================================================================

class RRBLUP:
    def __init__(self): self.model = RidgeCV(alphas=np.logspace(-3, 3, 30))
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)


class GBLUP:
    def __init__(self): self.model = RidgeCV(alphas=np.logspace(-3, 3, 20))
    def fit(self, G_train, y_train): self.model.fit(G_train, y_train); return self
    def predict(self, G_test_train): return self.model.predict(G_test_train)


class XGBoostModel:
    def __init__(self, n_estimators=500, max_depth=6, lr=0.05):
        self.params = {
            'n_estimators': n_estimators, 'max_depth': max_depth,
            'learning_rate': lr, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'random_state': RANDOM_SEED, 'n_jobs': 8, 'verbosity': 0}
    def fit(self, X, y): self.model = xgb.XGBRegressor(**self.params); self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)


class ElasticNetModel:
    def __init__(self):
        self.model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                                  alphas=np.logspace(-4, 2, 20),
                                  cv=3, random_state=RANDOM_SEED, max_iter=5000, n_jobs=8)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)


class GWASWeightedRRBLUP:
    def __init__(self):
        self.weights = None; self.model = RidgeCV(alphas=np.logspace(-3, 3, 30))
    def fit(self, X, y):
        y_c = y - y.mean(); X_c = X - X.mean(axis=0)
        denom = np.std(y_c) * len(y) * np.sqrt(np.sum(X_c ** 2, axis=0) + 1e-12)
        self.weights = np.abs(np.dot(y_c, X_c) / denom)
        self.weights = self.weights / self.weights.mean()
        X_weighted = X * self.weights[None, :]
        self.model.fit(X_weighted, y); return self
    def predict(self, X):
        X_weighted = X * self.weights[None, :]
        return self.model.predict(X_weighted)


# ============================================================================
# Section G: GWAS / Marker Selection / Model Factory
# ============================================================================

def gwas_select(X, y, top_k):
    y_c = y - y.mean(); X_c = X - X.mean(axis=0)
    num = np.dot(y_c, X_c)
    denom = np.std(y_c) * len(y) * np.sqrt(np.sum(X_c**2, axis=0) + 1e-12)
    return np.argsort(np.abs(num / denom))[-top_k:]


def maf_filter(X, threshold=MAF_THRESHOLD):
    af = X.mean(axis=0) / 2.0; maf = np.minimum(af, 1.0 - af)
    return np.where(maf >= threshold)[0]


def _select_dl_markers(Xtr_raw, Xte_raw, ytr, gidx_gwas, vt_maf, n_snps):
    """Select markers for DL models based on global MARKER_SELECTOR config."""
    if MARKER_SELECTOR == 'haplotype':
        gidx_dl = haplotype_select(Xtr_raw, ytr, n_snps, vt_maf)
    elif MARKER_SELECTOR == 'hybrid':
        gidx_dl = hybrid_select(Xtr_raw, ytr, n_snps, vt_maf, gwas_frac=HAPLO_GWAS_FRAC)
    else:
        gidx_dl = gidx_gwas
    Xtr_dl = Xtr_raw[:, gidx_dl]; Xte_dl = Xte_raw[:, gidx_dl]
    vt_dl = vt_maf[gidx_dl] if vt_maf is not None else None
    sc_dl = StandardScaler(); Xtr_dl_s = sc_dl.fit_transform(Xtr_dl).astype(np.float32)
    Xte_dl_s = sc_dl.transform(Xte_dl).astype(np.float32)
    return gidx_dl, vt_dl, Xtr_dl_s, Xte_dl_s


def create_model(name, n_snps, overrides=None):
    o = overrides or {}
    if name == 'FGN': return FourierGenomicNet(n_snps=n_snps, hidden=64, dropout=0.35)
    if name == 'MICNN': return MultiScaleInceptionCNN(n_snps=n_snps, hidden=48, dropout=0.35)
    if name == 'FGN v2': return FGNv2(n_snps=n_snps, hidden=64, dropout=0.35)
    if name == 'MICNN v2': return MICNNv2(n_snps=n_snps, hidden=40, dropout=0.35, spp_bins=(1, 2, 4))
    if name == 'FGN v3': return FGNv3(n_snps=n_snps, hidden=o.get('hidden', 48), dropout=o.get('dropout', 0.35), marker_types=o.get('marker_types'))
    if name == 'FusionNet': return FusionNet(n_snps=n_snps, hidden_dim=o.get('hidden_dim', 48), dropout=o.get('dropout', 0.35))
    if name == 'PreFGN': return PreFGN(n_snps=n_snps, hidden=o.get('hidden', 64), dropout=o.get('dropout', 0.35), marker_types=o.get('marker_types'))
    if name == 'DeepKernelGP':
        encoder = GenomicEncoder(n_snps=n_snps, latent_dim=o.get('latent_dim', 24), hidden1=o.get('hidden1', 256), hidden2=o.get('hidden2', 128), dropout=o.get('dropout', 0.3))
        return DeepKernelGP(encoder, outputscale=o.get('outputscale', 1.0), lengthscale=o.get('lengthscale', 1.0), noise=o.get('noise', 0.1))
    raise ValueError(f"Unknown model: {name}")


RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

MARKER_TYPE_MODELS = {'FGN v3', 'PreFGN'}

TRAD_NAMES = ['RRBLUP', 'GBLUP', 'XGBoost', 'ElasticNet', 'GWAS_RRBLUP']
DL_BASE_NAMES = ['FGN', 'MICNN', 'FGN v2', 'MICNN v2', 'FGN v3', 'PreFGN', 'DeepKernelGP']
DL_NAMES = DL_BASE_NAMES + ['FusionNet']
EXTRA_NAMES = ['ResFGN']
ALL_NAMES = TRAD_NAMES + DL_NAMES + EXTRA_NAMES


def _make_trad_configs(G_train, G_te_tr, n_snps, n_train):
    """Per-fold traditional model configs — GBLUP closures capture G matrices."""
    return [
        ('RRBLUP', lambda: RRBLUP(), lambda m, Xs, yt: m.fit(Xs, yt), lambda m, Xs: m.predict(Xs), n_snps + 1),
        ('GBLUP', lambda: GBLUP(), lambda m, _x, yt: m.fit(G_train, yt), lambda m, _x: m.predict(G_te_tr), n_train + 1),
        ('XGBoost', lambda: XGBoostModel(n_estimators=300), lambda m, Xs, yt: m.fit(Xs, yt), lambda m, Xs: m.predict(Xs), 300 * 6 * 2),
        ('ElasticNet', lambda: ElasticNetModel(), lambda m, Xs, yt: m.fit(Xs, yt), lambda m, Xs: m.predict(Xs), n_snps + 1),
        ('GWAS_RRBLUP', lambda: GWASWeightedRRBLUP(), lambda m, Xs, yt: m.fit(Xs, yt), lambda m, Xs: m.predict(Xs), n_snps * 2 + 1),
    ]


# ============================================================================
# Section H: Stacking / Ensemble Functions
# ============================================================================

def fit_resfgn_components(X, y, n_snps, cv=3, fusion_overrides=None):
    o = fusion_overrides or {}
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=cv)
    ridge.fit(X, y); residuals = y - ridge.predict(X)
    fusion = FusionNet(n_snps=n_snps, hidden_dim=o.get('hidden_dim', 48), dropout=o.get('dropout', 0.35)).to(DEVICE)
    fusion = train_torch_model(fusion, X, residuals, epochs=300, batch_size=64,
                               lr=o.get('lr', 1e-3), weight_decay=o.get('weight_decay', 1e-3),
                               patience=o.get('patience', 30))
    return ridge, fusion


def stacking_evaluate(oof_preds_dict, targets, n_folds=5):
    base_names = list(oof_preds_dict.keys())
    X_meta = np.column_stack([oof_preds_dict[m] for m in base_names])
    meta = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=5)
    meta.fit(X_meta, targets)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    sp = np.zeros(len(targets))
    for tr, te in kf.split(X_meta):
        m = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=3)
        m.fit(X_meta[tr], targets[tr]); sp[te] = m.predict(X_meta[te])
    return {'R2': float(r2_score(targets, sp)), 'Correlation': float(pearsonr(targets, sp)[0]),
            'Meta_weights': meta.coef_.tolist(), 'Meta_intercept': float(meta.intercept_),
            'Base_models': base_names}


def _add_stacking_to_results(oof_dl, oof_trad, y, trait_res, folds_run):
    """Run 3 stacking ensembles and add results to trait_res. Only when folds >= 3."""
    if folds_run < 3: return
    n_cv = min(5, folds_run)
    sr_dl = stacking_evaluate(oof_dl, y, n_folds=n_cv)
    trait_res['Stacking (DL)'] = {'R2': sr_dl['R2'], 'Correlation': sr_dl['Correlation'],
                                  'RMSE': 0.0, 'Type': TYPE_ENS,
                                  'Meta_weights': sr_dl.get('Meta_weights', []),
                                  'Base_models': sr_dl.get('Base_models', [])}
    oof_all = {**oof_trad, **oof_dl}
    sr_all = stacking_evaluate(oof_all, y, n_folds=n_cv)
    trait_res['Stacking (All)'] = {'R2': sr_all['R2'], 'Correlation': sr_all['Correlation'],
                                   'RMSE': 0.0, 'Type': TYPE_ENS,
                                   'Meta_weights': sr_all.get('Meta_weights', []),
                                   'Base_models': sr_all.get('Base_models', [])}
    tsr = stacking_evaluate(oof_trad, y, n_folds=min(5, len(TRAD_NAMES)))
    trait_res['Trad Ensemble'] = {'R2': tsr['R2'], 'Correlation': tsr['Correlation'],
                                  'RMSE': 0.0, 'Type': TYPE_ENS,
                                  'Meta_weights': tsr.get('Meta_weights', []),
                                  'Base_models': tsr.get('Base_models', [])}
    print(f"  {'Stacking (DL)':<24s} {sr_dl['R2']:8.4f} {sr_dl['Correlation']:8.4f}")
    print(f"  {'Stacking (All)':<24s} {sr_all['R2']:8.4f} {sr_all['Correlation']:8.4f}")
    weights_str = dict(zip(sr_all['Base_models'], [f'{w:.3f}' for w in sr_all['Meta_weights']]))
    print(f"    All weights: {weights_str}")
    print(f"  {'Trad Ensemble':<24s} {tsr['R2']:8.4f} {tsr['Correlation']:8.4f}")


def deploy_models(X, y, vt, n_snps, trait_name, output_dir, tuned_params, quick_test=False):
    """Fit all models on full data and save to disk for later inference."""
    import pickle
    deploy_dir = output_dir / f"deployed_{trait_name}"
    if deploy_dir.exists(): import shutil; shutil.rmtree(str(deploy_dir))
    deploy_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Deploying models on full dataset ({len(y)} samples) ...")

    gidx = gwas_select(X, y, n_snps)
    X_f = X[:, gidx]; vt_f = vt[gidx] if vt is not None else None
    sc = StandardScaler(); X_s = sc.fit_transform(X_f).astype(np.float32)
    with open(deploy_dir / "scaler.pkl", 'wb') as f: pickle.dump(sc, f)

    G_full = X_s @ X_s.T / n_snps
    trad_models = {'RRBLUP': (RRBLUP(), X_s),
                   'GBLUP': (GBLUP(), G_full),
                   'XGBoost': (XGBoostModel(n_estimators=300), X_s),
                   'ElasticNet': (ElasticNetModel(), X_s),
                   'GWAS_RRBLUP': (GWASWeightedRRBLUP(), X_s)}
    for tname, (tm, X_in) in trad_models.items():
        tm.fit(X_in, y)
        with open(deploy_dir / f"{tname}.pkl", 'wb') as f: pickle.dump(tm, f)
        print(f"    [saved] {tname}.pkl")

    for mname in DL_NAMES:
        tp = tuned_params.get(mname, {}) if tuned_params else {}
        if vt_f is not None and mname in MARKER_TYPE_MODELS:
            model = create_model(mname, n_snps, overrides=dict(tp, marker_types=vt_f))
        else:
            model = create_model(mname, n_snps, overrides=tp)
        if mname == 'PreFGN' and not quick_test:
            pretrain_prefgn_v2(model, X_s, epochs=50, lr=1e-3, patience=10)
        if mname == 'DeepKernelGP':
            model = train_dkgp(model, X_s, y, epochs=tp.get('epochs', 200),
                               lr=tp.get('lr', 5e-3), patience=tp.get('patience', 15), verbose=False)
            model.fit(torch.FloatTensor(X_s), torch.FloatTensor(y))
        else:
            bs = 64 if mname == 'FusionNet' else 128
            lr = tp.get('lr', 1e-3 if mname == 'FusionNet' else 2e-3)
            wd = tp.get('weight_decay', 1e-3); pat = tp.get('patience', 30)
            model = train_torch_model(model, X_s, y, epochs=300, batch_size=bs, lr=lr, weight_decay=wd, patience=pat)
        torch.save(model.state_dict(), deploy_dir / f"{mname}.pt")
        print(f"    [saved] {mname}.pt")
        del model

    ridge, res_model = fit_resfgn_components(X_s, y, n_snps, cv=3, fusion_overrides=tuned_params.get('FusionNet') if tuned_params else None)
    with open(deploy_dir / "ResFGN_ridge.pkl", 'wb') as f: pickle.dump(ridge, f)
    torch.save(res_model.state_dict(), deploy_dir / "ResFGN_fusion.pt")
    print(f"    [saved] ResFGN_ridge.pkl + ResFGN_fusion.pt")
    del res_model; torch.cuda.empty_cache()

    meta = {'trait': trait_name, 'n_snps': n_snps, 'gwas_indices': gidx.tolist(),
            'n_samples': len(y), 'models': list(trad_models.keys()) + DL_NAMES + ['ResFGN']}
    with open(deploy_dir / "deployment_meta.json", 'w') as f: json.dump(meta, f, indent=2)
    print(f"    [saved] deployment_meta")


def tune_model_hyperparams(model_name, X_train, y_train, n_snps, n_trials=15):
    try: import optuna
    except ImportError: print(f"    [SKIP] Optuna not installed, using defaults"); return {}, 0.0
    n_val = max(16, int(len(y_train) * 0.2))
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.permutation(len(y_train))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    def objective(trial):
        if model_name == 'FGN v3':
            overrides = {'hidden': trial.suggest_categorical('hidden', [32, 48, 64]),
                         'dropout': trial.suggest_float('dropout', 0.2, 0.5),
                         'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
                         'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
                         'patience': trial.suggest_int('patience', 20, 50)}
            model = FGNv3(n_snps=n_snps, hidden=overrides['hidden'], dropout=overrides['dropout'])
            bs = 128
        elif model_name == 'FusionNet':
            overrides = {'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 48, 64]),
                         'dropout': trial.suggest_float('dropout', 0.2, 0.5),
                         'lr': trial.suggest_float('lr', 5e-4, 3e-3, log=True),
                         'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
                         'patience': trial.suggest_int('patience', 20, 50)}
            model = FusionNet(n_snps=n_snps, hidden_dim=overrides['hidden_dim'], dropout=overrides['dropout'])
            bs = 64
        elif model_name == 'DeepKernelGP':
            overrides = {'latent_dim': trial.suggest_categorical('latent_dim', [16, 24, 32]),
                         'hidden1': trial.suggest_categorical('hidden1', [128, 256]),
                         'hidden2': trial.suggest_categorical('hidden2', [64, 128]),
                         'dropout': trial.suggest_float('dropout', 0.1, 0.4),
                         'lr': trial.suggest_float('lr', 1e-3, 1e-2, log=True),
                         'epochs': trial.suggest_int('epochs', 150, 300),
                         'patience': trial.suggest_int('patience', 10, 25)}
            encoder = GenomicEncoder(n_snps=n_snps, latent_dim=overrides['latent_dim'],
                                     hidden1=overrides['hidden1'], hidden2=overrides['hidden2'],
                                     dropout=overrides['dropout'])
            model = DeepKernelGP(encoder)
            model = train_dkgp(model, X_tr, y_tr, epochs=overrides['epochs'], lr=overrides['lr'],
                               patience=overrides['patience'], verbose=False)
            model.fit(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
            mean, _ = model.predict(torch.FloatTensor(X_val))
            return float(r2_score(y_val, mean.cpu().numpy()))
        else: raise ValueError(f"Unknown model for tuning: {model_name}")
        model = train_torch_model(model, X_tr, y_tr, epochs=300, batch_size=bs,
                                  lr=overrides.get('lr', 2e-3),
                                  weight_decay=overrides.get('weight_decay', 1e-3),
                                  patience=overrides.get('patience', 30))
        preds = predict_torch_model(model, X_val)
        return float(r2_score(y_val, preds))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ============================================================================
# Section I: Wheat Pipeline
# ============================================================================

def _print_final_summary(all_results, traits_run, output_dir, total_t0, crop_name):
    eval_models = list(all_results[traits_run[0]].keys())
    print(f"\n{'='*80}\nOVERALL SUMMARY\n{'='*80}")
    print(f"  {'Model':<25s} {'Type':>12s} {'Mean R2':>8s} {'Mean Corr':>10s} {'Best':>8s} {'Worst':>8s}")
    print(f"  {'-'*80}")
    summ = {m: {'R2': [], 'Corr': []} for m in eval_models}
    for t in traits_run:
        for m in eval_models:
            if m in all_results[t]:
                summ[m]['R2'].append(all_results[t][m]['R2'])
                summ[m]['Corr'].append(all_results[t][m].get('Correlation', 0))
    ranked = sorted([(m, np.mean(summ[m]['R2'])) for m in eval_models], key=lambda x: x[1], reverse=True)
    for m, _ in ranked:
        mean_r2 = np.mean(summ[m]['R2']); mean_corr = np.mean(summ[m]['Corr'])
        best = max(summ[m]['R2']); worst = min(summ[m]['R2'])
        mtype = all_results[traits_run[0]][m].get('Type', '')
        print(f"  {m:<25s} {mtype:>12s} {mean_r2:8.4f} {mean_corr:10.4f} {best:8.4f} {worst:8.4f}")
    print(f"\n  Ranking:")
    for i, (m, r) in enumerate(ranked, 1):
        marker = " <-- BEST" if i == 1 else ""
        print(f"  {i:2d}. [{all_results[traits_run[0]][m].get('Type', ''):>12s}] {m:<22s} {r:.4f}{marker}")
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(output_dir / f"ensemble_final_{ts}.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_dir}")
    print(f"Total time: {(time.time()-total_t0)/60:.1f} min")
    print(f"\n{crop_name} Done!")

def load_wheat_data():
    import cyvcf2
    print(f"\n{'='*70}\nWheat VCF Data Loading (SNP + INDEL + SV)\n{'='*70}")

    # 1. Phenotypes
    print("\n[1/4] Loading phenotypes ...")
    df = pd.read_csv(WHEAT_PHENO, sep='\t', header=0)
    n_samples = len(df); traits = {}
    for col in df.columns:
        if col.lower() in ('sample', 'id', 'name', 'accession', 'line'): continue
        try:
            vals = pd.to_numeric(df[col], errors='coerce').values.astype(np.float32)
            mask = ~np.isnan(vals)
            if mask.sum() > 0.5 * len(vals):
                traits[col] = (vals, mask)
        except (ValueError, TypeError): pass
    if not traits:
        df = pd.read_csv(WHEAT_PHENO, sep='\t', header=None)
        vals = pd.to_numeric(df.iloc[:, 1], errors='coerce').values.astype(np.float32)
        mask = ~np.isnan(vals); traits = {'Phenotype': (vals, mask)}
    print(f"  共检测到 {len(traits)} 个性状")
    y_sample_counts = [mask.sum() for _, (_, mask) in traits.items()]
    n_phe = max(y_sample_counts)
    print(f"  表型最大样本数: {n_phe}")

    # 2. VCF IDs
    print("\n[2/4] Loading VCF IDs ...")
    if os.path.exists(WHEAT_VCF_ID):
        vcf_ids = pd.read_csv(WHEAT_VCF_ID, header=None, sep=r'\s+')
        n_vcf_id = len(vcf_ids)
    else: n_vcf_id = n_phe

    # 3. VCF genotypes
    print("\n[3/4] Loading VCF genotypes ...")
    min_samples = min(n_phe, n_vcf_id)
    vcf_configs = [("SNP", WHEAT_SNP_VCF), ("INDEL", WHEAT_INDEL_VCF), ("SV", WHEAT_SV_VCF)]
    X_parts = []; variant_type_ids = []
    for vtype_id, (vtype, vpath) in enumerate(vcf_configs):
        print(f"  [{vtype}] {os.path.basename(vpath)}")
        if not os.path.exists(vpath): print(f"    WARNING: 文件不存在, 跳过"); continue
        try:
            vcf = cyvcf2.VCF(vpath)
            n_vcf = len(vcf.samples); actual_n = min(min_samples, n_vcf)
            genotypes = []; variant_count = 0
            for variant in vcf:
                gt = variant.gt_types
                if len(gt) >= actual_n:
                    gt_slice = gt[:actual_n]
                    gt_arr = np.where(gt_slice == 3, 0, np.clip(gt_slice, 0, 2)).astype(np.int8)
                    genotypes.append(gt_arr)
                    variant_count += 1
                    if variant_count >= WHEAT_MAX_VARIANTS_PER_TYPE: break
            X_v = np.array(genotypes, dtype=np.float32).T
            X_parts.append(X_v)
            variant_type_ids.append(np.full(X_v.shape[1], vtype_id, dtype=np.int32))
            print(f"    提取 {X_v.shape[1]} 个变异位点")
        except Exception as e: print(f"    ERROR: {e}, 跳过")

    if not X_parts: raise RuntimeError("未能加载任何 VCF 数据!")
    X_all = np.hstack(X_parts); vt_all = np.concatenate(variant_type_ids)
    print(f"\n  合并基因型矩阵: {X_all.shape}")
    print(f"  变异类型分布: SNP={int(np.sum(vt_all==0))}, INDEL={int(np.sum(vt_all==1))}, SV={int(np.sum(vt_all==2))}")

    # 4. Per-trait data
    print("\n[4/4] Preparing trait-specific data ...")
    trait_data = {}
    for tname, (y_full, mask) in traits.items():
        n_common = min(len(y_full), X_all.shape[0])
        y_t, X_t = y_full[:n_common], X_all[:n_common]
        mask_t = mask[:n_common]; X_t, y_t = X_t[mask_t], y_t[mask_t]
        vt_t = vt_all.copy()
        vars_per_marker = np.var(X_t, axis=0)
        keep = vars_per_marker >= 0.005
        if keep.sum() < X_t.shape[1]: X_t = X_t[:, keep]; vt_t = vt_t[keep]
        trait_data[tname] = (X_t.astype(np.float32), y_t.astype(np.float32), vt_t)
        print(f"  {tname}: {X_t.shape}")
    return trait_data


def run_wheat(quick_test=True):
    _script_dir = Path(__file__).resolve().parent
    output_dir = _script_dir / "results" / "wheat_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}  |  Quick test: {quick_test}")
    if DEVICE.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")
    trait_data = load_wheat_data()
    trait_names = sorted(trait_data.keys())
    print(f"\n实验性状: {trait_names}")

    traits_run = trait_names[:1] if quick_test else trait_names
    folds_run = min(2, N_FOLDS) if quick_test else N_FOLDS
    if quick_test: print(f"  [QUICK TEST] {len(traits_run)} trait x {folds_run} folds")

    all_results = {}; total_t0 = time.time()

    for t_idx, trait in enumerate(traits_run):
        print(f"\n{'='*80}\n  TRAIT [{t_idx+1}/{len(traits_run)}]: {trait}\n{'='*80}")
        X_all, y, vt_all = trait_data[trait]; y = y.astype(np.float32)
        n_snps = min(GWAS_TOP_K, max(50, X_all.shape[1] - 50))
        print(f"  {len(y)} samples, {X_all.shape[1]} markers -> {n_snps} GWAS-selected")

        # AutoML
        tuned_params = {}
        if not quick_test:
            print(f"\n  [AutoML] Tuning hyperparams ...")
            maf_tune = maf_filter(X_all)
            if len(maf_tune) >= n_snps:
                gidx_t = gwas_select(X_all[:, maf_tune], y, n_snps)
                X_tune = X_all[:, maf_tune][:, gidx_t]
            else: X_tune = X_all[:, gwas_select(X_all, y, n_snps)]
            sc_tune = StandardScaler(); X_tune_s = sc_tune.fit_transform(X_tune).astype(np.float32)
            for tune_name in ['FGN v3', 'FusionNet']:
                best_p, best_r2 = tune_model_hyperparams(tune_name, X_tune_s, y, n_snps, n_trials=15)
                tuned_params[tune_name] = best_p
                pstr = ', '.join(f'{k}={v}' for k, v in best_p.items())
                print(f"    {tune_name}: val R²={best_r2:.4f}  [{pstr}]")

        kf = KFold(n_splits=folds_run, shuffle=True, random_state=RANDOM_SEED)
        results = {m: {'preds': [], 'targets': [], 'params': 0, 'time': 0.0} for m in ALL_NAMES}
        oof_trad = {m: np.zeros(len(y)) for m in TRAD_NAMES}
        oof_dl = {m: np.zeros(len(y)) for m in DL_BASE_NAMES}

        dl_models = {mname: create_model(mname, n_snps) for mname in DL_NAMES}

        for fi, (tr, te) in enumerate(kf.split(X_all)):
            print(f"\n  --- Fold {fi+1}/{folds_run} ---")
            Xtr_raw, Xte_raw = X_all[tr], X_all[te]
            ytr, yte = y[tr], y[te]
            maf_idx = maf_filter(Xtr_raw)
            if len(maf_idx) >= n_snps: Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]; vt_maf = vt_all[maf_idx]
            else: vt_maf = vt_all

            # Traditional models: ALWAYS use GWAS markers
            gidx_gwas = gwas_select(Xtr_raw, ytr, n_snps)
            Xtr_trad = Xtr_raw[:, gidx_gwas]; Xte_trad = Xte_raw[:, gidx_gwas]
            sc_trad = StandardScaler(); Xtr_trad_s = sc_trad.fit_transform(Xtr_trad).astype(np.float32)
            Xte_trad_s = sc_trad.transform(Xte_trad).astype(np.float32)
            G_fold_train = Xtr_trad_s @ Xtr_trad_s.T / n_snps
            G_fold_te_tr = Xte_trad_s @ Xtr_trad_s.T / n_snps

            trad_configs = _make_trad_configs(G_fold_train, G_fold_te_tr, n_snps, len(tr))
            for tname, build_fn, fit_fn, pred_fn, param_count in trad_configs:
                t0 = time.time(); tmodel = build_fn()
                fit_fn(tmodel, Xtr_trad_s, ytr)
                preds = pred_fn(tmodel, Xte_trad_s)
                results[tname]['preds'].extend(preds.tolist())
                results[tname]['targets'].extend(yte.tolist())
                results[tname]['time'] += time.time() - t0
                if fi == 0: results[tname]['params'] = param_count
                oof_trad[tname][te] = preds
                print(f"    {tname:<16s} R²={r2_score(yte, preds):+.4f}")

            gidx_dl, vt_dl, Xtr_dl_s, Xte_dl_s = _select_dl_markers(
                Xtr_raw, Xte_raw, ytr, gidx_gwas, vt_maf, n_snps)

            # DL models
            for mi, mname in enumerate(DL_NAMES):
                tp = tuned_params.get(mname, {})
                if mname in MARKER_TYPE_MODELS:
                    dl_models[mname] = create_model(mname, n_snps, overrides=dict(tp, marker_types=vt_dl))
                if fi == 0: results[mname]['params'] = sum(p.numel() for p in dl_models[mname].parameters())
                t0 = time.time()
                if mname == 'PreFGN' and not quick_test:
                    pretrain_prefgn_v2(dl_models[mname], Xtr_dl_s, epochs=50, lr=1e-3, patience=10)
                if mname == 'DeepKernelGP':
                    model = train_dkgp(dl_models[mname], Xtr_dl_s, ytr, epochs=tp.get('epochs', 200),
                                       lr=tp.get('lr', 5e-3), patience=tp.get('patience', 15), verbose=False)
                    model.fit(torch.FloatTensor(Xtr_dl_s), torch.FloatTensor(ytr))
                    mean, _ = model.predict(torch.FloatTensor(Xte_dl_s)); preds = mean.cpu().numpy()
                else:
                    bs = 64 if mname == 'FusionNet' else 128
                    lr = tp.get('lr', 1e-3 if mname == 'FusionNet' else 2e-3)
                    wd = tp.get('weight_decay', 1e-3); pat = tp.get('patience', 30)
                    model = train_torch_model(dl_models[mname], Xtr_dl_s, ytr, epochs=300,
                                              batch_size=bs, lr=lr, weight_decay=wd, patience=pat)
                    preds = predict_torch_model(model, Xte_dl_s)
                elapsed = time.time() - t0
                results[mname]['preds'].extend(preds.tolist())
                results[mname]['targets'].extend(yte.tolist())
                results[mname]['time'] += elapsed
                print(f"    {mname:<16s} R²={r2_score(yte, preds):+.4f}  ({elapsed:.1f}s)")
                if mname in DL_BASE_NAMES: oof_dl[mname][te] = preds
                if mname not in MARKER_TYPE_MODELS:
                    dl_models[mname] = create_model(mname, n_snps, overrides=tp)

            # ResFGN
            t0 = time.time()
            ridge, res_model = fit_resfgn_components(Xtr_dl_s, ytr, n_snps, cv=3, fusion_overrides=tuned_params.get('FusionNet'))
            pred_te_ridge = ridge.predict(Xte_dl_s)
            res_pred = predict_torch_model(res_model, Xte_dl_s)
            final_pred = pred_te_ridge + res_pred
            elapsed = time.time() - t0
            if fi == 0: results['ResFGN']['params'] = (n_snps+1) + sum(p.numel() for p in res_model.parameters())
            results['ResFGN']['preds'].extend(final_pred.tolist())
            results['ResFGN']['targets'].extend(yte.tolist())
            results['ResFGN']['time'] += elapsed
            print(f"    {'ResFGN':<16s} R²={r2_score(yte, final_pred):+.4f}  ({elapsed:.1f}s)")
            torch.cuda.empty_cache()

        # Trait summary
        print(f"\n  {'-'*70}\n  {trait} Final Results:\n  {'Model':<16s} {'R²':>8s} {'Corr':>8s} {'RMSE':>8s} {'Time':>8s}\n  {'-'*70}")
        trait_res = {}
        for mname in ALL_NAMES:
            p = np.array(results[mname]['preds']); t = np.array(results[mname]['targets'])
            r2_v = float(r2_score(t, p)); corr_v = float(pearsonr(t, p)[0])
            rmse_v = float(np.sqrt(np.mean((p-t)**2)))
            mtype = _model_type(mname)
            tag_map = {TYPE_TRAD: ' [Trad]', TYPE_HYBRID: ' [Hybrid]', TYPE_DL: ' [DL]'}
            tag = tag_map.get(mtype, '')
            trait_res[mname] = {'R2': r2_v, 'Correlation': corr_v, 'RMSE': rmse_v, 'Type': mtype, 'Time': results[mname]['time']/folds_run}
            print(f"  {mname+tag:<24s} {r2_v:8.4f} {corr_v:8.4f} {rmse_v:8.4f} {results[mname]['time']/folds_run:7.1f}s")

        _add_stacking_to_results(oof_dl, oof_trad, y, trait_res, folds_run)
        all_results[trait] = trait_res
        with open(output_dir / "ensemble_intermediate.json", 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        if not quick_test:
            deploy_models(X_all, y, vt_all, n_snps, trait, output_dir, tuned_params, quick_test)

    if not quick_test:
        _print_final_summary(all_results, traits_run, output_dir, total_t0, 'Wheat')


# ============================================================================
# Section J: Rice Pipeline
# ============================================================================

RICE_TRAITS = ['Heading_date', 'Plant_height', 'Num_panicles', 'Num_effective_panicles',
               'Yield', 'Grain_weight', 'Spikelet_length', 'Grain_length', 'Grain_width', 'Grain_thickness']


def load_rice_data():
    """Load pre-processed rice data from genotype_matrix.npz + trait_data.json."""
    print(f"\n{'='*70}\nRice Data Loading\n{'='*70}")
    data = np.load(RICE_DATA_DIR + "/genotype_matrix.npz", allow_pickle=True)
    G = data['G']
    print(f"  Genotype: {G.shape}")
    with open(RICE_DATA_DIR + "/trait_data.json") as f:
        trait_info = json.load(f)
    trait_data = {}
    for t in RICE_TRAITS:
        if t not in trait_info:
            print(f"  {t}: SKIP (not in trait_data.json)")
            continue
        td = trait_info[t]
        idxs = td['genotype_indices']
        y = np.array(td['values']).astype(np.float32)
        X_t = G[idxs]
        mask = ~np.isnan(y)
        trait_data[t] = (X_t[mask], y[mask])
        print(f"  {t}: {mask.sum()} samples")
    return trait_data


def run_rice(quick_test=True):
    _script_dir = Path(__file__).resolve().parent
    output_dir = _script_dir / "results" / "rice_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}  |  Quick test: {quick_test}")
    if DEVICE.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")
    trait_data = load_rice_data()
    trait_names = sorted(trait_data.keys())
    print(f"\n实验性状: {trait_names}")

    traits_run = trait_names[:1] if quick_test else trait_names
    folds_run = min(2, N_FOLDS) if quick_test else N_FOLDS
    if quick_test: print(f"  [QUICK TEST] {len(traits_run)} trait x {folds_run} folds")

    all_results = {}; total_t0 = time.time()

    for t_idx, trait in enumerate(traits_run):
        print(f"\n{'='*80}\n  TRAIT [{t_idx+1}/{len(traits_run)}]: {trait}\n{'='*80}")
        X_all, y = trait_data[trait]; y = y.astype(np.float32)
        n_snps = min(GWAS_TOP_K, max(50, X_all.shape[1] - 50))
        print(f"  {len(y)} samples, {X_all.shape[1]} markers -> {n_snps} GWAS-selected")

        tuned_params = {}
        if not quick_test:
            print(f"\n  [AutoML] Tuning ...")
            maf_tune = maf_filter(X_all)
            if len(maf_tune) >= n_snps:
                gidx_t = gwas_select(X_all[:, maf_tune], y, n_snps)
                X_tune = X_all[:, maf_tune][:, gidx_t]
            else: X_tune = X_all[:, gwas_select(X_all, y, n_snps)]
            sc_tune = StandardScaler(); X_tune_s = sc_tune.fit_transform(X_tune).astype(np.float32)
            for tune_name in ['FGN v3', 'FusionNet']:
                best_p, best_r2 = tune_model_hyperparams(tune_name, X_tune_s, y, n_snps, n_trials=15)
                tuned_params[tune_name] = best_p
                print(f"    {tune_name}: val R²={best_r2:.4f}")

        kf = KFold(n_splits=folds_run, shuffle=True, random_state=RANDOM_SEED)
        results = {m: {'preds': [], 'targets': [], 'params': 0, 'time': 0.0} for m in ALL_NAMES}
        oof_trad = {m: np.zeros(len(y)) for m in TRAD_NAMES}
        oof_dl = {m: np.zeros(len(y)) for m in DL_BASE_NAMES}
        dl_models = {mname: create_model(mname, n_snps) for mname in DL_NAMES}

        for fi, (tr, te) in enumerate(kf.split(X_all)):
            print(f"\n  --- Fold {fi+1}/{folds_run} ---")
            Xtr_raw, Xte_raw = X_all[tr], X_all[te]; ytr, yte = y[tr], y[te]
            maf_idx = maf_filter(Xtr_raw)
            if len(maf_idx) >= n_snps: Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]
            gidx_gwas = gwas_select(Xtr_raw, ytr, n_snps)
            Xtr_trad = Xtr_raw[:, gidx_gwas]; Xte_trad = Xte_raw[:, gidx_gwas]
            sc_trad = StandardScaler(); Xtr_trad_s = sc_trad.fit_transform(Xtr_trad).astype(np.float32)
            Xte_trad_s = sc_trad.transform(Xte_trad).astype(np.float32)
            G_fold_train = Xtr_trad_s @ Xtr_trad_s.T / n_snps; G_fold_te_tr = Xte_trad_s @ Xtr_trad_s.T / n_snps

            trad_configs = _make_trad_configs(G_fold_train, G_fold_te_tr, n_snps, len(tr))
            for tname, build_fn, fit_fn, pred_fn, param_count in trad_configs:
                t0 = time.time(); tmodel = build_fn(); fit_fn(tmodel, Xtr_trad_s, ytr)
                preds = pred_fn(tmodel, Xte_trad_s)
                results[tname]['preds'].extend(preds.tolist()); results[tname]['targets'].extend(yte.tolist())
                results[tname]['time'] += time.time() - t0
                if fi == 0: results[tname]['params'] = param_count
                oof_trad[tname][te] = preds
                print(f"    {tname:<16s} R²={r2_score(yte, preds):+.4f}")

            # DL marker selection (configurable)
            gidx_dl, vt_dl, Xtr_dl_s, Xte_dl_s = _select_dl_markers(
                Xtr_raw, Xte_raw, ytr, gidx_gwas, None, n_snps)

            for mi, mname in enumerate(DL_NAMES):
                if fi == 0: results[mname]['params'] = sum(p.numel() for p in dl_models[mname].parameters())
                t0 = time.time(); tp = tuned_params.get(mname, {})
                if mname == 'PreFGN' and not quick_test:
                    pretrain_prefgn_v2(dl_models[mname], Xtr_dl_s, epochs=50, lr=1e-3, patience=10)
                if mname == 'DeepKernelGP':
                    model = train_dkgp(dl_models[mname], Xtr_dl_s, ytr, epochs=tp.get('epochs', 200),
                                       lr=tp.get('lr', 5e-3), patience=tp.get('patience', 15), verbose=False)
                    model.fit(torch.FloatTensor(Xtr_dl_s), torch.FloatTensor(ytr))
                    mean, _ = model.predict(torch.FloatTensor(Xte_dl_s)); preds = mean.cpu().numpy()
                else:
                    bs = 64 if mname == 'FusionNet' else 128
                    lr = tp.get('lr', 1e-3 if mname == 'FusionNet' else 2e-3); wd = tp.get('weight_decay', 1e-3); pat = tp.get('patience', 30)
                    model = train_torch_model(dl_models[mname], Xtr_dl_s, ytr, epochs=300, batch_size=bs, lr=lr, weight_decay=wd, patience=pat)
                    preds = predict_torch_model(model, Xte_dl_s)
                elapsed = time.time() - t0
                results[mname]['preds'].extend(preds.tolist()); results[mname]['targets'].extend(yte.tolist()); results[mname]['time'] += elapsed
                print(f"    {mname:<16s} R²={r2_score(yte, preds):+.4f}  ({elapsed:.1f}s)")
                if mname in DL_BASE_NAMES: oof_dl[mname][te] = preds
                dl_models[mname] = create_model(mname, n_snps, overrides=tuned_params.get(mname, {}) or {})

            # ResFGN
            t0 = time.time()
            ridge, res_model = fit_resfgn_components(Xtr_dl_s, ytr, n_snps, cv=3, fusion_overrides=tuned_params.get('FusionNet'))
            pred_te_ridge = ridge.predict(Xte_dl_s); res_pred = predict_torch_model(res_model, Xte_dl_s)
            final_pred = pred_te_ridge + res_pred; elapsed = time.time() - t0
            if fi == 0: results['ResFGN']['params'] = (n_snps+1) + sum(p.numel() for p in res_model.parameters())
            results['ResFGN']['preds'].extend(final_pred.tolist()); results['ResFGN']['targets'].extend(yte.tolist()); results['ResFGN']['time'] += elapsed
            print(f"    {'ResFGN':<16s} R²={r2_score(yte, final_pred):+.4f}  ({elapsed:.1f}s)")
            torch.cuda.empty_cache()

        # Summary
        print(f"\n  {trait} Final Results:")
        trait_res = {}
        for mname in ALL_NAMES:
            p = np.array(results[mname]['preds']); t = np.array(results[mname]['targets'])
            r2_v = float(r2_score(t, p)); corr_v = float(pearsonr(t, p)[0]); rmse_v = float(np.sqrt(np.mean((p-t)**2)))
            mtype = _model_type(mname)
            trait_res[mname] = {'R2': r2_v, 'Correlation': corr_v, 'RMSE': rmse_v, 'Type': mtype, 'Time': results[mname]['time']/folds_run}
            print(f"  {mname:<20s} R²={r2_v:+.4f}  Corr={corr_v:+.4f}  RMSE={rmse_v:.4f}")

        _add_stacking_to_results(oof_dl, oof_trad, y, trait_res, folds_run)
        all_results[trait] = trait_res
        with open(output_dir / "ensemble_intermediate.json", 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        if not quick_test:
            deploy_models(X_all, y, None, n_snps, trait, output_dir, tuned_params, quick_test)

    if not quick_test:
        _print_final_summary(all_results, traits_run, output_dir, total_t0, 'Rice')
    else:
        print("\nRice Quick Test Done!")


# ============================================================================
# Section K: Maize Pipeline
# ============================================================================

def load_iranian_data(max_markers=None):
    print(f"\n{'='*70}\nIranian Maize Data Loading\n{'='*70}")
    print("\n[1/3] Loading genotype matrix ..."); t0 = time.time()
    N_META = 17
    nrows = max_markers if max_markers else None
    with open(MAIZE_DATA_DIR + "/Iranian_Samples.csv") as f:
        header_line = f.readline().strip().split(',')
    sample_ids = [str(c).strip() for c in header_line[N_META:] if c.strip() and c.strip() != '*']
    df_geno = pd.read_csv(MAIZE_DATA_DIR + "/Iranian_Samples.csv", skiprows=8, header=None, nrows=nrows, low_memory=False)
    X_raw = df_geno.iloc[:, N_META:].values.T; del df_geno
    X_num = np.select([X_raw == '0', X_raw == '1', X_raw == '2'], [0.0, 1.0, 2.0], default=np.nan).astype(np.float32)
    del X_raw
    col_means = np.nanmean(X_num, axis=0); nan_mask = np.isnan(X_num)
    X_num = np.where(nan_mask, col_means, X_num)
    missing_pct = nan_mask.sum() / nan_mask.size * 100
    print(f"  Genotype: {X_num.shape} [missing={missing_pct:.1f}%] [{time.time()-t0:.1f}s]")

    print("\n[2/3] Loading phenotypes ..."); t0 = time.time()
    pheno = pd.read_csv(MAIZE_DATA_DIR + "/phenotype_iranian.csv")
    pheno_clean = pheno.iloc[1:].copy()
    pheno_clean.columns = ['GID', 'Heat_dtm', 'Heat_dth', 'Drought_dtm', 'Drought_dth']
    pheno_clean['GID'] = pheno_clean['GID'].apply(lambda x: str(int(float(x))) if pd.notna(x) else None)
    for c in ['Heat_dtm', 'Heat_dth', 'Drought_dtm', 'Drought_dth']:
        pheno_clean[c] = pd.to_numeric(pheno_clean[c], errors='coerce')
    pheno_clean = pheno_clean.dropna(subset=['Heat_dtm', 'Heat_dth', 'Drought_dtm', 'Drought_dth'])
    print(f"  Phenotypes: {pheno_clean.shape[0]} samples")

    print("\n[3/3] Aligning samples ...")
    geno_id_set = set(sample_ids)
    pheno_clean = pheno_clean[pheno_clean['GID'].isin(geno_id_set)]
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    aligned_indices = [id_to_idx[sid] for sid in pheno_clean['GID'].values]
    X = X_num[aligned_indices]
    y_dict = {}
    for trait in ['Heat_dtm', 'Heat_dth', 'Drought_dtm', 'Drought_dth']:
        y_dict[trait] = pheno_clean[trait].values.astype(np.float32)
    print(f"  Final: {X.shape[0]} samples, {X.shape[1]} markers, {len(y_dict)} traits")
    return X, y_dict


def run_maize(quick_test=True):
    _script_dir = Path(__file__).resolve().parent
    output_dir = _script_dir / "results" / "maize_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}  |  Quick test: {quick_test}")
    if DEVICE.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")
    max_markers = 10000 if quick_test else None
    X_all, y_dict = load_iranian_data(max_markers=max_markers)
    traits = sorted(y_dict.keys())
    print(f"\nTraits: {traits}")

    var_thresh = 0.005; vars_per_marker = np.var(X_all, axis=0); keep = vars_per_marker >= var_thresh
    if keep.sum() < X_all.shape[1]: X_all = X_all[:, keep]
    print(f"  Low-variance filter: {X_all.shape[1]} markers kept")

    traits_run = traits[:1] if quick_test else traits
    folds_run = min(2, N_FOLDS) if quick_test else N_FOLDS
    total_t0 = time.time(); all_results = {}

    for trait in traits_run:
        print(f"\n{'='*60}\nTrait: {trait}\n{'='*60}")
        y = y_dict[trait].astype(np.float32)
        n_snps = min(GWAS_TOP_K, max(50, X_all.shape[1] - 50))
        print(f"  {len(y)} samples, {X_all.shape[1]} markers -> {n_snps} GWAS-selected")

        kf = KFold(n_splits=folds_run, shuffle=True, random_state=RANDOM_SEED)
        results = {m: {'preds': [], 'targets': [], 'params': 0, 'time': 0.0} for m in ALL_NAMES}
        oof_trad = {m: np.zeros(len(y)) for m in TRAD_NAMES}
        oof_dl = {m: np.zeros(len(y)) for m in DL_BASE_NAMES}
        dl_models = {mname: create_model(mname, n_snps) for mname in DL_NAMES}

        for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X_all)):
            print(f"\n  --- Fold {fold_i+1}/{folds_run} ---")
            Xtr_raw, Xte_raw = X_all[tr_idx], X_all[te_idx]; ytr, yte = y[tr_idx], y[te_idx]
            maf_idx = maf_filter(Xtr_raw, MAF_THRESHOLD)
            if len(maf_idx) >= n_snps:
                gidx = gwas_select(Xtr_raw[:, maf_idx], ytr, n_snps); gidx = maf_idx[gidx]
            else: gidx = gwas_select(Xtr_raw, ytr, n_snps)
            Xtr = Xtr_raw[:, gidx]; Xte = Xte_raw[:, gidx]
            sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr).astype(np.float32); Xte_s = sc.transform(Xte).astype(np.float32)

            G_fold_train = Xtr_s @ Xtr_s.T / n_snps; G_fold_te_tr = Xte_s @ Xtr_s.T / n_snps

            trad_configs = _make_trad_configs(G_fold_train, G_fold_te_tr, n_snps, len(tr_idx))
            for tname, build_fn, fit_fn, pred_fn, param_count in trad_configs:
                t0 = time.time(); tmodel = build_fn(); fit_fn(tmodel, Xtr_s, ytr)
                preds = pred_fn(tmodel, Xte_s)
                results[tname]['preds'].extend(preds.tolist()); results[tname]['targets'].extend(yte.tolist())
                results[tname]['time'] += time.time() - t0
                if fold_i == 0: results[tname]['params'] = param_count
                oof_trad[tname][te_idx] = preds
                print(f"    {tname:<16s} R²={r2_score(yte, preds):+.4f}")

            # DL
            for mi, mname in enumerate(DL_NAMES):
                if fold_i == 0: results[mname]['params'] = sum(p.numel() for p in dl_models[mname].parameters())
                t0 = time.time()
                if mname == 'PreFGN' and not quick_test:
                    pretrain_prefgn_v2(dl_models[mname], Xtr_s, epochs=50, lr=1e-3, patience=10)
                if mname == 'DeepKernelGP':
                    model = train_dkgp(dl_models[mname], Xtr_s, ytr, epochs=200, lr=5e-3, patience=15, verbose=False)
                    model.fit(torch.FloatTensor(Xtr_s), torch.FloatTensor(ytr))
                    mean, _ = model.predict(torch.FloatTensor(Xte_s)); preds = mean.cpu().numpy()
                else:
                    model = train_torch_model(dl_models[mname], Xtr_s, ytr, epochs=300, batch_size=128, lr=2e-3, weight_decay=1e-3, patience=30)
                    preds = predict_torch_model(model, Xte_s)
                elapsed = time.time() - t0
                results[mname]['preds'].extend(preds.tolist()); results[mname]['targets'].extend(yte.tolist()); results[mname]['time'] += elapsed
                print(f"    {mname:<16s} R²={r2_score(yte, preds):+.4f}  ({elapsed:.1f}s)")
                if mname in DL_BASE_NAMES: oof_dl[mname][te_idx] = preds
                dl_models[mname] = create_model(mname, n_snps)

            # ResFGN
            t0 = time.time()
            ridge, res_model = fit_resfgn_components(Xtr_s, ytr, n_snps, cv=3)
            pred_te_ridge = ridge.predict(Xte_s); res_pred = predict_torch_model(res_model, Xte_s)
            final_pred = pred_te_ridge + res_pred; elapsed = time.time() - t0
            if fold_i == 0: results['ResFGN']['params'] = (n_snps+1) + sum(p.numel() for p in res_model.parameters())
            results['ResFGN']['preds'].extend(final_pred.tolist()); results['ResFGN']['targets'].extend(yte.tolist()); results['ResFGN']['time'] += elapsed
            print(f"    {'ResFGN':<16s} R²={r2_score(yte, final_pred):+.4f}  ({elapsed:.1f}s)")
            torch.cuda.empty_cache()

        # Summary
        trait_res = {}
        for mname in ALL_NAMES:
            p = np.array(results[mname]['preds']); t = np.array(results[mname]['targets'])
            r2_v = float(r2_score(t, p)); corr_v = float(pearsonr(t, p)[0]); rmse_v = float(np.sqrt(np.mean((p-t)**2)))
            mtype = _model_type(mname)
            trait_res[mname] = {'R2': r2_v, 'Correlation': corr_v, 'RMSE': rmse_v, 'Type': mtype, 'Time': results[mname]['time']/folds_run}
            print(f"  {mname:<20s} R²={r2_v:+.4f}  Corr={corr_v:+.4f}  RMSE={rmse_v:.4f}")

        _add_stacking_to_results(oof_dl, oof_trad, y, trait_res, folds_run)
        all_results[trait] = trait_res
        with open(output_dir / "ensemble_intermediate.json", 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        if not quick_test:
            deploy_models(X_all, y, None, n_snps, trait, output_dir, None, quick_test)

    if not quick_test:
        _print_final_summary(all_results, traits_run, output_dir, total_t0, 'Maize')
    else:
        print("\nMaize Quick Test Done!")


# ============================================================================
# Section L: Dispatcher
# ============================================================================

if __name__ == '__main__':
    crop = sys.argv[1] if len(sys.argv) > 1 else ''
    full_mode = '--full' in sys.argv

    if crop not in ('wheat', 'rice', 'maize', 'all'):
        print("Usage: python genomic_ensemble.py <wheat|rice|maize|all> [--full]")
        print("  wheat  — Run wheat ensemble pipeline")
        print("  rice   — Run rice ensemble pipeline")
        print("  maize  — Run maize ensemble pipeline")
        print("  all    — Run all three pipelines")
        print("  --full — Full mode (all traits x 5 folds)")
        sys.exit(1)

    print(f"\n{'#'*80}")
    print(f"  Genomic Prediction Ensemble — {crop.upper()}")
    print(f"  Mode: {'FULL' if full_mode else 'QUICK TEST'}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")

    if crop in ('wheat', 'all'): run_wheat(quick_test=not full_mode)
    if crop in ('rice', 'all'): run_rice(quick_test=not full_mode)
    if crop in ('maize', 'all'): run_maize(quick_test=not full_mode)

    print(f"\n{'#'*80}")
    print(f"  All done! Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")
