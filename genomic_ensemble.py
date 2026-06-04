#!/usr/bin/env python
"""
Genomic Prediction Ensemble — Consolidated Self-Contained Script
==================================================================
Wheat + Wheat2000 + Rice + Maize + Soybean + WheatGABI pipelines bundled
into one file for HPC deployment.  No local imports — all model definitions,
training utilities, and shared modules are inlined.

Usage:
  python genomic_ensemble.py wheat --no-plots       # Quick test: 1 trait x 2 folds
  python genomic_ensemble.py wheat --full           # Full: all traits x 5 folds
  python genomic_ensemble.py all --full             # Run all six datasets
  python genomic_ensemble.py plot --plot-only       # Regenerate figures only
"""

import json, time, os, sys, random, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from scipy.stats import pearsonr
import xgboost as xgb

# Fix Windows console encoding for Unicode characters (R², Δ, etc.)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

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

# Project root — auto-detect local vs HPC
_PROJECT_DIR_HPC = "/storage/public/home/2024110093/genomic_prediction"
PROJECT_DIR = _PROJECT_DIR_HPC if os.path.isdir(_PROJECT_DIR_HPC) else os.path.dirname(os.path.abspath(__file__))

# Wheat data paths
WHEAT_DATA_BASE = "/storage/public/home/2024110093/data/Variation/CSIAAS/"
WHEAT_SNP_VCF   = WHEAT_DATA_BASE + "Core819Samples_snp.filter.final.id_gt.813m.vcf.gz"
WHEAT_INDEL_VCF = WHEAT_DATA_BASE + "Core819Samples_indel.filter.final.id_gt.813m.vcf.gz"
WHEAT_SV_VCF    = WHEAT_DATA_BASE + "SV.new.vcf.gz"
WHEAT_PHENO     = WHEAT_DATA_BASE + "Phe.txt"
WHEAT_VCF_ID    = WHEAT_DATA_BASE + "VCFID.txt"
WHEAT_MAX_VARIANTS_PER_TYPE = 15000

# Wheat2000 CSV data paths (dnngp format: binary marker matrix + per-trait phenotype files)
WHEAT2000_DATA_DIR = PROJECT_DIR + "/dnngp_data/wheat2000/SNP_origin"
WHEAT2000_GENO = WHEAT2000_DATA_DIR + "/2000gene.csv"
WHEAT2000_TRAITS = {
    'tkw':    '2000_1_phe.txt',   # 千粒重 (thousand kernel weight)
    'testw':  '2000_2_phe.txt',   # 容重 (test weight)
    'length': '2000_3_phe.txt',   # 粒长 (kernel length)
    'width':  '2000_4_phe.txt',   # 粒宽 (kernel width)
    'Hard':   '2000_5_phe.txt',   # 硬度 (hardness)
    'Prot':   '2000_6_phe.txt',   # 蛋白质含量 (protein content)
}

# Rice data paths
RICE_DATA_DIR = PROJECT_DIR + "/results/rice_data"

# Maize data paths
MAIZE_DATA_DIR = PROJECT_DIR + "/data2"

# Soybean data paths (EasyGeSe benchmark — SoySNP50K chip)
SOYBEAN_DATA_DIR = PROJECT_DIR + "/results/soybean_SoySNP50K"
SOYBEAN_TRAIT_NAMES = ['Canopy_wilting', 'Water_use_efficiency']

# Wheat GABI data paths (EasyGeSe benchmark — iSELECT 90k chip)
WHEAT_GABI_DATA_DIR = PROJECT_DIR + "/results/wheat_GABI"

# Seed / deterministic helpers
def set_global_seed(seed=RANDOM_SEED):
    """Set all process-local RNGs used by this script."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_global_seed(RANDOM_SEED)
# PYTHONHASHSEED must be set at process launch: export PYTHONHASHSEED=42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _stable_seed(*parts, base=RANDOM_SEED):
    """Stable 31-bit seed independent of Python's hash randomization."""
    text = '|'.join(str(p) for p in (base, *parts))
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16) % (2**31 - 1)


def _as_model_input(X):
    """Return genotype features in the canonical no-scaler representation."""
    return np.asarray(X, dtype=np.float32)


def _safe_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    val = pearsonr(y_true, y_pred)[0]
    return 0.0 if np.isnan(val) else float(val)


def _metric_values(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return (float(r2_score(y_true, y_pred)),
            _safe_corr(y_true, y_pred),
            float(np.sqrt(np.mean((y_pred - y_true) ** 2))))

TYPE_TRAD = 'Traditional'
TYPE_DL = 'DL'
TYPE_ENS = 'Ensemble'

def _model_type(mname):
    if mname in ('Stacking (DL)', 'Stacking (All)', 'Trad Ensemble',
                 'Stacking (Pruned)', 'Stacking (Greedy)', 'Stacking (R²+Greedy)'): return TYPE_ENS
    if mname in ('RRBLUP', 'GBLUP', 'XGBoost', 'ElasticNet', 'GWAS_RRBLUP'): return TYPE_TRAD
    return TYPE_DL

# ============================================================================
# Section A: Shared NN Utilities (from genomic_nn_models.py)
# ============================================================================



# ============================================================================
# Section C: Haplotype Scoring (from haplotype_scoring.py)
# ============================================================================
DEFAULT_WINDOW = 50
DEFAULT_R2_THRESH = 0.6
MIN_MAF = 1e-4


def _compute_univariate_effects(X, y):
    y_c = y - y.mean(); X_c = X - X.mean(axis=0, keepdims=True)
    var_x = np.maximum(X_c.var(axis=0), 1e-8)
    cov_xy = X_c.T @ y_c / len(y_c)
    return np.abs(cov_xy / var_x)


def compute_haplotype_scores(X, y, variant_types=None, maf=None):
    """变异类型不参与打分, 仅传回 components 供后验富集分析"""
    p = X.shape[1]
    if maf is None:
        af = X.mean(axis=0) / 2.0
        maf = np.minimum(af, 1.0 - af)
    rarity_w = np.maximum(-np.log10(np.maximum(maf, MIN_MAF)), 1.0)
    effects = _compute_univariate_effects(X, y)
    scores = rarity_w * (1.0 + effects)
    return scores, {'rarity': rarity_w, 'effect': effects,
                    'variant_types': variant_types}


def eb_shrink_effects(effects, maf, n_bins=20):
    """经验贝叶斯效应收缩 (GeneBayes 启发, s41588-024-01820-9)"""
    p = len(effects); maf = np.asarray(maf, dtype=np.float64)
    effects = np.asarray(effects, dtype=np.float64)
    log_maf = np.log10(np.maximum(maf, MIN_MAF))
    bins = np.percentile(log_maf, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-8; bins[-1] += 1e-8
    bin_idx = np.clip(np.digitize(log_maf, bins) - 1, 0, n_bins - 1)
    shrunk = np.zeros(p)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 5: shrunk[mask] = effects[mask]; continue
        group_mean = np.mean(effects[mask]); group_var = np.var(effects[mask])
        if group_var < 1e-12: shrunk[mask] = group_mean; continue
        data_precision = maf[mask] * (1.0 - maf[mask]) + 1e-8
        data_noise = 1.0 / data_precision; avg_noise = np.mean(data_noise)
        signal_var = max(group_var - avg_noise, 0.0)
        lam = data_noise / (data_noise + signal_var + 1e-12)
        shrunk[mask] = (1.0 - lam) * effects[mask] + lam * group_mean
    return np.abs(shrunk)


def ld_aware_scores(X, raw_scores, window=DEFAULT_WINDOW, r2_thresh=DEFAULT_R2_THRESH):
    """LD 感知软加权, O(p×window) 用位置字典"""
    n, p = X.shape
    X_c = X - X.mean(axis=0, keepdims=True)
    X_n = X_c / (np.linalg.norm(X_c, axis=0, keepdims=True) + 1e-12)
    order = np.argsort(-raw_scores)
    adjusted = raw_scores.copy().astype(np.float64)
    pos_map = {}
    for idx in order:
        max_r2 = 0.0; xj = X_n[:, idx]
        for pos in range(max(0, idx - window), min(p, idx + window + 1)):
            if pos in pos_map:
                _, px = pos_map[pos]
                r2 = np.dot(xj, px) ** 2
                if r2 > max_r2: max_r2 = r2
        if max_r2 > r2_thresh:
            adjusted[idx] *= max(1.0 - np.sqrt(max_r2), 0.01)
        pos_map[idx] = (idx, xj)
    return adjusted


def haplotype_select_eb(X, y, k, variant_types=None, window=DEFAULT_WINDOW,
                        r2_thresh=DEFAULT_R2_THRESH, n_eb_bins=20):
    """GeneBayes 增强版: EB收缩 + LD软加权"""
    af = X.mean(axis=0) / 2.0; maf = np.minimum(af, 1.0 - af)
    effects_raw = _compute_univariate_effects(X, y)
    effects_shrunk = eb_shrink_effects(effects_raw, maf, n_bins=n_eb_bins)
    rarity_w = np.maximum(-np.log10(np.maximum(maf, MIN_MAF)), 1.0)
    raw_scores = rarity_w * (1.0 + effects_shrunk)
    n_cand = min(3 * k, X.shape[1])
    cand_idx = np.argsort(-raw_scores)[:n_cand]
    X_cand = X[:, cand_idx]; scores_cand = raw_scores[cand_idx]
    adjusted = ld_aware_scores(X_cand, scores_cand, window, r2_thresh)
    return np.sort(cand_idx[np.argsort(-adjusted)[:k]])


def ld_prune_markers(X, scores, window=DEFAULT_WINDOW, r2_thresh=DEFAULT_R2_THRESH):
    """LD 剪枝, O(p×window) 用位置集合"""
    n, p = X.shape
    X_c = X - X.mean(axis=0, keepdims=True)
    X_n = X_c / (np.linalg.norm(X_c, axis=0, keepdims=True) + 1e-12)
    order = np.argsort(-scores); kept = []; kept_set = set()
    for idx in order:
        redundant = False; xj = X_n[:, idx]
        for pos in range(max(0, idx - window), min(p, idx + window + 1)):
            if pos in kept_set:
                if np.dot(xj, X_n[:, pos]) ** 2 > r2_thresh:
                    redundant = True; break
        if not redundant: kept.append(idx); kept_set.add(idx)
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
    """FGN — FFT spectral + time-domain dual path."""
    def __init__(self, n_snps, hidden=64, dropout=0.35, input_dropout=0.0, n_spec=32, droppath=0.0):
        super().__init__()
        self.n_freq = n_snps // 2 + 1
        self.input_dropout = input_dropout
        self.droppath = droppath
        self.n_spec = n_spec
        self.spec_r = nn.Parameter(torch.randn(1, n_spec, self.n_freq) * 0.02)
        self.spec_i = nn.Parameter(torch.randn(1, n_spec, self.n_freq) * 0.02)
        self.freq_conv = nn.Sequential(
            nn.Conv1d(n_spec, hidden, 7, padding=3), nn.BatchNorm1d(hidden), nn.GELU(),
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
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        xc = torch.fft.rfft(x, dim=1)
        xr = xc.real.unsqueeze(1).expand(-1, self.n_spec, -1) * self.spec_r
        xi = xc.imag.unsqueeze(1).expand(-1, self.n_spec, -1) * self.spec_i
        fp = self.pool(self.freq_conv(xr + xi)).squeeze(-1)
        tp = self.pool(self.time_conv(x.unsqueeze(1))).squeeze(-1)
        if self.training and self.droppath > 0:
            r = torch.rand(1, device=x.device).item()
            if r < self.droppath:
                fp = fp * 0
            elif r < self.droppath * 2:
                tp = tp * 0
        return self.head(torch.cat([fp, tp], dim=1))


class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(ch, ch//r, 1),
                                nn.GELU(), nn.Conv1d(ch//r, ch, 1), nn.Sigmoid())
    def forward(self, x):
        return x * self.se(x)


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


class _SpectralBranch(nn.Module):
    """Separate real/imag FFT conv paths + wavelet + time — shared by FGNv4/AdditiveGenomicNet.

    If max_freq is set (< n_freq), only the first max_freq frequency components are used,
    acting as low-pass filtering that removes high-frequency noise. This is biologically
    motivated: LD blocks span dozens to hundreds of SNPs, corresponding to low frequencies.
    """
    def __init__(self, n_freq, n_spec, ch, dropout, max_freq=None):
        super().__init__()
        self.n_spec = n_spec
        self.max_freq = min(max_freq or n_freq, n_freq)  # clamp to FFT output size
        self.spec_r = nn.Parameter(torch.randn(1, n_spec, self.max_freq) * 0.02)
        self.spec_i = nn.Parameter(torch.randn(1, n_spec, self.max_freq) * 0.02)
        self.freq_conv_r = nn.Sequential(
            nn.Conv1d(n_spec, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.freq_conv_i = nn.Sequential(
            nn.Conv1d(n_spec, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.wavelet = HaarWaveletDecomp()
        self.wavelet_conv = nn.Sequential(
            nn.Conv1d(2, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.time_conv = nn.Sequential(
            nn.Conv1d(1, ch, 21, padding=10), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        xc = torch.fft.rfft(x, dim=1)[:, :self.max_freq]
        xr = xc.real.unsqueeze(1).expand(-1, self.n_spec, -1) * self.spec_r
        xi = xc.imag.unsqueeze(1).expand(-1, self.n_spec, -1) * self.spec_i
        fp_r = self.pool(self.freq_conv_r(xr)).squeeze(-1)
        fp_i = self.pool(self.freq_conv_i(xi)).squeeze(-1)
        cA, cD = self.wavelet(x)
        wp = self.pool(self.wavelet_conv(torch.cat([cA, cD], dim=1))).squeeze(-1)
        tp = self.pool(self.time_conv(x.unsqueeze(1))).squeeze(-1)
        return fp_r, fp_i, wp, tp


class FGNv4(nn.Module):
    """FGN v4: Complex-aware spectral + wavelet + time + colsample dropout"""
    def __init__(self, n_snps, hidden=64, dropout=0.35, input_dropout=0.0, n_spec=24):
        super().__init__()
        ch = hidden // 2
        self.input_dropout = input_dropout
        self.spec = _SpectralBranch(n_snps // 2 + 1, n_spec, ch, dropout)
        self.head = nn.Sequential(
            nn.Linear(ch * 4, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        return self.head(torch.cat(self.spec(x), dim=1))


class FGNv5(nn.Module):
    """FGN v5: Complex-aware spectral + multi-scale time conv (k=7,31,101).

    Extends v4's proven complex spectral branch with three additional parallel
    time-domain paths at different kernel widths, capturing LD at short/medium/
    long ranges. The original time conv (k=21) is retained alongside.
    """
    def __init__(self, n_snps, hidden=64, dropout=0.35, input_dropout=0.0):
        super().__init__()
        ch = hidden // 2
        self.input_dropout = input_dropout
        self.spec = _SpectralBranch(n_snps // 2 + 1, 24, ch, dropout)
        # Additional multi-scale time conv paths (spec already has k=21)
        self.time_k7 = nn.Sequential(
            nn.Conv1d(1, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.time_k51 = nn.Sequential(
            nn.Conv1d(1, ch, 51, padding=25), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.time_k101 = nn.Sequential(
            nn.Conv1d(1, ch, 101, padding=50), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.pool = nn.AdaptiveAvgPool1d(1)
        total_ch = ch * 7  # fp_r, fp_i, wp, tp(k21), tp_k7, tp_k51, tp_k101
        self.head = nn.Sequential(
            nn.Linear(total_ch, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        fp_r, fp_i, wp, tp_orig = self.spec(x)
        x_u = x.unsqueeze(1)
        tp_k7 = self.pool(self.time_k7(x_u)).squeeze(-1)
        tp_k51 = self.pool(self.time_k51(x_u)).squeeze(-1)
        tp_k101 = self.pool(self.time_k101(x_u)).squeeze(-1)
        return self.head(torch.cat([fp_r, fp_i, wp, tp_orig, tp_k7, tp_k51, tp_k101], dim=1))


class FGNv6(nn.Module):
    """FGN v6: Low-frequency focused complex spectral + wavelet + time.

    Key innovation: truncates FFT to only low frequencies (max_freq=384 for
    5000 SNPs). This removes high-frequency noise that corresponds to single-
    SNP fluctuations, keeping only patterns spanning ≥13 SNPs — matching
    typical LD block sizes. Benefits:
    - 85% fewer spectral weight params → less overfitting
    - Built-in denoising via low-pass filtering
    - Forces model to learn from genome-wide trends instead of individual SNPs
    """
    def __init__(self, n_snps, hidden=64, dropout=0.35, input_dropout=0.0, max_freq=384):
        super().__init__()
        ch = hidden // 2
        self.input_dropout = input_dropout
        self.spec = _SpectralBranch(n_snps // 2 + 1, 24, ch, dropout, max_freq=max_freq)
        self.head = nn.Sequential(
            nn.Linear(ch * 4, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        return self.head(torch.cat(self.spec(x), dim=1))


class FGNv7(nn.Module):
    """FGN v7: FGN v4 + learned per-SNP importance weights.

    Key innovation: before FFT, genotype is multiplied by learned SNP weights
    (sigmoid-bounded). This gives the model explicit feature selection capability
    — important SNPs get weight near 1.0, noisy SNPs near 0.0. This mimics
    XGBoost's implicit feature selection via tree splits, but in a differentiable
    way compatible with spectral processing.

    Weight initialization is near 1.0 (sigmoid(2)=0.88) so the model starts
    using all SNPs and gradually down-weights noise.
    """
    def __init__(self, n_snps, hidden=64, dropout=0.35, input_dropout=0.0):
        super().__init__()
        ch = hidden // 2
        self.input_dropout = input_dropout
        self.snp_weight = nn.Parameter(torch.full((1, n_snps), 2.0))
        self.spec = _SpectralBranch(n_snps // 2 + 1, 24, ch, dropout)
        self.head = nn.Sequential(
            nn.Linear(ch * 4, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        w = torch.sigmoid(self.snp_weight)
        return self.head(torch.cat(self.spec(x * w), dim=1))


class FGNv8(nn.Module):
    """FGN v8: Adaptive linear+spectral fusion with learnable mixing coefficient.

    Core insight: high-heritability traits need mostly linear (additive) modeling;
    low-heritability traits benefit from FGN's spectral bias. Instead of hard-coding
    the balance, α is learned per-trait via sigmoid gating.

    - α → 1: mostly linear, spectral path suppressed (high-h² traits)
    - α → 0: mostly spectral, linear path suppressed (low-h² traits)
    """
    def __init__(self, n_snps, hidden=64, dropout=0.35, input_dropout=0.0):
        super().__init__()
        ch = hidden // 2
        self.input_dropout = input_dropout
        # Strong additive path — captures linear SNP effects
        self.additive = nn.Linear(n_snps, 1, bias=False)
        nn.init.normal_(self.additive.weight, std=1.0 / np.sqrt(n_snps))
        # Spectral path — FGN v4 complex-aware branch
        self.spec = _SpectralBranch(n_snps // 2 + 1, 24, ch, dropout)
        self.spec_head = nn.Sequential(
            nn.Linear(ch * 4, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
        # Learnable mixing coefficient (sigmoid → bounded (0,1))
        # Init at 2.5 → sigmoid≈0.92, favoring linear path initially
        self.logit_alpha = nn.Parameter(torch.tensor([2.5]))

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        lin = self.additive(x)
        spec = self.spec_head(torch.cat(self.spec(x), dim=1))
        alpha = torch.sigmoid(self.logit_alpha)
        return alpha * lin + (1.0 - alpha) * spec


class FGNv9(nn.Module):
    """FGN v9: DCT-based spectral processing + wavelet + time.

    Replaces FFT with Discrete Cosine Transform (DCT-II). Motivation:
    - FFT assumes periodic boundary (SNPs on chr1 wrap to chrN) → artificial
      high-frequency noise at chromosome boundaries
    - DCT assumes symmetric boundary → no artificial discontinuities
    - DCT has better energy compaction for smooth signals → more of the genomic
      signal is concentrated in fewer coefficients
    - DCT is purely real → simpler spectral path (no real/imag split needed)

    Uses n_dct=384 coefficients (matching v6's max_freq) for built-in denoising.
    """
    def __init__(self, n_snps, hidden=64, dropout=0.35, input_dropout=0.0, n_dct=384):
        super().__init__()
        ch = hidden // 2
        self.input_dropout = input_dropout
        # Pre-compute DCT-II basis matrix
        n = torch.arange(n_snps).float()
        k = torch.arange(n_dct).float().unsqueeze(1)
        dct_basis = torch.cos(np.pi * k * (n + 0.5) / n_snps)
        # Normalize: DCT-II standard scaling
        dct_basis[0] *= 1.0 / np.sqrt(2)
        dct_basis *= np.sqrt(2.0 / n_snps)
        self.register_buffer('dct_basis', dct_basis)  # (n_dct, n_snps)
        self.n_spec = 24
        self.spec_weight = nn.Parameter(torch.randn(1, self.n_spec, n_dct) * 0.02)
        self.freq_conv = nn.Sequential(
            nn.Conv1d(self.n_spec, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.wavelet = HaarWaveletDecomp()
        self.wavelet_conv = nn.Sequential(
            nn.Conv1d(2, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.time_conv = nn.Sequential(
            nn.Conv1d(1, ch, 21, padding=10), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(ch * 3, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x_dct = x @ self.dct_basis.T  # (B, n_snps) @ (n_snps, n_dct) → (B, n_dct)
        x_spec = x_dct.unsqueeze(1).expand(-1, self.n_spec, -1) * self.spec_weight
        fp = self.pool(self.freq_conv(x_spec)).squeeze(-1)
        cA, cD = self.wavelet(x)
        wp = self.pool(self.wavelet_conv(torch.cat([cA, cD], dim=1))).squeeze(-1)
        tp = self.pool(self.time_conv(x.unsqueeze(1))).squeeze(-1)
        return self.head(torch.cat([fp, wp, tp], dim=1))


class FGNv10(nn.Module):
    """FGN v10: FGN v7 SNP attention + explicit additive linear path.

    The additive path (Linear n_snps→1) uses RAW SNPs without attention weighting,
    directly capturing pure additive effects the way RRBLUP/XGBoost do. The spectral
    paths use attention-weighted SNPs for non-additive signal. This dual-path design
    targets the gap between FGN and XGBoost on high-heritability traits.
    """
    def __init__(self, n_snps, hidden=64, dropout=0.35, input_dropout=0.0):
        super().__init__()
        ch = hidden // 2
        self.input_dropout = input_dropout
        self.snp_weight = nn.Parameter(torch.full((1, n_snps), 2.0))
        self.additive = nn.Linear(n_snps, 1, bias=False)
        nn.init.normal_(self.additive.weight, std=1.0 / np.sqrt(n_snps))
        self.spec = _SpectralBranch(n_snps // 2 + 1, 24, ch, dropout)
        self.head = nn.Sequential(
            nn.Linear(ch * 4 + 1, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        w = torch.sigmoid(self.snp_weight)
        add_out = self.additive(x)  # raw SNPs for pure additive signal
        spec_out = self.spec(x * w)  # attention-weighted for spectral
        return self.head(torch.cat([*spec_out, add_out], dim=1))


class FGNv11(nn.Module):
    """FGN v11: Dual spectral paths — FFT + DCT — in a single model.

    FFT captures sharp transitions (periodic boundary), DCT captures smooth trends
    (symmetric boundary). Stacking evidence shows they provide complementary signals.
    Four paths: FFT combined (real+imag), DCT, wavelet, time-domain conv.
    """
    def __init__(self, n_snps, hidden=64, dropout=0.35, input_dropout=0.0, n_dct=384):
        super().__init__()
        ch = hidden // 2
        self.input_dropout = input_dropout
        n_freq = n_snps // 2 + 1

        # --- FFT path ---
        n_spec = 24
        self.spec_r = nn.Parameter(torch.randn(1, n_spec, n_freq) * 0.02)
        self.spec_i = nn.Parameter(torch.randn(1, n_spec, n_freq) * 0.02)
        self.fft_conv = nn.Sequential(
            nn.Conv1d(n_spec * 2, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))

        # --- DCT path ---
        n = torch.arange(n_snps).float()
        k = torch.arange(n_dct).float().unsqueeze(1)
        dct_basis = torch.cos(np.pi * k * (n + 0.5) / n_snps)
        dct_basis[0] *= 1.0 / np.sqrt(2)
        dct_basis *= np.sqrt(2.0 / n_snps)
        self.register_buffer('dct_basis', dct_basis)  # (n_dct, n_snps)
        self.n_spec_dct = 24
        self.spec_weight_dct = nn.Parameter(torch.randn(1, self.n_spec_dct, n_dct) * 0.02)
        self.dct_conv = nn.Sequential(
            nn.Conv1d(self.n_spec_dct, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))

        # --- Wavelet path ---
        self.wavelet = HaarWaveletDecomp()
        self.wavelet_conv = nn.Sequential(
            nn.Conv1d(2, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))

        # --- Time-domain path ---
        self.time_conv = nn.Sequential(
            nn.Conv1d(1, ch, 21, padding=10), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout * 0.5))

        self.pool = nn.AdaptiveAvgPool1d(1)
        # 4 paths × ch = 4×32 = 128 for hidden=64
        self.head = nn.Sequential(
            nn.Linear(ch * 4, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)

        # FFT
        x_fft = torch.fft.rfft(x, dim=1)
        real, imag = x_fft.real, x_fft.imag
        sr = real.unsqueeze(1).expand(-1, 24, -1) * self.spec_r
        si = imag.unsqueeze(1).expand(-1, 24, -1) * self.spec_i
        fft_feat = self.pool(self.fft_conv(torch.cat([sr, si], dim=1))).squeeze(-1)

        # DCT
        x_dct = x @ self.dct_basis.T
        x_dct = x_dct.unsqueeze(1).expand(-1, self.n_spec_dct, -1) * self.spec_weight_dct
        dct_feat = self.pool(self.dct_conv(x_dct)).squeeze(-1)

        # Wavelet
        cA, cD = self.wavelet(x)
        wv_feat = self.pool(self.wavelet_conv(torch.cat([cA, cD], dim=1))).squeeze(-1)

        # Time conv
        tp_feat = self.pool(self.time_conv(x.unsqueeze(1))).squeeze(-1)

        return self.head(torch.cat([fft_feat, dct_feat, wv_feat, tp_feat], dim=1))


class FGNplus(nn.Module):
    """Improved FGN: additive skip + input dropout on top of FGN architecture.

    Mirrors FGN's spectral+time paths exactly, but adds:
    - Additive skip connection preserves linear SNP effects
    - Input dropout acts as colsample regularization
    """
    def __init__(self, n_snps, hidden=64, dropout=0.35, input_dropout=0.0):
        super().__init__()
        self.n_freq = n_snps // 2 + 1
        self.input_dropout = input_dropout

        # Additive path — captures linear SNP effects (same as RRBLUP)
        self.additive = nn.Linear(n_snps, 1, bias=False)
        nn.init.normal_(self.additive.weight, std=1.0 / np.sqrt(n_snps))

        # Spectral path: same as FGN (real+imag combined)
        self.spec_r = nn.Parameter(torch.randn(1, 32, self.n_freq) * 0.02)
        self.spec_i = nn.Parameter(torch.randn(1, 32, self.n_freq) * 0.02)
        self.freq_conv = nn.Sequential(
            nn.Conv1d(32, hidden, 7, padding=3), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.time_conv = nn.Sequential(
            nn.Conv1d(1, hidden, 21, padding=10), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        x_do = F.dropout(x, p=self.input_dropout, training=self.training)
        add_out = self.additive(x_do)
        xc = torch.fft.rfft(x_do, dim=1)
        xr = xc.real.unsqueeze(1).expand(-1, 32, -1) * self.spec_r
        xi = xc.imag.unsqueeze(1).expand(-1, 32, -1) * self.spec_i
        fp = self.pool(self.freq_conv(xr + xi)).squeeze(-1)
        tp = self.pool(self.time_conv(x_do.unsqueeze(1))).squeeze(-1)
        return add_out + self.head(torch.cat([fp, tp], dim=1))


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


class AdditiveGenomicNet(nn.Module):
    """End-to-end additive: linear SNP effects + nonlinear spectral residual."""
    def __init__(self, n_snps, hidden=48, dropout=0.35, input_dropout=0.0):
        super().__init__()
        ch = hidden // 2
        self.input_dropout = input_dropout
        self.additive = nn.Linear(n_snps, 1, bias=False)
        self.spec = _SpectralBranch(n_snps // 2 + 1, 16, ch, dropout)
        self.nonlinear_head = nn.Sequential(
            nn.Linear(ch * 4, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        x_do = F.dropout(x, p=self.input_dropout, training=self.training)
        lin = self.additive(x_do)
        nonlin = self.nonlinear_head(torch.cat(self.spec(x_do), dim=1))
        return lin + nonlin


# ============================================================================
# Section D2: GenomicFM — Factorization Machine for Genomic Prediction
# ============================================================================

class GenomicFM(nn.Module):
    """Linear + learned embedding + MLP for genomic prediction.

    Linear path captures additive SNP effects; a small MLP on learned
    feature embeddings captures nonlinear interactions.
    ~5500 params at k=4 for n_snps=1230.
    """
    def __init__(self, n_snps, k=4, dropout=0.2, mlp_hidden=16):
        super().__init__()
        self.linear = nn.Linear(n_snps, 1)         # additive SNP effects
        self.V = nn.Parameter(torch.randn(n_snps, k))  # SNP embeddings, scaled in forward
        self.mlp = nn.Sequential(
            nn.Linear(k, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1))

    def forward(self, x):
        fm_linear = self.linear(x)                       # (batch, 1)
        scale = self.V.shape[0] ** 0.5
        xv = (x @ self.V) / scale                         # (batch, k), scaled to unit variance
        mlp_out = self.mlp(xv)                            # (batch, 1)
        return fm_linear + mlp_out


class FGN_PCA(nn.Module):
    """FGN-PCA: PCA-optimized FGN — direct projection + conv + FM interactions."""
    def __init__(self, n_features, hidden=64, dropout=0.35, fm_k=4):
        super().__init__()
        self.direct_proj = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.conv_path = nn.Sequential(
            nn.Conv1d(1, hidden, 21, padding=10), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout * 0.5))
        self.V = nn.Parameter(torch.randn(n_features, fm_k) * 0.01)
        self.fm_scale = nn.Parameter(torch.tensor(0.5))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + fm_k, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        dp = self.direct_proj(x)
        cp = self.pool(self.conv_path(x.unsqueeze(1))).squeeze(-1)
        scale = x.shape[1] ** 0.5
        fm_out = 0.5 * ((x @ self.V) ** 2 - (x ** 2) @ (self.V ** 2)) / scale
        fm_out = fm_out * torch.tanh(self.fm_scale)
        return self.head(torch.cat([dp, cp, fm_out], dim=1))


# ============================================================================
# Section E: Bagged TinyNet Ensemble (Random-Forest style)
# ============================================================================

class WheatGPModel(nn.Module):
    """WheatGP: 5-slice CNN + LSTM — from WheatGP paper (2022)."""
    CNN_OUT_CH = 8
    POOL_SIZE = 16

    def __init__(self, n_features, n_subnetworks=5, hidden_dim=128):
        super().__init__()
        if n_features < n_subnetworks:
            raise ValueError(f"n_features ({n_features}) < n_subnetworks ({n_subnetworks})")
        self.n_subnetworks = n_subnetworks
        self.chunk_size = n_features // n_subnetworks

        self.subnetworks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(2, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(4, self.CNN_OUT_CH, kernel_size=9, padding=4),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.AdaptiveAvgPool1d(self.POOL_SIZE)
            )
            for _ in range(n_subnetworks)
        ])

        lstm_in = self.CNN_OUT_CH * self.POOL_SIZE * n_subnetworks
        self.lstm = nn.LSTM(lstm_in, hidden_dim, batch_first=True)
        self.lstm_drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)
        out_dim = self.CNN_OUT_CH * self.POOL_SIZE
        outputs = []
        for i, subnet in enumerate(self.subnetworks):
            start = i * self.chunk_size
            end = (start + self.chunk_size if i < self.n_subnetworks - 1
                   else x.size(1))
            chunk = x[:, start:end].unsqueeze(1)
            outputs.append(subnet(chunk).view(batch_size, out_dim))
        combined = torch.cat(outputs, dim=1).unsqueeze(1)
        lstm_out, _ = self.lstm(combined)
        lstm_out = self.lstm_drop(lstm_out)
        return self.fc(lstm_out[:, -1, :]).squeeze(-1)


class TinySNPNet(nn.Module):
    """Ultra-lightweight network for bagging ensemble — ~200 params."""
    def __init__(self, n_features, hidden=8, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


def train_bagged_ensemble(X_train, y_train, n_estimators=200, n_colsample=50,
                          hidden=8, dropout=0.3, epochs=20, lr=0.01, weight_decay=1e-3):
    """Train a bagged ensemble of TinySNPNets, each on a random feature subset."""
    n_total_features = X_train.shape[1]
    n_colsample = min(n_colsample, n_total_features)
    models = []
    feat_indices = []
    n_total = len(y_train)
    n_val = max(1, int(n_total * 0.15))
    rng = np.random.RandomState(RANDOM_SEED)

    for i in range(n_estimators):
        # Random feature subset (like colsample_bytree)
        feat_idx = rng.choice(n_total_features, n_colsample, replace=False)
        X_sub = X_train[:, feat_idx].astype(np.float32)

        # Random bootstrap sample (like subsample)
        boot_idx = rng.choice(n_total, int(n_total * 0.7), replace=True)
        X_boot, y_boot = X_sub[boot_idx], y_train[boot_idx].astype(np.float32)

        # Train/val split
        idx = rng.permutation(len(y_boot))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xt = torch.FloatTensor(X_boot[tr_idx]).to(DEVICE)
        yt = torch.FloatTensor(y_boot[tr_idx]).to(DEVICE)
        Xv = torch.FloatTensor(X_boot[val_idx]).to(DEVICE)
        yv = torch.FloatTensor(y_boot[val_idx]).to(DEVICE)

        model = TinySNPNet(n_colsample, hidden=hidden, dropout=dropout).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        crit = nn.MSELoss()
        best_state, best_loss, wait = None, float('inf'), 0

        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            loss = crit(model(Xt).squeeze(), yt)
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                vl = crit(model(Xv).squeeze(), yv).item()
            if vl < best_loss:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_loss = vl
                wait = 0
            else:
                wait += 1
                if wait >= 5: break

        model.load_state_dict(best_state)
        model.cpu()
        models.append(model)
        feat_indices.append(feat_idx)

    return models, feat_indices


def predict_bagged_ensemble(models, feat_indices, X):
    """Predict using bagged ensemble — average of all TinySNPNets."""
    preds = np.zeros(len(X), dtype=np.float32)
    for model, feat_idx in zip(models, feat_indices):
        X_sub = X[:, feat_idx].astype(np.float32)
        Xt = torch.FloatTensor(X_sub).to(DEVICE)
        model = model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            p = model(Xt).squeeze().cpu().numpy()
        preds += p
        model.cpu()
    return preds / len(models)


# ============================================================================
# Section F: Common Training Functions
# ============================================================================

def train_torch_model(model, X_train, y_train,
                      epochs=300, batch_size=128, lr=1e-3, weight_decay=1e-4,
                      patience=30, val_ratio=0.15, grad_clip=1.0,
                      use_swa=False, use_mixup=False, mixup_alpha=0.4,
                      label_smooth=0.0, colsample=1.0, l1_lambda=0.0,
                      seed=None):
    if seed is not None:
        set_global_seed(seed)
    model = model.to(DEVICE)
    n_total = len(X_train)
    n_snps = X_train.shape[1]
    n_val = max(1, int(n_total * val_ratio))
    base_seed = RANDOM_SEED if seed is None else int(seed)
    rng = np.random.RandomState(base_seed)
    torch_gen = torch.Generator(device=DEVICE.type if DEVICE.type == 'cuda' else 'cpu')
    torch_gen.manual_seed(base_seed)
    dl_gen = torch.Generator()
    dl_gen.manual_seed(base_seed)
    idx = rng.permutation(n_total)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xt_full = torch.FloatTensor(X_train[tr_idx]).to(DEVICE)
    yt = torch.FloatTensor(y_train[tr_idx]).to(DEVICE)
    Xv_full = torch.FloatTensor(X_train[val_idx]).to(DEVICE)
    yv = torch.FloatTensor(y_train[val_idx]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2, eta_min=lr*0.01)
    crit = nn.MSELoss()
    best_state, best_loss, wait = None, float('inf'), 0
    swa_state, swa_n, swa_start = None, 0, max(1, int(epochs * 0.3))
    for ep in range(epochs):
        model.train()
        # Per-epoch colsample: same feature mask for all samples this epoch
        if colsample < 1.0:
            cmask = torch.rand(n_snps, device=DEVICE, generator=torch_gen) < colsample
            Xt = Xt_full * cmask.float()
            Xv = Xv_full * cmask.float()
        else:
            Xt, Xv = Xt_full, Xv_full
        bs = min(batch_size, len(tr_idx))
        # Avoid batch of size 1 (kills BatchNorm): absorb singleton into previous batch
        while len(tr_idx) % bs == 1 and bs > 1:
            bs += 1
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=True, generator=dl_gen)
        for bx, by in dl:
            if use_mixup and ep >= 5:
                lam = rng.beta(mixup_alpha, mixup_alpha)
                lam = max(lam, 1.0 - lam)
                perm = torch.randperm(bx.size(0), device=DEVICE, generator=torch_gen)
                bx = lam * bx + (1.0 - lam) * bx[perm]
                by = lam * by + (1.0 - lam) * by[perm]
            if label_smooth > 0:
                noise = torch.randn(by.shape, device=by.device, dtype=by.dtype, generator=torch_gen) * label_smooth
                by = by + noise
            opt.zero_grad()
            loss = crit(model(bx).squeeze(), by)
            if l1_lambda > 0:
                for name, param in model.named_parameters():
                    if 'snp_weight' in name:
                        loss = loss + l1_lambda * param.abs().sum()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        model.eval()
        sch.step()
        with torch.no_grad():
            vl = crit(model(Xv).squeeze(), yv).item()
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
        if use_swa and ep >= swa_start:
            if swa_state is None:
                swa_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                swa_n = 1
            else:
                for k in swa_state:
                    swa_state[k] = (swa_state[k] * swa_n + model.state_dict()[k].cpu().clone()) / (swa_n + 1)
                swa_n += 1
    if use_swa and swa_state is not None:
        model.load_state_dict(swa_state)
    else:
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
    def __init__(self, n_estimators=500, max_depth=6, lr=0.05, seed=RANDOM_SEED):
        self.params = {
            'n_estimators': n_estimators, 'max_depth': max_depth,
            'learning_rate': lr, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'random_state': int(seed), 'n_jobs': 8, 'verbosity': 0}
    def fit(self, X, y): self.model = xgb.XGBRegressor(**self.params); self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)


class ElasticNetModel:
    def __init__(self, seed=RANDOM_SEED):
        self.model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                                  alphas=np.logspace(-4, 2, 20),
                                  cv=3, random_state=int(seed), max_iter=5000, n_jobs=8)
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


def random_snp_select(X, n_snps, seed):
    """Randomly select n_snps indices from X columns (deterministic, seed-based)."""
    rng = np.random.RandomState(seed)
    return rng.choice(X.shape[1], min(n_snps, X.shape[1]), replace=False)


def maf_filter(X, threshold=MAF_THRESHOLD):
    af = X.mean(axis=0) / 2.0; maf = np.minimum(af, 1.0 - af)
    return np.where(maf >= threshold)[0]


def _make_candidate_universe(X_raw, vt_all, n_requested):
    """Build the candidate universe from markers that pass the MAF filter.

    The function never falls back to unfiltered markers: if too few markers pass
    MAF, the selected SNP count is reduced instead of reintroducing low-MAF
    variants.  Callers inside CV must pass only the outer training fold.
    """
    requested = min(int(n_requested), X_raw.shape[1])
    maf_idx = maf_filter(X_raw)
    if len(maf_idx) <= 0:
        raise ValueError(
            f"No SNP candidates pass MAF >= {MAF_THRESHOLD:.3f}; "
            "refusing unfiltered fallback")

    X_cand = X_raw[:, maf_idx]
    vt_cand = vt_all[maf_idx] if vt_all is not None else None
    cand_orig_idx = maf_idx
    n_selected = min(requested, X_cand.shape[1])
    if n_selected <= 0:
        raise ValueError("No SNP candidates available after MAF filtering")
    if n_selected < requested:
        print(f"    [WARN] MAF retained {X_cand.shape[1]} markers; "
              f"selected SNP count reduced from {requested} to {n_selected} "
              "without low-MAF fallback")
    return X_cand, vt_cand, cand_orig_idx, n_selected


def _make_fold_candidate_universe(Xtr_raw, Xte_raw, vt_all, n_requested):
    """Build the shared post-MAF candidate universe for one CV fold."""
    Xtr_cand, vt_cand, cand_orig_idx, n_selected = _make_candidate_universe(
        Xtr_raw, vt_all, n_requested)
    Xte_cand = Xte_raw[:, cand_orig_idx]
    return Xtr_cand, Xte_cand, vt_cand, cand_orig_idx, n_selected


def _make_tuning_matrix(X_train_fold, y_train_fold, n_requested):
    """Use fold-train-only GWAS features for an inner hyperparameter search.

    Do not call this on full trait data when reporting outer-CV metrics; the
    MAF filter and GWAS selection must be fitted only inside each outer training
    fold to avoid test-fold label leakage.
    """
    X_cand, _, _, n_selected = _make_candidate_universe(
        X_train_fold, None, n_requested)
    gidx_t = gwas_select(X_cand, y_train_fold, n_selected)
    return _as_model_input(X_cand[:, gidx_t]), n_selected


def _select_dl_markers(Xtr_cand, Xte_cand, ytr, gidx_gwas, vt_cand, n_selected):
    """Select markers for DL models from the same fold candidate universe."""
    if MARKER_SELECTOR == 'haplotype':
        gidx_dl = haplotype_select(Xtr_cand, ytr, n_selected, vt_cand)
    elif MARKER_SELECTOR == 'hybrid':
        gidx_dl = hybrid_select(Xtr_cand, ytr, n_selected, vt_cand, gwas_frac=HAPLO_GWAS_FRAC)
    else:
        gidx_dl = gidx_gwas
    Xtr_dl = _as_model_input(Xtr_cand[:, gidx_dl])
    Xte_dl = _as_model_input(Xte_cand[:, gidx_dl])
    vt_dl = vt_cand[gidx_dl] if vt_cand is not None else None
    return gidx_dl, vt_dl, Xtr_dl, Xte_dl


def _run_fold_random_pass(Xtr_cand, Xte_cand, ytr, yte, te_idx, n_selected, fold_i,
                          results, oof_trad_r, oof_dl_r, tuned_params=None,
                          seed_context=''):
    """Train all models on random SNPs from the same post-MAF candidate universe."""
    gidx_r = random_snp_select(
        Xtr_cand, n_selected,
        seed=_stable_seed(seed_context, fold_i, 'random_snp_selection'))
    Xtr_rs = _as_model_input(Xtr_cand[:, gidx_r])
    Xte_rs = _as_model_input(Xte_cand[:, gidx_r])

    G_fold_train_r = Xtr_rs @ Xtr_rs.T / n_selected
    G_fold_te_tr_r = Xte_rs @ Xtr_rs.T / n_selected

    # Traditional models (random SNPs); paired seed matches GWAS arm.
    trad_seed = _stable_seed(seed_context, fold_i, 'traditional', 'train')
    trad_configs_r = _make_trad_configs(G_fold_train_r, G_fold_te_tr_r, n_selected, len(Xtr_rs), seed=trad_seed)
    for tname, build_fn, fit_fn, pred_fn, param_count in trad_configs_r:
        set_global_seed(_stable_seed(seed_context, fold_i, tname, 'train'))
        t0 = time.time()
        tmodel = build_fn()
        fit_fn(tmodel, Xtr_rs, ytr)
        preds = np.asarray(pred_fn(tmodel, Xte_rs), dtype=np.float32)
        rname = f'{tname}_random'
        results[rname]['preds'].extend(preds.tolist())
        results[rname]['targets'].extend(yte.tolist())
        results[rname]['time'] += time.time() - t0
        if fold_i == 0:
            results[rname]['params'] = param_count
        oof_trad_r[tname][te_idx] = preds

    # DL models (random SNPs); paired seed matches GWAS arm for the same fold/model.
    for mname in DL_NAMES:
        hp = _get_train_hparams(mname, tuned_params)
        train_seed = _stable_seed(seed_context, fold_i, mname, 'train')
        set_global_seed(train_seed)
        model_tp = (tuned_params or {}).get(mname, {})
        model = create_model(mname, n_selected, overrides=model_tp)
        t0 = time.time()
        if fold_i == 0:
            results[f'{mname}_random']['params'] = sum(p.numel() for p in model.parameters())
        if DEVICE.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        model = train_torch_model(model, Xtr_rs, ytr, epochs=300,
                                  batch_size=hp['batch_size'], lr=hp['lr'],
                                  weight_decay=hp['weight_decay'],
                                  patience=hp['patience'], seed=train_seed)
        if DEVICE.type == 'cuda':
            results[f'{mname}_random']['gpu_mem'] = max(
                results[f'{mname}_random']['gpu_mem'],
                torch.cuda.max_memory_allocated() / (1024 * 1024))
        preds = predict_torch_model(model, Xte_rs)
        elapsed = time.time() - t0
        rname = f'{mname}_random'
        results[rname]['preds'].extend(preds.tolist())
        results[rname]['targets'].extend(yte.tolist())
        results[rname]['time'] += elapsed
        print(f"    {rname:<22s} R2={r2_score(yte, preds):+.4f}  ({elapsed:.1f}s)")
        oof_dl_r[mname][te_idx] = preds
        del model
    torch.cuda.empty_cache()

def create_model(name, n_snps, overrides=None):
    o = overrides or {}
    if name == 'FGN': return FourierGenomicNet(n_snps=n_snps, hidden=o.get('hidden', 64), dropout=o.get('dropout', 0.35), input_dropout=o.get('input_dropout', 0.0), n_spec=o.get('n_spec', 32), droppath=o.get('droppath', 0.0))
    if name == 'FGN v2': return FGNv2(n_snps=n_snps, hidden=64, dropout=0.35)
    if name == 'FGN v4': return FGNv4(n_snps=n_snps, hidden=o.get('hidden', 64), dropout=o.get('dropout', 0.35), input_dropout=o.get('input_dropout', 0.0), n_spec=o.get('n_spec', 24))
    if name == 'FGN v5': return FGNv5(n_snps=n_snps, hidden=o.get('hidden', 64), dropout=o.get('dropout', 0.35), input_dropout=o.get('input_dropout', 0.0))
    if name == 'FGN v6': return FGNv6(n_snps=n_snps, hidden=o.get('hidden', 64), dropout=o.get('dropout', 0.35), input_dropout=o.get('input_dropout', 0.0), max_freq=o.get('max_freq', 384))
    if name == 'FGN v7': return FGNv7(n_snps=n_snps, hidden=o.get('hidden', 64), dropout=o.get('dropout', 0.35), input_dropout=o.get('input_dropout', 0.0))
    if name == 'FGN v8': return FGNv8(n_snps=n_snps, hidden=o.get('hidden', 64), dropout=o.get('dropout', 0.35), input_dropout=o.get('input_dropout', 0.0))
    if name == 'FGN v9': return FGNv9(n_snps=n_snps, hidden=o.get('hidden', 64), dropout=o.get('dropout', 0.35), input_dropout=o.get('input_dropout', 0.0), n_dct=o.get('n_dct', 384))
    if name == 'FGN v10': return FGNv10(n_snps=n_snps, hidden=o.get('hidden', 64), dropout=o.get('dropout', 0.35), input_dropout=o.get('input_dropout', 0.0))
    if name == 'FGN v11': return FGNv11(n_snps=n_snps, hidden=o.get('hidden', 64), dropout=o.get('dropout', 0.35), input_dropout=o.get('input_dropout', 0.0), n_dct=o.get('n_dct', 384))
    if name == 'FusionNet': return FusionNet(n_snps=n_snps, hidden_dim=o.get('hidden_dim', 48), dropout=o.get('dropout', 0.35))
    if name == 'AdditiveGenomicNet': return AdditiveGenomicNet(n_snps=n_snps, hidden=o.get('hidden', 48), dropout=o.get('dropout', 0.35), input_dropout=o.get('input_dropout', 0.0))
    if name == 'FGNplus':
        return FGNplus(n_snps=n_snps, hidden=o.get('hidden', 64),
                       dropout=o.get('dropout', 0.35),
                       input_dropout=o.get('input_dropout', 0.0))
    if name == 'FGN PCA': return FGN_PCA(n_features=n_snps, hidden=o.get('hidden', 64), dropout=o.get('dropout', 0.35), fm_k=o.get('fm_k', 4))
    if name == 'GenomicFM':
        return GenomicFM(n_snps=n_snps, k=o.get('k', 4),
                         dropout=o.get('dropout', 0.2),
                         mlp_hidden=o.get('mlp_hidden', 16))
    if name == 'WheatGP':
        return WheatGPModel(n_features=n_snps,
                            n_subnetworks=o.get('n_subnetworks', 5),
                            hidden_dim=o.get('hidden_dim', 128))
    raise ValueError(f"Unknown model: {name}")


RIDGE_ALPHAS = np.logspace(-3, 5, 50)  # wide 50-point log grid for better alpha selection
LASSO_ALPHAS = np.logspace(-4, 2, 30)
ENET_ALPHAS = np.logspace(-4, 2, 20)

TRAD_NAMES = ['RRBLUP', 'GBLUP', 'XGBoost', 'ElasticNet', 'GWAS_RRBLUP']
# FGN v8 is intentionally excluded: it is a documented failed model variant.
DL_BASE_NAMES = ['FGN', 'FGN v2', 'FGN v4', 'FGN v5', 'FGN v6', 'FGN v7', 'FGN v9', 'FGN v10', 'FGN v11', 'FGNplus', 'GenomicFM', 'FGN PCA', 'WheatGP']
DL_NAMES = DL_BASE_NAMES + ['FusionNet', 'AdditiveGenomicNet']
ALL_NAMES = TRAD_NAMES + DL_NAMES

# ---- GWAS vs Random SNP control experiment ----
# Each base model gets a _random twin trained on randomly-selected SNPs
# (same CV folds, same hyperparams, different input features)
ALL_NAMES_RANDOM = [f'{m}_random' for m in ALL_NAMES]
TRAD_NAMES_R = [f'{m}_random' for m in TRAD_NAMES]
DL_NAMES_R = [f'{m}_random' for m in DL_NAMES]
DL_BASE_NAMES_R = [f'{m}_random' for m in DL_BASE_NAMES]

# Per-model batch size overrides (default 128; 32 for FGN-prefixed models)
_MODEL_BS = {'FusionNet': 64, 'AdditiveGenomicNet': 64, 'WheatGP': 32, 'GenomicFM': 32}


def _get_batch_size(mname):
    """Return training batch size for a DL model name."""
    if mname in _MODEL_BS:
        return _MODEL_BS[mname]
    return 32 if mname.startswith('FGN') else 128


def _get_train_hparams(mname, tuned_params=None):
    """Return the canonical DL training hyperparameters for one model."""
    tp = (tuned_params or {}).get(mname, {})
    return {
        'batch_size': _get_batch_size(mname),
        'lr': tp.get('lr', 1e-3 if mname == 'FusionNet' else 2e-3),
        'weight_decay': tp.get('weight_decay', 5e-3 if mname == 'AdditiveGenomicNet' else 1e-3),
        'patience': tp.get('patience', 30),
    }


def _make_trad_configs(G_train, G_te_tr, n_snps, n_train, seed=RANDOM_SEED):
    """Per-fold traditional model configs — GBLUP closures capture G matrices."""
    seed = int(seed)
    return [
        ('RRBLUP', lambda: RRBLUP(), lambda m, Xs, yt: m.fit(Xs, yt), lambda m, Xs: m.predict(Xs), n_snps + 1),
        ('GBLUP', lambda: GBLUP(), lambda m, _x, yt: m.fit(G_train, yt), lambda m, _x: m.predict(G_te_tr), n_train + 1),
        ('XGBoost', lambda: XGBoostModel(n_estimators=300, seed=seed), lambda m, Xs, yt: m.fit(Xs, yt), lambda m, Xs: m.predict(Xs), 300 * 6 * 2),
        ('ElasticNet', lambda: ElasticNetModel(seed=seed), lambda m, Xs, yt: m.fit(Xs, yt), lambda m, Xs: m.predict(Xs), n_snps + 1),
        ('GWAS_RRBLUP', lambda: GWASWeightedRRBLUP(), lambda m, Xs, yt: m.fit(Xs, yt), lambda m, Xs: m.predict(Xs), n_snps * 2 + 1),
    ]


# ============================================================================
# Section H: Stacking / Ensemble Functions (Enhanced)
# ============================================================================

def _prune_correlated(oof_preds_dict, targets, corr_threshold=0.995):
    """Remove redundant base models whose OOF predictions are too correlated.

    For each pair with |r| > corr_threshold, keep the model with higher R².
    Returns filtered dict and list of removed names.
    """
    names = list(oof_preds_dict.keys())
    if len(names) <= 1:
        return oof_preds_dict, []
    preds = np.column_stack([oof_preds_dict[m] for m in names])
    corr = np.corrcoef(preds.T)
    np.fill_diagonal(corr, 0)
    # Rank by individual R² descending
    r2_scores = {m: float(r2_score(targets, oof_preds_dict[m])) for m in names}
    ranked = sorted(names, key=lambda m: r2_scores[m], reverse=True)
    kept, removed = [], set()
    for m in ranked:
        if m in removed:
            continue
        kept.append(m)
        mi = names.index(m)
        for j, other in enumerate(names):
            if other != m and other not in removed and other not in kept:
                if abs(corr[mi, j]) > corr_threshold:
                    removed.add(other)
    pruned = {m: oof_preds_dict[m] for m in kept}
    return pruned, list(removed)


def _filter_by_r2(oof_preds_dict, targets, threshold=0.0):
    """Remove models with R² < threshold before stacking.

    Models with negative R² are worse than predicting the mean — they add pure
    noise to the meta-learner and should be excluded.
    """
    kept = {}
    removed = []
    for mname, preds in oof_preds_dict.items():
        if float(r2_score(targets, preds)) >= threshold:
            kept[mname] = preds
        else:
            removed.append(mname)
    return kept, removed


def _greedy_forward_select(oof_preds_dict, targets, meta_type='ElasticNet',
                           min_gain=0.0005, max_models=10):
    """Greedy forward model selection via nested 3-fold CV on meta-features.

    Starts from the single best model, iteratively adds the model that gives
    the largest R² improvement. Stops when gain < min_gain.

    Args:
        oof_preds_dict: {model_name: OOF_predictions_array}
        targets: phenotype values
        meta_type: 'Ridge', 'Lasso', or 'ElasticNet'
        min_gain: minimum R² improvement to keep adding models
        max_models: hard cap on number of base models

    Returns:
        selected_names: ordered list of selected model names
    """
    names = list(oof_preds_dict.keys())
    if len(names) <= 1:
        return names

    # Score each model individually
    scores = {m: float(r2_score(targets, oof_preds_dict[m])) for m in names}
    ranked = sorted(names, key=lambda m: scores[m], reverse=True)

    selected = [ranked[0]]
    pool = ranked[1:]

    inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)

    def _eval_subset(sel):
        X = np.column_stack([oof_preds_dict[m] for m in sel])
        preds = np.zeros(len(targets))
        for itr, ite in inner_kf.split(X):
            if meta_type == 'Ridge':
                m = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=3)
            elif meta_type == 'Lasso':
                m = LassoCV(alphas=LASSO_ALPHAS, cv=3, max_iter=10000, random_state=42)
            elif meta_type == 'ElasticNet':
                m = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                                 alphas=ENET_ALPHAS, cv=3, max_iter=10000, random_state=42)
            else:
                m = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=3)
            m.fit(X[itr], targets[itr])
            preds[ite] = m.predict(X[ite])
        return float(r2_score(targets, preds))

    best_r2 = _eval_subset(selected)

    while pool and len(selected) < max_models:
        gains = []
        for cand in pool[:min(12, len(pool))]:  # test up to 12 candidates per round
            trial_r2 = _eval_subset(selected + [cand])
            gains.append((cand, trial_r2 - best_r2, trial_r2))
        gains.sort(key=lambda x: -x[2])
        best_cand, best_gain, best_trial_r2 = gains[0]
        if best_gain < min_gain:
            break
        selected.append(best_cand)
        pool.remove(best_cand)
        best_r2 = best_trial_r2

    return selected


def stacking_evaluate(oof_preds_dict, targets, n_folds=5,
                      meta_type='ElasticNet', prune_corr=True):
    """Evaluate stacking from original-order OOF base predictions.

    Returns honest inner-CV OOF predictions, R², correlation, and true RMSE.
    """
    targets = np.asarray(targets, dtype=np.float64)
    oof_preds_dict = {k: np.asarray(v, dtype=np.float64) for k, v in oof_preds_dict.items()}
    original_names = list(oof_preds_dict.keys())
    pruned_names = []

    if not original_names:
        fallback = np.full(len(targets), float(np.mean(targets)), dtype=np.float64)
        r2_v, corr_v, rmse_v = _metric_values(targets, fallback)
        return {'R2': r2_v, 'Correlation': corr_v, 'RMSE': rmse_v,
                'OOF_predictions': fallback, 'Meta_weights': [],
                'Meta_intercept': float(np.mean(targets)), 'Base_models': [],
                'Pruned_models': [], 'Meta_type': meta_type}

    if prune_corr and len(original_names) > 2:
        oof_preds_dict, pruned_names = _prune_correlated(oof_preds_dict, targets)

    base_names = list(oof_preds_dict.keys())
    if len(base_names) < 2:
        only = base_names[0]
        sp = np.asarray(oof_preds_dict[only], dtype=np.float64)
        r2_v, corr_v, rmse_v = _metric_values(targets, sp)
        return {'R2': r2_v, 'Correlation': corr_v, 'RMSE': rmse_v,
                'OOF_predictions': sp, 'Meta_weights': [1.0],
                'Meta_intercept': 0.0, 'Base_models': base_names,
                'Pruned_models': pruned_names, 'Meta_type': meta_type}

    X_meta = np.column_stack([oof_preds_dict[m] for m in base_names])

    n_splits = min(n_folds, max(2, len(targets) // 3), len(targets))
    sp = np.zeros(len(targets), dtype=np.float64)

    if n_splits < 2:
        sp[:] = np.mean(targets)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        for tr, te in kf.split(X_meta):
            inner_cv = min(3, max(2, len(tr) // 3), len(tr))
            if meta_type == 'Lasso':
                m = LassoCV(alphas=LASSO_ALPHAS, cv=inner_cv, max_iter=10000, random_state=RANDOM_SEED)
            elif meta_type == 'ElasticNet':
                m = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                                 alphas=ENET_ALPHAS, cv=inner_cv, max_iter=10000,
                                 random_state=RANDOM_SEED)
            else:
                m = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=inner_cv)
            m.fit(X_meta[tr], targets[tr])
            sp[te] = m.predict(X_meta[te])

    final_cv = min(5, max(2, len(targets) // 3), len(targets))
    if meta_type == 'Lasso':
        final_meta = LassoCV(alphas=LASSO_ALPHAS, cv=final_cv, max_iter=10000, random_state=RANDOM_SEED)
    elif meta_type == 'ElasticNet':
        final_meta = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                                  alphas=ENET_ALPHAS, cv=final_cv, max_iter=10000,
                                  random_state=RANDOM_SEED)
    else:
        final_meta = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=final_cv)
    final_meta.fit(X_meta, targets)

    r2_v, corr_v, rmse_v = _metric_values(targets, sp)
    result = {'R2': r2_v,
              'Correlation': corr_v,
              'RMSE': rmse_v,
              'OOF_predictions': sp,
              'Meta_weights': np.ravel(final_meta.coef_).astype(float).tolist(),
              'Meta_intercept': float(final_meta.intercept_),
              'Base_models': base_names,
              'Meta_type': meta_type}
    if pruned_names:
        result['Pruned_models'] = pruned_names
    return result


def stacking_evaluate_greedy(oof_preds_dict, targets, n_folds=5, meta_type='ElasticNet'):
    """Full stacking pipeline: prune → greedy select → evaluate with inner CV.

    This is the recommended variant that automatically finds the optimal model
    subset.  Uses ElasticNet as default meta-learner — L1 prunes weak models
    while L2 provides stability among correlated base learners.
    """
    # Step 1: correlation pruning
    pruned_dict, pruned_names = _prune_correlated(oof_preds_dict, targets)

    # Step 2: greedy forward selection
    if len(pruned_dict) > 2:
        selected = _greedy_forward_select(pruned_dict, targets, meta_type=meta_type)
    else:
        selected = list(pruned_dict.keys())

    # Step 3: evaluate selected subset
    selected_dict = {m: oof_preds_dict[m] for m in selected}
    result = stacking_evaluate(selected_dict, targets, n_folds=n_folds,
                               meta_type=meta_type, prune_corr=False)
    if pruned_names:
        result['Pruned_models'] = pruned_names
    result['Greedy_selected'] = selected
    return result


def _add_stacking_to_results(oof_dl, oof_trad, y, trait_res, folds_run, suffix='', stacking_oof=None):
    """Run stacking ensembles and add results to trait_res without best-single flooring."""
    if folds_run < 3:
        return {}

    y = np.asarray(y, dtype=np.float64)
    n_cv = min(5, folds_run)
    selection = 'random' if suffix else 'gwas'

    def _display_names(names):
        if not suffix:
            return list(names)
        return [name if str(name).endswith(suffix) else f'{name}{suffix}' for name in names]

    def _record(name, result):
        out_name = name + suffix
        entry = {'R2': float(result.get('R2', np.nan)),
                 'Correlation': float(result.get('Correlation', 0.0)),
                 'RMSE': float(result.get('RMSE', np.nan)),
                 'Type': TYPE_ENS,
                 'DisplayName': out_name,
                 'Selection': selection,
                 'Meta_weights': result.get('Meta_weights', []),
                 'Meta_intercept': float(result.get('Meta_intercept', 0.0)),
                 'Base_models': _display_names(result.get('Base_models', [])),
                 'Meta_type': result.get('Meta_type', '')}
        if 'OOF_predictions' in result and stacking_oof is not None:
            stacking_oof[out_name] = np.asarray(result['OOF_predictions'], dtype=np.float32)
        if result.get('Pruned_models'):
            entry['Pruned_models'] = _display_names(result.get('Pruned_models', []))
        if result.get('Greedy_selected'):
            entry['Greedy_selected'] = _display_names(result.get('Greedy_selected', []))
        if result.get('R2_filtered'):
            entry['R2_filtered'] = _display_names(result.get('R2_filtered', []))
        trait_res[out_name] = entry
        return entry

    oof_all = {**oof_trad, **oof_dl}

    sr_dl = stacking_evaluate(oof_dl, y, n_folds=n_cv, meta_type='Ridge', prune_corr=False)
    _record('Stacking (DL)', sr_dl)

    sr_all = stacking_evaluate(oof_all, y, n_folds=n_cv, meta_type='Ridge', prune_corr=False)
    _record('Stacking (All)', sr_all)

    tsr = stacking_evaluate(oof_trad, y, n_folds=min(5, len(TRAD_NAMES)),
                            meta_type='Ridge', prune_corr=False)
    _record('Trad Ensemble', tsr)

    sp = stacking_evaluate(oof_all, y, n_folds=n_cv, meta_type='ElasticNet', prune_corr=True)
    _record('Stacking (Pruned)', sp)

    sg = stacking_evaluate_greedy(oof_all, y, n_folds=n_cv, meta_type='ElasticNet')
    _record('Stacking (Greedy)', sg)

    oof_r2_filtered, r2_removed = _filter_by_r2(oof_all, y, threshold=0.0)
    if len(oof_r2_filtered) >= 2:
        srg = stacking_evaluate_greedy(oof_r2_filtered, y, n_folds=n_cv, meta_type='ElasticNet')
        srg['R2_filtered'] = r2_removed
        _record('Stacking (R²+Greedy)', srg)
    else:
        if len(oof_r2_filtered) == 1:
            mname = list(oof_r2_filtered.keys())[0]
            fallback = np.asarray(oof_r2_filtered[mname], dtype=np.float64)
            r2_v, corr_v, rmse_v = _metric_values(y, fallback)
            srg = {'R2': r2_v, 'Correlation': corr_v, 'RMSE': rmse_v,
                   'OOF_predictions': fallback,
                   'Base_models': [mname], 'Greedy_selected': [mname],
                   'R2_filtered': r2_removed, 'Meta_weights': [1.0],
                   'Meta_intercept': 0.0, 'Meta_type': 'fallback'}
        else:
            fallback = np.full(len(y), float(np.mean(y)), dtype=np.float64)
            r2_v, corr_v, rmse_v = _metric_values(y, fallback)
            srg = {'R2': r2_v, 'Correlation': corr_v, 'RMSE': rmse_v,
                   'OOF_predictions': fallback,
                   'Base_models': [], 'Greedy_selected': [],
                   'R2_filtered': r2_removed, 'Meta_weights': [],
                   'Meta_intercept': float(np.mean(y)), 'Meta_type': 'mean_fallback'}
        _record('Stacking (R²+Greedy)', srg)

    n_greedy_pool = len(oof_all) - len(sg.get('Pruned_models', []))
    prefix = ' [random]' if suffix else ' [gwas]'
    print(f"  {'Stacking (DL)'+prefix:<33s} {sr_dl['R2']:8.4f} {sr_dl['Correlation']:8.4f} RMSE={sr_dl['RMSE']:.4f}")
    print(f"  {'Stacking (All)'+prefix:<33s} {sr_all['R2']:8.4f} {sr_all['Correlation']:8.4f} RMSE={sr_all['RMSE']:.4f}")
    print(f"  {'Trad Ensemble'+prefix:<33s} {tsr['R2']:8.4f} {tsr['Correlation']:8.4f} RMSE={tsr['RMSE']:.4f}")
    print(f"  {'Stacking (Pruned)'+prefix:<33s} {sp['R2']:8.4f} {sp['Correlation']:8.4f}  "
          f"[ElasticNet, pruned={len(sp.get('Pruned_models', []))}]")
    print(f"  {'Stacking (Greedy)'+prefix:<33s} {sg['R2']:8.4f} {sg['Correlation']:8.4f}  "
          f"[ElasticNet, selected={len(sg.get('Greedy_selected', []))}/{n_greedy_pool}]")
    gs = _display_names(sg.get('Greedy_selected', []))
    if gs:
        print(f"    Greedy selected: {gs}")
    if len(oof_r2_filtered) >= 2:
        print(f"  {'Stacking (R²+Greedy)'+prefix:<33s} {srg['R2']:8.4f} {srg['Correlation']:8.4f}  "
              f"[R²-filter removed {len(r2_removed)}: {_display_names(r2_removed)}]")
        rgs = _display_names(srg.get('Greedy_selected', []))
        if rgs:
            print(f"    R²+Greedy selected: {rgs}")
    else:
        print(f"  {'Stacking (R²+Greedy)'+prefix:<33s} fallback after R² filter ({len(oof_r2_filtered)} kept)")
    return stacking_oof or {}


def deploy_models(X, y, n_snps, trait_name, output_dir, tuned_params, quick_test=False):
    """Fit all models on full raw genotype data and save to disk for later inference."""
    import pickle
    deploy_dir = output_dir / f"deployed_{trait_name}"
    if deploy_dir.exists(): import shutil; shutil.rmtree(str(deploy_dir))
    deploy_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Deploying models on full dataset ({len(y)} samples, raw genotype input) ...")

    X = _as_model_input(X)
    y = np.asarray(y, dtype=np.float32)
    X_cand, _, cand_orig_idx, n_selected = _make_candidate_universe(X, None, n_snps)
    gidx_rel = gwas_select(X_cand, y, n_selected)
    gidx = cand_orig_idx[gidx_rel]
    X_selected = _as_model_input(X[:, gidx])

    G_full = X_selected @ X_selected.T / float(n_selected)
    trad_models = {'RRBLUP': (RRBLUP(), X_selected),
                   'GBLUP': (GBLUP(), G_full),
                   'XGBoost': (XGBoostModel(n_estimators=300, seed=_stable_seed(trait_name, 'XGBoost', 'deploy')), X_selected),
                   'ElasticNet': (ElasticNetModel(seed=_stable_seed(trait_name, 'ElasticNet', 'deploy')), X_selected),
                   'GWAS_RRBLUP': (GWASWeightedRRBLUP(), X_selected)}
    for tname, (tm, X_in) in trad_models.items():
        tm.fit(X_in, y)
        with open(deploy_dir / f"{tname}.pkl", 'wb') as f: pickle.dump(tm, f)
        print(f"    [saved] {tname}.pkl")

    for mname in DL_NAMES:
        tp = tuned_params.get(mname, {}) if tuned_params else {}
        hp = _get_train_hparams(mname, tuned_params)
        train_seed = _stable_seed(trait_name, mname, 'deploy_train')
        set_global_seed(train_seed)
        model = create_model(mname, n_selected, overrides=tp)
        model = train_torch_model(model, X_selected, y, epochs=300,
                                  batch_size=hp['batch_size'], lr=hp['lr'],
                                  weight_decay=hp['weight_decay'], patience=hp['patience'],
                                  seed=train_seed)
        torch.save(model.state_dict(), deploy_dir / f"{mname}.pt")
        print(f"    [saved] {mname}.pt")
        del model
    torch.cuda.empty_cache()

    meta = {'trait': trait_name, 'n_snps': n_selected, 'gwas_indices': gidx.tolist(),
            'n_samples': len(y), 'input': 'raw_genotype_float32',
            'fit_scope': 'full_data_deployment_not_cv_metric',
            'models': list(trad_models.keys()) + DL_NAMES}
    with open(deploy_dir / "deployment_meta.json", 'w', encoding='utf-8') as f: json.dump(meta, f, indent=2)
    print(f"    [saved] deployment_meta")


def tune_model_hyperparams(model_name, X_train, y_train, n_snps, n_trials=15, seed=RANDOM_SEED):
    """Tune one DL model using only the data supplied by the caller.

    For paper CV metrics, callers must pass an outer training fold only.  This
    function performs its own inner hold-out split and must never receive full
    trait data to choose hyperparameters later reused across outer CV folds.
    """
    try: import optuna
    except ImportError: print(f"    [SKIP] Optuna not installed, using defaults"); return {}, 0.0
    seed = int(seed)
    set_global_seed(seed)
    X_train = _as_model_input(X_train)
    y_train = np.asarray(y_train, dtype=np.float32)
    n_val = max(16, int(len(y_train) * 0.2))
    n_val = min(n_val, max(1, len(y_train) - 2))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y_train))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    def objective(trial):
        trial_seed = _stable_seed(seed, model_name, trial.number, 'optuna_trial')
        set_global_seed(trial_seed)
        if model_name == 'FGN':
            overrides = {'hidden': trial.suggest_categorical('hidden', [32, 48, 64]),
                         'dropout': trial.suggest_float('dropout', 0.3, 0.55),
                         'input_dropout': trial.suggest_float('input_dropout', 0.0, 0.2),
                         'n_spec': trial.suggest_categorical('n_spec', [8, 16, 24]),
                         'droppath': trial.suggest_float('droppath', 0.0, 0.25),
                         'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
                         'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
                         'patience': trial.suggest_int('patience', 20, 50)}
            model = FourierGenomicNet(n_snps=n_snps, hidden=overrides['hidden'],
                                      dropout=overrides['dropout'],
                                      input_dropout=overrides['input_dropout'],
                                      n_spec=overrides['n_spec'],
                                      droppath=overrides['droppath'])
            bs = _get_batch_size(model_name)
        elif model_name == 'FGNplus':
            overrides = {'hidden': trial.suggest_categorical('hidden', [48, 64, 96]),
                         'dropout': trial.suggest_float('dropout', 0.25, 0.5),
                         'input_dropout': trial.suggest_float('input_dropout', 0.1, 0.4),
                         'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
                         'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
                         'patience': trial.suggest_int('patience', 20, 50)}
            model = FGNplus(n_snps=n_snps, hidden=overrides['hidden'],
                            dropout=overrides['dropout'],
                            input_dropout=overrides['input_dropout'])
            bs = _get_batch_size(model_name)
        elif model_name == 'FGN v4':
            overrides = {'hidden': trial.suggest_categorical('hidden', [48, 64, 96]),
                         'dropout': trial.suggest_float('dropout', 0.2, 0.5),
                         'input_dropout': trial.suggest_float('input_dropout', 0.1, 0.4),
                         'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
                         'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
                         'patience': trial.suggest_int('patience', 20, 50)}
            model = FGNv4(n_snps=n_snps, hidden=overrides['hidden'], dropout=overrides['dropout'],
                          input_dropout=overrides['input_dropout'])
            bs = _get_batch_size(model_name)
        elif model_name == 'FusionNet':
            overrides = {'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 48, 64]),
                         'dropout': trial.suggest_float('dropout', 0.2, 0.5),
                         'lr': trial.suggest_float('lr', 5e-4, 3e-3, log=True),
                         'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
                         'patience': trial.suggest_int('patience', 20, 50)}
            model = FusionNet(n_snps=n_snps, hidden_dim=overrides['hidden_dim'], dropout=overrides['dropout'])
            bs = _get_batch_size(model_name)
        elif model_name == 'AdditiveGenomicNet':
            overrides = {'hidden': trial.suggest_categorical('hidden', [32, 48, 64]),
                         'dropout': trial.suggest_float('dropout', 0.2, 0.5),
                         'input_dropout': trial.suggest_float('input_dropout', 0.1, 0.4),
                         'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
                         'weight_decay': trial.suggest_float('weight_decay', 5e-4, 1e-2, log=True),
                         'patience': trial.suggest_int('patience', 20, 50)}
            model = AdditiveGenomicNet(n_snps=n_snps, hidden=overrides['hidden'], dropout=overrides['dropout'],
                                       input_dropout=overrides['input_dropout'])
            bs = _get_batch_size(model_name)
        elif model_name == 'GenomicFM':
            overrides = {'k': trial.suggest_categorical('k', [2, 4, 8]),
                         'dropout': trial.suggest_float('dropout', 0.1, 0.4),
                         'mlp_hidden': trial.suggest_categorical('mlp_hidden', [8, 16, 24]),
                         'lr': trial.suggest_float('lr', 1e-3, 5e-3, log=True),
                         'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
                         'patience': trial.suggest_int('patience', 20, 50)}
            model = GenomicFM(n_snps=n_snps, k=overrides['k'],
                              dropout=overrides['dropout'],
                              mlp_hidden=overrides['mlp_hidden'])
            bs = _get_batch_size(model_name)
            model = train_torch_model(model, X_tr, y_tr, epochs=200, batch_size=bs,
                                       lr=overrides['lr'], weight_decay=overrides['weight_decay'],
                                       patience=overrides['patience'], val_ratio=0.2,
                                       seed=trial_seed)
            preds = predict_torch_model(model, X_val)
            return float(r2_score(y_val, preds))
        else: raise ValueError(f"Unknown model for tuning: {model_name}")
        model = train_torch_model(model, X_tr, y_tr, epochs=300, batch_size=bs,
                                  lr=overrides.get('lr', 2e-3),
                                  weight_decay=overrides.get('weight_decay', 1e-3),
                                  patience=overrides.get('patience', 30),
                                  seed=trial_seed)
        preds = predict_torch_model(model, X_val)
        return float(r2_score(y_val, preds))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed),
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
    with open(output_dir / f"ensemble_final_{ts}.json", 'w', encoding='utf-8') as f:
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
    skipped_id, skipped_nonnum, skipped_missing = [], [], []
    print(f"  文件: {WHEAT_PHENO}")
    print(f"  行数: {n_samples}, 列数: {len(df.columns)}")
    print(f"  列名: {list(df.columns)}")
    for col in df.columns:
        if col.lower() in ('sample', 'id', 'name', 'accession', 'line'):
            skipped_id.append(col); continue
        try:
            vals = pd.to_numeric(df[col], errors='coerce').values.astype(np.float32)
            mask = ~np.isnan(vals)
            valid_pct = mask.sum() / len(vals) * 100
            if valid_pct > 50:
                traits[col] = (vals, mask)
                print(f"    [OK] {col}: {mask.sum()}/{len(vals)} ({valid_pct:.1f}%) valid")
            else:
                skipped_missing.append(f"{col}({valid_pct:.1f}%)")
        except (ValueError, TypeError):
            skipped_nonnum.append(col)
    if skipped_id: print(f"  跳过ID列: {skipped_id}")
    if skipped_nonnum: print(f"  跳过非数值列: {skipped_nonnum}")
    if skipped_missing: print(f"  跳过缺失过多列: {skipped_missing}")
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


def load_wheat2000_data():
    """Load wheat2000 CSV-format data from dnngp_data.

    Genotype: 2000 lines × 33,709 binary SNPs (0/1) in 2000gene.csv
    Phenotype: 6 traits in separate .txt files, one value per line with header.

    Returns dict: {trait_name: (X, y, None)}
        X — (n_samples, n_markers) float32 genotype matrix
        y — (n_samples,) float32 phenotype values
        vt — None (no variant type annotation in CSV format)
    """
    print(f"\n{'='*70}\nWheat2000 CSV Data Loading\n{'='*70}")

    # 1. Genotype matrix (skip row-0 column-headers, col-0 row-index)
    print(f"\n[1/2] Loading genotype matrix ...")
    X_all = pd.read_csv(WHEAT2000_GENO, skiprows=1, index_col=0,
                        dtype=np.float32, header=None).values
    n_samples, n_markers = X_all.shape
    print(f"  文件: {WHEAT2000_GENO}")
    print(f"  基因型矩阵: {n_samples} samples × {n_markers} markers (binary 0/1)")

    # 2. Phenotype files
    print(f"\n[2/2] Loading phenotypes ...")
    trait_data = {}
    for tname, fname in WHEAT2000_TRAITS.items():
        fpath = os.path.join(WHEAT2000_DATA_DIR, fname)
        with open(fpath) as f:
            body = f.read().strip().split('\n')
        if len(body) < 2:
            raise ValueError(f"Phenotype file too short: {fpath} ({len(body)} lines)")
        y = np.array([float(l) for l in body[1:]], dtype=np.float32)
        trait_data[tname] = (X_all[:len(y)], y, None)
        print(f"  {tname}: {len(y)} samples, "
              f"mean={np.mean(y):.4f}, std={np.std(y):.4f}")
    return trait_data


def _run_trait_pipeline(X_all, y, vt_all, trait_name, folds_run, output_dir,
                        quick_test):
    """Run the full GWAS-vs-Random ensemble pipeline for one trait."""
    X_all = _as_model_input(X_all)
    y = np.asarray(y, dtype=np.float32)
    if vt_all is not None:
        vt_all = np.asarray(vt_all)
    n_requested = min(GWAS_TOP_K, max(50, X_all.shape[1] - 50))
    seed_context = f"{output_dir.name}:{trait_name}"
    print(f"  {len(y)} samples, {X_all.shape[1]} markers -> up to {n_requested} selected")
    print("  Input representation: raw genotype float32 (no feature standardization)")

    # CV metrics use fixed hyperparameters.  Any future tuning must happen
    # inside each outer training fold only; full-data tuning is deployment-only.
    tuned_params = {}
    if not quick_test:
        print("\n  [AutoML] Skipped during CV to avoid test-fold label leakage; "
              "using fixed hyperparameters.")

    kf = KFold(n_splits=folds_run, shuffle=True, random_state=RANDOM_SEED)
    results = {m: {'preds': [], 'targets': [], 'params': 0, 'time': 0.0, 'gpu_mem': 0.0}
               for m in ALL_NAMES + ALL_NAMES_RANDOM}
    oof_trad = {m: np.zeros(len(y), dtype=np.float32) for m in TRAD_NAMES}
    oof_dl = {m: np.zeros(len(y), dtype=np.float32) for m in DL_NAMES}
    oof_trad_r = {m: np.zeros(len(y), dtype=np.float32) for m in TRAD_NAMES}
    oof_dl_r = {m: np.zeros(len(y), dtype=np.float32) for m in DL_NAMES}

    for fi, (tr, te) in enumerate(kf.split(X_all)):
        print(f"\n  --- Fold {fi+1}/{folds_run} ---")
        Xtr_raw, Xte_raw = _as_model_input(X_all[tr]), _as_model_input(X_all[te])
        ytr, yte = y[tr], y[te]
        Xtr_cand, Xte_cand, vt_cand, _, n_selected = _make_fold_candidate_universe(
            Xtr_raw, Xte_raw, vt_all, n_requested)
        print(f"    Candidate universe after MAF: {Xtr_cand.shape[1]} markers; selected={n_selected}")

        gidx_gwas = gwas_select(Xtr_cand, ytr, n_selected)
        Xtr_gwas = _as_model_input(Xtr_cand[:, gidx_gwas])
        Xte_gwas = _as_model_input(Xte_cand[:, gidx_gwas])
        G_fold_train = Xtr_gwas @ Xtr_gwas.T / float(n_selected)
        G_fold_te_tr = Xte_gwas @ Xtr_gwas.T / float(n_selected)

        trad_seed = _stable_seed(seed_context, fi, 'traditional', 'train')
        trad_configs = _make_trad_configs(
            G_fold_train, G_fold_te_tr, n_selected, len(tr), seed=trad_seed)
        for tname, build_fn, fit_fn, pred_fn, param_count in trad_configs:
            set_global_seed(_stable_seed(seed_context, fi, tname, 'train'))
            t0 = time.time()
            tmodel = build_fn()
            fit_fn(tmodel, Xtr_gwas, ytr)
            preds = np.asarray(pred_fn(tmodel, Xte_gwas), dtype=np.float32)
            results[tname]['preds'].extend(preds.tolist())
            results[tname]['targets'].extend(yte.tolist())
            results[tname]['time'] += time.time() - t0
            if fi == 0:
                results[tname]['params'] = param_count
            oof_trad[tname][te] = preds
            print(f"    {tname:<16s} R2={r2_score(yte, preds):+.4f}")

        gidx_dl, _, Xtr_dl, Xte_dl = _select_dl_markers(
            Xtr_cand, Xte_cand, ytr, gidx_gwas, vt_cand, n_selected)
        for mname in DL_NAMES:
            hp = _get_train_hparams(mname, tuned_params)
            train_seed = _stable_seed(seed_context, fi, mname, 'train')
            set_global_seed(train_seed)
            model_tp = (tuned_params or {}).get(mname, {})
            model = create_model(mname, n_selected, overrides=model_tp)
            t0 = time.time()
            if fi == 0:
                results[mname]['params'] = sum(p.numel() for p in model.parameters())
            if DEVICE.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            model = train_torch_model(model, Xtr_dl, ytr, epochs=300,
                                      batch_size=hp['batch_size'], lr=hp['lr'],
                                      weight_decay=hp['weight_decay'],
                                      patience=hp['patience'], seed=train_seed)
            if DEVICE.type == 'cuda':
                results[mname]['gpu_mem'] = max(
                    results[mname]['gpu_mem'],
                    torch.cuda.max_memory_allocated() / (1024 * 1024))
            preds = np.asarray(predict_torch_model(model, Xte_dl), dtype=np.float32)
            elapsed = time.time() - t0
            results[mname]['preds'].extend(preds.tolist())
            results[mname]['targets'].extend(yte.tolist())
            results[mname]['time'] += elapsed
            oof_dl[mname][te] = preds
            print(f"    {mname:<22s} R2={r2_score(yte, preds):+.4f}  ({elapsed:.1f}s)")
            del model
        torch.cuda.empty_cache()

        _run_fold_random_pass(Xtr_cand, Xte_cand, ytr, yte, te, n_selected, fi,
                              results, oof_trad_r, oof_dl_r, tuned_params,
                              seed_context=seed_context)

    print(f"\n  {'-'*70}\n  {trait_name} Final Results:\n  "
          f"{'Model':<28s} {'R2':>8s} {'Corr':>8s} {'RMSE':>8s} {'Time':>8s}\n  {'-'*70}")
    trait_res = {}

    def _summarize(names, selection):
        for mname in names:
            p = np.asarray(results[mname]['preds'], dtype=np.float32)
            t = np.asarray(results[mname]['targets'], dtype=np.float32)
            r2_v, corr_v, rmse_v = _metric_values(t, p)
            base = mname.replace('_random', '')
            mtype = _model_type(base)
            trait_res[mname] = {
                'R2': r2_v, 'Correlation': corr_v, 'RMSE': rmse_v,
                'Type': mtype, 'Selection': selection, 'DisplayName': mname,
                'Time': results[mname]['time'] / folds_run,
                'Params': results[mname].get('params', 0),
                'GPUMem': results[mname].get('gpu_mem', 0.0)}
            print(f"  {mname:<28s} {r2_v:8.4f} {corr_v:8.4f} "
                  f"{rmse_v:8.4f} {results[mname]['time']/folds_run:7.1f}s")

    _summarize(ALL_NAMES, 'gwas')
    _summarize(ALL_NAMES_RANDOM, 'random')

    stacking_oof = {}
    stacking_oof_r = {}
    _add_stacking_to_results(oof_dl, oof_trad, y, trait_res, folds_run,
                             suffix='', stacking_oof=stacking_oof)
    _add_stacking_to_results(oof_dl_r, oof_trad_r, y, trait_res, folds_run,
                             suffix='_random', stacking_oof=stacking_oof_r)

    oof_gwas_named = {**oof_trad, **oof_dl, **stacking_oof}
    oof_random_named = {**{f'{m}_random': v for m, v in oof_trad_r.items()},
                        **{f'{m}_random': v for m, v in oof_dl_r.items()},
                        **stacking_oof_r}
    _save_oof_npz(oof_gwas_named, y, output_dir, trait_name)
    _save_oof_npz(oof_random_named, y, output_dir, f"{trait_name}_random")

    if not quick_test:
        deploy_models(X_all, y, n_requested, trait_name, output_dir,
                      tuned_params, quick_test)
    return trait_res


def _run_ensemble_traits(trait_data, output_dir, crop_label, quick_test):
    """Load and run ensemble pipeline across all traits. Returns all_results."""
    trait_names = sorted(trait_data.keys())
    print(f"\n实验性状: {trait_names}")

    traits_run = trait_names[:1] if quick_test else trait_names
    folds_run = min(2, N_FOLDS) if quick_test else N_FOLDS
    if quick_test:
        print(f"  [QUICK TEST] {len(traits_run)} trait x {folds_run} folds")

    all_results = {}
    total_t0 = time.time()

    for t_idx, trait in enumerate(traits_run):
        print(f"\n{'='*80}\n  TRAIT [{t_idx+1}/{len(traits_run)}]: {trait}\n{'='*80}")
        item = trait_data[trait]
        if len(item) == 2:
            X_all, y = item
            vt_all = None
        else:
            X_all, y, vt_all = item
        trait_res = _run_trait_pipeline(
            X_all, y, vt_all, trait, folds_run, output_dir, quick_test)
        all_results[trait] = trait_res
        with open(output_dir / "ensemble_intermediate.json", 'w',
                  encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    if not quick_test:
        _print_final_summary(all_results, traits_run, output_dir, total_t0,
                             crop_label)
    return all_results


def run_wheat2000(quick_test=True):
    """Wheat2000 ensemble pipeline on CSV-format dnngp data
    (2000 lines × 33K binary SNPs, 6 agronomic traits)."""
    _script_dir = Path(__file__).resolve().parent
    output_dir = _script_dir / "results" / "wheat2000_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}  |  Quick test: {quick_test}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    trait_data = load_wheat2000_data()
    _run_ensemble_traits(trait_data, output_dir, 'Wheat2000', quick_test)


def run_wheat(quick_test=True):
    _script_dir = Path(__file__).resolve().parent
    output_dir = _script_dir / "results" / "wheat_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}  |  Quick test: {quick_test}")
    if DEVICE.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")
    trait_data = load_wheat_data()
    _run_ensemble_traits(trait_data, output_dir, 'Wheat', quick_test)


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
        mask = ~np.isnan(y) & (y > -8)  # -9 is missing-value sentinel
        trait_data[t] = (_as_model_input(X_t[mask]), y[mask])
        if mask.sum() < len(y):
            print(f"  {t}: {mask.sum()} samples (removed {len(y) - mask.sum()} sentinel -9)")
    return trait_data


def run_rice(quick_test=True):
    _script_dir = Path(__file__).resolve().parent
    output_dir = _script_dir / "results" / "rice_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}  |  Quick test: {quick_test}")
    if DEVICE.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")
    trait_data = load_rice_data()
    _run_ensemble_traits(trait_data, output_dir, 'Rice', quick_test)


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

    var_thresh = 0.005
    vars_per_marker = np.var(X_all, axis=0)
    keep = vars_per_marker >= var_thresh
    if keep.sum() < X_all.shape[1]:
        n_before = X_all.shape[1]
        X_all = X_all[:, keep]
        print(f"  Low-variance filter: {n_before} -> {X_all.shape[1]} markers kept")
    else:
        print(f"  Low-variance filter: all {X_all.shape[1]} markers kept")

    trait_data = {trait: (_as_model_input(X_all), y.astype(np.float32))
                  for trait, y in y_dict.items()}
    _run_ensemble_traits(trait_data, output_dir, 'Maize', quick_test)

# ============================================================================
# Section K2: EasyGeSe Benchmark Data Loader + Soybean / Wheat GABI Pipelines
# ============================================================================

def load_easygese_data(data_dir, trait_names=None):
    """Generic loader for EasyGeSe-style genotype NPZ + trait JSON data.

    The loader keeps the downstream contract ``{trait_name: (X, y)}`` but is
    tolerant to common schema variants: genotype arrays named ``G``/``X``/etc.,
    numeric ``trait_00`` keys, a nested ``traits`` dict, or semantic top-level
    trait keys.
    """
    genotype_path = os.path.join(data_dir, "genotype_matrix.npz")
    npz = np.load(genotype_path, allow_pickle=True)
    try:
        keys = list(npz.files)
        preferred = ['G', 'genotype', 'genotypes', 'X', 'matrix', 'genotype_matrix']
        G = None
        g_key = None
        for key in preferred + [k for k in keys if k not in preferred]:
            if key not in keys:
                continue
            arr = np.asarray(npz[key])
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                G = arr
                g_key = key
                break
        if G is None:
            raise ValueError(f"No 2D numeric genotype matrix found in {genotype_path}; "
                             f"available keys={keys}")
    finally:
        npz.close()

    G = _as_model_input(G)
    if g_key != 'G':
        print(f"  [INFO] Using genotype matrix key '{g_key}' from {genotype_path}")

    with open(os.path.join(data_dir, "trait_data.json"), encoding='utf-8') as f:
        trait_info = json.load(f)

    # Low-variance filter on full matrix (once, not per-trait)
    var_thresh = 0.005
    vars_per_marker = np.var(G, axis=0)
    keep = vars_per_marker >= var_thresh
    if keep.sum() < G.shape[1]:
        n_before = G.shape[1]
        G = G[:, keep]
        print(f"  Low-variance filter: {n_before} -> {G.shape[1]} markers kept")

    metadata_keys = {'_trait_names', 'trait_names', 'samples', 'sample_ids', 'metadata'}

    def _numeric_trait_sort(keys_in):
        num = [k for k in keys_in if isinstance(k, str) and k.startswith('trait_') and k[6:].isdigit()]
        if num:
            return sorted(num, key=lambda k: int(k[6:]))
        return list(keys_in)

    num_keys = _numeric_trait_sort(trait_info.keys())
    num_keys = [k for k in num_keys if isinstance(k, str) and k.startswith('trait_') and k[6:].isdigit()]
    if num_keys:
        trait_items = [(k, trait_info[k]) for k in num_keys]
    else:
        traits_obj = trait_info.get('traits')
        if isinstance(traits_obj, dict):
            trait_items = [(k, traits_obj[k]) for k in _numeric_trait_sort(traits_obj.keys())]
        else:
            trait_items = [(k, v) for k, v in trait_info.items()
                           if k not in metadata_keys and isinstance(v, (dict, list))]
        if not trait_items:
            raise ValueError(f"No trait entries found in {data_dir}/trait_data.json")

    if trait_names is None:
        trait_names = trait_info.get('_trait_names', trait_info.get('trait_names', None))
    if isinstance(trait_names, str):
        trait_names = [trait_names]
    if trait_names is not None and not isinstance(trait_names, dict):
        trait_names = list(trait_names)
        if len(trait_names) < len(trait_items):
            print(f"  [WARN] Only {len(trait_names)} trait names for {len(trait_items)} keys; "
                  "using provided names + key fallback")

    def _display_name(i, key):
        if isinstance(trait_names, dict):
            return str(trait_names.get(key, key))
        if trait_names is not None and i < len(trait_names):
            return str(trait_names[i])
        return str(key)

    def _to_float(value):
        if value is None:
            return np.nan
        if isinstance(value, str):
            s = value.strip()
            if s.lower() in ('', 'na', 'nan', 'null', 'none', 'missing'):
                return np.nan
            value = s
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    def _extract_values(td):
        if isinstance(td, dict):
            for key in ('values', 'phenotypes', 'phenotype', 'y', 'trait_values'):
                if key in td:
                    return td[key]
        elif isinstance(td, list):
            return td
        return None

    def _extract_indices(td, n_values):
        if isinstance(td, dict):
            for key in ('genotype_indices', 'sample_indices', 'indices'):
                if key in td:
                    return td[key]
        return np.arange(n_values)

    print(f"  {G.shape[0]} samples x {G.shape[1]} SNPs  |  {len(trait_items)} traits")

    trait_data = {}
    for i, (key, td) in enumerate(trait_items):
        tname = _display_name(i, key)
        raw_values = _extract_values(td)
        if raw_values is None:
            print(f"  [WARN] Trait {key}: no values field found, skipping")
            continue
        y = np.asarray([_to_float(v) for v in raw_values], dtype=np.float32)
        try:
            idxs = np.asarray(_extract_indices(td, len(y)), dtype=np.int64)
        except (TypeError, ValueError):
            print(f"  [WARN] Trait {key}: invalid sample indices, skipping")
            continue
        if len(idxs) != len(y):
            n = min(len(idxs), len(y))
            print(f"  [WARN] Trait {key}: {len(idxs)} indices vs {len(y)} values; truncating to {n}")
            idxs, y = idxs[:n], y[:n]
        in_bounds = (idxs >= 0) & (idxs < G.shape[0])
        if not np.all(in_bounds):
            print(f"  [WARN] Trait {key}: dropping {np.sum(~in_bounds)} out-of-range indices")
            idxs, y = idxs[in_bounds], y[in_bounds]
        mask = np.isfinite(y)
        if not np.any(mask):
            print(f"  [WARN] Trait {key}: no finite phenotype values after filtering, skipping")
            continue
        trait_data[tname] = (_as_model_input(G[idxs][mask]), y[mask])

    if not trait_data:
        raise ValueError(f"No usable traits found in {data_dir}/trait_data.json")
    return trait_data


def run_soybean(quick_test=True):
    """Soybean SoySNP50K ensemble pipeline."""
    _script_dir = Path(__file__).resolve().parent
    output_dir = _script_dir / "results" / "soybean_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}  |  Quick test: {quick_test}")
    if DEVICE.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")
    trait_data = load_easygese_data(SOYBEAN_DATA_DIR, SOYBEAN_TRAIT_NAMES)
    _run_ensemble_traits(trait_data, output_dir, 'Soybean', quick_test)


def run_wheat_gabi(quick_test=True):
    """Wheat GABI iSELECT 90k ensemble pipeline."""
    _script_dir = Path(__file__).resolve().parent
    output_dir = _script_dir / "results" / "wheat_gabi_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}  |  Quick test: {quick_test}")
    if DEVICE.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")
    trait_data = load_easygese_data(WHEAT_GABI_DATA_DIR)
    _run_ensemble_traits(trait_data, output_dir, 'Wheat GABI', quick_test)

def _save_oof_npz(oof_predictions, y_true, output_dir, trait_name):
    """Save original-sample-order OOF predictions as NPZ for plotting.

    ``oof_predictions`` must map display/model names to arrays indexed in the same
    order as the original trait sample vector ``y_true``.  This keeps ``_y_true`` in
    original sample order and supports base + stacking OOF in the same file.
    """
    oof_dir = output_dir / "oof_predictions"
    oof_dir.mkdir(parents=True, exist_ok=True)
    aligned_y = np.asarray(y_true, dtype=np.float32)
    data = {'_y_true': aligned_y}
    for m, preds in oof_predictions.items():
        p = np.asarray(preds, dtype=np.float32)
        if len(p) == len(aligned_y):
            data[m] = p
    path = oof_dir / f"{trait_name}_oof.npz"
    np.savez_compressed(path, **data)
    print(f"  OOF saved: {path}")

def _load_json_safe(path):
    if os.path.exists(str(path)):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# Module-level color constants — used by _model_color()
_MODEL_COLORS_TRAD = {'RRBLUP': '#90CAF9', 'GBLUP': '#64B5F6', 'XGBoost': '#1565C0',
                      'ElasticNet': '#42A5F5', 'GWAS_RRBLUP': '#1E88E5'}
_MODEL_COLORS_DL = {'FGN': '#FFB74D', 'FGN v2': '#FF9800', 'FGN v4': '#F57C00',
                    'FGN v5': '#E65100', 'FGN v6': '#BF360C', 'FGN v7': '#FFD54F',
                    'FGN v9': '#FFCC80', 'FGN v10': '#FFE082', 'FGN v11': '#FFECB3',
                    'FGNplus': '#A1887F', 'FGN PCA': '#BCAAA4', 'GenomicFM': '#D7CCC8',
                    'FusionNet': '#E91E63', 'AdditiveGenomicNet': '#F48FB1',
                    'WheatGP': '#78909C'}
_MODEL_COLORS_ENS = {'Stacking (DL)': '#66BB6A', 'Stacking (All)': '#2E7D32',
                     'Trad Ensemble': '#0D47A1', 'Stacking (Pruned)': '#43A047',
                     'Stacking (Greedy)': '#1B5E20', 'Stacking (R²+Greedy)': '#388E3C'}


def _model_color(name):
    # Strip _random suffix for color lookup (GWAS vs random share color)
    base = name.replace('_random', '')
    if base in _MODEL_COLORS_TRAD: return _MODEL_COLORS_TRAD[base]
    if base in _MODEL_COLORS_DL: return _MODEL_COLORS_DL[base]
    if base in _MODEL_COLORS_ENS: return _MODEL_COLORS_ENS[base]
    return '#BDBDBD'


def _mean_r2(data, model_name):
    r2s = [data[t][model_name]['R2'] for t in data if model_name in data[t]]
    return np.mean(r2s) if r2s else float('nan')




def _dataset_specs():
    """Unified plotting dataset specifications."""
    return [
        {'tag': 'Wheat', 'sub': 'wheat', 'bar_prefix': '08_wheat', 'scatter_prefix': '05_wheat', 'ncols': 2, 'figsize': (24, 14)},
        {'tag': 'Wheat2000', 'sub': 'wheat2000', 'bar_prefix': '08_wheat2000', 'scatter_prefix': '11_wheat2000', 'ncols': 2, 'figsize': (24, 18)},
        {'tag': 'Rice', 'sub': 'rice', 'bar_prefix': '09_rice', 'scatter_prefix': '05_rice', 'ncols': 2, 'figsize': (24, 30)},
        {'tag': 'Maize', 'sub': 'maize', 'bar_prefix': '10_maize', 'scatter_prefix': '06_maize', 'ncols': 2, 'figsize': (24, 12)},
        {'tag': 'Soybean', 'sub': 'soybean', 'bar_prefix': '10_soybean', 'scatter_prefix': '06_soybean', 'ncols': 2, 'figsize': (18, 8)},
        {'tag': 'WheatGABI', 'sub': 'wheat_gabi', 'bar_prefix': '10_wheat_gabi', 'scatter_prefix': '11_wheat_gabi', 'ncols': 4, 'figsize': (32, 24)},
    ]


def _load_plot_datasets():
    script_dir = Path(__file__).resolve().parent
    datasets = []
    for spec in _dataset_specs():
        d = _load_json_safe(script_dir / "results" / f"{spec['sub']}_ensemble" / "ensemble_intermediate.json")
        if d:
            datasets.append((spec['tag'], d, sorted(d.keys())))
    return datasets


def _dataset_spec_by_tag(tag):
    for spec in _dataset_specs():
        if spec['tag'] == tag:
            return spec
    return {'tag': tag, 'sub': tag.lower(), 'fig': 'xx', 'ncols': 2, 'figsize': (24, 18)}


def _is_random_model(name):
    return str(name).endswith('_random')


def _is_single_model(name):
    return 'Stacking' not in name and 'Ensemble' not in name


def _filter_models_for_arm(models, arm='gwas', include_ensemble=True):
    out = []
    for m in models:
        if arm == 'gwas' and _is_random_model(m):
            continue
        if arm == 'random' and not _is_random_model(m):
            continue
        if not include_ensemble and not _is_single_model(m):
            continue
        out.append(m)
    return out

def generate_bar_charts(fig_dir=None, datasets=None):
    """Generate bar chart figures from ensemble_intermediate.json files.

    Produces:
      01_per_dataset_bar_charts.png   — per-dataset model comparison (3 datasets)
      02_cross_dataset_comparison.png — models common to all datasets
      03_stacking_gain_scatter.png    — Stacking (Greedy) vs best single model
      07_combined_ranking.png         — multi-dataset average ranking
      08_wheat2000_per_trait_bars.png — wheat2000 per-trait model R²
      09_rice_per_trait_bars.png      — rice per-trait model R²
      10_maize_per_trait_bars.png     — maize per-trait model R²
    """
    import matplotlib.pyplot as plt

    SCRIPT_DIR = Path(__file__).resolve().parent
    fig_dir = Path(fig_dir) if fig_dir else SCRIPT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    DATASETS = datasets if datasets is not None else _load_plot_datasets()

    if len(DATASETS) < 1:
        print("  [plot] No dataset JSONs found for bar charts, skipping.")
        return

    n_datasets = len(DATASETS)
    print(f"\n[plot] Generating bar chart figures (01-03, 07-10) from {n_datasets} datasets...")

    # --- Fig 01: Per-dataset bar charts (3 panels) ---
    fig, axes = plt.subplots(1, n_datasets, figsize=(12 * n_datasets, 14), squeeze=False)
    fig.suptitle('Genomic Prediction Ensemble — Per-Dataset Model Comparison (5-fold CV R²)',
                 fontsize=22, fontweight='bold', y=1.01)
    for ax_idx, (dname, data, traits) in enumerate(DATASETS):
        ax = axes[0, ax_idx]
        all_models = _filter_models_for_arm(list(data[traits[0]].keys()), arm='gwas', include_ensemble=True)
        means = {m: _mean_r2(data, m) for m in all_models}
        sorted_m = sorted([m for m in all_models if means[m] > -1], key=lambda m: means[m], reverse=True)[:20]
        vals = [means[m] for m in sorted_m]
        colors = [_model_color(m) for m in sorted_m]
        x = np.arange(len(sorted_m))
        bars = ax.bar(x, vals, 0.7, color=colors, edgecolor='white', linewidth=0.8, zorder=3)
        if vals:
            best_idx = int(np.argmax(vals))
            bars[best_idx].set_edgecolor('#C62828'); bars[best_idx].set_linewidth(3.0)
            for i, (m, v) in enumerate(zip(sorted_m, vals)):
                if i < 8 or v == max(vals):
                    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom',
                            fontsize=8 if v != max(vals) else 10,
                            fontweight='bold' if v == max(vals) else 'normal',
                            color='#C62828' if v == max(vals) else '#555')
            ax.set_title(f'{dname} ({len(traits)} trait{"s" if len(traits)>1 else ""})  Best: {sorted_m[best_idx]} ({vals[best_idx]:.3f})',
                         fontsize=13, fontweight='bold')
            ax.set_ylim(min(-0.5, min(vals)-0.15), max(vals)+0.15)
        else:
            ax.set_title(f'{dname}: no GWAS models', fontsize=13, fontweight='bold')
        ax.axhline(y=0, color='#666', linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(sorted_m, rotation=55, ha='right', fontsize=7.5)
        ax.set_ylabel('R²', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir/'01_per_dataset_bar_charts.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  -> 01_per_dataset_bar_charts.png")

    # --- Fig 02: Cross-dataset comparison ---
    model_sets = [set(_filter_models_for_arm(data[traits[0]].keys(), arm='gwas', include_ensemble=True)) for _, data, traits in DATASETS]
    common = model_sets[0]
    for s in model_sets[1:]: common = common & s
    all_common = sorted(common, key=lambda m: np.mean([_mean_r2(d, m) for _, d, _ in DATASETS]), reverse=True)
    common_sorted = [m for m in all_common if min(_mean_r2(d, m) for _, d, _ in DATASETS) > -1]

    fig, ax = plt.subplots(figsize=(18, 10))
    x = np.arange(len(common_sorted)); bar_w = 0.25
    for bi, (dname, data, _) in enumerate(DATASETS):
        vals = [_mean_r2(data, m) for m in common_sorted]
        dataset_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#00897B']
        c = dataset_colors[bi % len(dataset_colors)]
        ax.bar(x + (bi - (n_datasets-1)/2) * bar_w, vals, bar_w, color=c, edgecolor='white',
               label=f'{dname} ({len(DATASETS[bi][2])} trait{"s" if len(DATASETS[bi][2])>1 else ""})', zorder=3)
    ax.axhline(y=0, color='#666', linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(common_sorted, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('R²', fontsize=13)
    ax.set_title(f'Cross-Dataset Comparison — Models Common to {n_datasets} Datasets', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right'); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir/'02_cross_dataset_comparison.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  -> 02_cross_dataset_comparison.png")

    # --- Fig 03: Stacking gain scatter ---
    fig, axes = plt.subplots(1, n_datasets, figsize=(8 * n_datasets, 8), squeeze=False)
    fig.suptitle('Stacking (Greedy) vs Best Single Model — Per Trait', fontsize=16, fontweight='bold')
    for ax_idx, (dname, data, traits) in enumerate(DATASETS):
        ax = axes[0, ax_idx]
        singles = _filter_models_for_arm([m for m in data[traits[0]].keys() if _is_single_model(m)], arm='gwas', include_ensemble=False)
        stk_key = 'Stacking (Greedy)' if 'Stacking (Greedy)' in data[traits[0]] else 'Stacking (All)'
        xs, ys = [], []
        for t in traits:
            bx = max(data[t][m]['R2'] for m in singles if m in data[t])
            by = data[t].get(stk_key, {}).get('R2', bx)
            xs.append(bx); ys.append(by)
        mn = min(min(xs), min(ys)) - 0.03; mx = max(max(xs), max(ys)) + 0.05
        ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.3, lw=1.5, label='y=x')
        for i, t in enumerate(traits):
            ax.scatter(xs[i], ys[i], s=180, edgecolors='#333', linewidth=1.2, zorder=4)
            ax.annotate(t.replace('_','\n')[:20], (xs[i], ys[i]), textcoords="offset points", xytext=(8, 10), fontsize=7)
        wins = sum(1 for yv, xv in zip(ys, xs) if yv > xv)
        ax.set_xlabel('Best Single Model R²'); ax.set_ylabel(f'{stk_key} R²')
        ax.set_title(f'{dname}: Stacking wins {wins}/{len(traits)}', fontsize=12, fontweight='bold')
        ax.set_xlim(mn, mx); ax.set_ylim(mn, mx); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir/'03_stacking_gain_scatter.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  -> 03_stacking_gain_scatter.png")

    # --- Fig 07: Combined ranking ---
    combined = {}
    for m in [x for x in all_common if not _is_random_model(x)]:
        v = np.mean([_mean_r2(d, m) for _, d, _ in DATASETS])
        if min(_mean_r2(d, m) for _, d, _ in DATASETS) > -1:
            combined[m] = v
    sorted_all = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    y_pos = range(len(sorted_all))
    models_r = [s[0] for s in sorted_all]; vals_r = [s[1] for s in sorted_all]
    bars = ax.barh(y_pos, vals_r, 0.7, color=[_model_color(m) for m in models_r],
                   edgecolor='white', linewidth=1, zorder=3)
    for i in range(min(3, len(sorted_all))): bars[i].set_edgecolor('#C62828'); bars[i].set_linewidth(2.5)
    for i, (m, v) in enumerate(zip(models_r, vals_r)):
        ax.text(v+0.005, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
        tag = 'Trad' if m in TRAD_NAMES else ('Ensemble' if 'Stacking' in m or 'Ensemble' in m else 'DL')
        ax.text(-0.35, i, f'{m} [{tag}]', va='center', ha='right', fontsize=9,
                fontweight='bold' if 'Stacking' in m or 'Ensemble' in m else 'normal')
    ax.set_yticks([]); ax.set_xlabel('Mean R² (all-dataset avg)', fontsize=12)
    ax.set_title('Multi-Dataset Combined Ranking', fontsize=15, fontweight='bold')
    ax.grid(axis='x', alpha=0.25); ax.set_xlim(-0.75, max(vals_r)+0.08 if vals_r else 1); ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fig_dir/'07_combined_ranking.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  -> 07_combined_ranking.png")

    # --- Figs 08-10: Per-trait bar charts (one figure per dataset, each trait = subplot) ---
    _per_trait_bar_figures(DATASETS, fig_dir)
    print("[plot] Bar chart figures done.")


def _per_trait_bar_figures(DATASETS, fig_dir):
    """Generate per-trait bar chart figures (one figure per dataset)."""
    import matplotlib.pyplot as plt

    layout = {spec['tag']: (spec.get('bar_prefix', spec.get('fig', 'xx')),
                            spec['ncols'], spec['figsize'], spec['sub'])
              for spec in _dataset_specs()}
    for dname, data, traits in DATASETS:
        fig_num, ncols, fsize, sub = layout.get(dname, ('xx', 2, (24, 18), dname.lower()))
        nrows = (len(traits) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=fsize)
        axes_arr = axes.flatten()

        # Base models only (no stacking/ensemble)
        base_models = _filter_models_for_arm(
            [m for m in data[traits[0]].keys() if _is_single_model(m)],
            arm='gwas', include_ensemble=False)
        for idx, trait in enumerate(traits):
            ax = axes_arr[idx]
            r2s = {m: data[trait][m]['R2'] for m in base_models if m in data[trait]}
            sorted_m = sorted(r2s, key=r2s.get, reverse=True)
            vals = [r2s[m] for m in sorted_m]
            colors = [_model_color(m) for m in sorted_m]
            x = np.arange(len(sorted_m))
            bars = ax.bar(x, vals, 0.65, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
            if vals:
                best_idx = np.argmax(vals)
                bars[best_idx].set_edgecolor('#C62828'); bars[best_idx].set_linewidth(2.5)
                ax.text(best_idx, vals[best_idx] + 0.03, f'{vals[best_idx]:.3f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color='#C62828')
                ax.set_title(f'{trait}  (best: {sorted_m[best_idx]}={vals[best_idx]:.3f})',
                             fontsize=9, fontweight='bold')
            ax.axhline(y=0, color='#666', linewidth=0.8)
            ax.set_xticks(x); ax.set_xticklabels(sorted_m, rotation=55, ha='right', fontsize=5.5)
            ax.set_ylabel('R²'); ax.grid(axis='y', alpha=0.2)
            y_min = min(-0.5, min(vals)-0.1) if vals else -0.5
            ax.set_ylim(y_min, max(vals)+0.15 if vals else 1)
        for idx in range(len(traits), len(axes_arr)):
            axes_arr[idx].axis('off')
        fig.suptitle(f'{dname}: Per-Trait Model R² Comparison', fontsize=14, fontweight='bold')
        fig.tight_layout()
        out = fig_dir / f'{fig_num}_per_trait_bars.png'
        fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  -> {out.name}")


def generate_scatter_plots(fig_dir=None, datasets=None):
    """Generate top-4 per-trait predicted-vs-true scatter plots for GWAS and random arms."""
    import matplotlib.pyplot as plt

    SCRIPT_DIR = Path(__file__).resolve().parent
    fig_dir = Path(fig_dir) if fig_dir else SCRIPT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n[plot] Generating top-4 per-trait scatter figures from OOF predictions...")
    generated = 0
    N_TOP = 4

    DATASETS = datasets if datasets is not None else _load_plot_datasets()

    for dname, results_d, traits in DATASETS:
        spec = _dataset_spec_by_tag(dname)
        sub = spec['sub']
        oof_dir = SCRIPT_DIR / "results" / f"{sub}_ensemble" / "oof_predictions"
        if not oof_dir.exists():
            continue
        for arm, suffix in [('gwas', ''), ('random', '_random')]:
            existing_npz = [oof_dir / f"{trait}{suffix}_oof.npz"
                            for trait in traits
                            if (oof_dir / f"{trait}{suffix}_oof.npz").exists()]
            if not existing_npz:
                continue
            nrows = len(traits)
            ncols = N_TOP
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.3, max(1, nrows) * 3.6), squeeze=False)
            fig.suptitle(f'{dname}: Top-{N_TOP} Single-Model OOF Predictions ({arm.upper()} SNPs)',
                         fontsize=14, fontweight='bold')
            any_trait = False
            for row_idx, trait in enumerate(traits):
                npz_path = oof_dir / f"{trait}{suffix}_oof.npz"
                if not npz_path.exists():
                    for ci in range(ncols):
                        axes[row_idx, ci].axis('off')
                    continue
                npz_data = np.load(npz_path, allow_pickle=True)
                y_true = npz_data['_y_true']
                single_r2 = {
                    m: results_d[trait][m]['R2']
                    for m in results_d[trait]
                    if _is_single_model(m)
                    and ((_is_random_model(m) and arm == 'random') or ((not _is_random_model(m)) and arm == 'gwas'))
                    and m in npz_data
                }
                top_models = sorted(single_r2, key=single_r2.get, reverse=True)[:N_TOP]
                if top_models:
                    any_trait = True
                yt_min, yt_max = float(y_true.min()), float(y_true.max())
                for col_idx, mname in enumerate(top_models):
                    ax = axes[row_idx, col_idx]
                    oof = np.asarray(npz_data[mname], dtype=np.float32)
                    ax.scatter(y_true, oof, alpha=0.4, s=6, c=_model_color(mname), edgecolors='none', zorder=3)
                    mn = min(yt_min, float(oof.min()))
                    mx = max(yt_max, float(oof.max()))
                    pad = (mx - mn) * 0.08
                    ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], '--', color='#E53935', alpha=0.4, lw=1.0)
                    ax.set_title(f'{mname}\nR²={single_r2[mname]:.3f}', fontsize=7, fontweight='bold')
                    ax.set_xlabel('True', fontsize=6)
                    ax.set_ylabel('Predicted', fontsize=6)
                    ax.tick_params(labelsize=5)
                    ax.grid(alpha=0.2)
                for col_idx in range(len(top_models), ncols):
                    axes[row_idx, col_idx].axis('off')
            if any_trait:
                fig.tight_layout()
                fig_prefix = spec.get('scatter_prefix', spec.get('fig', f'xx_{sub}'))
                out_path = fig_dir / f'{fig_prefix}_{arm}_top4_scatter.png'
                fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"  -> {out_path.name}")
                generated += 1
            else:
                plt.close()

    if generated:
        print("[plot] Scatter plots done.")
    else:
        print("[plot] No OOF NPZ files found — run full CV first to generate scatter plots.")


def generate_efficiency_plots(fig_dir=None, include_random=False, datasets=None):
    """Generate model efficiency comparison figures.

    Fig 12: per-fold runtime comparison (3 subplots, one per dataset).
    Fig 13: model parameter count comparison (single chart, log-scale).
    """
    import matplotlib.pyplot as plt

    SCRIPT_DIR = Path(__file__).resolve().parent
    fig_dir = Path(fig_dir) if fig_dir else SCRIPT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    DATASETS = datasets if datasets is not None else _load_plot_datasets()

    if not DATASETS:
        print("[plot] No results for efficiency plots, skipping.")
        return

    print("\n[plot] Generating efficiency comparison figures (12-13)...")

    # --- Fig 12: Per-fold runtime ---
    n = len(DATASETS)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 8))
    if n == 1:
        axes = [axes]
    for ax_idx, (dname, data, traits) in enumerate(DATASETS):
        ax = axes[ax_idx]
        base = [m for m in data[traits[0]].keys()
                if _is_single_model(m) and (include_random or not _is_random_model(m))]
        avg_time = {}
        for m in base:
            times = [data[t][m].get('Time', 0) for t in traits if m in data[t]]
            avg_time[m] = np.mean(times) if times else 0
        sorted_m = sorted(avg_time, key=avg_time.get, reverse=True)
        vals = [avg_time[m] for m in sorted_m]
        colors = [_model_color(m) for m in sorted_m]
        y_pos = np.arange(len(sorted_m))
        ax.barh(y_pos, vals, 0.7, color=colors, edgecolor='white', linewidth=0.8, zorder=3)
        for i, (m, v) in enumerate(zip(sorted_m, vals)):
            label = f'{v:.1f}s' if v < 60 else f'{v/60:.1f}min'
            ax.text(v + (max(vals) if vals else 1) * 0.01, i, label, va='center', fontsize=6.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_m, fontsize=7)
        ax.set_xlabel('Time per fold (seconds)')
        ax.set_title(f'{dname} ({len(traits)} traits)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    fig.suptitle('Model Runtime Comparison (5-fold CV, per-fold average)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_dir / '12_time_comparison.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  -> 12_time_comparison.png")

    # --- Fig 13: Parameter count ---
    all_models = {m for _, data, _ in DATASETS
                  for m in data[list(data.keys())[0]]
                  if _is_single_model(m) and (include_random or not _is_random_model(m))}

    param_counts = {}
    for m in all_models:
        counts = []
        for _, data, traits in DATASETS:
            if m in data[traits[0]]:
                counts.append(data[traits[0]][m].get('Params', 0))
        param_counts[m] = max(counts) if counts else 0

    sorted_m = sorted(param_counts, key=param_counts.get, reverse=True)
    vals = [param_counts[m] for m in sorted_m]
    colors = [_model_color(m) for m in sorted_m]

    fig, ax = plt.subplots(figsize=(14, 9))
    y_pos = np.arange(len(sorted_m))
    ax.barh(y_pos, vals, 0.7, color=colors, edgecolor='white', linewidth=0.8, zorder=3)
    ax.set_xscale('log')
    for i, (m, v) in enumerate(zip(sorted_m, vals)):
        if v >= 1000000:
            label = f'{v/1000000:.1f}M'
        elif v >= 1000:
            label = f'{v/1000:.0f}K'
        else:
            label = str(int(v))
        ax.text(v * 1.05, i, label, va='center', fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_m, fontsize=9)
    ax.set_xlabel('Number of Parameters (log scale)')
    ax.set_title('Model Parameter Count Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fig_dir / '13_params_comparison.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  -> 13_params_comparison.png")

    # --- Fig 14: GPU memory (actual measured, with param-based fallback) ---
    gpu_mem_mb = {}
    for m in all_models:
        mems = []
        for _, data, traits in DATASETS:
            if m in data[traits[0]]:
                mems.append(data[traits[0]][m].get('GPUMem', 0.0))
        gpu_mem_mb[m] = max(mems) if mems else 0.0

    # Fallback: if no actual GPU mem recorded, estimate from params
    for m in all_models:
        if gpu_mem_mb[m] == 0.0 and param_counts.get(m, 0) > 0:
            p = param_counts[m]
            gpu_mem_mb[m] = p * 16 / (1024 * 1024)  # estimated training mem

    sorted_m_mem = sorted(gpu_mem_mb, key=gpu_mem_mb.get, reverse=True)
    vals_mem = [gpu_mem_mb[m] for m in sorted_m_mem]
    colors_mem = [_model_color(m) for m in sorted_m_mem]

    fig, ax = plt.subplots(figsize=(14, 9))
    y_pos = np.arange(len(sorted_m_mem))
    ax.barh(y_pos, vals_mem, 0.7, color=colors_mem, edgecolor='white', linewidth=0.8, zorder=3)
    for i, (m, v) in enumerate(zip(sorted_m_mem, vals_mem)):
        if v >= 1000:
            label = f'{v/1000:.1f} GB'
        elif v >= 1:
            label = f'{v:.0f} MB'
        else:
            label = '<1 MB (≈ params × 16 B / 1024²)'
        ax.text(v + (max(vals_mem) if vals_mem else 1) * 0.01, i, label, va='center', fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_m_mem, fontsize=9)
    ax.set_xlabel('GPU Memory (MB) — measured via torch.cuda.max_memory_allocated')
    ax.set_title('GPU Memory Usage per Model During Training', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    if any(v > 0 for v in vals_mem):
        ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(fig_dir / '14_memory_comparison.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  -> 14_memory_comparison.png")
    print("[plot] Efficiency plots done.")


def generate_all_plots(fig_dir=None, include_scatter=True, include_efficiency=True, fail_soft=True):
    """Generate all standard figures while loading result JSONs only once.

    Returns ``True`` when every requested plotting function completed without
    raising, otherwise ``False``.  Non-exception skips (for example missing OOF
    NPZ files) remain normal successful outcomes.
    """
    datasets = _load_plot_datasets()
    ok = True

    def _call(label, fn, *args, **kwargs):
        nonlocal ok
        if fail_soft:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                ok = False
                print(f"  [WARNING] {label} generation failed: {e}")
                return None
        return fn(*args, **kwargs)

    _call('Bar chart', generate_bar_charts, fig_dir=fig_dir, datasets=datasets)
    if include_scatter:
        _call('Scatter plot', generate_scatter_plots, fig_dir=fig_dir, datasets=datasets)
    if include_efficiency:
        _call('Efficiency plot', generate_efficiency_plots, fig_dir=fig_dir, datasets=datasets)
    return ok



# ============================================================================
# Section M: Dispatcher
# ============================================================================

if __name__ == '__main__':
    crop = sys.argv[1] if len(sys.argv) > 1 else ''
    full_mode = '--full' in sys.argv
    no_plots = '--no-plots' in sys.argv
    plot_only = '--plot-only' in sys.argv or crop == 'plot'

    if crop not in ('wheat', 'wheat2000', 'rice', 'maize', 'soybean', 'wheatgabi', 'all', 'plot'):
        print("Usage: python genomic_ensemble.py <wheat|wheat2000|rice|maize|soybean|wheatgabi|all|plot> [--full] [--no-plots] [--plot-only]")
        print("  wheat     — Run wheat ensemble pipeline (VCF data)")
        print("  wheat2000 — Run wheat2000 ensemble pipeline (CSV data, 2000×33K)")
        print("  rice      — Run rice ensemble pipeline")
        print("  maize     — Run maize ensemble pipeline")
        print("  soybean   — Run soybean SoySNP50K ensemble pipeline")
        print("  wheatgabi — Run wheat GABI iSELECT 90k ensemble pipeline")
        print("  all       — Run all six crop/dataset pipelines")
        print("  plot      — Generate figures only")
        print("  --full    — Full mode (all traits x 5 folds)")
        print("  --no-plots   — Do not auto-generate figures after full run")
        print("  --plot-only  — Generate figures only")
        sys.exit(1)

    print(f"\n{'#'*80}")
    print(f"  Genomic Prediction Ensemble — {crop.upper()}")
    print(f"  Mode: {'FULL' if full_mode else 'QUICK TEST'}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")

    exit_code = 0

    def _run_all_plots():
        return generate_all_plots(fail_soft=True)

    if plot_only:
        print(f"\n{'#'*80}")
        print("  Generating visualization figures only...")
        print(f"{'#'*80}")
        if not _run_all_plots():
            exit_code = 2
    elif crop == 'all':
        import subprocess
        procs = []
        all_crops = ['wheat', 'wheat2000', 'rice', 'maize', 'soybean', 'wheatgabi']
        for c in all_crops:
            cmd = [sys.executable, __file__, c]
            if full_mode:
                cmd.append('--full')
            cmd.append('--no-plots')
            print(f"  Launching subprocess: {' '.join(cmd)}")
            procs.append(subprocess.Popen(cmd))
        for i, p in enumerate(procs):
            p.wait()
            print(f"  Subprocess {all_crops[i]} finished (rc={p.returncode})")
            if p.returncode:
                exit_code = p.returncode if exit_code == 0 else exit_code
        if full_mode and not no_plots:
            print(f"\n{'#'*80}")
            print("  Generating visualization figures once after all subprocesses...")
            print(f"{'#'*80}")
            if not _run_all_plots() and exit_code == 0:
                exit_code = 2
    else:
        if crop == 'wheat': run_wheat(quick_test=not full_mode)
        if crop == 'wheat2000': run_wheat2000(quick_test=not full_mode)
        if crop == 'rice': run_rice(quick_test=not full_mode)
        if crop == 'maize': run_maize(quick_test=not full_mode)
        if crop == 'soybean': run_soybean(quick_test=not full_mode)
        if crop == 'wheatgabi': run_wheat_gabi(quick_test=not full_mode)

    # Auto-generate visualization figures in full mode for single-dataset runs.
    if full_mode and not no_plots and not plot_only and crop != 'all':
        print(f"\n{'#'*80}")
        print(f"  Generating visualization figures...")
        print(f"{'#'*80}")
        if not _run_all_plots() and exit_code == 0:
            exit_code = 2

    print(f"\n{'#'*80}")
    if exit_code:
        print(f"  Finished with failures (exit={exit_code}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"  All done! Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")
    sys.exit(exit_code)
