"""
Rice Genomic Prediction — FGN + EFM + MICNN Ensemble
=====================================================
"局部-全局-统计" 正交集成体系

改进:
  FGN v2:  FFT + 可学习 Haar 小波双频域路径
  EFM v2:  + BatchNorm(SNP级) + Dropout 正则化
  MICNN v2: + Dilated Conv + Spatial Pyramid Pooling

集成:
  FusionNet: 三分支特征级硬融合 (联合训练)
  Stacking:  预测级软融合 (RidgeCV 元学习器)

对照: 保留 FGN/EFM/MICNN 原版作为基线
"""

import json, time, pickle, shutil
import numpy as np
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
from genomic_nn_models import (FGNEncoder, PreFGN, pretrain_prefgn, pretrain_prefgn_v2)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Config
# ============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "results" / "rice_data"
OUTPUT_DIR = _SCRIPT_DIR / "results" / "rice_ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS = 5
GWAS_TOP_K = 5000
MAF_THRESHOLD = 0.05  # 预过滤: 剔除 minor allele frequency < 5% 的稀有位点
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TYPE_TRAD = 'Traditional'
TYPE_DL = 'DL'
TYPE_ENS = 'Ensemble'
TYPE_HYBRID = 'Hybrid'

# True=仅1个性状x2折本地测试, False=完整实验
# 命令行: python rice_models_ensemble.py --full  覆盖为完整模式
QUICK_TEST = True

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

TRAITS = [
    'Heading_date', 'Plant_height', 'Num_panicles',
    'Num_effective_panicles', 'Yield', 'Grain_weight',
    'Spikelet_length', 'Grain_length', 'Grain_width', 'Grain_thickness'
]

# ============================================================================
# 训练工具
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

    dl = DataLoader(TensorDataset(Xt, yt),
                    batch_size=min(batch_size, len(tr_idx)), shuffle=True)

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
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model.cpu()


def predict_torch_model(model, X):
    model = model.to(DEVICE)
    Xt = torch.FloatTensor(X).to(DEVICE)
    model.eval()
    with torch.no_grad():
        return model(Xt).squeeze().cpu().numpy()


# ============================================================================
# 原版模型 (基线对照)
# ============================================================================

class FourierGenomicNet(nn.Module):
    """FGN -- FFT频域+时域双路径"""
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


class EpistaticFM(nn.Module):
    """EFM -- 线性+FM二阶交互+深度MLP"""
    def __init__(self, n_snps, k=8, hidden=64, dropout=0.35):
        super().__init__()
        self.linear = nn.Linear(n_snps, 1, bias=True)
        self.V = nn.Parameter(torch.randn(n_snps, k) * 0.01)
        self.deep = nn.Sequential(
            nn.Linear(n_snps, hidden*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(1+1+hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        lo = self.linear(x)
        xv = x.unsqueeze(2) * self.V.unsqueeze(0)
        fm = 0.5 * (xv.sum(1).pow(2) - (xv.pow(2)).sum(1)).sum(1, keepdim=True)
        return self.head(torch.cat([lo, fm, self.deep(x)], dim=1))


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
    """MICNN -- 多尺度Inception+SE"""
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


# ============================================================================
# 改进版模型
# ============================================================================

class HaarWaveletDecomp(nn.Module):
    """可学习Haar小波分解 -- Conv1d实现, GPU原生, 无需pywt"""
    def __init__(self):
        super().__init__()
        s2 = np.sqrt(2)
        self.lo = nn.Parameter(torch.tensor([[[1., 1.]]]) / s2)
        self.hi = nn.Parameter(torch.tensor([[[1., -1.]]]) / s2)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return F.conv1d(x, self.lo, stride=2), F.conv1d(x, self.hi, stride=2)


class FGNv2(nn.Module):
    """FGN v2: FFT + Wavelet 双频域 + 时域残差"""
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


class EFMv2(nn.Module):
    """EFM v2: +FM Embedding Dropout (核心改进: 防止FM隐向量过拟合)"""
    def __init__(self, n_snps, k=8, hidden=64, dropout=0.35, fm_do=0.15):
        super().__init__()
        self.linear = nn.Linear(n_snps, 1, bias=True)
        self.V = nn.Parameter(torch.randn(n_snps, k) * 0.01)
        self.fm_do = fm_do
        self.deep = nn.Sequential(
            nn.Linear(n_snps, hidden*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(1+1+hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        lo = self.linear(x)
        Vd = F.dropout(self.V, p=self.fm_do, training=self.training)
        xv = x.unsqueeze(2) * Vd.unsqueeze(0)
        fm = 0.5 * (xv.sum(1).pow(2) - (xv.pow(2)).sum(1)).sum(1, keepdim=True)
        return self.head(torch.cat([lo, fm, self.deep(x)], dim=1))


class EFMv3(nn.Module):
    """EFM v3: Sparse-gated FM + LayerNorm, k=4, top-50% SNP gate"""
    def __init__(self, n_snps, k=4, hidden=64, dropout=0.35):
        super().__init__()
        self.linear = nn.Linear(n_snps, 1, bias=True)
        self.V = nn.Parameter(torch.randn(n_snps, k) * 0.005)
        self.fm_ln = nn.LayerNorm(k)
        self.deep = nn.Sequential(
            nn.Linear(n_snps, hidden*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(1+k+hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        lo = self.linear(x)
        with torch.no_grad():
            w = self.linear.weight.abs().squeeze()
            thresh = torch.quantile(w, 0.5)
            mask = (w >= thresh).float()
        x_gated = x * mask.unsqueeze(0)
        xv = x_gated.unsqueeze(2) * self.V.unsqueeze(0)
        fm = 0.5 * (xv.sum(1).pow(2) - (xv.pow(2)).sum(1))
        fm = self.fm_ln(fm)
        return self.head(torch.cat([lo, fm, self.deep(x)], dim=1))


class FGNv3(nn.Module):
    """FGN v3: composes FGNEncoder + regression head"""
    def __init__(self, n_snps, hidden=48, dropout=0.35):
        super().__init__()
        self.encoder = FGNEncoder(n_snps, hidden, dropout)
        total = self.encoder.out_dim
        self.head = nn.Sequential(
            nn.Linear(total, hidden*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.head(self.encoder(x))



class DilatedInceptionBlock(nn.Module):
    """带空洞卷积的Inception: k7(标准), k15(标准), k7d2(空洞), k7d4(空洞)"""
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
    """1D SPP: 多尺度自适应池化拼接"""
    def __init__(self, bins=(1, 2, 4, 8)):
        super().__init__()
        self.bins = bins

    def forward(self, x):
        B = x.size(0)
        return torch.cat([F.adaptive_avg_pool1d(x, b).view(B, -1) for b in self.bins], dim=1)


class MICNNv2(nn.Module):
    """MICNN v2: +DilatedConv +SPP替代全局池化"""
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


# ============================================================================
# 集成方案 A: FusionNet -- 特征级硬融合 (联合训练)
# ============================================================================

class FusionNet(nn.Module):
    """三分支特征级融合: FGN(频域) + EFM(统计) + MICNN(局部)"""
    def __init__(self, n_snps, hidden_dim=48, dropout=0.35):
        super().__init__()
        # FGN branch (轻量)
        self.n_freq = n_snps // 2 + 1
        self.spec_r = nn.Parameter(torch.randn(1, 16, self.n_freq) * 0.02)
        self.spec_i = nn.Parameter(torch.randn(1, 16, self.n_freq) * 0.02)
        self.fgn_conv = nn.Sequential(
            nn.Conv1d(16, hidden_dim, 7, padding=3), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(dropout*0.3))
        self.fgn_pool = nn.AdaptiveAvgPool1d(1)

        # EFM branch (轻量)
        self.efm_linear = nn.Linear(n_snps, 1)
        self.V_f = nn.Parameter(torch.randn(n_snps, 4) * 0.01)
        self.efm_deep = nn.Sequential(
            nn.Linear(n_snps, hidden_dim*2), nn.BatchNorm1d(hidden_dim*2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))

        # MICNN branch (轻量)
        self.micnn_stem = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 7, padding=3), nn.BatchNorm1d(hidden_dim), nn.GELU())
        self.micnn_block = DilatedInceptionBlock(hidden_dim, hidden_dim, dropout)
        self.micnn_se = SEBlock(hidden_dim)
        self.micnn_spp = SpatialPyramidPool1D(bins=(1, 2, 4))
        micnn_dim = hidden_dim * 7

        # Fusion head
        fusion_in = hidden_dim + (hidden_dim + 2) + micnn_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim*3), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*3, hidden_dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, 1))

    def forward(self, x):
        # FGN
        xc = torch.fft.rfft(x, dim=1)
        xr = xc.real.unsqueeze(1).expand(-1, 16, -1) * self.spec_r
        xi = xc.imag.unsqueeze(1).expand(-1, 16, -1) * self.spec_i
        fgn_f = self.fgn_pool(self.fgn_conv(xr+xi)).squeeze(-1)

        # EFM (加性+FM交互+深度特征)
        lo = self.efm_linear(x)
        Vd = F.dropout(self.V_f, p=0.1, training=self.training)
        xv = x.unsqueeze(2) * Vd.unsqueeze(0)
        fm = 0.5 * (xv.sum(1).pow(2) - (xv.pow(2)).sum(1)).sum(1, keepdim=True)
        efm_f = torch.cat([lo, fm, self.efm_deep(x)], dim=1)

        # MICNN
        m = self.micnn_stem(x.unsqueeze(1))
        m = self.micnn_se(self.micnn_block(m))
        micnn_f = self.micnn_spp(m)

        return self.fusion_head(torch.cat([fgn_f, efm_f, micnn_f], dim=1))


# ============================================================================
# GWAS 筛选
# ============================================================================

def gwas_select(X, y, top_k):
    """GWAS SNP筛选 (Pearson相关近似)"""
    y_c = y - y.mean()
    X_c = X - X.mean(axis=0)
    num = np.dot(y_c, X_c)
    denom = np.std(y_c) * len(y) * np.sqrt(np.sum(X_c**2, axis=0) + 1e-12)
    return np.argsort(np.abs(num / denom))[-top_k:]


def maf_filter(X, threshold=MAF_THRESHOLD):
    """预过滤稀有位点: 剔除 MAF < threshold 的 SNP (per-fold, no leakage)."""
    af = X.mean(axis=0) / 2.0
    maf = np.minimum(af, 1.0 - af)
    return np.where(maf >= threshold)[0]


# ============================================================================
# 传统模型 (RRBLUP, GBLUP, XGBoost, ElasticNet, GWAS_RRBLUP)
# ============================================================================

class RRBLUP:
    def __init__(self):
        self.model = RidgeCV(alphas=np.logspace(-3, 3, 30))

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class GBLUP:
    def __init__(self):
        self.model = RidgeCV(alphas=np.logspace(-3, 3, 20))

    def fit(self, G_train, y_train):
        self.model.fit(G_train, y_train)
        return self

    def predict(self, G_test_train):
        return self.model.predict(G_test_train)


class XGBoostModel:
    def __init__(self, n_estimators=500, max_depth=6, lr=0.05):
        self.params = {
            'n_estimators': n_estimators, 'max_depth': max_depth,
            'learning_rate': lr, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'random_state': RANDOM_SEED, 'n_jobs': 8, 'verbosity': 0
        }

    def fit(self, X, y):
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class ElasticNetModel:
    def __init__(self):
        self.model = ElasticNetCV(
            l1_ratio=[.1, .5, .7, .9, .95, 1],
            alphas=np.logspace(-4, 2, 20),
            cv=3, random_state=RANDOM_SEED, max_iter=5000, n_jobs=8
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class GWASWeightedRRBLUP:
    def __init__(self):
        self.weights = None
        self.model = RidgeCV(alphas=np.logspace(-3, 3, 30))

    def fit(self, X, y):
        y_c = y - y.mean()
        X_c = X - X.mean(axis=0)
        denom = np.std(y_c) * len(y) * np.sqrt(np.sum(X_c ** 2, axis=0) + 1e-12)
        self.weights = np.abs(np.dot(y_c, X_c) / denom)
        self.weights = self.weights / self.weights.mean()
        X_weighted = X * self.weights[None, :]
        self.model.fit(X_weighted, y)
        return self

    def predict(self, X):
        X_weighted = X * self.weights[None, :]
        return self.model.predict(X_weighted)


def evaluate_traditional_stacking(oof_trad, y):
    trad_names = list(oof_trad.keys())
    X_meta = np.column_stack([oof_trad[m] for m in trad_names])

    meta = RidgeCV(alphas=np.logspace(-3, 3, 20), fit_intercept=True, cv=5)
    meta.fit(X_meta, y)

    kf = KFold(n_splits=min(5, len(trad_names)), shuffle=True, random_state=RANDOM_SEED)
    sp = np.zeros(len(y))
    for tr, te in kf.split(X_meta):
        m = RidgeCV(alphas=np.logspace(-3, 3, 20), fit_intercept=True, cv=3)
        m.fit(X_meta[tr], y[tr])
        sp[te] = m.predict(X_meta[te])

    return {
        'R2': float(r2_score(y, sp)),
        'Correlation': float(pearsonr(y, sp)[0]),
        'Meta_weights': meta.coef_.tolist(),
        'Meta_intercept': float(meta.intercept_),
        'Base_models': trad_names
    }


# ============================================================================
# 模型工厂
# ============================================================================

def create_model(name, n_snps, overrides=None):
    """按名称创建单个模型。overrides: 可选超参覆盖 dict"""
    o = overrides or {}
    if name == 'FGN':
        return FourierGenomicNet(n_snps=n_snps, hidden=64, dropout=0.35)
    if name == 'EFM':
        return EpistaticFM(n_snps=n_snps, k=8, hidden=64, dropout=0.35)
    if name == 'MICNN':
        return MultiScaleInceptionCNN(n_snps=n_snps, hidden=48, dropout=0.35)
    if name == 'FGN v2':
        return FGNv2(n_snps=n_snps, hidden=64, dropout=0.35)
    if name == 'EFM v2':
        return EFMv2(n_snps=n_snps, k=8, hidden=64, dropout=0.35, fm_do=0.1)
    if name == 'MICNN v2':
        return MICNNv2(n_snps=n_snps, hidden=40, dropout=0.35, spp_bins=(1, 2, 4))
    if name == 'FGN v3':
        return FGNv3(n_snps=n_snps, hidden=o.get('hidden', 48),
                     dropout=o.get('dropout', 0.35))
    if name == 'EFM v3':
        return EFMv3(n_snps=n_snps, k=o.get('k', 4),
                     hidden=o.get('hidden', 64),
                     dropout=o.get('dropout', 0.35))
    if name == 'FusionNet':
        return FusionNet(n_snps=n_snps,
                         hidden_dim=o.get('hidden_dim', 48),
                         dropout=o.get('dropout', 0.35))
    if name == 'PreFGN':
        return PreFGN(n_snps=n_snps, hidden=o.get('hidden', 64),
                      dropout=o.get('dropout', 0.35))
    raise ValueError(f"Unknown model: {name}")


# ============================================================================
# 集成方案 B: Stacking (RidgeCV元学习器)
# ============================================================================

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def fit_resfgn_components(X, y, n_snps, cv=3, fusion_overrides=None):
    """Train RidgeCV + FusionNet on Ridge residuals. Returns (ridge, fusion_model)."""
    o = fusion_overrides or {}
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=cv)
    ridge.fit(X, y)
    residuals = y - ridge.predict(X)
    fusion = FusionNet(n_snps=n_snps,
                       hidden_dim=o.get('hidden_dim', 48),
                       dropout=o.get('dropout', 0.35)).to(DEVICE)
    fusion = train_torch_model(
        fusion, X, residuals,
        epochs=300, batch_size=64,
        lr=o.get('lr', 1e-3),
        weight_decay=o.get('weight_decay', 1e-3),
        patience=o.get('patience', 30))
    return ridge, fusion


def tune_model_hyperparams(model_name, X_train, y_train, n_snps, n_trials=15):
    """Optuna tune a model on given data. Returns (best_params, best_val_r2)."""
    try:
        import optuna
    except ImportError:
        print(f"    [SKIP] Optuna not installed, using defaults for {model_name}")
        return {}, 0.0

    n_val = max(16, int(len(y_train) * 0.2))
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(y_train))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    def objective(trial):
        if model_name == 'FGN v3':
            overrides = {
                'hidden': trial.suggest_categorical('hidden', [32, 48, 64]),
                'dropout': trial.suggest_float('dropout', 0.2, 0.5),
                'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
                'patience': trial.suggest_int('patience', 20, 50),
            }
            model = FGNv3(n_snps=n_snps, hidden=overrides['hidden'],
                          dropout=overrides['dropout'])
            bs = 128
        elif model_name == 'EFM v3':
            overrides = {
                'k': trial.suggest_categorical('k', [3, 4, 6]),
                'hidden': trial.suggest_categorical('hidden', [48, 64, 96]),
                'dropout': trial.suggest_float('dropout', 0.2, 0.5),
                'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
            }
            model = EFMv3(n_snps=n_snps, k=overrides['k'],
                          hidden=overrides['hidden'], dropout=overrides['dropout'])
            bs = 128
        elif model_name == 'FusionNet':
            overrides = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 48, 64]),
                'dropout': trial.suggest_float('dropout', 0.2, 0.5),
                'lr': trial.suggest_float('lr', 5e-4, 3e-3, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
                'patience': trial.suggest_int('patience', 20, 50),
            }
            model = FusionNet(n_snps=n_snps, hidden_dim=overrides['hidden_dim'],
                              dropout=overrides['dropout'])
            bs = 64
        else:
            raise ValueError(f"Unknown model for tuning: {model_name}")

        model = train_torch_model(
            model, X_tr, y_tr,
            epochs=300, batch_size=bs,
            lr=overrides.get('lr', 2e-3),
            weight_decay=overrides.get('weight_decay', 1e-3),
            patience=overrides.get('patience', 30))
        preds = predict_torch_model(model, X_val)
        return float(r2_score(y_val, preds))

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, study.best_value


def stacking_evaluate(oof_preds_dict, targets, n_folds=5):
    """Stacking评估: 各模型OOF预测作为元特征, RidgeCV融合"""
    base_names = list(oof_preds_dict.keys())
    X_meta = np.column_stack([oof_preds_dict[m] for m in base_names])

    meta = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=5)
    meta.fit(X_meta, targets)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    sp = np.zeros(len(targets))
    for tr, te in kf.split(X_meta):
        m = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=3)
        m.fit(X_meta[tr], targets[tr])
        sp[te] = m.predict(X_meta[te])

    return {
        'R2': float(r2_score(targets, sp)),
        'Correlation': float(pearsonr(targets, sp)[0]),
        'Meta_weights': meta.coef_.tolist(),
        'Meta_intercept': float(meta.intercept_),
        'Base_models': base_names
    }


# ============================================================================
# 主实验
# ============================================================================

def main():
    import sys
    quick_test = QUICK_TEST
    if '--full' in sys.argv:
        quick_test = False
        print("[CMD] --full: 切换到完整实验模式")

    print(f"Device: {DEVICE}  |  Quick test: {quick_test}")
    print(f"{'='*80}")
    print("Rice Genomic Prediction -- FGN+EFM+MICNN Ensemble System")
    print(f"{'='*80}")

    print("\nLoading data ...")
    data = np.load(DATA_DIR / "genotype_matrix.npz", allow_pickle=True)
    G = data['G']
    with open(DATA_DIR / "trait_data.json") as f:
        trait_data = json.load(f)
    print(f"  Genotype: {G.shape}")

    trad_names = ['RRBLUP', 'GBLUP', 'XGBoost', 'ElasticNet', 'GWAS_RRBLUP']
    dl_base_names = ['FGN', 'EFM', 'MICNN', 'FGN v2', 'EFM v2', 'MICNN v2',
                     'FGN v3', 'EFM v3', 'PreFGN']
    extra_names = ['ResFGN']
    dl_names = dl_base_names + ['FusionNet']
    all_names = trad_names + dl_names + extra_names

    traits_run = TRAITS[:1] if quick_test else TRAITS
    folds_run = min(2, N_FOLDS) if quick_test else N_FOLDS

    if quick_test:
        print(f"  [QUICK TEST] {len(traits_run)} trait x {folds_run} folds")

    all_results = {}
    total_t0 = time.time()

    for t_idx, trait in enumerate(traits_run):
        print(f"\n{'='*80}")
        print(f"  TRAIT [{t_idx+1}/{len(traits_run)}]: {trait}")
        print(f"{'='*80}")

        td = trait_data[trait]
        idxs = td['genotype_indices']
        y = np.array(td['values']).astype(np.float32)
        X_all = G[idxs]

        n_snps = min(GWAS_TOP_K, X_all.shape[1] - 50)
        print(f"  {len(y)} samples, {X_all.shape[1]} markers -> "
              f"{n_snps} GWAS-selected (per-fold, no leakage)")

        # ── AutoML tuning (full mode only, on held-out data before CV) ──
        tuned_params = {}
        if not quick_test:
            print(f"\n  [AutoML] Tuning hyperparams for new models...")
            maf_tune = maf_filter(X_all)
            if len(maf_tune) >= n_snps:
                gidx_t = gwas_select(X_all[:, maf_tune], y, n_snps)
                X_tune = X_all[:, maf_tune][:, gidx_t]
            else:
                X_tune = X_all[:, gwas_select(X_all, y, n_snps)]
            sc_tune = StandardScaler()
            X_tune_s = sc_tune.fit_transform(X_tune).astype(np.float32)

            for tune_name in ['FGN v3', 'EFM v3', 'FusionNet']:
                best_p, best_r2 = tune_model_hyperparams(
                    tune_name, X_tune_s, y, n_snps, n_trials=15)
                tuned_params[tune_name] = best_p
                pstr = ', '.join(f'{k}={v}' for k, v in best_p.items())
                print(f"    {tune_name}: val R²={best_r2:.4f}  [{pstr}]")

        kf = KFold(n_splits=folds_run, shuffle=True, random_state=RANDOM_SEED)
        results = {m: {'preds': [], 'targets': [], 'params': 0, 'time': 0.0}
                   for m in all_names}
        oof_trad = {m: np.zeros(len(y)) for m in trad_names}
        oof_dl = {m: np.zeros(len(y)) for m in dl_base_names}

        dl_models = {m: create_model(m, n_snps) for m in dl_names}

        for fi, (tr, te) in enumerate(kf.split(X_all)):
            print(f"\n  --- Fold {fi+1}/{folds_run} ---")
            Xtr_raw, Xte_raw = X_all[tr], X_all[te]
            ytr, yte = y[tr], y[te]

            maf_idx = maf_filter(Xtr_raw)
            if len(maf_idx) >= n_snps:
                Xtr_raw = Xtr_raw[:, maf_idx]
                Xte_raw = Xte_raw[:, maf_idx]

            gidx = gwas_select(Xtr_raw, ytr, n_snps)
            Xtr = Xtr_raw[:, gidx]
            Xte = Xte_raw[:, gidx]

            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
            Xte_s = sc.transform(Xte).astype(np.float32)

            G_fold_train = Xtr_s @ Xtr_s.T / n_snps
            G_fold_te_tr = Xte_s @ Xtr_s.T / n_snps

            # ── 传统模型 ──
            trad_configs = [
                ('RRBLUP', lambda: RRBLUP(),
                 lambda m, Xs, yt: m.fit(Xs, yt),
                 lambda m, Xs: m.predict(Xs), n_snps + 1),
                ('GBLUP', lambda: GBLUP(),
                 lambda m, _x, yt: m.fit(G_fold_train, yt),
                 lambda m, _x: m.predict(G_fold_te_tr), len(tr) + 1),
                ('XGBoost', lambda: XGBoostModel(n_estimators=300),
                 lambda m, Xs, yt: m.fit(Xs, yt),
                 lambda m, Xs: m.predict(Xs), 300 * 6 * 2),
                ('ElasticNet', lambda: ElasticNetModel(),
                 lambda m, Xs, yt: m.fit(Xs, yt),
                 lambda m, Xs: m.predict(Xs), n_snps + 1),
                ('GWAS_RRBLUP', lambda: GWASWeightedRRBLUP(),
                 lambda m, Xs, yt: m.fit(Xs, yt),
                 lambda m, Xs: m.predict(Xs), n_snps * 2 + 1),
            ]
            for tname, build_fn, fit_fn, pred_fn, param_count in trad_configs:
                t0 = time.time()
                tmodel = build_fn()
                fit_fn(tmodel, Xtr_s, ytr)
                preds = pred_fn(tmodel, Xte_s)
                results[tname]['preds'].extend(preds.tolist())
                results[tname]['targets'].extend(yte.tolist())
                results[tname]['time'] += time.time() - t0
                if fi == 0:
                    results[tname]['params'] = param_count
                oof_trad[tname][te] = preds
                print(f"    {tname:<16s} R2={r2_score(yte, preds):+.4f}")

            # ── DL 模型 ──
            for mi, mname in enumerate(dl_names):
                if fi == 0:
                    results[mname]['params'] = sum(p.numel() for p in dl_models[mname].parameters())

                t0 = time.time()
                tp = tuned_params.get(mname, {})

                if mname == 'PreFGN' and not quick_test:
                    pretrain_prefgn_v2(dl_models[mname], Xtr_s, epochs=50,
                                       lr=1e-3, patience=10)

                bs = 64 if mname == 'FusionNet' else 128
                lr = tp.get('lr', 1e-3 if mname == 'FusionNet' else 2e-3)
                wd = tp.get('weight_decay', 1e-3)
                pat = tp.get('patience', 30)
                model = train_torch_model(
                    dl_models[mname], Xtr_s, ytr,
                    epochs=300, batch_size=bs, lr=lr, weight_decay=wd, patience=pat)
                preds = predict_torch_model(model, Xte_s)
                elapsed = time.time() - t0

                results[mname]['preds'].extend(preds.tolist())
                results[mname]['targets'].extend(yte.tolist())
                results[mname]['time'] += elapsed
                fold_r2 = r2_score(yte, preds)
                print(f"    {mname:<16s} R2={fold_r2:+.4f}  ({elapsed:.1f}s)")

                if mname in dl_base_names:
                    oof_dl[mname][te] = preds

                dl_models[mname] = create_model(mname, n_snps,
                                                  overrides=tuned_params.get(mname))

            # ── ResFGN: Ridge + FusionNet residual learning ──
            t0 = time.time()
            ridge, res_model = fit_resfgn_components(
                Xtr_s, ytr, n_snps, cv=3,
                fusion_overrides=tuned_params.get('FusionNet'))
            pred_te_ridge = ridge.predict(Xte_s)
            res_pred = predict_torch_model(res_model, Xte_s)
            final_pred = pred_te_ridge + res_pred
            elapsed = time.time() - t0
            if fi == 0:
                results['ResFGN']['params'] = (n_snps + 1 +
                    sum(p.numel() for p in res_model.parameters()))
            results['ResFGN']['preds'].extend(final_pred.tolist())
            results['ResFGN']['targets'].extend(yte.tolist())
            results['ResFGN']['time'] += elapsed
            print(f"    {'ResFGN':<16s} R2={r2_score(yte, final_pred):+.4f}  ({elapsed:.1f}s)")

        # ── 性状汇总 ──
        print(f"\n  {'-'*70}")
        print(f"  {trait} Final Results (5-fold CV):")
        print(f"  {'Model':<16s} {'R2':>8s} {'Corr':>8s} {'RMSE':>8s} {'Time':>8s}")
        print(f"  {'-'*70}")

        trait_res = {}
        for mname in all_names:
            p = np.array(results[mname]['preds'])
            t = np.array(results[mname]['targets'])
            r2_v = float(r2_score(t, p))
            corr_v = float(pearsonr(t, p)[0])
            rmse_v = float(np.sqrt(np.mean((p-t)**2)))
            if mname in trad_names:
                tag, mtype = " [Trad]", TYPE_TRAD
            elif mname == 'ResFGN':
                tag, mtype = " [Hybrid]", TYPE_HYBRID
            else:
                tag, mtype = " [DL]", TYPE_DL
            trait_res[mname] = {
                'R2': r2_v, 'Correlation': corr_v, 'RMSE': rmse_v,
                'Type': mtype,
                'Time': results[mname]['time']/folds_run}
            print(f"  {mname+tag:<24s} {r2_v:8.4f} {corr_v:8.4f} {rmse_v:8.4f} "
                  f"{results[mname]['time']/folds_run:7.1f}s")

        # ── 集成 ──
        if folds_run >= 3:
            # Stacking (DL): 仅 DL 6 基础模型
            sr_dl = stacking_evaluate(oof_dl, y, n_folds=min(5, folds_run))
            trait_res['Stacking (DL)'] = {
                'R2': sr_dl['R2'], 'Correlation': sr_dl['Correlation'],
                'RMSE': 0.0, 'Type': TYPE_ENS,
                'Meta_weights': sr_dl['Meta_weights'], 'Base_models': sr_dl['Base_models']}
            print(f"  {'Stacking (DL)':<24s} {sr_dl['R2']:8.4f} {sr_dl['Correlation']:8.4f}")

            # Stacking (All): 传统 5 + DL 6 = 11 基础模型
            oof_all = {**oof_trad, **oof_dl}
            sr_all = stacking_evaluate(oof_all, y, n_folds=min(5, folds_run))
            trait_res['Stacking (All)'] = {
                'R2': sr_all['R2'], 'Correlation': sr_all['Correlation'],
                'RMSE': 0.0, 'Type': TYPE_ENS,
                'Meta_weights': sr_all['Meta_weights'], 'Base_models': sr_all['Base_models']}
            print(f"  {'Stacking (All)':<24s} {sr_all['R2']:8.4f} {sr_all['Correlation']:8.4f}")
            weights_str = dict(zip(sr_all['Base_models'], [f'{w:.3f}' for w in sr_all['Meta_weights']]))
            print(f"    All weights: {weights_str}")

            # Trad Ensemble: 仅传统 5 模型 Stacking
            tsr = evaluate_traditional_stacking(oof_trad, y)
            trait_res['Trad Ensemble'] = {
                'R2': tsr['R2'], 'Correlation': tsr['Correlation'],
                'RMSE': 0.0, 'Type': TYPE_ENS,
                'Meta_weights': tsr['Meta_weights'], 'Base_models': tsr['Base_models']}
            print(f"  {'Trad Ensemble':<24s} {tsr['R2']:8.4f} {tsr['Correlation']:8.4f}")

        all_results[trait] = trait_res

        # ── 最终部署: 全量数据重训 + 保存模型 ──
        print(f"\n  Deploying models on full dataset ({len(y)} samples) ...")
        deploy_dir = OUTPUT_DIR / "deployed_models" / trait
        if deploy_dir.exists():
            shutil.rmtree(deploy_dir)
        deploy_dir.mkdir(parents=True, exist_ok=True)

        maf_full = maf_filter(X_all)
        if len(maf_full) >= n_snps:
            gidx_f = gwas_select(X_all[:, maf_full], y, n_snps)
            gidx_full = maf_full[gidx_f]
        else:
            gidx_full = gwas_select(X_all, y, n_snps)
        X_full = X_all[:, gidx_full]
        sc_full = StandardScaler()
        X_full_s = sc_full.fit_transform(X_full).astype(np.float32)

        deployment_meta = {
            'gwas_indices': gidx_full.tolist(),
            'n_snps': n_snps,
            'trad_names': trad_names,
            'dl_base_names': dl_base_names,
        }

        # Traditional models
        for tname in trad_names:
            tmodel = None
            if tname == 'RRBLUP':
                tmodel = RRBLUP().fit(X_full_s, y)
            elif tname == 'GBLUP':
                G_full = X_full_s @ X_full_s.T / n_snps
                tmodel = GBLUP().fit(G_full, y)
            elif tname == 'XGBoost':
                tmodel = XGBoostModel(n_estimators=300).fit(X_full_s, y)
            elif tname == 'ElasticNet':
                tmodel = ElasticNetModel().fit(X_full_s, y)
            elif tname == 'GWAS_RRBLUP':
                tmodel = GWASWeightedRRBLUP().fit(X_full_s, y)
            pickle.dump(tmodel, open(deploy_dir / f"{tname}.pkl", 'wb'))
            print(f"    [saved] {tname}.pkl")

        # DL models
        for mname in dl_base_names + ['FusionNet']:
            tp = tuned_params.get(mname, {})
            model = create_model(mname, n_snps, overrides=tp)

            if mname == 'PreFGN' and not quick_test:
                pretrain_prefgn_v2(model, X_full_s, epochs=50,
                                   lr=1e-3, patience=10)

            model = train_torch_model(
                model, X_full_s, y,
                epochs=300,
                batch_size=64 if mname == 'FusionNet' else 128,
                lr=tp.get('lr', 1e-3 if mname == 'FusionNet' else 2e-3),
                weight_decay=tp.get('weight_decay', 1e-3),
                patience=tp.get('patience', 30))
            torch.save(model.state_dict(), deploy_dir / f"{mname}.pt")
            print(f"    [saved] {mname}.pt")

        # ResFGN deployment
        ridge_full, res_model_full = fit_resfgn_components(
            X_full_s, y, n_snps, cv=5,
            fusion_overrides=tuned_params.get('FusionNet'))
        pickle.dump(ridge_full, open(deploy_dir / "ResFGN_ridge.pkl", 'wb'))
        torch.save(res_model_full.state_dict(), deploy_dir / "ResFGN_fusion.pt")
        print(f"    [saved] ResFGN_ridge.pkl + ResFGN_fusion.pt")

        # Stacking meta-learners
        X_meta_dl = np.column_stack([oof_dl[m] for m in dl_base_names])
        meta_dl = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=5)
        meta_dl.fit(X_meta_dl, y)
        pickle.dump(meta_dl, open(deploy_dir / "Stacking_DL_meta.pkl", 'wb'))
        deployment_meta['meta_dl_coef'] = meta_dl.coef_.tolist()

        X_meta_all = np.column_stack([oof_trad[m] for m in trad_names]
                                     + [oof_dl[m] for m in dl_base_names])
        meta_all = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=5)
        meta_all.fit(X_meta_all, y)
        pickle.dump(meta_all, open(deploy_dir / "Stacking_All_meta.pkl", 'wb'))
        deployment_meta['meta_all_coef'] = meta_all.coef_.tolist()

        pickle.dump(sc_full, open(deploy_dir / "scaler.pkl", 'wb'))
        pickle.dump(deployment_meta, open(deploy_dir / "deployment_meta.pkl", 'wb'))
        print(f"    [saved] Stacking meta-learners, scaler, deployment_meta")

        with open(OUTPUT_DIR / "ensemble_intermediate.json", 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 总结 (完整模式)
    if not quick_test and len(traits_run) > 1:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")

        eval_models = list(all_results[traits_run[0]].keys())
        summ = {m: {'R2': [], 'Corr': []} for m in eval_models}
        for t in traits_run:
            for m in eval_models:
                if m in all_results[t]:
                    summ[m]['R2'].append(all_results[t][m]['R2'])
                    summ[m]['Corr'].append(all_results[t][m].get('Correlation', 0))

        print(f"\n  {'Model':<24s} {'Type':>12s} {'Mean R2':>10s} {'Mean Corr':>10s} {'Best':>10s} {'Worst':>10s}")
        print(f"  {'-'*80}")
        for m in eval_models:
            rs = summ[m]['R2']
            if rs:
                mtype = all_results[traits_run[0]][m].get('Type', 'DL')
                print(f"  {m:<24s} {mtype:>12s} {np.mean(rs):10.4f} {np.mean(summ[m]['Corr']):10.4f} "
                      f"{np.max(rs):10.4f} {np.min(rs):10.4f}")

        ranked = sorted([(m, np.mean(summ[m]['R2'])) for m in eval_models],
                        key=lambda x: x[1], reverse=True)
        print("\n  Ranking:")
        for i, (m, r) in enumerate(ranked, 1):
            mtype = all_results[traits_run[0]][m].get('Type', 'DL')
            print(f"    {i:2d}. [{mtype:>11s}] {m}: {r:.4f}")

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(OUTPUT_DIR / f"ensemble_final_{ts}.json", 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # 画图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 13))
        colors = plt.cm.tab20(np.linspace(0, 1, len(eval_models)))
        for i, m in enumerate(eval_models):
            rs = [all_results[t][m]['R2'] for t in traits_run if m in all_results[t]]
            ax1.bar(np.arange(len(traits_run)) + (i-len(eval_models)/2+0.5)*0.08,
                    rs, 0.08, label=m, color=colors[i], alpha=0.85)
        ax1.set_ylabel('R2')
        ax1.set_title('Rice Genomic Prediction -- Ensemble System (5-fold CV)')
        ax1.set_xticks(np.arange(len(traits_run)))
        ax1.set_xticklabels([t[:15] for t in traits_run], rotation=45, ha='right')
        ax1.legend(ncol=3, fontsize=6)
        ax1.axhline(0, c='k', lw=0.5)
        ax1.grid(axis='y', alpha=0.3)

        ns = [m for m,_ in ranked]
        bc = [colors[eval_models.index(m)] for m in ns]
        bars = ax2.barh([m[:30] for m in ns], [np.mean(summ[m]['R2']) for m in ns], color=bc, alpha=0.85)
        ax2.set_xlabel('Mean R2')
        ax2.set_title('Overall Ranking')
        for b, v in zip(bars, [np.mean(summ[m]['R2']) for m in ns]):
            ax2.text(b.get_width()+0.005, b.get_y()+b.get_height()/2,
                     f'{v:.4f}', va='center', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"ensemble_comparison_{ts}.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nResults saved to: {OUTPUT_DIR}")
        print(f"Total time: {(time.time()-total_t0)/60:.1f} min")

    print("\nDone!")


if __name__ == '__main__':
    main()
