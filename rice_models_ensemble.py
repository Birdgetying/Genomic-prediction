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

import json, time
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr

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
# Config
# ============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "results" / "rice_data"
OUTPUT_DIR = _SCRIPT_DIR / "results" / "rice_ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS = 5
GWAS_TOP_K = 3000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def train_torch_model(model, X_train, y_train, X_val, y_val,
                      epochs=300, batch_size=128, lr=1e-3, weight_decay=1e-4,
                      patience=30):
    model = model.to(DEVICE)
    Xt = torch.FloatTensor(X_train).to(DEVICE)
    yt = torch.FloatTensor(y_train).to(DEVICE)
    Xv = torch.FloatTensor(X_val).to(DEVICE)
    yv = torch.FloatTensor(y_val).to(DEVICE)

    dl = DataLoader(TensorDataset(Xt, yt),
                    batch_size=min(batch_size, len(X_train)), shuffle=True)

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


# ============================================================================
# 模型工厂
# ============================================================================

def create_model(name, n_snps):
    """按名称创建单个模型"""
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
    if name == 'FusionNet':
        return FusionNet(n_snps=n_snps, hidden_dim=48, dropout=0.35)
    raise ValueError(f"Unknown model: {name}")


# ============================================================================
# 集成方案 B: Stacking (RidgeCV元学习器)
# ============================================================================

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

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

    base_names = ['FGN', 'EFM', 'MICNN', 'FGN v2', 'EFM v2', 'MICNN v2']
    ensemble_names = ['FusionNet']
    all_names = base_names + ensemble_names

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

        k = min(GWAS_TOP_K, X_all.shape[1] - 50)
        gidx = gwas_select(X_all, y, k)
        X_sel = X_all[:, gidx]
        n_snps = X_sel.shape[1]
        print(f"  {len(y)} samples, {X_all.shape[1]} markers -> {n_snps} GWAS-selected")

        models = {m: create_model(m, n_snps) for m in all_names}

        kf = KFold(n_splits=folds_run, shuffle=True, random_state=RANDOM_SEED)
        results = {m: {'preds': [], 'targets': [], 'params': 0, 'time': 0.0}
                   for m in all_names}
        oof = {m: np.zeros(len(y)) for m in base_names}

        for fi, (tr, te) in enumerate(kf.split(X_sel)):
            print(f"\n  --- Fold {fi+1}/{folds_run} ---")
            Xtr, Xte = X_sel[tr], X_sel[te]
            ytr, yte = y[tr], y[te]

            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
            Xte_s = sc.transform(Xte).astype(np.float32)

            fold_preds = {}

            for mi, mname in enumerate(all_names):
                if fi == 0:
                    results[mname]['params'] = sum(p.numel() for p in models[mname].parameters())

                t0 = time.time()
                bs = 64 if mname == 'FusionNet' else 128
                model = train_torch_model(
                    models[mname], Xtr_s, ytr, Xte_s, yte,
                    epochs=300, batch_size=bs, lr=2e-3, weight_decay=1e-3, patience=30)
                preds = predict_torch_model(model, Xte_s)
                elapsed = time.time() - t0

                results[mname]['preds'].extend(preds.tolist())
                results[mname]['targets'].extend(yte.tolist())
                results[mname]['time'] += elapsed
                fold_r2 = r2_score(yte, preds)
                print(f"    {mname:<16s} R2={fold_r2:+.4f}  ({elapsed:.1f}s)")

                if mname in base_names:
                    oof[mname][te] = preds
                    fold_preds[mname] = preds

                models[mname] = create_model(mname, n_snps)

            # 加权平均集成
            w = np.array([max(0.001, r2_score(yte, fold_preds[m])) for m in base_names])
            w = w / w.sum()
            wavg = np.zeros(len(yte))
            for mi, mname in enumerate(base_names):
                wavg += w[mi] * fold_preds[mname]
            print(f"    {'WeightedAvg':<16s} R2={r2_score(yte, wavg):+.4f}")

        # 性状汇总
        print(f"\n  {'-'*65}")
        print(f"  {trait} Final Results:")
        print(f"  {'Model':<16s} {'R2':>8s} {'Corr':>8s} {'RMSE':>8s} {'Params':>8s} {'Time':>7s}")
        print(f"  {'-'*65}")

        trait_res = {}
        for mname in all_names:
            p = np.array(results[mname]['preds'])
            t = np.array(results[mname]['targets'])
            r2_v = float(r2_score(t, p))
            corr_v = float(pearsonr(t, p)[0])
            rmse_v = float(np.sqrt(np.mean((p-t)**2)))
            trait_res[mname] = {
                'R2': r2_v, 'Correlation': corr_v, 'RMSE': rmse_v,
                'Params': results[mname]['params'],
                'Time': results[mname]['time']/folds_run}
            print(f"  {mname:<16s} {r2_v:8.4f} {corr_v:8.4f} {rmse_v:8.4f} "
                  f"{results[mname]['params']:8,} {results[mname]['time']/folds_run:7.1f}s")

        # Stacking
        if folds_run >= 3:
            sr = stacking_evaluate(oof, y, n_folds=min(5, folds_run))
            trait_res['Stacking'] = {
                'R2': sr['R2'], 'Correlation': sr['Correlation'],
                'RMSE': 0.0, 'Params': 0, 'Time': 0.0,
                'Meta_weights': sr['Meta_weights'], 'Base_models': sr['Base_models']}
            print(f"  {'Stacking':<16s} {sr['R2']:8.4f} {sr['Correlation']:8.4f}")
            weights_str = dict(zip(sr['Base_models'], [f'{w:.3f}' for w in sr['Meta_weights']]))
            print(f"    weights: {weights_str}")

        all_results[trait] = trait_res

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

        print(f"\n  {'Model':<16s} {'Mean R2':>10s} {'Mean Corr':>10s} {'Best':>10s} {'Worst':>10s}")
        print(f"  {'-'*60}")
        for m in eval_models:
            rs = summ[m]['R2']
            if rs:
                print(f"  {m:<16s} {np.mean(rs):10.4f} {np.mean(summ[m]['Corr']):10.4f} "
                      f"{np.max(rs):10.4f} {np.min(rs):10.4f}")

        ranked = sorted([(m, np.mean(summ[m]['R2'])) for m in eval_models],
                        key=lambda x: x[1], reverse=True)
        print("\n  Ranking:")
        for i, (m, r) in enumerate(ranked, 1):
            print(f"    {i}. {m}: {r:.4f}")

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(OUTPUT_DIR / f"ensemble_final_{ts}.json", 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # 画图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        colors = ['#2196F3','#4CAF50','#FF9800','#03A9F4','#8BC34A','#FFB74D','#E91E63','#9C27B0','#00BCD4']
        for i, m in enumerate(eval_models):
            rs = [all_results[t][m]['R2'] for t in traits_run if m in all_results[t]]
            ax1.bar(np.arange(len(traits_run)) + (i-len(eval_models)/2+0.5)*0.1,
                    rs, 0.1, label=m, color=colors[i%len(colors)], alpha=0.85)
        ax1.set_ylabel('R2')
        ax1.set_title('Rice Genomic Prediction -- Ensemble System (5-fold CV)')
        ax1.set_xticks(np.arange(len(traits_run)))
        ax1.set_xticklabels([t[:15] for t in traits_run], rotation=45, ha='right')
        ax1.legend(ncol=3, fontsize=7)
        ax1.axhline(0, c='k', lw=0.5)
        ax1.grid(axis='y', alpha=0.3)

        ns = [m for m,_ in ranked]
        bc = [colors[eval_models.index(m)%len(colors)] for m in ns]
        bars = ax2.barh([m[:25] for m in ns], [np.mean(summ[m]['R2']) for m in ns], color=bc, alpha=0.85)
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
