"""
Wheat Genomic Prediction — FGN + EFM + MICNN Ensemble System
=============================================================
"频域-统计-空间" 正交集成体系, 适配小麦 VCF 数据 (SNP+INDEL+SV)

数据: 819 小麦核心种质 (WATDE), SNP/INDEL/SV 三种变异类型
路径: /storage/public/home/2024110093/data/Variation/CSIAAS/

模型:
  Traditional: RRBLUP, GBLUP, XGBoost, ElasticNet, GWAS_RRBLUP, Trad Ensemble
  DL:         FGN / FGN v2 (FFT+Wavelet), EFM / EFM v2 (FM+MLP),
              MICNN / MICNN v2 (Inception+Dilated+SPP)
  DL集成:     FusionNet (三分支联合训练), Stacking (DL/All)

用法:
  python wheat_models_ensemble.py           # 快速测试 (1个性状 x 2折)
  python wheat_models_ensemble.py --full    # 完整实验 (所有性状 x 5折)
"""

import json, time, os, sys
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
# Config
# ============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent

# HPC 数据路径 (与 train_sv_pro_large.py 一致)
DATA_BASE_PATH = "/storage/public/home/2024110093/data/Variation/CSIAAS/"
SNP_VCF_PATH   = DATA_BASE_PATH + "Core819Samples_snp.filter.final.id_gt.813m.vcf.gz"
INDEL_VCF_PATH = DATA_BASE_PATH + "Core819Samples_indel.filter.final.id_gt.813m.vcf.gz"
SV_VCF_PATH    = DATA_BASE_PATH + "SV.new.vcf.gz"
PHENOTYPE_PATH = DATA_BASE_PATH + "Phe.txt"
VCF_ID_PATH    = DATA_BASE_PATH + "VCFID.txt"

OUTPUT_DIR = _SCRIPT_DIR / "results" / "wheat_ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS = 5
GWAS_TOP_K = 3000
MAX_VARIANTS_PER_TYPE = 15000  # 每种变异类型最多加载标记数

# GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

QUICK_TEST = True  # 命令行 --full 覆盖

TYPE_TRAD = 'Traditional'
TYPE_DL = 'DL'
TYPE_ENS = 'Ensemble'

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ============================================================================
# VCF 数据加载
# ============================================================================

def load_phenotypes(phenotype_path):
    """加载表型文件, 返回 {trait_name: y_array} 和样本数"""
    if not os.path.exists(phenotype_path):
        raise FileNotFoundError(f"表型文件不存在: {phenotype_path}")

    df = pd.read_csv(phenotype_path, sep='\t', header=0)
    n_samples = len(df)
    print(f"  表型文件: {n_samples} 样本, {df.shape[1]} 列")

    traits = {}
    for col in df.columns:
        if col.lower() in ('sample', 'id', 'name', 'accession', 'line'):
            continue
        try:
            vals = pd.to_numeric(df[col], errors='coerce').values.astype(np.float32)
            mask = ~np.isnan(vals)
            if mask.sum() > 0.5 * len(vals):  # 至少50%有效值
                traits[col] = (vals, mask)
                print(f"    -> 性状 '{col}': {mask.sum()} 有效, "
                      f"范围 [{vals[mask].min():.3f}, {vals[mask].max():.3f}]")
        except (ValueError, TypeError):
            pass

    if not traits:
        # 尝试无标题模式: 第一列ID, 第二列表型
        df = pd.read_csv(phenotype_path, sep='\t', header=None)
        vals = pd.to_numeric(df.iloc[:, 1], errors='coerce').values.astype(np.float32)
        mask = ~np.isnan(vals)
        traits = {'Phenotype': (vals, mask)}
        print(f"    -> 单性状模式: {mask.sum()} 有效")

    print(f"  共检测到 {len(traits)} 个性状")
    return traits


def load_vcf_genotypes(vcf_path, max_variants, min_samples):
    """使用 cyvcf2 加载 VCF 基因型, 返回 (samples x variants) 矩阵"""
    import cyvcf2

    vcf = cyvcf2.VCF(vcf_path)
    n_vcf = len(vcf.samples)
    actual_n = min(min_samples, n_vcf)

    genotypes = []
    variant_count = 0

    for variant in vcf:
        if variant_count >= max_variants:
            break
        gt = variant.gt_types
        if len(gt) >= actual_n:
            gt_numeric = []
            for g in gt[:actual_n]:
                if g == 0:      gt_numeric.append(0)
                elif g == 1:    gt_numeric.append(1)
                elif g == 2:    gt_numeric.append(2)
                else:           gt_numeric.append(0)
            genotypes.append(gt_numeric)
            variant_count += 1

    X = np.array(genotypes, dtype=np.float32).T
    print(f"    提取 {variant_count} 个变异位点, 矩阵 {X.shape}")
    return X


def load_all_wheat_data():
    """加载全部小麦数据: SNP+INDEL+SV 合并, 返回 {trait: X} 和 {trait: y}"""
    print(f"\n{'='*70}")
    print("Wheat VCF Data Loading (SNP + INDEL + SV)")
    print(f"{'='*70}")

    # 1. 加载表型
    print("\n[1/4] Loading phenotypes ...")
    traits = load_phenotypes(PHENOTYPE_PATH)

    # 确定最小样本数
    y_sample_counts = [mask.sum() for _, (_, mask) in traits.items()]
    n_phe = max(y_sample_counts)
    print(f"  表型最大样本数: {n_phe}")

    # 2. 加载 VCF ID
    print("\n[2/4] Loading VCF IDs ...")
    if os.path.exists(VCF_ID_PATH):
        vcf_ids = pd.read_csv(VCF_ID_PATH, header=None, sep=r'\s+')
        n_vcf_id = len(vcf_ids)
        print(f"  VCF ID 样本数: {n_vcf_id}")
    else:
        n_vcf_id = n_phe
        print(f"  VCF ID 文件不存在, 使用表型样本数: {n_vcf_id}")

    # 3. 加载各类型 VCF
    print("\n[3/4] Loading VCF genotypes ...")
    min_samples = min(n_phe, n_vcf_id)

    vcf_configs = [
        ("SNP",   SNP_VCF_PATH),
        ("INDEL", INDEL_VCF_PATH),
        ("SV",    SV_VCF_PATH),
    ]

    X_parts = []
    for vtype, vpath in vcf_configs:
        print(f"  [{vtype}] {os.path.basename(vpath)}")
        if not os.path.exists(vpath):
            print(f"    WARNING: 文件不存在, 跳过")
            continue
        try:
            X_v = load_vcf_genotypes(vpath, MAX_VARIANTS_PER_TYPE, min_samples)
            X_parts.append(X_v)
        except Exception as e:
            print(f"    ERROR: {e}, 跳过")

    if not X_parts:
        raise RuntimeError("未能加载任何 VCF 数据!")

    X_all = np.hstack(X_parts)
    print(f"\n  合并基因型矩阵: {X_all.shape} (samples x total_markers)")

    # 4. 为每个性状准备数据
    print("\n[4/4] Preparing trait-specific data ...")
    trait_data = {}
    for tname, (y_full, mask) in traits.items():
        n_common = min(len(y_full), X_all.shape[0])
        y_t = y_full[:n_common]
        X_t = X_all[:n_common]

        # 移除 NaN
        mask_t = mask[:n_common]
        X_t, y_t = X_t[mask_t], y_t[mask_t]

        # 过滤低方差标记
        var_thresh = 0.005
        vars_per_marker = np.var(X_t, axis=0)
        keep = vars_per_marker >= var_thresh
        if keep.sum() < X_t.shape[1]:
            X_t = X_t[:, keep]
            print(f"  {tname}: {X_t.shape} (过滤低方差后)  y∈[{y_t.min():.3f}, {y_t.max():.3f}]")
        else:
            print(f"  {tname}: {X_t.shape}  y∈[{y_t.min():.3f}, {y_t.max():.3f}]")

        trait_data[tname] = (X_t.astype(np.float32), y_t.astype(np.float32))

    return trait_data


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
        preds = model(Xt).squeeze().cpu().numpy()
    model.cpu()
    return preds


# ============================================================================
# 传统模型 (sklearn — 快速基线)
# ============================================================================

class RRBLUP:
    """Ridge Regression BLUP — alpha 范围 0.001~1000"""
    def __init__(self):
        self.model = RidgeCV(alphas=np.logspace(-3, 3, 30))

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class GBLUP:
    """Genomic BLUP — 基于基因组关系矩阵 (GRM) 的 Ridge 回归"""
    def __init__(self):
        self.model = RidgeCV(alphas=np.logspace(-3, 3, 20))

    def fit(self, G_train, y_train):
        self.model.fit(G_train, y_train)
        return self

    def predict(self, G_test_train):
        return self.model.predict(G_test_train)


class XGBoostModel:
    """XGBoost 回归 — 固定超参, 快速训练"""
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
    """ElasticNet CV — 自动搜索 l1_ratio 和 alpha"""
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
    """GWAS 加权 RRBLUP — 软权重 (Pearson 相关), 保留所有标记"""
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


# ============================================================================
# 传统模型 Stacking (Ridge 元学习器, 嵌套 CV 防泄露)
# ============================================================================

def evaluate_traditional_stacking(oof_trad, y):
    """复用主循环 OOF 预测, 仅做元学习器嵌套 CV 评估 (无重复训练)"""
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


def compute_grm(X):
    """计算基因组关系矩阵 G = XX^T / p (VanRaden 2008)"""
    X_s = X - X.mean(axis=0)
    return X_s @ X_s.T / X_s.shape[1]


# ============================================================================
# 原版 DL 模型 (基线对照)
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
    """可学习Haar小波分解"""
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
    """EFM v2: +FM Embedding Dropout"""
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
    """带空洞卷积的Inception"""
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
    """MICNN v2: +DilatedConv +SPP"""
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
# FusionNet -- 三分支特征级硬融合
# ============================================================================

class FusionNet(nn.Module):
    """三分支特征级融合: FGN(频域) + EFM(统计) + MICNN(局部)"""
    def __init__(self, n_snps, hidden_dim=48, dropout=0.35):
        super().__init__()
        # FGN branch
        self.n_freq = n_snps // 2 + 1
        self.spec_r = nn.Parameter(torch.randn(1, 16, self.n_freq) * 0.02)
        self.spec_i = nn.Parameter(torch.randn(1, 16, self.n_freq) * 0.02)
        self.fgn_conv = nn.Sequential(
            nn.Conv1d(16, hidden_dim, 7, padding=3), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(dropout*0.3))
        self.fgn_pool = nn.AdaptiveAvgPool1d(1)

        # EFM branch
        self.efm_linear = nn.Linear(n_snps, 1)
        self.V_f = nn.Parameter(torch.randn(n_snps, 4) * 0.01)
        self.efm_deep = nn.Sequential(
            nn.Linear(n_snps, hidden_dim*2), nn.BatchNorm1d(hidden_dim*2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))

        # MICNN branch
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

        # EFM
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
# Stacking 集成
# ============================================================================

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

def stacking_evaluate(oof_preds_dict, targets, n_folds=5):
    """Stacking: 各模型OOF预测作为元特征, RidgeCV融合"""
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
    global QUICK_TEST

    if '--full' in sys.argv:
        QUICK_TEST = False
        print("[CMD] --full: 完整实验模式")

    # 如果本地运行且数据在HPC不可用, 尝试本地路径
    global DATA_BASE_PATH, SNP_VCF_PATH, INDEL_VCF_PATH, SV_VCF_PATH
    global PHENOTYPE_PATH, VCF_ID_PATH

    if not os.path.exists(DATA_BASE_PATH):
        local_base = str(_SCRIPT_DIR / "Variation" / "CSIAAS") + "/"
        if os.path.exists(local_base):
            print(f"[WARNING] HPC路径不可用, 回退到本地: {local_base}")
            DATA_BASE_PATH = local_base
            SNP_VCF_PATH   = DATA_BASE_PATH + "Core819Samples_snp.filter.final.id_gt.813m.vcf.gz"
            INDEL_VCF_PATH = DATA_BASE_PATH + "Core819Samples_indel.filter.final.id_gt.813m.vcf.gz"
            SV_VCF_PATH    = DATA_BASE_PATH + "SV.new.vcf.gz"
            PHENOTYPE_PATH = DATA_BASE_PATH + "Phe.txt"
            VCF_ID_PATH    = DATA_BASE_PATH + "VCFID.txt"

    print(f"Device: {DEVICE}  |  Quick test: {QUICK_TEST}")
    print(f"Data base: {DATA_BASE_PATH}")
    print(f"{'='*80}")
    print("Wheat Genomic Prediction — FGN+EFM+MICNN Ensemble System")
    print(f"{'='*80}")

    if DEVICE.type == 'cuda':
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 加载数据
    trait_data = load_all_wheat_data()
    trait_names = sorted(trait_data.keys())
    print(f"\n实验性状: {trait_names}")

    trad_names = ['RRBLUP', 'GBLUP', 'XGBoost', 'ElasticNet', 'GWAS_RRBLUP']
    dl_base_names = ['FGN', 'EFM', 'MICNN', 'FGN v2', 'EFM v2', 'MICNN v2']
    dl_ensemble_names = ['FusionNet']
    dl_names = dl_base_names + dl_ensemble_names
    all_names = trad_names + dl_names

    traits_run = trait_names[:1] if QUICK_TEST else trait_names
    folds_run = min(2, N_FOLDS) if QUICK_TEST else N_FOLDS

    if QUICK_TEST:
        print(f"  [QUICK TEST] {len(traits_run)} trait x {folds_run} folds")

    all_results = {}
    total_t0 = time.time()

    for t_idx, trait in enumerate(traits_run):
        print(f"\n{'='*80}")
        print(f"  TRAIT [{t_idx+1}/{len(traits_run)}]: {trait}")
        print(f"{'='*80}")

        X_all, y = trait_data[trait]
        y = y.astype(np.float32)

        # GWAS 筛选
        k = min(GWAS_TOP_K, max(50, X_all.shape[1] - 50))
        if X_all.shape[1] > k:
            gidx = gwas_select(X_all, y, k)
            X_sel = X_all[:, gidx]
        else:
            X_sel = X_all
        n_snps = X_sel.shape[1]
        print(f"  {len(y)} samples, {X_all.shape[1]} markers -> {n_snps} GWAS-selected")

        # 预计算 GRM (GBLUP 和传统 Stacking 需要)
        G_mat = compute_grm(X_sel)
        print(f"  GRM: {G_mat.shape}")

        # DL 模型初始化 (每折重建)
        dl_models = {m: create_model(m, n_snps) for m in dl_names}

        kf = KFold(n_splits=folds_run, shuffle=True, random_state=RANDOM_SEED)
        results = {m: {'preds': [], 'targets': [], 'params': 0, 'time': 0.0}
                   for m in all_names}

        # OOF 预测 (用于 Stacking)
        oof_trad = {m: np.zeros(len(y)) for m in trad_names}
        oof_dl = {m: np.zeros(len(y)) for m in dl_base_names}

        for fi, (tr, te) in enumerate(kf.split(X_sel)):
            print(f"\n  --- Fold {fi+1}/{folds_run} ---")
            Xtr, Xte = X_sel[tr], X_sel[te]
            ytr, yte = y[tr], y[te]

            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
            Xte_s = sc.transform(Xte).astype(np.float32)

            # ── 传统模型 ──
            trad_configs = [
                ('RRBLUP', lambda: RRBLUP(), lambda m, Xtr, ytr: m.fit(Xtr, ytr),
                 lambda m, Xte: m.predict(Xte), n_snps + 1),
                ('GBLUP', lambda: GBLUP(),
                 lambda m, Xtr, ytr: m.fit(G_mat[tr][:, tr], ytr),
                 lambda m, Xte: m.predict(G_mat[te][:, tr]), len(tr) + 1),
                ('XGBoost', lambda: XGBoostModel(n_estimators=300),
                 lambda m, Xtr, ytr: m.fit(Xtr, ytr),
                 lambda m, Xte: m.predict(Xte), 300 * 6 * 2),
                ('ElasticNet', lambda: ElasticNetModel(),
                 lambda m, Xtr, ytr: m.fit(Xtr, ytr),
                 lambda m, Xte: m.predict(Xte), n_snps + 1),
                ('GWAS_RRBLUP', lambda: GWASWeightedRRBLUP(),
                 lambda m, Xtr, ytr: m.fit(Xtr, ytr),
                 lambda m, Xte: m.predict(Xte), n_snps * 2 + 1),
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

            # ── DL 模型 (PyTorch) ──
            for mi, mname in enumerate(dl_names):
                if fi == 0:
                    results[mname]['params'] = sum(p.numel() for p in dl_models[mname].parameters())

                t0 = time.time()
                bs = 64 if mname == 'FusionNet' else 128
                lr = 1e-3 if mname == 'FusionNet' else 2e-3
                model = train_torch_model(
                    dl_models[mname], Xtr_s, ytr, Xte_s, yte,
                    epochs=300, batch_size=bs, lr=lr, weight_decay=1e-3, patience=30)
                preds = predict_torch_model(model, Xte_s)
                elapsed = time.time() - t0

                results[mname]['preds'].extend(preds.tolist())
                results[mname]['targets'].extend(yte.tolist())
                results[mname]['time'] += elapsed
                fold_r2 = r2_score(yte, preds)
                print(f"    {mname:<16s} R2={fold_r2:+.4f}  ({elapsed:.1f}s)")

                if mname in dl_base_names:
                    oof_dl[mname][te] = preds

                dl_models[mname] = create_model(mname, n_snps)

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
            tag = " [Trad]" if mname in trad_names else " [DL]"
            trait_res[mname] = {
                'R2': r2_v, 'Correlation': corr_v, 'RMSE': rmse_v,
                'Type': TYPE_TRAD if mname in trad_names else TYPE_DL,
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

            # 传统 Stacking — 复用主循环 OOF, 仅元学习器评估
            tsr = evaluate_traditional_stacking(oof_trad, y)
            trait_res['Trad Ensemble'] = {
                'R2': tsr['R2'], 'Correlation': tsr['Correlation'],
                'RMSE': 0.0, 'Type': TYPE_ENS,
                'Meta_weights': tsr['Meta_weights'], 'Base_models': tsr['Base_models']}
            print(f"  {'Trad Ensemble':<24s} {tsr['R2']:8.4f} {tsr['Correlation']:8.4f}")

        all_results[trait] = trait_res

        with open(OUTPUT_DIR / "ensemble_intermediate.json", 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ══════════════════════════════════════════════════════════════════════
    # 总结 & 可视化
    # ══════════════════════════════════════════════════════════════════════
    if not QUICK_TEST and len(traits_run) > 1:
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

        print(f"\n  {'Model':<20s} {'Type':<12s} {'Mean R2':>10s} {'Mean Corr':>10s} {'Best':>10s} {'Worst':>10s}")
        print(f"  {'-'*72}")
        for m in eval_models:
            rs = summ[m]['R2']
            if rs:
                mtype = all_results[traits_run[0]][m].get('Type', TYPE_DL)
                print(f"  {m:<20s} {mtype:<12s} {np.mean(rs):10.4f} {np.mean(summ[m]['Corr']):10.4f} "
                      f"{np.max(rs):10.4f} {np.min(rs):10.4f}")

        ranked = sorted([(m, np.mean(summ[m]['R2'])) for m in eval_models],
                        key=lambda x: x[1], reverse=True)
        print("\n  Ranking:")
        for i, (m, r) in enumerate(ranked, 1):
            mtype = all_results[traits_run[0]][m].get('Type', '')
            marker = " <-- BEST" if i == 1 else ""
            print(f"    {i:2d}. {m:<22s} [{mtype}]  {r:.4f}{marker}")

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(OUTPUT_DIR / f"ensemble_final_{ts}.json", 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # ── 可视化 ──
        # 颜色方案: 传统=蓝色系, DL=暖色系, Ensemble=红/粉
        model_colors = {
            'RRBLUP': '#90CAF9', 'GBLUP': '#64B5F6', 'XGBoost': '#42A5F5',
            'ElasticNet': '#2196F3', 'GWAS_RRBLUP': '#1E88E5',
            'FGN': '#FFB74D', 'EFM': '#FFD54F', 'MICNN': '#FF8A65',
            'FGN v2': '#F57C00', 'EFM v2': '#FBC02D', 'MICNN v2': '#E64A19',
            'FusionNet': '#E91E63',
            'Stacking (DL)': '#C62828', 'Stacking (All)': '#B71C1C',
            'Trad Ensemble': '#1565C0',
        }

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14))

        # 子图1: 各性状 R² 柱状图
        x = np.arange(len(traits_run))
        n_models = len(eval_models)
        w = 0.8 / n_models
        for i, m in enumerate(eval_models):
            rs = [all_results[t][m]['R2'] for t in traits_run if m in all_results[t]]
            offset = (i - n_models/2 + 0.5) * w
            ax1.bar(x + offset, rs, w, label=m, color=model_colors.get(m, '#999'),
                    alpha=0.88, edgecolor='white', linewidth=0.3)
        ax1.set_ylabel('R²', fontsize=12)
        ax1.set_title('Wheat Genomic Prediction — All Models Comparison (5-fold CV R²)',
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([t[:22] for t in traits_run], rotation=45, ha='right', fontsize=8)
        ax1.legend(ncol=5, fontsize=6.5, loc='lower left')
        ax1.axhline(0, c='k', lw=0.5)
        ax1.grid(axis='y', alpha=0.25)

        # 子图2: 总体排名
        ns = [m for m, _ in ranked]
        vals = [np.mean(summ[m]['R2']) for m in ns]
        bar_colors = [model_colors.get(m, '#999') for m in ns]
        bars = ax2.barh(ns, vals, color=bar_colors, alpha=0.88, edgecolor='white', linewidth=0.3)
        ax2.set_xlabel('Mean R² across traits', fontsize=12)
        ax2.set_title('Overall Model Ranking', fontsize=14, fontweight='bold')
        for b, v, m in zip(bars, vals, ns):
            mtype = all_results[traits_run[0]][m].get('Type', TYPE_DL)
            label = f'{v:.4f} [{mtype[0]}]'
            ax2.text(b.get_width() + 0.005, b.get_y() + b.get_height()/2,
                     label, va='center', fontsize=7.5, fontweight='bold')
        ax2.grid(axis='x', alpha=0.25)
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"ensemble_comparison_{ts}.png", dpi=180, bbox_inches='tight')
        plt.close()

        print(f"\nResults saved to: {OUTPUT_DIR}")
        print(f"Total time: {(time.time()-total_t0)/60:.1f} min")

    print("\nDone!")


if __name__ == '__main__':
    main()
