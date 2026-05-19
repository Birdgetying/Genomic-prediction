"""
Maize Genomic Prediction — Iranian/Mexican Maize Ensemble
===========================================================
伊朗+墨西哥玉米数据: GWAS筛选 → FGN/MICNN/FusionNet/Stacking 集成

数据: data2/ — Iranian_Samples.csv (52K markers × 2.5K samples)
表型: phenotype_iranian.csv — Heat/Drought × DTM/DTH (4 traits)

用法:
  python maize_models_ensemble.py              # 快速测试 (1个性状 x 2折)
  python maize_models_ensemble.py --full       # 完整实验
  python maize_models_ensemble.py --mexican    # 使用墨西哥数据
"""

import json, time, os, sys, pickle, shutil
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
from genomic_nn_models import (FGNEncoder, PreFGN, pretrain_prefgn, pretrain_prefgn_v2)
from deep_kernel_gp import GenomicEncoder, DeepKernelGP, train_dkgp

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
DATA_DIR = _SCRIPT_DIR / "data2"
OUTPUT_DIR = _SCRIPT_DIR / "results" / "maize_ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS = 5
GWAS_TOP_K = 5000
MAF_THRESHOLD = 0.05
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TYPE_TRAD = 'Traditional'
TYPE_DL = 'DL'
TYPE_HYBRID = 'Hybrid'
TYPE_ENSEMBLE = 'Ensemble'

DATASET = 'iranian'  # 'iranian' | 'mexican'

# ============================================================================
# DL Model Definitions (same architecture as wheat/rice)
# ============================================================================

class FourierGenomicNet(nn.Module):
    """FGN: FFT频谱 + 深度MLP"""
    def __init__(self, n_snps, hidden=64, dropout=0.35):
        super().__init__()
        self.n_freq = n_snps // 2 + 1
        self.freq_linear = nn.Sequential(nn.Linear(self.n_freq, hidden*2), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(hidden*2, hidden))
        self.mlp = nn.Sequential(nn.Linear(n_snps, hidden*2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden*2, hidden))
        self.head = nn.Linear(hidden*2, 1)

    def forward(self, x):
        f = torch.fft.rfft(x, dim=1).abs()
        f_out = self.freq_linear(f)
        mlp_out = self.mlp(x)
        return self.head(torch.cat([f_out, mlp_out], dim=1)).squeeze(-1)



class MultiScaleInceptionCNN(nn.Module):
    """MICNN: 多尺度Inception卷积 + 膨胀卷积"""
    def __init__(self, n_snps, hidden=48, dropout=0.35):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden, 3, padding=1)
        self.inception = nn.ModuleList([
            nn.Conv1d(hidden, hidden//3, k, padding=k//2) for k in [3, 7, 15]
        ])
        self.dilated = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
            nn.GELU(), nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.gelu(self.conv1(x))
        branches = [conv(x) for conv in self.inception]
        x = torch.cat(branches, dim=1)
        x = self.dilated(x)
        return self.head(x).squeeze(-1)


class FGNv2(nn.Module):
    """FGN v2: FFT + 可学习 Haar 小波"""
    def __init__(self, n_snps, hidden=64, dropout=0.35):
        super().__init__()
        self.n_freq = n_snps // 2 + 1
        self.freq_mlp = nn.Sequential(nn.Linear(self.n_freq, hidden*2), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(hidden*2, hidden))
        self.wavelet = nn.Sequential(nn.Linear(n_snps, hidden*2), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden*2, hidden))
        self.head = nn.Linear(hidden*2, 1)

    def forward(self, x):
        f = torch.fft.rfft(x, dim=1).abs()
        return self.head(torch.cat([self.freq_mlp(f), self.wavelet(x)], dim=1)).squeeze(-1)



class MICNNv2(nn.Module):
    """MICNN v2: +Dilated Conv + Spatial Pyramid Pooling"""
    def __init__(self, n_snps, hidden=40, dropout=0.35, spp_bins=(1, 2, 4)):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden, 3, padding=1)
        self.dilated = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4))
        self.spp_bins = spp_bins
        spp_out = hidden * sum(spp_bins)
        self.head = nn.Sequential(nn.Linear(spp_out, hidden*2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden*2, 1))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.gelu(self.conv1(x))
        x = self.dilated(x)
        spp_feats = []
        for b in self.spp_bins:
            spp_feats.append(F.adaptive_avg_pool1d(x, b).flatten(1))
        return self.head(torch.cat(spp_feats, dim=1)).squeeze(-1)


class FGNv3(nn.Module):
    """FGN v3: 深度FGN编码器"""
    def __init__(self, n_snps, hidden=48, dropout=0.35, marker_types=None):
        super().__init__()
        self.encoder = FGNEncoder(n_snps, hidden, dropout, marker_types)

    def forward(self, x):
        return self.encoder(x)



class FusionNet(nn.Module):
    """三分支特征融合: FGN(频域) + EFM(统计) + MICNN(局部)"""
    def __init__(self, n_snps, hidden_dim=48, dropout=0.35):
        super().__init__()
        self.n_freq = n_snps // 2 + 1
        self.fgn_freq = nn.Sequential(nn.Linear(self.n_freq, hidden_dim), nn.GELU())
        self.fgn_linear = nn.Linear(n_snps, hidden_dim)
        self.efm_linear = nn.Linear(n_snps, 1)
        self.V_f = nn.Parameter(torch.randn(n_snps, 4) * 0.01)
        self.efm_mlp = nn.Sequential(
            nn.Linear(n_snps, hidden_dim*2), nn.BatchNorm1d(hidden_dim*2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim*2, hidden_dim))
        self.micnn = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.fuse = nn.Sequential(nn.Linear(hidden_dim*4, hidden_dim*2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim*2, 1))

    def forward(self, x):
        f_freq = self.fgn_freq(torch.fft.rfft(x, dim=1).abs())
        f_lin = self.fgn_linear(x)
        fm = 0.5 * ((x @ self.V_f)**2 - (x**2) @ (self.V_f**2)).sum(dim=1, keepdim=True)
        efm_deep = self.efm_mlp(x)
        micnn_feat = self.micnn(x.unsqueeze(1))
        return self.fuse(torch.cat([f_freq, f_lin, efm_deep, micnn_feat], dim=1)).squeeze(-1)


# ============================================================================
# Utility Functions
# ============================================================================

def gwas_select(X, y, k):
    """GWAS p-value 排序, 返回 top-k 标记索引"""
    y_c = y - y.mean()
    X_c = X - X.mean(axis=0)
    num = np.dot(y_c, X_c)
    denom = np.std(y_c) * len(y) * np.sqrt(np.sum(X_c ** 2, axis=0) + 1e-12)
    scores = np.abs(num / denom)
    return np.argsort(scores)[-k:]


def fit_traditional(model_name, X_train, y_train):
    """训练传统模型"""
    if model_name == 'RRBLUP':
        m = RidgeCV(alphas=np.logspace(-2, 5, 20))
    elif model_name == 'GBLUP':
        G = X_train @ X_train.T / X_train.shape[1]
        from sklearn.kernel_ridge import KernelRidge
        m = KernelRidge(alpha=1.0, kernel='precomputed')
        X_train = G
    elif model_name == 'XGBoost':
        m = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                             subsample=0.8, random_state=RANDOM_SEED, n_jobs=-1)
    elif model_name == 'ElasticNet':
        m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95], cv=3,
                         random_state=RANDOM_SEED, max_iter=5000)
    elif model_name == 'GWAS_RRBLUP':
        m = RidgeCV(alphas=np.logspace(-2, 5, 20))
    else:
        raise ValueError(f"Unknown traditional model: {model_name}")
    m.fit(X_train, y_train)
    return m


def create_model(name, n_snps, overrides=None):
    o = overrides or {}
    if name == 'FGN':
        return FourierGenomicNet(n_snps=n_snps, hidden=64, dropout=0.35)
    if name == 'MICNN':
        return MultiScaleInceptionCNN(n_snps=n_snps, hidden=48, dropout=0.35)
    if name == 'FGN v2':
        return FGNv2(n_snps=n_snps, hidden=64, dropout=0.35)
    if name == 'MICNN v2':
        return MICNNv2(n_snps=n_snps, hidden=40, dropout=0.35, spp_bins=(1, 2, 4))
    if name == 'FGN v3':
        return FGNv3(n_snps=n_snps, hidden=o.get('hidden', 48),
                     dropout=o.get('dropout', 0.35))
    if name == 'FusionNet':
        return FusionNet(n_snps=n_snps,
                         hidden_dim=o.get('hidden_dim', 48),
                         dropout=o.get('dropout', 0.35))
    if name == 'PreFGN':
        return PreFGN(n_snps=n_snps, hidden=o.get('hidden', 64),
                      dropout=o.get('dropout', 0.35))
    if name == 'DeepKernelGP':
        encoder = GenomicEncoder(n_snps=n_snps,
                                 latent_dim=o.get('latent_dim', 16),
                                 hidden_dim=o.get('hidden', 64),
                                 dropout=o.get('dropout', 0.2))
        return DeepKernelGP(encoder, latent_dim=o.get('latent_dim', 16))
    raise ValueError(f"Unknown model: {name}")


def train_dl_model(model, X_train, y_train, X_val=None, y_val=None,
                   epochs=300, lr=1e-3, batch_size=32, patience=30):
    """训练深度学习模型"""
    model = model.to(DEVICE)
    ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_loss, best_state, wait = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * len(bx)
        epoch_loss /= len(ds)

        scheduler.step(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model.cpu()


def train_dkgp_wrapper(model, X_train, y_train, epochs=200, lr=0.01):
    return train_dkgp(model, torch.FloatTensor(X_train), torch.FloatTensor(y_train),
                      epochs=epochs, lr=lr, verbose=False)


def tune_model_hyperparams(model_name, X_train, y_train, n_snps, n_trials=15):
    """AutoML tuning (simplified: random search over grid)"""
    best_score, best_overrides = -float('inf'), {}
    grids = {
        'FGN v3': {'hidden': [32, 48, 64, 96], 'dropout': [0.2, 0.35, 0.5]},
        'FusionNet': {'hidden_dim': [32, 48, 64], 'dropout': [0.2, 0.35, 0.5]},
        'DeepKernelGP': {'latent_dim': [8, 16, 32], 'hidden': [32, 48, 64], 'dropout': [0.1, 0.2, 0.35]},
        'PreFGN': {'hidden': [32, 48, 64], 'dropout': [0.2, 0.35, 0.5]},
    }
    grid = grids.get(model_name, {})
    if not grid:
        return {}

    np.random.seed(RANDOM_SEED)
    for _ in range(n_trials):
        overrides = {k: np.random.choice(v) for k, v in grid.items()}
        try:
            model = create_model(model_name, n_snps, overrides=overrides)
            model = train_dl_model(model, X_train, y_train, epochs=80, lr=5e-4, patience=15)
            with torch.no_grad():
                pred = model(torch.FloatTensor(X_train)).numpy()
            score = r2_score(y_train, pred)
            if score > best_score:
                best_score = score
                best_overrides = overrides
        except Exception:
            continue
    return best_overrides


def fit_resfgn_components(X, y, n_snps, cv=3):
    """ResFGN: RidgeCV拟合加性部分 + FusionNet拟合残差"""
    ridge = RidgeCV(alphas=np.logspace(-2, 5, 20))
    ridge.fit(X, y)
    y_pred_ridge = ridge.predict(X)
    residual = y - y_pred_ridge

    fusion = FusionNet(n_snps=n_snps, hidden_dim=48, dropout=0.35)
    fusion = train_dl_model(fusion, X, residual, epochs=200, lr=5e-4, patience=25)
    return ridge, fusion


# ============================================================================
# Data Loading
# ============================================================================

def load_iranian_data(max_markers=None):
    """加载伊朗玉米数据

    CSV结构: 前8行=元数据头, 前17列=标记属性, 第17列后=样本基因型(0/1/2/-)
    """
    print(f"\n{'='*70}")
    print("Iranian Maize Data Loading")
    print(f"{'='*70}")

    print("\n[1/3] Loading genotype matrix ...")
    t0 = time.time()

    N_META = 17
    nrows = max_markers if max_markers else None
    df_geno = pd.read_csv(DATA_DIR / "Iranian_Samples.csv",
                          skiprows=8, header=None, nrows=nrows, low_memory=False)

    # Extract sample IDs from original header (before skip)
    sample_ids_raw = pd.read_csv(DATA_DIR / "Iranian_Samples.csv", nrows=0).columns[N_META:]
    sample_ids = [str(c).strip() for c in sample_ids_raw if str(c).strip() and str(c).strip() != '*']

    # Genotype matrix: columns 17+ are samples, transpose to samples × markers
    X_raw = df_geno.iloc[:, N_META:].values.T
    del df_geno

    # Encode: '0'→0, '1'→1, '2'→2, '-/empty'→NaN via np.select (single pass)
    X_num = np.select(
        [X_raw == '0', X_raw == '1', X_raw == '2'],
        [0.0, 1.0, 2.0],
        default=np.nan
    ).astype(np.float32)
    del X_raw

    # Column-mean imputation with broadcasting (avoids materializing np.where tuple)
    col_means = np.nanmean(X_num, axis=0)
    nan_mask = np.isnan(X_num)
    X_num = np.where(nan_mask, col_means, X_num)
    missing_pct = nan_mask.sum() / nan_mask.size * 100
    print(f"  Genotype matrix: {X_num.shape} (samples × markers) "
          f"[missing={missing_pct:.1f}%] [{time.time()-t0:.1f}s]")

    # Load phenotypes
    print("\n[2/3] Loading phenotypes ...")
    pheno = pd.read_csv(DATA_DIR / "phenotype_iranian.csv")
    pheno_clean = pheno.iloc[1:].copy()
    pheno_clean.columns = ['GID', 'Heat_dtm', 'Heat_dth', 'Drought_dtm', 'Drought_dth']

    # Normalize GID: phenotype stores float (e.g. 156377.0), genotype stores int string
    pheno_clean['GID'] = pheno_clean['GID'].apply(
        lambda x: str(int(float(x))) if pd.notna(x) else None)
    for c in ['Heat_dtm', 'Heat_dth', 'Drought_dtm', 'Drought_dth']:
        pheno_clean[c] = pd.to_numeric(pheno_clean[c], errors='coerce')
    pheno_clean = pheno_clean.dropna(subset=['Heat_dtm', 'Heat_dth', 'Drought_dtm', 'Drought_dth'])
    print(f"  Phenotypes: {pheno_clean.shape[0]} samples, 4 traits")

    # Align samples
    print("\n[3/3] Aligning genotype and phenotype samples ...")
    geno_id_set = set(sample_ids)
    pheno_clean = pheno_clean[pheno_clean['GID'].isin(geno_id_set)]
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    aligned_indices = [id_to_idx[sid] for sid in pheno_clean['GID'].values]
    X = X_num[aligned_indices]
    y_dict = {}
    for trait in ['Heat_dtm', 'Heat_dth', 'Drought_dtm', 'Drought_dth']:
        y_dict[trait] = pheno_clean[trait].values.astype(np.float32)

    print(f"  Final: {X.shape[0]} samples, {X.shape[1]} markers, {len(y_dict)} traits")
    return X, y_dict, pheno_clean['GID'].values.tolist()


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    quick_test = '--full' not in sys.argv
    use_mexican = '--mexican' in sys.argv

    if use_mexican:
        print("ERROR: Mexican data not yet supported. Use Iranian data (default).")
        sys.exit(1)

    print("=" * 70)
    print("Maize Genomic Prediction — FGN+MICNN+EFM Ensemble System")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Quick Test: {quick_test}")
    print(f"  Dataset: Iranian")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load data (quick test: use 10K marker subset for speed)
    max_markers = 10000 if quick_test else None
    X_all, y_dict, sample_ids = load_iranian_data(max_markers=max_markers)
    traits = sorted(y_dict.keys())
    print(f"\nTraits: {traits}")

    # Filter low-variance markers
    var_thresh = 0.005
    vars_per_marker = np.var(X_all, axis=0)
    keep = vars_per_marker >= var_thresh
    if keep.sum() < X_all.shape[1]:
        X_all = X_all[:, keep]
        print(f"  Low-variance filter: {X_all.shape[1]} markers kept")

    # Model lists
    trad_names = ['RRBLUP', 'GBLUP', 'XGBoost', 'ElasticNet', 'GWAS_RRBLUP']
    dl_base_names = ['FGN', 'MICNN', 'FGN v2', 'MICNN v2',
                     'FGN v3', 'PreFGN', 'DeepKernelGP']
    dl_ensemble_names = ['FusionNet']
    extra_names = ['ResFGN']
    dl_names = dl_base_names + dl_ensemble_names
    all_names = trad_names + dl_names + extra_names

    traits_run = traits[:1] if quick_test else traits
    n_folds = 2 if quick_test else N_FOLDS
    total_t0 = time.time()
    all_results = {}

    for trait in traits_run:
        print(f"\n{'='*60}")
        print(f"Trait: {trait}")
        print(f"{'='*60}")

        y = y_dict[trait]
        y = y.astype(np.float32)

        n_snps = min(GWAS_TOP_K, max(50, X_all.shape[1] - 50))
        print(f"  {len(y)} samples, {X_all.shape[1]} markers -> ")
        print(f"  {n_snps} GWAS-selected (per-fold, no leakage)")

        results = {}
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

        # Per-fold storage
        fold_preds = {m: [] for m in all_names}
        fold_trues = []

        for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X_all)):
            print(f"\n  --- Fold {fold_i+1}/{n_folds} ---")
            Xtr_raw, Xte_raw = X_all[tr_idx], X_all[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]

            # GWAS selection
            maf = np.minimum(Xtr_raw.mean(axis=0)/2.0, 1.0 - Xtr_raw.mean(axis=0)/2.0)
            maf_idx = np.where(maf >= MAF_THRESHOLD)[0]
            if len(maf_idx) >= n_snps:
                gidx = gwas_select(Xtr_raw[:, maf_idx], ytr, n_snps)
                gidx = maf_idx[gidx]
            else:
                gidx = gwas_select(Xtr_raw, ytr, n_snps)

            Xtr = Xtr_raw[:, gidx]
            Xte = Xte_raw[:, gidx]

            # Scale
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
            Xte_s = sc.transform(Xte).astype(np.float32)

            fold_trues.extend(yte.tolist())

            # ── Traditional Models ──
            for mname in trad_names:
                t0 = time.time()
                try:
                    if mname == 'GBLUP':
                        G_tr = Xtr_s @ Xtr_s.T / n_snps
                        G_te = Xte_s @ Xtr_s.T / n_snps
                        from sklearn.kernel_ridge import KernelRidge
                        m = KernelRidge(alpha=1.0, kernel='precomputed')
                        m.fit(G_tr, ytr)
                        pred = m.predict(G_te)
                    else:
                        m = fit_traditional(mname, Xtr_s, ytr)
                        pred = m.predict(Xte_s)
                    fold_preds[mname].extend(pred.tolist())
                    r2 = r2_score(yte, pred)
                    print(f"    {mname:<15s} R²={r2:.4f} [{time.time()-t0:.1f}s]")
                except Exception as e:
                    print(f"    {mname:<15s} FAILED: {e}")
                    fold_preds[mname].extend([0]*len(yte))

            # ── DL Models ──
            for mname in dl_names:
                t0 = time.time()
                try:
                    model = create_model(mname, n_snps)
                    if mname == 'DeepKernelGP':
                        model = train_dkgp_wrapper(model, Xtr_s, ytr)
                    elif mname == 'PreFGN':
                        model = pretrain_prefgn_v2(model, torch.FloatTensor(Xtr_s),
                                                   epochs=60, lr=1e-3)
                        model = train_dl_model(model, Xtr_s, ytr)
                    else:
                        model = train_dl_model(model, Xtr_s, ytr)
                    model.eval()
                    with torch.no_grad():
                        pred = model(torch.FloatTensor(Xte_s)).numpy()
                    fold_preds[mname].extend(pred.tolist())
                    r2 = r2_score(yte, pred)
                    print(f"    {mname:<15s} R²={r2:.4f} [{time.time()-t0:.1f}s]")
                except Exception as e:
                    print(f"    {mname:<15s} FAILED: {e}")
                    fold_preds[mname].extend([0]*len(yte))

            # ── ResFGN ──
            t0 = time.time()
            try:
                ridge, fusion = fit_resfgn_components(Xtr_s, ytr, n_snps)
                y_pred_ridge = ridge.predict(Xte_s)
                res_pred = fusion(torch.FloatTensor(Xte_s)).detach().numpy()
                pred = y_pred_ridge + res_pred
                fold_preds['ResFGN'].extend(pred.tolist())
                r2 = r2_score(yte, pred)
                print(f"    {'ResFGN':<15s} R²={r2:.4f} [{time.time()-t0:.1f}s]")
            except Exception as e:
                print(f"    {'ResFGN':<15s} FAILED: {e}")
                fold_preds['ResFGN'].extend([0]*len(yte))

        # ── Aggregate per-fold results ──
        y_all_folds = np.array(fold_trues)

        for mname in all_names:
            preds = np.array(fold_preds[mname])
            mask = ~np.isnan(preds)
            if mask.sum() > 1:
                r2 = r2_score(y_all_folds[mask], preds[mask])
                corr = pearsonr(y_all_folds[mask], preds[mask])[0]
                rmse = np.sqrt(np.mean((y_all_folds[mask] - preds[mask])**2))
            else:
                r2, corr, rmse = -999, -999, -999

            if mname in trad_names:
                mtype = TYPE_TRAD
            elif mname == 'ResFGN':
                mtype = TYPE_HYBRID
            elif mname in ('Stacking (DL)', 'Stacking (All)', 'Trad Ensemble'):
                mtype = TYPE_ENSEMBLE
            else:
                mtype = TYPE_DL

            results[mname] = {'R2': r2, 'Correlation': corr, 'RMSE': rmse,
                              'Type': mtype}

        # ── Stacking (DL) ──
        dl_preds_train = {m: np.array(fold_preds[m]) for m in dl_names if m in fold_preds}
        if len(dl_preds_train) >= 2:
            meta_X = np.column_stack([v for v in dl_preds_train.values()])
            meta_ridge = RidgeCV(alphas=np.logspace(-2, 5, 20))
            meta_ridge.fit(meta_X, y_all_folds)
            stacking_pred = meta_ridge.predict(meta_X)
            results['Stacking (DL)'] = {
                'R2': r2_score(y_all_folds, stacking_pred),
                'Correlation': pearsonr(y_all_folds, stacking_pred)[0],
                'RMSE': 0.0, 'Type': TYPE_ENSEMBLE,
                'Meta_weights': meta_ridge.coef_.tolist(),
                'Base_models': list(dl_preds_train.keys())
            }
            print(f"\n  Stacking (DL) R²={results['Stacking (DL)']['R2']:.4f}")

        # ── Trad Ensemble ──
        trad_preds = {m: np.array(fold_preds[m]) for m in trad_names if m in fold_preds}
        if len(trad_preds) >= 2:
            trad_meta_X = np.column_stack([v for v in trad_preds.values()])
            trad_ridge = RidgeCV(alphas=np.logspace(-2, 5, 20))
            trad_ridge.fit(trad_meta_X, y_all_folds)
            trad_pred = trad_ridge.predict(trad_meta_X)
            results['Trad Ensemble'] = {
                'R2': r2_score(y_all_folds, trad_pred),
                'Correlation': pearsonr(y_all_folds, trad_pred)[0],
                'RMSE': 0.0, 'Type': TYPE_ENSEMBLE
            }

        # ── Stacking (All) ──
        all_preds = {m: np.array(fold_preds[m]) for m in all_names
                     if m in fold_preds and m not in ('ResFGN', 'Trad Ensemble',
                                                       'Stacking (DL)', 'Stacking (All)')}
        if len(all_preds) >= 2:
            all_meta_X = np.column_stack([v for v in all_preds.values()])
            all_ridge = RidgeCV(alphas=np.logspace(-2, 5, 20))
            all_ridge.fit(all_meta_X, y_all_folds)
            all_pred = all_ridge.predict(all_meta_X)
            results['Stacking (All)'] = {
                'R2': r2_score(y_all_folds, all_pred),
                'Correlation': pearsonr(y_all_folds, all_pred)[0],
                'RMSE': 0.0, 'Type': TYPE_ENSEMBLE,
                'Meta_weights': all_ridge.coef_.tolist(),
                'Base_models': list(all_preds.keys())
            }
            print(f"  Stacking (All) R²={results['Stacking (All)']['R2']:.4f}")

        all_results[trait] = results

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    if not quick_test:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")

        eval_models = list(all_results[traits_run[0]].keys())
        summ = {m: [] for m in eval_models}
        for t in traits_run:
            for m in eval_models:
                if m in all_results[t]:
                    summ[m].append(all_results[t][m]['R2'])

        ranked = sorted([(m, np.mean(summ[m])) for m in eval_models if summ[m]],
                        key=lambda x: x[1], reverse=True)
        for i, (m, r) in enumerate(ranked, 1):
            marker = " <-- BEST" if i == 1 else ""
            print(f"  {i:2d}. {m:<22s} {r:.4f}{marker}")

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(OUTPUT_DIR / f"ensemble_final_{ts}.json", 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print(f"Total time: {(time.time()-total_t0)/60:.1f} min")

    # Save intermediate
    with open(OUTPUT_DIR / "ensemble_intermediate.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nDone!")


if __name__ == '__main__':
    main()
