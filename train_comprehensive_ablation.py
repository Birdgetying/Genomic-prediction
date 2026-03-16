"""
================================================================================
综合消融实验与模型集成 - Comprehensive Ablation Study and Model Ensemble
================================================================================

功能：
1. 使用 sv_pro 数据路径训练多种基础模型 (RRBLUP, XGBoost, WheatGP, Enhanced, DNNGP)
2. 对增强模型进行消融实验：改变结构组件
3. 模型集成评估
4. 选出综合表现最好的模型

数据路径：严格使用 train_sv_pro.py 中的配置，禁止修改
"""

import os
import sys
import json
import time
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
import scipy.stats
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # 无显示器环境必须在import plt之前设置
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

# ============================================================================
# GPU初始化
# ============================================================================
def init_gpu():
    """初始化GPU环境"""
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("="*60)
    print("GPU INITIALIZATION")
    print("="*60)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[OK] CUDA is available")
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print(f"[INFO] Using CPU")
    
    print(f"\nFinal device: {device}")
    print("="*60 + "\n")
    return device

DEVICE = init_gpu()

def clear_gpu_memory():
    """清理GPU显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ============================================================================
# 配置 - 严格使用 sv_pro 的数据路径
# ============================================================================
class Config:
    """配置 - 数据路径与 train_sv_pro.py 完全一致"""
    # 数据路径 (禁止修改！与 train_sv_pro.py 完全一致)
    DATA_BASE_PATH = "/storage/public/home/2024110093/data/Variation/CSIAAS/"
    SV_VCF_PATH = DATA_BASE_PATH + "SV.new.vcf.gz"
    PHENOTYPE_PATH = DATA_BASE_PATH + "Phe.txt"
    VCF_ID_PATH = DATA_BASE_PATH + "VCFID.txt"
    
    # 数据加载参数
    MAX_VARIANTS = 10000  # 使用1万位点进行综合分析
    
    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 150
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.01
    GRADIENT_CLIP = 1.0
    
    # 增强模型基础参数
    EMBED_DIM = 128
    DEPTH = 3
    NUM_HEADS = 8
    DROP_RATE = 0.3
    DROP_PATH_RATE = 0.15
    CNN_CHANNELS = [16, 32, 64, 128]
    PATCH_SIZE = 50
    
    # RRBLUP
    RRBLUP_ALPHAS = np.logspace(-2, 5, 30)
    
    # XGBoost
    XGB_N_ESTIMATORS = 200
    XGB_MAX_DEPTH = 6
    XGB_LR = 0.1
    
    # 交叉验证
    N_FOLDS = 5
    EARLY_STOPPING_PATIENCE = 30
    
    # GPU
    USE_AMP = True
    
    # 输出
    OUTPUT_DIR = "./results/comprehensive_ablation"

# ============================================================================
# 高级正则化模块
# ============================================================================
class DropPath(nn.Module):
    """DropPath (Stochastic Depth)"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SpatialDropout1D(nn.Module):
    """Spatial Dropout"""
    def __init__(self, drop_prob: float = 0.2):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x
        x = x.permute(0, 2, 1)
        x = F.dropout2d(x.unsqueeze(-1), self.drop_prob, self.training).squeeze(-1)
        return x.permute(0, 2, 1)


class LayerScale(nn.Module):
    """Layer Scale"""
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class SwiGLU(nn.Module):
    """SwiGLU激活"""
    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: int, drop: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ============================================================================
# CNN模块
# ============================================================================
class LocalCNNExtractor(nn.Module):
    """CNN局部特征提取"""
    def __init__(self, in_channels: int = 1, channels: list = [8, 16, 32],
                 drop_rate: float = 0.2):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch),
                nn.GELU(),
            ])
            prev_ch = ch
        layers.append(SpatialDropout1D(drop_rate))
        self.conv = nn.Sequential(*layers)
        self.out_channels = channels[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        return x


# ============================================================================
# GWAS门控融合
# ============================================================================
class GWASGatedFusion(nn.Module):
    """GWAS先验门控融合"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor, gwas_weights: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if gwas_weights.dim() == 1:
            gwas_weights = gwas_weights.unsqueeze(0)
        if gwas_weights.size(0) == 1 and B > 1:
            gwas_weights = gwas_weights.expand(B, -1)
        gwas_embed = gwas_weights.unsqueeze(-1).expand(-1, -1, C)
        combined = torch.cat([x, gwas_embed], dim=-1)
        gate = self.gate(combined)
        out = gate * x + (1 - gate) * self.proj(gwas_embed)
        return out


# ============================================================================
# 多尺度注意力
# ============================================================================
class MultiScaleAttention(nn.Module):
    """多尺度注意力"""
    def __init__(self, dim: int, num_heads: int = 4,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out, attn.mean(dim=1)


# ============================================================================
# Transformer块
# ============================================================================
class EnhancedTransformerBlock(nn.Module):
    """增强版Transformer块"""
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.67,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 layer_scale_init: float = 1e-5, use_swiglu: bool = True,
                 use_layer_scale: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiScaleAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        
        if use_swiglu:
            self.mlp = SwiGLU(dim, hidden_dim, dim, drop)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(drop)
            )
        
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.ls1 = LayerScale(dim, layer_scale_init)
            self.ls2 = LayerScale(dim, layer_scale_init)
        
    def forward(self, x: torch.Tensor):
        attn_out, attn_weights = self.attn(self.norm1(x))
        if self.use_layer_scale:
            x = x + self.drop_path1(self.ls1(attn_out))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        else:
            x = x + self.drop_path1(attn_out)
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x, attn_weights


# ============================================================================
# 可配置增强模型 - 用于消融实验
# ============================================================================
class ConfigurableEnhancedNet(nn.Module):
    """
    可配置增强网络 - 支持消融实验
    
    可开关组件:
    - use_cnn: 是否使用CNN
    - use_transformer: 是否使用Transformer
    - use_gwas_fusion: 是否使用GWAS融合
    - use_drop_path: 是否使用DropPath
    - use_layer_scale: 是否使用LayerScale
    - use_swiglu: 是否使用SwiGLU
    - use_learnable_importance: 是否使用可学习重要性
    """
    def __init__(self, 
                 n_features: int,
                 embed_dim: int = 128,
                 depth: int = 3,
                 num_heads: int = 8,
                 patch_size: int = 50,
                 drop_rate: float = 0.2,
                 drop_path_rate: float = 0.1,
                 cnn_channels: list = [16, 32, 64],
                 # 消融开关
                 use_cnn: bool = True,
                 use_transformer: bool = True,
                 use_gwas_fusion: bool = True,
                 use_drop_path: bool = True,
                 use_layer_scale: bool = True,
                 use_swiglu: bool = True,
                 use_learnable_importance: bool = True):
        super().__init__()
        
        self.n_features = n_features
        self.patch_size = patch_size
        self.n_patches = (n_features + patch_size - 1) // patch_size
        self.embed_dim = embed_dim
        
        # 消融配置
        self.use_cnn = use_cnn
        self.use_transformer = use_transformer
        self.use_gwas_fusion = use_gwas_fusion
        self.use_drop_path = use_drop_path
        self.use_learnable_importance = use_learnable_importance
        
        # CNN模块
        if use_cnn:
            self.cnn = LocalCNNExtractor(1, cnn_channels, drop_rate)
            cnn_out_dim = cnn_channels[-1] * patch_size
        else:
            self.cnn = None
            cnn_out_dim = patch_size
        
        # Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Linear(cnn_out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5)
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # GWAS融合
        if use_gwas_fusion:
            self.gwas_fusion = GWASGatedFusion(embed_dim)
        else:
            self.gwas_fusion = None
        
        # Transformer
        if use_transformer:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate if use_drop_path else 0, depth)]
            self.blocks = nn.ModuleList([
                EnhancedTransformerBlock(
                    embed_dim, num_heads, 
                    drop=drop_rate, attn_drop=drop_rate, 
                    drop_path=dpr[i],
                    use_swiglu=use_swiglu,
                    use_layer_scale=use_layer_scale
                )
                for i in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.blocks = None
            self.norm = nn.LayerNorm(embed_dim)
        
        # 预测头
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # 可学习重要性
        if use_learnable_importance:
            self.feature_importance = nn.Parameter(torch.zeros(n_features))
        else:
            self.feature_importance = None
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor, gwas_weights: torch.Tensor = None) -> torch.Tensor:
        B = x.size(0)
        
        # 特征重要性加权
        if self.feature_importance is not None:
            importance = torch.sigmoid(self.feature_importance)
            x = x * importance.unsqueeze(0)
        
        # CNN特征
        if self.use_cnn:
            x_cnn = self.cnn(x)
        else:
            x_cnn = x.unsqueeze(1)
        
        # Reshape to patches
        L = x_cnn.size(2)
        if L % self.patch_size != 0:
            pad = self.patch_size - (L % self.patch_size)
            x_cnn = F.pad(x_cnn, (0, pad))
        x_cnn = x_cnn.view(B, x_cnn.size(1), self.n_patches, self.patch_size)
        x_cnn = x_cnn.permute(0, 2, 1, 3).reshape(B, self.n_patches, -1)
        
        # Patch embedding
        x_embed = self.patch_embed(x_cnn)
        
        # CLS token + position
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_embed = torch.cat([cls_tokens, x_embed], dim=1)
        x_embed = x_embed + self.pos_embed
        
        # GWAS融合
        if self.gwas_fusion is not None and gwas_weights is not None:
            gwas_patched = self._aggregate_gwas(gwas_weights)
            cls_part = x_embed[:, :1, :]
            patch_part = x_embed[:, 1:, :]
            patch_part_fused = self.gwas_fusion(patch_part, gwas_patched)
            x_embed = torch.cat([cls_part, patch_part_fused], dim=1)
        
        # Transformer
        if self.blocks is not None:
            for block in self.blocks:
                x_embed, _ = block(x_embed)
        x_embed = self.norm(x_embed)
        
        # 输出
        if self.use_transformer:
            cls_output = x_embed[:, 0]
        else:
            cls_output = x_embed.mean(dim=1)
        
        output = self.head(cls_output)
        return output.squeeze(-1)
    
    def _aggregate_gwas(self, gwas_weights: torch.Tensor) -> torch.Tensor:
        if gwas_weights.dim() == 1:
            gwas_weights = gwas_weights.unsqueeze(0)
        B = gwas_weights.size(0)
        L = gwas_weights.size(1)
        if L % self.patch_size != 0:
            pad = self.patch_size - (L % self.patch_size)
            gwas_weights = F.pad(gwas_weights, (0, pad))
        gwas_patched = gwas_weights.view(B, self.n_patches, self.patch_size)
        gwas_patched = gwas_patched.mean(dim=-1)
        return gwas_patched


# ============================================================================
# WheatGP模型
# ============================================================================
class WheatGPModel(nn.Module):
    """WheatGP模型"""
    def __init__(self, n_features, n_subnetworks=5, hidden_dim=128):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.chunk_size = n_features // n_subnetworks
        
        self.subnetworks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 2, kernel_size=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(2, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(4, 8, kernel_size=9, padding=1),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.AdaptiveAvgPool1d(16)
            )
            for _ in range(n_subnetworks)
        ])
        
        self.lstm = nn.LSTM(8 * 16 * n_subnetworks, hidden_dim, batch_first=True)
        self.lstm_drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        outputs = []
        
        for i, subnet in enumerate(self.subnetworks):
            start = i * self.chunk_size
            end = start + self.chunk_size if i < self.n_subnetworks - 1 else x.size(1)
            chunk = x[:, start:end].unsqueeze(1)
            out = subnet(chunk)
            outputs.append(out.view(batch_size, -1))
        
        combined = torch.cat(outputs, dim=1).unsqueeze(1)
        lstm_out, _ = self.lstm(combined)
        lstm_out = self.lstm_drop(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        return output.squeeze(-1)


# ============================================================================
# DNNGP模型
# ============================================================================
class DNNGPNet(nn.Module):
    """DNNGP模型"""
    def __init__(self, n_features, embed_dim=128, depth=2, num_heads=8,
                 patch_size=50, drop_rate=0.18, drop_path_rate=0.15,
                 cnn_channels=[16, 32, 64]):
        super().__init__()
        
        self.n_features = n_features
        self.patch_size = patch_size
        self.n_patches = (n_features + patch_size - 1) // patch_size
        
        cnn_layers = []
        in_ch = 1
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_ch = cnn_channels[-1]
        
        patch_dim = self.cnn_out_ch * patch_size
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim) * 0.02)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            self._make_transformer_block(embed_dim, num_heads, drop_rate, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _make_transformer_block(self, dim, num_heads, drop, drop_path):
        return nn.ModuleDict({
            'norm1': nn.LayerNorm(dim),
            'attn': nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True),
            'norm2': nn.LayerNorm(dim),
            'mlp': nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(dim * 2, dim),
                nn.Dropout(drop)
            ),
            'drop_path': DropPath(drop_path) if drop_path > 0. else nn.Identity()
        })
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.size(0)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        
        L = x.size(2)
        if L % self.patch_size != 0:
            pad = self.patch_size - (L % self.patch_size)
            x = F.pad(x, (0, pad))
        x = x.view(B, self.cnn_out_ch, self.n_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.n_patches, -1)
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x_norm = block['norm1'](x)
            attn_out, _ = block['attn'](x_norm, x_norm, x_norm)
            x = x + block['drop_path'](attn_out)
            x = x + block['drop_path'](block['mlp'](block['norm2'](x)))
        
        x = self.norm(x)
        cls_output = x[:, 0]
        output = self.head(cls_output)
        return output.squeeze(-1)


# ============================================================================
# 数据加载
# ============================================================================
def load_sv_data(config: Config):
    """加载SV数据 - 使用与 train_sv_pro.py 相同的逻辑"""
    print("\n" + "="*60)
    print("Loading SV Data")
    print("="*60)
    print(f"VCF Path: {config.SV_VCF_PATH}")
    print(f"Phenotype Path: {config.PHENOTYPE_PATH}")
    sys.stdout.flush()  # 确保日志立即输出
    
    X = None
    sample_ids = None
    
    # 尝试加载VCF
    try:
        import cyvcf2
        print("[INFO] Using cyvcf2 for VCF parsing")
        sys.stdout.flush()
        
        if not os.path.exists(config.SV_VCF_PATH):
            raise FileNotFoundError(f"VCF file not found: {config.SV_VCF_PATH}")
        
        vcf = cyvcf2.VCF(config.SV_VCF_PATH)
        sample_ids = vcf.samples
        print(f"  Samples in VCF: {len(sample_ids)}")
        sys.stdout.flush()
        
        genotypes = []
        variant_count = 0
        
        for variant in vcf:
            if variant_count >= config.MAX_VARIANTS:
                break
            gt = variant.gt_types
            genotypes.append(gt)
            variant_count += 1
            if variant_count % 2000 == 0:
                print(f"    Loaded {variant_count} variants...")
                sys.stdout.flush()
        
        vcf.close()
        X = np.array(genotypes, dtype=np.float32).T
        print(f"  Genotype matrix: {X.shape}")
        
    except ImportError:
        print("[WARNING] cyvcf2 not available, using simulated data")
    except FileNotFoundError as e:
        print(f"[WARNING] {e}, using simulated data")
    except Exception as e:
        print(f"[WARNING] VCF loading failed: {e}, using simulated data")
    
    # Fallback: 生成模拟数据
    if X is None:
        print("[INFO] Generating simulated genotype data...")
        np.random.seed(42)
        n_samples = 800
        n_variants = config.MAX_VARIANTS
        X = np.random.randint(0, 3, size=(n_samples, n_variants)).astype(np.float32)
        sample_ids = [f"Sample_{i}" for i in range(n_samples)]
        print(f"  Using simulated data: {X.shape}")
    
    # 加载表型
    print(f"\nLoading phenotype from: {config.PHENOTYPE_PATH}")
    try:
        pheno_df = pd.read_csv(config.PHENOTYPE_PATH, sep='\t')
        print(f"  Phenotype columns: {list(pheno_df.columns)}")
        
        # 获取表型列
        if 'DTH_sum' in pheno_df.columns:
            y_col = 'DTH_sum'
        elif len(pheno_df.columns) > 1:
            y_col = pheno_df.columns[1]
        else:
            y_col = pheno_df.columns[0]
        
        y = pheno_df[y_col].values.astype(np.float32)
        pheno_ids = pheno_df.iloc[:, 0].values if pheno_df.shape[1] > 1 else np.arange(len(y))
        
    except Exception as e:
        print(f"[WARNING] Failed to load phenotype: {e}")
        print("  Using simulated phenotype")
        y = np.random.randn(X.shape[0]).astype(np.float32) * 10 + 50
        pheno_ids = sample_ids
    
    # 匹配样本
    n_samples = min(X.shape[0], len(y))
    X = X[:n_samples]
    y = y[:n_samples]
    
    # 处理缺失值
    X = np.nan_to_num(X, nan=0.0)
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\nFinal data shape: X={X.shape}, y={y.shape}")
    print(f"Phenotype range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y


# ============================================================================
# 训练函数
# ============================================================================
def train_rrblup(X_train, X_val, y_train, y_val, config):
    """训练RRBLUP"""
    model = RidgeCV(alphas=config.RRBLUP_ALPHAS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    corr, _ = scipy.stats.pearsonr(y_val, y_pred)
    return y_pred, r2, corr


def train_xgboost(X_train, X_val, y_train, y_val, config, seed=42):
    """训练XGBoost"""
    model = xgb.XGBRegressor(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LR,
        random_state=seed,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    corr, _ = scipy.stats.pearsonr(y_val, y_pred)
    return y_pred, r2, corr


def train_deep_model(model, X_train, X_val, y_train, y_val, config, model_name=""):
    """训练深度学习模型"""
    model = model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                                   weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    criterion = nn.MSELoss()
    
    if config.USE_AMP and torch.cuda.is_available():
        scaler = GradScaler()
    else:
        scaler = None
    
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast('cuda'):
                pred = model(X_train_t)
                loss = criterion(pred, y_train_t)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(X_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
        
        scheduler.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            if scaler is not None:
                with autocast('cuda'):
                    val_pred = model(X_val_t)
            else:
                val_pred = model(X_val_t)
            val_loss = criterion(val_pred, torch.FloatTensor(y_val).to(DEVICE)).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                break
    
    # 加载最优权重
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val_t).cpu().numpy()
    
    r2 = r2_score(y_val, y_pred)
    corr, _ = scipy.stats.pearsonr(y_val, y_pred)
    
    # 清理
    del X_train_t, y_train_t, X_val_t
    clear_gpu_memory()
    
    return y_pred, r2, corr


# ============================================================================
# 消融实验配置
# ============================================================================
ABLATION_CONFIGS = {
    # 完整版
    'Enhanced_Full': {
        'use_cnn': True, 'use_transformer': True, 'use_gwas_fusion': True,
        'use_drop_path': True, 'use_layer_scale': True, 'use_swiglu': True,
        'use_learnable_importance': True
    },
    # 无CNN
    'Enhanced_NoCNN': {
        'use_cnn': False, 'use_transformer': True, 'use_gwas_fusion': True,
        'use_drop_path': True, 'use_layer_scale': True, 'use_swiglu': True,
        'use_learnable_importance': True
    },
    # 无Transformer (仅CNN)
    'Enhanced_NoTransformer': {
        'use_cnn': True, 'use_transformer': False, 'use_gwas_fusion': True,
        'use_drop_path': False, 'use_layer_scale': False, 'use_swiglu': False,
        'use_learnable_importance': True
    },
    # 无GWAS融合
    'Enhanced_NoGWAS': {
        'use_cnn': True, 'use_transformer': True, 'use_gwas_fusion': False,
        'use_drop_path': True, 'use_layer_scale': True, 'use_swiglu': True,
        'use_learnable_importance': True
    },
    # 无DropPath
    'Enhanced_NoDropPath': {
        'use_cnn': True, 'use_transformer': True, 'use_gwas_fusion': True,
        'use_drop_path': False, 'use_layer_scale': True, 'use_swiglu': True,
        'use_learnable_importance': True
    },
    # 无LayerScale
    'Enhanced_NoLayerScale': {
        'use_cnn': True, 'use_transformer': True, 'use_gwas_fusion': True,
        'use_drop_path': True, 'use_layer_scale': False, 'use_swiglu': True,
        'use_learnable_importance': True
    },
    # 无SwiGLU
    'Enhanced_NoSwiGLU': {
        'use_cnn': True, 'use_transformer': True, 'use_gwas_fusion': True,
        'use_drop_path': True, 'use_layer_scale': True, 'use_swiglu': False,
        'use_learnable_importance': True
    },
    # 无可学习重要性
    'Enhanced_NoImportance': {
        'use_cnn': True, 'use_transformer': True, 'use_gwas_fusion': True,
        'use_drop_path': True, 'use_layer_scale': True, 'use_swiglu': True,
        'use_learnable_importance': False
    },
    # 简化版 (仅CNN+Transformer)
    'Enhanced_Simple': {
        'use_cnn': True, 'use_transformer': True, 'use_gwas_fusion': False,
        'use_drop_path': False, 'use_layer_scale': False, 'use_swiglu': False,
        'use_learnable_importance': False
    },
}

# 不同深度配置
DEPTH_CONFIGS = {
    'Enhanced_Depth1': {'depth': 1},
    'Enhanced_Depth2': {'depth': 2},
    'Enhanced_Depth4': {'depth': 4},
    'Enhanced_Depth6': {'depth': 6},
}

# 不同embed_dim配置
EMBED_DIM_CONFIGS = {
    'Enhanced_Dim64': {'embed_dim': 64},
    'Enhanced_Dim256': {'embed_dim': 256},
}


# ============================================================================
# 主评估函数
# ============================================================================
def run_comprehensive_evaluation(X, y, config: Config):
    """运行综合评估"""
    print("\n" + "="*70)
    print("COMPREHENSIVE ABLATION STUDY AND MODEL ENSEMBLE")
    print("="*70)
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Folds: {config.N_FOLDS}")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_mean, y_std = y.mean(), y.std() + 1e-8
    y_norm = (y - y_mean) / y_std
    
    n_features = X_scaled.shape[1]
    
    # 结果存储
    all_results = {}
    all_predictions = {}
    
    # K折交叉验证
    try:
        y_binned = pd.qcut(y_norm, q=5, labels=False, duplicates='drop')
        kfold = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
        fold_iterator = list(kfold.split(X_scaled, y_binned))
    except:
        kfold = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
        fold_iterator = list(kfold.split(X_scaled))
    
    # ========== 1. 基础模型 ==========
    print("\n" + "="*60)
    print("PART 1: BASE MODELS")
    print("="*60)
    
    base_models = ['RRBLUP', 'XGBoost', 'WheatGP', 'DNNGP']
    
    for model_name in base_models:
        print(f"\n--- Training {model_name} ---")
        fold_r2s = []
        fold_corrs = []
        fold_preds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(fold_iterator):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_norm[train_idx], y_norm[val_idx]
            
            if model_name == 'RRBLUP':
                y_pred, r2, corr = train_rrblup(X_train, X_val, y_train, y_val, config)
            elif model_name == 'XGBoost':
                y_pred, r2, corr = train_xgboost(X_train, X_val, y_train, y_val, config)
            elif model_name == 'WheatGP':
                model = WheatGPModel(n_features)
                y_pred, r2, corr = train_deep_model(model, X_train, X_val, y_train, y_val, config, model_name)
            elif model_name == 'DNNGP':
                model = DNNGPNet(n_features, embed_dim=config.EMBED_DIM, depth=2)
                y_pred, r2, corr = train_deep_model(model, X_train, X_val, y_train, y_val, config, model_name)
            
            fold_r2s.append(r2)
            fold_corrs.append(corr)
            fold_preds.append((val_idx, y_pred))
            print(f"  Fold {fold_idx+1}: R²={r2:.4f}, Corr={corr:.4f}")
            clear_gpu_memory()
        
        mean_r2 = np.mean(fold_r2s)
        std_r2 = np.std(fold_r2s)
        mean_corr = np.mean(fold_corrs)
        
        all_results[model_name] = {
            'r2_mean': mean_r2, 'r2_std': std_r2,
            'corr_mean': mean_corr, 'fold_r2s': fold_r2s
        }
        all_predictions[model_name] = fold_preds
        print(f"  {model_name} Final: R²={mean_r2:.4f}±{std_r2:.4f}, Corr={mean_corr:.4f}")
    
    # ========== 2. 增强模型消融实验 ==========
    print("\n" + "="*60)
    print("PART 2: ENHANCED MODEL ABLATION STUDY")
    print("="*60)
    
    for ablation_name, ablation_config in ABLATION_CONFIGS.items():
        print(f"\n--- Training {ablation_name} ---")
        print(f"  Config: {ablation_config}")
        
        fold_r2s = []
        fold_corrs = []
        fold_preds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(fold_iterator):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_norm[train_idx], y_norm[val_idx]
            
            model = ConfigurableEnhancedNet(
                n_features=n_features,
                embed_dim=config.EMBED_DIM,
                depth=config.DEPTH,
                num_heads=config.NUM_HEADS,
                patch_size=config.PATCH_SIZE,
                drop_rate=config.DROP_RATE,
                drop_path_rate=config.DROP_PATH_RATE,
                cnn_channels=config.CNN_CHANNELS[:3],  # 使用前3个通道
                **ablation_config
            )
            
            y_pred, r2, corr = train_deep_model(model, X_train, X_val, y_train, y_val, config, ablation_name)
            fold_r2s.append(r2)
            fold_corrs.append(corr)
            fold_preds.append((val_idx, y_pred))
            print(f"  Fold {fold_idx+1}: R²={r2:.4f}, Corr={corr:.4f}")
            clear_gpu_memory()
        
        mean_r2 = np.mean(fold_r2s)
        std_r2 = np.std(fold_r2s)
        mean_corr = np.mean(fold_corrs)
        
        all_results[ablation_name] = {
            'r2_mean': mean_r2, 'r2_std': std_r2,
            'corr_mean': mean_corr, 'fold_r2s': fold_r2s,
            'config': ablation_config
        }
        all_predictions[ablation_name] = fold_preds
        print(f"  {ablation_name} Final: R²={mean_r2:.4f}±{std_r2:.4f}, Corr={mean_corr:.4f}")
    
    # ========== 3. 深度实验 ==========
    print("\n" + "="*60)
    print("PART 3: DEPTH ABLATION")
    print("="*60)
    
    for depth_name, depth_config in DEPTH_CONFIGS.items():
        print(f"\n--- Training {depth_name} ---")
        
        fold_r2s = []
        fold_corrs = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(fold_iterator):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_norm[train_idx], y_norm[val_idx]
            
            model = ConfigurableEnhancedNet(
                n_features=n_features,
                embed_dim=config.EMBED_DIM,
                depth=depth_config['depth'],
                num_heads=config.NUM_HEADS,
                patch_size=config.PATCH_SIZE,
                drop_rate=config.DROP_RATE,
                cnn_channels=config.CNN_CHANNELS[:3],
            )
            
            y_pred, r2, corr = train_deep_model(model, X_train, X_val, y_train, y_val, config, depth_name)
            fold_r2s.append(r2)
            fold_corrs.append(corr)
            print(f"  Fold {fold_idx+1}: R²={r2:.4f}, Corr={corr:.4f}")
            clear_gpu_memory()
        
        mean_r2 = np.mean(fold_r2s)
        std_r2 = np.std(fold_r2s)
        mean_corr = np.mean(fold_corrs)
        
        all_results[depth_name] = {
            'r2_mean': mean_r2, 'r2_std': std_r2,
            'corr_mean': mean_corr, 'fold_r2s': fold_r2s,
            'depth': depth_config['depth']
        }
        print(f"  {depth_name} Final: R²={mean_r2:.4f}±{std_r2:.4f}")
    
    # ========== 4. 模型集成 ==========
    print("\n" + "="*60)
    print("PART 4: MODEL ENSEMBLE")
    print("="*60)
    
    # 简单平均集成
    ensemble_models = ['RRBLUP', 'XGBoost', 'WheatGP', 'DNNGP', 'Enhanced_Full']
    available_models = [m for m in ensemble_models if m in all_predictions]
    
    if len(available_models) >= 2:
        print(f"\nEnsemble models: {available_models}")
        
        # 计算集成预测
        ensemble_r2s = []
        for fold_idx in range(config.N_FOLDS):
            fold_preds = {}
            for model_name in available_models:
                for idx, pred in all_predictions[model_name]:
                    if fold_idx == 0 or idx is not None:
                        val_idx, y_pred = all_predictions[model_name][fold_idx]
                        fold_preds[model_name] = y_pred
                        break
            
            # 简单平均
            avg_pred = np.mean(list(fold_preds.values()), axis=0)
            val_idx = fold_iterator[fold_idx][1]
            y_val = y_norm[val_idx]
            
            r2 = r2_score(y_val, avg_pred)
            ensemble_r2s.append(r2)
            print(f"  Fold {fold_idx+1} Ensemble: R²={r2:.4f}")
        
        mean_ensemble_r2 = np.mean(ensemble_r2s)
        all_results['Ensemble_Average'] = {
            'r2_mean': mean_ensemble_r2,
            'r2_std': np.std(ensemble_r2s),
            'models': available_models
        }
        print(f"\n  Ensemble Average Final: R²={mean_ensemble_r2:.4f}")
    
    # ========== 5. 汇总结果 ==========
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # 按R²排序
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['r2_mean'], reverse=True)
    
    print(f"\n{'Model':<30} {'R² (mean±std)':<20} {'Correlation':<15}")
    print("-"*70)
    
    for model_name, result in sorted_results:
        r2_str = f"{result['r2_mean']:.4f}±{result.get('r2_std', 0):.4f}"
        corr_str = f"{result.get('corr_mean', 'N/A'):.4f}" if 'corr_mean' in result else "N/A"
        print(f"{model_name:<30} {r2_str:<20} {corr_str:<15}")
    
    print("-"*70)
    
    # 最佳模型
    best_model = sorted_results[0]
    print(f"\n★ BEST MODEL: {best_model[0]}")
    print(f"   R² = {best_model[1]['r2_mean']:.4f}")
    
    # 保存结果
    results_file = os.path.join(config.OUTPUT_DIR, f'comprehensive_results_{int(time.time())}.json')
    
    # 转换numpy类型
    def convert_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump({
            'results': convert_to_native(all_results),
            'best_model': best_model[0],
            'best_r2': float(best_model[1]['r2_mean']),
            'timestamp': datetime.now().isoformat(),
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # 创建可视化
    create_visualization(all_results, config.OUTPUT_DIR)
    
    return all_results, best_model


def create_visualization(results: Dict, output_dir: str):
    """创建结果可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. 所有模型对比
    ax1 = axes[0, 0]
    models = list(results.keys())
    r2_means = [results[m]['r2_mean'] for m in models]
    r2_stds = [results[m].get('r2_std', 0) for m in models]
    
    colors = ['#4CAF50' if 'Enhanced' not in m else '#2196F3' for m in models]
    bars = ax1.barh(range(len(models)), r2_means, xerr=r2_stds, color=colors, alpha=0.8, capsize=3)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models, fontsize=8)
    ax1.set_xlabel('R² Score')
    ax1.set_title('All Models Comparison')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # 标注最优
    best_idx = np.argmax(r2_means)
    bars[best_idx].set_color('#FF5722')
    
    # 2. 基础模型 vs 增强模型消融
    ax2 = axes[0, 1]
    base_models = ['RRBLUP', 'XGBoost', 'WheatGP', 'DNNGP']
    ablation_models = [m for m in models if 'Enhanced' in m and 'Depth' not in m and 'Dim' not in m]
    
    base_r2 = [results[m]['r2_mean'] for m in base_models if m in results]
    ablation_r2 = [results[m]['r2_mean'] for m in ablation_models if m in results]
    
    x = np.arange(max(len(base_r2), len(ablation_r2)))
    width = 0.35
    
    if base_r2:
        ax2.bar(x[:len(base_r2)] - width/2, base_r2, width, label='Base Models', color='#4CAF50', alpha=0.8)
    if ablation_r2:
        ax2.bar(x[:len(ablation_r2)] + width/2, ablation_r2, width, label='Enhanced Variants', color='#2196F3', alpha=0.8)
    
    ax2.set_ylabel('R² Score')
    ax2.set_title('Base vs Enhanced Models')
    ax2.legend()
    ax2.set_xticks(x)
    
    # 3. 深度消融
    ax3 = axes[1, 0]
    depth_models = [m for m in models if 'Depth' in m]
    if depth_models:
        depths = [results[m].get('depth', int(m.split('Depth')[-1])) for m in depth_models]
        depth_r2 = [results[m]['r2_mean'] for m in depth_models]
        
        sorted_idx = np.argsort(depths)
        depths = [depths[i] for i in sorted_idx]
        depth_r2 = [depth_r2[i] for i in sorted_idx]
        
        ax3.plot(depths, depth_r2, 'o-', linewidth=2, markersize=10, color='#9C27B0')
        ax3.set_xlabel('Transformer Depth')
        ax3.set_ylabel('R² Score')
        ax3.set_title('Impact of Transformer Depth')
        ax3.grid(True, alpha=0.3)
    
    # 4. 消融实验详细对比
    ax4 = axes[1, 1]
    ablation_only = [m for m in models if 'Enhanced' in m and 'Depth' not in m and 'Dim' not in m]
    if ablation_only:
        ablation_r2 = [results[m]['r2_mean'] for m in ablation_only]
        
        # 简化名称
        short_names = [m.replace('Enhanced_', '') for m in ablation_only]
        
        ax4.barh(range(len(short_names)), ablation_r2, color='#2196F3', alpha=0.8)
        ax4.set_yticks(range(len(short_names)))
        ax4.set_yticklabels(short_names, fontsize=9)
        ax4.set_xlabel('R² Score')
        ax4.set_title('Enhanced Model Ablation Study')
        
        # 标注Full版本
        if 'Full' in short_names:
            full_idx = short_names.index('Full')
            ax4.get_children()[full_idx].set_color('#FF5722')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comprehensive_results_{int(time.time())}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_dir}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("="*70)
    print("COMPREHENSIVE ABLATION STUDY AND MODEL ENSEMBLE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    X, y = load_sv_data(config)
    
    # 运行综合评估
    results, best_model = run_comprehensive_evaluation(X, y, config)
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best Model: {best_model[0]} (R²={best_model[1]['r2_mean']:.4f})")


if __name__ == '__main__':
    main()
