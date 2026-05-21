# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**语言规则：所有回复必须使用中文。代码注释可以使用英文或中文，但面向用户的说明、分析、总结一律使用中文。**
**代码修改后保存git并推送，网络错误不重复尝试。**
**新建 Python 脚本 (.py) 或 Shell 脚本 (.sh) 后必须将其加入 .gitignore 白名单 (添加 `!文件名` 行)，否则 git 不会追踪该文件。**
**新建或修改 Python 脚本时必须设置全局随机种子，确保实验可复现：**
```python
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# PYTHONHASHSEED 必须在进程启动前设置（CPython 只在解释器初始化时读取）：
#   启动脚本前执行: export PYTHONHASHSEED=42
```

## Project Overview

Genomic prediction/selection research — reproducing and extending deep learning models for predicting crop and animal phenotypes from genomic variants (SNPs, INDELs, SVs). The primary reference paper is "Cropformer: An interpretable deep learning framework for crop genomic prediction" (Wang et al., 2025).

## Key Sub-Projects

- **`cropformer/`** — Custom Transformer-based genomic prediction framework (PyTorch). Core modules: `model.py` (GenomicTransformer, CropformerV3), `data.py` (data loading/generation), `trainer.py`.
- **`WheatGP/WheatGP-main/`** — CNN+LSTM model for wheat polyploid genomic prediction (PyTorch). Has a tkinter GUI.
- **`DNNGP-main/`** — Deep neural network for genomic prediction using TensorFlow 2.6. Used for baseline comparisons.
- **`CropG2P-main/`** — Another genomic prediction framework with data preprocessing pipeline and model explanation tools.
- **`cattle_data/genomic-FM-main/`** — Genomic foundation model for cattle (finetuning pretrained DNA language models like DNABERT, HyenaDNA, Nucleotide Transformer).

## Environment

- Python 3.9–3.10 with PyTorch ≥2.0 (primary), TensorFlow 2.6 (DNNGP only)
- GPU required for training; CUDA expected
- HPC cluster uses PBS/Torque job scheduler (`.jsub` files)
- Key deps: `numpy`, `pandas`, `torch`, `scikit-learn`, `xgboost`, `scipy`, `matplotlib`

```
pip install -r requirements.txt
```

## Running Models

**Main entry point (Cropformer):**
```
python run.py --mode train --n_samples 20000 --n_snps 5000
python run.py --mode demo
```

**Key training scripts (root directory, in approximate order of complexity):**
- `efficient_rrblup.py` — Fast RRBLUP baseline on wheat2000 PCA data
- `train_optimized_model.py` — GWAS-filtered + PCA + Transformer on wheat2000 (good starting point)
- `train_enhanced_automl.py` — CNN-Transformer with Optuna AutoML
- `train_enhanced_automl_separate_gwas.py` — Separate processing of SNP/INDEL/SV from VCF with GWAS integration
- `train_sv_enhanced.py` — Full pipeline with SV (structural variation) data
- `train_sv_pro_gwas.py` — Most advanced: GWAS-guided marker optimization across all variant types
- `train_comprehensive_ablation.py` — Ablation study comparing multiple model architectures + baselines

**GUI platform:**
```
python genomic_selection_gui.py  # Gradio web UI at http://127.0.0.1:7860
```

**Ensemble & comparison:**
```
python ensemble_models.py --dataset wheat599 --strategy weighted
python compare_all_models.py
```

**Haplotype analysis:**
```
python haplotype_phenotype_analysis.py
```

## Data Locations

| Directory | Content |
|-----------|---------|
| `dnngp_data/wheat599/` | 599 wheat lines, PCA features |
| `dnngp_data/wheat2000/SNP_pca/` | 2000 wheat lines, PCA-reduced SNPs (~1691 dims) |
| `dnngp_data/wheat2000/SNP_origin/` | Raw 33,710-dim SNP genotypes |
| `dnngp_data/maize1404/` | 1404 maize lines |
| `dnngp_data/tomato332/` | 332 tomato accessions |
| `data2/` | Iranian & Mexican wheat samples with phenotypes |
| `Variation/CSIAAS/` | VCF files (SNP, INDEL, SV) from cattle |
| `data/` | Simulated genomic data for algorithm testing |

## HPC Job Submission

作业提交到景行平台 (PBS/Torque):
```
jsub < submit_xxx.jsub
jjobs          # check queue
tail -f logs/xxx_output.*   # monitor logs
```

## Key Architectural Patterns

1. **Data preprocessing is critical**: PCA features should be mean-centered but NOT fully standardized (preserves variance structure). GWAS-guided marker selection significantly improves prediction.
2. **Cross-validation**: All experiments use 5-fold CV (often stratified/quantile-split for fair comparison).
3. **Baseline models**: RRBLUP (RidgeCV) and XGBoost are standard baselines. RRBLUP often outperforms deep models on linear-dominated genomic data.
4. **Model architecture pattern**: `Config` class at module level → data loading with train/test split → PCA/dimension reduction → PyTorch model (CNN+Transformer hybrid) → training loop with early stopping → result JSON + plots.
5. **Results output**: JSON files in `results/` with R², Pearson correlation, per-fold metrics. Plots saved as PNG.

## Output Directories

- `results/` — Model outputs (JSON metrics, PNG plots, saved model weights)
- `logs/` — Training logs from HPC jobs

## Note

Most root-level Python files are independent training scripts that share similar structure but target different datasets or model variants. They are NOT a unified package — each is self-contained with its own model definitions, config, and training loop. The `cropformer/` package is the exception (properly modularized).

## Failed Approaches (不要重复尝试)

以下方案已经验证无效，不要再浪费时间：

1. **残差学习 (Residual FGN)**: 用 RRBLUP/XGBoost 预测残差，再训练 FGN 拟合残差，最后相加。结果比单独 XGBoost 差（R2 0.149 vs 0.206）。
2. **L1 正则化 on snp_weight**: 在 FGN v7 的 snp_weight 上加 L1 惩罚，效果不显著。
3. **NGBoost (Neural Gradient Boosting)**: TinyNN + boosting，用户已确认此方案不可行。
4. **XGBoost 作为 stacking meta-learner**: 在 4-18 维 meta-feature 上 XGBoost 严重过拟合，Ridge/Lasso 远优于 XGBoost meta-learner。
5. **FGN v5 (复数分离卷积)**: 比 v4 差。
6. **FGN v6 (大核 k=31 时间卷积)**: 比 v4 差。
7. **FGN v9 (DCT 替代 FFT)**: 比 v4 差。
8. **FGN v10 (加性线性路径)**: 比 v7 差。
9. **FGN v11 (双 FFT+DCT)**: 与 v7 相当，但 fold 间不稳定。
10. **StandardScaler**: 去掉比保留好（+0.01），已采纳。不要再加回来。
11. **Mixup**: 对 v4 有益但对 v7 有害，效果不一致。

**当前有效的提升手段**:
- SWA (Stochastic Weight Averaging): 稳定 +0.01-0.02
- hidden=96 for FGN v4/v7: 轻微提升
- 去掉 StandardScaler: +0.01
- Stacking:比XGBoost提升0.02左右
