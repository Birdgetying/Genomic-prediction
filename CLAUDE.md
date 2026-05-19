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

## HPC 代码上传

修改代码后, 使用 `upload_to_hpc.sh` 一键上传到景行超算平台:

```
bash upload_to_hpc.sh              # 上传所有 git 追踪文件
bash upload_to_hpc.sh --dry-run    # 仅预览, 不实际传输
```

- 上传列表自动从 `git ls-files` 获取 (即 `.gitignore` 白名单)
- 自动排除 `.gitignore`、`CLAUDE.md`、`upload_to_hpc.sh` 三个本地专用文件
- 首次使用需修改脚本顶部 `SERVER` 变量为景行平台实际地址
- **新建文件后必须**:
  1. 在 `.gitignore` 添加 `!新文件名` 白名单条目
  2. `git add` + `git commit` 使其被 git 追踪
  3. 然后 `bash upload_to_hpc.sh` 即可自动包含新文件

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
