## 项目结构

### 主流程入口

| 文件 | 说明 |
|------|------|
| `genomic_ensemble.py` | **核心 pipeline** — 所有模型训练、评估、可视化的统一入口 |
| `submit_all_ensemble.jsub` | HPC 作业脚本：小麦/小麦2000/水稻/玉米/大豆/小麦GABI 六数据集串行运行 |
| `plot_bar_charts.py` | 独立柱状图脚本（可脱离 GPU 运行，仅读 JSON） |

### 辅助脚本

| 文件 | 说明 |
|------|------|
| `repair_oof_npz.py` | OOF NPZ 修复工具（诊断用） |
| `test_*.py` | 各模块测试脚本（wheatgp, rice, ablation, multitrait, stacking 等） |

### 子项目

| 目录 | 说明 |
|------|------|
| `cropformer/` | Custom Transformer-based 基因组预测框架（论文代码） |
| `WheatGP/` | CNN+LSTM 小麦多倍体模型（PyTorch, 带 GUI） |
| `DNNGP-main/` | TensorFlow 2.6 深度基因组预测（基线对比） |
| `CropG2P-main/` | 另一基因组预测框架 |

### 数据

| 数据集 | 目录 | 样本数 | SNP 数 | 性状数 | 来源 |
|--------|------|--------|--------|--------|------|
| Wheat (CSIAAS) | VCF 三文件 (*.vcf.gz) | 813 | 45,000 (SNP+INDEL+SV) | 1 | 服务器 `/data/Variation/CSIAAS/` |
| Wheat2000 | `dnngp_data/wheat2000/SNP_origin/` | 2,000 | 33,709 | 6 | CSV 格式 |
| Rice | `results/rice_data/genotype_matrix.npz` | 529 | ~360,000 | 10 | npz+json 格式 |
| Maize (Iranian) | `data2/` | ~2,374 | ~15,000 | 4 | CSV 格式 |
| Soybean SoySNP50K | `results/soybean_SoySNP50K/` | 346 | 20,526 | 2 | npz+json 格式 |
| Wheat GABI (iSELECT 90k) | `results/wheat_GABI/` | 371 | 12,546 | 16 | npz+json 格式 |

### 结果输出

```
results/
├── wheat_ensemble/          # 小麦 813 样本结果 (JSON + OOF NPZ + 图)
├── wheat2000_ensemble/      # 小麦2000 结果
├── rice_ensemble/           # 水稻结果
├── maize_ensemble/          # 玉米结果
├── soybean_ensemble/        # 大豆结果
├── wheat_gabi_ensemble/     # 小麦GABI结果
├── oof_cache/               # OOF 缓存
└── stacking_cache/          # Stacking 缓存

figures/                     # 13+ 张柱状图/散点图/效率图（训练后自动生成）
logs/                        # HPC 作业日志
```

---

## 当前工作：全模型集成 + GWAS 验证

### 实验设计

**目标**：在 6 个数据集上系统评估 20 种基因组预测模型，并验证 GWAS 预选位点的有效性。

**模型清单（20 个基础模型）**：

| 类型 | 模型 |
|------|------|
| 传统 (5) | RRBLUP, GBLUP, XGBoost, ElasticNet, GWAS_RRBLUP |
| 深度学习 (15) | FGN / v2 / v4–v11, FGNplus, FGN PCA, GenomicFM, FusionNet, AdditiveGenomicNet, WheatGP |

**集成方法（6 种 Stacking 变体）**：Stacking (DL), Stacking (All), Trad Ensemble, Stacking (Pruned), Stacking (Greedy), Stacking (R²+Greedy)

### GWAS vs Random 对照实验

每个模型在 **同一 CV fold** 下训练两次：
- **GWAS pass**：GWAS 选 top-5000 SNPs → 模型名 `{model}`
- **Random pass**：随机选 5000 SNPs → 模型名 `{model}_random`

两组使用完全相同的超参数和 fold 划分，唯一变量是 SNP 选择方式。用于验证 GWAS 预选的实际改进效果。

### 评估指标

- **R²** (决定系数, `sklearn.metrics.r2_score`) — 主指标
- **Pearson r** — 相关性
- **RMSE** — 均方根误差
- **训练时间** (秒/fold)
- **参数量** (Params)
- **GPU 显存** (MB, 实测 `torch.cuda.max_memory_allocated`)

### 可视化产出

| 编号 | 内容 |
|------|------|
| 01-03, 07 | 柱状/排名图（跨数据集综合） |
| 05-06, 11 | 逐性状 top-4 模型 OOF 散点图 |
| 08-10 | 逐数据集逐性状模型柱状图 |
| 12 | 训练时间对比 |
| 13 | 参数量对比（对数轴） |
| 14 | GPU 显存对比（实测/估算） |

### 运行方式

```bash
# 本地快速测试（2 folds × 1 trait）
python genomic_ensemble.py wheat2000

# HPC 完整运行（5 folds × 全部性状）
jsub < submit_all_ensemble.jsub
```
