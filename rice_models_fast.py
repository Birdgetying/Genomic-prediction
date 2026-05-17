"""
Rice Genomic Prediction — Fast Model Comparison
================================================
Fast models: RRBLUP, GBLUP, XGBoost, ElasticNet, GWAS-weighted RRBLUP, Stacking
Novel model: GWAS-Weighted Stacking Ensemble (GWSE)
Evaluation: 5-fold CV on all 10 traits

Strategy: Use fast classical models that train in seconds.
All 10 traits complete in ~15-20 minutes.
"""

import json
import time
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge, ElasticNetCV, LassoCV
from scipy.stats import pearsonr
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"D:\Desktop\论文复现\results\rice_data")
OUTPUT_DIR = Path(r"D:\Desktop\论文复现\results\rice_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS = 5
GWAS_TOP_K = 3000
np.random.seed(RANDOM_SEED)

TRAITS = [
    'Heading_date', 'Plant_height', 'Num_panicles',
    'Num_effective_panicles', 'Yield', 'Grain_weight',
    'Spikelet_length', 'Grain_length', 'Grain_width', 'Grain_thickness'
]


# ============================================================================
# 1. RRBLUP
# ============================================================================
class RRBLUP:
    def __init__(self):
        # alpha range: 0.001~1000, 30 points — better resolution in 0.1~10 region
        self.model = RidgeCV(alphas=np.logspace(-3, 3, 30))

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# ============================================================================
# 2. GBLUP
# ============================================================================
class GBLUP:
    def __init__(self):
        self.model = RidgeCV(alphas=np.logspace(-3, 3, 20))

    def fit(self, G_train, y_train):
        self.model.fit(G_train, y_train)
        return self

    def predict(self, G_test_train):
        return self.model.predict(G_test_train)


# ============================================================================
# 3. XGBoost
# ============================================================================
class XGBoostModel:
    def __init__(self, n_estimators=500, max_depth=6, lr=0.05):
        self.params = {
            'n_estimators': n_estimators, 'max_depth': max_depth,
            'learning_rate': lr, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbosity': 0
        }

    def fit(self, X, y):
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# ============================================================================
# 4. Elastic Net
# ============================================================================
class ElasticNetModel:
    def __init__(self):
        self.model = ElasticNetCV(
            l1_ratio=[.1, .5, .7, .9, .95, 1],
            alphas=np.logspace(-4, 2, 20),
            cv=3, random_state=RANDOM_SEED, max_iter=5000, n_jobs=-1
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# ============================================================================
# 5. GWAS-Weighted RRBLUP (novel variation)
# ============================================================================
class GWASWeightedRRBLUP:
    """
    GWAS-weighted RRBLUP: weights each marker by its GWAS signal strength.
    Instead of hard-threshold feature selection, uses soft weighting:
      X_weighted_j = X_j * w_j
    where w_j = |r(X_j, y)| (absolute Pearson correlation with phenotype).
    This retains all selected markers while emphasizing the most relevant ones.
    """
    def __init__(self, top_k=GWAS_TOP_K):
        self.top_k = top_k
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
# Fast GWAS selection
# ============================================================================
def gwas_select_fast(X, y, top_k):
    """Vectorized GWAS using Pearson correlation."""
    y_c = y - y.mean()
    X_c = X - X.mean(axis=0)
    # Pearson r for all SNPs
    num = np.dot(y_c, X_c)
    denom = np.std(y_c) * len(y) * np.sqrt(np.sum(X_c ** 2, axis=0) + 1e-12)
    scores = np.abs(num / denom)
    return np.argsort(scores)[-top_k:]


# ============================================================================
# Cross-Validation
# ============================================================================
def evaluate_all_models(X_all, y, G_mat, trait_name):
    """5-fold CV for all models on a single trait."""
    n = len(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    model_names = [
        'RRBLUP', 'GBLUP', 'XGBoost', 'ElasticNet',
        'GWAS_RRBLUP', 'Ensemble'
    ]
    results = {m: {'preds': [], 'targets': []} for m in model_names}

    for fi, (tr, te) in enumerate(kf.split(X_all)):
        print(f"    Fold {fi+1}/{N_FOLDS} ...", end=" ", flush=True)

        Xtr, Xte = X_all[tr], X_all[te]
        ytr, yte = y[tr], y[te]

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        fold_preds = {}

        # RRBLUP
        rr = RRBLUP().fit(Xtr_s, ytr)
        fold_preds['RRBLUP'] = rr.predict(Xte_s)

        # GBLUP
        gb = GBLUP().fit(G_mat[tr][:, tr], ytr)
        fold_preds['GBLUP'] = gb.predict(G_mat[te][:, tr])

        # XGBoost
        xm = XGBoostModel(n_estimators=300).fit(Xtr_s, ytr)
        fold_preds['XGBoost'] = xm.predict(Xte_s)

        # ElasticNet
        en = ElasticNetModel().fit(Xtr_s, ytr)
        fold_preds['ElasticNet'] = en.predict(Xte_s)

        # GWAS-Weighted RRBLUP
        gw = GWASWeightedRRBLUP().fit(Xtr_s, ytr)
        fold_preds['GWAS_RRBLUP'] = gw.predict(Xte_s)

        # Stacking: nested 3-fold CV to produce OOF meta-features (no leakage)
        n_tr = len(ytr)
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + fi)
        oof_preds = {m: np.zeros(n_tr) for m in ['RRBLUP', 'GBLUP', 'XGBoost',
                                                   'ElasticNet', 'GWAS_RRBLUP']}
        for itr, ite in inner_kf.split(range(n_tr)):
            X_itr = Xtr_s[itr]; X_ite = Xtr_s[ite]
            y_itr = ytr[itr]
            oof_preds['RRBLUP'][ite] = RRBLUP().fit(X_itr, y_itr).predict(X_ite)
            oof_preds['GBLUP'][ite] = GBLUP().fit(
                G_mat[tr][itr][:, itr], y_itr).predict(G_mat[tr][ite][:, itr])
            oof_preds['XGBoost'][ite] = XGBoostModel(n_estimators=300).fit(
                X_itr, y_itr).predict(X_ite)
            oof_preds['ElasticNet'][ite] = ElasticNetModel().fit(
                X_itr, y_itr).predict(X_ite)
            oof_preds['GWAS_RRBLUP'][ite] = GWASWeightedRRBLUP().fit(
                X_itr, y_itr).predict(X_ite)
        meta = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(
            np.column_stack([oof_preds[m] for m in oof_preds]), ytr)
        base_preds_te = np.column_stack([
            fold_preds['RRBLUP'], fold_preds['GBLUP'],
            fold_preds['XGBoost'], fold_preds['ElasticNet'],
            fold_preds['GWAS_RRBLUP']
        ])
        fold_preds['Ensemble'] = meta.predict(base_preds_te)

        for m in model_names:
            results[m]['preds'].extend(fold_preds[m].tolist())
            results[m]['targets'].extend(yte.tolist())

        r2s = {m: r2_score(yte, fold_preds[m]) for m in model_names}
        best = max(r2s, key=r2s.get)
        print(f"best={best}({r2s[best]:.4f})  "
              f"GWAS_RR={r2s['GWAS_RRBLUP']:.4f}  "
              f"Ens={r2s['Ensemble']:.4f}")

    final = {}
    for m in model_names:
        p = np.array(results[m]['preds'])
        t = np.array(results[m]['targets'])
        final[m] = {
            'R2': float(r2_score(t, p)),
            'Correlation': float(pearsonr(t, p)[0]),
            'RMSE': float(np.sqrt(np.mean((p - t) ** 2)))
        }
    return final


# ============================================================================
# Main
# ============================================================================
def main():
    print(f"{'='*70}")
    print("Rice Genomic Prediction — Fast Model Comparison")
    print(f"{'='*70}")

    # Load data
    print("\nLoading preprocessed data ...")
    data = np.load(DATA_DIR / "genotype_matrix.npz", allow_pickle=True)
    G = data['G']
    G_matrix = data['G_matrix']
    sample_names = list(data['sample_names'])
    with open(DATA_DIR / "trait_data.json") as f:
        trait_data = json.load(f)
    print(f"  Genotype: {G.shape}, G-matrix: {G_matrix.shape}")

    all_results = {}
    t0_trait = time.time()

    for trait in TRAITS:
        print(f"\n{'='*70}")
        print(f"  TRAIT: {trait}")
        print(f"{'='*70}")

        td = trait_data[trait]
        idxs = td['genotype_indices']
        y = np.array(td['values'])
        X_all = G[idxs]
        G_mat = G_matrix[idxs][:, idxs]

        # GWAS selection for models that need it
        k = min(GWAS_TOP_K, X_all.shape[1] - 50)
        gidx = gwas_select_fast(X_all, y, k)
        X_sel = X_all[:, gidx]
        print(f"  {len(y)} samples, {X_all.shape[1]} markers -> {X_sel.shape[1]} GWAS-selected")

        res = evaluate_all_models(X_sel, y, G_mat, trait)
        all_results[trait] = res

        print(f"\n  {'Model':<20s} {'R2':>8s} {'Corr':>8s} {'RMSE':>8s}")
        print(f"  {'-'*44}")
        for m, v in res.items():
            print(f"  {m:<20s} {v['R2']:8.4f} {v['Correlation']:8.4f} {v['RMSE']:8.4f}")

        # Save intermediate
        with open(OUTPUT_DIR / "results_intermediate.json", 'w') as f:
            json.dump({t: {m: {k: float(vv) for k, vv in mv.items()}
                           for m, mv in models.items()}
                       for t, models in all_results.items()}, f, indent=2)

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    model_names = ['RRBLUP', 'GBLUP', 'XGBoost', 'ElasticNet', 'GWAS_RRBLUP', 'Ensemble']
    summ = {m: {'R2': [], 'Corr': []} for m in model_names}

    for t in TRAITS:
        for m in model_names:
            if m in all_results[t]:
                summ[m]['R2'].append(all_results[t][m]['R2'])
                summ[m]['Corr'].append(all_results[t][m]['Correlation'])

    print(f"\n  {'Model':<20s} {'Mean R2':>10s} {'Mean Corr':>10s} {'Best':>10s} {'Worst':>10s}")
    print(f"  {'-'*60}")
    for m in model_names:
        rs = summ[m]['R2']
        cs = summ[m]['Corr']
        if rs:
            print(f"  {m:<20s} {np.mean(rs):10.4f} {np.mean(cs):10.4f} "
                  f"{np.max(rs):10.4f} {np.min(rs):10.4f}")

    ranked = sorted([(m, np.mean(summ[m]['R2'])) for m in model_names],
                    key=lambda x: x[1], reverse=True)
    print(f"\n  Model Ranking (by Mean R2):")
    for i, (m, r) in enumerate(ranked, 1):
        marker = " <-- NOVEL" if m in ('GWAS_RRBLUP', 'Ensemble') else ""
        print(f"    {i}. {m}: {r:.4f}{marker}")

    # ========================================================================
    # Save & Plot
    # ========================================================================
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(OUTPUT_DIR / f"final_results_{ts}.json", 'w') as f:
        json.dump({t: {m: {k: float(v) for k, v in mv.items()}
                       for m, mv in models.items()}
                   for t, models in all_results.items()}, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4']
    x = np.arange(len(TRAITS))
    w = 0.12
    for i, m in enumerate(model_names):
        rs = [all_results[t][m]['R2'] for t in TRAITS]
        ax1.bar(x + i * w - 2.5 * w, rs, w, label=m, color=colors[i], alpha=0.85)
    ax1.set_ylabel('R2'); ax1.set_title('Rice Genomic Prediction — Model Comparison')
    ax1.set_xticks(x); ax1.set_xticklabels([t[:12] for t in TRAITS], rotation=45, ha='right')
    ax1.legend(ncol=3, fontsize=7); ax1.axhline(0, c='k', lw=0.5); ax1.grid(axis='y', alpha=0.3)

    names_sorted = [m for m, _ in ranked]
    bars = ax2.barh(names_sorted, [np.mean(summ[m]['R2']) for m in names_sorted],
                     color=colors[:len(names_sorted)], alpha=0.85)
    ax2.set_xlabel('Mean R2'); ax2.set_title('Overall Model Ranking')
    for b, v in zip(bars, [np.mean(summ[m]['R2']) for m in names_sorted]):
        ax2.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2,
                 f'{v:.4f}', va='center', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"comparison_{ts}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Total time: {(time.time()-t0_trait)/60:.1f} min")


if __name__ == '__main__':
    main()
