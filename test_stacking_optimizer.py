#!/usr/bin/env python3
"""Stacking optimizer — rapid local testing of improved stacking strategies.

Uses genomic_ensemble.py for data loading & model training. Caches OOF predictions
to disk so subsequent runs can test new stacking algorithms without retraining.

Key improvements over the default stacking_evaluate():
  1. Correlation-based pruning — remove redundant models (r > 0.995)
  2. Greedy forward selection — auto-find optimal model subset via nested CV
  3. Multiple meta-learners — BayesianRidge, LassoCV (fine grid), ElasticNetCV
  4. Meta feature engineering — pairwise interactions, squared terms
  5. Convex weight optimization — scipy.optimize with simplex constraints
  6. Model clustering — cluster by prediction correlation, pick best per cluster
"""
import sys, os, time, json, random, pickle, warnings
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, BayesianRidge, HuberRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import xgboost as xgb

random.seed(42)
np.random.seed(42)

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import genomic_ensemble as ge

warnings.filterwarnings('ignore')

# ============================================================================
# Config
# ============================================================================
CACHE_DIR = "results/stacking_cache"
N_FOLDS = 3  # fast inner CV for stacking eval
RANDOM_SEED = 42
RIDGE_ALPHAS_FINE = np.logspace(-3, 5, 50)  # 50-point log grid
LASSO_ALPHAS = np.logspace(-4, 2, 30)
ENET_ALPHAS = np.logspace(-4, 2, 20)

os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================================
# Data loading & OOF caching
# ============================================================================

def load_rice_trait(trait_name):
    data = np.load("results/rice_data/genotype_matrix.npz", allow_pickle=True)
    G = data['G']
    with open("results/rice_data/trait_data.json") as f:
        trait_info = json.load(f)
    td = trait_info[trait_name]
    idxs = td['genotype_indices']
    y = np.array(td['values']).astype(np.float32)
    X_t = G[idxs]
    mask = ~np.isnan(y)
    return X_t[mask], y[mask]


def get_cache_path(trait_name):
    return os.path.join(CACHE_DIR, f"oof_{trait_name}.pkl")


def train_all_models_cached(trait_name, force_retrain=False):
    """Train all models with 3-fold CV, cache OOF predictions. Returns (oof_dict, y_all, metrics)."""
    cache_path = get_cache_path(trait_name)
    if not force_retrain and os.path.exists(cache_path):
        print(f"  [CACHE HIT] Loading OOF from {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data['oof'], data['y'], data['metrics']

    print(f"\n  [TRAINING] Full model training for {trait_name}...")
    X_all, y_all = load_rice_trait(trait_name)
    n_snps = min(ge.GWAS_TOP_K, max(50, X_all.shape[1] - 50))
    n_samples = len(y_all)

    trad_models = ge.TRAD_NAMES
    dl_models = ge.DL_NAMES
    all_models = trad_models + dl_models

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=ge.RANDOM_SEED)
    oof = {m: np.zeros(n_samples) for m in all_models}
    fold_r2 = {m: [] for m in all_models}

    for fi, (tr, te) in enumerate(kf.split(X_all)):
        print(f"\n  --- Fold {fi+1}/{N_FOLDS} ---")
        Xtr_raw, Xte_raw = X_all[tr], X_all[te]
        ytr, yte = y_all[tr], y_all[te]

        maf_idx = ge.maf_filter(Xtr_raw)
        if len(maf_idx) >= n_snps:
            Xtr_raw, Xte_raw = Xtr_raw[:, maf_idx], Xte_raw[:, maf_idx]

        gidx = ge.gwas_select(Xtr_raw, ytr, n_snps)
        Xtr = Xtr_raw[:, gidx]; Xte = Xte_raw[:, gidx]
        Xtr_s = Xtr.astype(np.float32); Xte_s = Xte.astype(np.float32)

        # Traditional models
        G_train = Xtr_s @ Xtr_s.T / n_snps
        G_te_tr = Xte_s @ Xtr_s.T / n_snps
        trad_configs = ge._make_trad_configs(G_train, G_te_tr, n_snps, len(tr))
        for tname, build_fn, fit_fn, pred_fn, param_count in trad_configs:
            tmodel = build_fn()
            fit_fn(tmodel, Xtr_s, ytr)
            preds = pred_fn(tmodel, Xte_s)
            oof[tname][te] = preds
            fold_r2[tname].append(float(r2_score(yte, preds)))

        # DL models
        for mname in dl_models:
            try:
                overrides = {'hidden': 96} if mname in ('FGN v4', 'FGN v7') else {}
                model = ge.create_model(mname, n_snps, overrides=overrides)
                bs = 32 if mname.startswith('FGN') or mname in ('GenomicFM', 'FusionNet') else 64
                model = ge.train_torch_model(model, Xtr_s, ytr, epochs=300,
                                              batch_size=bs, lr=2e-3, weight_decay=1e-3,
                                              patience=35, use_swa=True)
                preds = ge.predict_torch_model(model, Xte_s)
                oof[mname][te] = preds
                fold_r2[mname].append(float(r2_score(yte, preds)))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"    {mname:<22s} FAILED: {e}")
                oof[mname][te] = np.mean(ytr)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute per-model metrics
    metrics = {}
    for mname in all_models:
        r2_v = float(r2_score(y_all, oof[mname]))
        corr_v = float(pearsonr(y_all, oof[mname])[0])
        metrics[mname] = {'R2': r2_v, 'r': corr_v}

    # Cache
    with open(cache_path, 'wb') as f:
        pickle.dump({'oof': oof, 'y': y_all, 'metrics': metrics}, f)
    print(f"  [CACHED] OOF saved to {cache_path}")

    return oof, y_all, metrics


# ============================================================================
# Phase 1: Correlation-based Pruning
# ============================================================================

def prune_correlated_models(oof, metrics, corr_threshold=0.995):
    """Remove redundant models: if two models have prediction correlation > threshold,
    keep only the one with higher R²."""
    model_names = list(oof.keys())
    # Build prediction matrix
    preds = np.column_stack([oof[m] for m in model_names])

    # Correlation matrix
    corr_mat = np.corrcoef(preds.T)
    np.fill_diagonal(corr_mat, 0)

    # Sort models by R² descending
    ranked = sorted(model_names, key=lambda m: metrics[m]['R2'], reverse=True)
    kept = []
    removed = set()

    for m in ranked:
        if m in removed:
            continue
        kept.append(m)
        m_idx = model_names.index(m)
        # Remove all models too correlated with this one
        for j, other in enumerate(model_names):
            if other != m and other not in removed and other not in kept:
                if abs(corr_mat[m_idx, j]) > corr_threshold:
                    removed.add(other)

    return kept, list(removed)


# ============================================================================
# Phase 2: Auto-select model subset via greedy forward selection (nested CV)
# ============================================================================

def build_X_stack(model_list, oof):
    return np.column_stack([oof[m] for m in model_list])


def greedy_forward_select(oof, y, metrics, model_names, meta_type='Ridge',
                          min_gain=0.0005, max_models=12):
    """Greedily add models that improve nested CV R².

    Args:
        oof: dict of model_name -> OOF prediction array
        y: target values
        metrics: dict of model_name -> {'R2': ..., 'r': ...}
        model_names: pool of candidate model names
        meta_type: 'Ridge' | 'Lasso' | 'ElasticNet'
        min_gain: minimum R² improvement to add a model
        max_models: maximum number of models to include

    Returns:
        selected: list of selected model names
        history: list of (model_added, R2_after, n_models)
    """
    # Sort candidates by individual R²
    remaining = sorted(model_names, key=lambda m: metrics[m]['R2'], reverse=True)

    selected = [remaining[0]]  # start with best
    remaining = remaining[1:]

    inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)

    def eval_subset(models):
        X = build_X_stack(models, oof)
        preds = np.zeros(len(y))
        for itr, ite in inner_kf.split(X):
            X_tr, X_te = X[itr], X[ite]
            if meta_type == 'Ridge':
                m = RidgeCV(alphas=RIDGE_ALPHAS_FINE, fit_intercept=True, cv=3)
            elif meta_type == 'Lasso':
                m = LassoCV(alphas=LASSO_ALPHAS, cv=3, max_iter=5000, random_state=42)
            elif meta_type == 'ElasticNet':
                m = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                                 alphas=ENET_ALPHAS, cv=3, max_iter=5000, random_state=42)
            m.fit(X_tr, y[itr])
            preds[ite] = m.predict(X_te)
        return float(r2_score(y, preds))

    best_r2 = eval_subset(selected)
    history = [('START', best_r2, 1)]

    while remaining and len(selected) < max_models:
        gains = []
        for cand in remaining[:min(10, len(remaining))]:  # test top 10 candidates
            trial_set = selected + [cand]
            trial_r2 = eval_subset(trial_set)
            gains.append((cand, trial_r2 - best_r2, trial_r2))

        gains.sort(key=lambda x: -x[2])  # sort by trial R2 descending
        best_cand, best_gain, best_trial_r2 = gains[0]

        if best_gain < min_gain:
            break

        selected.append(best_cand)
        remaining.remove(best_cand)
        best_r2 = best_trial_r2
        history.append((best_cand, best_r2, len(selected)))
        print(f"    + {best_cand:<22s} R2={best_r2:+.4f} (gain={best_gain:+.4f})")

    return selected, history


# ============================================================================
# Phase 3: Advanced Meta-learners
# ============================================================================

def eval_meta_learner(model_list, oof, y, meta_type, ml_params=None):
    """Evaluate a meta-learner on given model subset with nested 3-fold CV."""
    X = build_X_stack(model_list, oof)
    n_models = X.shape[1]
    if n_models < 1:
        return None

    inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
    preds = np.zeros(len(y))

    for itr, ite in inner_kf.split(X):
        X_tr, X_te = X[itr], X[ite]

        if meta_type == 'RidgeCV':
            m = RidgeCV(alphas=RIDGE_ALPHAS_FINE, fit_intercept=True, cv=5)
            m.fit(X_tr, y[itr])
            preds[ite] = m.predict(X_te)
            weights = m.coef_.copy()

        elif meta_type == 'LassoCV':
            m = LassoCV(alphas=LASSO_ALPHAS, cv=5, max_iter=10000, random_state=42)
            m.fit(X_tr, y[itr])
            preds[ite] = m.predict(X_te)
            weights = m.coef_.copy()

        elif meta_type == 'ElasticNetCV':
            m = ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, 1],
                             alphas=ENET_ALPHAS, cv=5, max_iter=10000, random_state=42)
            m.fit(X_tr, y[itr])
            preds[ite] = m.predict(X_te)
            weights = m.coef_.copy()

        elif meta_type == 'BayesianRidge':
            m = BayesianRidge(max_iter=500, tol=1e-5)
            m.fit(X_tr, y[itr])
            preds[ite] = m.predict(X_te)
            weights = m.coef_.copy()

        elif meta_type == 'Huber':
            m = HuberRegressor(max_iter=500, alpha=0.001)
            m.fit(X_tr, y[itr])
            preds[ite] = m.predict(X_te)
            weights = m.coef_.copy()

        elif meta_type == 'OptimizedWeights':
            # Convex optimization: minimize MSE with simplex constraints
            n = X_tr.shape[1]
            w0 = np.ones(n) / n

            def loss(w):
                pred = X_tr @ w
                return np.mean((y[itr] - pred) ** 2)

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = [(0, 1) for _ in range(n)]
            res = minimize(loss, w0, method='SLSQP', constraints=constraints,
                          bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
            w_opt = res.x
            w_opt = w_opt / w_opt.sum()  # ensure sum=1
            preds[ite] = X_te @ w_opt
            weights = w_opt.copy()

        elif meta_type == 'XGBoost':
            # XGBoost with strong regularization
            m = xgb.XGBRegressor(
                n_estimators=100, max_depth=2, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5,
                reg_lambda=2.0, min_child_weight=10, random_state=42, verbosity=0
            )
            m.fit(X_tr, y[itr])
            preds[ite] = m.predict(X_te)
            weights = m.feature_importances_.copy()

        elif meta_type == 'SimpleAvg':
            preds[ite] = np.mean(X_te, axis=1)
            weights = np.ones(n_models) / n_models

        elif meta_type == 'WeightedAvg':
            w = np.array([metrics_reference.get(m, 0.01) for m in model_list])
            w = np.maximum(w, 1e-6)
            w = w / w.sum()
            preds[ite] = X_te @ w
            weights = w.copy()

        elif meta_type == 'TopK_Avg':
            # Average top K=ceil(n/2) models by position in model_list (assumes sorted)
            k = max(2, (n_models + 1) // 2)
            preds[ite] = np.mean(X_te[:, :k], axis=1)
            w = np.zeros(n_models)
            w[:k] = 1.0 / k
            weights = w

        else:
            raise ValueError(f"Unknown meta_type: {meta_type}")

    r2_v = float(r2_score(y, preds))
    corr_v = float(pearsonr(y, preds)[0])
    n_nonzero = int(np.sum(np.abs(weights) > 1e-6))
    return {'R2': r2_v, 'Correlation': corr_v, 'n_active': n_nonzero,
            'weights': weights.tolist(), 'models': model_list, 'meta': meta_type}


# Global reference for WeightedAvg
metrics_reference = {}


# ============================================================================
# Phase 4: Meta Feature Engineering
# ============================================================================

def add_meta_features(X_meta, y, n_top=6):
    """Add pairwise interactions and squared terms for top-n models."""
    n_samples, n_base = X_meta.shape
    features = [X_meta]

    # Top models (most impactful) interactions
    k = min(n_top, n_base)
    for i in range(k):
        for j in range(i + 1, k):
            features.append((X_meta[:, i] * X_meta[:, j]).reshape(-1, 1))

    # Squared terms for top models
    for i in range(k):
        features.append((X_meta[:, i] ** 2).reshape(-1, 1))

    return np.hstack(features)


def eval_stacking_with_features(model_list, oof, y, meta_type='Ridge'):
    """Evaluate stacking with meta feature engineering."""
    X_base = build_X_stack(model_list, oof)
    n_base = X_base.shape[1]
    X_meta = add_meta_features(X_base, y, n_top=min(6, n_base))
    n_extra = X_meta.shape[1] - n_base

    inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
    preds = np.zeros(len(y))

    for itr, ite in inner_kf.split(X_meta):
        X_tr, X_te = X_meta[itr], X_meta[ite]
        m = RidgeCV(alphas=RIDGE_ALPHAS_FINE, fit_intercept=True, cv=5)
        m.fit(X_tr, y[itr])
        preds[ite] = m.predict(X_te)

    r2_v = float(r2_score(y, preds))
    corr_v = float(pearsonr(y, preds)[0])
    return {'R2': r2_v, 'Correlation': corr_v, 'n_features': n_extra, 'meta': f'Ridge+Poly(n={n_extra})'}


# ============================================================================
# Phase 5: Model Clustering
# ============================================================================

def cluster_models_by_prediction(oof, metrics, n_clusters='auto'):
    """Cluster models by prediction correlation, return best per cluster using
    simple correlation threshold-based clustering."""
    model_names = list(oof.keys())
    preds_mat = np.column_stack([oof[m] for m in model_names])
    corr = np.corrcoef(preds_mat.T)

    # Sort by R²
    ranked = sorted(model_names, key=lambda m: metrics[m]['R2'], reverse=True)
    clusters = []
    assigned = set()

    for m in ranked:
        if m in assigned:
            continue
        m_idx = model_names.index(m)
        cluster = [m]
        assigned.add(m)
        for j, other in enumerate(model_names):
            if other not in assigned:
                if abs(corr[m_idx, j]) > 0.97:  # high correlation = same cluster
                    cluster.append(other)
                    assigned.add(other)
        clusters.append(cluster)

    # Pick best from each cluster
    cluster_best = [c[0] for c in clusters]  # ranked already, so c[0] is best
    return clusters, cluster_best


# ============================================================================
# Main optimization driver
# ============================================================================

def run_optimization(trait_name="Plant_height", force_retrain=False):
    global metrics_reference

    print(f"\n{'='*70}")
    print(f"  Stacking Optimizer — {trait_name}")
    print(f"{'='*70}")

    oof, y, metrics = train_all_models_cached(trait_name, force_retrain=force_retrain)
    metrics_reference = {m: metrics[m]['R2'] for m in metrics}

    # Baseline: best single model
    best_single = max(metrics.items(), key=lambda x: x[1]['R2'])
    print(f"\n  Best single: {best_single[0]} (R²={best_single[1]['R2']:+.4f})")

    all_models = list(oof.keys())

    # ----------------------------------------------------------
    # Step 1: Correlation pruning
    # ----------------------------------------------------------
    print(f"\n  [Step 1] Correlation pruning (r > 0.995)...")
    pruned, removed = prune_correlated_models(oof, metrics, corr_threshold=0.995)
    print(f"    Pruned: {len(all_models)} -> {len(pruned)} models")
    if removed:
        print(f"    Removed: {removed}")

    # ----------------------------------------------------------
    # Step 2: Greedy forward selection (3 meta-learners)
    # ----------------------------------------------------------
    print(f"\n  [Step 2] Greedy forward selection...")
    all_results = {}

    for meta in ['Ridge', 'Lasso', 'ElasticNet']:
        print(f"\n    --- {meta} meta-learner ---")
        selected, history = greedy_forward_select(oof, y, metrics, pruned,
                                                   meta_type=meta, min_gain=0.0002, max_models=10)
        label = f"GreedySelect ({meta})"
        result = eval_meta_learner(selected, oof, y, meta_type=f"{meta}CV")
        if result:
            result['label'] = label
            all_results[label] = result
            print(f"    => R²={result['R2']:+.4f}, {len(selected)} models: {selected}")

    # ----------------------------------------------------------
    # Step 3: Clustering approach
    # ----------------------------------------------------------
    print(f"\n  [Step 3] Model clustering...")
    clusters, cluster_best = cluster_models_by_prediction(oof, metrics)
    print(f"    {len(clusters)} clusters: {[(c[0], len(c)) for c in clusters]}")

    # Test cluster-based selection
    for meta in ['RidgeCV', 'LassoCV', 'ElasticNetCV', 'BayesianRidge', 'OptimizedWeights']:
        label = f"ClusterBest ({meta})"
        result = eval_meta_learner(cluster_best, oof, y, meta)
        if result:
            result['label'] = label
            all_results[label] = result
            print(f"    {label}: R²={result['R2']:+.4f}")

    # ----------------------------------------------------------
    # Step 4: Meta feature engineering on selected models
    # ----------------------------------------------------------
    print(f"\n  [Step 4] Meta feature engineering...")
    # Use best model subset from greedy selection
    best_so_far = max(all_results.items(), key=lambda x: x[1]['R2'])
    best_models = best_so_far[1]['models']
    print(f"    Base subset: {best_so_far[0]} ({len(best_models)} models)")

    poly_result = eval_stacking_with_features(best_models, oof, y, 'Ridge')
    if poly_result:
        all_results['BestModels + PolyFeat'] = poly_result
        print(f"    Ridge+Poly: R²={poly_result['R2']:+.4f} (+{poly_result['n_features']} features)")

    # Also test poly on cluster best
    poly_cluster = eval_stacking_with_features(cluster_best[:8], oof, y, 'Ridge')
    if poly_cluster:
        all_results['ClusterBest + PolyFeat'] = poly_cluster
        print(f"    Cluster+Poly: R²={poly_cluster['R2']:+.4f}")

    # ----------------------------------------------------------
    # Step 5: Compare all strategies vs baselines
    # ----------------------------------------------------------
    print(f"\n  [Step 5] Comprehensive comparison...")

    # Baseline stacking variants (replicating genomic_ensemble.py defaults)
    trad_names = ge.TRAD_NAMES
    dl_names = [m for m in all_models if m not in trad_names]
    dl_base = [m for m in dl_names if m not in ('FusionNet', 'AdditiveGenomicNet')]

    # A: Stacking (DL) — current pipeline default (Ridge, no pruning)
    oof_dl_base = {m: oof[m] for m in dl_base if m in oof}
    sr_dl = ge.stacking_evaluate(oof_dl_base, y, n_folds=3, meta_type='Ridge', prune_corr=False)
    all_results['Current: Stacking (DL)'] = {
        'R2': sr_dl['R2'], 'Correlation': sr_dl['Correlation'],
        'n_active': len(sr_dl.get('Base_models', [])), 'meta': 'RidgeCV (current)',
        'models': sr_dl.get('Base_models', []), 'label': 'Current: Stacking (DL)'
    }
    print(f"    Current Stacking (DL):  R²={sr_dl['R2']:+.4f}")

    # B: Stacking (All) — current pipeline default (Ridge, no pruning)
    oof_all = {m: oof[m] for m in all_models}
    sr_all = ge.stacking_evaluate(oof_all, y, n_folds=3, meta_type='Ridge', prune_corr=False)
    all_results['Current: Stacking (All)'] = {
        'R2': sr_all['R2'], 'Correlation': sr_all['Correlation'],
        'n_active': len(sr_all.get('Base_models', [])), 'meta': 'RidgeCV (current)',
        'models': sr_all.get('Base_models', []), 'label': 'Current: Stacking (All)'
    }
    print(f"    Current Stacking (All): R²={sr_all['R2']:+.4f}")

    # C: Trad Ensemble — current pipeline default (Ridge, no pruning)
    oof_trad = {m: oof[m] for m in trad_names if m in oof}
    sr_trad = ge.stacking_evaluate(oof_trad, y, n_folds=3, meta_type='Ridge', prune_corr=False)
    all_results['Current: Trad Ensemble'] = {
        'R2': sr_trad['R2'], 'Correlation': sr_trad['Correlation'],
        'n_active': len(sr_trad.get('Base_models', [])), 'meta': 'RidgeCV (current)',
        'models': sr_trad.get('Base_models', []), 'label': 'Current: Trad Ensemble'
    }
    print(f"    Current Trad Ensemble:  R²={sr_trad['R2']:+.4f}")

    # D: Best single model
    all_results['Best Single Model'] = {
        'R2': best_single[1]['R2'], 'Correlation': best_single[1]['r'],
        'n_active': 1, 'meta': 'Single', 'models': [best_single[0]],
        'label': 'Best Single Model'
    }

    # ----------------------------------------------------------
    # Summary & Ranking
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  FINAL RANKING — {trait_name}")
    print(f"  {'Rank':<5s} {'Strategy':<40s} {'R²':>8s} {'Δ vs Best Single':>14s} {'n_models':>10s}")
    print(f"  {'-'*70}")
    sorted_results = sorted(all_results.items(), key=lambda x: -x[1]['R2'])
    for i, (name, res) in enumerate(sorted_results):
        delta = res['R2'] - best_single[1]['R2']
        n_mod = len(res.get('models', []))
        marker = " ***" if i == 0 else ""
        print(f"  {i+1:<5d} {name:<40s} {res['R2']:+.4f}  {delta:+.4f}        {n_mod:>5d}{marker}")

    # Highlight improvement over current Stacking (All)
    current_best_r2 = sr_all['R2']
    new_best_name, new_best = sorted_results[0]
    improvement = new_best['R2'] - current_best_r2
    print(f"\n  Improvement over current Stacking (All): {improvement:+.4f}")
    print(f"  Improvement over best single model:    {new_best['R2'] - best_single[1]['R2']:+.4f}")

    # Save
    os.makedirs("results", exist_ok=True)
    output = {
        'trait': trait_name,
        'best_single': {'name': best_single[0], 'R2': float(best_single[1]['R2'])},
        'current_stacking_all_r2': float(current_best_r2),
        'improvement': float(improvement),
        'results': {k: {kk: vv for kk, vv in v.items() if kk not in ('weights', 'models')}
                     for k, v in all_results.items()},
        'ranking': [(name, float(res['R2'])) for name, res in sorted_results],
        'best_strategy': {
            'name': new_best_name,
            'R2': float(new_best['R2']),
            'models': new_best.get('models', []),
            'meta_type': new_best.get('meta', 'unknown'),
        },
    }
    with open(f"results/stacking_opt_{trait_name}.json", 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to results/stacking_opt_{trait_name}.json")

    return all_results, sorted_results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stacking optimizer')
    parser.add_argument('--trait', type=str, default='Plant_height',
                        help='Trait name (default: Plant_height)')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain all models (ignore cache)')
    parser.add_argument('--traits', nargs='+', type=str, default=None,
                        help='Multiple traits to test')
    args = parser.parse_args()

    if args.traits:
        for t in args.traits:
            run_optimization(t, force_retrain=args.retrain)
    else:
        run_optimization(args.trait, force_retrain=args.retrain)
