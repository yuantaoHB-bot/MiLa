#!/usr/bin/env python3
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""Simple ML baselines (XGBoost, LightGBM, Logistic Regression, MLP)
using the exact same features and fold splits as M5.

Addresses reviewer concern: is deep learning necessary?

Feature sets tested:
  - Full: WSI(768) + Clinical(30) + LabGrid(2×12=24) = 822 features
  - NoWSI: Clinical(30) + LabGrid(24) = 54 features
  - ClinicalOnly: Clinical(30)
"""
import sys
sys.path.insert(0, f"{ROOT}/src")

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

BASE = Path(f"{ROOT}")
DATA_DIR = BASE / "dataset"
EMB_DIR = BASE / "outputs/embeddings"
LAB_DIR = DATA_DIR / "follow_up_labs_clean"
OUT_DIR = BASE / "workspace/supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
CLINICAL_FEATURES = [
    'age', 'gender', 'BMI', 'height', 'weight',
    'ALT', 'AST', 'ALP', 'GGT', 'TBIL', 'DBIL', 'Albumin', 'INR',
    'Creatinine', 'Uric_Acid', 'FBG', 'TG', 'TC', 'HDL', 'LDL', 'HbA1c', 'Platelet',
    'Steatosis_Grade', 'Lobular_Inflammation', 'Ballooning', 'NAS_Total',
    'SAF_S', 'SAF_A',
    'Impaired_glucose_tolerance', 'L_MAFLD'
]
LAB_COLS = ['ALT', 'AST', 'GGT', 'TBIL', 'Albumin', 'Platelet', 'FBG', 'TG', 'TC', 'Creatinine', 'INR']
ANNUAL_GRID_YEARS = 2


def load_wsi_embedding(patient_id, split):
    pid_str = str(patient_id).replace('_', '-')
    he_path = EMB_DIR / f"{split}_HE" / f"{pid_str}-HE_embed.pt"
    sr_path = EMB_DIR / f"{split}_SR" / f"{pid_str}-SR_embed.pt"
    he = torch.load(he_path, map_location='cpu', weights_only=True).squeeze() if he_path.exists() else torch.zeros(768)
    sr = torch.load(sr_path, map_location='cpu', weights_only=True).squeeze() if sr_path.exists() else torch.zeros(768)
    return torch.stack([he, sr]).mean(dim=0).numpy()


def build_annual_grid(patient_id, liver_progression, time_to_event_years):
    lab_file = LAB_DIR / f"{patient_id}_follow_up_labs.csv"
    grid = np.zeros((ANNUAL_GRID_YEARS, len(LAB_COLS) + 1), dtype=np.float32)
    if not lab_file.exists():
        return grid.flatten()
    df = pd.read_csv(lab_file).sort_values('days_from_biopsy')
    if liver_progression == 1:
        event_days = time_to_event_years * 365
        df = df[df['days_from_biopsy'] < event_days]
    last_values = None
    for year in range(ANNUAL_GRID_YEARS):
        end_day = (year + 1) * 365
        year_df = df[df['days_from_biopsy'] <= end_day]
        if len(year_df):
            observed = year_df[year_df['days_from_biopsy'] > year * 365]
            last_values = year_df.iloc[-1][LAB_COLS].values.astype(np.float32)
            grid[year, :-1] = last_values
            grid[year, -1] = 1.0 if len(observed) else 0.0
        elif last_values is not None:
            grid[year, :-1] = last_values
    return grid.flatten()


def load_split_data(split):
    clin = pd.read_csv(DATA_DIR / split / "clinical_data_clean.csv")
    labels = pd.read_csv(DATA_DIR / split / "patient_labels.csv")
    merged = clin.merge(labels, on='patient_id', how='inner')

    n = len(merged)
    X_clin = merged[CLINICAL_FEATURES].values.astype(np.float32)
    X_clin = np.nan_to_num(X_clin, nan=0.0)

    X_wsi = np.zeros((n, 768), dtype=np.float32)
    X_lab = np.zeros((n, ANNUAL_GRID_YEARS * (len(LAB_COLS) + 1)), dtype=np.float32)

    for i, row in merged.iterrows():
        idx = merged.index.get_loc(i)
        pid = str(row['patient_id'])
        X_wsi[idx] = load_wsi_embedding(pid, split)
        X_lab[idx] = build_annual_grid(
            pid,
            row['liver_progression_occurred'],
            row['time_to_event_years']
        )

    label_5y = ((merged['liver_progression_occurred'] == 1) &
                (merged['time_to_event_years'] <= 5)).astype(int).values
    label_10y = merged['liver_progression_occurred'].astype(int).values

    return {
        'X_clin': X_clin, 'X_wsi': X_wsi, 'X_lab': X_lab,
        'label_5y': label_5y, 'label_10y': label_10y,
        'patient_ids': merged['patient_id'].values,
        'merged': merged,
    }


def build_stratification_labels(data):
    label_10y = data['liver_progression_occurred'].astype(int).values
    label_5y = (
        (data['liver_progression_occurred'].astype(int) == 1) &
        (data['time_to_event_years'].astype(float) <= 5)
    ).astype(int).values
    return label_10y + 2 * label_5y


def bootstrap_auroc(y, s, n=2000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def get_feature_sets(X_wsi, X_clin, X_lab):
    return {
        'Full (WSI+Clinical+Labs)': np.hstack([X_wsi, X_clin, X_lab]),
        'NoWSI (Clinical+Labs)': np.hstack([X_clin, X_lab]),
        'ClinicalOnly': X_clin,
    }


def train_and_evaluate():
    print("Loading training data...")
    train_data = load_split_data('train')
    print("Loading validation data...")
    val_data = load_split_data('val')

    strat_labels = build_stratification_labels(train_data['merged'])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    models_config = {
        'XGBoost': lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
            random_state=SEED, eval_metric='logloss', verbosity=0,
            use_label_encoder=False
        ),
        'LightGBM': lambda: lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
            random_state=SEED, verbose=-1
        ),
        'LogisticRegression': lambda: LogisticRegression(
            C=1.0, max_iter=2000, random_state=SEED, solver='saga',
            penalty='elasticnet', l1_ratio=0.5
        ),
        'MLP': lambda: MLPClassifier(
            hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
            alpha=1e-3, max_iter=500, random_state=SEED, early_stopping=True,
            validation_fraction=0.15
        ),
    }

    all_results = {}

    for feat_name, X_train_feat in get_feature_sets(
        train_data['X_wsi'], train_data['X_clin'], train_data['X_lab']
    ).items():
        X_val_feat = get_feature_sets(
            val_data['X_wsi'], val_data['X_clin'], val_data['X_lab']
        )[feat_name]

        print(f"\n{'='*60}")
        print(f"Feature set: {feat_name} (dim={X_train_feat.shape[1]})")
        print(f"{'='*60}")

        for model_name, model_fn in models_config.items():
            print(f"\n--- {model_name} ---")
            fold_aurocs_3y = []
            fold_aurocs_8y = []
            val_probs_3y_folds = []
            val_probs_8y_folds = []

            for fold, (train_idx, val_idx) in enumerate(
                skf.split(np.zeros(len(strat_labels)), strat_labels)
            ):
                X_tr = X_train_feat[train_idx]
                X_va = X_train_feat[val_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)
                X_val_s = scaler.transform(X_val_feat)

                for label_name, y_all, ep_name in [
                    ('label_5y', train_data['label_5y'], '3y'),
                    ('label_10y', train_data['label_10y'], '8y'),
                ]:
                    y_tr = y_all[train_idx]
                    y_va = y_all[val_idx]

                    model = model_fn()
                    model.fit(X_tr_s, y_tr)

                    va_prob = model.predict_proba(X_va_s)[:, 1]
                    va_auc = roc_auc_score(y_va, va_prob) if len(np.unique(y_va)) > 1 else np.nan

                    val_prob = model.predict_proba(X_val_s)[:, 1]

                    if ep_name == '3y':
                        fold_aurocs_3y.append(va_auc)
                        val_probs_3y_folds.append(val_prob)
                    else:
                        fold_aurocs_8y.append(va_auc)
                        val_probs_8y_folds.append(val_prob)

            # Ensemble: average across folds
            val_prob_3y = np.mean(val_probs_3y_folds, axis=0)
            val_prob_8y = np.mean(val_probs_8y_folds, axis=0)

            val_auroc_3y = roc_auc_score(val_data['label_5y'], val_prob_3y)
            val_auroc_8y = roc_auc_score(val_data['label_10y'], val_prob_8y)
            ci_3y = bootstrap_auroc(val_data['label_5y'], val_prob_3y)
            ci_8y = bootstrap_auroc(val_data['label_10y'], val_prob_8y)

            auprc_3y = average_precision_score(val_data['label_5y'], val_prob_3y)
            auprc_8y = average_precision_score(val_data['label_10y'], val_prob_8y)
            brier_3y = brier_score_loss(val_data['label_5y'], val_prob_3y)
            brier_8y = brier_score_loss(val_data['label_10y'], val_prob_8y)

            key = f"{model_name}_{feat_name.split()[0]}"
            all_results[key] = {
                'model': model_name,
                'features': feat_name,
                'n_features': int(X_train_feat.shape[1]),
                'cv_auroc_3y_mean': round(np.nanmean(fold_aurocs_3y), 4),
                'cv_auroc_3y_std': round(np.nanstd(fold_aurocs_3y), 4),
                'cv_auroc_8y_mean': round(np.nanmean(fold_aurocs_8y), 4),
                'cv_auroc_8y_std': round(np.nanstd(fold_aurocs_8y), 4),
                'val_auroc_3y': round(val_auroc_3y, 4),
                'val_auroc_3y_ci': [round(ci_3y[0], 4), round(ci_3y[1], 4)],
                'val_auroc_8y': round(val_auroc_8y, 4),
                'val_auroc_8y_ci': [round(ci_8y[0], 4), round(ci_8y[1], 4)],
                'val_auprc_3y': round(auprc_3y, 4),
                'val_auprc_8y': round(auprc_8y, 4),
                'val_brier_3y': round(brier_3y, 4),
                'val_brier_8y': round(brier_8y, 4),
            }

            print(f"  CV 3y: {np.nanmean(fold_aurocs_3y):.4f}±{np.nanstd(fold_aurocs_3y):.4f}  "
                  f"CV 8y: {np.nanmean(fold_aurocs_8y):.4f}±{np.nanstd(fold_aurocs_8y):.4f}")
            print(f"  Val 3y: {val_auroc_3y:.4f} [{ci_3y[0]:.4f}–{ci_3y[1]:.4f}]  "
                  f"Val 8y: {val_auroc_8y:.4f} [{ci_8y[0]:.4f}–{ci_8y[1]:.4f}]")

    # Save results
    with open(OUT_DIR / "simple_baselines.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR / 'simple_baselines.json'}")

    # Print comparison table
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Model':<25} {'Features':<25} {'Val 3y AUROC':>14} {'Val 8y AUROC':>14}")
    print("-" * 80)

    # Add M5 reference
    print(f"{'M5 (Transformer+BiLSTM)':<25} {'Full (WSI+Clin+Labs)':<25} {'0.8446':>14} {'0.9020':>14}")
    print(f"{'M4 (Cross-Attention)':<25} {'WSI+Clinical':<25} {'0.7440':>14} {'0.8132':>14}")
    print(f"{'M3 (MLP)':<25} {'WSI-only':<25} {'0.6458':>14} {'0.7147':>14}")
    print("-" * 80)

    for key, r in sorted(all_results.items()):
        feat_short = r['features'].split('(')[0].strip()
        print(f"{r['model']:<25} {feat_short:<25} {r['val_auroc_3y']:>14.4f} {r['val_auroc_8y']:>14.4f}")

    return all_results


if __name__ == '__main__':
    results = train_and_evaluate()
