#!/usr/bin/env python
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""
Logistic regression comparator: Fibrosis_Stage + baseline lab features.

Trains logistic regression models using 5-fold CV (same StratifiedKFold
seed=42 splits as M5) on the training set, then evaluates on the external
validation set.

Four comparators:
  1. "fibrosis_only"        : Fibrosis_Stage alone (univariate)
  2. "fibrosis_clinical"    : Fibrosis_Stage + age + gender + BMI + DM (5 features)
  3. "fibrosis_all_labs"    : Fibrosis_Stage + 17 baseline lab features (18 features)

For each comparator: AUROC + bootstrap 95% CI on external validation for
both 3-year and 8-year endpoints.

Uses clinical_data.csv (NOT _clean) because it contains Fibrosis_Stage.
StandardScaler is fit on train fold only and applied to val fold + external val.

Outputs: workspace/supplementary/fibrosis_labs_lr.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = f"{ROOT}"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

SEED = 42
N_BOOTSTRAP = 1000
N_FOLDS = 5

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
VAL_PRED_CSV = os.path.join(
    PROJECT_ROOT,
    "outputs", "models",
    "val_predictions_auc_target_20260509_153628",
    "m5_labs_mask_val_predictions.csv",
)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "workspace", "supplementary", "fibrosis_labs_lr.json")


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
LAB_17 = [
    "ALT", "AST", "ALP", "GGT", "TBIL", "DBIL", "Albumin", "INR",
    "Creatinine", "Uric_Acid", "FBG", "TG", "TC", "HDL", "LDL",
    "HbA1c", "Platelet",
]

COMPARATORS = {
    "fibrosis_only": ["Fibrosis_Stage"],
    "fibrosis_clinical": ["Fibrosis_Stage", "age", "gender", "BMI", "Impaired_glucose_tolerance"],
    "fibrosis_all_labs": ["Fibrosis_Stage"] + LAB_17,
}


# ---------------------------------------------------------------------------
# Helpers (replicated from src/train.py for consistency)
# ---------------------------------------------------------------------------
def build_stratification_labels(data):
    """Reproduce the exact stratification used in training."""
    label_10y = data["liver_progression_occurred"].astype(int).values
    label_5y = (
        (data["liver_progression_occurred"].astype(int) == 1)
        & (data["time_to_event_years"].astype(float) <= 5)
    ).astype(int).values
    return label_10y + 2 * label_5y


def compute_labels(data):
    """Compute 5y and 10y binary labels."""
    prog = data["liver_progression_occurred"].astype(int).values
    tte = data["time_to_event_years"].astype(float).values
    label_5y = ((prog == 1) & (tte <= 5)).astype(int)
    label_10y = prog
    return label_5y, label_10y


def bootstrap_auroc(labels, scores, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    base_auc = roc_auc_score(labels, scores)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y = labels[idx]
        s = scores[idx]
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, s))
    aucs = np.array(aucs)
    return float(base_auc), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(SEED)

    # --- Load training data ---
    train_clin = pd.read_csv(os.path.join(DATA_DIR, "train", "clinical_data.csv"))
    train_labels = pd.read_csv(os.path.join(DATA_DIR, "train", "patient_labels.csv"))
    train_df = pd.merge(train_clin, train_labels, on="patient_id", how="inner")
    print(f"Training patients: {len(train_df)}")

    # --- Load external validation data ---
    val_clin = pd.read_csv(os.path.join(DATA_DIR, "val", "clinical_data.csv"))
    val_labels_df = pd.read_csv(VAL_PRED_CSV)

    # Merge to get val features + labels
    val_df = pd.merge(val_clin, val_labels_df[["patient_id", "label_5y", "label_10y"]],
                       on="patient_id", how="inner")
    print(f"Validation patients: {len(val_df)}")

    # Compute training labels
    train_label_5y, train_label_10y = compute_labels(train_df)

    # External val labels
    ext_label_5y = val_df["label_5y"].values
    ext_label_10y = val_df["label_10y"].values

    # Stratification for CV
    strat_labels = build_stratification_labels(train_df)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Check for missing values
    for col in set(sum(COMPARATORS.values(), [])):
        n_miss_train = train_df[col].isna().sum()
        n_miss_val = val_df[col].isna().sum()
        if n_miss_train > 0 or n_miss_val > 0:
            print(f"  WARNING: {col} has {n_miss_train} missing in train, {n_miss_val} missing in val")

    results = {
        "metadata": {
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_folds": N_FOLDS,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "note": "label_5y = 3-year post-landmark; label_10y = 8-year post-landmark",
        }
    }

    for comp_name, feature_cols in COMPARATORS.items():
        print(f"\n{'='*60}")
        print(f"Comparator: {comp_name}  ({len(feature_cols)} features)")
        print(f"  Features: {feature_cols}")
        print(f"{'='*60}")

        X_train_all = train_df[feature_cols].values.astype(np.float64)
        X_val_ext = val_df[feature_cols].values.astype(np.float64)

        # Handle any NaN by filling with column median (train)
        for col_i in range(X_train_all.shape[1]):
            mask_train = np.isnan(X_train_all[:, col_i])
            mask_val = np.isnan(X_val_ext[:, col_i])
            if mask_train.any() or mask_val.any():
                median_val = np.nanmedian(X_train_all[:, col_i])
                X_train_all[mask_train, col_i] = median_val
                X_val_ext[mask_val, col_i] = median_val

        comp_result = {"features": feature_cols, "n_features": len(feature_cols)}

        # --- 5-fold CV on training set ---
        cv_auroc_5y, cv_auroc_10y = [], []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(strat_labels)), strat_labels)):
            X_tr = X_train_all[tr_idx]
            X_va = X_train_all[va_idx]
            y_tr_5y = train_label_5y[tr_idx]
            y_va_5y = train_label_5y[va_idx]
            y_tr_10y = train_label_10y[tr_idx]
            y_va_10y = train_label_10y[va_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            # 3-year model
            lr_5y = LogisticRegression(
                max_iter=1000, random_state=SEED, solver="lbfgs", C=1.0
            )
            lr_5y.fit(X_tr_s, y_tr_5y)
            pred_5y = lr_5y.predict_proba(X_va_s)[:, 1]
            auc_5y = roc_auc_score(y_va_5y, pred_5y) if len(np.unique(y_va_5y)) > 1 else float("nan")

            # 8-year model
            lr_10y = LogisticRegression(
                max_iter=1000, random_state=SEED, solver="lbfgs", C=1.0
            )
            lr_10y.fit(X_tr_s, y_tr_10y)
            pred_10y = lr_10y.predict_proba(X_va_s)[:, 1]
            auc_10y = roc_auc_score(y_va_10y, pred_10y) if len(np.unique(y_va_10y)) > 1 else float("nan")

            cv_auroc_5y.append(auc_5y)
            cv_auroc_10y.append(auc_10y)
            print(f"  Fold {fold+1}: 3y AUROC={auc_5y:.4f}, 8y AUROC={auc_10y:.4f}")

        comp_result["cv_3y"] = {
            "auroc_mean": round(float(np.nanmean(cv_auroc_5y)), 4),
            "auroc_std": round(float(np.nanstd(cv_auroc_5y)), 4),
            "auroc_per_fold": [round(float(v), 4) for v in cv_auroc_5y],
        }
        comp_result["cv_8y"] = {
            "auroc_mean": round(float(np.nanmean(cv_auroc_10y)), 4),
            "auroc_std": round(float(np.nanstd(cv_auroc_10y)), 4),
            "auroc_per_fold": [round(float(v), 4) for v in cv_auroc_10y],
        }
        print(f"  CV mean: 3y={np.nanmean(cv_auroc_5y):.4f}+/-{np.nanstd(cv_auroc_5y):.4f}, "
              f"8y={np.nanmean(cv_auroc_10y):.4f}+/-{np.nanstd(cv_auroc_10y):.4f}")

        # --- External validation (train on ALL training data, test on val) ---
        scaler_full = StandardScaler()
        X_train_s = scaler_full.fit_transform(X_train_all)
        X_val_s = scaler_full.transform(X_val_ext)

        # 3-year
        lr_5y_full = LogisticRegression(
            max_iter=1000, random_state=SEED, solver="lbfgs", C=1.0
        )
        lr_5y_full.fit(X_train_s, train_label_5y)
        ext_pred_5y = lr_5y_full.predict_proba(X_val_s)[:, 1]
        auc_ext_5y, ci_lo_5y, ci_hi_5y = bootstrap_auroc(ext_label_5y, ext_pred_5y, N_BOOTSTRAP, SEED)

        # 8-year
        lr_10y_full = LogisticRegression(
            max_iter=1000, random_state=SEED, solver="lbfgs", C=1.0
        )
        lr_10y_full.fit(X_train_s, train_label_10y)
        ext_pred_10y = lr_10y_full.predict_proba(X_val_s)[:, 1]
        auc_ext_10y, ci_lo_10y, ci_hi_10y = bootstrap_auroc(ext_label_10y, ext_pred_10y, N_BOOTSTRAP, SEED)

        comp_result["external_val_3y"] = {
            "auroc": round(auc_ext_5y, 4),
            "ci_lower": round(ci_lo_5y, 4),
            "ci_upper": round(ci_hi_5y, 4),
        }
        comp_result["external_val_8y"] = {
            "auroc": round(auc_ext_10y, 4),
            "ci_lower": round(ci_lo_10y, 4),
            "ci_upper": round(ci_hi_10y, 4),
        }
        print(f"  External val: 3y={auc_ext_5y:.4f} [{ci_lo_5y:.4f}-{ci_hi_5y:.4f}], "
              f"8y={auc_ext_10y:.4f} [{ci_lo_10y:.4f}-{ci_hi_10y:.4f}]")

        # Store LR coefficients for the full-training model (for interpretability)
        coef_5y = lr_5y_full.coef_[0].tolist()
        coef_10y = lr_10y_full.coef_[0].tolist()
        comp_result["lr_coefficients_3y"] = {
            feat: round(c, 6) for feat, c in zip(feature_cols, coef_5y)
        }
        comp_result["lr_coefficients_8y"] = {
            feat: round(c, 6) for feat, c in zip(feature_cols, coef_10y)
        }

        results[comp_name] = comp_result

    # --- Save ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
