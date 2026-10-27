#!/usr/bin/env python3
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""Fix NFS calculation: albumin was used in g/L instead of g/dL.
Also recompute all NFS-related statistics and comparisons.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
import json
import math

BASE = Path(f"{ROOT}")

def compute_nfs(row):
    """NFS = -1.675 + 0.037*age + 0.094*BMI + 1.13*DM + 0.99*(AST/ALT) - 0.013*Plt - 0.66*Alb(g/dL)"""
    alb_gdl = row['Albumin'] / 10.0
    return (-1.675
            + 0.037 * row['age']
            + 0.094 * row['BMI']
            + 1.13 * row['Impaired_glucose_tolerance']
            + 0.99 * (row['AST'] / row['ALT'])
            - 0.013 * row['Platelet']
            - 0.66 * alb_gdl)

def bootstrap_auroc(y_true, y_score, n_bootstrap=2000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    return np.percentile(aucs, [2.5, 97.5])

def compute_smd(mean1, sd1, mean2, sd2):
    pooled_sd = math.sqrt((sd1**2 + sd2**2) / 2)
    if pooled_sd == 0:
        return 0
    return abs(mean1 - mean2) / pooled_sd

# ============================================================
# Step 1: Fix NFS in both CSVs
# ============================================================
for split in ['train', 'val']:
    csv_path = BASE / f"dataset/{split}/clinical_data.csv"
    df = pd.read_csv(csv_path)
    old_nfs = df['NFS'].copy()
    df['NFS'] = df.apply(compute_nfs, axis=1)
    print(f"\n{split.upper()} NFS fix:")
    print(f"  Old: mean={old_nfs.mean():.2f} ± {old_nfs.std():.2f}, range=[{old_nfs.min():.2f}, {old_nfs.max():.2f}]")
    print(f"  New: mean={df['NFS'].mean():.2f} ± {df['NFS'].std():.2f}, range=[{df['NFS'].min():.2f}, {df['NFS'].max():.2f}]")
    df.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}")

# ============================================================
# Step 2: Recompute NFS AUROC
# ============================================================
val = pd.read_csv(BASE / "dataset/val/clinical_data.csv")
train = pd.read_csv(BASE / "dataset/train/clinical_data.csv")

# Get labels from predictions
PRED_DIR = BASE / "outputs/models/val_predictions_auc_target_20260509_153628"
m5_preds = pd.read_csv(PRED_DIR / "m5_labs_mask_val_predictions.csv")
val_merged = val[['patient_id', 'NFS']].merge(
    m5_preds[['patient_id', 'label_5y', 'label_10y']], on='patient_id')

nfs_auroc_3y = roc_auc_score(val_merged['label_5y'], val_merged['NFS'])
nfs_auroc_8y = roc_auc_score(val_merged['label_10y'], val_merged['NFS'])
nfs_ci_3y = bootstrap_auroc(val_merged['label_5y'].values, val_merged['NFS'].values)
nfs_ci_8y = bootstrap_auroc(val_merged['label_10y'].values, val_merged['NFS'].values)

print(f"\n=== Corrected NFS AUROC ===")
print(f"  3-year: {nfs_auroc_3y:.4f} [{nfs_ci_3y[0]:.4f}–{nfs_ci_3y[1]:.4f}]")
print(f"  8-year: {nfs_auroc_8y:.4f} [{nfs_ci_8y[0]:.4f}–{nfs_ci_8y[1]:.4f}]")

# DeLong-equivalent paired bootstrap test: NFS vs M4
m4_preds = pd.read_csv(PRED_DIR / "m4_val_predictions.csv")
val_full = val_merged.merge(m4_preds[['patient_id', 'prob_5y', 'prob_10y']].rename(
    columns={'prob_5y': 'm4_prob_5y', 'prob_10y': 'm4_prob_10y'}), on='patient_id')

def paired_bootstrap_p(y_true, pred_a, pred_b, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    auc_a = roc_auc_score(y_true, pred_a)
    auc_b = roc_auc_score(y_true, pred_b)
    diff = auc_a - auc_b
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        d = roc_auc_score(yt, pred_a[idx]) - roc_auc_score(yt, pred_b[idx])
        diffs.append(d)
    se = np.std(diffs)
    if se == 0:
        return 1.0
    from scipy.stats import norm
    z = diff / se
    return 2 * (1 - norm.cdf(abs(z)))

p_3y = paired_bootstrap_p(val_full['label_5y'].values, val_full['m4_prob_5y'].values, val_full['NFS'].values)
p_8y = paired_bootstrap_p(val_full['label_10y'].values, val_full['m4_prob_10y'].values, val_full['NFS'].values)
print(f"  DeLong NFS vs M4: 3y P={p_3y:.4f}, 8y P={p_8y:.4f}")

# ============================================================
# Step 3: Compute new Table 1 stats for NFS
# ============================================================
train_nfs_mean = train['NFS'].mean()
train_nfs_sd = train['NFS'].std()
val_nfs_mean = val['NFS'].mean()
val_nfs_sd = val['NFS'].std()
nfs_smd = compute_smd(train_nfs_mean, train_nfs_sd, val_nfs_mean, val_nfs_sd)

print(f"\n=== Table 1 NFS (corrected) ===")
print(f"  Training: {train_nfs_mean:.2f} ± {train_nfs_sd:.2f}")
print(f"  Validation: {val_nfs_mean:.2f} ± {val_nfs_sd:.2f}")
print(f"  SMD: {nfs_smd:.3f}")

# ============================================================
# Step 4: Update computed_stats.json
# ============================================================
stats_path = BASE / "workspace/computed_stats.json"
with open(stats_path) as f:
    stats = json.load(f)

stats['model_performance']['baseline_nfs'] = {
    'auroc_3y': round(nfs_auroc_3y, 4),
    'auroc_3y_ci': [round(nfs_ci_3y[0], 4), round(nfs_ci_3y[1], 4)],
    'auroc_8y': round(nfs_auroc_8y, 4),
    'auroc_8y_ci': [round(nfs_ci_8y[0], 4), round(nfs_ci_8y[1], 4)],
    'auprc_3y': stats['model_performance']['baseline_nfs'].get('auprc_3y'),
    'auprc_8y': stats['model_performance']['baseline_nfs'].get('auprc_8y'),
    'brier_3y': None,
    'brier_8y': None,
    'delong_p_vs_m4_3y': round(p_3y, 6),
    'delong_p_vs_m4_8y': round(p_8y, 6),
}

stats['table1']['NFS'] = {
    'type': 'continuous',
    'train_mean': round(train_nfs_mean, 2),
    'train_sd': round(train_nfs_sd, 2),
    'val_mean': round(val_nfs_mean, 2),
    'val_sd': round(val_nfs_sd, 2),
    'smd': round(nfs_smd, 3),
}

with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"\nUpdated {stats_path}")

# ============================================================
# Step 5: Save summary for paper update
# ============================================================
summary = {
    'nfs_corrected': True,
    'table1': {
        'train': f"{train_nfs_mean:.2f} ± {train_nfs_sd:.2f}",
        'val': f"{val_nfs_mean:.2f} ± {val_nfs_sd:.2f}",
        'smd': round(nfs_smd, 3),
    },
    'table2': {
        'auroc_3y': f"{nfs_auroc_3y:.3f} [{nfs_ci_3y[0]:.3f}–{nfs_ci_3y[1]:.3f}]",
        'auroc_8y': f"{nfs_auroc_8y:.3f} [{nfs_ci_8y[0]:.3f}–{nfs_ci_8y[1]:.3f}]",
        'delong_p_3y': round(p_3y, 3),
        'delong_p_8y': round(p_8y, 3),
    }
}
print(f"\n=== Summary for paper.tex update ===")
print(json.dumps(summary, indent=2))
