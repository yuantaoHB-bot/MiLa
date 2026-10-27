#!/usr/bin/env python3
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""Full cirrhosis-only endpoint sensitivity analysis.

Recomputes AUROC, Brier score, DeLong p-values (vs M4 reference),
and MiLa threshold performance for the cirrhosis-only endpoint.
All baselines use corrected NFS (albumin g/dL fix already applied to
clinical_data.csv via fix_nfs.py).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss

BASE = Path(f"{ROOT}")
PRED_DIR = BASE / "outputs/models/val_predictions_auc_target_20260509_153628"
OUT_PATH = BASE / "workspace/supplementary/cirrhosis_only_full_results.json"

# ── 1. Labels ──────────────────────────────────────────────────────────────────
labels = pd.read_csv(BASE / "dataset/val/patient_labels.csv")
labels["cirrhosis_3y"] = (
    (labels["event_type"] == "Cirrhosis") &
    (labels["time_to_event_years"] <= 5)  # label_5y = 3-year post-landmark
).astype(int)
labels["cirrhosis_8y"] = (
    (labels["event_type"] == "Cirrhosis") &
    (labels["time_to_event_years"] <= 10)
).astype(int)

# ── 2. Clinical baselines ───────────────────────────────────────────────────────
clin = pd.read_csv(BASE / "dataset/val/clinical_data.csv")
clin = clin.merge(labels[["patient_id", "cirrhosis_3y", "cirrhosis_8y"]], on="patient_id")
# Merge model label columns too for composite reference
m5_preds = pd.read_csv(PRED_DIR / "m5_labs_mask_val_predictions.csv")
clin = clin.merge(m5_preds[["patient_id", "label_5y", "label_10y"]], on="patient_id")

# CORE score per patient
core_df = pd.read_csv(BASE / "workspace/supplementary/core_score_per_patient.csv")
clin = clin.merge(core_df[["patient_id", "core_risk"]], on="patient_id")

# ── 3. Helpers ─────────────────────────────────────────────────────────────────
def bootstrap_auroc(y_true, y_score, n=2000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    return np.percentile(aucs, [2.5, 97.5]).tolist()


def delong_pvalue(y_true, pred_a, pred_b):
    """DeLong test for correlated AUROCs (two-sided). Returns p-value."""
    # Hanley-McNeil approach via variance of the difference of AUROCs
    # Using the method from DeLong et al. 1988 (structural components).
    def structural_components(y_true, y_score):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        n_pos, n_neg = len(pos), len(neg)
        # V10: for each positive, fraction of negatives it beats
        V10 = np.array([(pos[i] > neg).mean() + 0.5 * (pos[i] == neg).mean()
                         for i in range(n_pos)])
        # V01: for each negative, fraction of positives it loses to
        V01 = np.array([(neg[j] < pos).mean() + 0.5 * (neg[j] == pos).mean()
                         for j in range(n_neg)])
        return V10, V01

    y = np.asarray(y_true)
    V10_a, V01_a = structural_components(y, pred_a)
    V10_b, V01_b = structural_components(y, pred_b)
    n_pos = V10_a.shape[0]
    n_neg = V01_a.shape[0]

    # Covariance matrix S of [theta_a, theta_b]
    S10 = np.cov(np.stack([V10_a, V10_b])) / n_pos
    S01 = np.cov(np.stack([V01_a, V01_b])) / n_neg
    S = S10 + S01  # 2×2 covariance

    auc_a = roc_auc_score(y, pred_a)
    auc_b = roc_auc_score(y, pred_b)
    diff = auc_a - auc_b
    # Variance of diff = S[0,0] + S[1,1] - 2*S[0,1]
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        return 1.0
    from scipy import stats
    z = diff / np.sqrt(var_diff)
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def threshold_performance(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {"sens": round(sens, 3), "spec": round(spec, 3),
            "ppv": round(ppv, 3), "npv": round(npv, 3)}


# ── 4. Model predictions ───────────────────────────────────────────────────────
model_files = {
    "M3":           "m3_val_predictions.csv",
    "M4":           "m4_val_predictions.csv",
    "MiLa_mask":   "m5_mask_only_val_predictions.csv",
    "MiLa_lab":    "m5_lab_only_val_predictions.csv",
    "MiLa":        "m5_labs_mask_val_predictions.csv",
}
preds = {}
for name, fname in model_files.items():
    df = pd.read_csv(PRED_DIR / fname)
    df = df.merge(labels[["patient_id", "cirrhosis_3y", "cirrhosis_8y"]], on="patient_id")
    preds[name] = df

# ── 5. Compute all AUROCs ──────────────────────────────────────────────────────
results = {}

print(f"{'Model/Score':<22} {'Cirrh 3y':>9} {'Cirrh 8y':>9}")
print("-" * 44)

# Deep learning models
for name, df in preds.items():
    a3 = roc_auc_score(df["cirrhosis_3y"], df["prob_5y"])
    a8 = roc_auc_score(df["cirrhosis_8y"], df["prob_10y"])
    ci3 = bootstrap_auroc(df["cirrhosis_3y"].values, df["prob_5y"].values)
    ci8 = bootstrap_auroc(df["cirrhosis_8y"].values, df["prob_10y"].values)
    b3 = brier_score_loss(df["cirrhosis_3y"], df["prob_5y"])
    b8 = brier_score_loss(df["cirrhosis_8y"], df["prob_10y"])
    # DeLong vs M4
    if name != "M4":
        m4_df = preds["M4"]
        p3 = delong_pvalue(df["cirrhosis_3y"].values, df["prob_5y"].values, m4_df["prob_5y"].values)
        p8 = delong_pvalue(df["cirrhosis_8y"].values, df["prob_10y"].values, m4_df["prob_10y"].values)
    else:
        p3, p8 = None, None
    results[name] = {
        "auroc_3y": round(a3, 4), "auroc_3y_ci": [round(x,4) for x in ci3],
        "auroc_8y": round(a8, 4), "auroc_8y_ci": [round(x,4) for x in ci8],
        "brier_3y": round(b3, 4), "brier_8y": round(b8, 4),
        "delong_p_vs_m4_3y": round(p3, 4) if p3 is not None else "ref",
        "delong_p_vs_m4_8y": round(p8, 4) if p8 is not None else "ref",
    }
    print(f"  {name:<20} {a3:>9.4f} {a8:>9.4f}")

# Clinical baselines
clinical_scores = {
    "FIB-4": "FIB4",
    "NFS":   "NFS",
    "APRI":  "APRI",
    "Fibrosis_stage": "Fibrosis_Stage",
    "CORE":  "core_risk",
}
for display, col in clinical_scores.items():
    y3 = clin["cirrhosis_3y"].values
    y8 = clin["cirrhosis_8y"].values
    s  = clin[col].values
    a3 = roc_auc_score(y3, s)
    a8 = roc_auc_score(y8, s)
    ci3 = bootstrap_auroc(y3, s)
    ci8 = bootstrap_auroc(y8, s)
    # DeLong vs M4
    m4 = preds["M4"]
    p3 = delong_pvalue(y3, s, m4["prob_5y"].values)
    p8 = delong_pvalue(y8, s, m4["prob_10y"].values)
    results[display] = {
        "auroc_3y": round(a3, 4), "auroc_3y_ci": [round(x,4) for x in ci3],
        "auroc_8y": round(a8, 4), "auroc_8y_ci": [round(x,4) for x in ci8],
        "brier_3y": None, "brier_8y": None,
        "delong_p_vs_m4_3y": round(p3, 4),
        "delong_p_vs_m4_8y": round(p8, 4),
    }
    print(f"  {display:<20} {a3:>9.4f} {a8:>9.4f}")

# ── 6. Threshold performance for MiLa (cirrhosis-only) ─────────────────────────
mila = preds["MiLa"]
threshold_results = {}
for thr in [0.10, 0.20, 0.30]:
    key = f"{int(thr*100)}pct"
    threshold_results[key] = {
        "3y": threshold_performance(mila["cirrhosis_3y"].values, mila["prob_5y"].values, thr),
        "8y": threshold_performance(mila["cirrhosis_8y"].values, mila["prob_10y"].values, thr),
    }
    t3 = threshold_results[key]["3y"]
    t8 = threshold_results[key]["8y"]
    print(f"\n  MiLa threshold {int(thr*100)}%:")
    print(f"    3y: Sens={t3['sens']:.3f} Spec={t3['spec']:.3f} PPV={t3['ppv']:.3f} NPV={t3['npv']:.3f}")
    print(f"    8y: Sens={t8['sens']:.3f} Spec={t8['spec']:.3f} PPV={t8['ppv']:.3f} NPV={t8['npv']:.3f}")

# ── 7. Event counts ─────────────────────────────────────────────────────────────
event_counts = {
    "n_val": len(labels),
    "composite_3y": int(clin["label_5y"].sum()),
    "composite_8y": int(clin["label_10y"].sum()),
    "cirrhosis_3y": int(labels["cirrhosis_3y"].sum()),
    "cirrhosis_8y": int(labels["cirrhosis_8y"].sum()),
    "reclassified_3y": int(clin["label_5y"].sum()) - int(labels["cirrhosis_3y"].sum()),
    "reclassified_8y": int(clin["label_10y"].sum()) - int(labels["cirrhosis_8y"].sum()),
}
print("\n=== Event counts ===")
for k, v in event_counts.items():
    print(f"  {k}: {v}")

# ── 8. Save ─────────────────────────────────────────────────────────────────────
output = {
    "description": "Full cirrhosis-only endpoint analysis with corrected NFS, all models/baselines, thresholds, DeLong",
    "event_counts": event_counts,
    "models": results,
    "mila_threshold_performance": threshold_results,
}
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUT_PATH}")
