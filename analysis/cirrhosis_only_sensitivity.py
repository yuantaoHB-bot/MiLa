#!/usr/bin/env python3
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""Cirrhosis-only sensitivity analysis.

Reclassify HCC and Liver_Death events as negative, keeping only
cirrhosis as the positive endpoint. Re-compute AUROC for all models.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
import json

BASE = Path(f"{ROOT}")
PRED_DIR = BASE / "outputs/models/val_predictions_auc_target_20260509_153628"

labels = pd.read_csv(BASE / "dataset/val/patient_labels.csv")

print("=== Event type distribution (validation) ===")
print(labels["event_type"].value_counts(dropna=False))
print()

# Build cirrhosis-only labels
# label_5y → 3y post-landmark = biopsy year 2-5; label_10y → 8y = biopsy year 2-10
labels["cirrhosis_5y"] = ((labels["event_type"] == "Cirrhosis") &
                           (labels["time_to_event_years"] <= 5)).astype(int)
labels["cirrhosis_10y"] = ((labels["event_type"] == "Cirrhosis") &
                            (labels["time_to_event_years"] <= 10)).astype(int)

# Also need original composite labels for comparison
labels["composite_5y"] = (labels["liver_progression_occurred"] == 1).astype(int)
# But composite_5y should match label_5y from predictions — let's merge to verify
m5 = pd.read_csv(PRED_DIR / "m5_labs_mask_val_predictions.csv")
labels = labels.merge(m5[["patient_id", "label_5y", "label_10y"]], on="patient_id")

print("=== Label comparison ===")
print(f"Composite 3y events: {labels['label_5y'].sum()}")
print(f"Cirrhosis-only 3y events: {labels['cirrhosis_5y'].sum()}")
print(f"Composite 8y events: {labels['label_10y'].sum()}")
print(f"Cirrhosis-only 8y events: {labels['cirrhosis_10y'].sum()}")
print()

# Identify reclassified patients
diff_3y = labels[(labels["label_5y"] == 1) & (labels["cirrhosis_5y"] == 0)]
diff_8y = labels[(labels["label_10y"] == 1) & (labels["cirrhosis_10y"] == 0)]
print(f"Patients reclassified (3y): {len(diff_3y)} — event types: {diff_3y['event_type'].tolist()}")
print(f"Patients reclassified (8y): {len(diff_8y)} — event types: {diff_8y['event_type'].tolist()}")
print()

# Load all model predictions
model_files = {
    "M3": "m3_val_predictions.csv",
    "M4": "m4_val_predictions.csv",
    "M5 labs+mask": "m5_labs_mask_val_predictions.csv",
    "M5 lab-only": "m5_lab_only_val_predictions.csv",
    "M5 mask-only": "m5_mask_only_val_predictions.csv",
}

# Also load clinical scores for FIB-4, NFS, APRI
val_clin = pd.read_csv(BASE / "dataset/val/clinical_data.csv")
clin_merged = val_clin[["patient_id", "FIB4", "NFS", "APRI"]].merge(
    labels[["patient_id", "cirrhosis_5y", "cirrhosis_10y", "label_5y", "label_10y"]],
    on="patient_id"
)

results = {}

def bootstrap_auroc(y_true, y_score, n_bootstrap=2000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    return np.percentile(aucs, [2.5, 97.5])

print("=== Cirrhosis-only AUROC (external validation) ===")
print(f"{'Model':<20} {'Composite 3y':>14} {'Cirrh-only 3y':>14} {'Composite 8y':>14} {'Cirrh-only 8y':>14}")
print("-" * 80)

for name, fname in model_files.items():
    pred = pd.read_csv(PRED_DIR / fname)
    merged = pred.merge(labels[["patient_id", "cirrhosis_5y", "cirrhosis_10y"]], on="patient_id")

    comp_3y = roc_auc_score(merged["label_5y"], merged["prob_5y"])
    cirr_3y = roc_auc_score(merged["cirrhosis_5y"], merged["prob_5y"])
    comp_8y = roc_auc_score(merged["label_10y"], merged["prob_10y"])
    cirr_8y = roc_auc_score(merged["cirrhosis_10y"], merged["prob_10y"])

    ci_cirr_3y = bootstrap_auroc(merged["cirrhosis_5y"].values, merged["prob_5y"].values)
    ci_cirr_8y = bootstrap_auroc(merged["cirrhosis_10y"].values, merged["prob_10y"].values)

    results[name] = {
        "composite_3y": round(comp_3y, 4),
        "cirrhosis_only_3y": round(cirr_3y, 4),
        "cirrhosis_only_3y_ci": [round(ci_cirr_3y[0], 4), round(ci_cirr_3y[1], 4)],
        "composite_8y": round(comp_8y, 4),
        "cirrhosis_only_8y": round(cirr_8y, 4),
        "cirrhosis_only_8y_ci": [round(ci_cirr_8y[0], 4), round(ci_cirr_8y[1], 4)],
    }
    print(f"{name:<20} {comp_3y:>14.4f} {cirr_3y:>14.4f} {comp_8y:>14.4f} {cirr_8y:>14.4f}")

# Traditional scores
for score_name, display_name in [("FIB4", "Baseline FIB-4"), ("NFS", "Baseline NFS"), ("APRI", "Baseline APRI")]:
    comp_3y = roc_auc_score(clin_merged["label_5y"], clin_merged[score_name])
    cirr_3y = roc_auc_score(clin_merged["cirrhosis_5y"], clin_merged[score_name])
    comp_8y = roc_auc_score(clin_merged["label_10y"], clin_merged[score_name])
    cirr_8y = roc_auc_score(clin_merged["cirrhosis_10y"], clin_merged[score_name])

    ci_cirr_3y = bootstrap_auroc(clin_merged["cirrhosis_5y"].values, clin_merged[score_name].values)
    ci_cirr_8y = bootstrap_auroc(clin_merged["cirrhosis_10y"].values, clin_merged[score_name].values)

    results[display_name] = {
        "composite_3y": round(comp_3y, 4),
        "cirrhosis_only_3y": round(cirr_3y, 4),
        "cirrhosis_only_3y_ci": [round(ci_cirr_3y[0], 4), round(ci_cirr_3y[1], 4)],
        "composite_8y": round(comp_8y, 4),
        "cirrhosis_only_8y": round(cirr_8y, 4),
        "cirrhosis_only_8y_ci": [round(ci_cirr_8y[0], 4), round(ci_cirr_8y[1], 4)],
    }
    print(f"{display_name:<20} {comp_3y:>14.4f} {cirr_3y:>14.4f} {comp_8y:>14.4f} {cirr_8y:>14.4f}")

# Save results
output = {
    "description": "Sensitivity analysis: cirrhosis-only endpoint (excluding HCC and liver-related death)",
    "n_total": len(labels),
    "composite_3y_events": int(labels["label_5y"].sum()),
    "cirrhosis_only_3y_events": int(labels["cirrhosis_5y"].sum()),
    "composite_8y_events": int(labels["label_10y"].sum()),
    "cirrhosis_only_8y_events": int(labels["cirrhosis_10y"].sum()),
    "reclassified_3y": len(diff_3y),
    "reclassified_8y": len(diff_8y),
    "models": results,
}

out_path = BASE / "workspace/supplementary/etable_cirrhosis_only.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
