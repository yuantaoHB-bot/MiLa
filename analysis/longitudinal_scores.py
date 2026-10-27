#!/usr/bin/env python
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""
Longitudinal traditional fibrosis scores vs. baseline scores.

Computes FIB-4, NFS, APRI using:
  - "baseline" values from clinical_data_clean.csv
  - "longitudinal" values: age + BMI + DM from baseline, but ALT, AST,
    Albumin, Platelet from the most recent follow-up observation within
    730 days (2 years).  Patients without eligible follow-up labs fall
    back to baseline values.

AUROC + bootstrap 95% CI for both 3-year and 8-year endpoints.

Labels come from the canonical val predictions CSV.

Outputs: workspace/supplementary/longitudinal_scores.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = f"{ROOT}"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

SEED = 42
N_BOOTSTRAP = 1000
DEVICE = "cpu"

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
FOLLOWUP_DIR = os.path.join(DATA_DIR, "follow_up_labs_clean")
VAL_PRED_CSV = os.path.join(
    PROJECT_ROOT,
    "outputs", "models",
    "val_predictions_auc_target_20260509_153628",
    "m5_labs_mask_val_predictions.csv",
)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "workspace", "supplementary", "longitudinal_scores.json")

ULN_AST = 40.0  # Upper limit of normal for AST


# ---------------------------------------------------------------------------
# Score formulas
# ---------------------------------------------------------------------------
def calc_fib4(age, ast, alt, platelet):
    """FIB-4 = (age x AST) / (Platelet x sqrt(ALT))"""
    alt_safe = max(alt, 0.1)
    platelet_safe = max(platelet, 0.1)
    return (age * ast) / (platelet_safe * np.sqrt(alt_safe))


def calc_nfs(age, bmi, dm, ast, alt, platelet, albumin_gL):
    """
    NFS = -1.675 + 0.037*age + 0.094*BMI + 1.13*DM
          + 0.99*(AST/ALT) - 0.013*Plt - 0.66*(Alb_g/dL)
    albumin_gL is in g/L; convert to g/dL by dividing by 10.
    """
    alt_safe = max(alt, 0.1)
    alb_gdl = albumin_gL / 10.0
    return (
        -1.675
        + 0.037 * age
        + 0.094 * bmi
        + 1.13 * dm
        + 0.99 * (ast / alt_safe)
        - 0.013 * platelet
        - 0.66 * alb_gdl
    )


def calc_apri(ast, platelet):
    """APRI = (AST / ULN_AST) / Platelet * 100"""
    platelet_safe = max(platelet, 0.1)
    return (ast / ULN_AST) / platelet_safe * 100.0


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_auroc(labels, scores, n_boot=1000, seed=42):
    """Compute AUROC + bootstrap 95% CI."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    base_auc = roc_auc_score(labels, scores)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y = labels[idx]
        s = scores[idx]
        # Skip degenerate samples (single class)
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, s))
    aucs = np.array(aucs)
    ci_lo = float(np.percentile(aucs, 2.5))
    ci_hi = float(np.percentile(aucs, 97.5))
    return float(base_auc), ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(SEED)

    # --- Load validation clinical data (clean) ---
    clin_clean = pd.read_csv(os.path.join(DATA_DIR, "val", "clinical_data_clean.csv"))
    # Also load the non-clean version for baseline FIB4/NFS/APRI if precomputed
    clin_orig = pd.read_csv(os.path.join(DATA_DIR, "val", "clinical_data.csv"))
    clin_orig = clin_orig.set_index("patient_id")

    # --- Load labels ---
    pred_df = pd.read_csv(VAL_PRED_CSV)
    pred_df = pred_df.set_index("patient_id")

    # Use the patient list from clin_clean (intersection with pred_df)
    pids = [pid for pid in clin_clean["patient_id"].tolist() if pid in pred_df.index]
    print(f"Validation patients: {len(pids)}")

    label_5y = np.array([int(pred_df.loc[pid, "label_5y"]) for pid in pids])
    label_10y = np.array([int(pred_df.loc[pid, "label_10y"]) for pid in pids])

    clin_clean = clin_clean.set_index("patient_id")

    # --- Compute baseline scores ---
    baseline_fib4 = []
    baseline_nfs = []
    baseline_apri = []
    for pid in pids:
        row = clin_orig.loc[pid]
        age = float(row["age"])
        bmi = float(row["BMI"])
        dm = int(row["Impaired_glucose_tolerance"])
        ast = float(row["AST"])
        alt = float(row["ALT"])
        platelet = float(row["Platelet"])
        albumin = float(row["Albumin"])

        baseline_fib4.append(calc_fib4(age, ast, alt, platelet))
        baseline_nfs.append(calc_nfs(age, bmi, dm, ast, alt, platelet, albumin))
        baseline_apri.append(calc_apri(ast, platelet))

    baseline_fib4 = np.array(baseline_fib4)
    baseline_nfs = np.array(baseline_nfs)
    baseline_apri = np.array(baseline_apri)

    # --- Compute longitudinal scores ---
    # Use most recent follow-up labs within 730 days for ALT, AST, Albumin, Platelet
    long_fib4 = []
    long_nfs = []
    long_apri = []
    n_with_followup = 0
    n_fallback = 0

    for pid in pids:
        # Baseline demographics (always from baseline)
        row_orig = clin_orig.loc[pid]
        age = float(row_orig["age"])
        bmi = float(row_orig["BMI"])
        dm = int(row_orig["Impaired_glucose_tolerance"])

        # Try loading follow-up labs
        fu_path = os.path.join(FOLLOWUP_DIR, f"{pid}_follow_up_labs.csv")
        used_followup = False

        if os.path.exists(fu_path):
            fu_df = pd.read_csv(fu_path)
            # Filter to within 730 days
            fu_eligible = fu_df[fu_df["days_from_biopsy"] <= 730].copy()
            if len(fu_eligible) > 0:
                # Take the most recent observation
                fu_eligible = fu_eligible.sort_values("days_from_biopsy")
                latest = fu_eligible.iloc[-1]
                alt = float(latest["ALT"])
                ast = float(latest["AST"])
                albumin = float(latest["Albumin"])
                platelet = float(latest["Platelet"])
                used_followup = True
                n_with_followup += 1

        if not used_followup:
            # Fall back to baseline
            alt = float(row_orig["ALT"])
            ast = float(row_orig["AST"])
            albumin = float(row_orig["Albumin"])
            platelet = float(row_orig["Platelet"])
            n_fallback += 1

        long_fib4.append(calc_fib4(age, ast, alt, platelet))
        long_nfs.append(calc_nfs(age, bmi, dm, ast, alt, platelet, albumin))
        long_apri.append(calc_apri(ast, platelet))

    long_fib4 = np.array(long_fib4)
    long_nfs = np.array(long_nfs)
    long_apri = np.array(long_apri)

    print(f"  Patients with follow-up labs (<=730d): {n_with_followup}")
    print(f"  Patients falling back to baseline:     {n_fallback}")

    # --- Compute AUROCs + Bootstrap CIs ---
    results = {"metadata": {
        "n_patients": len(pids),
        "n_with_followup": n_with_followup,
        "n_fallback_to_baseline": n_fallback,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "uln_ast": ULN_AST,
        "followup_window_days": 730,
        "note": "label_5y = 3-year post-landmark; label_10y = 8-year post-landmark",
    }}

    for score_name, bl_scores, lo_scores in [
        ("FIB4", baseline_fib4, long_fib4),
        ("NFS", baseline_nfs, long_nfs),
        ("APRI", baseline_apri, long_apri),
    ]:
        entry = {}
        for endpoint_label, labels in [("3y", label_5y), ("8y", label_10y)]:
            # Baseline
            auc_bl, ci_lo_bl, ci_hi_bl = bootstrap_auroc(labels, bl_scores, N_BOOTSTRAP, SEED)
            entry[f"baseline_{endpoint_label}"] = {
                "auroc": round(auc_bl, 4),
                "ci_lower": round(ci_lo_bl, 4),
                "ci_upper": round(ci_hi_bl, 4),
            }
            # Longitudinal
            auc_lo, ci_lo_lo, ci_hi_lo = bootstrap_auroc(labels, lo_scores, N_BOOTSTRAP, SEED)
            entry[f"longitudinal_{endpoint_label}"] = {
                "auroc": round(auc_lo, 4),
                "ci_lower": round(ci_lo_lo, 4),
                "ci_upper": round(ci_hi_lo, 4),
            }
            print(f"  {score_name} {endpoint_label}: baseline={auc_bl:.4f} [{ci_lo_bl:.4f}-{ci_hi_bl:.4f}]"
                  f"  longitudinal={auc_lo:.4f} [{ci_lo_lo:.4f}-{ci_hi_lo:.4f}]")

        results[score_name] = entry

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
