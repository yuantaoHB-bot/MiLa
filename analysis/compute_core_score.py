from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""
Compute CORE score (Strandberg et al., BMJ 2024) on val cohort.
Compare against MiLa (M5 labs_mask) using true landmark-based labels.

Formula source: strr083182.ww1.pdf (Supplementary p.16)
Knot locations:  Supplementary p.11

CORE variables: age (years), sex (1=male), GGT/AST/ALT (U/L → μkat/L by /60)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import json, os

# ---------------------------------------------------------------------------
# Truncated cubic spline helper
# ---------------------------------------------------------------------------
def tc(x):
    """(x)_+^3  =  x^3 if x > 0, else 0"""
    return np.where(x > 0, x ** 3, 0.0)


def core_risk(age, sex_male, ggt_ul, ast_ul, alt_ul):
    """
    10-year MALO probability from CORE score.

    Parameters
    ----------
    age      : array-like, years
    sex_male : array-like, 1=male 0=female
    ggt_ul   : array-like, GGT in U/L
    ast_ul   : array-like, AST in U/L
    alt_ul   : array-like, ALT in U/L

    Returns
    -------
    risk : ndarray of predicted 10-year probabilities
    """
    # Convert U/L → μkat/L (1 μkat/L = 60 U/L)
    ggt = np.clip(np.asarray(ggt_ul, float) / 60.0, 0.01, 17.0)
    ast = np.clip(np.asarray(ast_ul, float) / 60.0, 0.01, 10.0)
    alt = np.clip(np.asarray(alt_ul, float) / 60.0, 0.01, 10.0)
    age = np.asarray(age, float)
    sex = np.asarray(sex_male, float)

    lg, ls, la = np.log(ggt), np.log(ast), np.log(alt)

    # Confirmed knot positions (Supplementary p.11):
    # Age:  30.7, 39.3, 46.7, 56.2, 70.1   (OCR ≈ 30.65, 39.27, 46.68, 56.16, 70.52)
    # GGT (ln μkat/L): -1.966, -1.386, -0.942, +0.166, +1.609  → (lg+1.97),(lg+1.39),(lg+0.94),(lg-0.17),(lg-1.61)
    # AST (ln μkat/L): -1.561, -1.204, -0.994, -0.462, +0.593  → (ls+1.56),(ls+1.20),(ls+0.99),(ls+0.46),(ls-0.59)
    # ALT (ln μkat/L): -1.897, -1.273, -0.844, +0.010, +0.982  → (la+1.90),(la+1.27),(la+0.84),(la-0.01),(la-0.98)

    xb = (
        # Intercept
        -13.0001

        # Age (spline)
        + 0.1130476    * age
        - 0.0001670826 * tc(age - 30.65)
        + 0.0003334586 * tc(age - 39.27)
        - 0.0001490879 * tc(age - 46.68)
        - 1.425762e-5  * tc(age - 56.16)
        - 3.030396e-8  * tc(age - 70.52)

        # ln(GGT) (spline)
        - 0.9665283  * lg
        + 2.042459   * tc(lg + 1.97)
        - 6.212038   * tc(lg + 1.39)
        + 4.895392   * tc(lg + 0.94)
        - 0.8183846  * tc(lg - 0.17)
        + 0.09257177 * tc(lg - 1.61)

        # ln(AST) (spline)
        - 0.6254278 * ls
        + 0.9052493 * tc(ls + 1.56)
        + 3.456491  * tc(ls + 1.20)
        - 7.848813  * tc(ls + 0.99)
        + 4.072892  * tc(ls + 0.46)
        - 0.5858198 * tc(ls - 0.59)

        # ln(ALT) (spline)
        - 0.5258893 * la
        + 0.158157  * tc(la + 1.90)
        - 0.5817669 * tc(la + 1.27)
        + 0.2363219 * tc(la + 0.84)   # knot confirmed at 0.43 μkat/L → ln≈-0.844
        + 0.4381633 * tc(la - 0.01)
        - 0.2508752 * tc(la - 0.98)
    )

    # Sex (male) main + interaction
    xb += sex * (
        0.6098087
        - 0.04002686   * age
        + 0.0001498023 * tc(age - 30.65)
        - 0.0002338722 * tc(age - 39.27)
        - 8.857314e-6  * tc(age - 46.68)
        + 0.0002400746 * tc(age - 56.16)
        - 6.743156e-5  * tc(age - 70.52)

        - 0.2622872  * lg
        + 0.05660888 * tc(lg + 1.97)
        + 0.1045051  * tc(lg + 1.39)
        - 0.2941066  * tc(lg + 0.94)
        + 0.1626142  * tc(lg - 0.17)
        - 0.02962157 * tc(lg - 1.61)

        - 0.2470467  * ls
        - 1.211869   * tc(ls + 1.56)
        + 5.617974   * tc(ls + 1.20)
        - 5.453922   * tc(ls + 0.99)
        + 1.11022    * tc(ls + 0.46)
        - 0.06240334 * tc(ls - 0.59)

        - 0.03140536 * la
        - 0.1253678  * tc(la + 1.90)
        + 0.6473214  * tc(la + 1.27)
        - 0.6380499  * tc(la + 0.84)
        + 0.06804696 * tc(la - 0.01)
        + 0.04804941 * tc(la - 0.98)
    )

    # Age × ln(GGT) spline interaction
    xb += age * (
        - 0.005854575 * tc(lg + 1.97)
        + 0.01532457  * tc(lg + 1.39)
        - 0.01369761  * tc(lg + 0.94)
        + 0.006903603 * tc(lg - 0.17)
        - 0.002675983 * tc(lg - 1.61)
    )

    # ln(GGT) × [age spline + ln(AST) spline + ln(ALT) spline]
    xb += lg * (
        0.0152652    * age
        - 4.708568e-7 * tc(age - 30.65)
        + 0.0001232485* tc(age - 39.27)
        - 0.0001101636* tc(age - 46.68)
        + 4.541027e-6 * tc(age - 56.16)
        - 1.140946e-6 * tc(age - 70.52)

        - 0.8635879  * ls
        + 0.5645145  * tc(ls + 1.56)
        + 0.947918   * tc(ls + 1.20)
        - 2.820256   * tc(ls + 0.99)
        + 1.476022   * tc(ls + 0.46)
        - 0.1681977  * tc(ls - 0.59)

        + 0.01375014 * la
        - 0.1003385  * tc(la + 1.90)
        - 0.3823287  * tc(la + 1.27)
        + 0.9263131  * tc(la + 0.84)
        - 0.5563601  * tc(la - 0.01)
        + 0.1127141  * tc(la - 0.98)
    )

    # ln(AST) × [ln(GGT) spline + ln(ALT) spline]
    xb += ls * (
        0.2048662  * tc(lg + 1.97)
        + 0.344006 * tc(lg + 1.39)
        - 1.023491 * tc(lg + 0.94)
        + 0.5872282* tc(lg - 0.17)
        - 0.1126098* tc(lg - 1.61)

        - 0.03154256 * la
        + 0.04201595 * tc(la + 1.90)
        + 0.03201952 * tc(la + 1.27)
        - 0.5571034  * tc(la + 0.84)
        + 0.8491616  * tc(la - 0.01)
        - 0.3660936  * tc(la - 0.98)
    )

    # ln(ALT) × [ln(GGT) spline + ln(AST) spline]
    xb += la * (
        - 0.06489212* tc(lg + 1.97)
        - 0.2472643 * tc(lg + 1.39)
        + 0.5990766 * tc(lg + 0.94)
        - 0.3847223 * tc(lg - 0.17)
        + 0.0978021 * tc(lg - 1.61)

        - 0.07487616* tc(ls + 1.56)
        + 0.05706162* tc(ls + 1.20)
        - 0.9928076 * tc(ls + 0.99)
        + 1.243481  * tc(ls + 0.46)
        - 0.3826107 * tc(ls - 0.59)
    )

    return 1.0 / (1.0 + np.exp(-xb))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
BASE     = f"{ROOT}/dataset"
PRED_DIR = f"{ROOT}/outputs/models/val_predictions_auc_target_20260509_153628"

val_clin  = pd.read_csv(f"{BASE}/val/clinical_data_clean.csv")
val_preds = pd.read_csv(f"{PRED_DIR}/m5_labs_mask_val_predictions.csv")

# Merge clinical features with true landmark labels
df = val_clin.merge(val_preds[["patient_id", "label_5y", "label_10y"]], on="patient_id")
print(f"Val cohort merged: {len(df)} patients")

# ---------------------------------------------------------------------------
# Compute CORE score
# ---------------------------------------------------------------------------
df["core_risk"] = core_risk(df.age, df.gender, df.GGT, df.AST, df.ALT)

print(f"\nCORE score distribution:")
print(df["core_risk"].describe().round(4))

# Check for NaN/Inf
bad = (~np.isfinite(df["core_risk"])).sum()
if bad > 0:
    print(f"WARNING: {bad} non-finite CORE values")

# ---------------------------------------------------------------------------
# AUROC comparison
# ---------------------------------------------------------------------------
y3 = df["label_5y"].values    # internally named 5y = 3-year post-landmark
y8 = df["label_10y"].values   # internally named 10y = 8-year post-landmark
core = df["core_risk"].values

auc3_core = roc_auc_score(y3, core)
auc8_core = roc_auc_score(y8, core)

# MiLa AUROCs (from CLAUDE.md / manuscript)
mila_3y  = 0.8446
mila_8y  = 0.9020
fib4_3y  = 0.5786
fib4_8y  = 0.6685
nfs_3y   = 0.5762
nfs_8y   = 0.6454
apri_3y  = 0.7155
apri_8y  = 0.7660

print("\n" + "="*55)
print(f"{'Model':<30} {'3-year':>8} {'8-year':>8}")
print("="*55)
print(f"{'MiLa (M5 labs+mask)':<30} {mila_3y:>8.4f} {mila_8y:>8.4f}")
print("-"*55)
print(f"{'CORE (baseline only)':<30} {auc3_core:>8.4f} {auc8_core:>8.4f}")
print(f"{'FIB-4 (from manuscript)':<30} {fib4_3y:>8.4f} {fib4_8y:>8.4f}")
print(f"{'NFS (from manuscript)':<30} {nfs_3y:>8.4f} {nfs_8y:>8.4f}")
print(f"{'APRI (from manuscript)':<30} {apri_3y:>8.4f} {apri_8y:>8.4f}")
print("="*55)

print(f"\nΔ(MiLa - CORE): 3y = {mila_3y - auc3_core:+.4f}, "
      f"8y = {mila_8y - auc8_core:+.4f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = {
    "CORE_3y_AUROC": round(auc3_core, 4),
    "CORE_8y_AUROC": round(auc8_core, 4),
    "MiLa_3y_AUROC": mila_3y,
    "MiLa_8y_AUROC": mila_8y,
    "delta_3y": round(mila_3y - auc3_core, 4),
    "delta_8y": round(mila_8y - auc8_core, 4),
    "n_val": len(df),
    "events_3y": int(y3.sum()),
    "events_8y": int(y8.sum()),
}

out_path = f"{ROOT}/workspace/supplementary/core_score_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")

# Also save per-patient scores for potential bootstrap CI calculation
df[["patient_id", "label_5y", "label_10y", "core_risk"]].to_csv(
    f"{ROOT}/workspace/supplementary/core_score_per_patient.csv",
    index=False
)
print("Per-patient scores saved.")
