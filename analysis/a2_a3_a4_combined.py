#!/usr/bin/env python3
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""A2: Per-fold CV results table for all models.
   A3: DeLong test labs_mask vs lab_only.
   A4: Lab missingness pattern comparison by center.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

BASE = Path(f"{ROOT}")
SUP = BASE / "workspace/supplementary"
SUP.mkdir(parents=True, exist_ok=True)

# =========================================================
# A2: Per-fold CV results
# =========================================================
print("=" * 60)
print("A2: Per-fold CV results")
print("=" * 60)

model_files = {
    "M3": BASE / "outputs/models/m3_results.json",
    "M4": BASE / "outputs/models/m4_results.json",
    "M5 labs_mask": BASE / "outputs/models/m5_labs_mask/m5_results.json",
    "M5 lab_only": BASE / "outputs/models/m5_lab_only/m5_results.json",
    "M5 mask_only": BASE / "outputs/models/m5_mask_only/m5_results.json",
}

all_fold_data = {}
for model_name, fpath in model_files.items():
    with open(fpath) as f:
        data = json.load(f)
    fold_results = []
    for fr in data["fold_results"]:
        fold_results.append({
            "fold": fr["fold"],
            "auroc_3y": round(fr["auroc_5y"], 4),
            "auroc_8y": round(fr["auroc_10y"], 4),
            "n_train": fr["class_distribution"]["train"]["n_samples"],
            "n_val": fr["class_distribution"]["val"]["n_samples"],
            "n_pos_3y_val": fr["class_distribution"]["val"]["n_progression_5y"],
            "n_pos_8y_val": fr["class_distribution"]["val"]["n_progression_10y"],
        })
    all_fold_data[model_name] = fold_results

    aurocs_3y = [r["auroc_3y"] for r in fold_results]
    aurocs_8y = [r["auroc_8y"] for r in fold_results]
    print(f"\n{model_name}:")
    print(f"  3y: mean={np.mean(aurocs_3y):.4f} ± {np.std(aurocs_3y):.4f}  "
          f"range=[{min(aurocs_3y):.4f}, {max(aurocs_3y):.4f}]")
    print(f"  8y: mean={np.mean(aurocs_8y):.4f} ± {np.std(aurocs_8y):.4f}  "
          f"range=[{min(aurocs_8y):.4f}, {max(aurocs_8y):.4f}]")
    for r in fold_results:
        print(f"    Fold {r['fold']}: 3y={r['auroc_3y']:.4f}  8y={r['auroc_8y']:.4f}  "
              f"(val: {r['n_val']} pts, {r['n_pos_3y_val']} 3y-events, {r['n_pos_8y_val']} 8y-events)")

with open(SUP / "etable_perfold_cv.json", "w") as f:
    json.dump(all_fold_data, f, indent=2)
print(f"\nSaved to {SUP / 'etable_perfold_cv.json'}")

# =========================================================
# A3: DeLong test — labs_mask vs lab_only
# =========================================================
print("\n" + "=" * 60)
print("A3: DeLong test — M5 labs_mask vs M5 lab_only")
print("=" * 60)

PRED_DIR = BASE / "outputs/models/val_predictions_auc_target_20260509_153628"
m5_lm = pd.read_csv(PRED_DIR / "m5_labs_mask_val_predictions.csv")
m5_lo = pd.read_csv(PRED_DIR / "m5_lab_only_val_predictions.csv")

def delong_test(y_true, pred_a, pred_b):
    """Simplified paired bootstrap DeLong-equivalent test.
    Uses bootstrap approach (2000 iterations) for paired AUROC comparison.
    """
    from sklearn.metrics import roc_auc_score
    n = len(y_true)
    rng = np.random.RandomState(42)
    auc_a_orig = roc_auc_score(y_true, pred_a)
    auc_b_orig = roc_auc_score(y_true, pred_b)
    diff_orig = auc_a_orig - auc_b_orig

    diffs = []
    for _ in range(10000):
        idx = rng.choice(n, n, replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_a = roc_auc_score(yt, pred_a[idx])
        auc_b = roc_auc_score(yt, pred_b[idx])
        diffs.append(auc_a - auc_b)
    diffs = np.array(diffs)
    se = np.std(diffs)
    if se == 0:
        p_value = 1.0
    else:
        z = diff_orig / se
        p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    ci = np.percentile(diffs, [2.5, 97.5])
    return diff_orig, se, p_value, ci

for label_col, prob_col, name in [
    ("label_5y", "prob_5y", "3-year"),
    ("label_10y", "prob_10y", "8-year")]:

    y = m5_lm[label_col].values
    pa = m5_lm[prob_col].values
    pb = m5_lo[prob_col].values

    from sklearn.metrics import roc_auc_score
    auc_lm = roc_auc_score(y, pa)
    auc_lo = roc_auc_score(y, pb)
    diff, se, p, ci = delong_test(y, pa, pb)

    print(f"\n{name} endpoint:")
    print(f"  M5 labs_mask AUROC: {auc_lm:.4f}")
    print(f"  M5 lab_only AUROC:  {auc_lo:.4f}")
    print(f"  ΔAUROC:             {diff:+.4f}")
    print(f"  SE(ΔAUROC):         {se:.4f}")
    print(f"  95% CI(ΔAUROC):     [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  P-value:            {p:.4f}")

delong_results = {}
for label_col, prob_col, name in [("label_5y", "prob_5y", "3y"), ("label_10y", "prob_10y", "8y")]:
    y = m5_lm[label_col].values
    diff, se, p, ci = delong_test(y, m5_lm[prob_col].values, m5_lo[prob_col].values)
    delong_results[name] = {"delta": round(diff, 4), "se": round(se, 4),
                            "p": round(p, 4), "ci": [round(ci[0], 4), round(ci[1], 4)]}
with open(SUP / "delong_labsmask_vs_labonly.json", "w") as f:
    json.dump(delong_results, f, indent=2)
print(f"\nSaved to {SUP / 'delong_labsmask_vs_labonly.json'}")

# =========================================================
# A4: Lab missingness patterns by center
# =========================================================
print("\n" + "=" * 60)
print("A4: Lab missingness patterns by center")
print("=" * 60)

def get_obs_counts(labels_csv, lab_dir):
    labels = pd.read_csv(labels_csv)
    records = []
    for _, row in labels.iterrows():
        pid = str(row["patient_id"])
        lab_file = lab_dir / f"{pid}_follow_up_labs.csv"
        if lab_file.exists():
            labs = pd.read_csv(lab_file)
            n_within_730 = len(labs[labs["days_from_biopsy"] <= 730])
            has_year1 = int((labs["days_from_biopsy"] <= 365).any())
            has_year2 = int(((labs["days_from_biopsy"] > 365) & (labs["days_from_biopsy"] <= 730)).any())
        else:
            n_within_730 = 0
            has_year1 = 0
            has_year2 = 0
        records.append({"patient_id": pid, "obs_count": n_within_730,
                        "has_year1": has_year1, "has_year2": has_year2})
    return pd.DataFrame(records)

lab_dir = BASE / "dataset/follow_up_labs_clean"
train_obs = get_obs_counts(BASE / "dataset/train/patient_labels.csv", lab_dir)
val_obs = get_obs_counts(BASE / "dataset/val/patient_labels.csv", lab_dir)

train_obs["cohort"] = "Training"
val_obs["cohort"] = "Validation"
combined = pd.concat([train_obs, val_obs], ignore_index=True)

print("\nObservation counts (0-2y window):")
print(f"{'':>15} {'Training (n=180)':>20} {'Validation (n=94)':>20}")
for stat_name, stat_fn in [("Mean", "mean"), ("Median", "median"), ("SD", "std"),
                             ("Min", "min"), ("Max", "max")]:
    t_val = getattr(train_obs["obs_count"], stat_fn)()
    v_val = getattr(val_obs["obs_count"], stat_fn)()
    print(f"{stat_name:>15} {t_val:>20.2f} {v_val:>20.2f}")

# Distribution of obs counts
print("\nDistribution of observation counts:")
for count in range(6):
    t_n = (train_obs["obs_count"] == count).sum()
    v_n = (val_obs["obs_count"] == count).sum()
    t_pct = t_n / len(train_obs) * 100
    v_pct = v_n / len(val_obs) * 100
    if t_n > 0 or v_n > 0:
        print(f"  {count} observations: Train {t_n} ({t_pct:.1f}%)  Val {v_n} ({v_pct:.1f}%)")

# Year 1 vs Year 2 availability
print(f"\nYear 1 lab available: Train {train_obs['has_year1'].sum()}/{len(train_obs)} "
      f"({train_obs['has_year1'].mean()*100:.1f}%)  "
      f"Val {val_obs['has_year1'].sum()}/{len(val_obs)} ({val_obs['has_year1'].mean()*100:.1f}%)")
print(f"Year 2 lab available: Train {train_obs['has_year2'].sum()}/{len(train_obs)} "
      f"({train_obs['has_year2'].mean()*100:.1f}%)  "
      f"Val {val_obs['has_year2'].sum()}/{len(val_obs)} ({val_obs['has_year2'].mean()*100:.1f}%)")

# Mann-Whitney test for center difference
u_stat, u_p = sp_stats.mannwhitneyu(train_obs["obs_count"], val_obs["obs_count"], alternative="two-sided")
print(f"\nMann-Whitney U test (train vs val obs_count): U={u_stat:.1f}, P={u_p:.4f}")

# Chi-squared test for Year 2 availability
from scipy.stats import chi2_contingency
table = [[train_obs["has_year2"].sum(), len(train_obs) - train_obs["has_year2"].sum()],
         [val_obs["has_year2"].sum(), len(val_obs) - val_obs["has_year2"].sum()]]
chi2, chi_p, dof, expected = chi2_contingency(table)
print(f"Chi-squared test (Year 2 availability): χ²={chi2:.3f}, P={chi_p:.4f}")

missingness_results = {
    "train": {
        "n": len(train_obs),
        "obs_count_mean": round(train_obs["obs_count"].mean(), 2),
        "obs_count_median": float(train_obs["obs_count"].median()),
        "obs_count_sd": round(train_obs["obs_count"].std(), 2),
        "year1_available_pct": round(train_obs["has_year1"].mean() * 100, 1),
        "year2_available_pct": round(train_obs["has_year2"].mean() * 100, 1),
    },
    "val": {
        "n": len(val_obs),
        "obs_count_mean": round(val_obs["obs_count"].mean(), 2),
        "obs_count_median": float(val_obs["obs_count"].median()),
        "obs_count_sd": round(val_obs["obs_count"].std(), 2),
        "year1_available_pct": round(val_obs["has_year1"].mean() * 100, 1),
        "year2_available_pct": round(val_obs["has_year2"].mean() * 100, 1),
    },
    "mannwhitney_p": round(u_p, 4),
    "chi2_year2_p": round(chi_p, 4),
}
with open(SUP / "lab_missingness_by_center.json", "w") as f:
    json.dump(missingness_results, f, indent=2)
print(f"\nSaved to {SUP / 'lab_missingness_by_center.json'}")
