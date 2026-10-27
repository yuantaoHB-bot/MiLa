#!/usr/bin/env python3
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""Phase 1: Supplementary analyses.
1A: Fibrosis stage as comparator
1B: Holm multiple comparison correction
1C: M5 vs APRI DeLong
1D: AUPRC for all models
1E: Detailed lab missingness table
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from scipy import stats as sp_stats
import json

BASE = Path(f"{ROOT}")
PRED_DIR = BASE / "outputs/models/val_predictions_auc_target_20260509_153628"
SUP = BASE / "workspace/supplementary"
SUP.mkdir(parents=True, exist_ok=True)

with open(BASE / "workspace/computed_stats.json") as f:
    cstats = json.load(f)

val_clin = pd.read_csv(BASE / "dataset/val/clinical_data.csv")
train_clin = pd.read_csv(BASE / "dataset/train/clinical_data.csv")
val_labels = pd.read_csv(BASE / "dataset/val/patient_labels.csv")
train_labels = pd.read_csv(BASE / "dataset/train/patient_labels.csv")

m5 = pd.read_csv(PRED_DIR / "m5_labs_mask_val_predictions.csv")
m4 = pd.read_csv(PRED_DIR / "m4_val_predictions.csv")

# Merge labels
val = val_clin.merge(m5[['patient_id', 'label_5y', 'label_10y', 'prob_5y', 'prob_10y']], on='patient_id')
train = train_clin.merge(train_labels[['patient_id', 'liver_progression_occurred', 'time_to_event_years']], on='patient_id')
train['label_5y'] = ((train['liver_progression_occurred'] == 1) & (train['time_to_event_years'] <= 5)).astype(int)
train['label_10y'] = train['liver_progression_occurred'].astype(int)

def bootstrap_auroc(y, s, n=2000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2: continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(aucs, [2.5, 97.5])

def paired_bootstrap_p(y, pa, pb, n=10000, seed=42):
    rng = np.random.RandomState(seed)
    diff = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    diffs = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2: continue
        diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    se = np.std(diffs)
    if se == 0: return 1.0
    return 2 * (1 - sp_stats.norm.cdf(abs(diff / se)))

# ============================================================
# 1A: Fibrosis stage as comparator
# ============================================================
print("=" * 60)
print("1A: Fibrosis stage comparators")
print("=" * 60)

fib_results = {}

for label_col, prob_col, name in [('label_5y', 'prob_5y', '3y'), ('label_10y', 'prob_10y', '8y')]:
    y_val = val[label_col].values

    # Fibrosis stage alone
    fib_auroc = roc_auc_score(y_val, val['Fibrosis_Stage'].values)
    fib_ci = bootstrap_auroc(y_val, val['Fibrosis_Stage'].values)
    print(f"\n{name}: Fibrosis stage alone: AUROC={fib_auroc:.4f} [{fib_ci[0]:.4f}–{fib_ci[1]:.4f}]")

    # Fibrosis + age + sex + diabetes (logistic regression)
    features = ['Fibrosis_Stage', 'age', 'gender', 'Impaired_glucose_tolerance']
    X_train = train[features].values
    y_train = train[label_col].values
    X_val = val[features].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_probs = lr.predict_proba(X_val_s)[:, 1]
    lr_auroc = roc_auc_score(y_val, lr_probs)
    lr_ci = bootstrap_auroc(y_val, lr_probs)
    print(f"{name}: Fib+age+sex+DM LR: AUROC={lr_auroc:.4f} [{lr_ci[0]:.4f}–{lr_ci[1]:.4f}]")

    # M5 vs fibrosis stage DeLong
    m5_probs = val[prob_col].values
    p_m5_vs_fib = paired_bootstrap_p(y_val, m5_probs, val['Fibrosis_Stage'].values.astype(float))
    p_m5_vs_lr = paired_bootstrap_p(y_val, m5_probs, lr_probs)
    print(f"{name}: M5 vs Fib stage P={p_m5_vs_fib:.4f}, M5 vs Fib+clinical LR P={p_m5_vs_lr:.4f}")

    fib_results[name] = {
        'fibrosis_stage_auroc': round(fib_auroc, 4),
        'fibrosis_stage_ci': [round(fib_ci[0], 4), round(fib_ci[1], 4)],
        'fibrosis_clinical_lr_auroc': round(lr_auroc, 4),
        'fibrosis_clinical_lr_ci': [round(lr_ci[0], 4), round(lr_ci[1], 4)],
        'p_m5_vs_fibrosis': round(p_m5_vs_fib, 4),
        'p_m5_vs_fibrosis_lr': round(p_m5_vs_lr, 4),
    }

with open(SUP / "fibrosis_stage_comparator.json", "w") as f:
    json.dump(fib_results, f, indent=2)

# ============================================================
# 1B: Holm multiple comparison correction
# ============================================================
print("\n" + "=" * 60)
print("1B: Holm multiple comparison correction")
print("=" * 60)

# Collect all DeLong P values (vs M4) from computed_stats
raw_ps = {}
for model_key in ['m3', 'm5_labs_mask', 'm5_lab_only', 'm5_mask_only', 'baseline_fib4', 'baseline_nfs', 'baseline_apri']:
    perf = cstats['model_performance'][model_key]
    for endpoint in ['3y', '8y']:
        p = perf.get(f'delong_p_vs_m4_{endpoint}')
        if p is not None:
            raw_ps[f"{model_key}_{endpoint}"] = p

# Sort by P
sorted_ps = sorted(raw_ps.items(), key=lambda x: x[1])
n_tests = len(sorted_ps)
holm_results = {}
for rank, (name, p) in enumerate(sorted_ps, 1):
    holm_threshold = 0.05 / (n_tests - rank + 1)
    holm_sig = p <= holm_threshold
    holm_results[name] = {'raw_p': round(p, 6), 'holm_threshold': round(holm_threshold, 4), 'significant': holm_sig}
    print(f"  {name}: P={p:.6f}, Holm threshold={holm_threshold:.4f}, sig={holm_sig}")

with open(SUP / "holm_correction.json", "w") as f:
    json.dump(holm_results, f, indent=2)

# ============================================================
# 1C: M5 vs APRI DeLong
# ============================================================
print("\n" + "=" * 60)
print("1C: M5 vs APRI")
print("=" * 60)

for label_col, prob_col, name in [('label_5y', 'prob_5y', '3y'), ('label_10y', 'prob_10y', '8y')]:
    y = val[label_col].values
    p = paired_bootstrap_p(y, val[prob_col].values, val['APRI'].values.astype(float))
    print(f"  M5 vs APRI ({name}): P={p:.4f}")

# ============================================================
# 1D: AUPRC for all models
# ============================================================
print("\n" + "=" * 60)
print("1D: AUPRC for all models")
print("=" * 60)

auprc_results = {}
for model_name, fname in [("M3", "m3"), ("M4", "m4"), ("M5", "m5_labs_mask"),
                            ("M5-Lab", "m5_lab_only"), ("M5-Mask", "m5_mask_only")]:
    pred = pd.read_csv(PRED_DIR / f"{fname}_val_predictions.csv")
    for label_col, prob_col, ep in [('label_5y', 'prob_5y', '3y'), ('label_10y', 'prob_10y', '8y')]:
        auprc = average_precision_score(pred[label_col], pred[prob_col])
        key = f"{model_name}_{ep}"
        auprc_results[key] = round(auprc, 4)
        print(f"  {model_name} {ep}: AUPRC={auprc:.4f}")

# Traditional scores
for score_name, display in [('FIB4', 'FIB-4'), ('NFS', 'NFS'), ('APRI', 'APRI')]:
    for label_col, ep in [('label_5y', '3y'), ('label_10y', '8y')]:
        auprc = average_precision_score(val[label_col], val[score_name])
        key = f"{display}_{ep}"
        auprc_results[key] = round(auprc, 4)
        print(f"  {display} {ep}: AUPRC={auprc:.4f}")

with open(SUP / "auprc_all_models.json", "w") as f:
    json.dump(auprc_results, f, indent=2)

# ============================================================
# 1E: Detailed lab missingness
# ============================================================
print("\n" + "=" * 60)
print("1E: Detailed lab missingness by analyte")
print("=" * 60)

lab_dir = BASE / "dataset/follow_up_labs_clean"
analytes = ['ALT', 'AST', 'GGT', 'TBIL', 'Albumin', 'Platelet', 'FBG', 'TG', 'TC', 'Creatinine', 'INR']

def compute_analyte_missingness(labels_csv, lab_dir):
    labels = pd.read_csv(labels_csv)
    results = {a: {'year1_obs': 0, 'year2_obs': 0, 'total': len(labels)} for a in analytes}
    for _, row in labels.iterrows():
        pid = str(row['patient_id'])
        lab_file = lab_dir / f"{pid}_follow_up_labs.csv"
        if not lab_file.exists():
            continue
        labs = pd.read_csv(lab_file)
        y1 = labs[labs['days_from_biopsy'] <= 365]
        y2 = labs[(labs['days_from_biopsy'] > 365) & (labs['days_from_biopsy'] <= 730)]
        for a in analytes:
            if a in y1.columns and y1[a].notna().any():
                results[a]['year1_obs'] += 1
            if a in y2.columns and y2[a].notna().any():
                results[a]['year2_obs'] += 1
    return results

train_miss = compute_analyte_missingness(BASE / "dataset/train/patient_labels.csv", lab_dir)
val_miss = compute_analyte_missingness(BASE / "dataset/val/patient_labels.csv", lab_dir)

miss_table = {}
print(f"{'Analyte':>12} | {'Train Y1%':>10} {'Train Y2%':>10} | {'Val Y1%':>10} {'Val Y2%':>10}")
print("-" * 65)
for a in analytes:
    t1 = train_miss[a]['year1_obs'] / train_miss[a]['total'] * 100
    t2 = train_miss[a]['year2_obs'] / train_miss[a]['total'] * 100
    v1 = val_miss[a]['year1_obs'] / val_miss[a]['total'] * 100
    v2 = val_miss[a]['year2_obs'] / val_miss[a]['total'] * 100
    print(f"{a:>12} | {t1:>9.1f}% {t2:>9.1f}% | {v1:>9.1f}% {v2:>9.1f}%")
    miss_table[a] = {
        'train_year1_pct': round(t1, 1), 'train_year2_pct': round(t2, 1),
        'val_year1_pct': round(v1, 1), 'val_year2_pct': round(v2, 1),
    }

with open(SUP / "etable_lab_missingness_detail.json", "w") as f:
    json.dump(miss_table, f, indent=2)

print("\n=== All Phase 1 analyses complete ===")
