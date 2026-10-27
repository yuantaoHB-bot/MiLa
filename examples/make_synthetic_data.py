#!/usr/bin/env python3
"""Create a small SYNTHETIC cohort in the layout expected by run_validation.py.

Every value is drawn at random. No row corresponds to a real patient, and the identifiers (SYN_0001, ...) are
generated here. The data only demonstrate the file formats and let the pipeline run end to end; the resulting
metrics have no meaning (the labels are unrelated to the inputs, so AUROC is expected to be near 0.5).

    python examples/make_synthetic_data.py --out examples/synthetic --n 40
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LABS = ["ALT", "AST", "GGT", "TBIL", "Albumin", "Platelet", "FBG", "TG", "TC", "Creatinine", "INR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("examples/synthetic"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    val = a.out / "dataset" / "val"
    labs_dir = a.out / "dataset" / "follow_up_labs_clean"
    for d in (val, labs_dir, a.out / "embeddings" / "val_HE", a.out / "embeddings" / "val_SR"):
        d.mkdir(parents=True, exist_ok=True)

    ids = [f"SYN_{i:04d}" for i in range(1, a.n + 1)]
    n = a.n
    clin = pd.DataFrame({
        "patient_id": ids,
        "age": rng.integers(25, 75, n), "gender": rng.integers(0, 2, n),
        "BMI": rng.normal(26, 3.5, n).round(1), "height": rng.normal(168, 8, n).round(1),
        "weight": rng.normal(74, 12, n).round(1),
        "ALT": rng.gamma(4, 15, n).round(1), "AST": rng.gamma(4, 11, n).round(1),
        "ALP": rng.normal(85, 20, n).round(1), "GGT": rng.gamma(2.5, 25, n).round(1),
        "TBIL": rng.gamma(4, 4, n).round(1), "DBIL": rng.gamma(3, 1.5, n).round(1),
        "Albumin": rng.normal(43, 4, n).round(1), "INR": rng.normal(1.05, 0.08, n).round(2),
        "Creatinine": rng.normal(75, 14, n).round(1), "Uric_Acid": rng.normal(340, 70, n).round(1),
        "FBG": rng.normal(5.6, 1.1, n).round(2), "TG": rng.gamma(4, 0.4, n).round(2),
        "TC": rng.normal(4.8, 0.9, n).round(2), "HDL": rng.normal(1.2, 0.3, n).round(2),
        "LDL": rng.normal(2.8, 0.7, n).round(2), "HbA1c": rng.normal(6.0, 0.8, n).round(1),
        "Platelet": rng.integers(80, 320, n),
        "Steatosis_Grade": rng.integers(1, 4, n), "Lobular_Inflammation": rng.integers(0, 3, n),
        "Ballooning": rng.integers(0, 3, n), "NAS_Total": rng.integers(1, 8, n),
        "SAF_S": rng.integers(1, 4, n), "SAF_A": rng.integers(0, 5, n),
        "Impaired_glucose_tolerance": rng.integers(0, 2, n), "L_MAFLD": rng.integers(0, 2, n),
    })
    clin.to_csv(val / "clinical_data_clean.csv", index=False)

    event = rng.integers(0, 2, n)
    t = np.where(event == 1, rng.integers(1, 11, n), 10)
    lab = pd.DataFrame({
        "patient_id": ids, "baseline_diagnosis": "MASLD",
        "liver_progression_occurred": event,
        "event_type": np.where(event == 1, rng.choice(["Cirrhosis", "HCC", "Decompensation"], n), "None"),
        "time_to_event_years": t, "censored": 1 - event, "follow_up_years": 10,
    })
    lab.to_csv(val / "patient_labels.csv", index=False)

    for pid in ids:
        k = rng.integers(0, 6)
        days = np.sort(rng.integers(30, 1500, k))
        df = pd.DataFrame({"days_from_biopsy": days})
        for c in LABS:
            df[c] = rng.normal(clin.loc[clin.patient_id == pid, c].iloc[0], 5, k).round(1)
        df.to_csv(labs_dir / f"{pid}_follow_up_labs.csv", index=False)
        stem = pid.replace("_", "-")
        for stain in ("HE", "SR"):
            torch.save(torch.randn(1, 768), a.out / "embeddings" / f"val_{stain}" / f"{stem}-{stain}_embed.pt")
    print(f"wrote {n} synthetic patients to {a.out}")


if __name__ == "__main__":
    main()
