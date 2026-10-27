#!/usr/bin/env python3
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""
Sensitivity analysis R1.6 — MiLa (M5 labs_mask) inference with TBIL, Albumin,
and INR zeroed out.

Reviewer concern: these three tests of hepatic synthetic function may act as
proxy markers for cirrhosis, creating circularity.  This script re-runs the
5-fold ensemble on the external validation set after zeroing those columns and
reports whether predictive performance is materially reduced.

Grid layout clarification
--------------------------
The model ingests a [B, T=2, 12] tensor.  The T dimension is annual time slots
(year 1 = days 0-365, year 2 = days 366-730).  Within each time slot:
  cols 0-10 : normalized lab values  (ALT, AST, GGT, TBIL, Albumin, Platelet,
                                      FBG, TG, TC, Creatinine, INR)
  col  11   : per-year observation mask  (1 = at least one visit that year)

Note: the observation mask in col 11 is per-year, NOT per-lab.  Zeroing the
three lab value columns (3=TBIL, 4=Albumin, 10=INR) is the correct exclusion;
col 11 is left untouched because removing it would suppress all temporal
information for that year, not just the excluded analytes.

Usage
-----
    conda activate YT_AEGIS
    python workspace/analysis/r16_variable_exclusion.py

Output
------
    workspace/supplementary/r16_variable_exclusion_results.json
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = f"{ROOT}"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from dataset import MASLDDataset          # noqa: E402
from model_m5 import M5_FullTemporal      # noqa: E402

DEVICE     = "cpu"
SEED       = 42
N_FOLDS    = 5
N_BOOTSTRAP = 2000

DATA_DIR   = os.path.join(PROJECT_ROOT, "dataset")
WSI_DIR    = os.path.join(PROJECT_ROOT, "outputs", "embeddings")
MODEL_DIR  = os.path.join(PROJECT_ROOT, "outputs", "models", "m5_labs_mask")
VAL_PRED_CSV = os.path.join(
    PROJECT_ROOT,
    "outputs", "models",
    "val_predictions_auc_target_20260509_153628",
    "m5_labs_mask_val_predictions.csv",
)
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "workspace", "supplementary",
    "r16_variable_exclusion_results.json"
)

# Columns to exclude (0-indexed within the 11 lab columns)
EXCLUDE_COLS   = [3, 4, 10]          # TBIL, Albumin, INR
EXCLUDE_NAMES  = ["TBIL", "Albumin", "INR"]

# Known standard-model AUROCs (from val_predictions_auc_target_20260509_153628)
STANDARD_COMPOSITE_3Y = 0.8446
STANDARD_COMPOSITE_8Y = 0.9020


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def load_val_dataset() -> MASLDDataset:
    """Load external validation set with WSI features and annual lab grid."""
    return MASLDDataset(
        data_dir=DATA_DIR,
        split="val",
        use_wsi_features=True,
        wsi_feature_dir=WSI_DIR,
        use_followup_labs=True,
        use_clean_data=True,
        followup_mode="annual_grid",
        annual_grid_years=2,           # T = 2 annual slots
        annual_grid_variant="labs_mask",
    )


def collate_all(ds: MASLDDataset):
    """Return (wsi, clin, labs, patient_ids) tensors over the full dataset."""
    wsi_list, clin_list, labs_list, pids = [], [], [], []
    for i in range(len(ds)):
        s = ds[i]
        wsi_list.append(s["wsi_feature"])
        clin_list.append(s["clinical"])
        labs_list.append(s["followup_labs"])
        pids.append(s["patient_id"])
    wsi  = torch.stack(wsi_list)   # [N, 768]
    clin = torch.stack(clin_list)  # [N, 30]
    labs = torch.stack(labs_list)  # [N, 2, 12]
    return wsi, clin, labs, pids


def load_composite_labels(pids: list):
    """Load composite endpoint labels from canonical prediction CSV."""
    df = pd.read_csv(VAL_PRED_CSV).set_index("patient_id")
    label_5y  = np.array([int(df.loc[pid, "label_5y"])  for pid in pids])
    label_10y = np.array([int(df.loc[pid, "label_10y"]) for pid in pids])
    return label_5y, label_10y


def load_cirrhosis_labels(pids: list):
    """
    Build cirrhosis-only endpoint labels from val patient_labels.csv.
    cirrhosis_3y: event_type==Cirrhosis AND time_to_event_years <= 5
    cirrhosis_8y: event_type==Cirrhosis AND time_to_event_years <= 10
    (Matching the internal label_5y / label_10y naming convention.)
    """
    labels = pd.read_csv(
        os.path.join(DATA_DIR, "val", "patient_labels.csv")
    ).set_index("patient_id")
    circ_3y = np.array([
        int(
            labels.loc[pid, "event_type"] == "Cirrhosis"
            and labels.loc[pid, "time_to_event_years"] <= 5
        )
        for pid in pids
    ])
    circ_8y = np.array([
        int(
            labels.loc[pid, "event_type"] == "Cirrhosis"
            and labels.loc[pid, "time_to_event_years"] <= 10
        )
        for pid in pids
    ])
    return circ_3y, circ_8y


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def build_model() -> M5_FullTemporal:
    """
    Instantiate M5_FullTemporal with the same hyperparameters used during
    training (see outputs/models/m5_labs_mask/m5_results.json).
    """
    return M5_FullTemporal(
        wsi_dim=768,
        clinical_dim=30,
        lab_dim=12,      # 11 lab values + 1 per-year observation mask
        hidden_dim=512,
        fusion_dim=256,
        temporal_dim=128,
        lstm_hidden=128,
        dropout=0.4,
    )


def load_fold_models() -> list:
    """Load and return all 5 fold checkpoints (eval mode, CPU)."""
    models = []
    for fold in range(1, N_FOLDS + 1):
        ckpt = os.path.join(MODEL_DIR, f"m5_fold{fold}.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        m = build_model()
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        # Checkpoints may be saved as {"model_state_dict": ...} or bare state dict
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        m.load_state_dict(state)
        m.eval()
        models.append(m)
        print(f"  Loaded fold {fold}: {ckpt}")
    return models


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def ensemble_predict(
    models: list,
    wsi: torch.Tensor,
    clin: torch.Tensor,
    labs: torch.Tensor,
) -> tuple:
    """
    Average softmax probabilities from all fold models.

    Returns
    -------
    prob_5y  : np.ndarray  [N]
    prob_10y : np.ndarray  [N]
    """
    all_p5, all_p10 = [], []
    with torch.no_grad():
        for m in models:
            out  = m(wsi, clin, labs)
            p5   = F.softmax(out["logits_5y"],  dim=1)[:, 1]
            p10  = F.softmax(out["logits_10y"], dim=1)[:, 1]
            all_p5.append(p5)
            all_p10.append(p10)
    prob_5y  = torch.stack(all_p5).mean(dim=0).numpy()
    prob_10y = torch.stack(all_p10).mean(dim=0).numpy()
    return prob_5y, prob_10y


def apply_exclusion(labs: torch.Tensor, exclude_cols: list) -> torch.Tensor:
    """
    Return a copy of the lab tensor with specified lab-value columns zeroed.

    Parameters
    ----------
    labs        : [N, T, 12]  — original annual grid
    exclude_cols: column indices within the 11-lab positions to zero

    The per-year observation mask (col 11) is intentionally preserved: it
    indicates whether any labs were drawn that year, independent of which
    specific analytes are modelled.
    """
    labs_mod = labs.clone()
    for col in exclude_cols:
        labs_mod[:, :, col] = 0.0    # zero across all T time slots
    return labs_mod


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------
def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n: int = N_BOOTSTRAP,
    seed: int = SEED,
    alpha: float = 0.05,
) -> list:
    """Return [lower, upper] 95% CI for AUROC via bootstrap resampling."""
    rng  = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    lower = float(np.percentile(aucs, 100 * alpha / 2))
    upper = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return [round(lower, 4), round(upper, 4)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 1. Load validation data ---------------------------------------------------
    print("Loading external validation dataset …")
    ds   = load_val_dataset()
    wsi, clin, labs_orig, pids = collate_all(ds)
    N = wsi.shape[0]
    print(f"  {N} patients  |  lab grid shape: {list(labs_orig.shape)}")

    # 2. Labels ----------------------------------------------------------------
    label_5y,  label_10y  = load_composite_labels(pids)
    circ_3y,   circ_8y    = load_cirrhosis_labels(pids)
    print(f"  Composite  3y events: {label_5y.sum()}   8y events: {label_10y.sum()}")
    print(f"  Cirrhosis  3y events: {circ_3y.sum()}    8y events: {circ_8y.sum()}")

    # 3. Load 5-fold checkpoints -----------------------------------------------
    print("\nLoading 5-fold M5 checkpoints …")
    models = load_fold_models()

    # 4. Standard-model inference (sanity check) --------------------------------
    print("\nRunning standard (unmodified) inference …")
    prob_5y_std, prob_10y_std = ensemble_predict(models, wsi, clin, labs_orig)
    auroc_std_3y  = round(float(roc_auc_score(label_5y,  prob_5y_std)),  4)
    auroc_std_8y  = round(float(roc_auc_score(label_10y, prob_10y_std)), 4)
    print(f"  Sanity-check AUROC  3y: {auroc_std_3y}  (expected ~{STANDARD_COMPOSITE_3Y})")
    print(f"  Sanity-check AUROC  8y: {auroc_std_8y}  (expected ~{STANDARD_COMPOSITE_8Y})")

    # 5. Build modified lab grid (zero out TBIL, Albumin, INR) -----------------
    print(f"\nApplying exclusion: zeroing columns {EXCLUDE_COLS} "
          f"({', '.join(EXCLUDE_NAMES)}) in both annual time slots …")
    labs_mod = apply_exclusion(labs_orig, EXCLUDE_COLS)

    # Report fraction of non-zero entries removed as a sanity check
    n_affected = (labs_orig[:, :, EXCLUDE_COLS] != 0).sum().item()
    print(f"  Non-zero entries zeroed: {n_affected}")

    # 6. Excluded-model inference -----------------------------------------------
    print("\nRunning inference with excluded features …")
    prob_5y_ex, prob_10y_ex = ensemble_predict(models, wsi, clin, labs_mod)

    # Composite endpoint
    auroc_ex_3y = round(float(roc_auc_score(label_5y,  prob_5y_ex)),  4)
    auroc_ex_8y = round(float(roc_auc_score(label_10y, prob_10y_ex)), 4)
    print(f"  Excluded AUROC  3y (composite): {auroc_ex_3y}")
    print(f"  Excluded AUROC  8y (composite): {auroc_ex_8y}")

    # Cirrhosis-only endpoint
    auroc_circ_3y = round(float(roc_auc_score(circ_3y, prob_5y_ex)),  4)
    auroc_circ_8y = round(float(roc_auc_score(circ_8y, prob_10y_ex)), 4)
    print(f"  Excluded AUROC  3y (cirrhosis only): {auroc_circ_3y}")
    print(f"  Excluded AUROC  8y (cirrhosis only): {auroc_circ_8y}")

    # 7. Bootstrap 95% CIs ------------------------------------------------------
    print(f"\nComputing bootstrap CIs ({N_BOOTSTRAP} resamples) …")
    ci_ex_3y   = bootstrap_auroc_ci(label_5y,  prob_5y_ex)
    ci_ex_8y   = bootstrap_auroc_ci(label_10y, prob_10y_ex)
    ci_circ_3y = bootstrap_auroc_ci(circ_3y,   prob_5y_ex)
    ci_circ_8y = bootstrap_auroc_ci(circ_8y,   prob_10y_ex)

    # 8. Delta (absolute change from standard model) ---------------------------
    delta_3y = round(auroc_ex_3y - STANDARD_COMPOSITE_3Y, 4)
    delta_8y = round(auroc_ex_8y - STANDARD_COMPOSITE_8Y, 4)

    # 9. Compose interpretation ------------------------------------------------
    direction_3y = "decreased" if delta_3y < 0 else "increased"
    direction_8y = "decreased" if delta_8y < 0 else "increased"

    interpretation = (
        f"Zeroing TBIL, Albumin, and INR (hepatic synthetic function markers) "
        f"across both annual time slots {direction_3y} the 3-year composite AUROC "
        f"by {abs(delta_3y):.4f} (from {STANDARD_COMPOSITE_3Y} to {auroc_ex_3y}) "
        f"and {direction_8y} the 8-year composite AUROC "
        f"by {abs(delta_8y):.4f} (from {STANDARD_COMPOSITE_8Y} to {auroc_ex_8y}). "
        f"The cirrhosis-only AUROCs were {auroc_circ_3y} (3y) and {auroc_circ_8y} (8y). "
        f"These results suggest that the three hepatic synthetic function markers "
        f"{'contribute meaningfully' if abs(delta_3y) > 0.02 or abs(delta_8y) > 0.02 else 'do not drive'} "
        f"model performance, and that "
        f"{'circularity with the cirrhosis endpoint is a concern' if auroc_circ_3y > 0.80 or auroc_circ_8y > 0.85 else 'the model does not rely on circularity with the cirrhosis endpoint'}."
    )

    # 10. Save results ----------------------------------------------------------
    results = {
        "description": (
            "Sensitivity analysis R1.6: MiLa inference with TBIL, Albumin, INR "
            "zeroed out.  Addresses reviewer concern about circularity between "
            "hepatic synthetic function markers and the cirrhosis endpoint."
        ),
        "excluded_features": EXCLUDE_NAMES,
        "excluded_col_indices": EXCLUDE_COLS,
        "grid_layout_note": (
            "Grid shape is [N, T=2, 12] where cols 0-10 are normalized lab values "
            "(ALT,AST,GGT,TBIL,Albumin,Platelet,FBG,TG,TC,Creatinine,INR) and "
            "col 11 is the per-year observation mask.  Cols 3,4,10 were zeroed "
            "across both annual time slots; col 11 was preserved."
        ),
        "n_val_patients": int(N),
        "results": {
            "MiLa_standard_composite_3y": STANDARD_COMPOSITE_3Y,
            "MiLa_standard_composite_8y": STANDARD_COMPOSITE_8Y,
            "MiLa_sanitycheck_composite_3y": auroc_std_3y,
            "MiLa_sanitycheck_composite_8y": auroc_std_8y,
            "MiLa_excluded_composite_3y":   auroc_ex_3y,
            "MiLa_excluded_composite_8y":   auroc_ex_8y,
            "MiLa_excluded_cirrhosis_3y":   auroc_circ_3y,
            "MiLa_excluded_cirrhosis_8y":   auroc_circ_8y,
            "MiLa_excluded_3y_ci":          ci_ex_3y,
            "MiLa_excluded_8y_ci":          ci_ex_8y,
            "MiLa_excluded_cirrhosis_3y_ci": ci_circ_3y,
            "MiLa_excluded_cirrhosis_8y_ci": ci_circ_8y,
            "delta_composite_3y_vs_standard": delta_3y,
            "delta_composite_8y_vs_standard": delta_8y,
        },
        "interpretation": interpretation,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("\n=== Summary ===")
    print(f"  Standard model     3y / 8y : {STANDARD_COMPOSITE_3Y} / {STANDARD_COMPOSITE_8Y}")
    print(f"  Excluded TBIL/Alb/INR 3y/8y: {auroc_ex_3y} / {auroc_ex_8y}  "
          f"(delta {delta_3y:+.4f} / {delta_8y:+.4f})")
    print(f"  Cirrhosis-only     3y / 8y : {auroc_circ_3y} / {auroc_circ_8y}")


if __name__ == "__main__":
    main()

