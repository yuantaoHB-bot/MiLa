#!/usr/bin/env python
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""
Permutation test for M5 labs_mask model.

Loads the 5-fold M5 checkpoints and runs inference on the external
validation set under three conditions:
  - "original"      : no permutation (baseline)
  - "permute_labs"  : randomly shuffle lab values across patients (keep real masks)
  - "permute_mask"  : randomly shuffle mask values across patients (keep real labs)

For each condition the 5-fold ensemble probability (mean) is computed, then
AUROC is calculated for 3-year and 8-year endpoints.  Permutation conditions
are repeated 100 times; mean +/- std are reported.

Outputs: workspace/supplementary/permutation_test.json
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

from dataset import MASLDDataset
from model_m5 import M5_FullTemporal

SEED = 42
N_PERMUTATIONS = 100
N_FOLDS = 5
DEVICE = "cpu"

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
WSI_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings")
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models", "m5_labs_mask")
VAL_PRED_CSV = os.path.join(
    PROJECT_ROOT,
    "outputs", "models",
    "val_predictions_auc_target_20260509_153628",
    "m5_labs_mask_val_predictions.csv",
)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "workspace", "supplementary", "permutation_test.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_val_dataset() -> MASLDDataset:
    """Load the external validation dataset with WSI + follow-up labs."""
    ds = MASLDDataset(
        data_dir=DATA_DIR,
        split="val",
        use_wsi_features=True,
        wsi_feature_dir=WSI_DIR,
        use_followup_labs=True,
        use_clean_data=True,
        followup_mode="annual_grid",
        annual_grid_years=2,
        annual_grid_variant="labs_mask",
    )
    return ds


def collate_all(ds: MASLDDataset):
    """Collate the entire dataset into batched tensors."""
    wsi_list, clin_list, labs_list = [], [], []
    pids = []
    for i in range(len(ds)):
        sample = ds[i]
        wsi_list.append(sample["wsi_feature"])
        clin_list.append(sample["clinical"])
        labs_list.append(sample["followup_labs"])
        pids.append(sample["patient_id"])
    wsi = torch.stack(wsi_list)        # [N, 768]
    clin = torch.stack(clin_list)      # [N, 30]
    labs = torch.stack(labs_list)       # [N, 2, 12]  (11 labs + 1 mask)
    return wsi, clin, labs, pids


def load_labels(pids):
    """Load ground-truth labels from the val predictions CSV (canonical)."""
    df = pd.read_csv(VAL_PRED_CSV)
    df = df.set_index("patient_id")
    label_5y = np.array([int(df.loc[pid, "label_5y"]) for pid in pids])
    label_10y = np.array([int(df.loc[pid, "label_10y"]) for pid in pids])
    return label_5y, label_10y


def build_model() -> M5_FullTemporal:
    return M5_FullTemporal(
        wsi_dim=768,
        clinical_dim=30,
        lab_dim=12,   # 11 labs + 1 mask
        hidden_dim=512,
        fusion_dim=256,
        temporal_dim=128,
        lstm_hidden=128,
        dropout=0.4,
    )


def ensemble_predict(models, wsi, clin, labs):
    """Average softmax probabilities across fold models."""
    probs_5y_all, probs_10y_all = [], []
    for m in models:
        m.eval()
        with torch.no_grad():
            out = m(wsi, clin, labs)
            p5 = F.softmax(out["logits_5y"], dim=1)[:, 1]
            p10 = F.softmax(out["logits_10y"], dim=1)[:, 1]
            probs_5y_all.append(p5)
            probs_10y_all.append(p10)
    prob_5y = torch.stack(probs_5y_all).mean(dim=0).numpy()
    prob_10y = torch.stack(probs_10y_all).mean(dim=0).numpy()
    return prob_5y, prob_10y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    print("Loading validation dataset ...")
    ds = load_val_dataset()
    wsi, clin, labs_full, pids = collate_all(ds)
    label_5y, label_10y = load_labels(pids)
    N = wsi.shape[0]
    print(f"  {N} patients loaded.")

    # Load 5-fold models
    print("Loading 5-fold M5 checkpoints ...")
    models = []
    for fold in range(1, N_FOLDS + 1):
        ckpt_path = os.path.join(MODEL_DIR, f"m5_fold{fold}.pth")
        m = build_model()
        m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        m.to(DEVICE)
        m.eval()
        models.append(m)
        print(f"  Fold {fold} loaded from {ckpt_path}")

    results = {}

    # ---- Condition 1: original (no permutation) ----
    print("\n[1/3] Original (no permutation) ...")
    prob_5y, prob_10y = ensemble_predict(models, wsi, clin, labs_full)
    auc_5y = roc_auc_score(label_5y, prob_5y)
    auc_10y = roc_auc_score(label_10y, prob_10y)
    results["original"] = {
        "auroc_3y": round(float(auc_5y), 4),
        "auroc_8y": round(float(auc_10y), 4),
    }
    print(f"  3y AUROC = {auc_5y:.4f},  8y AUROC = {auc_10y:.4f}")

    # ---- Condition 2: permute_labs (shuffle lab values, keep masks) ----
    print(f"\n[2/3] Permute labs ({N_PERMUTATIONS} repetitions) ...")
    auc_5y_perm_labs, auc_10y_perm_labs = [], []
    rng = np.random.RandomState(SEED)
    for rep in range(N_PERMUTATIONS):
        perm_idx = rng.permutation(N)
        labs_perm = labs_full.clone()
        # labs_full shape: [N, 2, 12] — first 11 cols are labs, last col is mask
        labs_perm[:, :, :11] = labs_full[perm_idx, :, :11]   # shuffled labs
        # mask stays from original patient
        p5, p10 = ensemble_predict(models, wsi, clin, labs_perm)
        auc_5y_perm_labs.append(roc_auc_score(label_5y, p5))
        auc_10y_perm_labs.append(roc_auc_score(label_10y, p10))
        if (rep + 1) % 20 == 0:
            print(f"    rep {rep+1}/{N_PERMUTATIONS}")

    results["permute_labs"] = {
        "auroc_3y_mean": round(float(np.mean(auc_5y_perm_labs)), 4),
        "auroc_3y_std": round(float(np.std(auc_5y_perm_labs)), 4),
        "auroc_8y_mean": round(float(np.mean(auc_10y_perm_labs)), 4),
        "auroc_8y_std": round(float(np.std(auc_10y_perm_labs)), 4),
        "auroc_3y_all": [round(float(v), 4) for v in auc_5y_perm_labs],
        "auroc_8y_all": [round(float(v), 4) for v in auc_10y_perm_labs],
    }
    print(f"  3y AUROC = {np.mean(auc_5y_perm_labs):.4f} +/- {np.std(auc_5y_perm_labs):.4f}")
    print(f"  8y AUROC = {np.mean(auc_10y_perm_labs):.4f} +/- {np.std(auc_10y_perm_labs):.4f}")

    # ---- Condition 3: permute_mask (shuffle mask, keep labs) ----
    print(f"\n[3/3] Permute mask ({N_PERMUTATIONS} repetitions) ...")
    auc_5y_perm_mask, auc_10y_perm_mask = [], []
    rng2 = np.random.RandomState(SEED + 1)
    for rep in range(N_PERMUTATIONS):
        perm_idx = rng2.permutation(N)
        labs_perm = labs_full.clone()
        labs_perm[:, :, 11:] = labs_full[perm_idx, :, 11:]   # shuffled mask
        # labs stay from original patient
        p5, p10 = ensemble_predict(models, wsi, clin, labs_perm)
        auc_5y_perm_mask.append(roc_auc_score(label_5y, p5))
        auc_10y_perm_mask.append(roc_auc_score(label_10y, p10))
        if (rep + 1) % 20 == 0:
            print(f"    rep {rep+1}/{N_PERMUTATIONS}")

    results["permute_mask"] = {
        "auroc_3y_mean": round(float(np.mean(auc_5y_perm_mask)), 4),
        "auroc_3y_std": round(float(np.std(auc_5y_perm_mask)), 4),
        "auroc_8y_mean": round(float(np.mean(auc_10y_perm_mask)), 4),
        "auroc_8y_std": round(float(np.std(auc_10y_perm_mask)), 4),
        "auroc_3y_all": [round(float(v), 4) for v in auc_5y_perm_mask],
        "auroc_8y_all": [round(float(v), 4) for v in auc_10y_perm_mask],
    }
    print(f"  3y AUROC = {np.mean(auc_5y_perm_mask):.4f} +/- {np.std(auc_5y_perm_mask):.4f}")
    print(f"  8y AUROC = {np.mean(auc_10y_perm_mask):.4f} +/- {np.std(auc_10y_perm_mask):.4f}")

    # ---- Summary ----
    results["metadata"] = {
        "n_patients": N,
        "n_folds": N_FOLDS,
        "n_permutations": N_PERMUTATIONS,
        "seed": SEED,
        "device": DEVICE,
        "model_dir": MODEL_DIR,
        "note": "label_5y = 3-year post-landmark; label_10y = 8-year post-landmark",
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
