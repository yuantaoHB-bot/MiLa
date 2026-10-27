#!/usr/bin/env python3
"""
Re-run the external validation of all saved models on the 94-patient cohort.

This is the Mac/Linux entry point for reproducing the numbers reported in the
WJG submission (submission 7.26):

    MiLa (M5 labs_mask): 3-year AUROC 0.8446, 8-year AUROC 0.9020
    M4:                  3-year AUROC 0.7440, 8-year AUROC 0.8132
    M3:                  3-year AUROC 0.6458, 8-year AUROC 0.7147

It loads the 5 fold-specific checkpoints per model, averages their softmax
probabilities (the same ensemble used in the paper), and writes per-patient
predictions plus a metrics summary.

Usage
-----
    python run_validation.py                 # all models, CPU
    python run_validation.py --device mps    # Apple Silicon GPU (optional)
    python run_validation.py --models mila   # only MiLa

Outputs
-------
    outputs/validation_reproduced/
        m3_val_predictions.csv
        m4_val_predictions.csv
        m5_labs_mask_val_predictions.csv
        m5_lab_only_val_predictions.csv
        m5_mask_only_val_predictions.csv
        validation_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import MASLDDataset  # noqa: E402
from model_m3 import M3_WSIOnly  # noqa: E402
from model_m4 import M4_StaticMultimodal  # noqa: E402
from model_m5 import M5_FullTemporal  # noqa: E402

DATA_DIR = ROOT / "dataset"
WSI_DIR = ROOT / "embeddings"
MODEL_DIR = ROOT / "weights"
OUT_DIR = ROOT / "outputs" / "validation_reproduced"

# model_key -> (checkpoint subdir, checkpoint prefix, dataset variant, lab_dim)
MODEL_SPECS = {
    "m3": ("", "m3", None, None),
    "m4": ("", "m4", None, None),
    "m5_labs_mask": ("m5_labs_mask", "m5", "labs_mask", 12),
    "m5_lab_only": ("m5_lab_only", "m5", "lab_only", 11),
    "m5_mask_only": ("m5_mask_only", "m5", "mask_only", 1),
}

MODEL_ALIASES = {
    "mila": "m5_labs_mask",
    "m5": "m5_labs_mask",
    "labs_mask": "m5_labs_mask",
    "lab_only": "m5_lab_only",
    "mask_only": "m5_mask_only",
}

SEED = 42
N_BOOTSTRAP = 2000


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def bootstrap_auroc_ci(y_true: np.ndarray, y_score: np.ndarray,
                       n: int = N_BOOTSTRAP, seed: int = SEED) -> list[float]:
    """Percentile bootstrap 95% CI for AUROC."""
    rng = np.random.RandomState(seed)
    aucs: list[float] = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    return [round(float(np.percentile(aucs, 2.5)), 4),
            round(float(np.percentile(aucs, 97.5)), 4)]


def load_dataset(variant: str | None) -> MASLDDataset:
    """Load the external validation split with the inputs the model expects."""
    kwargs = dict(
        data_dir=str(DATA_DIR),
        split="val",
        use_wsi_features=True,
        wsi_feature_dir=str(WSI_DIR),
        use_followup_labs=variant is not None,
        use_clean_data=True,
    )
    if variant is not None:
        kwargs.update(
            followup_mode="annual_grid",
            annual_grid_years=2,
            annual_grid_variant=variant,
        )
    return MASLDDataset(**kwargs)


def collate(ds: MASLDDataset) -> dict:
    """Stack all samples into batch tensors."""
    wsi, clin, labs, pids = [], [], [], []
    y5, y10 = [], []
    for i in range(len(ds)):
        s = ds[i]
        wsi.append(s["wsi_feature"])
        clin.append(s["clinical"])
        pids.append(s["patient_id"])
        y5.append(int(s["label_5y"].item()))
        y10.append(int(s["label_10y"].item()))
        if "followup_labs" in s:
            labs.append(s["followup_labs"])
    out = {
        "wsi": torch.stack(wsi),
        "clin": torch.stack(clin),
        "pids": pids,
        "y5": np.asarray(y5),
        "y10": np.asarray(y10),
    }
    if labs:
        out["labs"] = torch.stack(labs)
    return out


def build_model(model_key: str, lab_dim: int | None):
    if model_key == "m3":
        return M3_WSIOnly(input_dim=768, hidden_dim=512, dropout=0.4)
    if model_key == "m4":
        return M4_StaticMultimodal(
            wsi_dim=768, clinical_dim=30, hidden_dim=512,
            fusion_dim=256, dropout=0.4,
        )
    return M5_FullTemporal(
        wsi_dim=768, clinical_dim=30, lab_dim=lab_dim, hidden_dim=512,
        fusion_dim=256, temporal_dim=128, lstm_hidden=128, dropout=0.4,
    )


def load_fold_models(model_key: str, lab_dim: int | None,
                     device: torch.device) -> list:
    subdir, prefix, _, _ = MODEL_SPECS[model_key]
    model_dir = MODEL_DIR / subdir if subdir else MODEL_DIR
    models = []
    for fold in range(1, 6):
        ckpt = model_dir / f"{prefix}_fold{fold}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        model = build_model(model_key, lab_dim)
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except TypeError:  # older PyTorch without weights_only
            state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models.append(model)
    return models


def ensemble_predict(models: list, batch: dict, model_key: str,
                     device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    wsi = batch["wsi"].to(device)
    clin = batch["clin"].to(device)
    labs = batch.get("labs")
    if labs is not None:
        labs = labs.to(device)

    probs_5y, probs_10y = [], []
    with torch.no_grad():
        for model in models:
            if model_key == "m3":
                out = model(wsi)
            elif model_key == "m4":
                out = model(wsi, clin)
            else:
                out = model(wsi, clin, labs)
            probs_5y.append(F.softmax(out["logits_5y"], dim=1)[:, 1].cpu())
            probs_10y.append(F.softmax(out["logits_10y"], dim=1)[:, 1].cpu())
    p5 = torch.stack(probs_5y).mean(dim=0).numpy()
    p10 = torch.stack(probs_10y).mean(dim=0).numpy()
    return p5, p10


def cirrhosis_labels(pids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Cirrhosis-only endpoint used in the sensitivity analysis."""
    labels = pd.read_csv(DATA_DIR / "val" / "patient_labels.csv").set_index("patient_id")
    c3, c8 = [], []
    for pid in pids:
        row = labels.loc[pid]
        is_cirr = row["event_type"] == "Cirrhosis"
        c3.append(int(is_cirr and row["time_to_event_years"] <= 5))
        c8.append(int(is_cirr and row["time_to_event_years"] <= 10))
    return np.asarray(c3), np.asarray(c8)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "auroc_ci": bootstrap_auroc_ci(y, p),
        "auprc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "events": int(y.sum()),
        "n": int(len(y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce external validation")
    parser.add_argument("--device", default="cpu",
                        help="cpu (default), mps (Apple Silicon), or cuda")
    parser.add_argument("--models", nargs="+", default=["m3", "m4", "m5_labs_mask"],
                        help="Model keys to run: m3 m4 m5_labs_mask (shipped); m5_lab_only m5_mask_only need their own checkpoints")
    parser.add_argument("--data-dir", type=Path, help="folder with val/clinical_data_clean.csv, val/patient_labels.csv, follow_up_labs_clean/")
    parser.add_argument("--embeddings-dir", type=Path, help="folder with val_HE/ and val_SR/ slide embeddings")
    parser.add_argument("--models-dir", type=Path, help="folder with the fold checkpoints (default: ./weights)")
    parser.add_argument("--out-dir", type=Path, help="output folder")
    args = parser.parse_args()
    global DATA_DIR, WSI_DIR, MODEL_DIR, OUT_DIR
    user_data = args.data_dir is not None
    DATA_DIR = args.data_dir or DATA_DIR
    WSI_DIR = args.embeddings_dir or WSI_DIR
    MODEL_DIR = args.models_dir or MODEL_DIR
    OUT_DIR = args.out_dir or OUT_DIR

    keys = [MODEL_ALIASES.get(k, k) for k in args.models]
    for k in keys:
        if k not in MODEL_SPECS:
            raise SystemExit(f"Unknown model: {k}. Choose from {list(MODEL_SPECS)}")

    set_seed(SEED)
    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for key in keys:
        _, _, variant, lab_dim = MODEL_SPECS[key]
        print(f"\n=== {key} ===")
        ds = load_dataset(variant)
        batch = collate(ds)
        print(f"  patients: {len(batch['pids'])}  |  events 3y/8y: "
              f"{batch['y5'].sum()}/{batch['y10'].sum()}")
        models = load_fold_models(key, lab_dim, device)
        p5, p10 = ensemble_predict(models, batch, key, device)

        c3, c8 = cirrhosis_labels(batch["pids"])
        row = {
            "composite_3y": metrics(batch["y5"], p5),
            "composite_8y": metrics(batch["y10"], p10),
            "cirrhosis_3y": metrics(c3, p5),
            "cirrhosis_8y": metrics(c8, p10),
        }
        summary[key] = row
        print(f"  composite 3y AUROC: {row['composite_3y']['auroc']:.4f} "
              f"" + ("(expected 0.8446 for MiLa on the study cohort)" if not user_data else ""))
        print(f"  composite 8y AUROC: {row['composite_8y']['auroc']:.4f} "
              f"" + ("(expected 0.9020 for MiLa on the study cohort)" if not user_data else ""))
        print(f"  cirrhosis 3y/8y: {row['cirrhosis_3y']['auroc']:.4f}/"
              f"{row['cirrhosis_8y']['auroc']:.4f}")

        pd.DataFrame({
            "patient_id": batch["pids"],
            "label_5y": batch["y5"],
            "label_10y": batch["y10"],
            "cirrhosis_3y": c3,
            "cirrhosis_8y": c8,
            "prob_5y": p5,
            "prob_10y": p10,
        }).to_csv(OUT_DIR / f"{key}_val_predictions.csv", index=False)

    with open(OUT_DIR / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {OUT_DIR}")
    if not user_data:
        print("Expected MiLa: composite 0.8446 / 0.9020; cirrhosis-only 0.8115 / 0.8242")


if __name__ == "__main__":
    main()
