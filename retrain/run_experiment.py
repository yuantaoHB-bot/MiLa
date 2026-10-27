#!/usr/bin/env python3
"""
Run one R1 retraining job: one experiment from experiments.json x one seed.

The training loop is a line-by-line copy of src/train.py::train_model (same
seeding, StratifiedKFold split, DataLoader settings, epoch loop, early stopping
and checkpoint rule). Experiment options only change the inputs:

    clinical feature set   add_features / drop_features / extra_drop
    longitudinal columns   drop_labs / extra_drop
    longitudinal content   lab_mode = "zeros" | "last_only"
    training population    exclude_train

On top of the original outputs it saves out-of-fold predictions (from each
fold's selected checkpoint) and the 5-fold-ensemble external-validation
predictions. Extra prediction passes run without a DataLoader and inside a
saved/restored torch RNG state, so they cannot shift the random stream that
the next fold consumes.

Usage
-----
    python run_experiment.py --exp ref_mila --seed 42
    python run_experiment.py --exp ref_mila --seed 42 --out-root runs_repro --threads 0
"""
import argparse
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from dataset import MASLDDataset  # noqa: E402
from model_m4 import M4_StaticMultimodal, M4_Trainer  # noqa: E402
from model_m5 import M5_FullTemporal, M5_Trainer  # noqa: E402
from train import set_seed, build_stratification_labels, summarize_fold_distribution  # noqa: E402

LAB_COLS = ["ALT", "AST", "GGT", "TBIL", "Albumin", "Platelet", "FBG", "TG", "TC", "Creatinine", "INR"]
GRID_COLS = {"labs_mask": LAB_COLS + ["mask"], "lab_only": LAB_COLS, "mask_only": ["mask"]}


class ExperimentDataset(MASLDDataset):
    """MASLDDataset with a configurable feature set and longitudinal grid."""

    def __init__(self, *, clinical_features, drop_labs=(), lab_mode=None, exclude_ids=(), **kwargs):
        super().__init__(**kwargs)
        if "Fibrosis_Stage" in clinical_features and "Fibrosis_Stage" not in self.data.columns:
            extra = pd.read_csv(os.path.join(kwargs["data_dir"], kwargs["split"], "clinical_data.csv"))
            n = len(self.data)
            self.data = self.data.merge(extra[["patient_id", "Fibrosis_Stage"]], on="patient_id",
                                        how="left", validate="one_to_one")
            assert len(self.data) == n and self.data["Fibrosis_Stage"].notna().all()
        if exclude_ids:
            found = set(self.data["patient_id"]) & set(exclude_ids)
            assert found == set(exclude_ids), f"exclusion IDs not found: {set(exclude_ids) - found}"
            self.data = self.data[~self.data["patient_id"].isin(exclude_ids)].reset_index(drop=True)
        self.patient_ids = self.data["patient_id"].tolist()
        missing = set(clinical_features) - set(self.data.columns)
        assert not missing, f"missing clinical columns: {missing}"
        self.clinical_features = list(clinical_features)
        self.lab_mode = lab_mode
        cols = GRID_COLS[self.annual_grid_variant] if self.use_followup_labs else []
        assert set(drop_labs) <= set(cols), f"cannot drop {set(drop_labs) - set(cols)}"
        self.keep_lab_idx = [i for i, c in enumerate(cols) if c not in set(drop_labs)]
        self.lab_cols = [cols[i] for i in self.keep_lab_idx]

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if "followup_labs" in item:
            grid = item["followup_labs"]
            if self.lab_mode == "zeros":
                grid = torch.zeros_like(grid)
            elif self.lab_mode == "last_only":
                grid = grid[-1:].repeat(grid.shape[0], 1)
            item["followup_labs"] = grid[:, self.keep_lab_idx]
        return item


def resolve(cfg, plan):
    """Turn one experiments.json entry into concrete feature and lab lists."""
    feats = list(plan["feature_sets"]["base30"])
    drop = list(plan[cfg["drop_features"]]) if "drop_features" in cfg else []
    drop += cfg.get("extra_drop", [])
    feats = [f for f in feats if f not in drop] + list(cfg.get("add_features", []))
    drop_labs = list(plan[cfg["drop_labs"]]) if "drop_labs" in cfg else []
    drop_labs += [c for c in cfg.get("extra_drop", []) if c in LAB_COLS]
    exclude = list(plan[cfg["exclude_train"]]) if "exclude_train" in cfg else []
    return feats, drop_labs, exclude


def make_dataset(cfg, split, feats, drop_labs, exclude, data_dir, wsi_dir):
    use_labs = cfg["model"] == "m5"
    return ExperimentDataset(
        clinical_features=feats,
        drop_labs=drop_labs if use_labs else (),
        lab_mode=cfg.get("lab_mode"),
        exclude_ids=exclude if split == "train" else (),
        data_dir=data_dir,
        split=split,
        use_wsi_features=True,
        wsi_feature_dir=wsi_dir,
        use_followup_labs=use_labs,
        use_clean_data=True,
        followup_mode="annual_grid",
        annual_grid_years=2,
        annual_grid_variant=cfg.get("variant", "labs_mask"),
    )


def build_model(cfg, clinical_dim, lab_dim, device, lr):
    if cfg["model"] == "m4":
        model = M4_StaticMultimodal(wsi_dim=768, clinical_dim=clinical_dim, hidden_dim=512,
                                    fusion_dim=256, dropout=0.4)
        return model, M4_Trainer(model, device=device, lr=lr)
    model = M5_FullTemporal(wsi_dim=768, clinical_dim=clinical_dim, lab_dim=lab_dim, hidden_dim=512,
                            fusion_dim=256, temporal_dim=128, lstm_hidden=128, dropout=0.4)
    return model, M5_Trainer(model, device=device, lr=lr)


def predict(model, dataset, indices, device, is_m5, batch_size=64):
    """Deterministic prediction pass without a DataLoader; leaves the torch RNG untouched."""
    rng_state = torch.get_rng_state()
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            items = [dataset[i] for i in indices[start:start + batch_size]]
            wsi = torch.stack([s["wsi_feature"] for s in items]).to(device)
            clin = torch.stack([s["clinical"] for s in items]).to(device)
            if is_m5:
                labs = torch.stack([s["followup_labs"] for s in items]).to(device)
                out = model(wsi, clin, labs)
            else:
                out = model(wsi, clin)
            p5 = F.softmax(out["logits_5y"], dim=1)[:, 1].cpu().numpy()
            p10 = F.softmax(out["logits_10y"], dim=1)[:, 1].cpu().numpy()
            for s, a, b in zip(items, p5, p10):
                rows.append({"patient_id": s["patient_id"], "label_5y": int(s["label_5y"]),
                             "label_10y": int(s["label_10y"]), "prob_5y": float(a), "prob_10y": float(b)})
    torch.set_rng_state(rng_state)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--plan", default=str(HERE / "experiments.json"))
    ap.add_argument("--data-dir", default=str(HERE.parent / "Mac_transfer_20261026" / "dataset"))
    ap.add_argument("--wsi-dir", default=str(HERE.parent / "Mac_transfer_20261026" / "outputs" / "embeddings"))
    ap.add_argument("--out-root", default=str(HERE / "runs"))
    ap.add_argument("--threads", type=int, default=0, help="torch intra-op threads; 0 = torch default")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (original: 4)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    cfg = plan["experiments"][args.exp]
    out_dir = Path(args.out_root) / args.exp / f"seed{args.seed}"
    if (out_dir / "DONE").exists():
        print(f"skip {out_dir}: already done")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    n_folds, num_epochs, batch_size, lr, device, seed = 5, 60, 16, 1e-4, args.device, args.seed
    is_m5 = cfg["model"] == "m5"
    feats, drop_labs, exclude = resolve(cfg, plan)
    t0 = time.time()

    # ---- copied from src/train.py::train_model (seeding, CV split, loop, early stopping) ----
    set_seed(seed)
    dataset = make_dataset(cfg, "train", feats, drop_labs, exclude, args.data_dir, args.wsi_dir)
    lab_dim = len(dataset.lab_cols) if is_m5 else None

    stratification_labels = build_stratification_labels(dataset.data)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results, epoch_log, oof = [], [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(stratification_labels)), stratification_labels)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        fold_distribution = summarize_fold_distribution(dataset.data, train_idx, val_idx)

        model, trainer = build_model(cfg, len(feats), lab_dim, device, lr)
        n_params = sum(p.numel() for p in model.parameters())
        save_path = out_dir / f"{cfg['model']}_fold{fold + 1}.pth"

        best_auroc_5y = 0.0
        best_auroc_10y = 0.0
        best_epoch = None
        patience_counter = 0
        patience = 10

        for epoch in range(num_epochs):
            train_metrics = trainer.train_epoch(train_loader, epoch)
            val_metrics = trainer.evaluate(val_loader)
            epoch_log.append({"fold": fold + 1, "epoch": epoch + 1, "train_loss": train_metrics["loss"],
                              "val_loss": val_metrics["loss"], "val_auroc_5y": val_metrics["auroc_5y"],
                              "val_auroc_10y": val_metrics["auroc_10y"]})
            avg_auroc = (val_metrics["auroc_5y"] + val_metrics["auroc_10y"]) / 2
            if avg_auroc > (best_auroc_5y + best_auroc_10y) / 2:
                best_auroc_5y = val_metrics["auroc_5y"]
                best_auroc_10y = val_metrics["auroc_10y"]
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        # ---- end of copied loop body ----

        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        fold_pred = predict(model, dataset, list(val_idx), device, is_m5)
        fold_pred.insert(1, "fold", fold + 1)
        oof.append(fold_pred)
        fold_results.append({"fold": fold + 1, "auroc_5y": best_auroc_5y, "auroc_10y": best_auroc_10y,
                             "best_epoch": best_epoch, "epochs_run": epoch + 1, "n_params": n_params,
                             "model_path": save_path.name, "class_distribution": fold_distribution})
        print(f"[{args.exp} seed{seed}] fold {fold + 1}: best epoch {best_epoch}, "
              f"AUROC 3y {best_auroc_5y:.4f} 8y {best_auroc_10y:.4f}", flush=True)

    # external validation: average the softmax of the 5 fold checkpoints
    val_ds = make_dataset(cfg, "val", feats, drop_labs, exclude, args.data_dir, args.wsi_dir)
    idx = list(range(len(val_ds)))
    per_fold = []
    for k in range(1, n_folds + 1):
        model, _ = build_model(cfg, len(feats), lab_dim, device, lr)
        model.load_state_dict(torch.load(out_dir / f"{cfg['model']}_fold{k}.pth", map_location=device, weights_only=True))
        per_fold.append(predict(model, val_ds, idx, device, is_m5))
    val = per_fold[0][["patient_id", "label_5y", "label_10y"]].copy()
    val["prob_5y"] = np.mean([p["prob_5y"].values for p in per_fold], axis=0)
    val["prob_10y"] = np.mean([p["prob_10y"].values for p in per_fold], axis=0)
    for k, p in enumerate(per_fold, 1):
        val[f"prob_5y_fold{k}"] = p["prob_5y"].values
        val[f"prob_10y_fold{k}"] = p["prob_10y"].values

    oof = pd.concat(oof, ignore_index=True)
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    val.to_csv(out_dir / "val_predictions.csv", index=False)
    pd.DataFrame(epoch_log).to_csv(out_dir / "epoch_log.csv", index=False)

    summary = {
        "exp": args.exp, "seed": seed, "model": cfg["model"], "comment": cfg.get("comment"),
        "clinical_features": feats, "lab_cols": dataset.lab_cols if is_m5 else None,
        "lab_mode": cfg.get("lab_mode"), "variant": cfg.get("variant"), "excluded_train_ids": exclude,
        "n_train": len(dataset), "n_val": len(val_ds), "clinical_dim": len(feats), "lab_dim": lab_dim,
        "n_params": fold_results[0]["n_params"],
        "cv_auroc_5y_mean": float(np.mean([r["auroc_5y"] for r in fold_results])),
        "cv_auroc_10y_mean": float(np.mean([r["auroc_10y"] for r in fold_results])),
        "oof_auroc_5y": float(roc_auc_score(oof["label_5y"], oof["prob_5y"])),
        "oof_auroc_10y": float(roc_auc_score(oof["label_10y"], oof["prob_10y"])),
        "val_auroc_5y": float(roc_auc_score(val["label_5y"], val["prob_5y"])),
        "val_auroc_10y": float(roc_auc_score(val["label_10y"], val["prob_10y"])),
        "fold_results": fold_results,
        "settings": {"n_folds": n_folds, "num_epochs": num_epochs, "batch_size": batch_size, "lr": lr,
                     "patience": 10, "device": device, "num_workers": args.num_workers,
                     "torch_threads": torch.get_num_threads(), "threads_arg": args.threads},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "sklearn": __import__("sklearn").__version__, "host": socket.gethostname(),
                        "platform": platform.platform()},
        "data_dir": args.data_dir, "wsi_dir": args.wsi_dir,
        "started": datetime.fromtimestamp(t0).isoformat(), "finished": datetime.now().isoformat(),
        "duration_sec": round(time.time() - t0, 1),
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2, ensure_ascii=False)
    (out_dir / "DONE").write_text(datetime.now().isoformat())
    print(f"[{args.exp} seed{seed}] external AUROC 3y {summary['val_auroc_5y']:.4f} "
          f"8y {summary['val_auroc_10y']:.4f}  ({summary['duration_sec']} s)", flush=True)


if __name__ == "__main__":
    main()
