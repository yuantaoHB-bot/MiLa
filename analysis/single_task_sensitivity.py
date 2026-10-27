#!/usr/bin/env python3
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""E4: Single-task vs multi-task sensitivity analysis.

Trains M5 with only one classification head active at a time:
- 3y-only: loss = loss_5y
- 8y-only: loss = loss_10y
- multi-task: loss = loss_5y + loss_10y (original, for reference)

Compares external validation AUROC to see if multi-task hurts or helps.
"""
import sys
sys.path.insert(0, f"{ROOT}/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from dataset import MASLDDataset
from model_m5 import M5_FullTemporal
from train import build_stratification_labels

BASE = Path(f"{ROOT}")
SEED = 42
N_FOLDS = 5
DEVICE = 'cpu'


def train_fold(model, train_loader, val_loader, task_mode='multi',
               n_epochs=60, patience=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    best_auroc = 0
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            wsi = batch['wsi_feature'].to(DEVICE)
            clin = batch['clinical'].to(DEVICE)
            labs = batch['followup_labs'].to(DEVICE)
            y5 = batch['label_5y'].to(DEVICE)
            y10 = batch['label_10y'].to(DEVICE)

            out = model(wsi, clin, labs)

            if task_mode == '3y_only':
                loss = criterion(out['logits_5y'], y5)
            elif task_mode == '8y_only':
                loss = criterion(out['logits_10y'], y10)
            else:
                loss = criterion(out['logits_5y'], y5) + criterion(out['logits_10y'], y10)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        all_p5, all_p10, all_y5, all_y10 = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch['wsi_feature'].to(DEVICE),
                           batch['clinical'].to(DEVICE),
                           batch['followup_labs'].to(DEVICE))
                all_p5.append(F.softmax(out['logits_5y'], dim=1)[:, 1].cpu())
                all_p10.append(F.softmax(out['logits_10y'], dim=1)[:, 1].cpu())
                all_y5.append(batch['label_5y'])
                all_y10.append(batch['label_10y'])

        p5 = torch.cat(all_p5).numpy()
        p10 = torch.cat(all_p10).numpy()
        y5v = torch.cat(all_y5).numpy()
        y10v = torch.cat(all_y10).numpy()

        if task_mode == '3y_only':
            metric = roc_auc_score(y5v, p5) if len(np.unique(y5v)) > 1 else 0.5
        elif task_mode == '8y_only':
            metric = roc_auc_score(y10v, p10) if len(np.unique(y10v)) > 1 else 0.5
        else:
            a5 = roc_auc_score(y5v, p5) if len(np.unique(y5v)) > 1 else 0.5
            a10 = roc_auc_score(y10v, p10) if len(np.unique(y10v)) > 1 else 0.5
            metric = (a5 + a10) / 2

        scheduler.step(metric)

        if metric > best_auroc:
            best_auroc = metric
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def bootstrap_auroc(y, s, n=2000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_dir = str(BASE / "dataset")
    wsi_dir = str(BASE / "outputs/embeddings")

    train_ds = MASLDDataset(
        data_dir=data_dir, split='train',
        use_wsi_features=True, wsi_feature_dir=wsi_dir,
        use_followup_labs=True, use_clean_data=True,
        followup_mode='annual_grid', annual_grid_years=2,
        annual_grid_variant='labs_mask'
    )
    val_ds = MASLDDataset(
        data_dir=data_dir, split='val',
        use_wsi_features=True, wsi_feature_dir=wsi_dir,
        use_followup_labs=True, use_clean_data=True,
        followup_mode='annual_grid', annual_grid_years=2,
        annual_grid_variant='labs_mask'
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    strat = build_stratification_labels(train_ds.data)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    results = {}

    for mode in ['multi', '3y_only', '8y_only']:
        print(f"\n{'='*60}")
        print(f"Task mode: {mode}")
        print(f"{'='*60}")

        val_probs_5y, val_probs_10y = [], []
        fold_aucs_5y, fold_aucs_10y = [], []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(strat)), strat)
        ):
            print(f"  Fold {fold+1}...", end=" ", flush=True)
            torch.manual_seed(SEED + fold)

            train_sub = Subset(train_ds, train_idx)
            val_sub = Subset(train_ds, val_idx)
            train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
            fold_val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

            model = M5_FullTemporal(wsi_dim=768, clinical_dim=30, lab_dim=12).to(DEVICE)
            model = train_fold(model, train_loader, fold_val_loader, task_mode=mode)

            # CV metrics
            model.eval()
            all_p5, all_p10, all_y5, all_y10 = [], [], [], []
            with torch.no_grad():
                for batch in fold_val_loader:
                    out = model(batch['wsi_feature'].to(DEVICE),
                               batch['clinical'].to(DEVICE),
                               batch['followup_labs'].to(DEVICE))
                    all_p5.append(F.softmax(out['logits_5y'], dim=1)[:, 1].cpu())
                    all_p10.append(F.softmax(out['logits_10y'], dim=1)[:, 1].cpu())
                    all_y5.append(batch['label_5y'])
                    all_y10.append(batch['label_10y'])

            p5 = torch.cat(all_p5).numpy()
            p10 = torch.cat(all_p10).numpy()
            y5 = torch.cat(all_y5).numpy()
            y10 = torch.cat(all_y10).numpy()

            a5 = roc_auc_score(y5, p5) if len(np.unique(y5)) > 1 else np.nan
            a10 = roc_auc_score(y10, p10)
            fold_aucs_5y.append(a5)
            fold_aucs_10y.append(a10)
            print(f"3y={a5:.4f}, 8y={a10:.4f}")

            # External val predictions
            vp5, vp10 = [], []
            with torch.no_grad():
                for batch in val_loader:
                    out = model(batch['wsi_feature'].to(DEVICE),
                               batch['clinical'].to(DEVICE),
                               batch['followup_labs'].to(DEVICE))
                    vp5.append(F.softmax(out['logits_5y'], dim=1)[:, 1].cpu().numpy())
                    vp10.append(F.softmax(out['logits_10y'], dim=1)[:, 1].cpu().numpy())
            val_probs_5y.append(np.concatenate(vp5))
            val_probs_10y.append(np.concatenate(vp10))

        # Ensemble
        val_p5 = np.mean(val_probs_5y, axis=0)
        val_p10 = np.mean(val_probs_10y, axis=0)

        all_y5, all_y10 = [], []
        for batch in DataLoader(val_ds, batch_size=len(val_ds)):
            all_y5 = batch['label_5y'].numpy()
            all_y10 = batch['label_10y'].numpy()

        auc3 = roc_auc_score(all_y5, val_p5)
        auc8 = roc_auc_score(all_y10, val_p10)
        ci3 = bootstrap_auroc(all_y5, val_p5)
        ci8 = bootstrap_auroc(all_y10, val_p10)

        print(f"\n  Val 3y: {auc3:.4f} [{ci3[0]:.4f}–{ci3[1]:.4f}]")
        print(f"  Val 8y: {auc8:.4f} [{ci8[0]:.4f}–{ci8[1]:.4f}]")

        results[mode] = {
            'val_auroc_3y': round(float(auc3), 4),
            'val_auroc_3y_ci': [round(float(ci3[0]), 4), round(float(ci3[1]), 4)],
            'val_auroc_8y': round(float(auc8), 4),
            'val_auroc_8y_ci': [round(float(ci8[0]), 4), round(float(ci8[1]), 4)],
            'cv_auroc_3y_mean': round(float(np.nanmean(fold_aucs_5y)), 4),
            'cv_auroc_8y_mean': round(float(np.nanmean(fold_aucs_10y)), 4),
        }

    # Summary
    print("\n" + "=" * 70)
    print("SINGLE-TASK vs MULTI-TASK COMPARISON")
    print("=" * 70)
    print(f"{'Mode':<15} {'Val 3y':>12} {'Val 8y':>12}")
    print("-" * 40)
    for mode, r in results.items():
        print(f"{mode:<15} {r['val_auroc_3y']:>12.4f} {r['val_auroc_8y']:>12.4f}")
    print(f"\n  Reference M5 (original): 3y=0.8446, 8y=0.9020")

    with open(BASE / "workspace/supplementary/single_vs_multitask.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved.")


if __name__ == '__main__':
    main()
