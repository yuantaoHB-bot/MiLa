#!/usr/bin/env python3
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""A4: Extract per-fold training AUROC from existing M5 checkpoints.

Loads each fold's best model, evaluates on its own training subset,
and reports the train-val AUROC gap to quantify overfitting.
"""
import sys
sys.path.insert(0, f"{ROOT}/src")

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from dataset import MASLDDataset
from model_m5 import M5_FullTemporal
from model_m4 import M4_StaticMultimodal
from model_m3 import M3_WSIOnly
from train import build_stratification_labels

BASE = Path(f"{ROOT}")
SEED = 42
N_FOLDS = 5
DEVICE = 'cpu'


def evaluate_model(model, loader, model_type='m5'):
    model.eval()
    all_p5, all_p10, all_y5, all_y10 = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            wsi = batch['wsi_feature'].to(DEVICE)
            if model_type == 'm3':
                out = model(wsi)
            elif model_type == 'm4':
                clin = batch['clinical'].to(DEVICE)
                out = model(wsi, clin)
            else:
                clin = batch['clinical'].to(DEVICE)
                labs = batch['followup_labs'].to(DEVICE)
                out = model(wsi, clin, labs)
            all_p5.append(F.softmax(out['logits_5y'], dim=1)[:, 1].cpu())
            all_p10.append(F.softmax(out['logits_10y'], dim=1)[:, 1].cpu())
            all_y5.append(batch['label_5y'])
            all_y10.append(batch['label_10y'])
    p5 = torch.cat(all_p5).numpy()
    p10 = torch.cat(all_p10).numpy()
    y5 = torch.cat(all_y5).numpy()
    y10 = torch.cat(all_y10).numpy()
    auc5 = roc_auc_score(y5, p5) if len(np.unique(y5)) > 1 else np.nan
    auc10 = roc_auc_score(y10, p10) if len(np.unique(y10)) > 1 else np.nan
    return auc5, auc10


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_dir = str(BASE / "dataset")
    wsi_dir = str(BASE / "outputs/embeddings")

    # Model configurations
    configs = [
        {
            'name': 'M3',
            'model_dir': BASE / "outputs/models",
            'ckpt_pattern': 'm3_fold{fold}.pth',
            'model_type': 'm3',
            'use_followup': False,
            'followup_mode': None,
            'variant': None,
        },
        {
            'name': 'M4',
            'model_dir': BASE / "outputs/models",
            'ckpt_pattern': 'm4_fold{fold}.pth',
            'model_type': 'm4',
            'use_followup': False,
            'followup_mode': None,
            'variant': None,
        },
        {
            'name': 'M5_labs_mask',
            'model_dir': BASE / "outputs/models/m5_labs_mask",
            'ckpt_pattern': 'm5_fold{fold}.pth',
            'model_type': 'm5',
            'use_followup': True,
            'followup_mode': 'annual_grid',
            'variant': 'labs_mask',
        },
        {
            'name': 'M5_lab_only',
            'model_dir': BASE / "outputs/models/m5_lab_only",
            'ckpt_pattern': 'm5_fold{fold}.pth',
            'model_type': 'm5',
            'use_followup': True,
            'followup_mode': 'annual_grid',
            'variant': 'lab_only',
        },
        {
            'name': 'M5_mask_only',
            'model_dir': BASE / "outputs/models/m5_mask_only",
            'ckpt_pattern': 'm5_fold{fold}.pth',
            'model_type': 'm5',
            'use_followup': True,
            'followup_mode': 'annual_grid',
            'variant': 'mask_only',
        },
    ]

    results = {}

    for cfg in configs:
        name = cfg['name']
        model_dir = cfg['model_dir']
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")

        # Check if checkpoints exist
        test_path = model_dir / cfg['ckpt_pattern'].format(fold=1)
        if not test_path.exists():
            print(f"  Skipping: {test_path} not found")
            continue

        # Load dataset matching model config
        train_ds = MASLDDataset(
            data_dir=data_dir, split='train',
            use_wsi_features=True, wsi_feature_dir=wsi_dir,
            use_followup_labs=cfg['use_followup'], use_clean_data=True,
            followup_mode=cfg['followup_mode'] or 'fixed_visits',
            annual_grid_years=2,
            annual_grid_variant=cfg['variant'] or 'labs_mask'
        )

        strat = build_stratification_labels(train_ds.data)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(strat)), strat)
        ):
            ckpt_path = model_dir / cfg['ckpt_pattern'].format(fold=fold+1)
            if not ckpt_path.exists():
                print(f"  Fold {fold+1}: checkpoint not found")
                continue

            # Instantiate model
            if cfg['model_type'] == 'm3':
                model = M3_WSIOnly(input_dim=768)
            elif cfg['model_type'] == 'm4':
                model = M4_StaticMultimodal(wsi_dim=768, clinical_dim=30)
            else:
                lab_dim = {'labs_mask': 12, 'lab_only': 11, 'mask_only': 1}[cfg['variant']]
                model = M5_FullTemporal(wsi_dim=768, clinical_dim=30, lab_dim=lab_dim)

            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)
            model.to(DEVICE)

            if fold == 0:
                n_params = count_params(model)
                print(f"  Parameters: {n_params:,}")

            train_sub = Subset(train_ds, train_idx)
            val_sub = Subset(train_ds, val_idx)
            train_loader = DataLoader(train_sub, batch_size=32, shuffle=False)
            val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

            train_auc5, train_auc10 = evaluate_model(model, train_loader, cfg['model_type'])
            val_auc5, val_auc10 = evaluate_model(model, val_loader, cfg['model_type'])

            fold_results.append({
                'fold': fold + 1,
                'train_auroc_3y': round(float(train_auc5), 4) if not np.isnan(train_auc5) else None,
                'train_auroc_8y': round(float(train_auc10), 4) if not np.isnan(train_auc10) else None,
                'val_auroc_3y': round(float(val_auc5), 4) if not np.isnan(val_auc5) else None,
                'val_auroc_8y': round(float(val_auc10), 4) if not np.isnan(val_auc10) else None,
            })

            gap_3y = train_auc5 - val_auc5 if not (np.isnan(train_auc5) or np.isnan(val_auc5)) else np.nan
            gap_8y = train_auc10 - val_auc10 if not (np.isnan(train_auc10) or np.isnan(val_auc10)) else np.nan

            print(f"  Fold {fold+1}: Train 3y={train_auc5:.4f} Val 3y={val_auc5:.4f} Gap={gap_3y:.4f}  |  "
                  f"Train 8y={train_auc10:.4f} Val 8y={val_auc10:.4f} Gap={gap_8y:.4f}")

        if fold_results:
            train_3y = [r['train_auroc_3y'] for r in fold_results if r['train_auroc_3y'] is not None]
            val_3y = [r['val_auroc_3y'] for r in fold_results if r['val_auroc_3y'] is not None]
            train_8y = [r['train_auroc_8y'] for r in fold_results if r['train_auroc_8y'] is not None]
            val_8y = [r['val_auroc_8y'] for r in fold_results if r['val_auroc_8y'] is not None]

            mean_gap_3y = np.mean(train_3y) - np.mean(val_3y) if train_3y and val_3y else None
            mean_gap_8y = np.mean(train_8y) - np.mean(val_8y) if train_8y and val_8y else None

            print(f"\n  Mean Train-Val Gap: 3y={mean_gap_3y:.4f}, 8y={mean_gap_8y:.4f}")

            results[name] = {
                'n_params': n_params,
                'fold_results': fold_results,
                'mean_train_auroc_3y': round(float(np.mean(train_3y)), 4) if train_3y else None,
                'mean_train_auroc_8y': round(float(np.mean(train_8y)), 4) if train_8y else None,
                'mean_val_auroc_3y': round(float(np.mean(val_3y)), 4) if val_3y else None,
                'mean_val_auroc_8y': round(float(np.mean(val_8y)), 4) if val_8y else None,
                'mean_gap_3y': round(float(mean_gap_3y), 4) if mean_gap_3y is not None else None,
                'mean_gap_8y': round(float(mean_gap_8y), 4) if mean_gap_8y is not None else None,
            }

    # Summary table
    print("\n" + "=" * 90)
    print("OVERFITTING ASSESSMENT: Train-Val AUROC Gap")
    print("=" * 90)
    print(f"{'Model':<20} {'Params':>10} {'Train 3y':>10} {'Val 3y':>10} {'Gap 3y':>10} {'Train 8y':>10} {'Val 8y':>10} {'Gap 8y':>10}")
    print("-" * 90)
    for name, r in results.items():
        print(f"{name:<20} {r['n_params']:>10,} "
              f"{r['mean_train_auroc_3y']:>10.4f} {r['mean_val_auroc_3y']:>10.4f} {r['mean_gap_3y']:>10.4f} "
              f"{r['mean_train_auroc_8y']:>10.4f} {r['mean_val_auroc_8y']:>10.4f} {r['mean_gap_8y']:>10.4f}")

    out_path = BASE / "workspace/supplementary/train_val_gap.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
