#!/usr/bin/env python3
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""A3: H&E-only vs Sirius-Red-only stain ablation.

Trains M3 (WSI-only model) using:
  1. H&E embeddings only
  2. Sirius Red embeddings only
  3. Dual-stain averaged (existing M3 result)

Also tests concatenation (1536→512) vs averaging (768→512).

Uses same fold splits and training procedure as original M3.
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
from torch.utils.data import DataLoader, Subset, Dataset
import pandas as pd
import os

BASE = Path(f"{ROOT}")
SEED = 42
N_FOLDS = 5
DEVICE = 'cpu'


class SingleStainDataset(Dataset):
    """Dataset that loads only one stain or concatenates both."""

    def __init__(self, data_dir, split, wsi_feature_dir, stain_mode='average'):
        """stain_mode: 'he_only', 'sr_only', 'average', 'concat'"""
        self.stain_mode = stain_mode
        self.split = split
        self.wsi_feature_dir = wsi_feature_dir

        clinical_path = os.path.join(data_dir, split, 'clinical_data_clean.csv')
        labels_path = os.path.join(data_dir, split, 'patient_labels.csv')
        clin = pd.read_csv(clinical_path)
        labels = pd.read_csv(labels_path)
        self.data = clin.merge(labels, on='patient_id', how='inner')
        self.patient_ids = self.data['patient_id'].tolist()

    def __len__(self):
        return len(self.data)

    def _load_emb(self, pid, stain):
        pid_str = str(pid).replace('_', '-')
        path = Path(self.wsi_feature_dir) / f"{self.split}_{stain}" / f"{pid_str}-{stain}_embed.pt"
        if path.exists():
            return torch.load(path, map_location='cpu', weights_only=True).squeeze()
        return torch.zeros(768)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pid = row['patient_id']

        he = self._load_emb(pid, 'HE')
        sr = self._load_emb(pid, 'SR')

        if self.stain_mode == 'he_only':
            wsi = he
        elif self.stain_mode == 'sr_only':
            wsi = sr
        elif self.stain_mode == 'concat':
            wsi = torch.cat([he, sr])
        else:
            wsi = torch.stack([he, sr]).mean(dim=0)

        label_5y = int(row['liver_progression_occurred'] == 1 and row['time_to_event_years'] <= 5)
        label_10y = int(row['liver_progression_occurred'] == 1)

        return {
            'wsi_feature': wsi,
            'label_5y': torch.tensor(label_5y, dtype=torch.long),
            'label_10y': torch.tensor(label_10y, dtype=torch.long),
            'patient_id': str(pid),
        }


class M3_Variant(nn.Module):
    """M3 with configurable input dimension for stain ablation."""

    def __init__(self, wsi_dim=768, dropout=0.4):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(wsi_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier_5y = nn.Linear(256, 2)
        self.classifier_10y = nn.Linear(256, 2)

    def forward(self, wsi_features):
        features = self.feature_extractor(wsi_features)
        return {
            'logits_5y': self.classifier_5y(features),
            'logits_10y': self.classifier_10y(features),
        }


def build_stratification_labels(data):
    label_10y = data['liver_progression_occurred'].astype(int).values
    label_5y = (
        (data['liver_progression_occurred'].astype(int) == 1) &
        (data['time_to_event_years'].astype(float) <= 5)
    ).astype(int).values
    return label_10y + 2 * label_5y


def train_fold(model, train_loader, val_loader, n_epochs=60, patience=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_auroc = 0
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            wsi = batch['wsi_feature'].to(DEVICE)
            y5 = batch['label_5y'].to(DEVICE)
            y10 = batch['label_10y'].to(DEVICE)
            out = model(wsi)
            loss = criterion(out['logits_5y'], y5) + criterion(out['logits_10y'], y10)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        all_p10, all_y10 = [], []
        with torch.no_grad():
            for batch in val_loader:
                wsi = batch['wsi_feature'].to(DEVICE)
                out = model(wsi)
                all_p10.append(F.softmax(out['logits_10y'], dim=1)[:, 1].cpu())
                all_y10.append(batch['label_10y'])
        p10 = torch.cat(all_p10).numpy()
        y10 = torch.cat(all_y10).numpy()
        auc = roc_auc_score(y10, p10) if len(np.unique(y10)) > 1 else 0.5

        if auc > best_auroc:
            best_auroc = auc
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

    stain_modes = {
        'H&E only': ('he_only', 768),
        'Sirius Red only': ('sr_only', 768),
        'Dual-stain average': ('average', 768),
        'Dual-stain concat': ('concat', 1536),
    }

    results = {}

    for mode_name, (stain_mode, wsi_dim) in stain_modes.items():
        print(f"\n{'='*60}")
        print(f"Stain mode: {mode_name} (dim={wsi_dim})")
        print(f"{'='*60}")

        train_ds = SingleStainDataset(data_dir, 'train', wsi_dir, stain_mode)
        val_ds = SingleStainDataset(data_dir, 'val', wsi_dir, stain_mode)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        strat = build_stratification_labels(train_ds.data)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        fold_aucs_3y, fold_aucs_8y = [], []
        val_probs_3y, val_probs_8y = [], []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(strat)), strat)
        ):
            print(f"  Fold {fold+1}...", end=" ")

            train_sub = Subset(train_ds, train_idx)
            val_sub = Subset(train_ds, val_idx)
            train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
            fold_val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

            model = M3_Variant(wsi_dim=wsi_dim).to(DEVICE)
            model = train_fold(model, train_loader, fold_val_loader)

            # CV metrics
            model.eval()
            all_p5, all_p10, all_y5, all_y10 = [], [], [], []
            with torch.no_grad():
                for batch in fold_val_loader:
                    out = model(batch['wsi_feature'].to(DEVICE))
                    all_p5.append(F.softmax(out['logits_5y'], dim=1)[:, 1].cpu())
                    all_p10.append(F.softmax(out['logits_10y'], dim=1)[:, 1].cpu())
                    all_y5.append(batch['label_5y'])
                    all_y10.append(batch['label_10y'])

            p5 = torch.cat(all_p5).numpy()
            p10 = torch.cat(all_p10).numpy()
            y5 = torch.cat(all_y5).numpy()
            y10 = torch.cat(all_y10).numpy()

            auc5 = roc_auc_score(y5, p5) if len(np.unique(y5)) > 1 else np.nan
            auc10 = roc_auc_score(y10, p10)
            fold_aucs_3y.append(auc5)
            fold_aucs_8y.append(auc10)
            print(f"3y={auc5:.4f}, 8y={auc10:.4f}")

            # External val predictions
            vp5, vp10 = [], []
            with torch.no_grad():
                for batch in val_loader:
                    out = model(batch['wsi_feature'].to(DEVICE))
                    vp5.append(F.softmax(out['logits_5y'], dim=1)[:, 1].cpu().numpy())
                    vp10.append(F.softmax(out['logits_10y'], dim=1)[:, 1].cpu().numpy())
            val_probs_3y.append(np.concatenate(vp5))
            val_probs_8y.append(np.concatenate(vp10))

        # Ensemble
        val_p3 = np.mean(val_probs_3y, axis=0)
        val_p8 = np.mean(val_probs_8y, axis=0)

        # Get labels
        all_y5, all_y10 = [], []
        for batch in DataLoader(val_ds, batch_size=len(val_ds)):
            all_y5 = batch['label_5y'].numpy()
            all_y10 = batch['label_10y'].numpy()

        auc3 = roc_auc_score(all_y5, val_p3)
        auc8 = roc_auc_score(all_y10, val_p8)
        ci3 = bootstrap_auroc(all_y5, val_p3)
        ci8 = bootstrap_auroc(all_y10, val_p8)

        print(f"\n  External Val 3y: {auc3:.4f} [{ci3[0]:.4f}–{ci3[1]:.4f}]")
        print(f"  External Val 8y: {auc8:.4f} [{ci8[0]:.4f}–{ci8[1]:.4f}]")

        results[mode_name] = {
            'stain_mode': stain_mode,
            'wsi_dim': wsi_dim,
            'cv_auroc_3y_mean': round(float(np.nanmean(fold_aucs_3y)), 4),
            'cv_auroc_3y_std': round(float(np.nanstd(fold_aucs_3y)), 4),
            'cv_auroc_8y_mean': round(float(np.nanmean(fold_aucs_8y)), 4),
            'cv_auroc_8y_std': round(float(np.nanstd(fold_aucs_8y)), 4),
            'val_auroc_3y': round(float(auc3), 4),
            'val_auroc_3y_ci': [round(float(ci3[0]), 4), round(float(ci3[1]), 4)],
            'val_auroc_8y': round(float(auc8), 4),
            'val_auroc_8y_ci': [round(float(ci8[0]), 4), round(float(ci8[1]), 4)],
        }

    # Summary
    print("\n" + "=" * 80)
    print("STAIN ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Mode':<25} {'Val 3y AUROC':>14} {'Val 8y AUROC':>14}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<25} {r['val_auroc_3y']:>14.4f} {r['val_auroc_8y']:>14.4f}")

    out_path = BASE / "workspace/supplementary/stain_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
