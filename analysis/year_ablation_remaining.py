#!/usr/bin/env python3
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""E5 续跑: 只跑 year2 和 both（year1 模式结果已硬编码）。"""
import sys
sys.path.insert(0, f"{ROOT}/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset, Dataset
from model_m5 import M5_FullTemporal
from train import build_stratification_labels

BASE = Path(f"{ROOT}")
SEED = 42
N_FOLDS = 5
DEVICE = 'cpu'
LAB_COLS = ['ALT', 'AST', 'GGT', 'TBIL', 'Albumin', 'Platelet', 'FBG', 'TG', 'TC', 'Creatinine', 'INR']

# year1 模式已完成的结果（直接硬编码，避免重复训练）
KNOWN_YEAR1 = {
    'n_years': 1,
    'val_auroc_3y': 0.8423,
    'val_auroc_3y_ci': [0.7432, 0.9248],
    'val_auroc_8y': 0.8785,
    'val_auroc_8y_ci': [0.8068, 0.9368],
}


class YearAblationDataset(Dataset):
    CLEAN_CLINICAL_FEATURES = [
        'age', 'gender', 'BMI', 'height', 'weight',
        'ALT', 'AST', 'ALP', 'GGT', 'TBIL', 'DBIL', 'Albumin', 'INR',
        'Creatinine', 'Uric_Acid', 'FBG', 'TG', 'TC', 'HDL', 'LDL', 'HbA1c', 'Platelet',
        'Steatosis_Grade', 'Lobular_Inflammation', 'Ballooning', 'NAS_Total',
        'SAF_S', 'SAF_A', 'Impaired_glucose_tolerance', 'L_MAFLD'
    ]

    def __init__(self, data_dir, split, wsi_feature_dir, year_mode='both'):
        self.year_mode = year_mode
        self.split = split
        self.wsi_feature_dir = wsi_feature_dir
        self.lab_dir = os.path.join(data_dir, 'follow_up_labs_clean')

        clin = pd.read_csv(os.path.join(data_dir, split, 'clinical_data_clean.csv'))
        labels = pd.read_csv(os.path.join(data_dir, split, 'patient_labels.csv'))
        self.data = clin.merge(labels, on='patient_id', how='inner')

    def __len__(self):
        return len(self.data)

    def _load_wsi(self, pid):
        pid_str = str(pid).replace('_', '-')
        he_path = Path(self.wsi_feature_dir) / f"{self.split}_HE" / f"{pid_str}-HE_embed.pt"
        sr_path = Path(self.wsi_feature_dir) / f"{self.split}_SR" / f"{pid_str}-SR_embed.pt"
        he = torch.load(he_path, map_location='cpu', weights_only=True).squeeze() if he_path.exists() else torch.zeros(768)
        sr = torch.load(sr_path, map_location='cpu', weights_only=True).squeeze() if sr_path.exists() else torch.zeros(768)
        return torch.stack([he, sr]).mean(dim=0)

    def _build_grid(self, pid, prog, tte):
        lab_file = Path(self.lab_dir) / f"{pid}_follow_up_labs.csv"
        n_years = 2 if self.year_mode == 'both' else 1
        grid = np.zeros((n_years, len(LAB_COLS) + 1), dtype=np.float32)

        if not lab_file.exists():
            return torch.tensor(grid, dtype=torch.float32)

        df = pd.read_csv(lab_file).sort_values('days_from_biopsy')
        if prog == 1:
            event_days = tte * 365
            df = df[df['days_from_biopsy'] < event_days]

        if self.year_mode == 'year1':
            year_df = df[df['days_from_biopsy'] <= 365]
            if len(year_df):
                grid[0, :-1] = year_df.iloc[-1][LAB_COLS].values.astype(np.float32)
                grid[0, -1] = 1.0
        elif self.year_mode == 'year2':
            year2_df = df[(df['days_from_biopsy'] > 365) & (df['days_from_biopsy'] <= 730)]
            if len(year2_df):
                grid[0, :-1] = year2_df.iloc[-1][LAB_COLS].values.astype(np.float32)
                grid[0, -1] = 1.0
            else:
                year1_df = df[df['days_from_biopsy'] <= 365]
                if len(year1_df):
                    grid[0, :-1] = year1_df.iloc[-1][LAB_COLS].values.astype(np.float32)
        else:
            last_values = None
            for year in range(2):
                end_day = (year + 1) * 365
                year_df = df[df['days_from_biopsy'] <= end_day]
                if len(year_df):
                    observed = year_df[year_df['days_from_biopsy'] > year * 365]
                    last_values = year_df.iloc[-1][LAB_COLS].values.astype(np.float32)
                    grid[year, :-1] = last_values
                    grid[year, -1] = 1.0 if len(observed) else 0.0
                elif last_values is not None:
                    grid[year, :-1] = last_values

        return torch.tensor(grid, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pid = row['patient_id']

        wsi = self._load_wsi(pid)
        clinical = torch.tensor(
            row[self.CLEAN_CLINICAL_FEATURES].values.astype(np.float32),
            dtype=torch.float32
        )
        labs = self._build_grid(pid, row['liver_progression_occurred'], row['time_to_event_years'])

        label_5y = int(row['liver_progression_occurred'] == 1 and row['time_to_event_years'] <= 5)
        label_10y = int(row['liver_progression_occurred'] == 1)

        return {
            'wsi_feature': wsi,
            'clinical': clinical,
            'followup_labs': labs,
            'label_5y': torch.tensor(label_5y, dtype=torch.long),
            'label_10y': torch.tensor(label_10y, dtype=torch.long),
            'patient_id': str(pid),
        }


def train_fold(model, train_loader, val_loader, n_epochs=60, patience=10):
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
            loss = criterion(out['logits_5y'], y5) + criterion(out['logits_10y'], y10)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        all_p10, all_y10 = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch['wsi_feature'].to(DEVICE),
                           batch['clinical'].to(DEVICE),
                           batch['followup_labs'].to(DEVICE))
                all_p10.append(F.softmax(out['logits_10y'], dim=1)[:, 1].cpu())
                all_y10.append(batch['label_10y'])

        p10 = torch.cat(all_p10).numpy()
        y10 = torch.cat(all_y10).numpy()
        monitor_auc = roc_auc_score(y10, p10)

        scheduler.step(monitor_auc)
        if monitor_auc > best_auroc:
            best_auroc = monitor_auc
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

    # 预填充已完成的 year1 模式
    results = {'year1': KNOWN_YEAR1}

    for year_mode in ['year2', 'both']:
        print(f"\n{'='*60}")
        print(f"Year mode: {year_mode}")
        print(f"{'='*60}")

        n_years = 1 if year_mode != 'both' else 2

        train_ds = YearAblationDataset(data_dir, 'train', wsi_dir, year_mode)
        val_ds = YearAblationDataset(data_dir, 'val', wsi_dir, year_mode)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        strat = build_stratification_labels(train_ds.data)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        val_probs_5y, val_probs_10y = [], []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(strat)), strat)
        ):
            print(f"  Fold {fold+1}...", end=" ", flush=True)
            torch.manual_seed(SEED + fold)

            train_sub = Subset(train_ds, train_idx)
            val_sub = Subset(train_ds, val_idx)
            train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
            fold_val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

            model = M5_FullTemporal(
                wsi_dim=768, clinical_dim=30, lab_dim=12,
                temporal_dim=128, lstm_hidden=128
            ).to(DEVICE)
            model = train_fold(model, train_loader, fold_val_loader)

            model.eval()
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

            fp10, fy10 = [], []
            with torch.no_grad():
                for batch in fold_val_loader:
                    out = model(batch['wsi_feature'].to(DEVICE),
                               batch['clinical'].to(DEVICE),
                               batch['followup_labs'].to(DEVICE))
                    fp10.append(F.softmax(out['logits_10y'], dim=1)[:, 1].cpu())
                    fy10.append(batch['label_10y'])
            fa10 = roc_auc_score(torch.cat(fy10).numpy(), torch.cat(fp10).numpy())
            print(f"8y={fa10:.4f}")

        val_p5 = np.mean(val_probs_5y, axis=0)
        val_p10 = np.mean(val_probs_10y, axis=0)

        for batch in DataLoader(val_ds, batch_size=len(val_ds)):
            all_y5 = batch['label_5y'].numpy()
            all_y10 = batch['label_10y'].numpy()

        auc3 = roc_auc_score(all_y5, val_p5)
        auc8 = roc_auc_score(all_y10, val_p10)
        ci3 = bootstrap_auroc(all_y5, val_p5)
        ci8 = bootstrap_auroc(all_y10, val_p10)

        print(f"\n  Val 3y: {auc3:.4f} [{ci3[0]:.4f}–{ci3[1]:.4f}]")
        print(f"  Val 8y: {auc8:.4f} [{ci8[0]:.4f}–{ci8[1]:.4f}]")

        results[year_mode] = {
            'n_years': n_years,
            'val_auroc_3y': round(float(auc3), 4),
            'val_auroc_3y_ci': [round(float(ci3[0]), 4), round(float(ci3[1]), 4)],
            'val_auroc_8y': round(float(auc8), 4),
            'val_auroc_8y_ci': [round(float(ci8[0]), 4), round(float(ci8[1]), 4)],
        }

        # 每个 mode 完成后立即写入，防止中断丢失
        with open(BASE / "workspace/supplementary/year_ablation.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [saved partial results]")

    print("\n" + "=" * 60)
    print("YEAR ABLATION SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<15} {'Val 3y':>12} {'Val 8y':>12}")
    print("-" * 40)
    for mode, r in results.items():
        print(f"{mode:<15} {r['val_auroc_3y']:>12.4f} {r['val_auroc_8y']:>12.4f}")
    print(f"{'M5 original':<15} {'0.8446':>12} {'0.9020':>12}")
    print(f"\nFinal saved to workspace/supplementary/year_ablation.json")


if __name__ == '__main__':
    main()
