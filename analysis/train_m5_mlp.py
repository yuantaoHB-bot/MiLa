#!/usr/bin/env python3
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
"""Train M5-MLP: simplified M5 where Transformer+BiLSTM is replaced by MLP.

Purpose: Architecture complexity ablation — does the Transformer+BiLSTM
temporal branch provide genuine value over a simple flatten+MLP?

Uses the exact same:
- Data loading (MASLDDataset with annual_grid, labs_mask, clean data)
- Fold splits (StratifiedKFold, seed=42, composite stratification)
- Training procedure (AdamW, CrossEntropy, ReduceLROnPlateau, gradient clip)
- External validation (5-fold ensemble averaging)
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
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Subset
from dataset import MASLDDataset
from train import build_stratification_labels

BASE = Path(f"{ROOT}")
SEED = 42
N_FOLDS = 5
DEVICE = 'cpu'


class M5_MLPTemporal(nn.Module):
    """M5 variant: MLP replaces Transformer+BiLSTM for temporal processing."""

    def __init__(self, wsi_dim=768, clinical_dim=30, lab_dim=12,
                 hidden_dim=512, fusion_dim=256, temporal_hidden=128,
                 dropout=0.4, annual_grid_years=2):
        super().__init__()

        self.annual_grid_years = annual_grid_years

        self.wsi_encoder = nn.Sequential(
            nn.Linear(wsi_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.static_fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        flat_lab_dim = annual_grid_years * lab_dim
        self.temporal_mlp = nn.Sequential(
            nn.Linear(flat_lab_dim, temporal_hidden),
            nn.BatchNorm1d(temporal_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden, temporal_hidden),
            nn.BatchNorm1d(temporal_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim + temporal_hidden, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier_5y = nn.Linear(fusion_dim, 2)
        self.classifier_10y = nn.Linear(fusion_dim, 2)

    def forward(self, wsi_features, clinical_features, lab_sequence):
        wsi_enc = self.wsi_encoder(wsi_features)
        clin_enc = self.clinical_encoder(clinical_features)
        static = self.static_fusion(torch.cat([wsi_enc, clin_enc], dim=1))

        lab_flat = lab_sequence.reshape(lab_sequence.size(0), -1)
        temporal = self.temporal_mlp(lab_flat)

        fused = self.final_fusion(torch.cat([static, temporal], dim=1))

        return {
            'logits_5y': self.classifier_5y(fused),
            'logits_10y': self.classifier_10y(fused),
            'features': fused,
            'static_features': static,
            'temporal_features': temporal,
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_fold(model, train_loader, val_loader, n_epochs=60, patience=10):
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

        # Evaluate
        model.eval()
        all_p5, all_p10, all_y5, all_y10 = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                wsi = batch['wsi_feature'].to(DEVICE)
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

        auc5 = roc_auc_score(y5, p5) if len(np.unique(y5)) > 1 else 0.5
        auc10 = roc_auc_score(y10, p10) if len(np.unique(y10)) > 1 else 0.5
        mean_auc = (auc5 + auc10) / 2

        scheduler.step(mean_auc)

        if mean_auc > best_auroc:
            best_auroc = mean_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auroc


def predict_with_model(model, loader):
    model.eval()
    all_p5, all_p10, all_y5, all_y10, all_pids = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            wsi = batch['wsi_feature'].to(DEVICE)
            clin = batch['clinical'].to(DEVICE)
            labs = batch['followup_labs'].to(DEVICE)
            out = model(wsi, clin, labs)
            all_p5.append(F.softmax(out['logits_5y'], dim=1)[:, 1].cpu().numpy())
            all_p10.append(F.softmax(out['logits_10y'], dim=1)[:, 1].cpu().numpy())
            all_y5.append(batch['label_5y'].numpy())
            all_y10.append(batch['label_10y'].numpy())
            all_pids.extend(batch['patient_id'])
    return {
        'prob_5y': np.concatenate(all_p5),
        'prob_10y': np.concatenate(all_p10),
        'label_5y': np.concatenate(all_y5),
        'label_10y': np.concatenate(all_y10),
        'patient_ids': all_pids,
    }


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

    # Load training dataset (same config as M5)
    train_dataset = MASLDDataset(
        data_dir=data_dir, split='train',
        use_wsi_features=True, wsi_feature_dir=wsi_dir,
        use_followup_labs=True, use_clean_data=True,
        followup_mode='annual_grid', annual_grid_years=2,
        annual_grid_variant='labs_mask'
    )

    val_dataset = MASLDDataset(
        data_dir=data_dir, split='val',
        use_wsi_features=True, wsi_feature_dir=wsi_dir,
        use_followup_labs=True, use_clean_data=True,
        followup_mode='annual_grid', annual_grid_years=2,
        annual_grid_variant='labs_mask'
    )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    strat_labels = build_stratification_labels(train_dataset.data)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Also train M5-MLP with lab_only and mask_only
    variants = {
        'labs_mask': 12,
        'lab_only': 11,
        'mask_only': 1,
    }

    all_results = {}

    for variant_name, lab_dim in variants.items():
        print(f"\n{'='*60}")
        print(f"M5-MLP variant: {variant_name} (lab_dim={lab_dim})")
        print(f"{'='*60}")

        # Reload datasets with correct variant
        train_ds = MASLDDataset(
            data_dir=data_dir, split='train',
            use_wsi_features=True, wsi_feature_dir=wsi_dir,
            use_followup_labs=True, use_clean_data=True,
            followup_mode='annual_grid', annual_grid_years=2,
            annual_grid_variant=variant_name
        )
        val_ds = MASLDDataset(
            data_dir=data_dir, split='val',
            use_wsi_features=True, wsi_feature_dir=wsi_dir,
            use_followup_labs=True, use_clean_data=True,
            followup_mode='annual_grid', annual_grid_years=2,
            annual_grid_variant=variant_name
        )
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        strat_labels = build_stratification_labels(train_ds.data)

        fold_aurocs_5y = []
        fold_aurocs_10y = []
        fold_train_aurocs_5y = []
        fold_train_aurocs_10y = []
        val_probs_5y_all = []
        val_probs_10y_all = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(strat_labels)), strat_labels)
        ):
            print(f"\n  Fold {fold+1}/{N_FOLDS}")

            train_sub = Subset(train_ds, train_idx)
            val_sub = Subset(train_ds, val_idx)

            train_loader = DataLoader(train_sub, batch_size=16, shuffle=True, drop_last=False)
            fold_val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

            model = M5_MLPTemporal(
                wsi_dim=768, clinical_dim=30, lab_dim=lab_dim,
                hidden_dim=512, fusion_dim=256, temporal_hidden=128,
                dropout=0.4, annual_grid_years=2
            ).to(DEVICE)

            if fold == 0 and variant_name == 'labs_mask':
                print(f"  Parameters: {count_parameters(model):,}")

            model, best_auc = train_one_fold(model, train_loader, fold_val_loader)
            print(f"  Best CV AUROC (mean): {best_auc:.4f}")

            # CV fold metrics
            fold_pred = predict_with_model(model, fold_val_loader)
            auc5 = roc_auc_score(fold_pred['label_5y'], fold_pred['prob_5y']) if len(np.unique(fold_pred['label_5y'])) > 1 else np.nan
            auc10 = roc_auc_score(fold_pred['label_10y'], fold_pred['prob_10y'])
            fold_aurocs_5y.append(auc5)
            fold_aurocs_10y.append(auc10)
            print(f"  Fold val AUROC: 3y={auc5:.4f}, 8y={auc10:.4f}")

            # Training set metrics (for train-val gap)
            train_loader_eval = DataLoader(train_sub, batch_size=32, shuffle=False)
            train_pred = predict_with_model(model, train_loader_eval)
            tr_auc5 = roc_auc_score(train_pred['label_5y'], train_pred['prob_5y']) if len(np.unique(train_pred['label_5y'])) > 1 else np.nan
            tr_auc10 = roc_auc_score(train_pred['label_10y'], train_pred['prob_10y'])
            fold_train_aurocs_5y.append(tr_auc5)
            fold_train_aurocs_10y.append(tr_auc10)
            print(f"  Fold train AUROC: 3y={tr_auc5:.4f}, 8y={tr_auc10:.4f}")

            # External validation predictions
            val_pred = predict_with_model(model, val_loader)
            val_probs_5y_all.append(val_pred['prob_5y'])
            val_probs_10y_all.append(val_pred['prob_10y'])

        # Ensemble
        val_prob_5y = np.mean(val_probs_5y_all, axis=0)
        val_prob_10y = np.mean(val_probs_10y_all, axis=0)
        val_labels = predict_with_model(model, val_loader)

        val_auc_3y = roc_auc_score(val_labels['label_5y'], val_prob_5y)
        val_auc_8y = roc_auc_score(val_labels['label_10y'], val_prob_10y)
        ci_3y = bootstrap_auroc(val_labels['label_5y'], val_prob_5y)
        ci_8y = bootstrap_auroc(val_labels['label_10y'], val_prob_10y)

        print(f"\n  === {variant_name} External Validation ===")
        print(f"  3y: {val_auc_3y:.4f} [{ci_3y[0]:.4f}–{ci_3y[1]:.4f}]")
        print(f"  8y: {val_auc_8y:.4f} [{ci_8y[0]:.4f}–{ci_8y[1]:.4f}]")
        print(f"  CV 3y: {np.nanmean(fold_aurocs_5y):.4f}±{np.nanstd(fold_aurocs_5y):.4f}")
        print(f"  CV 8y: {np.nanmean(fold_aurocs_10y):.4f}±{np.nanstd(fold_aurocs_10y):.4f}")
        print(f"  Train-Val gap 3y: {np.nanmean(fold_train_aurocs_5y) - np.nanmean(fold_aurocs_5y):.4f}")
        print(f"  Train-Val gap 8y: {np.nanmean(fold_train_aurocs_10y) - np.nanmean(fold_aurocs_10y):.4f}")

        all_results[f"M5-MLP_{variant_name}"] = {
            'variant': variant_name,
            'lab_dim': lab_dim,
            'n_params': count_parameters(M5_MLPTemporal(
                clinical_dim=30, lab_dim=lab_dim, annual_grid_years=2
            )),
            'cv_auroc_3y_mean': round(float(np.nanmean(fold_aurocs_5y)), 4),
            'cv_auroc_3y_std': round(float(np.nanstd(fold_aurocs_5y)), 4),
            'cv_auroc_8y_mean': round(float(np.nanmean(fold_aurocs_10y)), 4),
            'cv_auroc_8y_std': round(float(np.nanstd(fold_aurocs_10y)), 4),
            'cv_train_auroc_3y_mean': round(float(np.nanmean(fold_train_aurocs_5y)), 4),
            'cv_train_auroc_8y_mean': round(float(np.nanmean(fold_train_aurocs_10y)), 4),
            'cv_train_val_gap_3y': round(float(np.nanmean(fold_train_aurocs_5y) - np.nanmean(fold_aurocs_5y)), 4),
            'cv_train_val_gap_8y': round(float(np.nanmean(fold_train_aurocs_10y) - np.nanmean(fold_aurocs_10y)), 4),
            'val_auroc_3y': round(float(val_auc_3y), 4),
            'val_auroc_3y_ci': [round(float(ci_3y[0]), 4), round(float(ci_3y[1]), 4)],
            'val_auroc_8y': round(float(val_auc_8y), 4),
            'val_auroc_8y_ci': [round(float(ci_8y[0]), 4), round(float(ci_8y[1]), 4)],
        }

    # Print comparison
    print("\n" + "=" * 80)
    print("ARCHITECTURE COMPLEXITY COMPARISON")
    print("=" * 80)
    print(f"{'Model':<30} {'Params':>10} {'Val 3y':>10} {'Val 8y':>10}")
    print("-" * 65)

    # M5 (original) reference values
    m5_ref = {
        'labs_mask': {'params': '~1.6M', '3y': 0.8446, '8y': 0.9020},
        'lab_only': {'params': '~1.6M', '3y': 0.8399, '8y': 0.8902},
        'mask_only': {'params': '~1.6M', '3y': 0.7000, '8y': 0.7832},
    }
    for v in ['labs_mask', 'lab_only', 'mask_only']:
        ref = m5_ref[v]
        print(f"M5-Transformer {v:<15} {ref['params']:>10} {ref['3y']:>10.4f} {ref['8y']:>10.4f}")
        r = all_results[f"M5-MLP_{v}"]
        print(f"M5-MLP {v:<23} {r['n_params']:>10,} {r['val_auroc_3y']:>10.4f} {r['val_auroc_8y']:>10.4f}")
        print()

    # Save
    out_path = BASE / "workspace/supplementary/architecture_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
