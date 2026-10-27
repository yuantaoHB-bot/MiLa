"""
统一训练脚本
支持训练M3/M4/M5模型，包含5折交叉验证
"""
import os
import sys
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import json
from datetime import datetime

# 导入模型和数据集
from dataset import MASLDDataset
from model_m3 import M3_WSIOnly, M3_Trainer
from model_m4 import M4_StaticMultimodal, M4_Trainer
from model_m5 import M5_FullTemporal, M5_Trainer


def set_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_stratification_labels(data):
    label_10y = data['liver_progression_occurred'].astype(int).values
    label_5y = (
        (data['liver_progression_occurred'].astype(int) == 1) &
        (data['time_to_event_years'].astype(float) <= 5)
    ).astype(int).values
    return label_10y + 2 * label_5y


def summarize_fold_distribution(data, train_idx, val_idx):
    def summarize(indices):
        fold_data = data.iloc[list(indices)]
        label_5y = (
            (fold_data['liver_progression_occurred'].astype(int) == 1) &
            (fold_data['time_to_event_years'].astype(float) <= 5)
        )
        label_10y = fold_data['liver_progression_occurred'].astype(int)
        return {
            'n_samples': int(len(fold_data)),
            'n_progression_5y': int(label_5y.sum()),
            'n_nonprogression_5y': int(len(fold_data) - label_5y.sum()),
            'n_progression_10y': int(label_10y.sum()),
            'n_nonprogression_10y': int(len(fold_data) - label_10y.sum())
        }

    return {
        'train': summarize(train_idx),
        'val': summarize(val_idx)
    }


def build_training_metadata(
    model_name: str,
    data_dir: str,
    wsi_feature_dir: str,
    output_dir: str,
    n_folds: int,
    num_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    use_clean_data: bool
):
    return {
        'model': model_name,
        'data_dir': data_dir,
        'wsi_feature_dir': wsi_feature_dir,
        'output_dir': output_dir,
        'n_folds': n_folds,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'device': device,
        'seed': seed,
        'use_clean_data': use_clean_data
    }


def train_model(
    model_name: str,
    data_dir: str,
    wsi_feature_dir: str,
    output_dir: str,
    n_folds: int = 5,
    num_epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = 'cuda',
    seed: int = 42,
    use_clean_data: bool = False,
    followup_mode: str = 'fixed_visits',
    annual_grid_years: int = 5,
    annual_grid_variant: str = 'labs_mask'
):
    """
    训练模型（5折交叉验证）

    Args:
        model_name: 模型名称 ('m3', 'm4', 'm5')
        data_dir: 数据根目录
        wsi_feature_dir: WSI特征目录
        output_dir: 输出目录
        n_folds: 折数
        num_epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        device: 计算设备
        seed: 随机种子
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    print(f"=== 加载数据集 ===")
    use_followup = (model_name == 'm5')

    dataset = MASLDDataset(
        data_dir=data_dir,
        split='train',
        use_wsi_features=True,
        wsi_feature_dir=wsi_feature_dir,
        use_followup_labs=use_followup,
        use_clean_data=use_clean_data,
        followup_mode=followup_mode,
        annual_grid_years=annual_grid_years,
        annual_grid_variant=annual_grid_variant
    )

    print(f"数据集大小: {len(dataset)}")
    print(f"统计信息: {dataset.get_statistics()}")

    # 准备交叉验证
    stratification_labels = build_stratification_labels(dataset.data)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(stratification_labels)), stratification_labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*60}")

        # 创建数据加载器
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        print(f"训练集: {len(train_subset)} 样本")
        print(f"验证集: {len(val_subset)} 样本")
        fold_distribution = summarize_fold_distribution(dataset.data, train_idx, val_idx)
        print(f"5年阳性: train={fold_distribution['train']['n_progression_5y']}, val={fold_distribution['val']['n_progression_5y']}")
        print(f"10年阳性: train={fold_distribution['train']['n_progression_10y']}, val={fold_distribution['val']['n_progression_10y']}")

        # 创建模型
        if model_name == 'm3':
            model = M3_WSIOnly(
                input_dim=768,
                hidden_dim=512,
                dropout=0.4
            )
            trainer = M3_Trainer(model, device=device, lr=lr)

        elif model_name == 'm4':
            model = M4_StaticMultimodal(
                wsi_dim=768,
                clinical_dim=30 if use_clean_data else 35,
                hidden_dim=512,
                fusion_dim=256,
                dropout=0.4
            )
            trainer = M4_Trainer(model, device=device, lr=lr)

        elif model_name == 'm5':
            lab_dim = 12 if followup_mode != 'annual_grid' else {'labs_mask': 12, 'lab_only': 11, 'mask_only': 1}[annual_grid_variant]
            model = M5_FullTemporal(
                wsi_dim=768,
                clinical_dim=30 if use_clean_data else 35,
                lab_dim=lab_dim,
                hidden_dim=512,
                fusion_dim=256,
                temporal_dim=128,
                lstm_hidden=128,
                dropout=0.4
            )
            trainer = M5_Trainer(model, device=device, lr=lr)

        else:
            raise ValueError(f"未知模型: {model_name}")

        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 训练
        save_path = os.path.join(output_dir, f"{model_name}_fold{fold+1}.pth")

        best_auroc_5y = 0.0
        best_auroc_10y = 0.0
        patience_counter = 0
        patience = 10

        for epoch in range(num_epochs):
            # 训练
            train_metrics = trainer.train_epoch(train_loader, epoch)

            # 验证
            val_metrics = trainer.evaluate(val_loader)

            # 打印进度
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val AUROC 5y: {val_metrics['auroc_5y']:.4f}")
            print(f"  Val AUROC 10y: {val_metrics['auroc_10y']:.4f}")

            # 保存最佳模型
            avg_auroc = (val_metrics['auroc_5y'] + val_metrics['auroc_10y']) / 2
            if avg_auroc > (best_auroc_5y + best_auroc_10y) / 2:
                best_auroc_5y = val_metrics['auroc_5y']
                best_auroc_10y = val_metrics['auroc_10y']
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ 模型已保存")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发")
                    break

        # 记录fold结果
        fold_results.append({
            'fold': fold + 1,
            'auroc_5y': best_auroc_5y,
            'auroc_10y': best_auroc_10y,
            'model_path': save_path,
            'class_distribution': fold_distribution
        })

    # 计算平均结果
    avg_auroc_5y = np.mean([r['auroc_5y'] for r in fold_results])
    avg_auroc_10y = np.mean([r['auroc_10y'] for r in fold_results])
    std_auroc_5y = np.std([r['auroc_5y'] for r in fold_results])
    std_auroc_10y = np.std([r['auroc_10y'] for r in fold_results])

    print(f"\n{'='*60}")
    print(f"交叉验证结果 ({model_name.upper()})")
    print(f"{'='*60}")
    print(f"5年AUROC: {avg_auroc_5y:.4f} ± {std_auroc_5y:.4f}")
    print(f"10年AUROC: {avg_auroc_10y:.4f} ± {std_auroc_10y:.4f}")

    # 保存结果
    results = build_training_metadata(
        model_name=model_name,
        data_dir=data_dir,
        wsi_feature_dir=wsi_feature_dir,
        output_dir=output_dir,
        n_folds=n_folds,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
        use_clean_data=use_clean_data
    )
    results.update({
        'timestamp': datetime.now().isoformat(),
        'followup_mode': followup_mode,
        'annual_grid_years': annual_grid_years if followup_mode == 'annual_grid' else None,
        'annual_grid_variant': annual_grid_variant if followup_mode == 'annual_grid' else None,
        'avg_auroc_5y': float(avg_auroc_5y),
        'std_auroc_5y': float(std_auroc_5y),
        'avg_auroc_10y': float(avg_auroc_10y),
        'std_auroc_10y': float(std_auroc_10y),
        'fold_results': fold_results
    })

    results_path = os.path.join(output_dir, f"{model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练MASLD进展预测模型')
    parser.add_argument('--model', type=str, required=True, choices=['m3', 'm4', 'm5'],
                        help='模型名称')
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='数据根目录')
    parser.add_argument('--wsi_dir', type=str, default='outputs/embeddings/fused',
                        help='WSI特征目录')
    parser.add_argument('--output_dir', type=str, default='outputs/models',
                        help='输出目录')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='交叉验证折数')
    parser.add_argument('--epochs', type=int, default=60,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--clean-data', action='store_true',
                        help='使用清理后的临床和随访数据')
    parser.add_argument('--followup-mode', type=str, default='fixed_visits', choices=['fixed_visits', 'annual_grid'],
                        help='M5随访输入模式')
    parser.add_argument('--annual-grid-years', type=int, default=5,
                        help='annual_grid模式的年度网格长度')
    parser.add_argument('--annual-grid-variant', type=str, default='labs_mask', choices=['labs_mask', 'lab_only', 'mask_only'],
                        help='annual_grid模式的输入变体')

    args = parser.parse_args()

    output_dir = args.output_dir
    if args.clean_data and output_dir == 'outputs/models':
        output_dir = 'outputs/models_clean'

    train_model(
        model_name=args.model,
        data_dir=args.data_dir,
        wsi_feature_dir=args.wsi_dir,
        output_dir=output_dir,
        n_folds=args.n_folds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        use_clean_data=args.clean_data,
        followup_mode=args.followup_mode,
        annual_grid_years=args.annual_grid_years,
        annual_grid_variant=args.annual_grid_variant
    )
