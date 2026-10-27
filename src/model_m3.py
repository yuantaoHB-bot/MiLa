"""
M3: WSI-only模型
输入: GigaPath特征 [768]
输出: 5年/10年进展预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class M3_WSIOnly(nn.Module):
    """M3: WSI-only模型"""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.4,
        num_classes: int = 2
    ):
        """
        Args:
            input_dim: GigaPath特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout比例
            num_classes: 分类数（2: 进展/未进展）
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类头（双任务：5年和10年）
        self.classifier_5y = nn.Linear(hidden_dim // 2, num_classes)
        self.classifier_10y = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, wsi_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            wsi_features: WSI特征 [B, 768]

        Returns:
            Dict包含:
                - logits_5y: 5年预测logits [B, 2]
                - logits_10y: 10年预测logits [B, 2]
                - features: 提取的特征 [B, hidden_dim//2]
        """
        # 特征提取
        features = self.feature_extractor(wsi_features)  # [B, hidden_dim//2]

        # 双任务预测
        logits_5y = self.classifier_5y(features)  # [B, 2]
        logits_10y = self.classifier_10y(features)  # [B, 2]

        return {
            'logits_5y': logits_5y,
            'logits_10y': logits_10y,
            'features': features
        }

    def predict_proba(self, wsi_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测概率

        Args:
            wsi_features: WSI特征 [B, 768]

        Returns:
            (prob_5y, prob_10y): 5年和10年进展概率 [B, 2]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(wsi_features)
            prob_5y = F.softmax(outputs['logits_5y'], dim=1)
            prob_10y = F.softmax(outputs['logits_10y'], dim=1)
        return prob_5y, prob_10y


class M3_Trainer:
    """M3模型训练器"""

    def __init__(
        self,
        model: M3_WSIOnly,
        device: str = 'cuda',
        lr: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            model: M3模型
            device: 计算设备
            lr: 学习率
            weight_decay: 权重衰减
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch

        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        total_loss_5y = 0.0
        total_loss_10y = 0.0
        n_batches = 0

        for batch in train_loader:
            wsi_features = batch['wsi_feature'].to(self.device)
            labels_5y = batch['label_5y'].to(self.device)
            labels_10y = batch['label_10y'].to(self.device)

            # 前向传播
            outputs = self.model(wsi_features)

            # 计算损失（双任务）
            loss_5y = self.criterion(outputs['logits_5y'], labels_5y)
            loss_10y = self.criterion(outputs['logits_10y'], labels_10y)
            loss = loss_5y + loss_10y

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_loss_5y += loss_5y.item()
            total_loss_10y += loss_10y.item()
            n_batches += 1

        return {
            'loss': total_loss / n_batches,
            'loss_5y': total_loss_5y / n_batches,
            'loss_10y': total_loss_10y / n_batches
        }

    def evaluate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        评估模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            评估指标字典
        """
        self.model.eval()
        total_loss = 0.0
        all_probs_5y = []
        all_probs_10y = []
        all_labels_5y = []
        all_labels_10y = []

        with torch.no_grad():
            for batch in val_loader:
                wsi_features = batch['wsi_feature'].to(self.device)
                labels_5y = batch['label_5y'].to(self.device)
                labels_10y = batch['label_10y'].to(self.device)

                outputs = self.model(wsi_features)

                # 计算损失
                loss_5y = self.criterion(outputs['logits_5y'], labels_5y)
                loss_10y = self.criterion(outputs['logits_10y'], labels_10y)
                loss = loss_5y + loss_10y
                total_loss += loss.item()

                # 收集预测和标签
                probs_5y = F.softmax(outputs['logits_5y'], dim=1)
                probs_10y = F.softmax(outputs['logits_10y'], dim=1)

                all_probs_5y.append(probs_5y[:, 1].cpu())
                all_probs_10y.append(probs_10y[:, 1].cpu())
                all_labels_5y.append(labels_5y.cpu())
                all_labels_10y.append(labels_10y.cpu())

        # 合并所有批次
        all_probs_5y = torch.cat(all_probs_5y).numpy()
        all_probs_10y = torch.cat(all_probs_10y).numpy()
        all_labels_5y = torch.cat(all_labels_5y).numpy()
        all_labels_10y = torch.cat(all_labels_10y).numpy()

        # 计算AUROC和AUPRC
        from sklearn.metrics import roc_auc_score, average_precision_score

        auroc_5y = roc_auc_score(all_labels_5y, all_probs_5y)
        auroc_10y = roc_auc_score(all_labels_10y, all_probs_10y)
        auprc_5y = average_precision_score(all_labels_5y, all_probs_5y)
        auprc_10y = average_precision_score(all_labels_10y, all_probs_10y)

        return {
            'loss': total_loss / len(val_loader),
            'auroc_5y': auroc_5y,
            'auroc_10y': auroc_10y,
            'auprc_5y': auprc_5y,
            'auprc_10y': auprc_10y
        }

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 60,
        early_stopping_patience: int = 10,
        save_path: str = None
    ) -> Dict:
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            save_path: 模型保存路径

        Returns:
            训练历史
        """
        best_auroc = 0.0
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auroc_5y': [],
            'val_auroc_10y': []
        }

        for epoch in range(num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)

            # 验证
            val_metrics = self.evaluate(val_loader)

            # 记录历史
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_auroc_5y'].append(val_metrics['auroc_5y'])
            history['val_auroc_10y'].append(val_metrics['auroc_10y'])

            # 打印进度
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val AUROC 5y: {val_metrics['auroc_5y']:.4f}")
            print(f"  Val AUROC 10y: {val_metrics['auroc_10y']:.4f}")

            # 早停检查
            avg_auroc = (val_metrics['auroc_5y'] + val_metrics['auroc_10y']) / 2
            if avg_auroc > best_auroc:
                best_auroc = avg_auroc
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  ✓ 模型已保存 (AUROC: {best_auroc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"早停触发 (patience={early_stopping_patience})")
                    break

        return history


if __name__ == '__main__':
    # 测试M3模型
    print("=== 测试M3模型 ===\n")

    # 创建模型
    model = M3_WSIOnly(
        input_dim=768,
        hidden_dim=512,
        dropout=0.4
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    batch_size = 16
    wsi_features = torch.randn(batch_size, 768)

    outputs = model(wsi_features)
    print(f"\n输入形状: {wsi_features.shape}")
    print(f"输出形状:")
    print(f"  logits_5y: {outputs['logits_5y'].shape}")
    print(f"  logits_10y: {outputs['logits_10y'].shape}")
    print(f"  features: {outputs['features'].shape}")

    # 测试预测
    prob_5y, prob_10y = model.predict_proba(wsi_features)
    print(f"\n预测概率:")
    print(f"  prob_5y: {prob_5y.shape}, sum={prob_5y[0].sum():.4f}")
    print(f"  prob_10y: {prob_10y.shape}, sum={prob_10y[0].sum():.4f}")

    print("\n✓ M3模型测试完成")
