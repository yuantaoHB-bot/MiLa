"""
M4: 静态多模态模型
输入: WSI特征 [768] + 基线临床特征 [33]
输出: 5年/10年进展预测
使用Cross-Attention融合多模态特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class CrossAttentionFusion(nn.Module):
    """Cross-Attention融合模块"""

    def __init__(
        self,
        wsi_dim: int,
        clinical_dim: int,
        hidden_dim: int,
        num_heads: int = 4
    ):
        """
        Args:
            wsi_dim: WSI特征维度
            clinical_dim: 临床特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
        """
        super().__init__()

        self.wsi_proj = nn.Linear(wsi_dim, hidden_dim)
        self.clinical_proj = nn.Linear(clinical_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(
        self,
        wsi_features: torch.Tensor,
        clinical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-Attention融合

        Args:
            wsi_features: WSI特征 [B, wsi_dim]
            clinical_features: 临床特征 [B, clinical_dim]

        Returns:
            融合特征 [B, hidden_dim]
        """
        # 投影到相同维度
        wsi_proj = self.wsi_proj(wsi_features).unsqueeze(1)  # [B, 1, hidden_dim]
        clinical_proj = self.clinical_proj(clinical_features).unsqueeze(1)  # [B, 1, hidden_dim]

        # Cross-Attention: WSI作为query, Clinical作为key/value
        attn_output, _ = self.cross_attention(
            query=wsi_proj,
            key=clinical_proj,
            value=clinical_proj
        )  # [B, 1, hidden_dim]

        # 残差连接
        fused = self.layer_norm(wsi_proj + attn_output)

        # FFN
        fused = fused + self.ffn(fused)
        fused = self.layer_norm(fused)

        return fused.squeeze(1)  # [B, hidden_dim]


class M4_StaticMultimodal(nn.Module):
    """M4: 静态多模态模型"""

    def __init__(
        self,
        wsi_dim: int = 768,
        clinical_dim: int = 35,
        hidden_dim: int = 512,
        fusion_dim: int = 256,
        dropout: float = 0.4,
        num_classes: int = 2
    ):
        """
        Args:
            wsi_dim: WSI特征维度
            clinical_dim: 临床特征维度
            hidden_dim: 隐藏层维度
            fusion_dim: 融合后维度
            dropout: Dropout比例
            num_classes: 分类数
        """
        super().__init__()

        self.wsi_dim = wsi_dim
        self.clinical_dim = clinical_dim
        self.hidden_dim = hidden_dim

        # WSI特征编码器
        self.wsi_encoder = nn.Sequential(
            nn.Linear(wsi_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 临床特征编码器
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Cross-Attention融合
        self.fusion = CrossAttentionFusion(
            wsi_dim=hidden_dim,
            clinical_dim=hidden_dim // 2,
            hidden_dim=fusion_dim,
            num_heads=4
        )

        # 融合后的特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类头
        self.classifier_5y = nn.Linear(fusion_dim, num_classes)
        self.classifier_10y = nn.Linear(fusion_dim, num_classes)

    def forward(
        self,
        wsi_features: torch.Tensor,
        clinical_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            wsi_features: WSI特征 [B, 768]
            clinical_features: 临床特征 [B, 33]

        Returns:
            Dict包含:
                - logits_5y: 5年预测logits [B, 2]
                - logits_10y: 10年预测logits [B, 2]
                - features: 融合特征 [B, fusion_dim]
                - wsi_encoded: WSI编码 [B, hidden_dim]
                - clinical_encoded: 临床编码 [B, hidden_dim//2]
        """
        # 编码
        wsi_encoded = self.wsi_encoder(wsi_features)  # [B, hidden_dim]
        clinical_encoded = self.clinical_encoder(clinical_features)  # [B, hidden_dim//2]

        # 融合
        fused = self.fusion(wsi_encoded, clinical_encoded)  # [B, fusion_dim]

        # 特征提取
        features = self.feature_extractor(fused)  # [B, fusion_dim]

        # 分类
        logits_5y = self.classifier_5y(features)
        logits_10y = self.classifier_10y(features)

        return {
            'logits_5y': logits_5y,
            'logits_10y': logits_10y,
            'features': features,
            'wsi_encoded': wsi_encoded,
            'clinical_encoded': clinical_encoded
        }

    def predict_proba(
        self,
        wsi_features: torch.Tensor,
        clinical_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测概率"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(wsi_features, clinical_features)
            prob_5y = F.softmax(outputs['logits_5y'], dim=1)
            prob_10y = F.softmax(outputs['logits_10y'], dim=1)
        return prob_5y, prob_10y


class M4_Trainer:
    """M4模型训练器"""

    def __init__(
        self,
        model: M4_StaticMultimodal,
        device: str = 'cuda',
        lr: float = 5e-5,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            model: M4模型
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
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_loss_5y = 0.0
        total_loss_10y = 0.0
        n_batches = 0

        for batch in train_loader:
            wsi_features = batch['wsi_feature'].to(self.device)
            clinical_features = batch['clinical'].to(self.device)
            labels_5y = batch['label_5y'].to(self.device)
            labels_10y = batch['label_10y'].to(self.device)

            # 前向传播
            outputs = self.model(wsi_features, clinical_features)

            # 计算损失
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
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        all_probs_5y = []
        all_probs_10y = []
        all_labels_5y = []
        all_labels_10y = []

        with torch.no_grad():
            for batch in val_loader:
                wsi_features = batch['wsi_feature'].to(self.device)
                clinical_features = batch['clinical'].to(self.device)
                labels_5y = batch['label_5y'].to(self.device)
                labels_10y = batch['label_10y'].to(self.device)

                outputs = self.model(wsi_features, clinical_features)

                # 计算损失
                loss_5y = self.criterion(outputs['logits_5y'], labels_5y)
                loss_10y = self.criterion(outputs['logits_10y'], labels_10y)
                loss = loss_5y + loss_10y
                total_loss += loss.item()

                # 收集预测
                probs_5y = F.softmax(outputs['logits_5y'], dim=1)
                probs_10y = F.softmax(outputs['logits_10y'], dim=1)

                all_probs_5y.append(probs_5y[:, 1].cpu())
                all_probs_10y.append(probs_10y[:, 1].cpu())
                all_labels_5y.append(labels_5y.cpu())
                all_labels_10y.append(labels_10y.cpu())

        # 计算指标
        all_probs_5y = torch.cat(all_probs_5y).numpy()
        all_probs_10y = torch.cat(all_probs_10y).numpy()
        all_labels_5y = torch.cat(all_labels_5y).numpy()
        all_labels_10y = torch.cat(all_labels_10y).numpy()

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


if __name__ == '__main__':
    # 测试M4模型
    print("=== 测试M4模型 ===\n")

    # 创建模型
    model = M4_StaticMultimodal(
        wsi_dim=1280,
        clinical_dim=33,
        hidden_dim=512,
        fusion_dim=256,
        dropout=0.4
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    batch_size = 16
    wsi_features = torch.randn(batch_size, 768)
    clinical_features = torch.randn(batch_size, 33)

    outputs = model(wsi_features, clinical_features)
    print(f"\n输入形状:")
    print(f"  WSI: {wsi_features.shape}")
    print(f"  Clinical: {clinical_features.shape}")
    print(f"\n输出形状:")
    print(f"  logits_5y: {outputs['logits_5y'].shape}")
    print(f"  logits_10y: {outputs['logits_10y'].shape}")
    print(f"  features: {outputs['features'].shape}")
    print(f"  wsi_encoded: {outputs['wsi_encoded'].shape}")
    print(f"  clinical_encoded: {outputs['clinical_encoded'].shape}")

    # 测试预测
    prob_5y, prob_10y = model.predict_proba(wsi_features, clinical_features)
    print(f"\n预测概率:")
    print(f"  prob_5y: {prob_5y.shape}")
    print(f"  prob_10y: {prob_10y.shape}")

    print("\n✓ M4模型测试完成")
