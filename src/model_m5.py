"""
M5: 完整时序模型
输入: WSI特征 [768] + 基线临床 [33] + 随访检验时序 [3, n_labs]
输出: 5年/10年进展预测
架构: 静态多模态 + 时序编码器 + LSTM + 风险预测头
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        return x + self.pe[:, :x.size(1), :]


class LabTimeSeriesEncoder(nn.Module):
    """随访检验时序编码器"""

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: 检验指标数量
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            dropout: Dropout比例
        """
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, lab_sequence: torch.Tensor) -> torch.Tensor:
        """
        编码随访检验时序

        Args:
            lab_sequence: [B, T, input_dim] (T=3个时点)

        Returns:
            编码后的序列 [B, T, hidden_dim]
        """
        # 投影到隐藏维度
        x = self.input_proj(lab_sequence)  # [B, T, hidden_dim]

        # 添加位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer(x)  # [B, T, hidden_dim]

        # 输出投影
        x = self.output_proj(x)

        return x


class M5_FullTemporal(nn.Module):
    """M5: 完整时序模型"""

    def __init__(
        self,
        wsi_dim: int = 768,
        clinical_dim: int = 35,
        lab_dim: int = 11,
        hidden_dim: int = 512,
        fusion_dim: int = 256,
        temporal_dim: int = 128,
        lstm_hidden: int = 128,
        dropout: float = 0.4,
        num_classes: int = 2
    ):
        """
        Args:
            wsi_dim: WSI特征维度
            clinical_dim: 临床特征维度
            lab_dim: 随访检验指标数
            hidden_dim: 隐藏层维度
            fusion_dim: 静态融合维度
            temporal_dim: 时序编码维度
            lstm_hidden: LSTM隐藏维度
            dropout: Dropout比例
            num_classes: 分类数
        """
        super().__init__()

        # WSI编码器
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

        # 静态特征融合（简化版Cross-Attention）
        self.static_fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 时序编码器
        self.temporal_encoder = LabTimeSeriesEncoder(
            input_dim=lab_dim,
            hidden_dim=temporal_dim,
            num_layers=2,
            num_heads=4,
            dropout=dropout
        )

        # LSTM处理时序
        self.lstm = nn.LSTM(
            input_size=temporal_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=True
        )

        # 最终融合层
        final_input_dim = fusion_dim + lstm_hidden * 2  # 静态 + LSTM双向输出
        self.final_fusion = nn.Sequential(
            nn.Linear(final_input_dim, fusion_dim),
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
        clinical_features: torch.Tensor,
        lab_sequence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            wsi_features: WSI特征 [B, 768]
            clinical_features: 临床特征 [B, 33]
            lab_sequence: 随访检验序列 [B, 3, 10]

        Returns:
            Dict包含:
                - logits_5y: 5年预测 [B, 2]
                - logits_10y: 10年预测 [B, 2]
                - features: 最终特征 [B, fusion_dim]
                - static_features: 静态融合特征 [B, fusion_dim]
                - temporal_features: 时序特征 [B, lstm_hidden*2]
        """
        # 1. 编码静态特征
        wsi_encoded = self.wsi_encoder(wsi_features)  # [B, hidden_dim]
        clinical_encoded = self.clinical_encoder(clinical_features)  # [B, hidden_dim//2]

        # 2. 静态特征融合
        static_concat = torch.cat([wsi_encoded, clinical_encoded], dim=1)
        static_fused = self.static_fusion(static_concat)  # [B, fusion_dim]

        # 3. 时序编码
        temporal_encoded = self.temporal_encoder(lab_sequence)  # [B, 3, temporal_dim]

        # 4. LSTM处理时序
        lstm_out, (h_n, c_n) = self.lstm(temporal_encoded)  # [B, 3, lstm_hidden*2]
        # 取最后一个时间步的输出
        temporal_features = lstm_out[:, -1, :]  # [B, lstm_hidden*2]

        # 5. 最终融合
        final_concat = torch.cat([static_fused, temporal_features], dim=1)
        final_features = self.final_fusion(final_concat)  # [B, fusion_dim]

        # 6. 分类
        logits_5y = self.classifier_5y(final_features)
        logits_10y = self.classifier_10y(final_features)

        return {
            'logits_5y': logits_5y,
            'logits_10y': logits_10y,
            'features': final_features,
            'static_features': static_fused,
            'temporal_features': temporal_features
        }

    def predict_proba(
        self,
        wsi_features: torch.Tensor,
        clinical_features: torch.Tensor,
        lab_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测概率"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(wsi_features, clinical_features, lab_sequence)
            prob_5y = F.softmax(outputs['logits_5y'], dim=1)
            prob_10y = F.softmax(outputs['logits_10y'], dim=1)
        return prob_5y, prob_10y


class M5_Trainer:
    """M5模型训练器"""

    def __init__(
        self,
        model: M5_FullTemporal,
        device: str = 'cuda',
        lr: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            model: M5模型
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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

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
            lab_sequence = batch['followup_labs'].to(self.device)
            labels_5y = batch['label_5y'].to(self.device)
            labels_10y = batch['label_10y'].to(self.device)

            # 前向传播
            outputs = self.model(wsi_features, clinical_features, lab_sequence)

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
                lab_sequence = batch['followup_labs'].to(self.device)
                labels_5y = batch['label_5y'].to(self.device)
                labels_10y = batch['label_10y'].to(self.device)

                outputs = self.model(wsi_features, clinical_features, lab_sequence)

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
    # 测试M5模型
    print("=== 测试M5模型 ===\n")

    # 创建模型
    model = M5_FullTemporal(
        wsi_dim=1280,
        clinical_dim=33,
        lab_dim=10,
        hidden_dim=512,
        fusion_dim=256,
        temporal_dim=128,
        lstm_hidden=128,
        dropout=0.4
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    batch_size = 16
    wsi_features = torch.randn(batch_size, 768)
    clinical_features = torch.randn(batch_size, 33)
    lab_sequence = torch.randn(batch_size, 3, 10)  # 3个时点，10个指标

    outputs = model(wsi_features, clinical_features, lab_sequence)
    print(f"\n输入形状:")
    print(f"  WSI: {wsi_features.shape}")
    print(f"  Clinical: {clinical_features.shape}")
    print(f"  Lab sequence: {lab_sequence.shape}")
    print(f"\n输出形状:")
    print(f"  logits_5y: {outputs['logits_5y'].shape}")
    print(f"  logits_10y: {outputs['logits_10y'].shape}")
    print(f"  features: {outputs['features'].shape}")
    print(f"  static_features: {outputs['static_features'].shape}")
    print(f"  temporal_features: {outputs['temporal_features'].shape}")

    # 测试预测
    prob_5y, prob_10y = model.predict_proba(wsi_features, clinical_features, lab_sequence)
    print(f"\n预测概率:")
    print(f"  prob_5y: {prob_5y.shape}")
    print(f"  prob_10y: {prob_10y.shape}")

    print("\n✓ M5模型测试完成")
