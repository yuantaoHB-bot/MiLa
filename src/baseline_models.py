"""
基线模型实现: M0 (FIB-4), M1 (NFS), M2 (逻辑回归)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class FIB4Model:
    """M0: FIB-4阈值模型"""

    def __init__(self, threshold: float = 2.67):
        """
        Args:
            threshold: FIB-4阈值，>threshold为高风险
                      文献常用: 2.67 (高风险), 1.30 (低风险)
        """
        self.threshold = threshold
        self.name = "M0_FIB4"

    def predict_proba(self, fib4_scores: np.ndarray) -> np.ndarray:
        """
        预测概率（简化为0/1）

        Args:
            fib4_scores: FIB-4评分数组 [N]

        Returns:
            概率数组 [N, 2]，第二列为高风险概率
        """
        probs = (fib4_scores > self.threshold).astype(float)
        return np.stack([1 - probs, probs], axis=1)

    def evaluate(self, fib4_scores: np.ndarray, labels: np.ndarray) -> Dict:
        """
        评估模型性能

        Args:
            fib4_scores: FIB-4评分 [N]
            labels: 真实标签 [N]

        Returns:
            性能指标字典
        """
        probs = self.predict_proba(fib4_scores)[:, 1]

        # 使用FIB-4分数本身作为连续预测
        auroc = roc_auc_score(labels, fib4_scores)
        auprc = average_precision_score(labels, fib4_scores)

        # 使用阈值预测计算Brier Score
        brier = brier_score_loss(labels, probs)

        # 计算敏感性和特异性
        predictions = (fib4_scores > self.threshold).astype(int)
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'model': self.name,
            'auroc': auroc,
            'auprc': auprc,
            'brier_score': brier,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'threshold': self.threshold
        }


class NFSModel:
    """M1: NFS (NAFLD Fibrosis Score) 阈值模型"""

    def __init__(self, threshold: float = 0.676):
        """
        Args:
            threshold: NFS阈值，>threshold为高风险
                      文献常用: 0.676 (高风险), -1.455 (低风险)
        """
        self.threshold = threshold
        self.name = "M1_NFS"

    def predict_proba(self, nfs_scores: np.ndarray) -> np.ndarray:
        """预测概率"""
        probs = (nfs_scores > self.threshold).astype(float)
        return np.stack([1 - probs, probs], axis=1)

    def evaluate(self, nfs_scores: np.ndarray, labels: np.ndarray) -> Dict:
        """评估模型性能"""
        probs = self.predict_proba(nfs_scores)[:, 1]

        auroc = roc_auc_score(labels, nfs_scores)
        auprc = average_precision_score(labels, nfs_scores)
        brier = brier_score_loss(labels, probs)

        predictions = (nfs_scores > self.threshold).astype(int)
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'model': self.name,
            'auroc': auroc,
            'auprc': auprc,
            'brier_score': brier,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'threshold': self.threshold
        }


class LogisticRegressionModel:
    """M2: 逻辑回归模型（基线临床特征）"""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        """
        Args:
            C: 正则化强度的倒数
            max_iter: 最大迭代次数
        """
        self.C = C
        self.max_iter = max_iter
        self.model = None
        self.name = "M2_LogisticRegression"

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型

        Args:
            X: 特征矩阵 [N, D]
            y: 标签 [N]
        """
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=42,
            class_weight='balanced'  # 处理类别不平衡
        )
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征矩阵 [N, D]

        Returns:
            概率矩阵 [N, 2]
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit()")
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """评估模型性能"""
        probs = self.predict_proba(X)[:, 1]

        auroc = roc_auc_score(y, probs)
        auprc = average_precision_score(y, probs)
        brier = brier_score_loss(y, probs)

        predictions = (probs > 0.5).astype(int)
        tp = np.sum((predictions == 1) & (y == 1))
        tn = np.sum((predictions == 0) & (y == 0))
        fp = np.sum((predictions == 1) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'model': self.name,
            'auroc': auroc,
            'auprc': auprc,
            'brier_score': brier,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5
    ) -> Tuple[Dict, list]:
        """
        5折交叉验证

        Args:
            X: 特征矩阵 [N, D]
            y: 标签 [N]
            n_folds: 折数

        Returns:
            平均性能指标和每折结果列表
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_results = []
        aurocs, auprcs, briers = [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 训练
            self.fit(X_train, y_train)

            # 评估
            metrics = self.evaluate(X_val, y_val)
            metrics['fold'] = fold + 1
            fold_results.append(metrics)

            aurocs.append(metrics['auroc'])
            auprcs.append(metrics['auprc'])
            briers.append(metrics['brier_score'])

        # 计算平均值和标准差
        avg_metrics = {
            'model': self.name,
            'auroc_mean': np.mean(aurocs),
            'auroc_std': np.std(aurocs),
            'auprc_mean': np.mean(auprcs),
            'auprc_std': np.std(auprcs),
            'brier_mean': np.mean(briers),
            'brier_std': np.std(briers)
        }

        return avg_metrics, fold_results


if __name__ == '__main__':
    # 测试基线模型
    print("=== 测试基线模型 ===\n")

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 100

    # 模拟FIB-4和NFS评分
    fib4_scores = np.random.uniform(0.5, 5.0, n_samples)
    nfs_scores = np.random.uniform(-3.0, 3.0, n_samples)

    # 模拟标签（与评分正相关）
    labels = ((fib4_scores > 2.0) | (nfs_scores > 0.5)).astype(int)

    # 测试M0
    print("M0 (FIB-4):")
    m0 = FIB4Model(threshold=2.67)
    m0_metrics = m0.evaluate(fib4_scores, labels)
    for k, v in m0_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 测试M1
    print("\nM1 (NFS):")
    m1 = NFSModel(threshold=0.676)
    m1_metrics = m1.evaluate(nfs_scores, labels)
    for k, v in m1_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 测试M2
    print("\nM2 (Logistic Regression):")
    X = np.column_stack([fib4_scores, nfs_scores, np.random.randn(n_samples, 3)])
    m2 = LogisticRegressionModel()
    avg_metrics, fold_results = m2.cross_validate(X, labels, n_folds=5)

    print("  交叉验证结果:")
    for k, v in avg_metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print("\n✓ 基线模型测试完成")
