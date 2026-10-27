from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
"""
MASLD数据集加载器
支持加载基线临床数据、患者标签、随访检验时序数据
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional


class MASLDDataset(Dataset):
    """MASLD数据集加载器"""

    ORIGINAL_CLINICAL_FEATURES = [
        'Impaired_glucose_tolerance', 'gender', 'age', 'height', 'weight', 'BMI',
        'L_MAFLD', 'ALT', 'AST', 'ALP', 'GGT', 'INR', 'TBIL', 'DBIL',
        'Creatinine', 'Uric_Acid', 'FBG', 'TG', 'TC', 'HDL', 'LDL',
        'Albumin', 'Platelet', 'HbA1c',
        'Steatosis_Grade', 'Lobular_Inflammation', 'Ballooning', 'NAS_Total',
        'SAF_S', 'SAF_A', 'SAF_F', 'Fibrosis_Stage', 'FIB4', 'NFS', 'APRI'
    ]

    CLEAN_CLINICAL_FEATURES = [
        'age', 'gender', 'BMI', 'height', 'weight',
        'ALT', 'AST', 'ALP', 'GGT', 'TBIL', 'DBIL', 'Albumin', 'INR',
        'Creatinine', 'Uric_Acid', 'FBG', 'TG', 'TC', 'HDL', 'LDL', 'HbA1c', 'Platelet',
        'Steatosis_Grade', 'Lobular_Inflammation', 'Ballooning', 'NAS_Total',
        'SAF_S', 'SAF_A',
        'Impaired_glucose_tolerance', 'L_MAFLD'
    ]

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        use_wsi_features: bool = False,
        wsi_feature_dir: Optional[str] = None,
        use_followup_labs: bool = False,
        followup_labs_dir: Optional[str] = None,
        use_clean_data: bool = False,
        followup_mode: str = 'fixed_visits',
        annual_grid_years: int = 5,
        annual_grid_variant: str = 'labs_mask'
    ):
        """
        Args:
            data_dir: 数据根目录 (包含train/val/follow_up_labs等)
            split: 'train' 或 'val'
            use_wsi_features: 是否加载WSI特征
            wsi_feature_dir: WSI特征目录路径
            use_followup_labs: 是否加载随访检验数据
            followup_labs_dir: 随访检验数据目录
        """
        self.data_dir = data_dir
        self.split = split
        self.use_wsi_features = use_wsi_features
        self.wsi_feature_dir = wsi_feature_dir
        self.use_followup_labs = use_followup_labs
        self.use_clean_data = use_clean_data
        self.followup_mode = followup_mode
        self.annual_grid_years = annual_grid_years
        self.annual_grid_variant = annual_grid_variant
        self.followup_labs_dir = followup_labs_dir or os.path.join(
            data_dir,
            'follow_up_labs_clean' if use_clean_data else 'follow_up_labs'
        )

        # 加载基线临床数据
        clinical_filename = 'clinical_data_clean.csv' if use_clean_data else 'clinical_data.csv'
        clinical_path = os.path.join(data_dir, split, clinical_filename)
        self.clinical_data = pd.read_csv(clinical_path)

        # 加载患者标签
        labels_path = os.path.join(data_dir, split, 'patient_labels.csv')
        self.labels = pd.read_csv(labels_path)

        # 合并数据
        self.data = pd.merge(
            self.clinical_data,
            self.labels,
            on='patient_id',
            how='inner'
        )

        self.patient_ids = self.data['patient_id'].tolist()

        # 定义基线临床特征列
        self.clinical_features = (
            self.CLEAN_CLINICAL_FEATURES if use_clean_data else self.ORIGINAL_CLINICAL_FEATURES
        )

        # 验证所有特征列存在
        missing_cols = set(self.clinical_features) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"缺少临床特征列: {missing_cols}")

        self.max_followup_steps = 2

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        返回一个样本的所有数据

        Returns:
            Dict包含:
                - patient_id: 患者ID
                - clinical: 基线临床特征 [35]
                - label_5y: 5年进展标签 (0/1)
                - label_10y: 10年进展标签 (0/1)
                - time_to_event: 事件发生时间(年)
                - censored: 是否截尾
                - wsi_feature: WSI特征 [768] (如果use_wsi_features=True)
                - followup_labs: 随访检验 [2, n_labs] (如果use_followup_labs=True)
        """
        row = self.data.iloc[idx]
        patient_id = row['patient_id']

        # 基线临床特征
        clinical = torch.tensor(
            row[self.clinical_features].values.astype(np.float32),
            dtype=torch.float32
        )

        # 标签
        liver_progression = int(row['liver_progression_occurred'])
        time_to_event = float(row['time_to_event_years'])
        censored = int(row['censored'])

        # 5年和10年进展标签
        label_5y = 1 if (liver_progression == 1 and time_to_event <= 5) else 0
        label_10y = liver_progression

        result = {
            'patient_id': patient_id,
            'clinical': clinical,
            'label_5y': torch.tensor(label_5y, dtype=torch.long),
            'label_10y': torch.tensor(label_10y, dtype=torch.long),
            'time_to_event': torch.tensor(time_to_event, dtype=torch.float32),
            'censored': torch.tensor(censored, dtype=torch.long)
        }

        # 加载WSI特征（如果需要）
        if self.use_wsi_features and self.wsi_feature_dir:
            # 转换患者ID格式（下划线转连字符）
            patient_id_formatted = patient_id.replace('_', '-')

            # 加载 HE 和 SR 特征
            he_path = os.path.join(self.wsi_feature_dir, f"{self.split}_HE", f"{patient_id_formatted}-HE_embed.pt")
            sr_path = os.path.join(self.wsi_feature_dir, f"{self.split}_SR", f"{patient_id_formatted}-SR_embed.pt")

            wsi_features = []

            # 加载 HE 特征
            if os.path.exists(he_path):
                he_feature = torch.load(he_path, map_location='cpu', weights_only=True)
                if he_feature.dim() > 1:
                    he_feature = he_feature.squeeze()
                wsi_features.append(he_feature)
            else:
                print(f"Warning: HE feature not found for {patient_id} at {he_path}")
                wsi_features.append(torch.zeros(768, dtype=torch.float32))

            # 加载 SR 特征
            if os.path.exists(sr_path):
                sr_feature = torch.load(sr_path, map_location='cpu', weights_only=True)
                if sr_feature.dim() > 1:
                    sr_feature = sr_feature.squeeze()
                wsi_features.append(sr_feature)
            else:
                print(f"Warning: SR feature not found for {patient_id} at {sr_path}")
                wsi_features.append(torch.zeros(768, dtype=torch.float32))

            # 融合 HE 和 SR 特征（平均）
            wsi_feature = torch.stack(wsi_features).mean(dim=0)  # [768]
            result['wsi_feature'] = wsi_feature

        # 加载随访检验数据（如果需要）
        if self.use_followup_labs:
            followup_path = os.path.join(
                self.followup_labs_dir,
                f"{patient_id}_follow_up_labs.csv"
            )
            if os.path.exists(followup_path):
                followup_df = pd.read_csv(followup_path)
                if self.followup_mode == 'annual_grid':
                    followup_df = followup_df.sort_values('days_from_biopsy')
                    if liver_progression == 1:
                        event_days = time_to_event * 365
                        followup_df = followup_df[followup_df['days_from_biopsy'] < event_days]
                    lab_cols = [c for c in followup_df.columns if c != 'days_from_biopsy']
                    grid = np.zeros((self.annual_grid_years, len(lab_cols) + 1), dtype=np.float32)
                    last_values = None
                    for year in range(self.annual_grid_years):
                        end_day = (year + 1) * 365
                        year_df = followup_df[followup_df['days_from_biopsy'] <= end_day]
                        if len(year_df):
                            observed = year_df[year_df['days_from_biopsy'] > year * 365]
                            last_values = year_df.iloc[-1][lab_cols].values.astype(np.float32)
                            grid[year, :-1] = last_values
                            grid[year, -1] = 1.0 if len(observed) else 0.0
                        elif last_values is not None:
                            grid[year, :-1] = last_values
                    if self.annual_grid_variant == 'lab_only':
                        followup_labs = grid[:, :-1]
                    elif self.annual_grid_variant == 'mask_only':
                        followup_labs = grid[:, -1:]
                    else:
                        followup_labs = grid
                    result['followup_labs'] = torch.tensor(followup_labs, dtype=torch.float32)
                    result['followup_days'] = torch.arange(1, self.annual_grid_years + 1, dtype=torch.float32) * 365
                else:
                    lab_cols = [c for c in followup_df.columns if c != 'days_from_biopsy']
                    followup_values = followup_df[lab_cols].values.astype(np.float32)
                    followup_days = followup_df['days_from_biopsy'].values.astype(np.float32)

                    n_visits = min(len(followup_df), self.max_followup_steps)
                    padded_labs = np.zeros((self.max_followup_steps, len(lab_cols)), dtype=np.float32)
                    padded_days = np.zeros(self.max_followup_steps, dtype=np.float32)
                    padded_labs[:n_visits] = followup_values[:n_visits]
                    padded_days[:n_visits] = followup_days[:n_visits]

                    result['followup_labs'] = torch.tensor(padded_labs, dtype=torch.float32)
                    result['followup_days'] = torch.tensor(padded_days, dtype=torch.float32)
            else:
                if self.followup_mode == 'annual_grid':
                    missing_dim = 11 if self.annual_grid_variant == 'lab_only' else 1 if self.annual_grid_variant == 'mask_only' else 12
                    result['followup_labs'] = torch.zeros(self.annual_grid_years, missing_dim, dtype=torch.float32)
                    result['followup_days'] = torch.arange(1, self.annual_grid_years + 1, dtype=torch.float32) * 365
                else:
                    result['followup_labs'] = torch.zeros(self.max_followup_steps, 11, dtype=torch.float32)
                    result['followup_days'] = torch.zeros(self.max_followup_steps, dtype=torch.float32)

        return result

    def get_clinical_feature_names(self) -> list:
        """返回临床特征名称列表"""
        return self.clinical_features

    def get_statistics(self) -> Dict:
        """返回数据集统计信息"""
        stats = {
            'n_samples': len(self),
            'n_progression_5y': int(self.data.apply(
                lambda x: 1 if (x['liver_progression_occurred'] == 1 and x['time_to_event_years'] <= 5) else 0,
                axis=1
            ).sum()),
            'n_progression_10y': int(self.data['liver_progression_occurred'].sum()),
            'mean_age': float(self.data['age'].mean()),
            'mean_bmi': float(self.data['BMI'].mean()),
            'gender_distribution': self.data['gender'].value_counts().to_dict(),
        }
        if 'Fibrosis_Stage' in self.data.columns:
            stats['fibrosis_distribution'] = self.data['Fibrosis_Stage'].value_counts().to_dict()
        return stats


def calculate_fib4(age: float, ast: float, alt: float, platelet: float) -> float:
    """计算FIB-4评分"""
    return (age * ast) / (platelet * np.sqrt(alt))


def calculate_nfs(
    age: float, bmi: float, diabetes: int, ast: float, alt: float,
    platelet: float, albumin: float
) -> float:
    """计算NAFLD Fibrosis Score (NFS)"""
    ast_alt_ratio = ast / alt if alt > 0 else 0
    nfs = (
        -1.675 +
        0.037 * age +
        0.094 * bmi +
        1.13 * diabetes +
        0.99 * ast_alt_ratio -
        0.013 * platelet -
        0.66 * albumin
    )
    return nfs


if __name__ == '__main__':
    # 测试数据加载器
    data_dir = f'{ROOT}/dataset'

    print("=== 测试训练集加载 ===")
    train_dataset = MASLDDataset(data_dir, split='train')
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"统计信息: {train_dataset.get_statistics()}")

    # 测试单个样本
    sample = train_dataset[0]
    print(f"\n样本0:")
    print(f"  患者ID: {sample['patient_id']}")
    print(f"  临床特征维度: {sample['clinical'].shape}")
    print(f"  5年进展标签: {sample['label_5y'].item()}")
    print(f"  10年进展标签: {sample['label_10y'].item()}")

    print("\n=== 测试验证集加载 ===")
    val_dataset = MASLDDataset(data_dir, split='val')
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"统计信息: {val_dataset.get_statistics()}")
