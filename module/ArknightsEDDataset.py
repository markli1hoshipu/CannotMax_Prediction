import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ArknightsDataset(Dataset):
    def __init__(self, csv_file, max_value=None):
        data = pd.read_csv(csv_file, header=None)
        features = data.iloc[:, :-1].values.astype(np.float32)
        labels = data.iloc[:, -1].map({'L': 0, 'R': 1}).values
        labels = np.where((labels != 0) & (labels != 1), 0, labels).astype(np.float32)

        # 分割双方单位
        feature_count = features.shape[1]
        midpoint = feature_count // 2
        left_counts = np.abs(features[:, :midpoint])
        right_counts = np.abs(features[:, midpoint:])
        left_signs = np.sign(features[:, :midpoint])
        right_signs = np.sign(features[:, midpoint:])

        if max_value is not None:
            left_counts = np.clip(left_counts, 0, max_value)
            right_counts = np.clip(right_counts, 0, max_value)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 转换为 PyTorch 张量，并一次性加载到 GPU
        self.left_signs = torch.from_numpy(left_signs).to(device)
        self.right_signs = torch.from_numpy(right_signs).to(device)
        self.left_counts = torch.from_numpy(left_counts).to(device)
        self.right_counts = torch.from_numpy(right_counts).to(device)
        self.labels = torch.from_numpy(labels).float().to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.left_signs[idx],
            self.left_counts[idx],
            self.right_signs[idx],
            self.right_counts[idx],
            self.labels[idx],
        )