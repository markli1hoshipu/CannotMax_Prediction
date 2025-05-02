import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import sys
from module.TransformerNet import UnitAwareTransformer
from utils import get_config


# 配置参数（需要与训练时一致）


class SafetyChecker:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        CONFIG = get_config()
        self.model = self._load_model()
        self.model.eval()
        
        # 创建输出路径
        self.clean_csv_path = CONFIG["data_file"].replace(".csv", "_clean.csv")
        
    def _load_model(self):
        """加载训练好的模型"""
        CONFIG = get_config()
        model = UnitAwareTransformer(
            num_units=CONFIG["num_units"],
            embed_dim=CONFIG["embed_dim"],
            num_heads=CONFIG["num_heads"],
            num_layers=CONFIG["n_layers"]
        )
        #model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
        #model = torch.load(CONFIG["model_path"], map_location=device)
        try:
            model = torch.load(CONFIG["model_dir"], map_location=self.device, weights_only=False)
        except TypeError:  # 如果旧版本 PyTorch 不认识 weights_only
            model = torch.load(CONFIG["model_dir"], map_location=self.device)

        return model.to(self.device)
    
    def _process_row(self, row):
        CONFIG = get_config()
        """处理单行数据（与ArknightsDataset保持一致）"""
        features = row[:-1].astype(np.float32)
        midpoint = len(features) // 2
        
        # 处理左右特征
        left_sign = np.sign(features[:midpoint])
        left_count = np.clip(np.abs(features[:midpoint]), 0, CONFIG["max_feature_value"])
        right_sign = np.sign(features[midpoint:])
        right_count = np.clip(np.abs(features[midpoint:]), 0, CONFIG["max_feature_value"])
        
        # 转换为张量并添加batch维度
        return (
            torch.tensor(left_sign, dtype=torch.float32).unsqueeze(0).to(self.device),
            torch.tensor(left_count, dtype=torch.float32).unsqueeze(0).to(self.device),
            torch.tensor(right_sign, dtype=torch.float32).unsqueeze(0).to(self.device),
            torch.tensor(right_count, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
    
    def check_csv(self):
        CONFIG = get_config()
        """检查整个CSV文件并保存干净版本"""
        problematic_rows = []
        
        # 初始化干净文件（清空已存在内容）
        pd.DataFrame().to_csv(self.clean_csv_path, mode='w', header=False, index=False)
        
        chunk_size = 1000
        total_rows = sum(1 for _ in open(CONFIG["data_file"]))
        
        with tqdm(total=total_rows, desc="Checking rows") as pbar:
            for chunk in pd.read_csv(CONFIG["data_file"], header=None, chunksize=chunk_size):
                normal_chunk = []
                for idx, row in chunk.iterrows():
                    global_idx = pbar.n % total_rows + 1
                    try:
                        tensors = self._process_row(row.values)
                        with torch.no_grad():
                            output = self.model(*tensors)
                            if torch.isnan(output).any() or torch.isinf(output).any():
                                problematic_rows.append(global_idx)
                            else:
                                # 保留正常行数据
                                normal_chunk.append(row.values.tolist())
                    except Exception as e:
                        problematic_rows.append((global_idx, str(e)))
                    pbar.update(1)
                
                # 将本chunk的正常行写入文件
                if normal_chunk:
                    pd.DataFrame(normal_chunk).to_csv(
                        self.clean_csv_path,
                        mode='a',
                        header=False,
                        index=False
                    )
        
        print(f"\n已保存干净数据至: {self.clean_csv_path}")
        print("\n检查完成！发现异常的行：")
        for row in problematic_rows:
            print(f"行号 {row[0] if isinstance(row, tuple) else row}: ",
                  row[1] if isinstance(row, tuple) else "输出包含NaN/Inf")

if __name__ == "__main__":
    checker = SafetyChecker()
    checker.check_csv()