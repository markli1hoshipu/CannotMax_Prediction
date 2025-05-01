import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *

from module.ArknightsEDDataset import ArknightsDataset
from module.TransformerNet import UnitAwareTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(csv_file):
    """预处理CSV文件，将异常值修正为合理范围"""
    """预处理CSV文件，将异常值修正为合理范围"""
    print(f"预处理数据文件: {csv_file}")

    # 读取CSV文件
    data = pd.read_csv(csv_file, header=None)
    print(f"原始数据形状: {data.shape}")

    # 检查特征范围
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    # 统计极端值
    extreme_values = (np.abs(features) > 20).sum().sum()
    if extreme_values > 0:
        print(f"发现 {extreme_values} 个绝对值大于20的特征值")

    # 检查标签
    invalid_labels = labels.apply(lambda x: x not in ['L', 'R']).sum()
    if invalid_labels > 0:
        print(f"发现 {invalid_labels} 个无效标签")

    # 输出特征的范围信息
    feature_min = features.min().min()
    feature_max = features.max().max()
    feature_mean = features.mean().mean()
    feature_std = features.std().mean()

    print(f"特征值范围: [{feature_min}, {feature_max}]")
    print(f"特征值平均值: {feature_mean:.4f}, 标准差: {feature_std:.4f}")

    # 如果需要，可以在这里对数据进行更多的预处理
    # 例如：将极端值截断到合理范围

    return data.shape[1]


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for ls, lc, rs, rc, labels in train_loader:
        ls, lc, rs, rc, labels = [x.to(device) for x in (ls, lc, rs, rc, labels)]

        optimizer.zero_grad()

        # 检查输入值范围
        if torch.isnan(ls).any() or torch.isnan(lc).any() or torch.isnan(rs).any() or torch.isnan(rc).any():
            print("警告: 输入数据包含NaN，跳过该批次")
            continue

        if torch.isinf(ls).any() or torch.isinf(lc).any() or torch.isinf(rs).any() or torch.isinf(rc).any():
            print("警告: 输入数据包含Inf，跳过该批次")
            continue

        # 确保labels严格在0-1之间
        if (labels < 0).any() or (labels > 1).any():
            print("警告: 标签值不在[0,1]范围内，进行修正")
            labels = torch.clamp(labels, 0, 1)

        try:
            outputs = model(ls, lc, rs, rc).squeeze()

            # 确保输出在合理范围内
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("警告: 模型输出包含NaN或Inf，跳过该批次")
                continue

            # 确保输出严格在0-1之间，因为BCELoss需要
            if (outputs < 0).any() or (outputs > 1).any():
                print("警告: 模型输出不在[0,1]范围内，进行修正")
                outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)

            loss = criterion(outputs, labels)

            # 检查loss是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 损失值为 {loss.item()}, 跳过该批次")
                continue

            loss.backward()

            # 梯度裁剪，避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        except RuntimeError as e:
            print(f"警告: 训练过程中出错 - {str(e)}")
            continue

    return total_loss / max(1, len(train_loader)), 100 * correct / max(1, total)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for ls, lc, rs, rc, labels in data_loader:
            ls, lc, rs, rc, labels = [x.to(device) for x in (ls, lc, rs, rc, labels)]

            # 检查输入值范围
            if torch.isnan(ls).any() or torch.isnan(lc).any() or torch.isnan(rs).any() or torch.isnan(rc).any() or \
                    torch.isinf(ls).any() or torch.isinf(lc).any() or torch.isinf(rs).any() or torch.isinf(rc).any():
                print("警告: 评估时输入数据包含NaN或Inf，跳过该批次")
                continue

            # 确保labels严格在0-1之间
            if (labels < 0).any() or (labels > 1).any():
                labels = torch.clamp(labels, 0, 1)

            try:
                outputs = model(ls, lc, rs, rc).squeeze()

                # 确保输出在合理范围内
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("警告: 评估时模型输出包含NaN或Inf，跳过该批次")
                    continue

                # 确保输出严格在0-1之间，因为BCELoss需要
                if (outputs < 0).any() or (outputs > 1).any():
                    outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)

                loss = criterion(outputs, labels)

                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            except RuntimeError as e:
                print(f"警告: 评估过程中出错 - {str(e)}")
                continue

    return total_loss / max(1, len(data_loader)), 100 * correct / max(1, total)

def stratified_random_split(dataset, test_size=0.1, seed=42):
    labels = dataset.labels  # 假设 labels 是一个 GPU tensor
    if device != 'cpu':
        labels = labels.cpu()  # 移动到 CPU 上进行操作
    labels = labels.numpy()   # 转换为 numpy array

    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )
    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices)
    )

def main():
    # 配置参数
    config = get_config()

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)

    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # 设置设备
    print(f"使用设备: {device}")

    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")

        # 设置确定性计算以增加稳定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("警告: 未检测到GPU，将在CPU上运行训练，这可能会很慢!")

    # 先预处理数据，检查是否有异常值
    num_data=preprocess_data(config['data_file'])

    # 加载数据集
    dataset = ArknightsDataset(
        config['data_file'],
        max_value=config['max_feature_value']  # 使用最大值限制
    )

    # 数据集分割
    val_size = int(0.1 * len(dataset))  # 10% 验证集
    train_size = len(dataset) - val_size

    # 划分
    train_dataset, val_dataset = stratified_random_split(dataset, test_size=config['test_size'], seed=config['seed'])

    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=0,
    )

    # 初始化模型
    model = UnitAwareTransformer(
        num_units=(num_data - 1) // 2,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['n_layers']
    ).to(device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # 训练历史记录
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 训练设置
    best_acc = 0
    best_loss = float('inf')

    # 训练循环
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer)

        # 验证
        val_loss, val_acc = evaluate(
            model, val_loader, criterion)

        # 更新学习率
        scheduler.step()

        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 保存最佳模型（基于准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model_acc.pth'))
            torch.save(model, os.path.join(config['save_dir'], 'best_model_full.pth'))
            print("保存了新的最佳准确率模型!")

        # 保存最佳模型（基于损失）
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model_loss.pth'))
            print("保存了新的最佳损失模型!")

        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'config': config
        }, os.path.join(config['save_dir'], 'latest_checkpoint.pth'))

        # 打印训练信息
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print("-" * 40)

        # 绘制并保存训练历史
        # if (epoch + 1) % 5 == 0 or epoch == config['epochs'] - 1:
        #     plot_training_history(
        #         train_losses, val_losses, train_accs, val_accs,
        #         save_path=os.path.join(config['save_dir'], 'training_history.png')
        #     )

    print(f"训练完成! 最佳验证准确率: {best_acc:.2f}%, 最佳验证损失: {best_loss:.4f}")

    # 保存最终训练历史
    # plot_training_history(
    #     train_losses, val_losses, train_accs, val_accs,
    #     save_path=os.path.join(config['save_dir'], 'final_training_history.png')
    # )


if __name__ == "__main__":
    main()
