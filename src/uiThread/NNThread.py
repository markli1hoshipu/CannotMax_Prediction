import os
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal, QObject, Qt, Q_ARG
from PySide6.QtWidgets import QMessageBox

class NNWorker(QObject):
    """
    神经网络工作器，用于在子线程中执行神经网络相关操作
    """
    prediction_ready = Signal(float)  # 预测完成信号，携带预测结果
    error_occurred = Signal(str)     # 错误信号，携带错误信息
    model_loaded = Signal()          # 模型加载完成信号

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.model = None

    def load_model(self, model_path='models/best_model_full.pth'):
        """在子线程中加载模型"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"未找到训练好的模型文件 '{model_path}'，请先训练模型")

            try:
                model = torch.load(model_path, map_location=self.device, weights_only=False)
            except TypeError:  # 如果旧版本 PyTorch 不认识 weights_only
                model = torch.load(model_path, map_location=self.device)
            
            model.eval()
            self.model = model.to(self.device)
            self.model_loaded.emit()

        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            if "missing keys" in str(e):
                error_msg += "\n可能是模型结构不匹配，请重新训练模型"
            self.error_occurred.emit(error_msg)

    def predict(self, left_counts, right_counts):
        """在子线程中执行预测"""
        try:
            if self.model is None:
                raise RuntimeError("模型未正确初始化")

            # 转换为张量并处理符号和绝对值
            left_signs = torch.sign(torch.tensor(left_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
            left_counts = torch.abs(torch.tensor(left_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
            right_signs = torch.sign(torch.tensor(right_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
            right_counts = torch.abs(torch.tensor(right_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)

            # 预测流程
            with torch.no_grad():
                prediction = self.model(left_signs, left_counts, right_signs, right_counts).item()

                # 确保预测值在有效范围内
                if np.isnan(prediction) or np.isinf(prediction):
                    print("警告: 预测结果包含NaN或Inf，返回默认值0.5")
                    prediction = 0.5

                # 检查预测结果是否在[0,1]范围内
                if prediction < 0 or prediction > 1:
                    prediction = max(0, min(1, prediction))

            self.prediction_ready.emit(prediction)

        except Exception as e:
            error_msg = f"预测时发生错误: {str(e)}"
            self.error_occurred.emit(error_msg)
            self.prediction_ready.emit(0.5)  # 发送默认值


class NNThread(QThread):
    """
    神经网络线程，封装了神经网络工作器
    """
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            print("未检测到NVIDIA Cuda,请参考手册检查。将使用CPU运行")
        super().__init__()
        self.worker = NNWorker(device)
        self.worker.moveToThread(self)
        
        # 连接信号
        self.worker.error_occurred.connect(self.handle_error)
        
        # 线程启动时初始化
        self.started.connect(self.initialize)

    def initialize(self):
        """线程启动时初始化"""
        self.worker.load_model()

    def handle_error(self, error_msg):
        """处理错误信号"""
        # 这里可以记录日志或执行其他错误处理
        print(f"神经网络线程错误: {error_msg},尝试终止进程")
        self.stop()

    def request_prediction(self, left_counts, right_counts):
        """请求预测"""
        # 使用元调用确保在正确的线程中执行
        self.metaObject().invokeMethod(
            self.worker, 
            'predict', 
            Qt.QueuedConnection,
            Q_ARG(list, left_counts),
            Q_ARG(list, right_counts)
        )

    def stop(self):
        """停止线程"""
        self.quit()
        self.wait()