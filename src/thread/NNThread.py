import os
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal, QObject, Slot
from ..module.TransformerNet import UnitAwareTransformer
from ..utils import get_config


class NNWorker(QObject):
    """
    改进版神经网络工作器，优化了线程安全和错误处理
    """
    prediction_ready = Signal(float)         # 预测结果信号 (0-1之间的值)
    error_occurred = Signal(str)             # 错误信息信号
    model_loaded = Signal()                  # 模型加载完成信号
    predict_requested = Signal(list, list)   # 请求预测信号，供线程调用

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.model = None
        self._model_loaded = False

    @Slot()
    def load_model(self, model_path=get_config()['model_dir']):
        """自动选择加载策略"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            # 先尝试安全模式
            try:
                import torch.serialization
                torch.serialization.add_safe_globals([UnitAwareTransformer])
                try:
                    model = torch.load(model_path, map_location=self.device, weights_only=False)
                except TypeError:  # 如果旧版本 PyTorch 不认识 weights_only
                    model = torch.load(model_path, map_location=self.device)

            except:
                print("警告: 使用非安全模式加载模型，请确保模型来源可信")
                # 加载模型权重
                try:
                    model = torch.load(model_path, map_location=self.device, weights_only=False)
                except TypeError:  # 如果旧版本 PyTorch 不认识 weights_only
                    model = torch.load(model_path, map_location=self.device)

            model.eval()
            self.model = model.to(self.device)
            self._model_loaded = True
            self.model_loaded.emit()

        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            self.error_occurred.emit(error_msg)

    @Slot(list, list)
    def predict(self, left_counts, right_counts):
        """线程安全的预测方法"""
        try:
            if not self._model_loaded:
                raise RuntimeError("模型未加载或加载失败")

            if not all(isinstance(x, (int, float)) for x in left_counts + right_counts):
                raise ValueError("输入必须为数字列表")

            def prepare_input(data):
                data = np.array(data, dtype=np.float32)
                signs = np.sign(data)
                counts = np.abs(data)
                return (
                    torch.from_numpy(signs).unsqueeze(0).to(self.device),
                    torch.from_numpy(counts).unsqueeze(0).to(self.device)
                )

            left_signs, left_counts = prepare_input(left_counts)
            right_signs, right_counts = prepare_input(right_counts)

            with torch.no_grad():
                prediction = self.model(
                    left_signs, left_counts, 
                    right_signs, right_counts
                ).item()

                if not np.isfinite(prediction):
                    prediction = 0.5
                    print("警告: 预测结果为非数值，已重置为0.5")
                prediction = max(0.0, min(1.0, prediction))

            self.prediction_ready.emit(prediction)

        except Exception as e:
            error_msg = f"预测错误: {str(e)}"
            self.error_occurred.emit(error_msg)
            self.prediction_ready.emit(0.5)


class NNThread(QThread):
    """
    改进版神经网络线程，优化了资源管理和错误处理
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = NNWorker(self._detect_device())
        self.worker.moveToThread(self)

        # 信号连接
        self.worker.error_occurred.connect(self._handle_error)
        self.worker.predict_requested.connect(self.worker.predict)
        self.started.connect(self.worker.load_model)

    @staticmethod
    def _detect_device():
        """自动检测最佳计算设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        print("警告: 使用CPU模式，性能可能受限")
        return "cpu"

    def request_prediction(self, left_counts, right_counts):
        """线程安全的预测请求"""
        left_counts = list(map(int, left_counts))
        right_counts = list(map(int, right_counts))
        self.worker.predict_requested.emit(left_counts, right_counts)

    def _handle_error(self, error_msg):
        """集中错误处理"""
        print(f"神经网络错误: {error_msg}")

    def safe_stop(self):
        """安全停止线程"""
        if self.isRunning():
            self.quit()
            if not self.wait(2000):
                self.terminate()
            print("神经网络线程已安全停止")

    def __del__(self):
        """析构时确保线程停止"""
        self.safe_stop()
