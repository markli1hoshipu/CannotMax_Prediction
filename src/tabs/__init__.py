# src/ui/tabs/__init__.py

from .base_tab import BaseTab
from .predict_tab import PredictTab
from .manual_tab import ManualTab
from .adb_tab import ADBTab
from .opencv_tab import OpenCVTab
from .train_tab import TrainTab
from .legacy_tab import LegacyTab

__all__ = [
    'BaseTab',
    'PredictTab',
    'ManualTab',
    'ADBTab',
    'OpenCVTab',
    'TrainTab', 
    'LegacyTab'
]

"""
战斗预测系统 - 标签页模块

包含以下标签页实现：
- PredictTab: 预测功能页
- ManualTab: 手动录入页
- ADBTab: ADB录入页
- OpenCVTab: OpenCV录入页
- LegacyTab: 旧版UI页
"""