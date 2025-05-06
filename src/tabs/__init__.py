# src/ui/tabs/__init__.py

from .base_tab import BaseTab
from .adb_tab import ADBTab
from .opencv_tab import OpenCVTab
from .photopredict_tab import PhotoPredictTab
__all__ = [
    'BaseTab',
    'PredictTab',
    'ADBTab',
    'OpenCVTab',
    'PhotoPredictTab'
]