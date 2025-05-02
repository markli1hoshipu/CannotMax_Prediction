# tabs/base_tab.py
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtUiTools import QUiLoader
from ..utils import get_ui_path  # 导入工具函数
import os

class BaseTab(QWidget):
    """Tab页基类（子类必须指定UI文件名）"""
    UI_NAME = None  # 子类覆盖此属性（如 'adb_tab'）

    def __init__(self, parent=None):
        super().__init__(parent)
        if self.UI_NAME is None:
            raise NotImplementedError("子类必须指定UI_NAME")
            
        self.ui_path = get_ui_path(self.UI_NAME)  # 使用工具函数
        self._load_ui()
        self.setup_ui()
        self.setup_connections()
    
    def _load_ui(self):
        """加载UI文件"""
        if not os.path.exists(self.ui_path):
            raise FileNotFoundError(f"UI文件不存在: {self.ui_path}")
        
        loader = QUiLoader()
        self.ui = loader.load(self.ui_path)
        
        # layout = QVBoxLayout(self)
        # layout.addWidget(self.ui)
        # self.setLayout(layout)