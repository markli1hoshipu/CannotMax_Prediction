from .base_tab import BaseTab
from PySide6.QtWidgets import QVBoxLayout

class OpenCVTab(BaseTab):
    UI_NAME = 'opencv_tab'
    def setup_ui(self):
        # 设置布局
        layout = QVBoxLayout(self)
        layout.addWidget(self.ui)
        self.setLayout(layout)
        
    
    def setup_connections(self):
        if hasattr(self.ui, 'btnClear'):
            self.ui.btnClear.clicked.connect(self._clear_all_numbers)
    
    def _clear_all_numbers(self):
        print("Clear button clicked")
