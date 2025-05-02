from PySide6.QtWidgets import QMainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from src.module.TransformerNet import UnitAwareTransformer
import os
import sys
from src.utils import get_config
from src.tabs import PredictTab, ManualTab, ADBTab, OpenCVTab, TrainTab, LegacyTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.tabs = {}
        self.setup_ui()
        self.setup_tabs()
    
    def setup_ui(self):
        """加载主窗口UI"""
        loader = QUiLoader()
        self.ui = loader.load(self.config['ui_file'])
        self.setCentralWidget(self.ui)
    
    def setup_tabs(self):
        """初始化所有Tab页"""
        
        self.tabs['PredictTab'] = PredictTab()
        self.ui.tabWidget.addTab(self.tabs['PredictTab'], "斗蛐蛐预测")
        
        self.tabs['ManualTab'] = ManualTab()
        self.ui.tabWidget.addTab(self.tabs['ManualTab'], "人工录入")

        self.tabs['ADBTab'] = ADBTab()
        self.ui.tabWidget.addTab(self.tabs['ADBTab'], "ADB录入")

        self.tabs['OpenCVTab'] = OpenCVTab()
        self.ui.tabWidget.addTab(self.tabs['OpenCVTab'], "OpenCV录入")

        self.tabs['TrainTab'] = TrainTab()
        self.ui.tabWidget.addTab(self.tabs['TrainTab'], "模型训练")

        self.tabs['LegacyTab'] = LegacyTab()
        self.ui.tabWidget.addTab(self.tabs['LegacyTab'], "旧模板展示")
        
    def closeEvent(self, event):
        # 清理所有tab页的资源
        for tab in self.tabs.values():
            if hasattr(tab, 'cleanup'):
                tab.cleanup()
        event.accept()

    def closeEvent(self, event):
        # 清理所有tab页的资源
        for tab in self.tabs.values():
            if hasattr(tab, 'cleanup'):
                tab.cleanup()
        event.accept()    

if __name__ == "__main__":
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"  # 可选：启用高DPI缩放
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)  # 关键设置

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.setWindowTitle("CannotMax")
    window.resize(900, 600)
    window.show()
    #app.exec()
    
    sys.exit(app.exec())