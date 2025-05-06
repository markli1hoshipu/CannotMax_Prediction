from PySide6.QtWidgets import QMainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from src.module.TransformerNet import UnitAwareTransformer
import os
import sys
from src.utils import get_config
from src.tabs import ADBTab, OpenCVTab, PhotoPredictTab

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
        
 

        self.tabs['PhotoPredictTab'] = PhotoPredictTab()
        self.ui.tabWidget.addTab(self.tabs['PhotoPredictTab'], "斗蛐蛐")

        # self.tabs['ADBTab'] = ADBTab()
        # self.ui.tabWidget.addTab(self.tabs['ADBTab'], "ADB录入")

        # self.tabs['OpenCVTab'] = OpenCVTab()
        # self.ui.tabWidget.addTab(self.tabs['OpenCVTab'], "OpenCV录入")

        
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
    window.resize(1200, 800)
    window.show()
    #app.exec()
    
    sys.exit(app.exec())
