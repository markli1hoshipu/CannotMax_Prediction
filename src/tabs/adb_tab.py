from .base_tab import BaseTab
from PySide6.QtWidgets import QVBoxLayout
from ..utils import get_config

class ADBTab(BaseTab):

    UI_NAME = 'adb_tab'

    def setup_ui(self):
        # 设置布局
        self.device_serial = "" 
        self.statistics = {}
        self.config = get_config()

        layout = QVBoxLayout(self)
        layout.addWidget(self.ui)
        self.setLayout(layout)

    
    def setup_connections(self):
        self.ui.btnUpdate_2.clicked.connect(self._update_deviceserial)
        self.ui.btnAutoGetData_2.clicked.connect(self._auto_collect)

    def _update_deviceserial(self):
        self.device_serial = self.ui.lineSimuNum_2.text().strip()

    def _auto_collect(self):
        pass

if __name__ == "__main__":
    # 独立运行测试
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    tab = ADBTab()
    tab.setWindowTitle("ADB Tab - Standalone")
    tab.show()
    app.exec()