from PySide6.QtCore import QThread, Signal, QProcess, QMutex, QMutexLocker
from ..adb_connect import loadData

class ADBThread(QThread):
    """ADB截图获取线程"""
    
    # 定义信号
    adb_connected = Signal()            # ADB连接成功信号
    error_occurred = Signal(str)        # 错误发生信号，传递错误信息
    screenshot_obtained = Signal(object)  # 截图获取成功信号，传递截图数据

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._mutex = QMutex()  # 添加互斥锁
        self.adb_process = QProcess(self)  # 作为子对象创建，父对象析构时自动销毁

    def is_running(self):
        with QMutexLocker(self._mutex):  # 线程安全访问
            return self._running

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False
        self.quit()
        self.wait()  # 等待线程结束

    def run(self):
        """线程主逻辑"""
        self._running = True
        
        try:
            # 获取截图
            screenshot = self._get_screenshot()
            if screenshot:
                self.screenshot_obtained.emit(screenshot)
                
        except Exception as e:
            self.error_occurred.emit(f"获取截图过程中发生错误: {str(e)}")
        finally:
            self._running = False

    def _get_screenshot(self):
        """通过ADB获取截图"""
        self._connect_adb()
        return loadData.capture_screenshot()

    def _connect_adb(self):
        """连接ADB设备"""
        self.adb_process = QProcess()
        self.adb_process.finished.connect(self._on_adb_finished)
        self.adb_process.errorOccurred.connect(self._on_adb_error)
        
        command = f"{loadData.adb_path} connect {loadData.device_serial}"
        self.adb_process.start(command.split())

    def _on_adb_finished(self, exit_code):
        """ADB连接完成处理"""
        if exit_code == 0:
            self.adb_connected.emit()
        else:
            self.error_occurred.emit(f"ADB连接失败，错误码: {exit_code}")

    def _on_adb_error(self, error):
        """ADB错误处理"""
        self.error_occurred.emit(f"ADB进程错误: {error}")