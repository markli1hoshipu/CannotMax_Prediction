from PySide6.QtCore import QThread, Signal ,QProcess,QMutex, QMutexLocker
from ..recogize import loadData, recognize

class RecoThread(QThread):
    """OCR识别线程"""
    
    # 定义信号
    recognition_complete = Signal(list)  # 识别完成信号，传递结果列表
    update_entry = Signal(dict)         # 更新条目信号，传递条目数据
    adb_connected = Signal()            # ADB连接成功信号
    error_occurred = Signal(str)        # 错误发生信号，传递错误信息

    def __init__(self, parent=None):
        super().__init__(parent)
        self._auto_fetch = False
        self._no_region = True
        self._first_recognize = True
        self._main_roi = None
        self._running = False
        
        self._mutex = QMutex()  # 添加互斥锁
        # 其他初始化...
        self.adb_process = QProcess(self)  # 作为子对象创建，父对象析构时自动销毁

    def set_parameters(self, auto_fetch=False, no_region=True, main_roi=None):
        """设置识别参数"""
        self._auto_fetch = auto_fetch
        self._no_region = no_region
        self._main_roi = main_roi

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
            
            # 执行OCR识别
            if self._main_roi and screenshot:
                results = recognize.process_regions(self._main_roi, screenshot=screenshot)
                self.recognition_complete.emit(results)
                self._process_results(results)
                
        except Exception as e:
            self.error_occurred.emit(f"识别过程中发生错误: {str(e)}")
        finally:
            self._running = False

    def _get_screenshot(self):
        """获取截图"""
        # 自动获取模式下从adb加载截图
        if self._auto_fetch:
            return loadData.capture_screenshot()

        # 未选择区域时从adb获取截图
        if self._no_region:
            if self._first_recognize:
                self._connect_adb()
                self._first_recognize = False
            return loadData.capture_screenshot()

        return None

    def _connect_adb(self):
        self.adb_process = QProcess()
        self.adb_process.finished.connect(self._on_adb_finished)
        self.adb_process.errorOccurred.connect(self._on_adb_error)
        
        command = f"{loadData.adb_path} connect {loadData.device_serial}"
        self.adb_process.start(command.split())

    def _on_adb_finished(self, exit_code):
        if exit_code == 0:
            self.adb_connected.emit()
        else:
            self.error_occurred.emit(f"ADB连接失败，错误码: {exit_code}")

    def _on_adb_error(self, error):
        self.error_occurred.emit(f"ADB进程错误: {error}")

    def _process_results(self, results):
        """处理识别结果"""
        for res in results:
            if 'error' not in res and res.get('matched_id', 0) != 0:
                self.update_entry.emit({
                    'region_id': res['region_id'],
                    'matched_id': res['matched_id'],
                    'number': res['number'],
                    'has_data': bool(res['number'])
                })
