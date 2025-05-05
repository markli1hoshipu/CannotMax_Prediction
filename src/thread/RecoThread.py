from PySide6.QtCore import QThread, Signal ,QProcess,QMutex, QMutexLocker
from ..recognize import ImageRecognizer

class RecoThread(QThread):
    """OCR识别线程"""
    
    # 定义信号
    recognition_complete = Signal(list)  # 识别完成信号，传递结果列表
    update_entry = Signal(dict)         # 更新条目信号，传递条目数据
    error_occurred = Signal(str)        # 错误发生信号，传递错误信息

    def __init__(self, parent=None):
        super().__init__(parent)
        self._first_recognize = True
        self._main_roi = None
        self._running = False
        self.recogizer = ImageRecognizer()
        
    def is_running(self):
        return self._running

    def stop(self):
        self._running = False
        self.quit()
        self.wait()  # 等待线程结束

    def run(self,adbshot=None,cvshot=None,detectway="None",selectROI=False):
        """线程主逻辑"""
        self._running = True
        screenshot = None
        try:
            if detectway == "adb":
                screenshot = adbshot
            elif detectway == "cv":
                screenshot = cvshot
            # 执行OCR识别
            
            if selectROI:
                self._main_roi = self.recogizer.select_roi(screenshot)
            
            if self._main_roi and screenshot:
                results = self.recogizer.process_regions(self._main_roi, screenshot=screenshot)
                self.recognition_complete.emit(results)
                self._process_results(results)
                
        except Exception as e:
            self.error_occurred.emit(f"识别过程中发生错误: {str(e)}")
        finally:
            self._running = False

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
