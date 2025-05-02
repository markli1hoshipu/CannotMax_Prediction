from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QGuiApplication, QImage, QPixmap

class ScreenCapture(QObject):
    """单次截图工具（基于 QScreen.grabWindow）"""

    screenshot_captured = Signal(QImage)  # 截图完成信号
    error_occurred = Signal(str)         # 错误信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self._screen = QGuiApplication.primaryScreen()  # 获取主屏幕

    def capture(self, target_window=None):
        """
        手动触发单次截图
        Args:
            target_window: 目标窗口（None 表示整个屏幕）
        """
        try:
            if not self._screen:
                raise RuntimeError("无法获取屏幕对象")

            # 执行截图
            pixmap = (
                self._screen.grabWindow(target_window)
                if target_window
                else self._screen.grabWindow(0)  # 0 表示整个屏幕
            )
            
            if pixmap.isNull():
                raise RuntimeError("截图失败（返回空图像）")

            # 发送截图结果
            self.screenshot_captured.emit(pixmap.toImage())

        except Exception as e:
            self.error_occurred.emit(f"截图失败: {str(e)}")