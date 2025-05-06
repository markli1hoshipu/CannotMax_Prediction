# from PySide6.QtCore import QObject, Signal
# from PySide6.QtGui import QGuiApplication, QImage, QPixmap

# class ScreenCapture(QObject):
#     """单次截图工具（基于 QScreen.grabWindow）"""

#     screenshot_captured = Signal(QImage)  # 截图完成信号
#     error_occurred = Signal(str)         # 错误信号

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self._screen = QGuiApplication.primaryScreen()  # 获取主屏幕

#     def capture(self, target_window=None):
#         """
#         手动触发单次截图
#         Args:
#             target_window: 目标窗口（None 表示整个屏幕）
#         """
#         try:
#             if not self._screen:
#                 raise RuntimeError("无法获取屏幕对象")

#             # 执行截图
#             pixmap = (
#                 self._screen.grabWindow(target_window)
#                 if target_window
#                 else self._screen.grabWindow(0)  # 0 表示整个屏幕
#             )
            
#             if pixmap.isNull():
#                 raise RuntimeError("截图失败（返回空图像）")

#             # 发送截图结果
#             self.screenshot_captured.emit(pixmap.toImage())

#         except Exception as e:
#             self.error_occurred.emit(f"截图失败: {str(e)}")
from PySide6.QtCore import Qt, QPoint, QRect, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor, QGuiApplication
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QMessageBox
import numpy as np
from PIL import ImageGrab
import cv2

class ROISelector(QWidget):
    roi_selected = Signal(tuple)  # 发送((x1,y1),(x2,y2))
    selection_cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择区域")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        
        # 初始化变量
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.is_selecting = False
        self.roi_box = []
        
        # 设置全屏
        screen = QGuiApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        
        # 添加提示标签
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                color: red;
                font-size: 20px;
                background-color: rgba(0,0,0,0.5);
                padding: 10px;
            }
        """)
        self.label.setText("拖动鼠标选择区域\n按Enter确认 | ESC取消")
        self.label.move(50, 50)
        self.label.adjustSize()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_selecting = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_selecting:
            self.is_selecting = False
            self.end_point = event.pos()
            self.roi_box = [
                (self.start_point.x(), self.start_point.y()),
                (self.end_point.x(), self.end_point.y())
            ]
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return and len(self.roi_box) == 2:
            # 标准化坐标
            x1, y1 = min(self.roi_box[0][0], self.roi_box[1][0]), min(self.roi_box[0][1], self.roi_box[1][1])
            x2, y2 = max(self.roi_box[0][0], self.roi_box[1][0]), max(self.roi_box[0][1], self.roi_box[1][1])
            self.roi_selected.emit(((x1, y1), (x2, y2)))
            self.close()
        elif event.key() == Qt.Key_Escape:
            self.selection_cancelled.emit()
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setOpacity(0.7)  # 半透明背景
        
        # 绘制全屏截图背景
        screenshot = QGuiApplication.primaryScreen().grabWindow(0)
        painter.drawPixmap(self.rect(), screenshot)
        
        # 绘制选区矩形
        if self.is_selecting or len(self.roi_box) == 2:
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.setPen(QPen(Qt.red, 2, Qt.DashLine))
            painter.setBrush(QColor(0, 0, 255, 50))
            painter.drawRect(rect)