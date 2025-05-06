import platform
from PySide6.QtCore import Qt, QRect, QPoint, Signal, QSize
from PySide6.QtGui import QPainter, QPen, QColor, QScreen
from PySide6.QtWidgets import QApplication, QWidget, QRubberBand

class ScreenSelectionWidget(QWidget):
    """
    一个覆盖屏幕的透明窗口，允许用户通过拖拽选择一个矩形区域。
    选择完成后发出 'area_selected(QRect)' 信号，包含所选区域的屏幕坐标。
    PySide6 版本。
    """
    area_selected = Signal(QRect)  # PySide6 使用 Signal 而不是 pyqtSignal

    def __init__(self, screen: QScreen):
        super().__init__()
        self.screen_geometry = screen.geometry()
        self.setGeometry(self.screen_geometry)
        self.setWindowTitle('选择识别区域')
        # 设置窗口标志：无边框、保持最前、半透明
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        raw_pixel_ratio = screen.devicePixelRatio()
        if platform.system() == 'Darwin': 
            self.device_pixel_ratio = raw_pixel_ratio / 2.0
        else:
            self.device_pixel_ratio = raw_pixel_ratio
        self.rubber_band = None
        self.origin = QPoint()
        self.current_pos = QPoint()  # 存储鼠标移动或释放时的全局坐标
        self.preview_rect = QRect()  # 存储鼠标释放后的预览矩形

        # 添加一个半透明的黑色背景以便看清选区
        self.overlay_color = QColor(0, 0, 0, 100)  # 半透明黑色

    def paintEvent(self, event):
        """绘制半透明背景和预览选框"""
        painter = QPainter(self)
        # 绘制半透明背景
        painter.fillRect(self.rect(), self.overlay_color)

        # 如果有预览选区 (全局坐标)，将其转换为局部坐标进行绘制
        if self.preview_rect.isValid():
            draw_rect = self.preview_rect.translated(-self.geometry().topLeft())
            pen = QPen(QColor('red'), 2, Qt.PenStyle.SolidLine)  # 2px 红色实线
            painter.setPen(pen)
            painter.drawRect(draw_rect)  # 绘制局部坐标矩形

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # 清除之前的预览框
            self.preview_rect = QRect()
            self.update()  # 触发重绘以清除红框

            self.origin = event.globalPosition().toPoint()  # 记录全局逻辑坐标起点
            if not self.rubber_band:
                self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)

            # 将全局起点映射到窗口局部坐标，用于设置橡皮筋初始位置
            local_origin = self.mapFromGlobal(self.origin)
            self.rubber_band.setGeometry(QRect(local_origin, QSize()))  # 使用局部坐标
            self.rubber_band.show()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.rubber_band and not self.origin.isNull() and event.buttons() & Qt.MouseButton.LeftButton:
            self.current_pos = event.globalPosition().toPoint()  # 记录当前全局逻辑坐标

            # 将全局起点和当前点映射到窗口局部坐标，用于更新橡皮筋
            local_origin = self.mapFromGlobal(self.origin)
            local_current = self.mapFromGlobal(self.current_pos)
            self.rubber_band.setGeometry(QRect(local_origin, local_current).normalized())  # 使用局部坐标
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rubber_band:
            self.current_pos = event.globalPosition().toPoint()  # 记录释放时的全局逻辑坐标
            self.rubber_band.hide()  # 隐藏橡皮筋

            # 直接使用全局起点和全局释放点计算最终的全局逻辑坐标矩形
            self.preview_rect = QRect(self.origin, self.current_pos).normalized()

            # 检查计算出的全局选区是否有效
            if not (self.preview_rect.isValid() and self.preview_rect.width() > 0 and self.preview_rect.height() > 0):
                self.preview_rect = QRect()  # 重置预览

            self.update()  # 触发重绘以显示（或清除）红框
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        # 按 Enter 键确认当前预览选区
        if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            if self.preview_rect.isValid():
                # 将逻辑坐标转换为物理像素坐标
                physical_rect = QRect(
                    int(self.preview_rect.left() * self.device_pixel_ratio),
                    int(self.preview_rect.top() * self.device_pixel_ratio),
                    int(self.preview_rect.width() * self.device_pixel_ratio),
                    int(self.preview_rect.height() * self.device_pixel_ratio)
                )

                # 发射物理像素坐标
                self.area_selected.emit(physical_rect)
                self.close()
                event.accept()
            else:
                event.ignore()  # 没有有效选区，忽略 Enter

        # 按 Esc 键取消选择并关闭窗口
        elif key == Qt.Key.Key_Escape:
            self.area_selected.emit(QRect())  # 发出空矩形信号
            self.close()
            event.accept()
        else:
            super().keyPressEvent(event)