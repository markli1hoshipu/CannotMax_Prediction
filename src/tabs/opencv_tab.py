from .base_tab import BaseTab
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QTableWidget, 
                              QHeaderView, QSpinBox, QHBoxLayout, QLabel, QWidget, 
                              QVBoxLayout,QTableWidgetItem)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Slot, Qt
from src.recognize import detect_enemies
from src.utils import get_config
import os

class OpenCVTab(BaseTab):
    UI_NAME = 'opencv_tab'
    def setup_ui(self):
        # 设置布局
        layout = QVBoxLayout(self)
        layout.addWidget(self.ui)
        self.setLayout(layout)

    
        self.config = get_config()
        self.game_mode = "单人" 
        self.is_invest = True
        self.clear_tables()
        self._init_enemy_selection_table()

    def setup_connections(self):
        self.ui.btnCapture.clicked.connect(self._on_area_capture)

        self.ui.rdbtn1P_2.clicked.connect(lambda: self.update_gamemode("单人"))
        self.ui.rdbtn30P_2.clicked.connect(lambda: self.update_gamemode("30人"))
        self.ui.btnClear_2.clicked.connect(self._clear_all_numbers)
        self.ui.btnCheck_2.clicked.connect(self._transfer_valid_data)

        self.ui.btnLeftWin.clicked.connect(self.left_win_collect)
        self.ui.btnRightWin.clicked.connect(self.right_win_collect)

    @Slot()
    def _on_area_capture(self):
        """区域截图"""
        try:
            self._set_capture_ui_state(True, "请用鼠标选择区域 (ESC取消)")
            
            # 使用OpenCV版本的选择器
            from utils import select_roi
            roi = select_roi()
            
            if roi:
                # 获取选区的截图
                (x1, y1), (x2, y2) = roi
                screenshot = np.array(ImageGrab.grab())
                cropped = screenshot[y1:y2, x1:x2]
                
                # 转换为QPixmap显示
                height, width, _ = cropped.shape
                bytes_per_line = 3 * width
                q_img = QImage(cropped.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                self.ui.labelScreenshot.setPixmap(
                    pixmap.scaled(
                        self.ui.labelScreenshot.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )
                
                # 识别敌人
                enemy_list = detect_enemies(pixmap)
                self._update_table(enemy_list)
                
        except Exception as e:
            self._show_error(f"截图失败: {str(e)}")
        finally:
            self._set_capture_ui_state(False)

    def _update_table(self, enemy_list):
        pass
    
    def clear_tables(self):
        """清空所有表格数据"""
        tables = [
            self.ui.tableSelectEnemy_2,
            self.ui.tableDispLeft_2,
            self.ui.tableDispRight_2
        ]
        for table in tables:
            table.setRowCount(0)
            table.clearContents()
        
    def _init_enemy_selection_table(self):
        for table in [self.ui.tableDispLeft_2, self.ui.tableDispRight_2]:
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["敌人", "数量"])
            table.setColumnWidth(0, 80)
            table.setColumnWidth(1, 60)
            
        """初始化敌人选择表格（自适应行高+大图显示）"""
        table = self.ui.tableSelectEnemy_2
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["敌人", "左数量", "右数量"])
        
        # 设置列宽策略
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # 图片列自适应
        header.setSectionResizeMode(1, QHeaderView.Fixed)    # 固定宽度
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.resizeSection(1, 60)  # 加宽数值输入列
        header.resizeSection(2, 60)

        # 禁用默认行高设置，启用内容自适应
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        # 图片目录路径
        image_dir = self.config['ui_photos_direc']
        
        try:
            # 获取并排序PNG文件
            png_files = sorted(
                [f for f in os.listdir(image_dir) if f.lower().endswith('.png')],
                key=lambda x: int(os.path.splitext(x)[0])
            )
            
            for row, filename in enumerate(png_files):
                table.insertRow(row)
                
                # 第一列：图片显示（带悬停提示）
                cell_widget = QWidget()
                layout = QHBoxLayout(cell_widget)
                layout.setContentsMargins(2,2,2,2)  # 增加垂直边距
                layout.setAlignment(Qt.AlignCenter)
                
                # 图片标签（放大显示）
                icon_label = QLabel()
                pixmap = QPixmap(os.path.join(image_dir, filename))
                if not pixmap.isNull():
                    # 设置大尺寸显示（保留原始比例）
                    scaled_pixmap = pixmap.scaledToHeight(
                        52,  # 固定高度，宽度按比例自动计算
                        Qt.SmoothTransformation
                    )
                    icon_label.setPixmap(scaled_pixmap)
                    icon_label.setToolTip(f"ID: {os.path.splitext(filename)[0]}")
                
                layout.addWidget(icon_label)
                table.setCellWidget(row, 0, cell_widget)
                
                # 数值输入列
                spin_left = QSpinBox()
                spin_left.setRange(0, 999)
                spin_left.setAlignment(Qt.AlignCenter)
                table.setCellWidget(row, 1, spin_left)
                
                spin_right = QSpinBox()
                spin_right.setRange(0, 999)
                spin_right.setAlignment(Qt.AlignCenter)
                table.setCellWidget(row, 2, spin_right)
                
                # 设置该行最小高度（根据图片实际高度+边距）
                table.setRowHeight(row, scaled_pixmap.height())
                
        except Exception as e:
            print(f"加载敌人图片失败: {e}")

    def _clear_all_numbers(self):
        """清空所有数量输入框（btnClear功能）"""
        table = self.ui.tableSelectEnemy_2
        for row in range(table.rowCount()):
            # 获取左数量输入框
            spin_left = table.cellWidget(row, 1)
            if isinstance(spin_left, QSpinBox):
                spin_left.setValue(0)
            
            # 获取右数量输入框
            spin_right = table.cellWidget(row, 2)
            if isinstance(spin_right, QSpinBox):
                spin_right.setValue(0)
    
    def _transfer_valid_data(self):
        """修复版：稳定处理多行数据转移"""
        src_table = self.ui.tableSelectEnemy_2
        left_table = self.ui.tableDispLeft_2
        right_table = self.ui.tableDispRight_2
        
        # 清空目标表格时保留列设置
        left_table.clearContents()
        right_table.clearContents()
        left_table.setRowCount(0)
        right_table.setRowCount(0)
        
        # 强制立即应用表格样式
        self._force_table_style(left_table)
        self._force_table_style(right_table)
        
        # 遍历源表格
        for src_row in range(src_table.rowCount()):
            img_widget = src_table.cellWidget(src_row, 0)
            spin_left = src_table.cellWidget(src_row, 1)
            spin_right = src_table.cellWidget(src_row, 2)
            
            if not all([img_widget, spin_left, spin_right]):
                continue
                
            left_val = spin_left.value()
            right_val = spin_right.value()
            pixmap = img_widget.findChild(QLabel).pixmap()
            
            # 左数量处理
            if left_val > 0:
                left_row = left_table.rowCount()
                left_table.insertRow(left_row)
                
                # 图片单元格（固定高度）
                left_img_cell = self._create_image_cell(pixmap, target_height=52)
                left_table.setCellWidget(left_row, 0, left_img_cell)
                
                # SpinBox单元格
                left_spin_cell = self._create_spinbox(left_val)
                left_table.setCellWidget(left_row, 1, left_spin_cell)
                
                # 显式设置行高
                left_table.setRowHeight(left_row, left_img_cell.sizeHint().height())

            # 右数量处理（相同逻辑）
            if right_val > 0:
                right_row = right_table.rowCount()
                right_table.insertRow(right_row)
                
                right_img_cell = self._create_image_cell(pixmap, target_height=52)
                right_table.setCellWidget(right_row, 0, right_img_cell)
                
                right_spin_cell = self._create_spinbox(right_val)
                right_table.setCellWidget(right_row, 1, right_spin_cell)
                
                right_table.setRowHeight(right_row, right_img_cell.sizeHint().height())

    def _force_table_style(self, table):
        """强制刷新表格样式"""
        table.style().unpolish(table)
        table.style().polish(table)
        table.updateGeometry()
        table.viewport().update()

    def _create_image_cell(self, pixmap, target_height):
        """创建标准化图片单元格（带高度控制）"""
        cell = QWidget()
        layout = QHBoxLayout(cell)
        layout.setContentsMargins(2,2,2,2)
        
        label = QLabel()
        scaled_pix = pixmap.scaledToHeight(
            target_height - 2,  # 减去边距
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pix)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        # 固定单元格最小尺寸
        cell.setMinimumSize(scaled_pix.width() , target_height)
        return cell

    def _create_spinbox(self, value):
        """创建标准化SpinBox"""
        spin = QSpinBox()
        spin.setRange(0, 999)
        spin.setValue(value)
        spin.setAlignment(Qt.AlignCenter)
        spin.setReadOnly(True)

        return spin

    def left_win_collect(self):
        pass
    def right_win_collect(self):
        pass