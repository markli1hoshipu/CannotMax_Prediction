# pyside6-designer
import sys
import os

# 将 src 目录添加到 sys.path
sys.path.append(os.path.abspath('./src'))

# 现在可以导入 utils
from src.utils import *

from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QTableWidget, 
                              QHeaderView, QSpinBox, QHBoxLayout, QLabel, QWidget, 
                              QVBoxLayout,QTableWidgetItem)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

class BattlePredictor(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 加载UI文件
        self.config = get_config()
        self.ui = QUiLoader().load(self.config['ui_file'])
        self.setCentralWidget(self.ui)
        
        # 初始化界面
        self.initialize()
        
    def initialize(self):
        """初始化界面所有组件"""
        # 0. 为所有QPushButton连接空函数
        self._connect_buttons()
        
        # 1. 初始化单选按钮（默认选择rdbtn1P）
        self.ui.rdbtn1P.setChecked(True)
        
        # 2. 初始化勾选投资模式
        self.ui.ckbxInvest.setChecked(True)
        
        # 3. 设置训练时长初始值
        if hasattr(self.ui, 'lineTrainTime'):  # 检查对象是否存在
            self.ui.lineTrainTime.setText("-1")
        
        # 4. 清空三个表格
        self.clear_tables()
        
        # 5. 初始化敌人选择表格
        self._init_enemy_selection_table()
    
    def clear_tables(self):
        """清空所有表格数据"""
        tables = [
            self.ui.tableSelectEnemy,
            self.ui.tableDispLeft,
            self.ui.tableDispRight
        ]
        for table in tables:
            table.setRowCount(0)
            table.clearContents()
        
    def _init_enemy_selection_table(self):
        for table in [self.ui.tableDispLeft, self.ui.tableDispRight]:
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["敌人", "数量"])
            table.setColumnWidth(0, 80)
            table.setColumnWidth(1, 60)
            
        """初始化敌人选择表格（自适应行高+大图显示）"""
        table = self.ui.tableSelectEnemy
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

    def _connect_buttons(self):
        """连接按钮信号"""
        self.ui.btnClear.clicked.connect(self._clear_all_numbers)
        self.ui.btnCheck.clicked.connect(self._transfer_valid_data)

    def _clear_all_numbers(self):
        """清空所有数量输入框（btnClear功能）"""
        table = self.ui.tableSelectEnemy
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
        src_table = self.ui.tableSelectEnemy
        left_table = self.ui.tableDispLeft
        right_table = self.ui.tableDispRight
        
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

        return spin


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序样式（可选）
    app.setStyle('Fusion')
    
    window = BattlePredictor()
    window.setWindowTitle("战斗预测系统")
    window.resize(900, 600)
    window.show()
    
    sys.exit(app.exec())