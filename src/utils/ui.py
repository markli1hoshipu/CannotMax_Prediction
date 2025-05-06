import os
from PySide6.QtWidgets import QSpinBox

def get_ui_path(tab_name):
    """
    获取Tab页对应的UI文件路径
    :param tab_name: UI文件名（不带后缀），如 'adb_tab'
    :return: 绝对路径字符串
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))  # 项目根目录
    return os.path.abspath(os.path.join(
        base_dir,
        'ui',
        f'{tab_name}.ui'
    ))

def get_spinbox_column_data(table_widget, column_index):
    """读取指定列的 QSpinBox 值，返回整数列表"""
    column_data = []
    for row in range(table_widget.rowCount()):
        # 获取单元格的 QSpinBox 控件
        spinbox = table_widget.cellWidget(row, column_index)
        if isinstance(spinbox, QSpinBox):  # 确保是 SpinBox
            column_data.append(spinbox.value())
        else:
            column_data.append(0)  # 默认值（或根据需求返回 None）
    return column_data