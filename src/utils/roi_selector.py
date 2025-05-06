import cv2
import numpy as np
import os
from PIL import ImageGrab
from .config import get_config

# 全局变量
roi_box = []  # 存储选区坐标
drawing = False  # 标记是否正在绘制
current_img = None  # 当前显示的图像

def mouse_callback(event, x, y, flags, param):
    """改进的鼠标回调函数，实时显示选区"""
    global roi_box, drawing, current_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_box = [(x, y)]
        drawing = True
        
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # 实时显示选区
        display_img = current_img.copy()
        cv2.rectangle(display_img, roi_box[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", display_img)
        
    elif event == cv2.EVENT_LBUTTONUP:
        roi_box.append((x, y))
        drawing = False
        # 显示最终选区
        display_img = current_img.copy()
        cv2.rectangle(display_img, roi_box[0], roi_box[1], (0, 0, 255), 2)
        cv2.imshow("Select ROI", display_img)

def select_roi():
    """改进的交互式区域选择"""
    global roi_box, drawing, current_img
    
    while True:
        # 重置状态
        roi_box = []
        drawing = False
        
        # 获取初始截图
        screenshot = np.array(ImageGrab.grab())
        current_img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        # 添加操作提示
        cv2.putText(current_img, 
                   "Drag to select area | ENTER:confirm | ESC:retry",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 尝试显示示例图片（如果存在）
        example_path = os.path.join(os.path.dirname(__file__), "...", "resources", "eg.png")
        example_path = os.path.normpath(example_path)
        
        if os.path.exists(example_path):
            try:
                example_img = cv2.imread(example_path)
                if example_img is not None:
                    cv2.imshow("Example (Close this first)", example_img)
            except Exception as e:
                print(f"无法加载示例图片: {e}")

        # 显示主窗口
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", 1280, 720)
        cv2.setMouseCallback("Select ROI", mouse_callback, current_img)
        cv2.imshow("Select ROI", current_img)

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 13 and len(roi_box) == 2:  # Enter确认
            # 标准化坐标 (x1,y1)为左上角，(x2,y2)为右下角
            x1, y1 = min(roi_box[0][0], roi_box[1][0]), min(roi_box[0][1], roi_box[1][1])
            x2, y2 = max(roi_box[0][0], roi_box[1][0]), max(roi_box[0][1], roi_box[1][1])
            return [(x1, y1), (x2, y2)]
        elif key == 27:  # ESC重试
            continue