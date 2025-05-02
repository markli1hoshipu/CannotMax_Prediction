import subprocess
import time
import cv2
import numpy as np

class AutoGameController:
    def __init__(self, adb_path=r".\src\outerTools\platform-tools\adb.exe", serial='127.0.0.1:5555'):
        self.adb_path = adb_path
        self.manual_serial = serial
        self.device_serial = None
        self.process_images = [cv2.imread(f'resources/process/{i}.png') for i in range(16)]  # 16个模板
        self.screen_width = 1920
        self.screen_height = 1080
        
        self.relative_points = [
            (0.9297, 0.8833),  # 右ALL、返回主页、加入赛事、开始游戏
            (0.0713, 0.8833),  # 左ALL
            (0.8281, 0.8833),  # 右礼物、自娱自乐
            (0.1640, 0.8833),  # 左礼物
            (0.4979, 0.6324),  # 本轮观望
        ]
        
        # 初始化设备连接
        self._initialize_device()

    def set_device_serial(self, serial):
        """设置设备序列号"""
        self.manual_serial = serial
        self._initialize_device()

    def _initialize_device(self):
        """初始化设备连接"""
        try:
            self.device_serial = self._get_device_serial()
            print(f"最终使用设备: {self.device_serial}")
            self._connect_to_emulator()
        except RuntimeError as e:
            print(f"错误: {str(e)}")
            raise

    def _get_device_serial(self):
        """获取设备序列号"""
        try:
            # 使用当前的manual_serial值
            subprocess.run(f'{self.adb_path} connect {self.manual_serial}', shell=True, check=True)

            # 检查手动设备是否在线
            result = subprocess.run(
                f'{self.adb_path} devices',
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )

            devices = []
            for line in result.stdout.split('\n'):
                if '\tdevice' in line:
                    dev = line.split('\t')[0]
                    devices.append(dev)
                    if dev == self.manual_serial:
                        return dev

            # 自动选择第一个可用设备
            if devices:
                print(f"自动选择设备: {devices[0]}")
                return devices[0]

            print("未找到连接的Android设备")
            return None

        except Exception as e:
            print(f"设备检测失败: {str(e)}")
            return None

    def _connect_to_emulator(self):
        """连接到模拟器"""
        try:
            subprocess.run(f'{self.adb_path} connect {self.device_serial}', shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ADB connect command failed: {e}")
        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure adb is installed and added to the system PATH.")

    def capture_screenshot(self):
        """捕获屏幕截图"""
        try:
            # 获取二进制图像数据
            screenshot_data = subprocess.check_output(
                f'{self.adb_path} -s {self.device_serial} exec-out screencap -p',
                shell=True
            )

            # 将二进制数据转换为numpy数组
            img_array = np.frombuffer(screenshot_data, dtype=np.uint8)

            # 使用OpenCV解码图像
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            return img
        except subprocess.CalledProcessError as e:
            print(f"Screenshot capture failed: {e}")
            return None
        except Exception as e:
            print(f"Image processing error: {e}")
            return None

    def _match_images(self, screenshot, templates):
        """匹配图像模板"""
        screenshot_quarter = screenshot[int(screenshot.shape[0] * 3 / 4):, :]
        results = []
        for idx, template in enumerate(templates):
            template_quarter = template[int(template.shape[0] * 3 / 4):, :]
            res = cv2.matchTemplate(screenshot_quarter, template_quarter, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            results.append((idx, max_val))
        return results

    def _click(self, point):
        """点击屏幕指定位置"""
        x, y = point
        x_coord = int(x * self.screen_width)
        y_coord = int(y * self.screen_height)
        print(f"点击坐标: ({x_coord}, {y_coord})")
        subprocess.run(f'{self.adb_path} -s {self.device_serial} shell input tap {x_coord} {y_coord}', shell=True)

    def _operation_simple(self, results):
        """简单操作逻辑"""
        for idx, score in results:
            if score > 0.6:  # 假设匹配阈值为 0.8
                if idx == 0:  # 加入赛事
                    self._click(self.relative_points[0])
                    print("加入赛事")
                elif idx == 1:  # 自娱自乐
                    self._click(self.relative_points[2])
                    print("自娱自乐")
                elif idx == 2:  # 开始游戏
                    self._click(self.relative_points[0])
                    print("开始游戏")
                elif idx in [3, 4, 5]:  # 本轮观望
                    self._click(self.relative_points[4])
                    print("本轮观望")
                elif idx in [10, 11]:
                    print("下一轮")
                elif idx in [6, 7]:
                    print("等待战斗结束")
                elif idx == 12:  # 返回主页
                    self._click(self.relative_points[0])
                    print("返回主页")
                break  # 匹配到第一个结果后退出

    def _operation(self, results):
        """复杂操作逻辑"""
        for idx, score in results:
            if score > 0.6:  # 假设匹配阈值为 0.8
                if idx in [3, 4, 5]:
                    # 识别怪物类型数量，导入模型进行预测
                    prediction = 0.6
                    # 根据预测结果点击投资左/右
                    if prediction > 0.5:
                        self._click(self.relative_points[1])  # 投资右
                        print("投资右")
                    else:
                        self._click(self.relative_points[0])  # 投资左
                        print("投资左")
                elif idx in [1, 5]:
                    self._click(self.relative_points[2])  # 点击省点饭钱
                    print("点击省点饭钱")
                elif idx == 2:
                    self._click(self.relative_points[3])  # 点击敬请见证
                    print("点击敬请见证")
                elif idx in [3, 4]:
                    # 保存数据
                    self._click(self.relative_points[4])  # 点击下一轮
                    print("点击下一轮")
                elif idx == 6:
                    print("等待战斗结束")
                break  # 匹配到第一个结果后退出

    def start_auto_play(self, simple_mode=True):
        """启动自动游戏"""
        while True:
            screenshot = self.capture_screenshot()
            if screenshot is not None:
                results = self._match_images(screenshot, self.process_images)
                results = sorted(results, key=lambda x: x[1], reverse=True)
                print("匹配结果：", results[0])
                if simple_mode:
                    self._operation_simple(results)
                else:
                    self._operation(results)
            time.sleep(2)

    def single_operation(self):
        """执行单次操作"""
        screenshot = self.capture_screenshot()
        if screenshot is not None:
            results = self._match_images(screenshot, self.process_images)
            results = sorted(results, key=lambda x: x[1], reverse=True)
            print("匹配结果：", results[0])
            self._operation_simple(results)
            return results
        return None