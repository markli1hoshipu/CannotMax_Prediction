# Arknights Neural Network

这是一个基于深度学习的明日方舟游戏辅助工具，用于自动识别游戏画面中的单位并预测战斗结果。

## 功能特点

- 自动识别游戏画面中的单位类型和数量
- 使用深度学习模型预测战斗结果
- 支持自动数据收集和训练 (暂未实现)
- 提供图形用户界面进行操作
- 支持自动投资和战斗预测

## 环境要求

- Python 3.7+
- Windows 10/11
- NVIDIA GPU (推荐，用于加速模型训练)

## 安装步骤 （需要修改）

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd [项目目录]
```

2. 创建并激活虚拟环境（推荐）：
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. 安装依赖包：
```bash
pip install -r requirements.txt
```

4. 安装Tesseract OCR：
- 下载并安装 [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- 将Tesseract安装目录添加到系统环境变量PATH中

5. 修改你的设备序列号:
- 在`loadData.py`中找到以下代码：
```python
   device_serial = '127.0.0.1:5555'  # 指定设备序列号
```
- 若你使用MUMU或雷电模拟器，你可以从以下网址得知如何找到设备序列号
- 雷电 https://help.ldmnq.com/docs/LD9adbserver
- MUMU https://mumu.163.com/help/20230214/35047_1073151.html


## 使用方法

1. 运行主程序：
```bash
python main.py
```

2. 在图形界面中：
   - 点击"自动获取数据"通过adb链接来自动收集模拟器训练数据
   - 可通过点击单人按钮来选择单人或30人模式（注意：30人模式要点击投资勾选框）
   - 投资勾选框来选择是投资或观望
   - 使用"填写√"和"填写×"按钮记录预测结果
   - 使用"预测"按钮获取战斗结果预测
   - 点击选择范围按钮来选择识别范围（主要用于截取直播画面进行预测）
     - 框选后按回车键确认，框选错误按esc键重新框选
   - 使用"识别"按钮手动识别当前画面
   - 输入模拟器序列号后点击更新按钮对模拟器进行重新连接


3. 训练模型：
   - 数据收集完成后，先进行数据清洗，然后训练：
   ```bash
   python train.py
   ```

## 注意事项

- 选择范围时要尽可能按例子框选，且尽可能将直播画面放大再进行框选
- 建议在训练模型前收集足够的数据
- 自动获取数据功能游戏窗口可在后台运行
- 按ESC键可以随时停止自动获取数据

## 文件说明

- `main.py`: 主程序，包含GUI界面和主要功能
- `train.py`: 模型训练脚本
- `loadData.py`: 数据加载和处理模块
- `recognize.py`: 图像识别模块
- `requirements.txt`: 项目依赖包列表
- `data_cleaning.py`: 数据清洗

## 贡献

欢迎提交问题和改进建议！ 