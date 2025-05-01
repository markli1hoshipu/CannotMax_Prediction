import json

# 从JSON文件加载
with open('config.json', 'r') as f:
    loaded_config = json.load(f)

# 现在loaded_config就是原来的字典
print(loaded_config)