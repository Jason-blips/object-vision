# 项目统一配置：数据目录、模型路径、输入尺寸等
# 训练与推理共用，便于扩展任意类别

import os

# 数据与模型路径
DATA_DIR = "data"
MODEL_PATH = "model.keras"
MODEL_FINAL_PATH = "model_final.keras"   # 最后一轮权重（可选保留）
CLASSES_JSON = "model_classes.json"     # 训练时写入，推理时读取，保证类别顺序一致

# 图像与模型结构
IMAGE_SIZE = (224, 224)   # MobileNetV2 推荐
CONF_THRESHOLD = 0.5      # 实时识别时，低于此置信度用红色显示

# 训练超参数（可按需调整以逐步提高准确率）
BATCH_SIZE = 8
SEED = 123
VALIDATION_SPLIT = 0.2
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
FINE_TUNE_AT_LAYERS = 100   # 解冻 base 最后 N 层
