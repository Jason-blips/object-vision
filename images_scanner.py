"""
实时摄像头图像分类：加载 model.keras 与 model_classes.json，按训练时的类别顺序推理。
若尚未训练，请先运行 train_model.py 并确保 data/ 下有至少两个类别的图片。
"""
import json
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import MODEL_PATH, CLASSES_JSON, IMAGE_SIZE, CONF_THRESHOLD

# 注册自定义层/函数，避免加载 SavedModel 时报错
tf.keras.utils.get_custom_objects()["preprocess_input"] = preprocess_input

# ----- 加载类别列表（与训练时顺序一致）-----
if not os.path.isfile(CLASSES_JSON):
    print("❌ 未找到类别文件:", CLASSES_JSON)
    print("   请先运行 train_model.py 完成训练，会自动生成该文件。")
    exit(1)

with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# ----- 加载模型 -----
if not os.path.isfile(MODEL_PATH):
    print("❌ 未找到模型文件:", MODEL_PATH)
    print("   请先运行 train_model.py 完成训练。")
    exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ 模型与类别已加载:", MODEL_PATH, "->", class_names)

# ----- 摄像头 -----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit(1)

print("🎥 摄像头已开启，按 Q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, IMAGE_SIZE)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # 预测
    pred = model.predict(img, verbose=0)[0]

    # 显示各类别概率
    h, w, _ = frame.shape
    y0 = 30
    for i, name in enumerate(class_names):
        prob = float(pred[i])
        text = f"{name}: {prob:.2f}"
        color = (0, 255, 0) if prob > CONF_THRESHOLD else (0, 0, 255)
        cv2.putText(frame, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y0 += 30

    # 中心显示最高置信度预测
    max_idx = int(np.argmax(pred))
    max_prob = float(pred[max_idx])
    if max_prob > CONF_THRESHOLD:
        text = f"Pred: {class_names[max_idx]} ({max_prob:.2f})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cx = w // 2 - text_size[0] // 2
        cv2.putText(frame, text, (cx, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-time Object Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
