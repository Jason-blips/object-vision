"""
从零训练一个简单 CNN（无预训练权重）：
- 数据：data/ 下按文件夹分类，少量数据增强
- 模型：五层卷积 + 池化 + 全连接，权重随机初始化，自己学特征
- 训练：单阶段、固定学习率
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

from config import (
    IMAGE_SIZE,
    MODEL_PATH,
    MODEL_META_JSON,
    CLASSES_JSON,
    BATCH_SIZE,
    SEED,
    VALIDATION_SPLIT,
    INITIAL_EPOCHS,
)
from train_utils import load_data, save_class_names, get_class_weights

# ----- 加载数据集 -----
train_ds_raw, val_ds_raw, class_names, num_classes, train_labels = load_data()

print("Class:", class_names, "->", num_classes, "classes")
save_class_names(class_names)

# 从零训练的模型用简单 [0, 1] 归一化，推理时也用同一方式
with open(MODEL_META_JSON, "w", encoding = "UTF-8") as f:
    json.dump({"preprocess": "custom"}, f, indent = 2)

class_weights = get_class_weights(train_labels)
print("Class weights:", class_weights)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    train_ds_raw.cache()
    .shuffle(1000)
    .prefetch(buffer_size = AUTOTUNE)
)
val_ds = val_ds_raw.cache().prefetch(buffer_size = AUTOTUNE)

# ----- 数据增强 -----
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),

])

# ----- 从零训练的小型 CNN（无预训练） ----
# 输入 0~255，模型内先 Rescaling 到 [0, 1]，再卷积三次
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1.0 / 255, input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    layers.Conv2D(32, (3, 3), padding = "same", activation = "relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), padding = "same", activation = "relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(96, (3, 3), padding = "same", activation = "relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])
model.summary()

# ----- 编译与训练 -----
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 最佳模型保存到 model.keras，不在此处覆盖
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True,
        monitor="val_accuracy",
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=6,
        restore_best_weights=True,
        monitor="val_loss",
    ),
]

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = INITIAL_EPOCHS,
    callbacks = callbacks,
    class_weight = class_weights,
)

# ----- 训练曲线 -----
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Accuracy")
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")
plt.tight_layout()
plt.savefig("training_curves_basic.png", dpi = 120)
plt.show()

# # ----- 第二阶段：微调 base 最后几层 -----
# base_model.trainable = True
# fine_tune_at = len(base_model.layers) - FINE_TUNE_AT_LAYERS
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable = False
# for layer in base_model.layers[fine_tune_at:]:
#     layer.trainable = True

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"],
# )

# total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
# history_fine = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=total_epochs,
#     initial_epoch=history.epoch[-1],
#     callbacks=callbacks,
# )

# ----- 保存：最佳权重已在 ModelCheckpoint 中保存为 model.keras -----
# 最后一轮单独保存，避免覆盖“最佳模型”
# model.save(MODEL_FINAL_PATH)
print("Best model (by val_accuracy) saved to:", MODEL_PATH)
# print("Final epoch weights saved to:", MODEL_FINAL_PATH)
