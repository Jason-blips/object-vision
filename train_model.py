"""
从 data/ 下按子文件夹读取类别，训练 MobileNetV2 分类模型。
类别数不固定：任意 data/<类别名> 均可，训练后会把类别列表写入 model_classes.json 供推理使用。
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import (
    DATA_DIR,
    IMAGE_SIZE,
    MODEL_PATH,
    MODEL_FINAL_PATH,
    CLASSES_JSON,
    BATCH_SIZE,
    SEED,
    VALIDATION_SPLIT,
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_AT_LAYERS,
)

# ----- 加载数据集 -----
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
)

# 类别名与训练时顺序一致（按目录名字母序）
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes found:", class_names, "->", num_classes, "classes")

# 保存类别列表，推理时读取，避免顺序不一致
with open(CLASSES_JSON, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print("Saved class names to", CLASSES_JSON)

# ----- 数据管道 -----
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ----- 数据增强 -----
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.1, 0.1),
])

# ----- 基座 + 分类头 -----
base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

model = models.Sequential([
    data_augmentation,
    layers.Lambda(preprocess_input, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax"),
])
model.summary()

# ----- 编译与回调 -----
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
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
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
    ),
]

# ----- 类别权重 -----
train_labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels,
)
class_weights = dict(enumerate(weights))
print("Class weights:", class_weights)

# ----- 第一阶段：只训头 -----
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
)

# ----- 训练曲线 -----
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(len(acc))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Train Acc")
plt.plot(epochs_range, val_acc, label="Val Acc")
plt.legend()
plt.title("Accuracy")
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.legend()
plt.title("Loss")
plt.show()

# ----- 第二阶段：微调 base 最后几层 -----
base_model.trainable = True
fine_tune_at = len(base_model.layers) - FINE_TUNE_AT_LAYERS
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=callbacks,
)

# ----- 保存：最佳权重已在 ModelCheckpoint 中保存为 model.keras -----
# 最后一轮单独保存，避免覆盖“最佳模型”
model.save(MODEL_FINAL_PATH)
print("Best model (by val_accuracy) saved to:", MODEL_PATH)
print("Final epoch weights saved to:", MODEL_FINAL_PATH)
