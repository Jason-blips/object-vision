"""
训练辅助：加载 data/ 下按文件夹分类的数据集、保存类别名、计算类别权重
"""
import json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import (
    DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    VALIDATION_SPLIT,
    CLASSES_JSON,
)


def load_data():
    """
    从 data/ 按子文件夹名加载图像，划分训练/验证集。
    返回: train_ds_raw, val_ds_raw, class_names, num_classes, train_labels
    """
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
    )
    class_names = train_ds_raw.class_names
    num_classes = len(class_names)

    # 收集训练集标签（用于类别权重），然后重新创建训练集（因上一句会消费掉 dataset）
    train_labels = np.concatenate([y.numpy() for _, y in train_ds_raw], axis=0)
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
    )

    return train_ds_raw, val_ds_raw, class_names, num_classes, train_labels


def save_class_names(class_names):
    """将类别名列表写入 CLASSES_JSON，推理时读取以保持顺序一致。"""
    with open(CLASSES_JSON, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)


def get_class_weights(train_labels):
    """
    根据训练集标签计算平衡类别权重，返回 {class_index: weight} 供 model.fit(class_weights=...)。
    """
    classes = np.unique(train_labels)
    weights = compute_class_weight(
        "balanced",
        classes=classes,
        y=train_labels,
    )
    return dict(zip(classes, weights))
