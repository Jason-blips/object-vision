# 图像识别 / 实时分类项目

通过爬取图片 → 清洗 → 训练 → 实时摄像头推理，支持**任意类别**，可自行增加类别并反复训练以提升准确率。

## 环境

```bash
pip install -r requirements.txt
```

## 目录与流程

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 1. 爬图 | `search_cup_images.py` | 按关键词下载图片到 `data/<类别名>/`，可修改 `CrawlerConfig.keywords` 增加类别 |
| 2. 清洗 | `clean_images.py` | 删除损坏图片、将非 RGB 转为 RGB |
| 3. 训练 | `train_model.py` | 读取 `data/` 下所有子文件夹作为类别，训练 MobileNetV2，保存 `model.keras` 和 `model_classes.json` |
| 4. 推理 | `images_scanner.py` | 打开摄像头，加载模型与类别列表，实时显示预测（按 Q 退出） |

统一配置在 `config.py`（数据目录、模型路径、图像尺寸、置信度阈值、训练轮数等），便于调参。

## 扩展类别

1. 在 `search_cup_images.py` 的 `CrawlerConfig.keywords` 中增加新关键词（如 `["cup", "earphone", "glasses", "keyboard"]`）。
2. 运行爬虫，得到 `data/keyboard/` 等新目录。
3. 运行 `clean_images.py` 清理新数据。
4. 重新运行 `train_model.py`，会自动识别所有子文件夹并更新 `model_classes.json`。
5. 运行 `images_scanner.py` 即可用新模型实时识别。

## 提高准确率建议

- 每类至少几十张以上、尽量多样（角度、光照、背景）。
- 在 `config.py` 中可调：`BATCH_SIZE`、`INITIAL_EPOCHS`、`FINE_TUNE_EPOCHS`、`FINE_TUNE_AT_LAYERS`。
- 训练时最佳模型按验证集准确率保存为 `model.keras`，最后一轮权重另存为 `model_final.keras`，可按需选用。
