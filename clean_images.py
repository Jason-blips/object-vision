import os
from PIL import Image

# 与 config 一致：数据集根目录，其下为各类别子文件夹
from config import DATA_DIR
data_dir = DATA_DIR

# 统计
converted, deleted = 0, 0

for root, dirs, files in os.walk(data_dir):
    for file in files:
        file_path = os.path.join(root, file)
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # 跳过非图片文件

        try:
            with Image.open(file_path) as img:
                # 检查是否损坏
                img.verify()
        except Exception as e:
            print(f"❌ 无法打开或已损坏：{file_path}")
            os.remove(file_path)
            deleted += 1
            continue

        try:
            # 再次打开（因为 verify() 会关闭文件）
            img = Image.open(file_path)

            # 如果不是RGB就转换
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img.save(file_path)
                print(f"🔄 转换为RGB: {file_path}")
                converted += 1
        except Exception as e:
            print(f"⚠️ 转换失败：{file_path} -> {e}")
            os.remove(file_path)
            deleted += 1

print("\n✅ 清理完成！")
print(f"共转换为RGB的图片: {converted}")
print(f"共删除损坏的图片: {deleted}")
