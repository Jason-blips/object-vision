"""
按关键词爬取图片，每个关键词对应 data/<类别名> 一个子目录。
与 train_model 一致：data/cup、data/earphone、data/glasses 等，便于扩展任意类别。
"""
from icrawler.builtin import GoogleImageCrawler
import os
import time

from config import DATA_DIR


# ==========================
# 1. 配置参数（可扩展任意类别）
# ==========================
class CrawlerConfig:
    def __init__(self):
        self.root_dir = DATA_DIR   # 与训练一致：data/
        self.keywords = ["cup", "earphone", "glasses"]  # 可增删，每类一个子文件夹
        self.max_num_per_keyword = 150
        self.min_size = (200, 200)
        self.max_size = (4000, 4000)
        self.threads = {
            'feeder': 2,
            'parser': 4,
            'downloader': 8,
        }
        self.request_kwargs = {
            'timeout': 10,
            'retries': 3,
        }


# ==========================
# 2. 爬虫类
# ==========================
class OptimizedImageCrawler:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

    def setup_crawler(self, save_dir):
        """创建爬虫，图片保存到 save_dir（即 data/<类别名>）"""
        return GoogleImageCrawler(
            feeder_threads=2,
            parser_threads=4,
            downloader_threads=8,
            storage={'root_dir': save_dir},
        )

    def crawl_with_retry(self, keyword, save_dir, max_num, retry_times=2):
        for attempt in range(retry_times):
            try:
                crawler = self.setup_crawler(save_dir)
                crawler.crawl(
                    keyword=keyword,
                    max_num=max_num,
                    min_size=(200, 200),
                    max_size=(4000, 4000),
                    file_idx_offset=0,
                )
                return True
            except Exception as e:
                print(f"⚠️ 第{attempt + 1}次尝试失败: {e}")
                if attempt < retry_times - 1:
                    time.sleep(5)
                else:
                    return False
        return False


# ==========================
# 3. 批量爬取：每个关键词 -> data/<关键词>
# ==========================
def batch_crawl_images(config=None):
    config = config or CrawlerConfig()
    crawler = OptimizedImageCrawler(config.root_dir)

    total_downloaded = 0
    start_time = time.time()

    print("🚀 开始批量爬取图片...")
    print(f"📁 根目录: {os.path.abspath(config.root_dir)}")
    print(f"🔍 关键词（类别）: {config.keywords}")

    for keyword in config.keywords:
        # 每个类别一个目录：data/cup, data/earphone, data/glasses ...
        class_dir = os.path.join(config.root_dir, keyword.replace(' ', '_'))
        os.makedirs(class_dir, exist_ok=True)
        print(f"\n📸 正在爬取: '{keyword}' -> {class_dir}")

        success = crawler.crawl_with_retry(
            keyword=keyword,
            save_dir=class_dir,
            max_num=config.max_num_per_keyword,
        )

        if success:
            count = len([f for f in os.listdir(class_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total_downloaded += count
            print(f"✅ '{keyword}' 完成, 下载 {count} 张")
        else:
            print(f"❌ '{keyword}' 爬取失败")

        time.sleep(3)

    elapsed = time.time() - start_time
    print("\n🎉 所有任务完成!")
    print(f"📊 总下载: {total_downloaded} 张")
    print(f"⏱️ 耗时: {elapsed:.2f} 秒")
    if elapsed > 0:
        print(f"📈 平均: {total_downloaded / elapsed:.2f} 张/秒")


# ==========================
# 4. 后处理统计（扫描整个 data）
# ==========================
def post_process_images(main_dir=None):
    main_dir = main_dir or DATA_DIR
    if not os.path.exists(main_dir):
        print("❌ 目录不存在:", main_dir)
        return

    total_files = 0
    for root, dirs, files in os.walk(main_dir):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_files += len(image_files)
        if image_files:
            print(f"📂 {os.path.basename(root)}: {len(image_files)} 张")

    print(f"\n📋 共 {total_files} 张图片")


if __name__ == "__main__":
    batch_crawl_images()
    print("\n" + "=" * 50)
    post_process_images()
