# Seed Dream Image Generation System
import os
import sys
import time
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

from Samples.Generator import SeedreamGenerator

# ==================== 配置 ====================
CONFIG = {
    "csv_path": "/Users/haoqian3/Research/AnimationGEN/TestSet/test_data_7B_v16.csv",
    "model_name": "seedream-4-5-251128",
    "size": "2K",
    "response_format": "url",
    "image_base_url": "https://raw.githubusercontent.com/DumanHaoqian/Seedream/main/45/anime_sketch",
    "output_dir": "/Users/haoqian3/Research/AnimationGEN/Seedream/45/Samples/output",
    "progress_file": "/Users/haoqian3/Research/AnimationGEN/Seedream/45/Samples/progress.json",
    "failed_log_file": "/Users/haoqian3/Research/AnimationGEN/Seedream/45/Samples/failed_generations.log",
    
    # 安全措施配置
    "sleep_between_requests": 2,  # 每次请求之间的休眠时间（秒）
    "sleep_after_error": 10,      # 错误后的休眠时间（秒）
    "max_retries": 3,             # 最大重试次数
    "retry_delay": 5,             # 重试间隔基准时间（秒）
    "skip_existing": True,        # 是否跳过已存在的图片
}

# ==================== Logging配置 ====================
def setup_logging():
    """配置日志系统"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 文件handler
    log_file = f"/Users/haoqian3/Research/AnimationGEN/Seedream/45/Samples/generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 配置root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)

logger = setup_logging()


# ==================== 进度跟踪 ====================
class ProgressTracker:
    """跟踪生成进度和时间估计"""
    
    def __init__(self, total_items: int, progress_file: str):
        self.total_items = total_items
        self.progress_file = progress_file
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()
        self.generation_times: List[float] = []
        self.completed_items: set = set()
        
        # 加载之前的进度
        self.load_progress()
    
    def load_progress(self):
        """从文件加载进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.completed_items = set(data.get('completed_items', []))
                    self.generation_times = data.get('generation_times', [])[-100:]  # 只保留最近100个时间
                    logger.info(f"Loaded progress: {len(self.completed_items)} items previously completed")
            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}")
    
    def save_progress(self):
        """保存进度到文件"""
        try:
            data = {
                'completed_items': list(self.completed_items),
                'generation_times': self.generation_times[-100:],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")
    
    def is_completed(self, item_id: str) -> bool:
        """检查项目是否已完成"""
        return item_id in self.completed_items
    
    def mark_completed(self, item_id: str, generation_time: float):
        """标记项目为已完成"""
        self.completed += 1
        self.completed_items.add(item_id)
        self.generation_times.append(generation_time)
        
        # 每完成10个保存一次进度
        if self.completed % 10 == 0:
            self.save_progress()
    
    def mark_failed(self, item_id: str):
        """标记项目为失败"""
        self.failed += 1
    
    def mark_skipped(self):
        """标记项目为跳过"""
        self.skipped += 1
    
    def get_avg_generation_time(self) -> float:
        """获取平均生成时间"""
        if not self.generation_times:
            return 30.0  # 默认估计30秒
        return sum(self.generation_times) / len(self.generation_times)
    
    def get_eta(self, current_index: int) -> str:
        """计算预计剩余时间"""
        remaining = self.total_items - current_index - 1
        if remaining <= 0:
            return "即将完成"
        
        avg_time = self.get_avg_generation_time()
        eta_seconds = remaining * avg_time
        eta = timedelta(seconds=int(eta_seconds))
        
        return str(eta)
    
    def get_elapsed_time(self) -> str:
        """获取已用时间"""
        elapsed = time.time() - self.start_time
        return str(timedelta(seconds=int(elapsed)))
    
    def print_summary(self):
        """打印最终摘要"""
        elapsed = self.get_elapsed_time()
        avg_time = self.get_avg_generation_time()
        
        logger.info("=" * 60)
        logger.info("Generation Summary")
        logger.info("=" * 60)
        logger.info(f"Total items: {self.total_items}")
        logger.info(f"Completed: {self.completed}")
        logger.info(f"Failed: {self.failed}")
        logger.info(f"Skipped (already existed): {self.skipped}")
        logger.info(f"Total time: {elapsed}")
        logger.info(f"Average generation time: {avg_time:.2f} seconds")
        logger.info("=" * 60)


# ==================== 失败记录 ====================
class FailedItemLogger:
    """记录失败的项目，便于后续重试"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.failed_items: List[dict] = []
    
    def log_failure(self, shot_id: str, frame_id: str, prompt: str, error: str):
        """记录失败项目"""
        item = {
            'timestamp': datetime.now().isoformat(),
            'shot_id': shot_id,
            'frame_id': frame_id,
            'prompt': prompt[:200],  # 截断过长的prompt
            'error': str(error)
        }
        self.failed_items.append(item)
        
        # 立即写入文件
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write failure log: {e}")
    
    def get_failed_count(self) -> int:
        return len(self.failed_items)


# ==================== 主程序 ====================
def generate_single_image(
    row: pd.Series,
    config: dict,
    progress_tracker: ProgressTracker,
    failed_logger: FailedItemLogger,
    current_index: int
) -> Tuple[bool, Optional[float]]:
    """
    生成单张图片
    
    Returns:
        (success: bool, generation_time: Optional[float])
    """
    shot_id = row["shot_id"]
    frame_id = row["frame_id"]
    prompt = row["prompt"]
    item_id = f"{shot_id}/{frame_id}"
    
    # 检查是否已完成（从进度文件）
    if progress_tracker.is_completed(item_id):
        logger.info(f"[{current_index + 1}/{progress_tracker.total_items}] Skipping (in progress file): {item_id}")
        progress_tracker.mark_skipped()
        return True, None
    
    image_path = f"{config['image_base_url']}/{shot_id}/{frame_id}_out.png"
    
    # 打印详细信息
    logger.info("=" * 60)
    logger.info(f"[{current_index + 1}/{progress_tracker.total_items}] Processing: {item_id}")
    logger.info(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
    logger.info(f"Image URL: {image_path}")
    logger.info(f"Elapsed: {progress_tracker.get_elapsed_time()} | ETA: {progress_tracker.get_eta(current_index)}")
    logger.info(f"Avg generation time: {progress_tracker.get_avg_generation_time():.2f}s")
    
    try:
        start_time = time.time()
        
        generator = SeedreamGenerator(
            model_name=config["model_name"],
            prompt=prompt,
            image_path=image_path,
            size=config["size"],
            response_format=config["response_format"],
            shot_id=shot_id,
            frame_id=frame_id,
            max_retries=config["max_retries"],
            retry_delay=config["retry_delay"]
        )
        
        result = generator.generate(skip_existing=config["skip_existing"])
        generation_time = time.time() - start_time
        
        if result:
            logger.info(f"✓ Successfully generated: {item_id} in {generation_time:.2f}s")
            progress_tracker.mark_completed(item_id, generation_time)
            return True, generation_time
        else:
            logger.error(f"✗ Failed to generate: {item_id}")
            progress_tracker.mark_failed(item_id)
            failed_logger.log_failure(shot_id, frame_id, prompt, "Generation returned None")
            return False, None
            
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt detected. Saving progress...")
        progress_tracker.save_progress()
        raise
        
    except Exception as e:
        logger.error(f"✗ Exception during generation of {item_id}: {e}")
        progress_tracker.mark_failed(item_id)
        failed_logger.log_failure(shot_id, frame_id, prompt, str(e))
        return False, None


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Seedream Image Generation System Started")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 加载数据
    try:
        df = pd.read_csv(CONFIG["csv_path"])
        logger.info(f"Loaded {len(df)} items from CSV")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return
    
    # 初始化跟踪器
    progress_tracker = ProgressTracker(len(df), CONFIG["progress_file"])
    failed_logger = FailedItemLogger(CONFIG["failed_log_file"])
    
    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    logger.info(f"Starting generation of {len(df)} images...")
    logger.info(f"Sleep between requests: {CONFIG['sleep_between_requests']}s")
    logger.info(f"Skip existing: {CONFIG['skip_existing']}")
    
    try:
        for index, row in df.iterrows():
            success, gen_time = generate_single_image(
                row, CONFIG, progress_tracker, failed_logger, index
            )
            
            # 请求间休眠（rate limiting）
            if index < len(df) - 1:  # 不是最后一个
                if not success:
                    # 失败后休眠更长时间
                    sleep_time = CONFIG["sleep_after_error"]
                    logger.info(f"Sleeping {sleep_time}s after error...")
                else:
                    sleep_time = CONFIG["sleep_between_requests"]
                    logger.debug(f"Sleeping {sleep_time}s before next request...")
                
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.warning("\n" + "=" * 60)
        logger.warning("Generation interrupted by user")
        logger.warning("Progress has been saved. Run again to continue.")
        logger.warning("=" * 60)
    
    finally:
        # 保存最终进度
        progress_tracker.save_progress()
        progress_tracker.print_summary()
        
        if failed_logger.get_failed_count() > 0:
            logger.warning(f"Failed items logged to: {CONFIG['failed_log_file']}")
            logger.warning("You can retry failed items later.")


if __name__ == "__main__":
    main()
