import os
import time
import logging
import requests
from openai import OpenAI
from typing import Optional

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SeedreamGenerator:
    '''Single Image reference generation with robust error handling'''
    
    # 默认配置
    DEFAULT_TIMEOUT = 60  # 下载图片超时时间（秒）
    DEFAULT_MAX_RETRIES = 3  # 最大重试次数
    DEFAULT_RETRY_DELAY = 5  # 重试间隔（秒）
    
    def __init__(
        self, 
        model_name: str, 
        prompt: str, 
        image_path: str, 
        size: str, 
        response_format: str, 
        shot_id: str, 
        frame_id: str,
        api_key: Optional[str] = None,
        base_url: str = "https://ark.ap-southeast.bytepluses.com/api/v3",
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: int = DEFAULT_RETRY_DELAY,
        timeout: int = DEFAULT_TIMEOUT
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.image_path = image_path
        self.size = size
        self.response_format = response_format
        self.shot_id = shot_id
        self.frame_id = frame_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        self.extra_body = {
            "image": [image_path],
            "watermark": False,
            "sequential_image_generation": "disabled",
        }

        resolved_api_key =  "dcd63e69-449a-414b-8747-f0626a271e09"
        if not resolved_api_key:
            logger.warning("API key not provided. Using hardcoded key (not recommended for production)")
            resolved_api_key = "dcd63e69-449a-414b-8747-f0626a271e09"
        
        self.client = OpenAI(
            base_url=base_url, 
            api_key=resolved_api_key,
        )

    def get_save_path(self) -> str:
        output_dir = f"/Users/haoqian3/Research/AnimationGEN/Seedream/45/Samples/output/{self.shot_id}"
        os.makedirs(output_dir, exist_ok=True)
        return f"{output_dir}/{self.frame_id}.png"

    def is_already_generated(self) -> bool:
        """检查图片是否已经生成"""
        save_path = self.get_save_path()
        if os.path.exists(save_path):
            # 检查文件是否有效（大小大于0）
            if os.path.getsize(save_path) > 0:
                return True
        return False

    def save_image(self, image_url: str, save_path: str) -> bool:
        """下载并保存图片，带重试机制"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading image (attempt {attempt + 1}/{self.max_retries})...")
                response = requests.get(image_url, timeout=self.timeout)
                response.raise_for_status()  # 检查HTTP状态码
                
                # 检查内容是否为空
                if len(response.content) == 0:
                    raise ValueError("Downloaded image is empty")
                
                with open(save_path, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"Image saved successfully: {save_path} ({len(response.content) / 1024:.2f} KB)")
                return True
                
            except requests.exceptions.Timeout:
                logger.warning(f"Download timeout (attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download failed (attempt {attempt + 1}/{self.max_retries}): {e}")
            except Exception as e:
                logger.warning(f"Unexpected error during download (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        logger.error(f"Failed to download image after {self.max_retries} attempts")
        return False

    def generate(self, skip_existing: bool = True) -> Optional[str]:
        """
        生成图片，带完整错误处理和重试机制
        
        Args:
            skip_existing: 是否跳过已存在的图片
            
        Returns:
            成功返回保存路径，失败返回None
        """
        save_path = self.get_save_path()
        
        # 检查是否已生成
        if skip_existing and self.is_already_generated():
            logger.info(f"Image already exists, skipping: {save_path}")
            return save_path
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Generating image (attempt {attempt + 1}/{self.max_retries})...")
                logger.debug(f"Model: {self.model_name}, Size: {self.size}")
                logger.debug(f"Prompt: {self.prompt[:100]}..." if len(self.prompt) > 100 else f"Prompt: {self.prompt}")
                
                start_time = time.time()
                
                imagesResponse = self.client.images.generate(
                    model=self.model_name,
                    prompt=self.prompt,
                    size=self.size,
                    response_format=self.response_format,
                    extra_body=self.extra_body,
                )
                
                generation_time = time.time() - start_time
                logger.info(f"API call completed in {generation_time:.2f} seconds")
                
                # 检查响应
                if not imagesResponse.data or len(imagesResponse.data) == 0:
                    raise ValueError("API returned empty response")
                
                image_url = imagesResponse.data[0].url
                if not image_url:
                    raise ValueError("API returned empty image URL")
                
                # 下载并保存图片
                if self.save_image(image_url, save_path):
                    return save_path
                else:
                    raise Exception("Failed to save image")
                    
            except Exception as e:
                logger.error(f"Generation failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    # 指数退避
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        logger.error(f"Failed to generate image after {self.max_retries} attempts: {self.shot_id}/{self.frame_id}")
        return None
