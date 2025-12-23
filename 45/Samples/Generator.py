import os
import string
import requests
from openai import OpenAI

class SeedreamGenerator:
    '''Single Image reference generation'''
    def __init__(self, model_name, prompt, image_path, size, response_format, api_key, base_url, shot_id:str, frame_id:str):
        self.model_name = model_name
        self.prompt = prompt
        self.image_path = image_path
        self.size = size
        self.response_format = response_format
        self.extra_body = {
            "image": [image_path],
            "watermark": False,
            "sequential_image_generation": "disabled",
        }
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            base_url="https://ark.ap-southeast.bytepluses.com/api/v3", 
            api_key="dcd63e69-449a-414b-8747-f0626a271e09",
        )
        self.shot_id = shot_id
        self.frame_id = frame_id


    def get_save_path(self):
        os.makedirs(f"/Users/haoqian3/Research/AnimationGEN/Seedream/45/Samples/output/{self.shot_id}", exist_ok=True)
        return f"/Users/haoqian3/Research/AnimationGEN/Seedream/45/Samples/output/{self.shot_id}/{self.frame_id}.png"

    def save_image(self, image_url, save_path):
        response = requests.get(image_url)
        with open(save_path, "wb") as f:
            f.write(response.content)

    def generate(self):
        imagesResponse = self.client.images.generate(
            model=self.model_name,
            prompt=self.prompt,
            size=self.size,
            response_format=self.response_format,
            extra_body=self.extra_body,
        )
        save_path = self.get_save_path()
        self.save_image(imagesResponse.data[0].url, save_path)
        return save_path

