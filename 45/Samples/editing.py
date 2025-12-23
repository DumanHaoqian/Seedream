import os
from openai import OpenAI


# Make sure that you have stored the API Key in the environment variable ARK_API_KEY 
# Initialize the Ark client to read your API Key from an environment variable 
client = OpenAI( 
    # This is the default path. You can configure it based on the service location  
    base_url="https://ark.ap-southeast.bytepluses.com/api/v3", 
    # Get your Key authentication from the environment variable. This is the default mode and you can modify it as required 
    #api_key=os.environ.get("ARK_API_KEY"), 
    api_key="dcd63e69-449a-414b-8747-f0626a271e09",
) 
 
imagesResponse = client.images.generate( 
    model="seedream-4-5-251128", 
    prompt="Replace the clothing in image 1 with the outfit from image 2.",
    size="2K",
    response_format="url",
    extra_body = {
        "image": ["https://ark-doc.tos-ap-southeast-1.bytepluses.com/doc_image/seedream4_imagesToimage_1.png", 
        "https://ark-doc.tos-ap-southeast-1.bytepluses.com/doc_image/seedream4_imagesToimage_2.png"],
        "watermark": False,
        "sequential_image_generation": "disabled",
    }
) 
 
print(imagesResponse.data[0].url)