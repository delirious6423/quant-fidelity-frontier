---
license: mit
base_model:
- Qwen/Qwen3.8-27B
tags:
- Thor
- T5000
- Jetson
- NVIDIA
- NVFP4
---
Important: This is a NVFP4 model specifically for TensorRT-Edge-LLM on NVIDIA Jetson Thor T5000. vLLM may NOT load it. Be careful before you download it.

Usage:
1. download the model to your local folder, for example, /home/sd1/models
2. create a folder for model cache, for example, /home/sd1/data
3. launch the docker: (it will build TensorRT engine for the 1st time, then reuse the cache later)

```bash
docker run -d --runtime nvidia --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --name qwen38-mtp --restart always \
    -p 8089:8001 \
    -v /home/sicore/data:/data \
    -v /home/sicore/models:/models \
    harbor.sicoreai.com/thor/tensorrt-edge-llm:0.10.1 \
    tensorrt-edgellm-serve /models/Qwen3.8-27B-NVFP4 \
      --served-model-name qwen3.8-27b \
      --api-key "sk-your.key" \
      --cache-dir /data/edgellm-cache \
      --enable-auto-tool-choice \
      --tool-call-parser auto \
      --reasoning-parser qwen3 \
      --max-batch-size 4 \
      --max-input-len 32768 \
      --max-kv-cache-capacity 65536 \
      --max-verify-tree-size 6 \
      --max-draft-tree-size 6 \
      --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
      --enable-batching \
      --port 8001
```

Code example for text inference:

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your.key",
    base_url="http://[your ip]:8089/v1/", 
)

completion = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[
        {"role": "system", "content": "You're a helpful assistant"},
        {"role": "user", "content": "Please introduce Beijing, then let me know who are you"},
    ],
    timeout=300 
)
#print(completion)
#print(completion.choices[0].message.content)

choice = completion.choices[0]
msg = choice.message
text = msg.content or msg.reasoning
print(text)
```

Code example for image inference:

```python
import base64
from openai import OpenAI

# --- Configuration ---

api_key = "sk-your.key"
image_path = "D:\\download\\NewImage.png"  # Replace with the actual path to your image

# --- OpenAI Client Initialization ---

client = OpenAI(
    api_key=api_key,
    base_url="http://[your ip]:8089/v1",
)

# --- Base64 Image Encoding ---
def encode_image_to_base64(path):
    """Encodes an image file to a base64 string."""
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Get the base64 encoded string of the image
base64_image = encode_image_to_base64(image_path)

# --- API Request ---
try:
    response = client.chat.completions.create(
        model="qwen3.8-27b",  
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please OCR the image content."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        timeout=300  
    )
    print(response.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")
```
    
Performance test result:

![image](https://cdn-uploads.huggingface.co/production/uploads/6aa13f41d7bb14ed26fcdb83/QpHXZB78JO99RNymc3STj.png)