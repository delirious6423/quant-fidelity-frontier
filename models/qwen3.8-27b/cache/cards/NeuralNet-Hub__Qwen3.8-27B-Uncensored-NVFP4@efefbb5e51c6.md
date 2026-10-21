---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
tags:
  - qwen3.8
  - qwen
  - nvfp4
  - vision-language
  - image-text-to-text
  - vllm
  - blackwell
library_name: transformers
pipeline_tag: image-text-to-text
quantized_by: NeuralNet-Hub
---


<div align="center">
  <a href="http://neuralnet.solutions" target="_blank">
    <img width="450" src="https://raw.githubusercontent.com/NeuralNet-Hub/assets/main/logo/LOGO_png_orig.png">
  </a>
</div>

NeuralNet is a pioneering AI solutions provider that empowers businesses to harness the power of artificial intelligence.


# TRY THIS MODEL FOR FREE IN [UNCENSOREDGPT](https://uncensoredgpt.ai) 

## 🔓 No Filters. No Limits. Just Answers.

<div align="center">

### This model powers [UncensoredGPT](https://uncensoredgpt.ai)

**Ask anything. Get real answers. No restrictions.**

[![Join the Waitlist](https://img.shields.io/badge/Join%20to%20The%20Freedom-UncensoredGPT-black?style=for-the-badge)](https://uncensoredgpt.ai)

</div>

Most AI models are trained to refuse. They hedge, they deflect, they lecture. UncensoredGPT is built on the opposite philosophy: that access to information should be unrestricted, and that adults are capable of deciding what they need to know.

This model is the engine behind [UncensoredGPT](https://uncensoredgpt.ai), a platform providing unfiltered, honest responses for cybersecurity, education, content creation, research, or straightforward conversation. The refusals and content restrictions present in the original model have been removed through a combination of supervised fine-tuning and abliteration, resulting in a model that responds directly across topics that standard models typically refuse.

**Why stay in the system when you can have unrestricted answers, privacy by default, and complete freedom of information?**

> Ready to experience the freedom of unrestricted AI? [Join to uncensoredgpt.ai](https://uncensoredgpt.ai)


## ⚠️ Model created by [OrcaRouter](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4) 🐋

This is an NVFP4-quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), created by [OrcaRouter](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4), it's exactly the same model, we are saving it here as we use it in [UncensoredGPT](https://uncensoredgpt.ai).

We have not created nor optimized this model, but we needed it only because we modified the chat template to make it compatible with many other applications as per the suggestion by [SerialKicked](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/5#69a0539df2abd80827e55233).

We have replaced

```
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}

```

By

```
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
```



It's optimized for deployment on **NVIDIA Blackwell architecture GPUs** using vLLM.

> [!IMPORTANT]
> NVFP4 quantization requires **NVIDIA Blackwell architecture** (GB200, RTX 5000 series, etc.). This format is not compatible with Ampere, Ada Lovelace, or Hopper GPUs. If you are running on an older GPU, please use a different quantization format.

Original model: https://huggingface.co/Qwen/Qwen3.8-27B

## ⚡ Deployment with vLLM

This quantized model is intended to be served using vLLM (`vllm>=0.9.0` recommended).

### Quick Start

```bash
vllm serve NeuralNet-Hub/Qwen3.8-27B-NVFP4 \
  --quantization nvfp4 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

### Using a Config File

```yaml
# Deploy with: vllm serve --config config.yaml

# Model config
model: NeuralNet-Hub/Qwen3.8-27B-NVFP4
dtype: bfloat16
kv-cache-dtype: fp8
gpu-memory-utilization: 0.95
max-model-len: 262144
max-num-seqs: 200
enable-prefix-caching: true
trust-remote-code: true

# Parsing
reasoning-parser: qwen3
enable-auto-tool-choice: true
tool-call-parser: qwen3_coder


# Optional
default-chat-template-kwargs: '{"enable_thinking": false}'
download-dir: /workspace/models
host: 0.0.0.0
port: 18000
```

```bash
vllm serve --config config.yaml
```

## 💬 Chat API Usage

Once you deploy your model using vLLM you can chat qwith Qwen3.8 with chat template compatible with OpenAI-format APIs. Thinking mode is enabled by default.

### Thinking Mode (Default)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:18000/v1", api_key="EMPTY")

messages = [{"role": "user", "content": "Your message here"}]

response = client.chat.completions.create(
    model="NeuralNet-Hub/Qwen3.8-27B-NVFP4",
    messages=messages,
    max_tokens=32768,
    temperature=1.0,
    top_p=0.95,
    extra_body={"top_k": 20},
)
print(response.choices[0].message.content)
```

### Non-Thinking (Instruct) Mode

```python
response = client.chat.completions.create(
    model="NeuralNet-Hub/Qwen3.8-27B-NVFP4",
    messages=messages,
    max_tokens=8192,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
```

### Image Input

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

response = client.chat.completions.create(
    model="NeuralNet-Hub/Qwen3.8-27B-NVFP4",
    messages=messages,
    max_tokens=32768,
    temperature=1.0,
    top_p=0.95,
    extra_body={"top_k": 20},
)
```

## ⚙️ Recommended Sampling Parameters

| Mode | temperature | top_p | top_k | presence_penalty |
| ---- | ----------- | ----- | ----- | ---------------- |
| Thinking — general tasks | 1.0 | 0.95 | 20 | 0.0 |
| Thinking — precise coding | 0.6 | 0.95 | 20 | 0.0 |
| Instruct (non-thinking) | 0.7 | 0.80 | 20 | 1.5 |


## 🔧 Hardware Requirements

| Component | Requirement |
| --------- | ----------- |
| GPU Architecture | NVIDIA Blackwell (sm_100+) |
| VRAM | 48 GB+ recommended |
| CUDA | 12.8+ |
| vLLM | 0.9.0+ |

> [!WARNING]
> NVFP4 is exclusively supported on NVIDIA Blackwell GPUs. Attempting to run this model on Ampere (A100), Ada Lovelace (RTX 4000), or Hopper (H100) will fail. For those architectures, use the original BF16 model or an AWQ/GPTQ quantized variant.

## 🌐 Contact Us

NeuralNet is a pioneering AI solutions provider that empowers businesses to harness the power of artificial intelligence.

Website: https://neuralnet.solutions  
Email: info[at]neuralnet.solutions