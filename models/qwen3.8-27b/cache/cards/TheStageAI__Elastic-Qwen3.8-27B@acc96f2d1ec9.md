---
license: other
license_name: thestageai-elastic
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
---

# Elastic model: Qwen3.8-27B

## Overview

---

ElasticModels are the models produced by TheStage AI ANNA: Automated Neural Networks Accelerator. ANNA allows you to control model size, latency and quality with a simple slider movement, routing different compression algorithms to different layers. For each model, we have produced a series of optimized models:

- **XL**: The original model without quantization, 50.2 GiB, 2,025 tok/s; GSM8K-Platinum 0.959.
- **L**: Quantized model from TheStage, 22.8 GiB, 2,795 tok/s; GSM8K-Platinum 0.959.
- **M**: Faster quantized model from TheStage, 20.4 GiB, 2,883 tok/s; GSM8K-Platinum 0.959.
- **S**: The fastest quantized model from TheStage, 17.9 GiB, 2,943 tok/s; GSM8K-Platinum 0.959.

Speed is output tokens per second on one B200 with 64 parallel requests, without speculative decoding. Models can be served with thestage-vllm, an OpenAI-compatible server (see Usage and Invocation sections).


## Usage

---

### System Requirements

| **Property**| **Value** |
 | ---  | ---  |
| **GPU** | B200 or RTX PRO 6000 |
| **NVIDIA driver** | R580 or newer (CUDA 13) |
| **CUDA toolkit** | 13.x with cuBLAS development headers |
| **Python Version** | 3.12 |
| **CPU** | Intel/AMD x86_64 |
| **Package** | thestage-vllm (vLLM 0.31.0) |


### Installation

Install thestage-vllm in a Python 3.12 environment:

```bash
pip install 'thestage-vllm==0.0.2' --extra-index-url https://thestage.jfrog.io/artifactory/api/pypi/pypi-thestage-ai-production/simple
```

Log in and generate an access token on your profile page at [app.thestage.ai](https://app.thestage.ai):

```bash
export THESTAGE_AUTH_TOKEN='<YOUR_ACCESS_TOKEN>'
```

### Serving

With access to the Qwen3.8-27B offer on [app.thestage.ai](https://app.thestage.ai), launch the S variant:

```bash
thestage-vllm-serve Qwen/Qwen3.8-27B --mode S --reasoning-parser qwen3 --language-model-only
```

The server is ready when `/health` returns 200. It listens on port 8000 (`--port` changes it). `thestage-vllm-serve` takes the same arguments as `vllm serve`; `--mode` picks a size from the Overview. One server runs on one B200 or RTX PRO 6000 GPU.

Enable multi-token prediction (MTP) for speculative decoding. `num_speculative_tokens` sets the number of draft tokens per step:

```bash
thestage-vllm-serve Qwen/Qwen3.8-27B --mode S --reasoning-parser qwen3 --language-model-only --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The first start builds caches that make later starts faster. To reuse them on other servers, restart the server once with the same command, stop it, then save the caches:

```bash
thestage-vllm-start-cache build --out /volume/start-cache/model.tar
```

Copy the saved bundle to `/volume/start-cache` on the other server, then set:

```bash
export THESTAGE_START_CACHE_BUNDLE=/volume/start-cache
```

Serving starts a paid TheStage session and requires access to the model's offer. When the session ends, new requests return HTTP 402. Restart the server to begin a new paid session.


## Invocation

---

Call the OpenAI-compatible endpoint with curl:

```bash
curl -X POST 'http://127.0.0.1:8000/v1/chat/completions' \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "Qwen/Qwen3.8-27B",
        "messages":[{"role":"user","content":"Define AI"}]
    }'
```

Or use the OpenAI Python client (`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[
        {"role": "user", "content": "Define AI"}
    ]
)

print(response.choices[0].message.content)
```

`--reasoning-parser qwen3` returns the model's thinking separately in `message.reasoning`, so `message.content` holds only the answer.


## Quality Benchmarks

---

We have used the `lm_eval` library to validate the models. For each model size (S, M, L, XL), we have run GSM8K-Platinum (0-shot, 1209 problems, 3 seeds) with the chat template, reasoning enabled and Qwen's recommended sampling settings.

![Quality Benchmarking](assets/quality.png)

### Quality Benchmark Results

| **Metric/Model Size**| **S**| **M**| **L**| **XL**| **vLLM**| **SGLang** |
 | ---  | ---  | ---  | ---  | ---  | ---  | ---  |
| **GSM8K-Platinum** | 0.959 | 0.959 | 0.959 | 0.959 | 0.959 | 0.959 |


S, M, L: TheStage checkpoints. XL, vLLM and SGLang share one quality measurement of the original model. All were served by thestage-vllm on vLLM 0.31.0 on one B200 and scored with lm-eval 0.4.12; mean of 3 seeds.



## Datasets

---

- **GSM8K-Platinum**: Grade-school math word problems from the GSM8K test set with label errors removed (`madrylab/gsm8k-platinum`, 1209 problems). Only the final answer is scored. Results are averaged over seeds 42, 1234 and 4158 without speculative decoding.

**Reproduce quality.** Start the server without speculative decoding (S shown; `--mode` picks the size):

```bash
thestage-vllm-serve Qwen/Qwen3.8-27B --mode S --port 8021 --served-model-name model --max-model-len 36864 \
  --max-num-seqs 64 --max-num-batched-tokens 2560 --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
  --seed 0 --generation-config vllm --trust-remote-code --language-model-only --reasoning-parser qwen3
```

Then run once per seed (42 shown):

```bash
lm_eval --model local-chat-completions --tasks gsm8k_platinum_cot_llama --apply_chat_template --num_fewshot 0 \
  --batch_size 1 --log_samples --seed 42 \
  --model_args "model=model,base_url=http://127.0.0.1:8021/v1/chat/completions,num_concurrent=64,max_retries=3,timeout=3600,tokenized_requests=False,tokenizer_backend=None,seed=42" \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0,max_gen_toks=32768,seed=42"
```

## Metrics

---

- **Accuracy**: The proportion of final answers that exactly match the correct answer. Higher is better.


## Latency Benchmarks

---

We measured output TPS (tokens per second) for each model size with 1 and 64 parallel requests, with and without MTP. The chart shows 64 parallel requests with MTP.

![Latency Benchmarking](assets/latency.png)

### Latency Benchmark Results

Total output tokens per second across concurrent requests. Higher is better. S, M, L: TheStage checkpoints; XL: the original model; all four on thestage-vllm. vLLM, SGLang: the original model on stock vLLM 0.31.0 and SGLang 0.5.20. Results use the same B200 host. Compare throughput within this card because host hardware affects performance. XL and vLLM use the same weights. Their 2% difference with MTP at 64 requests is within run-to-run variation. The chart displays XL at the vLLM value. The table reports the measured throughput.

| **GPU/Model Size**| **S**| **M**| **L**| **XL**| **vLLM**| **SGLang** |
 | ---  | ---  | ---  | ---  | ---  | ---  | ---  |
| **B200, 1 request** | 183 | 165 | 149 | 94 | 92 | 92 |
| **B200, 64 requests** | 2943 | 2883 | 2795 | 2025 | 1910 | 1764 |
| **B200, 1 request, MTP** | 337 | 300 | 267 | 198 | 191 | 193 |
| **B200, 64 requests, MTP** | 4075 | 3911 | 3649 | 2538 | 2577 | 2197 |


## Benchmarking Methodology

---

The benchmarking was performed on a single B200 GPU, one server at a time. All columns of this card were measured on one host in one session.

> **Algorithm summary:**
> 1. Start the server (thestage-vllm, stock vLLM 0.31.0 or SGLang 0.5.20) with the same limits for every engine: 2048-token context, 64 parallel sequences, 90% GPU memory, prefix caching off.
> 2. Send the same random prompts to every engine, with 384-640 input tokens and fixed output lengths of 60-180 tokens.
> 3. Run at 1, 8, 32 and 64 parallel requests, without and with MTP (3 draft tokens per step).
> 4. Measure output tokens per second. At 64 parallel requests, report the median of 3 runs.

Example server configuration:

```bash
thestage-vllm-serve Qwen/Qwen3.8-27B --mode S --max-model-len 2048 --max-num-seqs 64 \
  --max-num-batched-tokens 2560 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --language-model-only \
  [--speculative-config '{"method":"mtp","num_speculative_tokens":3}']
```


## Links

---

* __Platform__: [app.thestage.ai](https://app.thestage.ai)
* __Subscribe for updates__: [TheStageAI X](https://x.com/TheStageAI)
* __Contact email__: contact@thestage.ai
