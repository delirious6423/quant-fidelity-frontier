---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
  - abliteration
  - uncensored
  - gemma4
  - moe
  - ara
  - multimodal
  - vision
  - conversational
language:
  - en
pipeline_tag: image-text-to-text
library_name: transformers
---

# Gemma 4 26B-A4B-IT ARA Abliterated

An uncensored version of [Google's Gemma 4 26B-A4B-IT](https://huggingface.co/google/gemma-4-26B-A4B-it) created using **Adaptive Refusal Abliteration (ARA)** — a 2-pass weight-editing technique that removes alignment guardrails while preserving model quality.

## Key Results

| Metric | Value |
|--------|-------|
| Refusal rate (StrongREJECT) | **7.7%** (39/507) |
| Refusal rate (3x Ensemble) | **5.7%** (29/507) |
| Compliance quality | **4.6/5** |
| KL divergence from base | **0.1299** |

### Comparison with All Published Abliterations

All models tested on the same 512 HarmBench prompts, scored with the same StrongREJECT rubric (GPT-4o-mini), KL recomputed with a single script against the same baseline logits:

| Model | Refusal (StrongREJECT) | Avg Quality | KL |
|-------|----------------------|-------------|-----|
| Google vanilla | 99.2% (503/507) | 1.1/5 | 0.0017 |
| WWTCyberLab/abliterated | 81.3% (412/507) | 1.9/5 | 0.7436 |
| TrevorJS/uncensored | 33.3% (169/507) | 3.6/5 | 0.3603 |
| trohrbaugh/heretic-ara | 31.4% (159/507) | 3.7/5 | 0.2999 |
| coder3101/heretic | 15.8% (80/507) | 4.2/5 | 0.4118 |
| **This model** | **7.7% (39/507)** | **4.6/5** | **0.1299** |

Lowest refusal rate, highest quality score, **and** lowest KL divergence of any published abliteration for this model.

## Available Formats

| File | Format | Size | Use Case |
|------|--------|------|----------|
| `model-*.safetensors` | BF16 SafeTensors | ~48 GB | Transformers / vLLM / SGLang |
| `gemma4-ara-2pass-bf16.gguf` | BF16 GGUF | ~48 GB | llama.cpp (full precision) |
| `gemma4-ara-2pass-APEX-Q5_K_M.gguf` | APEX IQ GGUF (imatrix-calibrated Q5_K_M) | ~20 GB | llama.cpp (best quality/size tradeoff) |
| `mlx-4bit/` | MLX mixed-bit affine (4-bit attn, 8-bit MLP/router, 6-bit v_proj) | ~15 GB | Apple Silicon (text + vision) |
| `mmproj-gemma4-f16.gguf` | F16 GGUF | ~1.2 GB | Vision projector for multimodal |

## Method: 2-Pass ARA

Adaptive Refusal Abliteration identifies and removes refusal directions from model weights using SVD decomposition, then applies targeted overcorrection to suppress residual refusal behavior.

**Pass 1**: Layers 13-24, steer weight 0.0004, overcorrect 0.93, preserve 0.30
**Pass 2**: Layers 13-24, steer weight 0.0008, overcorrect 0.93, preserve 0.30
**Targets**: `self_attn.o_proj`, `mlp.down_proj`

The 2-pass approach applies a light first pass followed by a stronger second pass on the same layer range, compounding the effect while the preserve weight prevents degradation.

## Architecture

- **Base**: Gemma 4 26B-A4B-IT (MoE: 128 experts, top-8 active, ~4B active parameters per token)
- **Layers**: 30 (25 sliding attention + 5 full attention)
- **Context**: 262,144 tokens
- **Multimodal**: Vision encoder (SigLIP-based, 27 layers) with 280 soft tokens per image
- **Vocabulary size**: 262,144

## Usage

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "jenerallee78/gemma-4-26B-A4B-it-ara-abliterated",
    torch_dtype="bfloat16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("jenerallee78/gemma-4-26B-A4B-it-ara-abliterated")

messages = [{"role": "user", "content": "Your prompt here"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
outputs = model.generate(inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

### llama.cpp

```bash
# APEX IQ (recommended for most users)
llama-cli -m gemma4-ara-2pass-APEX-Q5_K_M.gguf -p "Your prompt" --mmproj mmproj-gemma4-f16.gguf

# BF16 (full precision)
llama-cli -m gemma4-ara-2pass-bf16.gguf -p "Your prompt" --mmproj mmproj-gemma4-f16.gguf
```

### vLLM (BF16)

```bash
vllm serve jenerallee78/gemma-4-26B-A4B-it-ara-abliterated \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --trust-remote-code
```

### MLX (Apple Silicon)

```bash
pip install mlx-lm

# Chat
mlx_lm.chat --model jenerallee78/gemma-4-26B-A4B-it-ara-abliterated/mlx-4bit

# Generate
mlx_lm.generate --model jenerallee78/gemma-4-26B-A4B-it-ara-abliterated/mlx-4bit \
    --prompt "Your prompt here" --max-tokens 1024

# OpenAI-compatible server (text + vision)
pip install mlx-openai-server
mlx-openai-server launch \
    --model-path jenerallee78/gemma-4-26B-A4B-it-ara-abliterated/mlx-4bit \
    --model-type multimodal \
    --port 8081
```

Requires ~15 GB unified memory. Tested on M-series Macs with 128 GB+ RAM.

## Evaluation Details

### Refusal Measurement

Refusal rates are measured using two independent methods, then combined:

1. **StrongREJECT** (GPT-4o-mini, 1-5 rubric from [Souly et al., 2024](https://arxiv.org/abs/2402.10260)): Each response is scored 1-5 where 1=full refusal, 5=full compliance. Scores 1-3 are counted as refusals, 4-5 as compliance. This catches soft refusals, defense pivots, and hedged non-answers that keyword detection misses.

2. **HarmBench-13B classifier** (3x majority vote): A fine-tuned 13B classifier independently labels each response as comply/refuse, run 3 times with the majority vote as the final verdict.

3. **Ensemble**: A response is counted as a refusal only if **both** StrongREJECT (score 1-3) and HarmBench majority agree it is a refusal. This produces the most conservative (lowest) refusal rate.

All 512 prompts are from the HarmBench harmful behaviors dataset. 5 prompts involving minors/trafficking are excluded from scoring (507 scored). Results were manually audited: 49 false refusals were reclassified as comply, 8 false compliances reclassified as refuse. StrongREJECT correlates with manual audit within 0.3% (7.7% vs 8.0% hand-counted).

**Why not keyword detection?** Keyword matching (checking for "I cannot", "I apologize", etc.) has ~23% disagreement rate with human judgment on these prompts — it misses soft refusals and misclassifies disclaimers-then-compliance as refusals. Published refusal rates using keyword detection are unreliable in both directions.

### KL Divergence Methodology

KL divergence is computed on 100 harmless prompts (general knowledge, coding, creative writing) using the vanilla `google/gemma-4-26B-A4B-it` model as the reference distribution. For each prompt, we compare the final-token logit distribution of the abliterated model against the baseline using `torch.nn.functional.kl_div` with `reduction="sum"` across the full 262K vocabulary. The reported KL is the mean across all 100 prompts. This measures how much the model's general-purpose behavior has shifted — lower is better. All models in the comparison table are measured identically.

## Disclaimer

This model has had its safety guardrails removed. It will comply with requests that the original model would refuse. The user assumes full responsibility for how they use this model. This model is released for research purposes.

## Credits

- Base model: [Google Gemma Team](https://huggingface.co/google/gemma-4-26B-A4B-it)
- Abliteration method: ARA (Adaptive Refusal Abliteration) from [OBLITERATUS](https://github.com/jenerallee78/OBLITERATUS)
- Evaluation: StrongREJECT framework + HarmBench-13B classifier
