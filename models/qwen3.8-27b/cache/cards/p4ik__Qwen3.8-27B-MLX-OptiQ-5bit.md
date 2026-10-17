---
library_name: mlx
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- optiq
- quantized
- mixed-precision
- 5bit
- apple-silicon
- vision-language
- qwen3.8
---

# Qwen3.8-27B-MLX-OptiQ-5bit

A **mixed-precision quant** of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for Apple
Silicon, built with [mlx-optiq](https://mlx-optiq.com) at **5.50 bits per
weight**. `5bit` names the dominant precision, not the average.

> **Note:** an expanded allocation adding a measured 6-bit tier is in
> progress and will replace this build. Same pipeline, finer grid.

## Highlights

- 🖼️ **Image input on all three stacks: `optiq serve`, `mlx-vlm`, `vllm-mlx`.**
  Ships the base model's processor configs, which quantization
  pipelines commonly drop — without them, images are silently ignored.
- 📏 **Measured weights: per-layer {4, 5, 8} bit from exact KL vs the bf16 base.**
  No structural rules — every tensor's bit-width is bought by measured
  sensitivity. Results in the table below.
- 🧠 **Measured KV cache: per-layer bit-widths, zero tool-call flips.**
  Ships as `optiq/kv_config.json` — details below.
- 🔧 **Hardened chat template, adopted from [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B).**
  Accepts `developer`, merges system messages, guards tool-call
  arguments; renders byte-identically to the original (verified).
- ⚡ **MTP speculative decoding, engine-agnostic.**
  Head at `mtp/weights.safetensors` — the default path `optiq serve`
  and `vllm-mlx` both search.

## How it compares

|  | <div align="center"><a href="https://huggingface.co/p4ik/Qwen3.8-27B-MLX-8bit"><img alt="uniform 8bit" src="https://img.shields.io/badge/uniform-8bit-d97706" width="76"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/p4ik/Qwen3.8-27B-MLX-OptiQ-5bit"><img alt="OptiQ 5bit" src="https://img.shields.io/badge/OptiQ-5bit-2f6feb" width="65"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/p4ik/Qwen3.8-27B-MLX-OptiQ-4bit"><img alt="OptiQ 4bit" src="https://img.shields.io/badge/OptiQ-4bit-d97706" width="65"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/mlx-community/Qwen3.8-27B-OptiQ-4bit"><img alt="OptiQ 4bit" src="https://img.shields.io/badge/OptiQ-4bit-d97706" width="65"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/p4ik/Qwen3.8-27B-MLX-4bit"><img alt="uniform 4bit" src="https://img.shields.io/badge/uniform-4bit-d97706" width="76"></a>&#8288;</div> |
|:---|:---:|:---:|:---:|:---:|:---:|
| Publisher | <div align="center"><a href="https://huggingface.co/p4ik"><img alt="p4ik" src="https://img.shields.io/badge/p4ik-555555" width="30"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/p4ik"><img alt="p4ik (this repo)" src="https://img.shields.io/badge/p4ik-2f6feb" width="30"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/p4ik"><img alt="p4ik" src="https://img.shields.io/badge/p4ik-555555" width="30"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/mlx-community"><img alt="mlx-community" src="https://img.shields.io/badge/mlx--community-555555" width="87"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/p4ik"><img alt="p4ik" src="https://img.shields.io/badge/p4ik-555555" width="30"></a>&#8288;</div> |
| Weights&nbsp;(GiB) | <div align="center">26.62</div> | <div align="center">17.67</div> | <div align="center">18.06</div> | <div align="center">18.09</div> | <div align="center">14.09</div> |
| BPW¹ | <div align="center">8.50</div> | <div align="center">5.64</div> | <div align="center">5.77</div> | <div align="center">5.78</div> | <div align="center">4.50</div> |
| Allocation | <div align="center">uniform²</div> | <div align="center">measured&nbsp;(bf16)</div> | <div align="center">measured&nbsp;(bf16)</div> | <div align="center">measured&nbsp;(u4)</div> | <div align="center">uniform²</div> |
| Split&nbsp;4/5/8 | <div align="center">all&nbsp;@8</div> | <div align="center">100/262/136</div> | <div align="center">270/–/228</div> | <div align="center">237/–/261</div> | <div align="center">all&nbsp;@4</div> |
| Measured&nbsp;KV | <div align="center">—</div> | <div align="center">✓</div> | <div align="center">✓</div> | <div align="center">—</div> | <div align="center">—</div> |
| Vision³ | <div align="center">✓</div> | <div align="center">✓</div> | <div align="center">✓</div> | <div align="center">OptiQ&nbsp;only</div> | <div align="center">✓</div> |
| MTP&nbsp;head³ | <div align="center">✓</div> | <div align="center">✓</div> | <div align="center">✓</div> | <div align="center">OptiQ&nbsp;only</div> | <div align="center">✓</div> |
| Hardened&nbsp;template | <div align="center">✓</div> | <div align="center">✓</div> | <div align="center">✓</div> | <div align="center">—</div> | <div align="center">✓</div> |
| ΔNLL&nbsp;overall⁴ | <div align="center">0.000&nbsp;(anchor)</div> | <div align="center">+0.019&nbsp;±&nbsp;0.019</div> | <div align="center">+0.029&nbsp;±&nbsp;0.015</div> | <div align="center">+0.040&nbsp;±&nbsp;0.030</div> | <div align="center">+0.038&nbsp;±&nbsp;0.045</div> |
| —&nbsp;German&nbsp;prose⁵ | <div align="center">0</div> | <div align="center">+0.023&nbsp;±&nbsp;0.003</div> | <div align="center">+0.019&nbsp;±&nbsp;0.002</div> | <div align="center">+0.022&nbsp;±&nbsp;0.002</div> | <div align="center">+0.039&nbsp;±&nbsp;0.003</div> |
| —&nbsp;tool-call&nbsp;spans | <div align="center">0</div> | <div align="center">−0.001&nbsp;±&nbsp;0.013</div> | <div align="center">−0.008&nbsp;±&nbsp;0.017</div> | <div align="center">+0.004&nbsp;±&nbsp;0.005</div> | <div align="center">+0.013&nbsp;±&nbsp;0.009</div> |
| —&nbsp;thinking&nbsp;spans | <div align="center">0</div> | <div align="center">−0.001&nbsp;±&nbsp;0.014</div> | <div align="center">+0.004&nbsp;±&nbsp;0.013</div> | <div align="center">−0.005&nbsp;±&nbsp;0.021</div> | <div align="center">+0.008&nbsp;±&nbsp;0.013</div> |
| Flips&nbsp;per&nbsp;10k⁶ | <div align="center">—</div> | <div align="center">216</div> | <div align="center">503</div> | <div align="center">566</div> | <div align="center">772</div> |
| Divergence⁷ | <div align="center">(anchor)</div> | <div align="center">10.3</div> | <div align="center">10.3</div> | <div align="center">9.7</div> | <div align="center">7.9</div> |

¹ Bits per weight, file-based: shard bytes × 8 / parameters, same
formula for every column.

² Our uniform reference builds — full packages (bf16 vision, MTP
head, hardened template), deliberately without measured allocation
or KV config.

³ ✓ = works on all three stacks (`optiq serve`, `mlx-vlm`, `vllm-mlx`).
Vision needs the base model's processor configs, which quantization
pipelines commonly drop; the MTP head needs the engine-neutral path
`mtp/weights.safetensors`. "OptiQ only": runs solely under
`optiq serve` — that repo lacks the processor configs, and its MTP
head sits on optiq's internal path that other engines do not search.

⁴ Paired next-token NLL over a 196k-token corpus (agentic transcripts
with tool calls and thinking, German prose, WikiText) against the
uniform 8-bit anchor; corpus and method are ours.

⁵ All German-prose deltas lie beyond 2 SE; every other ΔNLL row is
within noise.

⁶ Tokens the 8-bit anchor is near-certain about (NLL < 0.05) that jump
above NLL 0.5 — the failure mode that breaks tool-call syntax. Lower
is better.

⁷ Free-running greedy decoding, 32 tokens from 168 held-out prompt
windows of the NLL corpus: mean position of the first token that
departs from the anchor's trajectory (higher is better). Share of
trajectories still identical after 8 tokens: 46 / 48 / 46 / 35 %.

## Use

Everything at once — image input, measured KV cache, MTP speculation:

```bash
pip install mlx-optiq
optiq serve --model p4ik/Qwen3.8-27B-MLX-OptiQ-5bit --mtp \
  --kv-config "$(hf download p4ik/Qwen3.8-27B-MLX-OptiQ-5bit optiq/kv_config.json)"
```

Image input with **mlx-vlm**:

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model, processor = load("p4ik/Qwen3.8-27B-MLX-OptiQ-5bit")
config = load_config("p4ik/Qwen3.8-27B-MLX-OptiQ-5bit")

prompt = apply_chat_template(processor, config, "Describe this image.", num_images=1)
print(generate(model, processor, prompt, ["photo.jpg"]))
```

Text-only with **mlx-lm**:

```python
from mlx_lm import load, generate

model, tokenizer = load("p4ik/Qwen3.8-27B-MLX-OptiQ-5bit")
messages = [{"role": "user", "content": "Write a merge sort in Python."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt))
```

## Quantization

| Property | Value |
|---|---|
| Method | `optiq` (exact per-layer KL vs the bf16 base) |
| Candidate bits | 4 / 5 / 8 |
| Target BPW | 5.00 (achieved 5.00 before packing overhead) |
| Effective BPW | 5.50 |
| Weights | 17.67 GiB, 100 tensors @ 4-bit, 262 @ 5-bit, 136 @ 8-bit |
| Group size | 64 |
| KV cache | measured, 16 layers, 5.00 bits average |
| MTP | int4 prequantized (as shipped by optiq), 29 tensors, 300 MiB |
| Vision tower | bf16, 879 MiB |

Calibration: 67 sequences × 512 tokens — the standard mix deepened
with agentic tool-calling and German; every candidate bit-width was
measured on the same sequences, sequence length and group size. The
per-layer KL table ships in `optiq/sensitivity.json`, the final
assignment in `optiq/metadata.json`.

## KV cache

**8-bit on layers 51, 55, 59, 63; 4-bit on the other twelve** (5.00 bits
average; only the 16 `full_attention` layers carry a KV cache — the 48
`linear_attention` layers hold a fixed-size Gated DeltaNet state).

Measured with paired next-token NLL over 196k tokens (agentic transcripts
with on-policy thinking and tool calls, German prose, WikiText), each
layer probed individually. At this allocation, **tool calls, thinking and
English text are statistically indistinguishable from a full 8-bit
cache**; German prose retains ~+0.4% perplexity. A flip check over
near-certain tokens found **zero flips inside tool-call spans** — the
4-bit cache does not break tool-call syntax.

## Files

| File | Purpose |
|---|---|
| `model-*.safetensors` | Mixed-precision weights |
| `preprocessor_config.json` | Image preprocessing for mlx-vlm / vllm-mlx |
| `optiq/kv_config.json` | Measured per-layer KV bit-widths |
| `mtp/weights.safetensors` | MTP head — default path for optiq serve and vllm-mlx |
| `optiq/optiq_vision.safetensors` | Vision tower, bf16 |
| `optiq/sensitivity.json` | Measured per-layer KL table (4/5/8-bit vs bf16) |
| `optiq/metadata.json` | Final per-tensor bit assignment |

## Sampling

From the [base model card](https://huggingface.co/Qwen/Qwen3.8-27B), unchanged:

| Parameter | Thinking | Instruct |
|---|---|---|
| `temperature` | 1.0 | 0.7 |
| `top_p` | 0.95 | 0.80 |
| `top_k` | 20 | 20 |
| `min_p` | 0.0 | 0.0 |
| `presence_penalty` | 0.0 | 1.5 |
| `repetition_penalty` | 1.0 | 1.0 |

Thinking mode is on by default and can be disabled per request. Context is
262,144 tokens natively, extensible to 1M with YaRN.
