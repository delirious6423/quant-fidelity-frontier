---
library_name: mlx
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- quantized
- 4bit
- apple-silicon
- vision-language
- qwen3.8
---

# Qwen3.8-27B-MLX-4bit

A **uniform 4-bit** MLX quantization (group size 64) of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for Apple
Silicon — as a full package: bf16 vision tower, MTP
speculative-decoding head, processor configs, hardened chat template.
Every layer at 4 bits, no mixed precision.

> **Tip:** the smallest full package of this base. For quality per
> byte take the [OptiQ-5bit](https://huggingface.co/p4ik/Qwen3.8-27B-MLX-OptiQ-5bit)
> — a third of the flip rate at +3.6 GiB.

## Highlights

- 🖼️ **Image input on all three stacks: `optiq serve`, `mlx-vlm`, `vllm-mlx`.**
  Ships the base model's processor configs, which quantization
  pipelines commonly drop — without them, images are silently ignored.
- ⚡ **MTP speculative decoding, engine-agnostic.**
  Head at `mtp/weights.safetensors` — the default path `optiq serve`
  and `vllm-mlx` both search.
- 🔧 **Hardened chat template, adopted from [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B).**
  Accepts `developer`, merges system messages, guards tool-call
  arguments; renders byte-identically to the original (verified).

## How it compares

|  | <div align="center"><a href="https://huggingface.co/p4ik/Qwen3.8-27B-MLX-8bit"><img alt="uniform 8bit" src="https://img.shields.io/badge/uniform-8bit-d97706" width="76"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/p4ik/Qwen3.8-27B-MLX-OptiQ-5bit"><img alt="OptiQ 5bit" src="https://img.shields.io/badge/OptiQ-5bit-d97706" width="65"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/p4ik/Qwen3.8-27B-MLX-OptiQ-4bit"><img alt="OptiQ 4bit" src="https://img.shields.io/badge/OptiQ-4bit-d97706" width="65"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/mlx-community/Qwen3.8-27B-OptiQ-4bit"><img alt="OptiQ 4bit" src="https://img.shields.io/badge/OptiQ-4bit-d97706" width="65"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/p4ik/Qwen3.8-27B-MLX-4bit"><img alt="uniform 4bit" src="https://img.shields.io/badge/uniform-4bit-2f6feb" width="76"></a>&#8288;</div> |
|:---|:---:|:---:|:---:|:---:|:---:|
| Publisher | <div align="center"><a href="https://huggingface.co/p4ik"><img alt="p4ik" src="https://img.shields.io/badge/p4ik-555555" width="30"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/p4ik"><img alt="p4ik" src="https://img.shields.io/badge/p4ik-555555" width="30"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/p4ik"><img alt="p4ik" src="https://img.shields.io/badge/p4ik-555555" width="30"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/mlx-community"><img alt="mlx-community" src="https://img.shields.io/badge/mlx--community-555555" width="87"></a>&#8288;</div> | <div align="center"><a href="https://huggingface.co/p4ik"><img alt="p4ik (this repo)" src="https://img.shields.io/badge/p4ik-2f6feb" width="30"></a>&#8288;</div> |
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

Image input, uniform 4-bit KV cache, MTP speculation — one line:

```bash
pip install mlx-optiq
optiq serve --model p4ik/Qwen3.8-27B-MLX-4bit --mtp --kv-bits 4
```

`--kv-bits 4` (group size 64) matches the cache to the all-4-bit
footprint — the smallest total. For measured per-layer KV, take the
OptiQ repos. Text-only use works with plain `mlx-lm`; image input
also runs under `mlx-vlm` and `vllm-mlx`.

## Files

| File | Purpose |
|---|---|
| `model-*.safetensors` | Uniform 4-bit weights, group size 64 |
| `preprocessor_config.json` | Image preprocessing for mlx-vlm / vllm-mlx |
| `mtp/weights.safetensors` | MTP head — default path for optiq serve and vllm-mlx |
| `optiq/optiq_vision.safetensors` | Vision tower, bf16 |

No measured KV config and no sensitivity table — those are products of
the measured OptiQ builds.

## Sampling

From the [base model card](https://huggingface.co/Qwen/Qwen3.8-27B),
unchanged: temperature 1.0 / top_p 0.95 (thinking), 0.7 / 0.8 (instruct).
