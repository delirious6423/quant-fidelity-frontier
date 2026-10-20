---
library_name: mlx
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
- vi
- th
- id
- hi
- ru
- pl
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mlx
- liquid
- lfm2.5
- edge
- moe
- mxfp4
- 4-bit
- text-generation
- conversational
---

# randmaru/LFM2.5-24B-A2B-mlx-mxfp4

An **MXFP4** MLX quantization of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) for Apple Silicon inference, produced with `mlx-lm` **0.32.0** / `mlx` **0.32.3**.

Weights are quantized to 4-bit float (MXFP4, group size 32, E8M0 shared exponent); the MoE router gates are kept at 8-bit (affine, group size 64). The result is a single `model.safetensors` of **12.74 GB (11.86 GiB)**, averaging **≈4.25 bits per weight** — about 0.83 GB smaller than Liquid AI's own 4-bit MLX build of the same model.

This is an unofficial, community quantization and is not affiliated with or endorsed by Liquid AI. All model weights, architecture, and license terms come from the base model.

## Model details

From the [base model card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B):

- **Total parameters**: 24.0B (**2.45B active** per token)
- **Architecture**: 40 layers — 30 double-gated short-convolution blocks + 10 GQA attention layers (`layer_types` alternates `conv` with a `full_attention` layer every 4th block); `hidden_size` 2048, `moe_intermediate_size` 1536, 2 dense layers
- **MoE**: 64 experts per block, **top-4** routing, expert bias enabled
- **Vocabulary**: 128,000, embeddings tied to the output head
- **Context length**: 131,072 tokens
- **Type**: text-only reasoning model (no vision tower) — it emits an explicit chain of thought before the final answer
- **Languages**: English, Arabic, Chinese, French, German, Italian, Japanese, Korean, Portuguese, Spanish, Vietnamese, Thai, Indonesian, Hindi, Russian, Polish
- **Recommended generation settings**: `temperature: 0.1`, `top_k: 50`, `repetition_penalty: 1.05`

## Quantization

| Property | Value |
|---|---|
| Mode | `mxfp4` (4-bit float with a shared E8M0 exponent per group) |
| Group size | 32 |
| Weight storage | `U32` packed 4-bit weights + `U8` E8M0 scales |
| Scales overhead | 1 byte per 32 weights → 4.25 bpw |
| MoE router gates | kept at 8-bit (group size 64, BF16 scales and biases) |
| Embeddings | tied to the head, quantized with the rest of the weights |
| Unquantized | norms and short-conv weights in BF16; MoE `expert_bias` in F32 (0.7 MB total) |
| Output layout | a single `model.safetensors` (no shards) |
| Effective bits per weight | **≈4.25** |

The per-layer 8-bit exception for `feed_forward.gate` is written explicitly into `config.json`; it is the standard behaviour of `mlx_lm.utils.quantize_model` for this architecture, not a manual tweak.

## Comparison with Liquid AI's own 4-bit build

The direct apples-to-apples reference is [LiquidAI/LFM2.5-24B-A2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) — same base model, same runtime (MLX), same target hardware. The [GGUF repository](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) is listed as a secondary reference because it targets a different runtime (llama.cpp).

| | **`mxfp4`** (this repo) | **MLX 4-bit** (Liquid AI) | **GGUF Q4_K_M** (Liquid AI) |
|---|---|---|---|
| Reference | this repo | [LiquidAI/LFM2.5-24B-A2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) | [LiquidAI/LFM2.5-24B-A2B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) |
| Quantization scheme | MXFP4 — 4-bit float, shared E8M0 exponent, group 32 | affine 4-bit, group 64 | k-quant `Q4_K_M` (mixed 4/6-bit sub-blocks) |
| Weight storage | `U32` packed + `U8` scales | packed 4-bit + BF16 scale/bias | GGUF blocks with FP scales |
| Kept at higher precision | MoE router gates → 8-bit (group 64) | `embed_tokens` → 6-bit; MoE router gates → float32 | mixed-precision sub-blocks |
| Layout | 1 shard | 3 shards | 1 file |
| Weight bytes | **12,738,163,212** | 13,567,294,335 | 14,526,258,464 |
| Weight size | **12.74 GB** (11.86 GiB) | 13.57 GB (12.64 GiB) | 14.53 GB (13.53 GiB) |
| Total repo download | **12,756,149,125** = **12.76 GB** (11.88 GiB) | 13,585,304,293 = 13.59 GB (12.65 GiB) | 14.53 GB (13.53 GiB) |
| Effective bits per weight | **≈4.25** | ≈4.53 | ≈4.85 |
| Runtime | `mlx-lm` | `mlx-lm` | `llama.cpp`, LM Studio |
| Hardware | Apple Silicon (MLX / Metal) | Apple Silicon (MLX / Metal) | Metal, CUDA, ROCm, CPU |

### Where the size difference comes from

| Component | Bytes | Share |
|---|---|---|
| Packed 4-bit weights (`U32`) | 11,988,107,264 (11.99 GB) | 94.11% |
| E8M0 scales (`U8`, one per 32 weights) | 748,945,408 (749.0 MB) | 5.88% |
| BF16 unquantized (norms, short-conv) | 702,976 | 0.01% |
| MoE gate scales + biases (BF16) | 311,296 | <0.01% |
| MoE `expert_bias` (F32) | 9,728 | <0.01% |

The whole difference is bit-rate arithmetic, not packing tricks:

- **MXFP4, group 32**: 4 bits per weight + an 8-bit shared exponent per group → 4 + 8/32 = **4.25 bpw**.
- **affine 4-bit, group 64**: 4 bits per weight + a BF16 scale and a BF16 bias per group → 4 + 32/64 = **4.5 bpw**, pulled to ≈4.53 by the 6-bit embeddings.
- **`Q4_K_M`**: mixes 4-bit and 6-bit sub-blocks and carries more per-block scaling metadata → ≈**4.85 bpw**.

So MXFP4 lands ~0.83 GB (6.1%) under Liquid AI's 4-bit MLX build and ~1.77 GB (12.2%) under GGUF `Q4_K_M`, mostly because the shared-exponent format needs no per-group bias term and only one byte of scale per 32 weights.

### Trade-offs

- **Size vs precision**: MXFP4 is the smallest of the three at ≈4.25 bpw. It buys that with a shared exponent per group instead of per-weight-range affine scaling.
- **Numerics are not equivalent**: MXFP4 is a float format (values are `E2M1` × `2^e`), affine quantization is integer-based with a learned offset. The smaller footprint does not automatically mean lower accuracy, but it does mean different error characteristics.
- **What Liquid AI preserved**: their MLX 4-bit build keeps token embeddings at 6 bits and the MoE router gates unquantized in float32, which is generally protective for routing accuracy. This MXFP4 build instead uses a finer group size (32 vs 64) and 8-bit routers — a different trade between scale overhead and router fidelity.
- **Benchmarks**: no accuracy benchmarks are published for this quantization. The base model card carries Liquid AI's official figures for the original checkpoint; if you need numbers, evaluate this build on your own task against the reference.

## Requirements

`mlx-lm` **0.32.0 or newer**:

```bash
pip install -U "mlx-lm>=0.32.0"
```

Earlier `mlx-lm` versions load `lfm2_moe` models but route the experts incorrectly — the same caveat stated on Liquid AI's own MLX card. Loading MXFP4 weights additionally requires an MLX version with `mxfp4` support (built and tested here with `mlx` 0.32.3).

## Usage

CLI generation from the downloaded folder:

```bash
mlx_lm.generate --model ./LFM2.5-24B-A2B-mlx-mxfp4 --prompt "What is C. elegans?" --max-tokens 4096
```

OpenAI-compatible server:

```bash
mlx_lm.server --model ./LFM2.5-24B-A2B-mlx-mxfp4
```

Python, with the base model's recommended sampling settings:

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

model, tokenizer = load("./LFM2.5-24B-A2B-mlx-mxfp4")

prompt = "What is C. elegans?"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

sampler = make_sampler(temp=0.1, top_k=50)
logits_processors = make_logits_processors(repetition_penalty=1.05)

generate(
    model, tokenizer, prompt=prompt, max_tokens=4096,
    sampler=sampler, logits_processors=logits_processors, verbose=True,
)
```

Smoke test on the machine that produced this build: load peak memory 12.756 GB, 137 tok/s generation.

### Chat template

LFM2.5 uses a ChatML-like format, so `tokenizer.apply_chat_template()` is the safe path — it handles the reasoning-model chain of thought and function calling. Tool use works by passing `tools=` to `apply_chat_template`. Both are covered in Liquid AI's [chat template](https://docs.liquid.ai/lfm/key-concepts/chat-template) and [tool use](https://docs.liquid.ai/lfm/key-concepts/tool-use) docs.

## Files

| File | Size |
|---|---|
| `model.safetensors` | 12,738,163,212 |
| `tokenizer.json` | 17,905,598 |
| `model.safetensors.index.json` | 55,126 |
| `LICENSE` | 10,574 |
| `config.json` | 8,446 |
| `chat_template.jinja` | 5,443 |
| `tokenizer_config.json` | 397 |
| `generation_config.json` | 329 |

## License

LFM Open License v1.0 (`lfm1.0`) — see the [LICENSE](LICENSE) file or the [base model card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) for the full terms.

## Citation

```bibtex
@article{liquidAI202624BA2B,
  author  = {Liquid AI},
  title   = {LFM2.5-24B-A2B: Built for Fast On-Device Agents},
  journal = {Liquid AI Blog},
  year    = {2026},
  note    = {www.liquid.ai/blog/lfm2-5-24b-a2b},
}
```

```bibtex
@article{liquidai2025lfm2,
  title   = {LFM2 Technical Report},
  author  = {Liquid AI},
  journal = {arXiv preprint arXiv:2511.23404},
  year    = {2025}
}
```