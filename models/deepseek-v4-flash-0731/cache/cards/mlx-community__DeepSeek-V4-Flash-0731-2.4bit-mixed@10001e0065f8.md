---
license: mit
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- oq
- quantized
- moe
- deepseek
---
# DeepSeek-V4-Flash-0731-2.4bit-mixed

Mixed-precision MLX quantization of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (284B total, ~304B with MTP, ~13.8B active per token), at **2.44 bits/weight effective**, 92.8 GB on disk. Produced with [oMLX](https://github.com/jundot/omlx) oQ level 2 enhanced and then hand-tuned: I requantized attention and the MTP experts myself, outside the oQ allocator, to get under 2.5 bpw without touching what the model actually needs. For Apple Silicon.

- **92.8 GB** on disk (86.5 GiB), down from 155 GB in the source's native FP8 e4m3 + FP4-expert format
- 43 layers, 256 routed experts + 1 shared, top-6; MLA with grouped output projections, the sparse-attention indexer (`index_topk 512`), hyper-connections with Sinkhorn, YaRN to 1M context
- **All three DSpark MTP blocks are preserved** (19.8B params)
- Peak memory in my smoke test: **84.4 GB** at short context, so this wants a 128 GB Mac
- Converted and tested on a **Macbook Pro M5 Max 128GB 40 GPU**

## Requirements

**oMLX 0.5.7 or newer.** mlx-lm does not support the `deepseek_v4` architecture. There are half a dozen open PRs ([mlx-lm#1189](https://github.com/ml-explore/mlx-lm/pull/1189) among them) and I haven't checked myself whether any of them work. oMLX loads this repo directly from the HF cache with no model setting.

The upstream checkpoint ships no chat template, and I kept it that way rather than inventing one. Use oMLX's `apply_chat_template`, as in Usage below.

## Quantization

The base is 2-bit affine at group size 128 and it carries the routed experts, which are 84% of the file. Everything read on every token got promoted: attention to 6 bits at group size 128, and the shared expert, embeddings and `lm_head` to 8 bits at group size 64. The MTP blocks' own experts sit at 3 bits, above the main model's 2, because there are only three of them and they cost almost nothing.

| Group | Tensors | Bits | Group size | On disk |
|---|---|---|---|---|
| routed experts (main model) | 129 | 2 | 128 | 77.9 GB |
| attention (`wq_a/wq_b/wkv/wo_a/wo_b`, compressor, indexer) | 396 | 6 | 128 | 4.0 GB |
| shared experts | 138 | 8 | 64 | 1.2 GB |
| `embed_tokens`, `lm_head` | 2 | 8 | 64 | 1.1 GB |
| MTP routed experts | 9 | 3 | 128 | 8.4 GB (whole MTP stack) |

Output is standard MLX affine quantization, so no custom kernels or runtime are required.

## Conversion check

Smoke-tested after conversion with oMLX's loader, three ways: thinking mode greedy, chat mode greedy, and thinking mode at temperature 0.6 with repetition penalty 1.1. Coherent in all three, English throughout, solved `17 * 24 = 408` by the distributive property with the `<think>` block opening and closing correctly. No repetition loops. Greedy decoding is the thing to check on a build this aggressive: the earlier uniform 2-bit attempts only held together with sampling on, and collapsed into repeated fragments or drifted into Chinese mid-answer under greedy.

## Performance

Measured with oMLX's benchmark harness on a **Macbook Pro M5 Max 128GB 40 GPU**, single request, 128 generated tokens, speculative decoding off.

| prompt | gen tok/s | prefill tok/s | TTFT ms | peak GB |
|---|---|---|---|---|
| 1k | 36.1 | 498.3 | 2055 | 79.76 |
| 4k | 33.8 | 431.6 | 9490 | 79.77 |
| 8k | 33.1 | 371.1 | 22078 | 79.86 |
| 16k | 32.6 | 352.6 | 46470 | 80.17 |
| 32k | 31.5 | 336.6 | 97347 | 80.80 |

Two things worth pointing out. Generation barely degrades with context, 33.8 tok/s at 4k against 31.5 at 32k, which is the sparse-attention indexer doing its job: `index_topk 512` bounds how many tokens enter the attention computation regardless of prompt length. And peak memory rises by **1.03 GB across an 8x jump in context**, because MLA compresses the KV cache hard. That is what makes a 304B model with 32k of context livable on a 128 GB machine. The cost sits in prefill: 97 seconds to first token at 32k.

Continuous batching at 1k prompt / 128 generated:

| batch | tg tok/s | speedup | TTFT ms | E2E s |
|---|---|---|---|---|
| 1 | 36.1 | 1.00x | 2055 | 5.61 |
| 2 | 43.9 | 1.22x | 4714 | 10.54 |
| 4 | 64.1 | 1.78x | 7488 | 16.92 |
| 8 | 83.2 | 2.30x | 15511 | 28.30 |

## Accuracy

mmlu_pro, n=600 seeded samples, thinking off, greedy, identical questions on both rows. The bf16 row is the hosted API measured the same way, through an OpenAI-compatible endpoint with the same prompts, the same `max_tokens` and temperature 0. Standard error at this n is about 2 points on each row.

| Variant | Size | bpw | mmlu_pro (n=600) |
|---|---|---|---|
| [**DeepSeek-V4-Flash-0731-2.4bit-mixed**](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed) (this repo) | 92.8 GB | 2.44 | 0.573 |
| [DeepSeek V4 Flash 0731 (API, bf16)](https://openrouter.ai/deepseek/deepseek-v4-flash-0731) | — | 16 | 0.647 |

That is 7.3 points below full precision, which at this sample size sits around 2.6 standard errors: enough to point at a real loss, not enough to pin its size. A drop of roughly this shape is what you would expect from quantization at 2.44 bpw, and I would treat it as such rather than as a measured constant.

Treat this as a rough sighting, not a verdict. One benchmark at n=600 covers a narrow slice of what the model does, with no long-context work, no agentic loops and no real code, which is most of what this checkpoint was built for. I ran it to size the cost of the bit-width, not to certify the model. Test it on your own workload before trusting any of it.

## Usage

```bash
# oMLX discovers this from the HF cache and serves it as deepseek-v4-flash-0731-2-4bit-mixed
omlx serve
```

```python
from omlx.utils.model_loading import load_text_model
from omlx.patches.deepseek_v4.chat_template_v4 import apply_chat_template
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

model, tok = load_text_model("mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed")
prompt = apply_chat_template(
    [{"role": "user", "content": "Explain Bayes' theorem in two sentences."}],
    add_generation_prompt=True,
    thinking_mode="thinking",   # or "chat"
)
print(generate(model, tok, prompt, max_tokens=512, sampler=make_sampler(temp=0.0)))
```

## License

[MIT](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE), inherited from the base model. Refer to the original model card for architecture, benchmarks, and intended use.
