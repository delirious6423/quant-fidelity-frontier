---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: colibri
pipeline_tag: image-text-to-text
tags:
  - colibri
  - moe
  - int4
  - cpu
  - expert-streaming
language:
  - en
  - zh
  - it
---

# GLM-5.3-Flash for colibrì (int4-gs64)

An int4 group-scaled conversion of
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), packed
in the container format that [colibrì](https://github.com/JustVugg/colibri)
streams from disk.

**This is not the official checkpoint.** The weights are Z.ai's; this repository
only changes how they are stored. It will not load in transformers or vLLM. It
loads in colibrì's `glm53` engine and nothing else.

Text and vision both work.

## What it is for

Running a 321B multimodal model on a machine with 25 GB of RAM, by keeping the
routed experts on disk and reading only the ones each token selects.

| | |
|---|---|
| on disk | 194.7 GB across 62 shards |
| resident RAM | about 12 GB at `GLM53_BITS=4` |
| KV state | 33 KB per token (1.1 GB at 32k context) |
| decode | about 44 s/token cold, about 20 s/token with a warm expert cache |

Those decode numbers are honest and they are the point of the next section.

## Read this before you download 195 GB

**The disk is the wall.** One token touches 42 sparse layers times 8 experts
times 14.2 MB, which is 4.8 GB of weights read per token. Measured with
`O_DIRECT`, the reference drive saturates near 200 MB/s, so there is a **floor
of roughly 24 seconds per token** on that hardware. No CPU and no GPU moves it.
Faster storage does; fewer bytes would.

So this is a model you can run, not one you can chat with quickly. If that
trade is not what you want, the smaller families colibrì supports (OLMoE,
Qwen3.6) are interactive on ordinary hardware.

## Use

```bash
git clone https://github.com/JustVugg/colibri && cd colibri/c
make glm53

./coli chat  --model /path/to/this/repo --no-think
./coli serve --model /path/to/this/repo
./coli web   --model /path/to/this/repo
```

### Images

Paste a path straight into `coli chat`, attach or drop a file in `coli web`, or
send an OpenAI `image_url` part with a base64 data URI or a local path.

`GLM53_MAX_IMAGE_TOKENS` is the setting that matters. The checkpoint's own
ceiling is 8000 tokens per image, which on a disk-streaming engine is a prefill
nobody will sit through. 256 keeps ordinary text legible, 64 keeps shapes and
colours. The image is shrunk, not cropped.

### Reasoning

The generation prompt opens `<think>` and the model closes it, which is what the
official template does. `--no-think` closes it immediately and the answer starts
at the first word. `--effort low|high|max` picks the level the template
understands. Both are time controls on this engine rather than matters of taste.

## What the conversion did

Routed experts become **int4 group-scaled at 64**: `name` as U8 packed nibbles
plus `name.qs` as F32 scales, stored flat, with shapes taken from `config.json`.
Everything else stays **BF16**, which is 9.7B parameters of 321 and about 18 GB,
so the precision of the dense set is a load-time choice and retuning it never
means downloading the source repository again.

Every tensor kind is classified explicitly during conversion and an
unrecognised name stops it, because a converter that silently skips what it does
not know produces a checkpoint that loads and is quietly missing a tensor.

Reproduce it with:

```bash
python3 tools/convert_glm53.py --outdir /path/glm53_i4 --min-free-gb 30
```

On the reference machine that was 62 shards, 194.7 GB out, about 25 hours. Peak
disk is the output plus a single 5 GB source shard, never the source
repository's 328 GB.

## Architecture, as read from this container

45 text layers plus one MTP, hidden 4096, vocab 154880, context up to 1,048,576.
34 layers use KDA linear attention and 11 use DSA. Attention is MLA with
`q_lora_rank` 1536, `kv_lora_rank` 512, 64 heads, `qk_nope_head_dim` 256 and
`qk_rope_head_dim` **0**, so there is no rotary component. 288 routed experts
with top-8 selection plus one shared expert, `moe_intermediate_size` 2048, the
first 3 layers dense. A 24-block ViT tower at patch 14 and 448 px.

The KV cache is absorbed rather than expanded. `kv_b_proj` is folded into both
ends, so the cache holds the 512-wide latent instead of expanded keys:

```
score_j = q · (W_k c_j) = (W_kᵀ q) · c_j
out     = Σ_j a_j (W_v c_j) = W_v (Σ_j a_j c_j)
```

An identity, not an approximation, and the reason the KV state is 33 KB per
token instead of 1.39 MB.

## Quality

No independent benchmark of this container has been published yet. On GLM-5.2,
group-scaled int4 measured within noise of int8 on packaged benchmarks, while
per-row int4 scales cost materially more, which is why this uses groups of 64.
If you measure this one, please open an issue on the colibrì repository with the
numbers.

## Licence and attribution

MIT, inherited from
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash). The
model is Z.ai's work; this repository redistributes a requantised copy of their
weights and claims nothing beyond the conversion.

Engine: [github.com/JustVugg/colibri](https://github.com/JustVugg/colibri) ·
engine notes: [`docs/glm53-flash.md`](https://github.com/JustVugg/colibri/blob/main/docs/glm53-flash.md)
