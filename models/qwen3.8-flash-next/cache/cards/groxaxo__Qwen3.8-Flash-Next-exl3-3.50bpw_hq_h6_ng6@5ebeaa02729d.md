---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
  - exl3
  - exllamav3
  - qwen3.8
  - flash-next
  - moe
  - mtp
  - quantized
  - hq
language:
  - en
  - zh
pipeline_tag: image-text-to-text
---

# Qwen3.8-Flash-Next — EXL3 3.50 HQ (h6 / ng6)

EXL3 quantization of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
Built with [ExLlamaV3](https://github.com/turboderp-org/exllamav3) **1.5.0** on **3× RTX 3090 24 GB**.

This is **not** a Transformers / vLLM / SGLang checkpoint. The original Qwen README that
landed in the first upload was leftover BF16 documentation. Load it with ExLlamaV3
(TabbyAPI, or any exllamav3 server). `tensor_parallel` is **not** implemented for Qwen4Exp.

Recipe is a HQ 3.50-bpw cut of turboderp’s official-style
[`4.05bpw_h6_ng6`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/4.05bpw_h6_ng6)
branch: same head/n-gram layout, HQ on, bitrate 4.05 → 3.50, vision 6 bpw.

| | |
|---|---|
| Source | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (BF16, 336 GB / 131 shards) |
| Tight / parent EXL3 recipe | [turboderp/Qwen3.8-Flash-Next-exl3 `4.05bpw_h6_ng6`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/4.05bpw_h6_ng6) |
| Size on disk | **93 GB** (8 shards + 37 GB n-gram table) |
| Effective bitrate (`quantization_config.bits`) | **3.52** (target 3.50 HQ) |
| `lm_head` | 6 bpw |
| n-gram (PLE) table | 6 bpw |
| MTP head | 4 bpw (native `component="mtp"`, 6203 `mtp.*` keys in shard 8) |
| Vision tower | 6 bpw |
| Codebook / scales | `mul1` / `out_scales=always` |
| Calibration | 250 rows × 2048 cols |
| ExLlamaV3 | 1.5.0 |

## Related models

- **Base (BF16):** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Tight EXL3 this recipe was cut from:** [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) — specifically the [`4.05bpw_h6_ng6`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/4.05bpw_h6_ng6) branch
- **Unc sibling (different weights, not this tree):** [Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-4bpw](https://huggingface.co/Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-4bpw)

Do not mix this directory with the Unc 4.05 tree or with `mtp-Qwen38-Unc-Q8_0.gguf`. MTP here is **inside this checkpoint**.

## Conversion

```bash
python convert.py \
  --in_dir Qwen3.8-Flash-Next \
  --out_dir Qwen3.8-Flash-Next-exl3-3.50bpw_hq_h6_ng6 \
  --bits 3.50 \
  --head_bits 6 \
  --ngram_bits 6 \
  --mtp_bits 4 \
  --vision_bits 6 \
  --hq \
  --out_scales always \
  --codebook mul1 \
  --cal_rows 250 \
  --cal_cols 2048 \
  --checkpoint_interval 120 \
  --devices 0,1,2
```

`--hq` is what makes this the 3.50 HQ cut. `tensor_parallel` must stay false at serve time.

## Serve on 3× RTX 3090 (measured, not a guess)

Host: 3× GeForce RTX 3090 24 GB, driver 595.58.03, CUDA 13.2, power limit 245 W, no NVLink.
Runtime: exllamav3 1.5.0 layer-split autosplit, Sharp `chat_template.jinja`, n-gram table in RAM.
Cache length must be a **multiple of 256**. `gpu_split 23.5,23.5,23.5` **is a real cap** on this box; `24,24,24` is what actually uses the cards.

### Load ceilings (`gpu_split=24,24,24`, chunk 256)

| KV | Largest cache that loads | First fail |
|---|---|---|
| q8 | **606208** tokens | 622592 |
| q6 | **720896** tokens | 753664 |

Production on this box (fits native MTP + batch 2):

```text
CACHE_KV_BITS=8
CACHE_SIZE=400128
GPU_SPLIT=24,24,24
CHUNK_SIZE=256
MAX_BATCH_SIZE=2
NGRAM_RAM=1
ENABLE_MTP=1
NUM_DRAFT_TOKENS=3
DEFAULT_MAX_TOKENS=40000
```

TabbyAPI-style:

```yaml
model:
  model_name: Qwen3.8-Flash-Next-exl3-3.50bpw_hq_h6_ng6
  cache_size: 400128
  cache_mode: "8,8"
  ngram_ram: true
  tensor_parallel: false   # required for Qwen4Exp
  max_batch_size: 2
  chunk_size: 256
  gpu_split: [24, 24, 24]

draft_model:
  draft_mode: mtp
  num_draft_tokens: 3
```

`tensor_parallel: true` raises `NotImplementedError: Tensor-parallel is not currently implemented for Qwen4ExpForConditionalGeneration`.

### Throughput on this box (existing reports, not re-run)

Sources: `exl3-350hq-375-2026-09-15/data/q8_kv_bench.json`, `JOURNEY.md`, and
`exl3-350hq-mtp-draft-q8-2026-09-16/{README,BLOG,data/*.json}`.

**Early q8 / 160k / split 23.5, MTP off** (cloned from the Unc 4.05 family, then KV set to 8):

- Load 32.6 s. VRAM after load: 23247 / 22889 / 16603 MiB.
- 194-token prompt + 256 new: TTFT 0.236 s, **65.9 decode tok/s** after TTFT.
- 160256 / 163840 / 180224 q8 did **not** load on split 23.5.

**After moving to split 24,24,24, still MTP off, q8 524288 live:**

- Proxy stream 256: TTFT 0.255 s, **67.9 decode tok/s**.
- q8 live 524288; q8 load ceiling 606208.

**Native MTP draft-window sweep** (q8 400128, thinking off, n=1, short prose mean):

| tag | draft N | mean tok/s | accept (prose / prose2 / mid) |
|---|---|---|---|
| off | 0 | 64.67 | — |
| d2 | 2 | 84.38 | 0.54 / 0.57 / 0.56 |
| **d3** | **3** | **88.16** | 0.44 / 0.55 / 0.50 |
| d4 | 4 | 80.71 | 0.37 / 0.41 / 0.40 |
| d6 | 6 | 62.35 | 0.23 / 0.28 / 0.30 |

Draft **3** is the keep setting here. Draft 6 is slower than MTP off.

**Instruct vs thinking on d3** (cap 16384, n=2 except mid; thinking `presence_penalty=0.0` in the bench, production thinking still uses 1.0):

| mode | short-prose median tok/s | mid ~3.8k | finish |
|---|---|---|---|
| instruct | 78.4 / 81.2 | 77.7 | stop |
| thinking | 110.2 / 133.2 | 131.7 | stop at 16k (4k often `length`) |

**Concurrency 2** through the OpenAI-compatible proxy (`active=2/2`, max 512): instruct aggregate **111.2 tok/s** (~56/req) vs 81.4 solo.

Samplers used on the proxy:

- Instruct: temp 0.7, top_p 0.80, top_k 20, presence_penalty 1.5, thinking off
- Thinking / medium / xhigh: temp 1.0, top_p 0.95, top_k 20, presence_penalty 1.0, thinking on

These numbers are **this** 3×3090 workstation, this checkpoint, these prompts. They are not a quality score vs BF16 and they are not a claim about other GPUs.

## Caveats (from the same reports)

- First request after reload is slow (mid-ctx looked ~25 tok/s cold, ~78 tok/s warm).
- Thinking CoT length is unstable (565 vs 5819 new tokens on the same prompt).
- Long-context (30k–75k) traffic was measured earlier **without** MTP at ~50 tok/s; that point was not re-measured with d3.
- Vision/mmproj was not brought up on this family (text-only serve).
- Cache seq len must be a multiple of 256; chunk 128 is invalid.

## License

Same as the source: Qwen Community 1.0. See
[Qwen/Qwen3.8-Flash-Next LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
