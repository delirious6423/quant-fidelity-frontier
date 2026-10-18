---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
tags:
  - hlwq
  - hadamard-lloyd-quantization
  - deepseek
  - engram
  - sglang
  - rtx-pro-6000
  - sm120
language:
  - en
  - pt
---

# DeepSeek-V4.1-Flash · HLWQ Q4 Engram

> Sibling repo with 5-bit tables (half the reconstruction error, +25 GB): [DeepSeek-V4.1-Flash-HLWQ-Engram-Q5](https://huggingface.co/caiovicentino1/DeepSeek-V4.1-Flash-HLWQ-Engram-Q5).

**What this is:** a complete, drop-in copy of [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) in which the two Engram hash tables (`layers.1.engram.embed` and `layers.14.engram.embed`, 384 M rows × 256 each, ~196 B parameters) are re-encoded with **HLWQ (Hadamard-Lloyd Weight Quantization)** at 4 bits per weight. All 46 other shards are byte-identical to the official release; shards 47 and 48 keep their small `q/k/wkv` tensors and drop the FP8 tables. Download this repo, run `serve.sh`, done.

**Why:** the FP8 tables take ~203 GB of host RAM (or NVMe) when served with the row-store adapters used on consumer/workstation GPUs. HLWQ Q4 halves that to **~101 GB**, and the download drops from 510 GB to **~408 GB**, with a per-row reconstruction error of **9.7 % relative RMSE** (cosine 0.9954), and on the same server and prompts, next-token quality matches FP8 on prose, costs about 0.015 nats/token on code and about 0.04 nats/token on long-context retrieval prompts, with all needles still retrieved (table below).

| | FP8 (official) | HLWQ Q4 (this repo) |
|---|---|---|
| Bytes per row (256 dims) | 264 (256 fp8 + 8 e8m0) | **132** (128 packed nibbles + 2 × fp16 norms) |
| Both layers on disk / RAM | ~203 GB | **~101 GB** |
| Rel. RMSE vs FP8 rows | — | 0.0967 |
| Mean cosine vs FP8 rows | — | 0.99536 |
| Lookup cost (1024 rows, RAM mode) | ~50 µs | ~50 µs + 3 tiny torch ops |

## Quick start (4× RTX PRO 6000 or any 4× 96 GB SM120 box, ≥ 128 GB host RAM)

```bash
hf download caiovicentino1/DeepSeek-V4.1-Flash-HLWQ-Engram-Q4 --local-dir /models/v41-hlwq
docker run --gpus all --ipc=host --shm-size 32g -v /models/v41-hlwq:/model -p 8000:8000 \
  lmsysorg/sglang@sha256:ee51777250c457d339518c2687e92dc81c83142d7ab7a23d68efb20204a9a61f \
  bash -c "python /model/patches/patch_hibrido.py && SGLANG_SM120_PREFILL_BACKEND=triton API_KEY=change-me bash /model/serve.sh"
```

`serve.sh` sets `PYTHONPATH=/model/adapter` (the Engram row store + HLWQ dequant), points `DSV41_SOURCE` at the model, serves the tables from host RAM (`OFFLOAD_MODE=nvme` if you have < 128 GB), and launches SGLang with the flags validated here (TP4/EP4, `dsv4` attention, MXFP4 MoE, DSpark block 5, 16 concurrent requests, 409 600 context). Every knob is an env var; see the header of the script. The OpenAI-compatible API answers on port 8000 as `deepseek-v4.1-flash`; pass `chat_template_kwargs: {"thinking": true}` to enable reasoning (off by default in this build).

## Serving on SM120: pin the image, and `patch_hibrido.py` is the one you need

`lmsysorg/sglang:dev-dsv41` is a moving tag. It was rebuilt twice inside 24 h in September 2026, and the rebuild changed *which* of the two patches below still applies. Pin by digest:

```
lmsysorg/sglang@sha256:ee51777250c457d339518c2687e92dc81c83142d7ab7a23d68efb20204a9a61f
```

That digest is not ours. It was pinned and verified on 4× RTX PRO 6000 by [@mmssix](https://huggingface.co/mmssix) in [discussion #1](https://huggingface.co/caiovicentino1/DeepSeek-V4.1-Flash-HLWQ-Engram-Q4/discussions/1). The Vast instance we developed on no longer exists and all we kept in-repo is a truncated `sha256:c4ca6511…` in a comment header, so we cite the digest that a reader can actually verify instead of inventing one.

| patch | on the pinned digest | what it does |
|---|---|---|
| `patches/patch_hibrido.py` | **required** | anchors on `_sm120_default_backend`; creates `SGLANG_SM120_PREFILL_BACKEND`, which routes query batches > 64 tokens to `flash_mla_sparse_decode_triton` while decode stays on FlashInfer |
| `patches/patch_mask.py` | **inert — keep it applied anyway** | masks index lanes ≥ `topk_length` to −1 before the FlashInfer prefill. It anchors on the old `_sm120_prefill_backend` symbol and now fails with `AssertionError: âncora não encontrada`. Defence-in-depth for a build that restores the FlashInfer prefill path |

Earlier versions of this README had those two the other way round. They were written against the older image, where the mask patch applied and was the faster route.

**The underlying kernel bug is not fixed, so the patch step is not removable.** On the pinned digest, flashinfer is still `0.6.18`, `flash_mla_with_kvcache_sm120()` still dispatches to `_flash_mla_sm120_prefill()` whenever `q.shape[0] > 64`, and that function still passes `idx` / `topk_length` straight into `_sparse_mla_sm120_paged_attention` with no lane masking — only the decode path masks. Skipping the patch step gives **silently corrupted output for any prompt longer than 64 tokens**, not an error ([flashinfer-ai/flashinfer#5095](https://github.com/flashinfer-ai/flashinfer/issues/5095)).

Re-anchoring the mask by hand does not rescue the FlashInfer route on this build either. The model then loads completely and dies on the first request:

```
RuntimeError: Check failed: (ok) is false: Unsupported sparse-MLA prefill configuration:
model=DSV4 num_heads=16 topk=128 page_block_size=64 topk_extra=512 extra_page_block_size=128
```

`_flash_mla_sm120_prefill` splits the **main** KV source into 64-token pages but hands the **extra** source through unsplit at 128, and no kernel variant covers that pair. The source comment `patch_mask.py` anchors on described exactly that fix — 64-token pages for both sources, with independent scratch buffers — so the capability regressed out of the image rather than never having existed.

**What the Triton route costs, and it is prefill only:**

| prefill path | tok/s |
|---|---|
| FlashInfer + the −1 lane mask (older image, ours, 2026-09-10) | 6.4 k |
| `SGLANG_SM120_PREFILL_BACKEND=triton` (ours, same box) | 2.3–2.5 k |
| `SGLANG_SM120_PREFILL_BACKEND=triton` (mmssix, NVMe Engram, 409 k context) | 1.67–2.03 k |

Roughly 3× of prefill. Decode is untouched, and on the pinned digest there is no faster alternative. Note this is a narrower fallback than the workaround in [sgl-project/sglang#31578](https://github.com/sgl-project/sglang/issues/31578), which sets `SGLANG_SM120_FLASHMLA_BACKEND=triton` and sends decode to Triton as well.

**Two things that will save you a crash or an hour:**

- Do not raise `--chunked-prefill-size` to 8192: the V4.1 sparse indexer OOMs on long low-ratio prefills. `serve.sh` defaults to **2048**, which is where we settled.
- To confirm the prefill path is sound rather than silently inert, run `tools/diag_sm120_boundary.py` against the live server. It walks the prompt from ~18 to ~72 tokens and prints where the answers stop making sense; degradation starting just above 64 prompt tokens is the signature of an unmasked FlashInfer prefill. About a minute.

Blackwell datacenter GPUs (B200/B300) do not need `patches/` at all; only SM120 does.

## Files

```
model-000{01..46}-of-00048.safetensors   byte-identical to deepseek-ai/DeepSeek-V4.1-Flash
model-00047-of-00048.safetensors         layers.1.engram.{q_weight,k_weight,wkv.*} only (157 MB)
model-00048-of-00048.safetensors         layers.14.engram.{q_weight,k_weight,wkv.*} only (157 MB)
model.safetensors.index.json             official index minus the two FP8 Engram tables
engram_hlwq_q4_layer1-0000{1,2}-of-00002.safetensors    layers.1.engram.embed.hlwq_codes  U8  [rows, 128]  (two row-range parts, ~23.6 GiB each)
                                                        layers.1.engram.embed.hlwq_norms  F16 [rows, 2]
engram_hlwq_q4_layer14-0000{1,2}-of-00002.safetensors   same for layer 14
hlwq_config_layer{1,14}.json             nbits, centroids, packing, part list with row ranges, reconstruction stats, provenance
config.json, tokenizer*, encoding/, inference/, LICENSE   from the official release
adapter/                                 engram_backend_hlwq.py, row_store_hlwq.cpp, sitecustomize.py
patches/                                 SM120 prefill patches; patch_hibrido.py is REQUIRED (see patches/README.md)
tools/                                   hlwq_engram_quant.py (the quantizer), split_hlwq.py, ab_tf2.py / ab_engram.py (the validation),
                                         diag_sm120_boundary.py (checks the 64-token prefill boundary)
serve.sh                                 one-command launcher
```

## How HLWQ encodes a row

Each 256-dim row is split into two blocks of 128:

1. **Normalize** the block by its L2 norm (stored as fp16).
2. **Rotate** with the normalized Walsh-Hadamard matrix H₁₂₈. After rotation the coordinates are close to N(0, 1/128).
3. **Scale** by √128 and map each coordinate to the nearest of 16 **Lloyd-Max centroids** for N(0,1) (100 fixed-point iterations of the conditional-expectation update).
4. **Pack** the 4-bit codes two per byte (low nibble = even index).

Dequantization is the mirror image and costs three tensor ops per lookup: `centroids[codes] → (·) @ H₁₂₈ → × norm`. Because H is orthogonal and its own inverse, nothing else is stored. The centroids used are in `hlwq_config_layer*.json`.

No calibration data, no Hessian, no gradient: the quantizer is deterministic and runs on CPU (≈ 0.2 M rows/s per 96 threads).

## Serving (SGLang, 4× RTX PRO 6000, no NVLink)

This repo was produced and validated on the recipe from [0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000) (SGLang `dev-dsv41`, Engram tables served from host RAM or NVMe through a `cudaLaunchHostFunc` row store). `adapter/sitecustomize.py` autodetects `hlwq_config_layer1.json` in the model directory and installs `engram_backend_hlwq.py`; with the official FP8 checkpoint it falls back to 0xSero's FP8 row store, so the same adapter serves both. Everything else (hashing, gating, `wkv`, TP all-reduce) is the model's own code. The SM120 prefill fix is in `patches/`; which patch you need depends on the image build, so read [Serving on SM120](#serving-on-sm120-pin-the-image-and-patch_hibridopy-is-the-one-you-need) before bring-up (context in [flashinfer#5095](https://github.com/flashinfer-ai/flashinfer/issues/5095)).

## Validation (same server, same prompts, FP8 vs HLWQ)

Measured on 4× RTX PRO 6000 (SM120), SGLang `dev-dsv41`, DSpark block 5, TP4/EP4, 2026-09-10.

**Teacher-forced next-token probability** on 25 fixed FP8 greedy continuations (10 pt-BR prose, 10 code, 5 needle-in-a-haystack from 3 k to 52 k prompt tokens), 8 positions each (16 for needles). Lower NLL is better. Both columns were measured with the server idle, where two consecutive passes reproduce to the fourth decimal; under concurrent traffic the same metric moves by up to ±0.015 because batching changes the numerics, so compare only idle-to-idle.

| Category | FP8 NLL / token | HLWQ Q4 NLL / token | top-1 agreement FP8 | top-1 agreement HLWQ |
|---|---|---|---|---|
| pt-BR prose (80 positions) | 0.0668 | 0.0601 | 98.8 % | 100.0 % |
| code (80 positions) | 0.1086 | 0.1233 | 97.5 % | 93.8 % |
| needle, 3–52 k ctx (37 positions) | 0.0840 | 0.1228 | 94.6 % | 97.3 % |

**Needle-in-a-haystack (free generation):** 5/5 facts retrieved correctly at 3.4 k, 9.6 k, 22 k, 39 k and 52 k prompt tokens (identical answers to FP8; wording differs in one case). Token-level identity to the FP8 greedy output was 79 %, against 88 % for FP8 re-run against itself under load, so the difference is within the server's own batching non-determinism.

**Functional gates:** 7/9 on the same functional gate suite as FP8 (the two failures are test-harness false negatives shared with FP8: `usage.cached_tokens` is not reported by this build, and a strict string match on a vision answer that is actually correct). Long generation (6.4 k tokens, pt-BR) coherent, tool calling and vision unchanged.

**Speed:** unchanged. Batch-1 prose 100 tok/s, batch-1 code 170 tok/s, 8 concurrent streams 592 tok/s aggregate (idle server), prefill 4.9 k tok/s at 29 k prompt tokens; the dequant adds three small tensor ops per Engram lookup, invisible next to the 40 ms forward.

## Limitations and honesty notes

- 4-bit Lloyd-Max on a Gaussian has an inherent ~9.7 % relative RMSE per row; the tables are already FP8 in the official checkpoint, so this is a second quantization. Use the numbers above, not intuition, to decide whether it fits your use.
- Validated only through SGLang with the row-store adapter on SM120. MLX/GGUF runtimes have their own Engram quantizers (e.g. mixed 4/8-bit with 6-bit Engram); this repo does not target them.
- Video and multi-image inputs were not tested (the base recipe does not support them either).

## Attribution

HLWQ (Hadamard-Lloyd Weight Quantization) is the weight-quantization method by Caio Vicentino ([arXiv:2603.29078](https://arxiv.org/abs/2603.29078)). The rotation-plus-scalar-codebook lineage goes back to **PolarQuant** by Han, Kacham, Karbasi, Mirrokni and Zandieh, 2025 ([arXiv:2502.02617](https://arxiv.org/abs/2502.02617)), a KV-cache method; the name HLWQ replaces the author's earlier "PolarQuant" branding to disambiguate from that work. Base model © DeepSeek, MIT license. Serving recipe by 0xSero. The SM120 bring-up on current `dev-dsv41` builds — the pinned image digest, the anchor drift between the two patches and the `Unsupported sparse-MLA prefill configuration` diagnosis — is [@mmssix](https://huggingface.co/mmssix)'s, reported in [discussion #1](https://huggingface.co/caiovicentino1/DeepSeek-V4.1-Flash-HLWQ-Engram-Q4/discussions/1).
