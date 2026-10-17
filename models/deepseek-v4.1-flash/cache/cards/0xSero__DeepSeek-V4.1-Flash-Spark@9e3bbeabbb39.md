---
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
license: mit
library_name: vllm
pipeline_tag: image-text-to-text
language:
- en
- zh
tags:
- deepseek-v4.1
- exl3
- exllamav3
- trellis
- mixed-precision
- mixture-of-experts
- dgx-spark
- vision
- long-context
---

# DeepSeek-V4.1-Flash-Spark

DeepSeek-V4.1-Flash with every backbone routed expert re-encoded as **EXL3 trellis** (MUL1 codebook) at a
per-expert K2 / K3 / K5 mix, sized to serve on **two NVIDIA DGX Spark** (GB10, 2 x 128 GB unified memory,
tensor parallel 2). Average routed-expert rate **2.77 bpw**. Everything else — attention, shared experts,
routers, Engram tables, the DSpark/MTP draft layers and their experts, the vision tower, embeddings and
head — is the upstream checkpoint, byte for byte, in its native FP8 / BF16 / MXFP4 formats.

> [!IMPORTANT]
> EXL3 experts need the EXL3 MoE path in the companion vLLM image (below). Stock vLLM and plain
> `transformers` cannot run this checkpoint. Two Engram shards are stored as split parts (HF limits files
> to 50 GB); run `./reassemble.sh` once after download.

## At a glance

| | |
|---|---|
| Base | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) @ `fb2764a5cf321eaa5070ca8f9e892818f477c16d` |
| Architecture | 40 layers, 384 routed experts (top-6) + 1 shared, hidden 5120; 3 DSpark/MTP draft layers; Engram on layers 1 and 14; 32-layer vision tower |
| Routed experts (layers 0-39) | EXL3 trellis, MUL1 codebook: **K2 7,279 · K3 6,299 · K5 1,782** of 15,360 experts; **2.77 bpw** average |
| Everything else | upstream tensors, byte-identical (FP8 e4m3 / ue8m0 scales, BF16, MXFP4 for the draft experts) |
| Engram tables | native FP8, 2 x 98.3 GB, read from disk at serve time |
| Size | native part 221.5 GB + EXL3 banks 240.4 GB = **~462 GB** |
| Context | 1,048,576 native; served at **262,144** |
| Vision / tools / reasoning | yes / yes / yes (smoke-tested on the served stack) |
| License | MIT (DeepSeek), carried from the base |

## Quality

Offline full-vocabulary token-wise KL divergence against the native checkpoint (teacher), 64 windows per
panel, bootstrap 95 % CI. Lower KLD and higher top-1 agreement are better.

| Panel | Mean KLD (nats) [95 % CI] | Top-1 agreement |
|---|---:|---:|
| v3.1 panel | **0.0742** [0.0644, 0.0865] | **0.917** |
| legacy panel | **0.0449** | **0.926** |

Paired against the previous allocation at 2.68 bpw (same K2/K3 banks, 1,782 experts left as native
MXFP4), this build moves −1.5 % on v3.1 and +0.8 % on legacy KLD, both inside the noise band; it was
accepted because every routed expert now uses one kernel family.

## Serving on 2 x DGX Spark

Historical measurements on host-built image `sha256:5668e35e5ee021da4b9ce88a1964caf0e082e8ca01d81d1d64b6213a55c6add5`
and the full upstream checkpoint layout, TP2 over the Sparks' RoCE link, FP8 KV cache, DSpark speculative
decoding (7 draft tokens), Engram tables on disk, CUDA graphs `FULL_DECODE_ONLY`, 2 concurrent sequences:

| | |
|---|---|
| KV cache | **2,023,717 tokens** |
| Max context | 262,144 |
| Prefill | **2,050 tok/s** at 8k · **2,061 tok/s** at 32k |
| Code decode, 1 stream | **~39-41 tok/s** |
| Code decode, 2 streams | **60 tok/s** aggregate |
| Prose decode, 1 stream | **~29 tok/s** |
| Load time | ~11-13 min (EXL3 banks ~15 s per layer) |

All decode runs used the model's default sampling and stopped naturally (3k-15k tokens); no output caps.
Smoke checks passed for text, tool calls and vision.

## Published image and validation status

The public CI image is
`ghcr.io/0xsero/deepseek-v4.1-flash-spark@sha256:3cbc8ec016f5fbfc82eba3480de12399cbce31e7b76aa3108b8fe8246c42a79d`.
[GitHub-hosted ARM64 build 37138684406](https://github.com/0xSero/local-ai-images/actions/runs/37138684406)
published it from source `93e001cf4ecafeb71c6e737eeefad9a9b4ce9235` on `main`, with verified signed provenance.
Fresh model acceptance on this exact image and this repository's compact layout is still pending in
[registry PR #154](https://github.com/0xSero/local-ai-registry/pull/154). The historical measurements above
have not been transferred to this image or layout.

## Run it

1. Download and rebuild the two Engram shards:

```bash
hf download 0xSero/DeepSeek-V4.1-Flash-Spark --revision 08ac8b3defc9a239ba0baf51059687c689406519 --local-dir ~/models/DeepSeek-V4.1-Flash-Spark
cd ~/models/DeepSeek-V4.1-Flash-Spark && ./reassemble.sh     # cat parts, sha256 check
```

Allow about 665 GB per node during download and reassembly, plus image and cache space.

2. The [two-Spark launcher](https://github.com/0xSero/DeepSeek-V4.1-Flash-Two-Sparks)
   mounts the repository root at `/model` and writes a separate plan copy whose bank paths are absolute
   `/model/exl3/...` paths. It mounts that copy at `/plans` and keeps the checkpoint read-only.
   The pending registry recipe instead uses the original relative plan with workdir `/model/exl3`.

3. To exercise the CI image, set `IMG` to the exact public digest above when running `scripts/launch.sh`
   from the launcher repository; also set `WORKER` to your second Spark's SSH target as its instructions
   describe. The launcher's default still points to the historical host image. This is a candidate
   configuration pending the acceptance run linked above. Key flags:

```bash
vllm serve /model --tensor-parallel-size 2 --nnodes 2 --dtype bfloat16 \
  --kv-cache-dtype fp8 --block-size 256 --swa-block-size 128 --kv-cache-memory-bytes 2700000000 \
  --max-model-len 262144 --max-num-seqs 2 --max-num-batched-tokens 2048 \
  --attention-backend B12X --linear-backend b12x --moe-backend b12x \
  --engram-config '{"table_memory":"disk","cpu_offload":false}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_tensor_parallel_size":2}' \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice
# launcher env: ST_EXL3_PLAN=/plans/J268-ho-v31-K5n.json (absolute qdir), ST_EXL3_PREFILL=st
```

The validated runs mounted the full upstream checkpoint as `/model`; this repo carries the same tensors
minus the backbone routed experts, which the EXL3 path never reads.

## Files

| Path | What it is |
|---|---|
| `model-00001..00004-of-00006.safetensors` | upstream non-expert tensors, byte copies (18.4 GB) |
| `model-00005/00006-of-00006.safetensors.part00..05` | Engram shards (layers 1 and 14) split into ≤17 GB parts |
| `model.safetensors.index.json` | weight map for the six reassembled shards |
| `reassemble.sh`, `sha256-manifest.txt` | Engram rebuild + whole-file hashes of the native shards |
| `exl3/qn/K5/L??.part.*` | K5 expert banks, layers 0-39 |
| `exl3/q31/K{2,3}/L??.part.*` | K2 / K3 expert banks, layers 0-11 |
| `exl3/q31s/K{2,3}/L??.part.*` | K2 / K3 expert banks, layers 12-39 |
| `exl3/plan/*.json` | per-layer, per-expert K allocation (`qdir` relative to `exl3/`) |
| `config.json`, `tokenizer*.json`, `encoding/`, `inference/`, `LICENSE` | upstream, unchanged |
| `repack-report.json` | repack verification receipt |

## Verification

- Dropped exactly the 92,160 backbone routed-expert tensors (`layers.{0-39}.ffn.experts.{0-383}.w{1,2,3}.{weight,scale}`
  = 40 x 384 x 6). Nothing else was dropped; the `mtp.*` draft experts are kept.
- Every kept tensor's bytes were hashed at read time and re-hashed from the written shard: all 3,913
  non-Engram tensors match, dtypes and shapes match. Engram shards are verified by whole-file sha256
  against the source. Receipt: `repack-report.json`.

## Provenance

| | |
|---|---|
| Source | `deepseek-ai/DeepSeek-V4.1-Flash` rev `fb2764a5cf321eaa5070ca8f9e892818f477c16d` |
| Allocation plan | `J268-ho-v31-K5n` (night-5 deploy): K2/K3 banks from the v3.1 calibration run, the remaining 1,782 experts re-encoded at K5 |
| Encoder / kernels | exllamav3 1.5.3 `d3739fd393337b1ff4d6c2a342b12f0c87a9592f` |
| Serving runtime | local-inference-lab vLLM r38 `66c293578412417476f842c1da5805d3a3d959a8` + EXL3 MoE method |
| Pipeline code | `a9ea365bbb8a3b5d4eaaf7097c3f3995e217f85e` |

## License and credits

MIT, carried from DeepSeek's release; this repo adds no terms. Thanks to DeepSeek for the model,
[turboderp](https://github.com/turboderp-org/exllamav3) for ExLlamaV3 / EXL3 (MIT) — the trellis
encoder and kernels used here — and Local Inference Lab and the vLLM community for the Spark serving stack.
