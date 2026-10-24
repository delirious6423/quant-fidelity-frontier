---
language:
- en
- zh
license: apache-2.0
tags:
- qwen
- qwen4exp
- model
- compressedsensors
- compression
- w4a16
- fp8
- quantized
- moe
- rocm
- rdna3
library_name: vllm
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-Flash-Next
---

# Qwen3.8-Flash-Next W4A16-FP8NG-INT4 (ROCm / RDNA3)

A hand-built, VLLM-0.32-compatible conversion of **Qwen/Qwen3.8-Flash-Next**
(an ~180B-parameter MoE-hybrid, architecture `Qwen4ExpForConditionalGeneration`)
for **AMD Radeon (ROCm, RDNA3 / gfx1100)**. It runs on **4x RX 7900 XTX
(96 GB VRAM)** with tensor-parallel size 4.

This checkpoint is **not** a standard upstream quant. There is no ready-made
compressed-tensors INT4 checkpoint on the Hub for this model, and the official
FP8 / INT4-AutoRound artifacts do not load on the new RDNA3-WNA16 backend, so it
was produced locally by converting the raw bf16 base model through a custom
4-step pipeline (see below).

## What is quantized

| Component | Method | Size |
|---|---|---|
| MoE experts (`gate_up_proj` / `down_proj`) | **int4** group-32 symmetric (CT-W4A16) | ~61.7 GB VRAM |
| ngram / PLE embedding table | **fp8** (`float8_e4m3fn`), pinned CPU-RAM (not VRAM) | ~51 GB RAM total |
| `self_attn.o_proj` + `linear_attn.out_proj` | **int4** W4A16 | small |
| linear / attention / embeddings / lm_head | bf16 (left as-is) | ~8.4 GB VRAM |

> **Why only `o_proj`/`out_proj` are int4 among the dense layers** (empirically
> established via 7 serve runs on gfx1100): on Qwen4Exp, vLLM can load through
> CT-W4A16-Linear **only non-fused** `RowParallelLinear` layers. The fused
> attention projections (`qkv_proj`, `in_proj_qkvz`, `in_proj_ba`),
> `shared_expert` (FusedMoE) and the hyper-connection mixers (`quant_config=None`)
> cannot be quantized through this path on RDNA3.

## Context-length caveat

- fp8 **KV-cache is impossible** for this model: Qwen4Exp QSA attention hard
  requires a **BF16 main KV cache** (`QSA requires a BF16 main KV cache`).
- The KV budget **drifts between restarts** (0.99–1.22 GiB/GPU, depends on
  background load/memory): `131072 -> 93296 -> 73696` tokens.
- This repo's launcher uses a conservative, stable **`--max-model-len 70000`**.
  **Full 131072 is not achievable on 4x 24 GB cards for this model.**

## Quantized file layout

Each quantized `[...].weight` becomes three on-disk tensors (CT-W4A16 for dense
Linear):

```
...weight        bf16  [out, in]     (removed)
...weight_packed int32 [out, in//8]   packed along K, raw nibbles signed+8
...weight_scale  fp16 [out, in/32]    per-group scale
...weight_shape  int64 [in, out]      original full shape
```

MoE experts use the standard fused CT names `w13_weight_packed` /
`w2_weight_packed` (+ `_scale`), matching `RoutedExperts` / FusedMoE loaders.

## File tree

```
model-00001..00131-of-00131.safetensors   # 131 shards (122 GB)
config.json / generation_config.json      # + quantization_config (group_1/group_2)
tokenizer.json / tokenizer_config.json / vocab.json / merges.txt
chat_template.jinja
model.safetensors.index.json
scripts/                                  # the 4-step conversion pipeline (English)
  convert_full.py            # step 1: MoE experts bf16 -> int4 (CT-W4A16)
  convert_ngram_fp8.py       # step 2: ngram/PLE table bf16 -> fp8
  convert_rename_experts.py  # step 3: rename experts -> w13/w2_weight_packed
  convert_linear_int4.py     # step 4: o_proj/out_proj bf16 -> int4 W4A16
  e2e_backend_check.py       # oracle: verify RDNA3-WNA16 backend is selected
  format_control_run.py      # control run of the int4 format on a real MoE layer
  rdna3_kernel_smoke.py      # smoke test of the RDNA3 W4A16 MoE kernel
```

## Conversion pipeline

The order is **mandatory** (each script depends on the previous output). All
scripts run from the vLLM venv with `LD_LIBRARY_PATH` as in the launcher.

```bash
.venv/bin/python convert_full.py            # base bf16 -> W4A16-INT4     (171 GB)
.venv/bin/python convert_ngram_fp8.py       # INT4 -> W4A16-FP8NG         (123 GB)
.venv/bin/python convert_rename_experts.py  # FP8NG -> experts renamed    (in-place)
.venv/bin/python convert_linear_int4.py     # FP8NG -> ...-LinearInt4     (122 GB)
```

Key pitfalls (all hit and fixed during development):

- **`ple_embedding_dtype` must live inside `text_config`** in `config.json` for
  the fp8 ngram path to activate (otherwise the ngram table stays bf16, ~102 GB,
  and allocation OOMs).
- **On-disk expert names must be the layer-parameter names** (`w13_weight_packed`
  / `w2_weight_packed`), not the base names (`gate_up_proj` / `down_proj`) —
  otherwise `AttributeError: no parameter 'w13_weight'`.
- **tmpfs `/tmp`**: the CUDA/Triton JIT writes temporary objects to `/tmp`
  (tmpfs); a full `/tmp` -> `LLVM ERROR: Disk quota exceeded`. Free `/tmp`
  before a run.
- **Chunked quantization**: quantizing the full `[512,1280,2560]` expert in fp32
  at once OOMs 24 GB; `convert_full.py` / `convert_linear_int4.py` chunk it.

## Serve (vLLM 0.32.0+rocm723)

```bash
MODEL=/path/to/this/repo
ROCm flags: TP=4, --gpu-memory-utilization 0.97, --max-num-seqs 1

.venv/bin/vllm serve "$MODEL" \
    --served-model-name flashnext-int4 \
    --host 0.0.0.0 --port 4239 \
    --tensor-parallel-size 4 \
    --max-model-len 70000 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.97 \
    --trust-remote-code \
    --attention-backend ROCM_SEGMENTED_ATTN \
    --engram-config '{"cpu_offload": true, "dp_shared_memory": false}'
```

Required flags:

| Flag | Why |
|---|---|
| `--attention-backend ROCM_SEGMENTED_ATTN` | native attention for gfx1100 (fix #59132) |
| `--engram-config '{"cpu_offload": true, ...}'` | keep the ngram/PLE table in pinned CPU-RAM, not VRAM |
| `--gpu-memory-utilization 0.97` | squeeze max weights+KV out of 96 GB |
| `--max-num-seqs 1` | minimize the KV budget |
| `--trust-remote-code` | custom Qwen4Exp architecture |

Measured throughput (this config, llm-bench, 4x RX 7900 XTX):

```
Short (61 tok): TTFT 3.47s, prefill 17.6 tok/s, gen 64.4 tok/s
Long  (50,235 tok): TTFT 19.94s, prefill 2519.8 tok/s, gen 65.4 tok/s
```

## Limitations

- **Context ceiling ~70K** on 4x 24 GB cards (see "Context-length caveat");
  131K is not reachable for this model on 96 GB VRAM.
- fp8-KV / fp8 weights are **not supported** on RDNA3 by this vLLM build.
- Requires **vLLM 0.32.0+rocm723** and ROCm/RDNA3 (gfx1100); other backends are
  not validated.
- ~122 GB download; ~650+ GB free disk needed to rebuild from base bf16
  (360 GB) through the intermediate checkpoints.

## License

The base model is distributed under the Apache-2.0 license. This repository
inherits that license (see `LICENSE`). Please also review and comply with
Qwen's model-card terms when using this model.