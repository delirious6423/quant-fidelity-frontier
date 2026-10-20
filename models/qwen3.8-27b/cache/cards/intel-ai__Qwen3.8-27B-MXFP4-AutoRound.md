---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
- quantized
- mxfp8
- mxfp4
- mixed-precision
- compressed-tensors
- auto-round
- vllm
- qwen3_5
- text-generation-inference
---

# Qwen3.8-27B-MXFP8Attn-MXFP4MLP-lmheadMXFP4

**Mixed-precision** OCP-Microscaling quantized checkpoint of **Qwen/Qwen3.8-27B** — a dense native
vision-language model — produced with [Intel AutoRound](https://github.com/intel/auto-round)
`0.17.0` in **RTN** mode (`--iters 0`, no gradient tuning) and exported in **compressed-tensors**
format (`format: mixed-precision`) for direct loading by vLLM.

The recipe applies two block-FP precisions at once:

- **MXFP8** (8-bit block FP, E4M3 elements + E8M0 group scale) on the **attention path** — both the
  gated `self_attn.{q,k,v,o}_proj` and the hybrid `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`;
- **MXFP4** (4-bit block FP, E2M1 elements + E8M0 group scale) on the **MLP path** (every text
  `mlp.{gate,up,down}_proj`) **and on `lm_head`**.

Quantizing `lm_head` to MXFP4 is the distinguishing feature of this checkpoint: it brings the
quantized-module coverage in line with the nvidia **NVFP4** reference recipe (which also quantizes
the output projection), closing the last gap between the two. It is essentially lossless on the
text tasks below (gsm8k even ticks up).

> This card documents a locally produced research artifact. Every quantization and evaluation
> number below was measured on the machine described in [Reproducibility](#5-reproducibility). The
> "lm_head=BF16" column in §3 is the sibling checkpoint produced with the *identical* recipe but
> `lm_head` left in BF16 — it is the controlled baseline for isolating the effect of quantizing
> `lm_head`.

## 1. Model summary

| | Base model | This checkpoint |
|---|---|---|
| Name | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | Qwen3.8-27B-MXFP8Attn-MXFP4MLP-lmheadMXFP4 |
| Architecture | `Qwen3_5ForConditionalGeneration` (dense VLM: GDN + gated-attention hybrid text tower, vision tower, 1-layer MTP) | identical |
| Parameters | 27.781 B | 27.781 B (unchanged; weights re-encoded) |
| Weight dtype | BF16 (1,199 tensors) | 208 × `F8_E4M3` (MXFP8) + 193 × `U8` packed FP4 + 401 × `U8` (E8M0 scales) + 783 × `BF16` |
| Size on disk (safetensors) | 55.56 GB / 51.75 GiB | **21.57 GB / 20.09 GiB**, 5 shards |
| Compression ratio | 1.00× | **≈ 2.58×** |
| Quantization | — | Mixed **W8A8 MXFP8** (attention) + **W4A4 MXFP4** (MLP + `lm_head`), `group_size=32`, symmetric, dynamic activations |
| Format | safetensors | `compressed-tensors` / `mixed-precision` (vLLM-native) |
| Context length | 262,144 native | unchanged |
| License | Apache-2.0 | Apache-2.0 (quantization adds no new restriction) |

**Not quantized** (`quantization_config.ignore`, 210 entries + all norms / MTP):

| Group | Reason |
|---|---|
| `model.language_model.embed_tokens` | Input embedding is least tolerant to low bits; kept BF16 |
| `linear_attn.in_proj_a` / `in_proj_b` | vLLM fuses them into `in_proj_ba` with `N=48+48=96`; the MXFP8 FlashInfer CuTe-DSL kernel asserts `N >= 128` → engine crashes at `profile_run` if quantized |
| `linear_attn.conv1d` | 3-D depthwise conv, not a Linear layer |
| `model.visual.*` | Vision tower kept in BF16; text evals do not exercise it and no MX kernel path exists there |
| `mtp.*` (1-layer MTP draft head) | Not in `layer_config`; auto-round saves it separately in BF16 |

Note `lm_head` is **not** in `ignore` here — it is explicitly placed in the MXFP4 group.

## 2. Precision plan (which layers got how many bits)

Quantized Linear layers in the final artifact: **401** (193 MXFP4 + 208 MXFP8).

| Bit-width | Modules | Count |
|---|---|---|
| **4 bit (MXFP4)** | text `mlp.{gate,up,down}_proj` ×64 → 192 · **`lm_head` → 1** | **193** |
| **8 bit (MXFP8)** | `self_attn.{q,k,v,o}_proj` → 64 · `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` → 144 | **208** |
| 16 bit (BF16) | `embed_tokens`, vision tower, `in_proj_a/b`, `conv1d`, MTP head, all norms | ignore + norms |

The config exposes this as two compressed-tensors groups:

```jsonc
"config_groups": {
  "group_0": {                       // MXFP4
    "targets": ["re:.*lm_head", "re:.*language_model.*mlp.*proj.*"],
    "weights":            {"num_bits": 4, "type": "float", "group_size": 32, "symmetric": true, "strategy": "group"},
    "input_activations":  {"num_bits": 4, "type": "float", "dynamic": true}
  },
  "group_1": {                       // MXFP8 (everything else matched by class name)
    "targets": ["Linear"],
    "weights":            {"num_bits": 8, "type": "float", "group_size": 32, "symmetric": true, "strategy": "group"},
    "input_activations":  {"num_bits": 8, "type": "float", "dynamic": true}
  }
}
```

> **Why the `group_0` targets are regexes, not leaf names.** `Qwen3_5ForConditionalGeneration`
> exposes the language model under a prefix (`language_model.*`), so at inference time `lm_head`
> and the MLP projections carry that prefix. Exact leaf-name targets never match those runtime
> paths — the layer is silently skipped and vLLM then fails to find `lm_head.weight_packed`.
> Substring regex targets match regardless of the prefix. The MLP regex
> `re:.*language_model.*mlp.*proj.*` also covers vLLM's fused `gate_up_proj` via the
> compressed-tensors fused-component expansion in `find_matched_target`. See §7.

## 3. Evaluation results

Harness: `lm-eval 0.4.13` with the `vllm` backend (**vLLM nightly `0.31.1rc1.dev292`**, required —
see §7), `seed=42`, `batch_size=64`, single B200 GPU, `language_model_only=True`. Serving args:

```
max_model_len=8192, max_num_batched_tokens=32768, max_num_seqs=128, add_bos_token=True,
gpu_memory_utilization=0.85, dtype=bfloat16, max_gen_toks=2048, enable_prefix_caching=False,
reasoning_parser=qwen3, language_model_only=True, enable_thinking=False, enforce_eager=True
```

`lm_head=BF16` = the sibling checkpoint `Qwen3.8-27B-MXFP8Attn-MXFP4MLP` (same MXFP8-attn /
MXFP4-mlp recipe, `lm_head` left BF16). `lm_head=MXFP4` = **this checkpoint**.

| Task | Setting | Metric | lm_head=BF16 | **lm_head=MXFP4 (this)** | Δ |
|---|---|---|---|---|---|
| mmlu | 0-shot, 57 subtasks | acc | 0.8156 | **0.8115** | −0.0041 |
| hellaswag | 0-shot | acc | 0.6243 | 0.6205 | −0.0038 |
| hellaswag | 0-shot | acc_norm | 0.8186 | 0.8137 | −0.0049 |
| piqa | 0-shot | acc | 0.8020 | 0.8009 | −0.0011 |
| piqa | 0-shot | acc_norm | 0.8101 | 0.8063 | −0.0038 |
| gsm8k | 5-shot, `--apply_chat_template --fewshot_as_multiturn` | exact_match (flexible) | 0.9606 | **0.9629** | +0.0023 |
| gsm8k | 5-shot | exact_match (strict) | 0.9606 | **0.9613** | +0.0007 |

**Every delta sits inside the task standard error (±0.003–0.009); gsm8k even improves slightly.
Quantizing `lm_head` to MXFP4 is effectively lossless** on these text tasks, while aligning the
quantized-module coverage with the NVFP4 reference. A smoke test confirmed the MXFP4 `lm_head`
GEMM actually executes at inference (`fp4_gemm ...(2560, 248320)` in the vLLM log), not a silent
fall-through to BF16.

## 4. Reproducibility

### 4.1 Hardware / software

| | Value |
|---|---|
| GPU | NVIDIA **B200** (183 GB) |
| auto-round | **0.17.0** + two local patches (§4.3) |
| vLLM (eval) | **nightly `0.31.1rc1.dev292`** (V1 engine; MXFP4 `lm_head` support — system 0.22.1 does **not** work, see §7) |
| lm-eval | 0.4.13 |
| Format | compressed-tensors / `mixed-precision` |

### 4.2 Quantize

```bash
auto-round /home/kaokaolv/models/Qwen3.8-27B \
  --iters 0 \
  --scheme MXFP8 \
  --ignore_layers "visual,embed_tokens,linear_attn.in_proj_a,linear_attn.in_proj_b" \
  --layer_config "{language_model.*mlp:{scheme:MXFP4},lm_head:{scheme:MXFP4}}" \
  --format llm_compressor \
  --device auto \
  --output_dir ./Qwen3.8-27B-MXFP8Attn-MXFP4MLP-lmheadMXFP4
```

`--scheme MXFP8` sets the global 8-bit base precision; `--layer_config` overrides the text MLP and
`lm_head` to MXFP4. Relative to the `lm_head=BF16` sibling recipe, the only changes are: drop
`lm_head` from `--ignore_layers`, and add `lm_head:{scheme:MXFP4}` to `--layer_config`.

### 4.3 Required auto-round patches

This checkpoint cannot be produced by stock auto-round `0.17.0`; two fixes are needed (both are
in the companion PR `fix/mxfp4-lm-head-and-mixed-targets`):

1. **Pack outside-block layers in the RTN route** (`compressors/orchestrator.py`). The `--iters 0`
   remaining-layer loop quantized `lm_head` but never called `immediate_pack` + `shard_writer.write`
   (unlike the data-driven route), so `lm_head` serialized as a raw `WrapperWALayer`
   (`lm_head.orig_layer.weight`) with no `weight_packed`/`weight_scale`.
2. **Emit regex targets for mixed-precision override groups** (`export/.../export_to_fp.py`). Exact
   leaf names never match the prefixed runtime paths (`language_model.lm_head`); the fix writes
   `re:` substring targets (see §2 / §7).

### 4.4 Evaluate

```bash
# MUST use the vLLM nightly venv (system vLLM 0.22.1 lacks MXFP4 lm_head support)
CKPT=./Qwen3.8-27B-MXFP8Attn-MXFP4MLP-lmheadMXFP4/Qwen3.8-27B-mxfp-w8g32
ARGS="pretrained=${CKPT},tensor_parallel_size=1,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.85,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,reasoning_parser=qwen3,language_model_only=True,enable_thinking=False,enforce_eager=True"

lm_eval --model vllm --model_args "$ARGS" --tasks piqa,mmlu,hellaswag --batch_size 64 --seed 42 \
        --output_path lm_eval_results_lmhead_mxfp4
lm_eval --model vllm --model_args "$ARGS" --tasks gsm8k --batch_size 64 --seed 42 \
        --apply_chat_template --fewshot_as_multiturn --output_path lm_eval_results_lmhead_mxfp4
```

## 5. Usage

```bash
# Blackwell-class GPU + a vLLM nightly carrying the MXFP4 lm_head path
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_COMPILE_CACHE=1   # after changing scheme, avoid stale inductor graphs
vllm serve ./Qwen3.8-27B-MXFP8Attn-MXFP4MLP-lmheadMXFP4/Qwen3.8-27B-mxfp-w8g32 \
  --tensor-parallel-size 1 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --language-model-only --reasoning-parser qwen3
```

```python
from vllm import LLM, SamplingParams
llm = LLM(model="./Qwen3.8-27B-MXFP8Attn-MXFP4MLP-lmheadMXFP4/Qwen3.8-27B-mxfp-w8g32",
          max_model_len=8192, gpu_memory_utilization=0.85)
print(llm.generate("The capital of France is", SamplingParams(max_tokens=16))[0].outputs[0].text)
```

## 6. Known issues and caveats

- **Requires a vLLM nightly with MXFP4 `lm_head` support.** System vLLM 0.22.1 cannot load this
  checkpoint (no MXFP4 output-projection path). Validated on nightly `0.31.1rc1.dev292`.
- **First engine start is slow.** The first `fp4_gemm` CuTe-DSL kernel JIT-compiles (~10 min,
  cached under `~/.cache/flashinfer`); a `No tuned config covers fp4_gemm` line is a perf hint, not
  an error.
- **`group_0` targets must be regexes.** If exported as exact leaf names, vLLM silently leaves
  `lm_head` / MLP unquantized and then fails with
  `no module named 'lm_head.weight_packed' ... available: {'lm_head.weight'}`. This card's config
  uses prefix-robust regexes (`re:.*lm_head`, `re:.*language_model.*mlp.*proj.*`).
- **`in_proj_a` / `in_proj_b` must stay BF16** — fused `N=96 < 128` crashes the MXFP8 kernel at
  startup.
- **Text scores only.** Images/videos load (vision tower is untouched BF16) but no multimodal
  benchmark was run; the MTP head (BF16) is not exercised under `language_model_only=True`.
- **BF16 context for §3** is the `lm_head=BF16` sibling, not the upstream base checkpoint; the
  deltas isolate the `lm_head` MXFP4 effect on the same mixed-precision stack.

## 7. License and attribution

The base [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) checkpoint is Apache-2.0;
this quantized derivative is offered under the same license. Quantization was performed with
[Intel AutoRound](https://github.com/intel/auto-round) (Apache-2.0); the artifact is served through
[vLLM](https://github.com/vllm-project/vllm) using FlashInfer / CUTLASS DSL kernels. The MXFP8 /
MXFP4 formats follow the [OCP Microscaling Formats (MX) specification](https://opencompute.org)
(E8M0 shared scale per 32-element block; E4M3 elements for MXFP8, E2M1 for MXFP4).
