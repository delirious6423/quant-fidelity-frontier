---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
- quantized
- mxfp8
- compressed-tensors
- auto-round
- vllm
- qwen3_5
- text-generation-inference
---

# Qwen3.8-27B-MXFP8-AutoRound

MXFP8 (OCP Microscaling, 8-bit block FP with E8M0 group scales) quantized checkpoint of
**Qwen3.8-27B** — a dense native vision-language model — produced with
[Intel AutoRound](https://github.com/intel/auto-round) `0.16.0` in **model-free RTN** mode
(no calibration dataset, no model load) and exported in **compressed-tensors** format
(`format: mxfp8-quantized`) for direct loading by vLLM.

The quantization touches only the **text tower Linear layers**. Embeddings, `lm_head`, the
vision tower, the MTP block and the small GDN projections `in_proj_a/in_proj_b` stay in BF16.

> This card documents a locally produced research artifact. Every quantization and evaluation
> number below was measured on the machine described in [Reproducibility](#reproducibility); the
> only figures taken from elsewhere are the BF16 baseline scores of the base checkpoint, which are
> labelled as such where they appear (§3.1).

## 1. Model summary

| | Base model | This checkpoint |
|---|---|---|
| Name | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | Qwen3.8-27B-MXFP8-AutoRound |
| Architecture | `Qwen3_5ForConditionalGeneration` (dense VLM: GDN + gated attention hybrid text tower, vision tower, 1-layer MTP) | identical |
| Parameters | 27.781 B | 27.781 B (unchanged; weights re-encoded) |
| Weight dtype | BF16 (1,199 tensors) | 400 × `F8_E4M3` + 400 × `U8` (E8M0 scales) + 799 × `BF16` = 1,599 tensors |
| Size on disk (safetensors) | 55.56 GB / 51.75 GiB, 18 shards | **32.00 GB / 29.80 GiB**, 18 shards |
| Compression ratio | 1.00× | **1.74×** |
| Effective bits/param | 16.0 | 8.99 (weights only) · **9.21** including E8M0 group scales |
| Quantization | — | W8A8 MXFP8, `group_size=32`, symmetric, dynamic activations |
| Format | safetensors | `compressed-tensors` / `mxfp8-quantized` (vLLM-native) |
| Context length | 262,144 native | unchanged |
| License | Apache-2.0 | Apache-2.0 (quantization adds no new restriction) |

**Not quantized** (266 entries in `quantization_config.ignore`):

| Group | Count | Reason |
|---|---|---|
| `lm_head`, `model.language_model.embed_tokens` | 2 | 1.27 B params each; output layer is the least tolerant to low bits |
| `linear_attn.in_proj_a` / `in_proj_b` | 96 | vLLM fuses them into `in_proj_ba` with `N=48+48=96`; the MXFP8 FlashInfer CuTe-DSL kernel asserts `N >= 128` → engine crashes at `profile_run` if quantized |
| `linear_attn.conv1d` | 48 | 3-D depthwise conv, not a Linear layer |
| `model.visual.*` | 112 | Vision tower kept in BF16; text evals do not exercise it and no MX kernel path exists there |
| `mtp.*` | 8 | Multi-token-propose head, not on the main serving path |

## 2. Precision plan (what got 8 bits)

Measured with `audit_precision_plan.py` (tensor-level, read-only from the BF16 safetensors
headers), which also validates the vLLM MX-kernel constraints (`K % 32 == 0`, `N >= 128`)
*before* the quantization job is launched:

| Bit-width | Modules | Params | Share of Linear |
|---|---|---|---|
| **8 bit (MXFP8)** | `mlp.{gate,up,down}_proj` ×64 · `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` ×48 · `self_attn.{q,k,v,o}_proj` ×16 | 24.327 B | 87.6 % |
| 16 bit (BF16) | embeddings + `lm_head`, vision tower, MTP, `in_proj_a/b` | 3.450 B | 12.4 % |

Quantized layers in the final artifact: **400** = 192 MLP + 144 GDN + 64 full-attention —
matching the audit table exactly.

## 3. Evaluation results

Harness: `lm-eval` 0.4.13 with the `vllm` backend, `seed=42`, `batch_size=64`, single B300 GPU.
Serving args (identical for every run):

```
max_model_len=8192, max_num_batched_tokens=32768, max_num_seqs=128, add_bos_token=True,
gpu_memory_utilization=0.85, dtype=bfloat16, max_gen_toks=2048, enable_prefix_caching=False,
reasoning_parser=qwen3, language_model_only=True, enable_thinking=False
```

| Task | Setting | Metric | Score | stderr | Samples |
|---|---|---|---|---|---|
| gsm8k | 5-shot, `--apply_chat_template --fewshot_as_multiturn` | exact_match (strict **and** flexible) | **0.9719** | ±0.0045 | 1,319 |
| piqa | 0-shot | acc | 0.8085 | ±0.0092 | 1,838 |
| piqa | 0-shot | acc_norm | 0.8156 | ±0.0090 | 1,838 |
| mmlu | 0-shot, 57 subtasks | acc | **0.8348** | ±0.0030 | 14,042 |
| hellaswag | 0-shot | acc | 0.6361 | ±0.0048 | 10,042 |
| hellaswag | 0-shot | acc_norm | 0.8283 | ±0.0038 | 10,042 |

MMLU category aggregates: STEM 0.8297 · Other 0.8574 · Social Sciences 0.9067 · Humanities 0.7762.

Raw files: `lm_eval_results/....../results_2026-09-21T09-10-40*.json`
(gsm8k, 642 s) and `results_2026-09-21T09-34-59*.json` (piqa/mmlu/hellaswag, 965 s).

### 3.1 MXFP8 vs. BF16 baseline

Baseline is the reported BF16 table for [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
on the same four tasks and metric variants (gsm8k strict-match; 0-shot log-likelihood `acc` for
piqa / mmlu / hellaswag):

| | gsm8k (strict) | mmlu | piqa (acc) | hellaswag (acc) | **AVG** |
|---|---|---|---|---|---|
| BF16 `Qwen/Qwen3.8-27B` | 0.9742 | 0.8340 | 0.8096 | 0.6369 | 0.8137 |
| **MXFP8 (this checkpoint)** | 0.9719 | 0.8348 | 0.8085 | 0.6361 | **0.8128** |
| Δ | −0.23 pp | +0.08 pp | −0.11 pp | −0.08 pp | **−0.09 pp** |

The largest single-task regression is 0.23 pp on gsm8k, which sits inside that task's own standard
error (±0.45 pp); MMLU is marginally *above* baseline. **Average loss ≈0.1 pp for 1.74×
compression** — effectively lossless at this bit-width.



## 4. Usage

### 4.1 vLLM (recommended path)

MXFP8 GEMM is served by `FlashInferCutedslMxfp8LinearKernel`; this requires a Blackwell-class GPU
(SM100/SM103) and a vLLM build that (a) registers `model_type=qwen3_5` and (b) carries the MXFP8
CuTe-DSL linear kernel — see the pinned build in §5.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve intel-ai/Qwen3.8-27B-MXFP8-CT-AutoRound \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --language-model-only \
  --reasoning-parser qwen3
```

Observed on 1× B300 (267 GiB usable, `max_model_len=4096`, from the smoke run
`smoke_MXFP8-AutoRound.log`): weights **28.35 GiB** loaded in 24.6 s, peak activation 4.47 GiB,
CUDA-graph pool 0.69 GiB, remaining 193.26 GiB → KV cache of **1,837,738 tokens** (~449×
concurrency at 4,096 tokens/request). The first engine start (profile + cudagraph capture +
FlashInfer autotune) took **504.5 s**; subsequent starts reuse the inductor/FlashInfer caches
under `~/.cache/`.

For image/video inputs drop `--language-model-only`; the vision tower is still BF16, but note that
only the text path was validated for this artifact (§7).

### 4.2 Transformers (loading only, no MXFP8 kernel)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "intel-ai/Qwen3.8-27B-MXFP8-CT-AutoRound"
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, dtype="auto", device_map="auto")
```

Verified with `transformers==5.16.1` + `compressed-tensors==0.17.0`; the only hard requirement is a
`transformers` build that registers `model_type=qwen3_5` in `CONFIG_MAPPING` (older builds raise
`ValueError: The checkpoint you are trying to load has model type qwen3_5 but Transformers does not
recognize this architecture`). Generation via the plain PyTorch dequant path is functionally
correct but far slower than vLLM — use vLLM for throughput work.

Default sampling from `generation_config.json`: `temperature=1.0, top_k=20, top_p=0.95`,
`eos_token_id=[248046, 248044]`.

## 5. Reproducibility

### 5.1 Hardware / OS

| | Value |
|---|---|
| GPU | 4× **NVIDIA B300 SXM6 AC**, 275,040 MiB each, compute capability **10.3** (`sm103`) |
| Driver | **580.159.04** (NVIDIA UNIX open kernel module, x86_64) |
| CUDA | runtime **13.3.33** (`/usr/local/cuda-13.3`); PyTorch built against **CUDA 13.0** |
| cuDNN | system 9.23.0; in-env `nvidia-cudnn-cu12` 9.10.2.21 |
| CPU | 2× Intel Xeon 6776P, 64 cores/socket, 2 threads/core (256 logical), 4 NUMA nodes |
| RAM | 4,031 GB |
| OS / kernel | Ubuntu 24.04.4 LTS (Noble), Linux 6.8.0-124-generic, glibc 2.39, GCC 13.3.0 |

Quantization itself is CPU/disk-bound (model-free streaming rewrite of the shards) and used
**no** calibration data; the evaluation used a single GPU (`CUDA_VISIBLE_DEVICES=3`).

### 5.2 Python packages that matter

```
auto-round            0.16.0        (editable @ auto-round, git 51003909 = v0.14.0-155)
vllm                  0.28.1rc1.dev312+g41848caa6   (editable @ vllm, git 41848caa6)
torch                 2.13.0+cu130
transformers          5.16.1
compressed-tensors    0.17.0
lm_eval               0.4.13
accelerate            1.14.0
datasets              5.0.1
tokenizers            0.23.1
safetensors           0.8.0
huggingface-hub       1.29.0
numpy                 2.2.6
triton                3.7.1
flashinfer-python     0.6.18
flashinfer-cubin      0.6.18
nvidia-cutlass-dsl    4.6.2
```



### 5.3 Environment variables used

```bash
export HF_HOME=/models/huggingface               # dataset/tokenizer cache
export AR_MODEL_FREE_SHARD_PARALLELISM=1         # caps host RAM during the streaming shard rewrite
export CUDA_VISIBLE_DEVICES=2                    # quant wrapper default; check nvidia-smi first
export VLLM_WORKER_MULTIPROC_METHOD=spawn        # required by the vLLM V1 engine here
```

## 6. Reproduce the artifact

### 6.1 Quantize 

```bash
auto-round \
  --model_name Qwen/Qwen3.8-27B \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers visual,lm_head,embed_tokens,mtp,in_proj_a,in_proj_b \
  --format llm_compressor \
  --device_map auto \
  --output_dir ./Qwen3.8-27B-MXFP8-CT-AutoRound
```

Measured run (2026-09-21 08:38–08:40 UTC): **85.36 s** wall, **3.28 GB** peak RAM, 18/18 shards,
400 quantized layers, 266 ignored layers. Two auto-round behaviours are worth knowing:

- `MXFP optimized RTN is enabled` — per group it evaluates the baseline E8M0 scale together with
  2× and 0.5× candidates and keeps the best. Pass `--disable_opt_rtn` for plain RTN.
- Auto-round additionally force-ignores layers model-free RTN cannot handle
  (`embed_tokens`, `linear_attn.conv1d`, `rotary_emb`, `visual.pos_embed`) and logs them as
  `Detected N layer(s) incompatible with model-free RTN` — that warning is expected, not an error.

Do **not** add `mlp.gate` to `--ignore_layers`: this is a dense model, and the substring match
would silently skip all 64 `mlp.gate_proj` (5.7 B params).


### 6.2 Evaluate

```bash

# gsm8k (chat template + multi-turn few-shot) — ~11 min; the lm_eval call for this is
CKPT=intel-ai/Qwen3.8-27B-MXFP8-CT-AutoRound/
ARGS="pretrained=${CKPT},tensor_parallel_size=1,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.85,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,reasoning_parser=qwen3,language_model_only=True,enable_thinking=False"
lm_eval --model vllm --model_args "$ARGS" --tasks gsm8k --batch_size 64 --seed 42 \
        --apply_chat_template --fewshot_as_multiturn \
        --output_path lm_eval_results

# 0-shot suite (piqa, mmlu, hellaswag) — log-likelihood scoring, no chat template. ~16 min on 1× B300:

lm_eval --model vllm --model_args "$ARGS" \
        --tasks piqa,mmlu,hellaswag --batch_size 64 --seed 42 \
        --output_path lm_eval_results
```

Results are written to `lm_eval_results/<sanitized-ckpt-path>/results_<timestamp>.json`. The
`language_model_only=True` / `enable_thinking=False` / `reasoning_parser=qwen3` combination defines
the protocol — keep it identical for any comparison, and remember the 0-shot suite is run **without**
`--apply_chat_template` (log-likelihood scoring), while gsm8k uses the chat template.

To re-derive the BF16 baseline on this stack (a cross-check on the upstream table used in §3.1),
run the same two commands against the base checkpoint directory
(`Qwen/Qwen3.8-27B`) instead of the quantized one. Note the base model needs no
`reasoning_parser`/`language_model_only` change — the args are identical; only `pretrained=` moves.

## 7. Known issues and caveats

- **`in_proj_a` / `in_proj_b` must stay BF16.** The previous artifact
  `qwen3.8-27B-MXFP8-model-free` (same recipe without those two in `--ignore_layers`) is
  self-consistent and passes `check_quant_export.py`, but kills the vLLM engine at start-up with
  `mm_mxfp8 requires N >= 128, got N=96`. It is kept in the workspace only as evidence — do not
  deploy it.
- **Only the text path is validated.** Images/videos load (vision tower is untouched BF16) but no
  multimodal benchmark was run for this artifact.
- **BF16 baseline is the upstream-reported table**, not a re-run on this stack (§3.1). The task
  selection and metric variants match, but the serving build may differ; treat the deltas as
  indicative. Re-running `Qwen3.8-27B` locally (§6.4) closes that gap if you need it exact.
- **`triton_kernels.matmul_ogs` import warning** appears in the vLLM log. It is non-fatal on
  Blackwell — the MXFP8 path resolves to `FlashInferCutedslMxfp8LinearKernel`.
- **First engine start is slow** (~8 min: profile, cudagraph capture, FlashInfer autotune). Warm
  starts are much faster.
- **Tokenizer/preprocessor files are inherited from the base checkpoint** (`crc32.txt` covers only
  those non-weight files: `chat_template.jinja`, `tokenizer*`, `merges.txt`, `vocab.json`,
  `generation_config.json`, `preprocessor_config.json`). The `model-*.safetensors` shards,
  `config.json` and `quantization_config.json` are produced by this quantization run.
- **`quant_lm_head` was not used** and is not recommended here (1.27 B params in the head; MXFP8
  would buy ~0.6 GB and risk the exact failure mode this recipe is designed to avoid).

## 8. License and attribution

The base [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) checkpoint is released under
**Apache-2.0** (see `LICENSE` in this repository); this quantized derivative is offered under the
same license. Quantization was
performed with [Intel AutoRound](https://github.com/intel/auto-round) (Apache-2.0); the artifact
is served through [vLLM](https://github.com/vllm-project/vllm) using FlashInfer / CUTLASS DSL
kernels. The MXFP8 format follows the [OCP Microscaling Formats (MX) specification](https://opencompute.org)
(E8M0 shared scale per 32-element block).

Full design rationale, the tensor-level parameter census, the failed-artifact post-mortem and the
MXFP4 variants of this recipe are in `Qwen3.8-27B-quantization-recipe.md` in the workspace root.
