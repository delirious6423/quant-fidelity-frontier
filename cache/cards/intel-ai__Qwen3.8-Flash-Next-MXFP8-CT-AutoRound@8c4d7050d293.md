---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- quantized
- mxfp8
- compressed-tensors
- auto-round
- vllm
- qwen4_exp
- moe
---

# Qwen3.8-Flash-Next-MXFP8-AutoRound

MXFP8 (OCP Microscaling, 8-bit block FP with E8M0 group scales) quantized checkpoint of
**Qwen3.8-Flash-Next** — a 180 B-parameter sparse-MoE vision-language model — produced with
[Intel AutoRound](https://github.com/intel/auto-round) `0.16.0` in **model-free RTN** mode
(no calibration dataset, no model load), exported as **compressed-tensors** (`format:
mxfp8-quantized`) for direct loading by vLLM.

All quantizable Linear layers of the text tower — **including all 73 728 routed expert
matrices** — are MXFP8 W8A8. Embeddings, `lm_head`, the vision tower, the whole MTP block, the PLE
n-gram table, hyper-connection modules, the MoE routers and the small GDN projections
`in_proj_a/in_proj_b` stay in BF16.

> **Read this first: two things about this checkpoint are unusual and both are structural, not
> configuration mistakes.** (1) The compression ratio is capped near **1.5×**, because a 51.2 B
> parameter *n-gram embedding table* (102.4 GB, 28.4 % of all params) is a lookup table, not a
> Linear layer — MX formats cannot touch it. (2) Serving it requires **TP ≥ 2 plus
> `moe_backend=marlin`** on this vLLM build, for two independent measured reasons in §7.

Every quantization and evaluation number below was measured on the machine in
[Reproducibility](#5-reproducibility); the BF16 row in §3.3 is the reference table supplied for the
base checkpoint rather than a local run. Headline result: **AVG −0.26 pp, but MMLU drops 1.37 pp
(−4.8σ)** — see §3.3 before treating this as a lossless quant.

## 1. Model summary

| | Base model | This checkpoint |
|---|---|---|
| Name | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Qwen3.8-Flash-Next-MXFP8-CT-AutoRound |
| Architecture | `Qwen4ExpForConditionalGeneration` / `model_type: qwen4_exp` (VL, hybrid GDN + Qwen Sparse Attention, 512-expert MoE, PLE n-gram layer, hyper-connections, 1-layer MTP) | identical |
| Parameters | 180.000 B total ≈ 125 B MoE + 51.2 B n-gram table + 4 B MTP; ~6.7 B active per token | 180.000 B (unchanged; weights re-encoded) |
| Layers | 48 = 36 × linear_attention (Gated DeltaNet) + 12 × full_attention (`full_attention_interval=4`) | identical |
| MoE | 512 routed experts, 10 active + 1 shared, `moe_intermediate_size=640` | identical |
| Weight tensors (index entries) | 1 658, all BF16 — experts stored as **fused 3-D** `[512, N, K]` | 150 720 = 73 896 × `F8_E4M3` + 73 896 × `U8` (E8M0 scales) + 2 925 × `BF16` + 3 × `I64` |
| Size on disk (safetensors) | 360.00 GB / 335.28 GiB, 131 shards | **240.39 GB / 223.88 GiB**, 131 shards |
| Compression ratio | 1.00× | **1.50×** (capped by the n-gram table, see §2) |
| Effective bits/param | 16.0 | 10.51 (weights only) · **10.68** on disk incl. E8M0 scales |
| Quantization | — | W8A8 MXFP8, `group_size=32`, symmetric, dynamic activations |
| Format | safetensors | `compressed-tensors` / `mxfp8-quantized` (vLLM-native) |
| Context length | 262,144 native (extensible to 1 M) | unchanged |
| License | Qwen Community License 1.0 | same (see §8) |

The tensor-count blow-up (1 658 → 150 720) is expected, not duplication: auto-round's
`split_fused_expert_tensors` expands each fused `[512, N, K]` expert block into individually named
`mlp.experts.<i>.{gate,up,down}_proj` 2-D layers (and splits `gate_up_proj` along the output dim),
so the export carries one weight + one scale tensor per expert matrix. That is also why
`quantization_config.config_groups.group_0.targets` is the literal count **1** here — targets are the
generic `Linear`, with the 73 896 quantized layer names implied by it and the exceptions listed in
`ignore`.

**Minimum deployment: 2 GPUs.** With 1 GPU the engine runs out of memory regardless of batch size,
because the n-gram table needs a second full 95.37 GiB copy when it cannot be sharded (§7).

## 2. Precision plan

Measured with `audit_precision_plan.py` (tensor-level, reads only `config.json` + safetensors
headers) and cross-checked against the exported artifact. Of the 76 297 2-D Linear layers in the
model, **73 896 are MXFP8** and 2 401 stay BF16:

| Bit-width | Modules (count) | Params | Share of Linear |
|---|---|---|---|
| **8 bit (MXFP8)** | `mlp.experts.{gate,up,down}_proj` 48 L × 512 E = 73 728 · `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` 36×3 = 108 · `self_attn.{q,k,v,o}_proj` 12×4 = 48 · `self_attn.indexer.index_qk_proj` 12 | 123.489 B | 68.6 % |
| 16 bit (BF16) | `embed_tokens`, `lm_head`, vision tower, all of `mtp.*`, PLE + n-gram table, hyper-connection, routers, `in_proj_a/b` | 56.506 B | 31.4 % |

The 8-bit column is the audit's predicted 123.725 B (74 040 layers) minus the 0.236 B of
shared-expert layers that this auto-round build moved into `ignore` at export time — see §3.4, which
also gives the as-exported layer counts. Layer counts in the table are the as-exported (measured)
ones: 73 896 tensors carry a `weight_scale`.

Kept in BF16 — all 2 439 `ignore` entries, grouped:

| Group | Entries | Why it must stay in BF16 |
|---|---|---|
| `mtp.*` routed experts | 1 536 | Speculative-decode draft module; not on the main serving path |
| hyper-connection (`input_mix_weight_{down,up}`, `block_inject_weight`, top-level mixer) | 290 | Engine-side structural residual mixing; `block_inject_weight` is `[4, 10240]` (N=4, no MX kernel can take it) |
| `mlp.shared_expert.{gate,up,down}_proj` | 144 | Kept BF16 by this auto-round build at export time — see the deviation note in §3.4 |
| PLE n-gram embedding shards | 128 | 51.2 B / 102.4 GB lookup table (`[2 500 012, 160]` × 128 shards), not a Linear layer |
| `model.visual.*` | 112 | Text evals do not exercise it; no MX kernel path for the tower |
| `linear_attn.in_proj_a` / `in_proj_b` | 72 | vLLM fuses them into `in_proj_ba` with `N=48+48=96`; `mm_mxfp8` asserts `N >= 128` → engine dies at `profile_run` (measured on the 27B sibling) |
| `mlp.gate` (MoE router) | 48 | 512-way softmax routing; quantizing it changes the routing distribution directly |
| `mlp.shared_expert_gate` | 48 | `[1, 2560]`, N=1 — unquantizable |
| `linear_attn.conv1d` | 36 | 3-D depthwise conv, not a Linear layer |
| `mtp.*` non-expert (attention, MLP, fc, indexer) | 20 | Same as the MTP experts above |
| PLE projections / `conv1d` | 3 | Lookup-adjacent, kept with the PLE block |
| `lm_head`, `embed_tokens` | 2 | 0.636 B each; output layer is the least bit-tolerant |

**Why the ratio is 1.5× and not 2×**: 95.6 % of the parameters are routed experts (67.1 %) plus the
n-gram table (28.4 %). Only ~123.5 B params are quantizable, and the table's 102.4 GB is immovable.
Even the more aggressive MXFP4 recipe can only reach ≈180 GB. If your deployment target is memory,
the n-gram table — not the weight precision — is the thing to plan around.

## 3. Evaluation results

Harness: `lm-eval` 0.4.13, `vllm` backend, **TP=2 on 2× B300**, `seed=42`, `batch_size=32`. Full
model args (identical for every run; this is exactly the JSON passed to `--model_args`):

```json
{"pretrained":"./Qwen3.8-Flash-Next-MXFP8-CT-AutoRound","tensor_parallel_size":2,
 "max_model_len":8192,"max_num_batched_tokens":16384,"max_num_seqs":64,"add_bos_token":true,
 "gpu_memory_utilization":0.85,"dtype":"bfloat16","max_gen_toks":2048,"enable_prefix_caching":false,
 "language_model_only":true,"trust_remote_code":true,"enable_thinking":false,
 "reasoning_parser":"qwen3","kernel_config":{"moe_backend":"marlin"}}
```

### 3.1 Scores

| Task | Setting | Metric | Score | stderr | Samples |
|---|---|---|---|---|---|
| gsm8k | 5-shot, chat template, `--fewshot_as_multiturn` | exact_match (**strict**) | **0.9704** | ±0.0047 | 1,319 |
| gsm8k | same | exact_match (flexible) | 0.9712 | ±0.0046 | 1,319 |
| piqa | 0-shot | acc | 0.8210 | ±0.0089 | 1,838 |
| piqa | 0-shot | acc_norm | 0.8313 | ±0.0087 | 1,838 |
| mmlu | 0-shot, 57 subjects | acc | **0.8514** | ±0.0029 | 14,042 |
| hellaswag | 0-shot | acc | 0.6913 | ±0.0046 | 10,042 |
| hellaswag | 0-shot | acc_norm | 0.8771 | ±0.0033 | 10,042 |
| **AVG** (gsm8k strict, mmlu, piqa acc, hellaswag acc) | | | **0.8335** | | |

MMLU category aggregates: STEM 0.8570 · Other 0.8761 · Social Sciences 0.9093 · Humanities 0.7936.

Wall time: gsm8k 1,265 s; piqa+mmlu+hellaswag 2,525 s. Raw files:
`lm_eval_results/....../results_2026-09-21T11-08-08*.json`
and `…T11-51-11*.json`.

### 3.2 Functional smoke test (passed)

`smoke_vllm_ckpt.py`, TP=2, `max_model_len=4096`, 3 deterministic prompts → `SMOKE OK`:

```
'The capital of France is' -> ' Paris. The capital of Germany is Berlin. The capital of Italy is Rome…'
'1+1=2. 2+2=4. 4+4='       -> '8. 8+8=16. 16+16=32. 32+32=64. 64+64=128…'
'def quicksort(arr):'      -> runnable recursive quicksort implementation
```

Engine accounting from that run (`smoke_fn_mxfp8_marlin.log`): weights **111.49 GiB per rank**
(223 GiB total across TP=2), load 252.7 s / 592.7 s per rank, `Using 'MARLIN' MxFp8 MoE backend`,
`Using FlashInferCutedslMxfp8LinearKernel for MXFP8 GEMM`, initial profiling/warmup 8.94 s, then
**109.69 GiB free → KV cache of 2,945,536 tokens**.

### 3.3 vs. the BF16 base model

BF16 reference for `Qwen/Qwen3.8-Flash-Next` on the same four tasks and metric variants
(gsm8k strict-match; 0-shot log-likelihood `acc` for piqa / mmlu / hellaswag):

| | gsm8k (strict) | mmlu | piqa (acc) | hellaswag (acc) | **AVG** |
|---|---|---|---|---|---|
| BF16 `Qwen/Qwen3.8-Flash-Next` | 0.9674 | 0.8652 | 0.8194 | 0.6927 | 0.8362 |
| **MXFP8 (this checkpoint)** | 0.9704 | 0.8514 | 0.8210 | 0.6913 | **0.8335** |
| Δ | **+0.30 pp** | **−1.37 pp** | +0.16 pp | −0.14 pp | **−0.26 pp** |
| Δ in units of MXFP8 stderr | +0.6σ (±0.47) | **−4.8σ (±0.29)** | +0.2σ (±0.89) | −0.3σ (±0.46) | — |

Reading it honestly:

- **gsm8k, piqa, hellaswag are unchanged.** All three deltas are well inside one standard error, and
  two of them are positive. gsm8k even gains 0.30 pp, which is noise on a ±0.47 pp metric.
- **MMLU is a real regression, not noise: −1.37 pp at −4.8σ.** Treating the BF16 run as carrying a
  comparable error bar gives a combined ±0.41 pp, so it is still ≈3.3σ — it survives the more
  conservative reading. Only the MMLU aggregate was supplied for BF16, so the drop cannot be
  attributed to specific subjects; the MXFP8 category split is in §3.1 for reference. **Do not
  describe this checkpoint as lossless** — the honest summary is *≈0.3 pp average loss, with a
  ~1.4 pp knowledge-benchmark cost*.
- **The cause is not isolated by these runs.** The regression appeared on the variant where all
  123.5 B routed-expert params go through W8A8. The MXFP4-Mixed variant (M0 — experts at W4A4, rest
  MXFP8) has **not** been evaluated here, and it is not obvious which way it moves: fewer bits per
  expert could deepen the drop, while a different scale-quantization path could behave differently.
  The experiment that would actually localise this is one that keeps the routed experts in BF16 and
  MXFP8's everything else — no run in this workspace isolates that.

Caveat on the comparison: the BF16 numbers are the reference table supplied for the base checkpoint,
not a run produced in this workspace, so no stderr was recorded for them. The MXFP8 side is fully
reproducible from §6.4. To close the gap with matched error bars, run the §6.4 commands against
`Qwen/Qwen3.8-Flash-Next` under `TP=4 CUDA_VISIBLE_DEVICES=0,1,2,3`.

### 3.4 Deviation from the predicted allocation (measured)

The audit predicted 74 040 quantized layers; the artifact has **73 896** — a difference of exactly
**144 = `mlp.shared_expert.{gate,up,down}_proj` × 48**. Auto-round's matcher assigns those layers
`bits=8`, but the model-free export writes them into `ignore` instead (147 `shared_expert.` entries,
zero `weight_scale` tensors). Cost is negligible (≈0.24 B params left in BF16, +0.12 GB) and it is
*safer* for vLLM, but it means this artifact is **not** byte-isomorphic with the
`INCModel3/Qwen3.8-Flash-Next-MXFP4-Mixed-CT-AutoRound` reference, whose shared expert is MXFP8.

## 4. Usage

### 4.1 vLLM (the path that was validated)

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve intel-ai/Qwen3.8-Flash-Next-MXFP8-CT-AutoRound \
  --tensor-parallel-size 2 \
  --moe-backend marlin \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

`--moe-backend marlin` is **required on this build at TP=2/TP=4**: with vLLM's automatic backend
selection (FlashInfer TRTLLM) the MXFP8 MoE weight-scale swizzle crashes at load — see §7. Equivalent
form for Python `LLM(...)` / lm_eval: `kernel_config={"moe_backend": "marlin"}`.

Hard constraints for `qwen4_exp` (violating any of these fails at start-up):

- `PP=1` only (PLE needs the raw `input_ids`); `TP` must divide both `linear_num_value_heads=48` and
  `num_attention_heads=24`.
- **No KV-cache quantization** (`Qwen4Exp QSA does not support KV quantization`), so the
  `-FP8KV` / `-FP8Attn` variants used on other models in this family will not start here.
- **No expert parallelism** (`CutlassExpertsMxfp4`/Cutlass MoE require `ep_size == 1`, no clean fallback).
- No speculative decoding during evaluation: the compressed-tensors `ignore` list is not remapped
  onto the MTP draft model.
- Do **not** set `VLLM_GDN_DECODE_KERNEL=cuda`; let vLLM fall back from CUDA to Triton by itself.
- Budget VRAM for the n-gram table on top of the weights: 95.37 GiB total (51.2 B × 2 B), sharded
  along the vocab dim, so ≈47.7 GiB/rank at TP=2 and ≈23.8 GiB/rank at TP=4. Measured at TP=2:
  weights 111.49 GiB/rank + n-gram share, leaving 109.69 GiB for KV cache at
  `gpu_memory_utilization=0.85`.

### 4.2 Transformers (weights-only inspection, not a fast path)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "intel-ai/Qwen3.8-Flash-Next-MXFP8-CT-AutoRound"
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt, dtype="auto", device_map="auto",
                                             trust_remote_code=True)
```

Verified with `transformers==5.16.1` + `compressed-tensors==0.17.0`; the only real requirement is a
`transformers` build that registers `model_type=qwen4_exp`. Note the `ARCHITECTURES`-level caveat:
older transformers releases that do not know `qwen4_exp` raise
`ValueError: ... model type qwen4_exp but Transformers does not recognize this architecture`.
Default sampling (`generation_config.json`): `temperature=1.0, top_k=20, top_p=0.95`,
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
| RAM | 4,031 GB (quantization peak was 4.51 GB) |
| OS / kernel | Ubuntu 24.04.4 LTS (Noble), Linux 6.8.0-124-generic, glibc 2.39, GCC 13.3.0 |
| Storage | NFS mount, 10 TB volume (2.0 TB free at run time) — 336 GB source + 224 GB output |

Quantization is CPU/disk-bound and used **no** calibration data and effectively no GPU. Evaluation
used 2 GPUs; the smoke run used 2.

### 5.2 Python packages

```
auto-round            0.16.0        (editable @ auto-round, git 51003909 = v0.14.0-155)
vllm                  0.28.1rc1.dev312+g41848caa6.d20260917
                                   (editable @ muse_support/vllm, git 41848caa6, + 2 local diffs below)
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

**Local diffs present in the vLLM tree** (neither is on this artifact's MXFP8 path, but they change
what a "clean checkout at `41848caa6`" would do):

- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py`
  (+14/−2): swizzled MXFP4 expert scales are reshaped to the **padded** dims. Affects the
  MXFP4-Mixed variant, not this checkpoint.
- `vllm/model_executor/layers/fused_moe/routed_experts.py` (+9/−1): chunk fused `w13` on the output
  dim for transposed checkpoints.
- `.../quantization/utils/flashinfer_utils.py` — **unpatched on purpose**; this is the file holding
  the MXFP8 MoE swizzle bug that §7 works around with `--moe-backend marlin`.



### 5.3 Environment variables

```bash
export HF_HOME=/models/huggingface               # dataset + tokenizer cache (bench sets pre-cached)
export AR_MODEL_FREE_SHARD_PARALLELISM=1         # caps host RAM during the 360 GB streaming rewrite
export CUDA_VISIBLE_DEVICES=1,2                  # quant wrapper uses :2; eval defaults to 1,2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## 6. Reproduce the artifact

### 6.1 Quantize


```bash
auto-round \
  --model_name Qwen/Qwen3.8-Flash-Next \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,ple,mtp,hyper_connection \
  --format llm_compressor \
  --device_map auto \
  --output_dir ./Qwen3.8-Flash-Next-MXFP8-CT-AutoRound
```

Measured (2026-09-21 09:23:56 → 09:38:42 UTC): **885.71 s** wall, **4.51 GB** peak RAM, 131/131
shards, 73 896 quantized layers, 2 439 ignored, no `falling back` line. Two auto-round behaviours
to expect in the log:

- `MXFP optimized RTN is enabled` — per 32-element group it evaluates the baseline E8M0 scale plus
  2× and 0.5× candidates and keeps the best; `--disable_opt_rtn` for plain RTN.
- `Detected 5 layer(s) incompatible with model-free RTN: … embed_tokens, ple.conv1d,
  ple.ple_embedding.ngram_embedding, linear_attn.conv1d, rotary_emb, visual.pos_embed …` — these are
  force-ignored automatically. Expected, not an error.

Checklist-specific warnings that do **not** apply here but are worth knowing before you reuse a
list from the dense sibling model:

- This **is** a MoE, so `mlp.gate` **must** be in `--ignore_layers` (it is the router). On the dense
  Qwen3.8-27B the same string would silently skip every `mlp.gate_proj` — the two lists are not
  interchangeable.
- `in_proj_a` and `in_proj_b` must be ignored **together**: vLLM loads them as one fused
  `in_proj_ba`, and mixed schemes across the shards are rejected.

### 6.3 Validate

Common environment:

```bash
export HF_HOME=/models/huggingface
export CUDA_VISIBLE_DEVICES=1,2                    # 2 GPUs = TP=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CKPT=intel-ai/Qwen3.8-Flash-Next-MXFP8-CT-AutoRound
OUTDIR=./lm_eval_results
```

`--model_args` **must be passed as one JSON object** on this model: this lm_eval build tries
`json.loads` on the first token of the argument and refuses to fall back when the string contains
`{`, so a comma-separated `k=v` list with the nested `kernel_config` dict dies with
`Invalid JSON` (§7).

gsm8k (5-shot, chat template, multi-turn few-shot) — 1,265 s:

```bash
lm_eval --model vllm --model_args '{
  "pretrained": "'"$CKPT"'",
  "tensor_parallel_size": 2,
  "max_model_len": 8192,
  "max_num_batched_tokens": 16384,
  "max_num_seqs": 64,
  "add_bos_token": true,
  "gpu_memory_utilization": 0.85,
  "dtype": "bfloat16",
  "max_gen_toks": 2048,
  "enable_prefix_caching": false,
  "language_model_only": true,
  "trust_remote_code": true,
  "enable_thinking": false,
  "reasoning_parser": "qwen3",
  "kernel_config": {"moe_backend": "marlin"}
}' --tasks gsm8k --batch_size 32 --seed 42 \
   --apply_chat_template --fewshot_as_multiturn --output_path "$OUTDIR"
```

0-shot suite (piqa, mmlu, hellaswag), log-likelihood scoring, no chat template — 2,525 s:

```bash
lm_eval --model vllm --model_args '{ …same JSON as above… }' \
        --tasks piqa,mmlu,hellaswag --batch_size 32 --seed 42 --output_path "$OUTDIR"
```

Quick sanity slice first (`--limit 16`, one task):

```bash
lm_eval --model vllm --model_args '{ …same JSON as above… }' \
        --tasks piqa --batch_size 32 --seed 42 --limit 16 --output_path "$OUTDIR"
```

Results land in `lm_eval_results/<sanitized-ckpt-path>/results_<timestamp>.json`. Protocol knobs
that must stay fixed for any comparison: `language_model_only=True`, `enable_thinking=False`,
`reasoning_parser=qwen3`, `dtype=bfloat16`, `seed=42`. (Upstream's own eval config uses
`enable_thinking: true`; flip that key in the JSON to match their convention — the numbers in §3.1
are the thinking-**off** protocol.) For the BF16 reference, point `pretrained` at
`Qwen/Qwen3.8-Flash-Next` and raise to `tensor_parallel_size: 4` with
`CUDA_VISIBLE_DEVICES=0,1,2,3`.

## 7. Known issues and caveats

- **MXFP8 MoE + TP>1 crashes with vLLM's automatic backend choice.** At
  `_shuffle_mxfp8_moe_weights` (`.../quantization/utils/flashinfer_utils.py:459-460`) the FlashInfer
  TRTLLM path swizzles the `down_proj` scales with columns padded to a multiple of 4, then reshapes
  to the **unpadded** shape:

  ```
  RuntimeError: shape '[512, 2560, 10]' is invalid for input of size 15728640
  ```

  Trigger condition is per-rank scale columns `moe_intermediate_size / TP / 32`: here `640` →
  TP=1 ⇒ 20 (fine), **TP=2 ⇒ 10 (pads → crash), TP=4 ⇒ 5 (pads → crash)**. Workaround used and
  verified end-to-end: `--moe-backend marlin`. Marlin is a dequant-oriented kernel, so these scores
  came from a **correct but not fastest** path; re-measure throughput if MoE GEMM performance matters
  (the `deep_gemm` / `humming` backends are not installed here; `triton` is rejected on CUDA for
  MXFP8: `No supported MXFP8 expert class for TRITON_MXFP8`). A real fix is upstream-side.
- **TP=1 does not work, and it is not the weights.** Two measured attempts (0.95 GPU util,
  `max_num_batched_tokens` 16384 then 2048) both OOM asking for exactly **95.37 GiB** = the PLE
  n-gram table (51.2 B × 2 B). At TP=1 it cannot be sharded and a second full copy is needed; batch
  size is irrelevant. TP≥2 shards it along the vocab dim.
- **Shared-expert allocation differs from the reference artifact** (§3.4): 144 layers stay BF16 here.
- **MMLU regresses 1.37 pp vs BF16** (−4.8σ on the MXFP8 stderr; §3.3) while gsm8k / piqa /
  hellaswag are flat. The BF16 reference is a supplied table without its own stderr, so re-run it at
  TP=4 if the paired error bars matter to your decision.
- **Multimodal path unvalidated.** The vision tower is untouched BF16 and would load, but all numbers
  here are `language_model_only=True`. Dropping that flag is untested for this artifact.
- **`triton_kernels.matmul_ogs` import errors** appear in every log. Non-fatal: that module is only
  needed if the MXFP4 MoE resolves to the Triton backend, which it does not here.
- **Long waits at start-up are normal, not a hang.** `No available shared memory broadcast block
  found in 60 seconds` with 0 % GPU util is vLLM's keepalive while a rank compiles or captures CUDA
  graphs — with 512 experts that is several minutes (weight load alone was 252–593 s/rank). Do not
  "fix" it with `enforce_eager=True`.
- **First engine start ≫ subsequent starts**: inductor AOT compile caches and the FlashInfer autotune
  cache (`~/.cache/vllm/`) are reused afterwards.
- **Do not reuse the 27B ignore list** (or vice versa): dense vs MoE flips the `mlp.gate` verdict, and
  this model additionally needs `ple`, `hyper_connection`, `block_inject_weight`,
  `mlp.shared_expert_gate`.

## 8. License and attribution

Base model [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is released
under the **Qwen Community License 1.0** (`LICENSE` in this repository), which this quantized
derivative inherits. Two obligations worth flagging before commercial use, taken from that text:
products above 100 M monthly active users or US$ 20 M monthly revenue must prominently display the
model name; and "Model as a Service" / "AI Work Assistant" businesses need a separate license from
Qwen (internal use is exempt). Read the `LICENSE` file itself rather than this summary.

Quantization performed with [Intel AutoRound](https://github.com/intel/auto-round) (Apache-2.0);
serving via [vLLM](https://github.com/vllm-project/vllm) with FlashInfer / CUTLASS DSL / Marlin
kernels; evaluation via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
0.4.13. MXFP8 follows the [OCP Microscaling Formats (MX) specification](https://opencompute.org)
(E8M0 shared scale per 32-element block).

Full design rationale, the fused-expert parameter census, the reverse-engineered comparison against
the HF reference artifact, the MoE swizzle post-mortem, and the MXFP4-Mixed (M0) and MXFP4 (M2)
variants of this recipe are in `Qwen3.8-Flash-Next-quantization-recipe.md` in the workspace root.
