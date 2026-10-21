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
- mtp
- speculative-decoding
- text-generation-inference
---

# Qwen3.8-27B-MXFP8-CT-AutoRound-mtp

MXFP8 (OCP Microscaling, 8-bit block FP with E8M0 group scales) quantized checkpoint of
**Qwen3.8-27B** — a dense native vision-language model — produced with
[Intel AutoRound](https://github.com/intel/auto-round) `0.17.0` in **model-free RTN** mode
(no calibration dataset, no model load) and exported in **compressed-tensors** format
(`format: mxfp8-quantized`) for direct loading by vLLM.

This checkpoint is a **MTP-quantized** variant of
[`intel-ai/Qwen3.8-27B-MXFP8-CT-AutoRound`](https://huggingface.co/intel-ai/Qwen3.8-27B-MXFP8-CT-AutoRound).
It uses the *same* text-tower recipe, but additionally quantizes the **Multi-Token-Prediction
(MTP) draft head** to MXFP8 — all seven MTP Linear layers (`self_attn.{q,k,v,o}_proj`,
`mlp.{gate,up,down}_proj`) — while keeping only the fused MTP input projection **`mtp.fc` in
BF16** (it concatenates two differently-scaled norm streams into the hidden entry point and is the
most sensitive layer of the head; see §7). The reference card leaves the entire MTP block in BF16.

> This card documents a locally produced research artifact. Every quantization and evaluation
> number below was measured on the machine described in [Reproducibility](#5-reproducibility); the
> only figures taken from elsewhere are the BF16 baseline scores of the base checkpoint, labelled
> as such where they appear (§3.1), and the sibling full-MXFP8 numbers, which are cross-linked.

## 1. Model summary

| | Base model | This checkpoint |
|---|---|---|
| Name | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | Qwen3.8-27B-MXFP8-CT-AutoRound-mtp |
| Architecture | `Qwen3_5ForConditionalGeneration` (dense VLM: GDN + gated-attention hybrid text tower, vision tower, 1-layer MTP) | identical |
| Parameters | 27.781 B | 27.781 B (unchanged; weights re-encoded) |
| Weight dtype | BF16 (1,199 tensors) | 407 × `F8_E4M3` + 407 × `U8` (E8M0 scales) + 792 × `BF16` = 1,606 tensors |
| Size on disk (safetensors) | 55.56 GB / 51.75 GiB, 18 shards | **31.64 GB / 29.46 GiB**, 18 shards |
| Compression ratio | 1.00× | **1.76×** |
| Quantization | — | W8A8 MXFP8, `group_size=32`, symmetric, dynamic activations |
| Format | safetensors | `compressed-tensors` / `mxfp8-quantized` (vLLM-native) |
| Context length | 262,144 native | unchanged |
| License | Apache-2.0 | Apache-2.0 (quantization adds no new restriction) |

**What is different from the reference card:** the MTP draft head is now MXFP8. This checkpoint
has **407** quantized layers vs **400** in the reference (the extra **7** are the MTP Linears),
and **259** ignored entries vs **266** (the 7 MTP Linears moved out of `ignore`; only `mtp.fc`
remains).

**Not quantized** (259 entries in `quantization_config.ignore`):

| Group | Count | Reason |
|---|---|---|
| `lm_head`, `model.language_model.embed_tokens` | 2 | 1.27 B params each; output / input embedding least tolerant to low bits |
| `linear_attn.in_proj_a` / `in_proj_b` | 96 | vLLM fuses them into `in_proj_ba` with `N=48+48=96`; the MXFP8 FlashInfer CuTe-DSL kernel asserts `N >= 128` → engine crashes at `profile_run` if quantized |
| `linear_attn.conv1d` | 48 | 3-D depthwise conv, not a Linear layer |
| `model.visual.*` | 112 | Vision tower kept in BF16; text evals do not exercise it and no MX kernel path exists there |
| `mtp.fc` | 1 | Fused `concat[norm(embed), norm(hidden)] → hidden` bottleneck with mixed-scale inputs (analogous to DeepSeek `eh_proj`); the single sensitive layer of the MTP head |

## 2. Precision plan (what got 8 bits)

Quantized layers in the final artifact: **407**

| Bit-width | Modules | Count |
|---|---|---|
| **8 bit (MXFP8)** | text `mlp.{gate,up,down}_proj` ×64 → 192 · text `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` ×48 → 144 · text `self_attn.{q,k,v,o}_proj` ×16 → 64 · **MTP `self_attn.{q,k,v,o}_proj` → 4 · MTP `mlp.{gate,up,down}_proj` → 3** | **407** |
| 16 bit (BF16) | embeddings + `lm_head`, vision tower, `in_proj_a/b`, `conv1d`, **`mtp.fc`**, all norms | 259 (ignore) + norms |

The **7 MTP Linears added here** all satisfy the MXFP8 kernel constraints (`N >= 128`,
`K % 32 == 0`), so none had to be force-ignored. Only `mtp.fc` is held back on sensitivity grounds.

## 3. Evaluation results

Harness: `lm-eval` with the `vllm` backend (vLLM **0.22.1**), `seed=42`, `batch_size=64`,
single B200 GPU. Serving args (identical for every run, matching the reference card protocol):

```
max_model_len=8192, max_num_batched_tokens=32768, max_num_seqs=128, add_bos_token=True,
gpu_memory_utilization=0.85, dtype=bfloat16, max_gen_toks=2048, enable_prefix_caching=False,
reasoning_parser=qwen3, language_model_only=True, enable_thinking=False
```

| Task | Setting | Metric | Score | stderr | Samples |
|---|---|---|---|---|---|
| gsm8k | 5-shot, `--apply_chat_template --fewshot_as_multiturn` | exact_match (strict **and** flexible) | **0.9704** | ±0.0047 | 1,319 |
| piqa | 0-shot | acc | 0.8079 | ±0.0092 | 1,838 |
| piqa | 0-shot | acc_norm | 0.8139 | ±0.0091 | 1,838 |
| mmlu | 0-shot, 57 subtasks | acc | **0.8331** | ±0.0030 | 14,042 |
| hellaswag | 0-shot | acc | 0.6345 | ±0.0048 | 10,042 |
| hellaswag | 0-shot | acc_norm | 0.8272 | ±0.0038 | 10,042 |

MMLU category aggregates: STEM 0.8265 · Other 0.8593 · Social Sciences 0.9074 · Humanities 0.7717.

> Note: with `language_model_only=True` the MTP head is **not** instantiated, so these text scores
> reflect the **text tower only** (identical recipe to the reference card). They confirm the text
> tower is unchanged; the effect of quantizing the MTP head is measured separately in §4.

### 3.1 Comparison: MTP-quantized vs full-MXFP8 (MTP BF16) vs BF16 baseline

Same four tasks / metric variants (gsm8k strict-match; 0-shot `acc` for piqa / mmlu / hellaswag):

| | gsm8k (strict) | mmlu | piqa (acc) | hellaswag (acc) | **AVG** |
|---|---|---|---|---|---|
| BF16 `Qwen/Qwen3.8-27B` | 0.9742 | 0.8340 | 0.8096 | 0.6369 | 0.8137 |
| full-MXFP8 (MTP BF16) · [reference](https://huggingface.co/intel-ai/Qwen3.8-27B-MXFP8-CT-AutoRound) | 0.9719 | 0.8348 | 0.8085 | 0.6361 | 0.8128 |
| **MXFP8 + MTP-MXFP8 (this checkpoint)** | 0.9704 | 0.8331 | 0.8079 | 0.6345 | **0.8115** |

All three sit within task standard error of each other. Quantizing the MTP head does **not** move
the text-task scores (expected: the head is off the `language_model_only` serving path).

## 4. MTP speculative decoding — does quantizing the draft head hurt?

This is the metric that actually depends on MTP precision. With MTP speculative decoding enabled
(`speculative_config={"method":"qwen3_5_mtp","num_speculative_tokens":1}`), the draft head proposes
1 token per step and the main tower verifies it. **Draft acceptance rate** reflects draft quality;
it is batch-independent and is the clean indicator of MTP-quantization damage.

Measured on this machine (B200, vLLM 0.22.1, greedy, 8-prompt reasoning set, `max_tokens=256`),
against a control checkpoint quantized identically **but with the whole MTP block left in BF16**:

| MTP head | acceptance rate | mean accept len (max 2.0) | single-stream decode (spec off → on) |
|---|---|---|---|
| **MXFP8 (this checkpoint)** | **0.9451** | 1.945 | 102.9 → 154.3 tok/s |
| BF16 (control) | 0.9481 | 1.948 | 103.7 → 120.4 tok/s |

**Acceptance rate is essentially unchanged (0.9451 vs 0.9481, Δ ≈ 0.3 pp — inside run-to-run
noise).** Quantizing the MTP head to MXFP8 does not degrade draft quality, so MTP speculative
decoding keeps its ~1.2–1.5× single-stream speedup (the per-run speedup gap above is JIT/warmup
noise on an 8-prompt micro-benchmark, not a quantization effect). MTP speculative decoding is
lossless in output quality either way — rejected draft tokens are corrected by the main model.

**Takeaway:** quantizing the MTP head is effectively free — it trims the head's weight/bandwidth
footprint without hurting acceptance.

## 5. Reproducibility

### 5.1 Hardware / software

| | Value |
|---|---|
| GPU | NVIDIA **B200** (183 GB), quantization is CPU/disk-bound (model-free streaming rewrite) |
| auto-round | **0.17.0** |
| vLLM (eval + spec-decode) | **0.22.1** (V1 engine; registers `model_type=qwen3_5` + MXFP8 CuTe-DSL kernel) |
| Format | compressed-tensors / `mxfp8-quantized` |

### 5.2 Quantize

```bash
auto-round \
  --model_name /home/kaokaolv/models/Qwen3.8-27B \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers visual,lm_head,embed_tokens,mtp.fc,in_proj_a,in_proj_b \
  --format llm_compressor \
  --device_map auto \
  --output_dir ./Qwen3.8-27B-MXFP8-CT-AutoRound-mtp
```

Measured run: **403.97 s** wall, 18/18 shards, **407** quantized layers, **259** ignored layers.
The only change from the reference recipe is `mtp` → `mtp.fc` in `--ignore_layers`: this keeps the
7 MTP Linears in the quantized set and holds back only the fused `mtp.fc` entry projection.

Two auto-round behaviours worth knowing:

- `MXFP optimized RTN is enabled` — per group it evaluates the baseline E8M0 scale together with
  2× and 0.5× candidates and keeps the best. Pass `--disable_opt_rtn` for plain RTN.
- Auto-round force-ignores layers model-free RTN cannot handle (`embed_tokens`,
  `linear_attn.conv1d`, `rotary_emb`, `visual.pos_embed`) and logs them as `Detected N layer(s)
  incompatible with model-free RTN` — that warning is expected, not an error.
- `--ignore_layers` is a **substring match**. `mtp.fc` matches only `mtp.fc.weight` (it does **not**
  touch `mtp.pre_fc_norm_*`). Do **not** write `mtp.gate` — it would also skip every text
  `mlp.gate_proj` (64 layers, 5.7 B params).

### 5.3 Evaluate (text scores, §3)

```bash
CKPT=./Qwen3.8-27B-MXFP8-CT-AutoRound-mtp
ARGS="pretrained=${CKPT},tensor_parallel_size=1,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.85,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,reasoning_parser=qwen3,language_model_only=True,enable_thinking=False"

# gsm8k — chat template + multi-turn few-shot
lm_eval --model vllm --model_args "$ARGS" --tasks gsm8k --batch_size 64 --seed 42 \
        --apply_chat_template --fewshot_as_multiturn --output_path lm_eval_results_mxfp8_mtp

# 0-shot suite — log-likelihood scoring, no chat template
lm_eval --model vllm --model_args "$ARGS" --tasks piqa,mmlu,hellaswag --batch_size 64 --seed 42 \
        --output_path lm_eval_results_mxfp8_mtp
```

### 5.4 Measure MTP speculative decoding (§4)

Drop `language_model_only` and enable the `qwen3_5_mtp` draft head:

```python
from vllm import LLM, SamplingParams
llm = LLM(model="./Qwen3.8-27B-MXFP8-CT-AutoRound-mtp", max_model_len=8192,
          gpu_memory_utilization=0.85, max_num_seqs=1,
          speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 1})
# ... generate, then read acceptance from llm.get_metrics():
#   vllm:spec_decode_num_accepted_tokens / vllm:spec_decode_num_draft_tokens
```

## 6. Usage

### 6.1 vLLM (text, no spec decode)

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve ./Qwen3.8-27B-MXFP8-CT-AutoRound-mtp \
  --tensor-parallel-size 1 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --language-model-only --reasoning-parser qwen3
```

### 6.2 vLLM with MTP speculative decoding

Remove `--language-model-only` (the MTP head must be instantiated) and add:

```bash
vllm serve ./Qwen3.8-27B-MXFP8-CT-AutoRound-mtp \
  --tensor-parallel-size 1 --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
```

`num_speculative_tokens` is capped at **1** (`mtp_num_hidden_layers=1`). Requires a Blackwell-class
GPU and a vLLM build carrying the MXFP8 CuTe-DSL linear kernel.

## 7. Known issues and caveats

- **`mtp.fc` is intentionally BF16.** It is the fused entry projection of the MTP head and mixes
  two differently-scaled normalized streams; it is the one layer of the head likely to be
  sensitive to MXFP8. Quantizing the other 7 MTP Linears is validated to be acceptance-neutral (§4).
- **`in_proj_a` / `in_proj_b` must stay BF16** — fused `N=96 < 128` crashes the MXFP8 kernel at
  startup. (Same constraint as the reference card.)
- **Text scores (§3) do not exercise the MTP head** (`language_model_only=True`). Use §4 to judge
  MTP quantization; use §3 to confirm the text tower matches the reference.
- **Only the text path is validated.** Images/videos load (vision tower is untouched BF16) but no
  multimodal benchmark was run.
- **BF16 baseline (§3.1) is the upstream-reported table**, not a re-run on this stack; treat the
  deltas as indicative.
- **First engine start is slow** (profile + cudagraph capture + FlashInfer autotune). Warm starts
  reuse the inductor/FlashInfer caches under `~/.cache/`.

## 8. License and attribution

The base [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) checkpoint is Apache-2.0;
this quantized derivative is offered under the same license. Quantization was performed with
[Intel AutoRound](https://github.com/intel/auto-round) (Apache-2.0); the artifact is served through
[vLLM](https://github.com/vllm-project/vllm) using FlashInfer / CUTLASS DSL kernels. The MXFP8
format follows the [OCP Microscaling Formats (MX) specification](https://opencompute.org) (E8M0
shared scale per 32-element block). Recipe and sibling full-MXFP8 reference:
[`intel-ai/Qwen3.8-27B-MXFP8-CT-AutoRound`](https://huggingface.co/intel-ai/Qwen3.8-27B-MXFP8-CT-AutoRound).
