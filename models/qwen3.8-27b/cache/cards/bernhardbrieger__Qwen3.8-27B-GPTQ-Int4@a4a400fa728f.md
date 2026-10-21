---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- gptq
- gptqmodel
- vllm
- int4
- 4-bit
- quantization
- mtp
- speculative-decoding
- vision
- conversational
- code
- long-context
library_name: transformers
quantized_by: Bernhard Brieger
---

# Qwen3.8-27B-GPTQ-Int4

Post-training GPTQ 4-bit quantization of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced on a
**RTX 5090 (32 GB) + Ryzen 9 9950X3D + 64 GB DDR5** system with
[GPTQModel](https://github.com/modelcloud/gptqmodel) 7.3.4. The checkpoint
ships the full multimodal architecture (language + vision tower) plus the
**Multi-Token-Prediction (MTP) draft layer**, and is verified serving 192k
context with an FP8 KV cache on [vLLM](https://github.com/vllm-project/vllm)
0.29.0.

Weight footprint drops from 55.56 GB (bf16, decimal) to 20.98 GB decimal (19.54 GiB)
— a 62.23 % reduction on the decimal-GB basis — while 11 precision-critical module
families are deliberately kept at full precision. Every number on this card is traceable to a file in this
repository (`config.json`, `quantize_config.json`, `quant_log.csv`,
`model.safetensors.index.json`) or to a command in the
[Reproducibility appendix](#6-reproducibility-appendix).

> **A note on how to read this card.** The base model publishes no perplexity
> or benchmark table, and its 55.6 GB bf16 weights cannot co-reside with
> this checkpoint on a 32 GB GPU (sequential single-model runs would fit; this
> card reports absolute numbers rather than a base-vs-quant diff). That
> limitation is stated rather than papered over.

---

## 1. Identity & scope

| Property | Value |
|---|---|
| **Base model** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) |
| **Architecture** | `Qwen3_5ForConditionalGeneration` (model_type `qwen3_5`), image-text-to-text |
| **Parameters** | 27,781,427,952 (~27.78 B), as reported by the base model |
| **Quantization** | GPTQ, **4-bit, group size 32, symmetric**, non-act-ordered |
| **Layers** | 64 — 48 Gated-DeltaNet linear-attention + 16 full-attention |
| **Native context** | 262,144 tokens (verified serving at 196,608) |
| **MTP** | 1 draft layer, 15 tensors, kept bf16, merged into the main shards |
| **License** | Apache-2.0 (base and derived) |
| **Quantized by** | Bernhard Brieger |
| **Quantization date** | 2026-08-21 |

**Intended use.** General instruction following, coding, tool/function calling,
and long-context workloads (retrieval over documents >100k tokens) on
hardware that fits ~20 GB of weights. The 4-bit weights use the standard GPTQ packing format (int32-packed,
group 32, symmetric). The serving path verified on this card is vLLM (Marlin
GPTQ kernels); other runtimes with GPTQ support may load the checkpoint, but
that is not verified here.

**Out of scope.** This is a plain quantization of the base model — no
fine-tuning, instruction distillation, or safety retraining was applied. It is
not intended as a substitute for the base model where full precision is
required, and the numbers below characterize this specific checkpoint, not the
base model in general.

**What is quantized vs. kept at bf16.** 400 linear modules (all `mlp`
projections and attention projections across the 64 layers) are quantized to
4-bit: that is **24,326,963,200 of the 27.78 B parameters (87.6 %)**. The
remaining **3.45 B parameters (12.4 %) stay at bf16**: the token-embedding and
`lm_head` matrices (1.27 B each), the MTP draft layer (425 M), the vision
tower (461 M), and the per-layer layernorms / GDN `conv1d` / `in_proj_a` /
`in_proj_b` / output-gate parameters.

---

## 2. Quantization recipe

Self-contained recipe — every parameter below is stored in
`quantize_config.json` in this repo.

### Toolchain

| Component | Version |
|---|---|
| GPTQModel | 7.3.4 |
| transformers | 5.15.0 |
| PyTorch | 2.13.0+cu130 |
| Python | 3.12 |
| GPU | 1× NVIDIA RTX 5090, 32 GB |
| CPU | AMD Ryzen 9 9950X3D, 16C/32T | calibration loop runs on CPU (per-layer block to GPU) |
| RAM | 64 GB DDR5 | calibration sequences staged on CPU |

### GPTQ configuration

| Parameter | Value | Note |
|---|---|---|
| `bits` | 4 | |
| `group_size` | 32 | uniform — full vLLM (Marlin) compatibility |
| `sym` | `true` | symmetric quantization |
| `desc_act` | `false` | non-act-ordered (deterministic group layout) |
| `lm_head` | `false` | `lm_head` not quantized |
| `act_group_aware` | `true` | GPTQModel default, pinned in metadata |
| `damp_percent` | 0.05 | auto-increment 0.01 on low damping |
| `true_sequential` | `true` | |
| `pack_dtype` | int32 | |
| Hessian staging dtype | float32 | |
| Fallback | `rtn` @ 0.5 % threshold | **never triggered** (0 fallbacks, §3) |

### Modules kept at bf16 (the 11 `dynamic` exclusion patterns)

GPTQ exclusions are *negative* patterns: a matched module is skipped and left
at bf16. The recipe below is what is stored in `quantize_config.json.dynamic`.
Each pattern is namespace-agnostic — it matches both the HuggingFace full
module path and vLLM's layer prefix, because both consumers anchor the same
regular expression at position 0.

| Pattern | What it protects | Why |
|---|---|---|
| `*.input_layernorm` | input layernorms | precision-critical; not in GPTQModel's supported-module set |
| `*.post_attention_layernorm` | post-attention layernorms | same |
| `*.self_attn.q_norm` | query normalization | same |
| `*.self_attn.k_norm` | key normalization | same |
| `*.norm` | all other norms (GDN `linear_attn.norm`, final norm, vision merger) | same; catches the rest |
| `*.linear_attn.conv1d` | GDN causal convolution | `nn.Conv1d` **is** a quantization candidate — load-bearing |
| `*.linear_attn.in_proj_a` | GDN `in_proj_a` | `nn.Linear` candidate — load-bearing |
| `*.linear_attn.in_proj_b` | GDN `in_proj_b` | `nn.Linear` candidate — load-bearing |
| `*.embed_tokens` | token embeddings | `nn.Embedding` candidate — load-bearing |
| `visual.*` | entire vision tower | vLLM quantizes its linears with the same config |
| `mtp.*` | MTP draft layer | keeps speculative-decoding draft weights exact |

> **Why the MTP pattern is written the way it is.** vLLM gates MTP on any
> `dynamic` key whose text contains `mtp` (`qwen3_5_mtp.py`). The form
> `(?:.*\.)?mtp\..*` satisfies that gate and matches a root `mtp.*` without
> false-positiving on e.g. `model.language_model.embed_tokens` (no dot before
> its "mtp"), unlike a naive `-:.*mtp.*`.

### Calibration corpus

| Metric | Value |
|---|---|
| **Sequences** | 2,026 |
| **Total tokens** | 1,802,645 |
| **Longest sequence** | 132,968 tokens |
| **Random seed** | 42 |
| **Ordering** | `desc` (longest first) |
| **Concat block size** | 40,960 (44 full blocks + 1 final; 0 % real-token loss) |
| **Calibration device** | CPU (per-layer block to GPU) |

Built from a **curated pool of public, non-gated sources** (see
[License & provenance](#7-license--provenance)): 11 programming languages, an
agent/tool-calling pool, general web/wiki prose, a multilingual (DE/FR/IT)
pool, and a math pool. The corpus mirrors the deployment profile —
code-heavy with a **2×128k long-document tail**. Note: calibration forward
passes are bounded at 40,960 tokens per block (§6.1), so the long tiers are
*content*-calibrated (their text goes through the quantization loop), not
exercised as 128k/160k-length forward passes.

| Pool | Budget (tok) | Used (tok) | Seqs | Length range |
|---|---:|---:|---:|---|
| code_typescript | 185,000 | 185,815 | 519 | 128–4,096 |
| code_python | 150,000 | 151,189 | 125 | 256–4,096 |
| code_javascript | 115,000 | 116,018 | 160 | 128–4,096 |
| code_csharp | 115,000 | 118,143 | 110 | 128–4,096 |
| code_java | 75,000 | 75,152 | 130 | 128–4,096 |
| code_cpp | 65,000 | 65,347 | 66 | 128–4,096 |
| code_rust | 50,000 | 50,031 | 97 | 128–4,096 |
| code_kotlin | 40,000 | 40,788 | 86 | 128–4,096 |
| code_shell | 45,000 | 45,404 | 55 | 128–4,096 |
| code_swift | 15,000 | 15,500 | 29 | 128–4,096 |
| code_sql | 15,000 | 17,268 | 6 | 128–4,096 |
| agent | 100,000 | 100,090 | 182 | 256–2,048 |
| general (web/wiki/config) | 125,000 | 125,756 | 131 | 128–2,048 |
| multilingual (wiki DE/FR/IT) | 65,000 | 65,152 | 135 | 128–1,024 |
| math | 35,000 | 35,045 | 183 | 64–1,500 |
| **Long tiers** | 12× | 12 tiers | 754 docs | 16k–132,968 |

The 12 long tiers (four at 32k, four at 16k, two at 64k, **two at 128k**) add
754 documents reaching up to 132,968 tokens. During quantization GPTQModel
repacks the corpus into 40,960-token blocks, so no forward pass ever exceeds
40,960 tokens — this bounds the GDN fp32 value-stream tensor and removes the
OOM on the longest sequences structurally.

### Run statistics

| Stage | Wall time |
|---|---:|
| Model load (bf16, meta-init → GPU per layer) | 2 s |
| **Quantize** | **90.2 min** |
| Save (6 shards) | 15 s |

Total wall time ~90 min on a single RTX 5090.

---

## 3. Quality verification

Four independent checks: the quantizer's own per-module loss, an absolute
perplexity, a functional text-smoke suite, and the 6-color vision suite —
all run against the live server.

### (a) Quantization loss (from `quant_log.csv`, 400 modules)

| Metric | Value |
|---|---:|
| Modules quantized | 400 (48 GDN × 6 + 16 full-attn × 7) |
| Mean per-module loss | 1.108e-05 |
| Max per-module loss | 6.713e-05 |
| Worst module | `L61 mlp.gate_proj` |
| Worst cluster | `mlp.gate_proj` at layers 58–61 (~6.3–6.7e-05) |
| **RTN fallbacks** | **0** (fallback configured but never triggered) |

The per-module loss is the squared quantization error GPTQ minimizes per
linear layer. A max of 6.7e-05 across all 400 modules, with zero modules ever
falling back to the RTN (round-to-nearest) strategy, indicates a clean,
uniform fit — no layer required a degraded fallback to converge.

### (b) Perplexity (absolute)

| Metric | Value |
|---|---|
| Perplexity | **2.729** |
| Tokens scored | n = 260 |
| Method | `max_tokens=1` + `prompt_logprobs=1` over a fixed, neutral English prose sample; PPL = exp(−mean logprob) over the input tokens |
| Runtime | live vLLM (this checkpoint), `scripts/measure_ppl_quant.py` |

This is an **absolute** perplexity on a single 260-token sample — a sanity
check that the dequantized model assigns sensible probabilities to plain
prose, not a quality benchmark. The base model publishes no perplexity, so
there is nothing to diff against; the number is reported as-is.

### (c) Functional smoke (3/3 pass, live server)

| Probe | Result | Detail |
|---|---|---|
| Coding generation | PASS | 2,273 chars in 11 s (`enable_thinking: false`) |
| Long-context needle | PASS | target located at **46,037-token** context, returned `ZEBRAX42` (2 s) |
| Tool / function calling | PASS | exact JSON: `{"name":"get_weather","arguments":{"city":"Graz","unit":"celsius"}}` (1 s) |

The needle probe exercises the long-context regime that the 2×128k calibration
tail was built to protect. The tool-call probe verifies the
`qwen3_coder` parser path end-to-end.

### (d) Vision probes (6/6 pass on vLLM, live server)

| Probe | Result | Detail |
|---|---|---|
| Solid-color suite | 6/6 PASS | red 128×128, green 128×128, blue 128×128, yellow 256×256, purple 64×64, cyan 512×512 — each answered with its exact color word |

Six deterministic solid-color PNGs (generated in-process, stdlib-only, no
asset files) were sent as base64 data-URLs through the OpenAI-compatible
`/v1/chat/completions` endpoint; each must be answered with its color word.
All six pass — the bf16 vision tower and the image-processor path are
intact in this quantized checkpoint, verified end-to-end on the vLLM 0.29.0
serving stack from §4. The same 6-probe set is used for the NVFP4 sibling
card, so the two are directly comparable.

### What is *not* verified here

- **No base-vs-quant benchmark table.** The base card publishes no
  benchmarks or perplexity, and the 55.6 GB bf16 base cannot co-reside with
  this checkpoint on the 32 GB test GPU (sequential single-model runs would
  fit; this card reports absolute numbers instead of a diff).
- **MTP speculative decoding** is preserved (15/15 tensors, bf16) but was
  served without the speculative flag in the runs above; see §4.

---

## 4. Deployment

Verified serving configuration on vLLM 0.29.0 (Marlin GPTQ kernels, FP8 KV
cache, FlashInfer attention), 196,608-token context on an RTX 5090 32 GB (0.95-utilization flag set):

```bash
vllm serve bernhardbrieger/Qwen3.8-27B-GPTQ-Int4 \
  --tensor-parallel-size 1 \
  --max-model-len 196608 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --attention-backend FLASHINFER
```

- **Weights (on disk):** 19.54 GiB (20.98 GB decimal); steady-state engine
  consumption 19.23 GiB (weights + non-torch, per vLLM's memory profiler). At 192k context
  with an FP8 KV cache, the 0.95-utilization engine fills ~29.9 GiB of the 31.36
  GiB available (weights + non-torch 19.23 + KV pool 8.18 + peak activation
  2.38 + CUDAGraph 0.06 GiB) — it fits, with `max_num_seqs 4`.
- **Startup:** ~46 s from process start to server ready (192k tier,
  measured from the server log).
- **Reasoning / tool parsing:** `--reasoning-parser qwen3` +
  `--tool-call-parser qwen3_coder` are what make structured tool-call output
  work (see the smoke probe in §3c).

### Optional: MTP speculative decoding

The MTP draft layer is present (15 tensors, bf16). To use it:

```bash
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

`num_speculative_tokens: 2` is the Qwen-recommended value for a single MTP
layer. The MTP weights remain fully usable **without** the flag (they are
simply unused), so the checkpoint runs as a normal model on any setup —
speculative decoding is opt-in. Note: with the MTP draft model enabled
the KV budget no longer fits at 163,840 / 196,608 context — MTP-1/MTP-2
work up to 131,072; for 160k+ context run without the speculative flag
(MTP-0).

### Loading outside vLLM

The verified serving path for this checkpoint is vLLM 0.29.0 (§4). Outside a
serving engine, the `qwen3_5` architecture (Qwen3.5 multimodal stack) is
**not** auto-resolved by `AutoModelForCausalLM` in transformers 5.17 (no
`qwen3_5` entry in its model mapping), so the auto-class fails at
model-type resolution. Load the explicit architecture class
(`Qwen3_5ForConditionalGeneration`) and use GPTQModel's loader to
dequantize the packed int32 weights.

### Performance (measured)

Measured 2026-09-21/22 with vLLM **0.29.0**, GPU util 0.95; 6 context
tiers × 3 MTP states; per-request (single stream) and 4-stream
aggregate wall-clock throughput, 1 warmup + 3 timed runs,
`max_tokens=256`, temperature 0.

| ctx | MTP-0 (1 stream) | MTP-0 (4 streams) | MTP-1 (1 stream) | MTP-1 (4 streams) | MTP-2 (1 stream) | MTP-2 (4 streams) |
|---|---|---|---|---|---|---|
| 32,768 | 20.44 | 24.77 | 11.65 | 6.77 | 8.79 | 9.53 |
| 65,536 | 9.81 | 10.64 | 5.07 | 5.52 | 3.90 | 4.22 |
| 131,072 | 4.48 | 4.48 | 2.10 | 2.14 | 1.45 | 1.50 |
| 163,840 | 2.42 | 3.10 | does not start | does not start | does not start | does not start |
| 196,608 | 1.47 | 1.47 | not executable | not executable | not executable | not executable |
| 262,144 | does not start | does not start | does not start | does not start | does not start | does not start |

(tok/s — 1 stream: per-request throughput; 4 streams: aggregate wall-clock across the 4 concurrent streams)

Cell notes: all runs at GPU util 0.95. *Does not start* (163,840 /
262,144) / *not executable* (196,608) — the MTP draft model exceeds the
remaining KV budget at those tiers; MTP-0 (base) is the only MTP state
at 160k+. The 262,144 tier is not reachable on this 32 GB card. MTP draft acceptance
(vLLM 0.29.0 @ 0.95 utilization, 32k–128k tiers, per-run spec-decode
metrics): MTP-1 mean acceptance length 1.84–1.98 (draft acceptance
84–98 %); MTP-2 2.44–2.80 (72–90 %). Per the table above, MTP is
net-negative per-request at the tiers where it runs on this
checkpoint — the draft-step overhead outruns the accepted tokens.

**Phase breakdown (single stream, MTP-0)** — prefill = prompt tokens /
TTFT p50; decode = per-token time (TPOT p50). 4-stream cells are not
decomposed: at 32k the KV pool caps effective concurrency at 4, at
64k at 3, and the excess requests queue server-side (their TTFT
includes the wait).

| Tier | TTFT p50 (s) | prefill (tok/s) | decode (ms/tok → tok/s) |
|---:|---:|---:|---:|
| 32,768 | 9.27 | 3,504 | 12.7 → 78 |
| 65,536 | 21.32 | 3,060 | 13.5 → 74 |
| 131,072 | 53.39 | 2,450 | 14.7 → 68 |
| 163,840 | 73.52 | 2,225 | 15.5 → 65 |
| 196,608 | 170.07 | 1,154 | 16.2 → 62 |

Decode (int4 Marlin GEMMs) is the context-independent phase and
stays ~60–78 tok/s per stream; prefill cost grows with context length
and is the bottleneck at the 160k/192k tiers.

The historical single-shot snapshot (vLLM 0.27.1, 2026-08-22, 0.90-util
flag set) predates the flag set verified here and is not reported on
this card.

---

## 5. Hardware requirements

| Deployment | Minimum VRAM |
|---|---:|
| Weights only (short context) | ~20 GB |
| 192k context, FP8 KV, 4 seqs (verified) | 32 GB (RTX 5090) |
| 163,840 / 196,608 context **with MTP** | does not fit — MTP draft OOMs; run MTP-0 |

The verified configuration uses the full 31.36 GiB of usable VRAM at
0.95 utilization. The 19.54 GiB (20.98 GB decimal) on-disk weight
footprint is a fixed cost: a GPU below ~20 GB cannot load this checkpoint
at any context length. The remaining budget goes to the KV cache, which
scales with context length and concurrency — it determines how much
context a given card can serve.

---

## 6. Reproducibility appendix

Every number in §1–§3 is produced by one of the following. Paths are relative
to this repository.

**Exact re-quantization needs the calibration corpus** (the file referenced in
the code below), which is documented in [`provenance.json`](provenance.json)
but not redistributed with this checkpoint.

### 6.1 Quantization

The checkpoint is produced by GPTQModel over a 2,026-sequence corpus:

```python
import json
from gptqmodel import GPTQConfig, GPTQModel

DYNAMIC = {
    "-:.*\.input_layernorm$": {},
    "-:.*\.post_attention_layernorm$": {},
    "-:.*\.self_attn\.q_norm$": {},
    "-:.*\.self_attn\.k_norm$": {},
    "-:.*\.norm$": {},
    "-:.*\.linear_attn\.conv1d$": {},
    "-:.*\.linear_attn\.in_proj_a$": {},
    "-:.*\.linear_attn\.in_proj_b$": {},
    "-:.*\.embed_tokens$": {},
    "-:(?:.*\.)?visual\..*": {},
    "-:(?:.*\.)?mtp\..*": {},
}

model = GPTQModel.load(
    "Qwen/Qwen3.8-27B",
    GPTQConfig(bits=4, group_size=32, desc_act=False, sym=True,
               lm_head=False, dynamic=DYNAMIC,
               offload_to_disk=True, offload_to_disk_path="output/offload",
               calibration_data_device="cpu"),
)
seqs = json.load(open("calibration_data.json"))["sequences"]  # see provenance.json
model.quantize(seqs, batch_size=1,
               calibration_sort="desc",
               calibration_concat_size=40960)
model.save("output/Qwen3.8-27B-GPTQ-Int4")
```

`offload_to_disk` streams finished layers to NVMe during the layer loop (the
default would otherwise stage them on the GPU). Per-module losses land in
`quant_log.csv`; the resolved config is written to `quantize_config.json`.

### 6.2 Perplexity

Against a running server on `:8000`, scoring the fixed 260-token sample with
input-token logprobs:

```bash
python scripts/measure_ppl_quant.py
# -> quant_ppl  n=260  perplexity=2.729
```

The script POSTs the fixed sample with `max_tokens=1, prompt_logprobs=1` and
computes `exp(-mean logprob)` over the prompt positions.

### 6.3 Functional smoke (text + vision)

```bash
python scripts/smoke_test_api.py
# -> 3 probes: coding / needle_40k / tool_call  (results in smoke_test_api.jsonl)

python scripts/vision_smoke.py
# -> 6 solid-color probes (red/green/blue 128x128, yellow 256x256,
#    purple 64x64, cyan 512x512); each must be answered with its
#    exact color word  (results in vision_smoke.json)
```

Both scripts are engine-agnostic (target via the `SMOKE_BASE` / `SMOKE_MODEL`
env vars) and hit any OpenAI-compatible server. The §3 results were taken
against the **vLLM 0.29.0** server from §4 — the verified serving path for
this checkpoint (GPTQ auto-detected from the checkpoint's
`quantization_config`, FP8 KV cache, 196,608-token context).

---

## 7. License & provenance

**Model weights.** Apache-2.0. The base
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) is Apache-2.0 and
this derived checkpoint is released under the same license.

**Calibration data.** The corpus is drawn entirely from public, non-gated
datasets. Every source, its license, and (for code) its pinned commit are
listed below; the full manifest is [`provenance.json`](provenance.json) in
this repository. The raw calibration sequences are not redistributed with
this checkpoint.

| Source | License | Used in |
|---|---|---|
| 28 code repositories | Apache-2.0 (13) · MIT (13) · BSD-2-Clause (1) · BSD-3-Clause (1) | code pools (11 languages) |
| [codeparrot/codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) | per-file upstream license (no dataset-level license declared) | Python pool |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (EN/DE/FR/IT) | CC BY-SA 3.0 + GFDL | wiki prose, long-document tail |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | MIT | math pool |
| [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | ODC-BY | agent / tool-calling pool |
| [HuggingFaceTB/SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/SmolLM-Corpus) (FineWeb-Edu dedup) | ODC-BY | web prose, long-document tail |

Each code repository is pinned to a specific commit in
[`provenance.json`](provenance.json) (repo + commit SHA + license), so the
code portion of the corpus is reconstructable byte-for-byte.

**Data attribution.** Wikipedia text © Wikipedia contributors, released under
CC BY-SA 3.0 and the GFDL. FineWeb-Edu and the Tülu-3 SFT mixture are used
under ODC-BY. Attribution is provided here in lieu of per-file notices.

*Card data updated 2026-09-23: vision verification added — §3d: 6-color solid-color probe suite 6/6 PASS against the vLLM 0.29.0 §4 serving stack (bf16 vision tower + image-processor path intact after quantization); §6.3 extended with `scripts/vision_smoke.py` (6 probes, engine-agnostic). This checkpoint is vLLM-only — no other engine was verified for it; phase breakdown added to the Performance section (TTFT, prefill tok/s + per-token decode time from the raw matrix runs).*

*Card data updated 2026-09-22: vLLM 0.29.0 (was 0.27.1); vLLM serving config + all vLLM performance values now 0.95-utilization only (0.27.1 historical table removed); 0.95 memory-budget breakdown; 62.23 % unit clarification (decimal-GB basis); MTP-at-160k fit note; loading notes; calibration wording tightened; internal terminology removed from the Performance section.*

**Base model references.**
- Base model card: <https://huggingface.co/Qwen/Qwen3.8-27B>
- Qwen3.5 / Qwen3.8 technical report: see the links on the base model card.
