---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
- quantized
- nyquant
- mixed-precision
- int4
- int3
- int2
- gguf
- awq
- vllm
- pascal
- p100
- qwen3.8
quantized_by: XGeorgeCostanzaX
---

# Qwen3.8-27B — Nyquant quantisations (preview)

Nyquant makes mixed-precision weight quantisations that balance model quality, size and hardware-specific inference.
These variants share [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (revision `1d4bf0f2`):
**NQ4MAX-B70** targets Intel Arc B70 with vLLM; **RC3Y** targets Tesla P100 with a patched llama.cpp (patch set included).
**This is a public preview.** Both downloads are early releases; read each variant's limitations before relying on it.
**q2lite (2-bit research)** now has a frozen D4 quality score in its packed FP32 reference runtime. Its equal-runtime check and release qualification remain pending.
**B3 (3-bit research, 22 October)** is a splice of public tensors at 12.04 GB that scores lower KLD than AP-IQ3_S, at equal size, on all four corpora of our board. It is not downloadable yet; see [B3](#b3--3-bit-research-result-22-october-2026).

## Variants at a glance

Bits below are average bits per weight of the **body**, not the whole file. Research sizes are targets, not releases.

| Variant | Bits (body bpw) | Size | Format / runtime | Best hardware | Quality headline | Speed headline | Status |
|---|---:|---|---|---|---|---|---|
| **NQ4MAX-B70** | **4.30** | 20.1 GB (18.7 GiB), 5 shards | Safetensors; AWQ-compatible vLLM export, no patches | Intel Arc B70 | +0.012 nats vs BF16; tested tasks within stated SEs | 58–59 t/s, c1 chat with MTP | **Public preview:** `NQ4MAX-B70-AWQ/` |
| **RC3Y (3-bit)** | **3.29** | 11.65 GB GGUF | Custom GGUF; requires the included llama.cpp patch set | Tesla P100 16 GB | Measured original-file D4 macro KLD 0.143355; behind AP-IQ3_S | 38.9 t/s with MTP K=2 (**development build**) | **Public preview:** `RC3Y/`; release checks still open |
| **B3 (3-bit research)** | Body bpw not asserted | 12,035,434,112 bytes complete (3.58 bpw) | Standard GGUF types; upstream llama.cpp | Not speed-tested | D4 macro KLD 0.099020051; lower than AP-IQ3_S on all 4 corpora at equal size | Not measured | **Research; provisional; no download yet** |
| **q2lite (2-bit research)** | Body bpw not asserted | 9,299,294,112 bytes complete | Packed FP32 reference | Deployment target unqualified | D4 macro KLD 0.18946722593101104; runtime parity pending | Not release-qualified | **Research; K3 fails; no download** |

> Downloadable here: **NQ4MAX-B70** (`NQ4MAX-B70-AWQ/`) and **RC3Y** (`RC3Y/`). There is no 2-bit download yet;
> q2lite and B3 are listed for their research results only.

## Which file should I use?

- **Intel Arc B70 / vLLM:** use the **NQ4MAX-B70 AWQ-compatible export** in `NQ4MAX-B70-AWQ/`.
  It is the higher-quality option here.
- **Tesla P100 16 GB:** **RC3Y**, `RC3Y/Qwen3.8-27B-RC3Y-P100.gguf`, is the speed-oriented option. Build llama.cpp with
  the patch set in `RC3Y/llama.cpp-p100-patches/`; upstream llama.cpp cannot load its custom tensor types.
- **Tesla P40 or other GPUs:** use standard community GGUFs today, for example
  [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) or [bartowski](https://huggingface.co/bartowski),
  choosing a size that fits your memory. IQ3_S and Q3_K_XL have better 3-bit quality on our board.
  RC3Y's fast path is P100-only; P40 and newer GPU fallbacks are untested and expected to be slow.
- **CPU:** prefer a standard community GGUF. RC3Y has a slow reference path, not a practical CPU speed target.

## Files

| Path | Size | Notes |
|---|---|---|
| `NQ4MAX-B70-AWQ/` | 20,085,536,800 bytes in 5 safetensors shards, plus tokenizer and config | SHA-256 list: `NQ4MAX-B70-AWQ/SHA256SUMS` |
| `RC3Y/Qwen3.8-27B-RC3Y-P100.gguf` | 11,650,719,040 bytes | SHA-256 `668faf132b4fcadcf6c36ebf834d83adac44ef621a2d3879342a793976f945ed` |
| `RC3Y/llama.cpp-p100-patches/` | 11 patches + `BUILD.md` | MIT; apply to llama.cpp `69f201a2` (b11551) |
| `LICENSE`, `NOTICE` | | Apache-2.0 and attributions |

## Quality board (D4, BF16 reference)

**Research result, with an open runtime gate.** q2lite's four-domain macro KLD is **0.18946722593101104** at **9,299,294,112 complete bytes**. It was scored in its **own packed FP32 reference runtime**; the public controls were scored in **llama.cpp using their original GGUFs**. An equal-runtime check is pending. These scores do not establish a best-quant claim or a working low-memory deployment.

**K3 fails:** the upper macro confidence bound is **0.21080927346309444**, above the required **0.205**. Passing paired comparisons in this cross-runtime board does not waive that gate.

![Frozen D4 macro KLD with 95% confidence intervals, including every public 2-bit control, the stronger AP-IQ3_S control and RC3Y. q2lite uses a separate packed FP32 runtime and fails K3.](assets/qwen-frozen-d4-20261016.png)

### 2-bit research and public controls

| Quant | Complete artifact bytes | Macro KLD ↓ | 95% window-bootstrap CI | Runtime |
|---|---:|---:|---|---|
| q2lite-native-late50 | 9,299,294,112 | 0.189467226 | [0.171098225, 0.210809274] | packed FP32 reference |
| AP-IQ2_S | 9,607,980,896 | 0.205070357 | [0.186003907, 0.226723904] | native GGUF / llama.cpp |
| UD-Q2_K_XL | 9,828,981,664 | 0.208431916 | [0.184685052, 0.235335855] | native GGUF / llama.cpp |
| bartowski-IQ2_M | 10,522,248,928 | 0.223455428 | [0.203403342, 0.244315571] | native GGUF / llama.cpp |
| ISTA GSQ-RCO-IQ2_S | 9,259,510,912 | 0.224526124 | [0.203575330, 0.247890046] | native GGUF / llama.cpp |
| UD-IQ2_S | 8,371,970,048 | 0.275715745 | [0.252054192, 0.302576649] | native GGUF / llama.cpp |
| bartowski-IQ2_S | 9,684,043,488 | 0.277955272 | [0.255661678, 0.301899920] | native GGUF / llama.cpp |
| bartowski-IQ2_XS | 9,088,526,048 | 0.316145283 | [0.291040614, 0.343021908] | native GGUF / llama.cpp |
| ByteShape-IQ2_XXS-2.56 | 8,843,953,536 | 0.332769478 | [0.308324662, 0.358405218] | native GGUF / llama.cpp |

### Around 3.5 bpw

![D4 macro KLD with 95% confidence intervals for files around 3.5 bits per weight: unsloth UD-Q3_K_XL, Nyquant B3, AP-IQ3_S, ISTA GSQ-RCO-IQ3_S, AP-IQ3_XS, RC3Y, AP-IQ3_XXS, unsloth UD-IQ3_XXS and ISTA GSQ-RCO-IQ3_XXS.](assets/qwen-d4-3bit-20261022.png)

| Quant | Complete artifact bytes | Macro KLD ↓ | 95% window-bootstrap CI | Scored |
|---|---:|---:|---|---|
| unsloth UD-Q3_K_XL | 13,146,393,504 | 0.083593631 | [0.071831769, 0.096703128] | 22 Oct |
| **Nyquant B3 (research splice)** | **12,035,434,112** | **0.099020051** | [0.087641131, 0.112176225] | 22 Oct |
| AP-IQ3_S | 12,040,882,016 | 0.105547323 | [0.093673296, 0.118701684] | 16 Oct; 22 Oct re-score 0.105547321 |
| ISTA GSQ-RCO-IQ3_S | 11,771,546,784 | 0.112602871 | [0.099541383, 0.128040858] | 16 Oct |
| AP-IQ3_XS | 11,489,008,096 | 0.116936116 | [0.103669694, 0.132281515] | 22 Oct |
| RC3Y-original | 11,650,719,040 | 0.143354810 | [0.127917392, 0.160995765] | 16 Oct |
| AP-IQ3_XXS | 10,737,389,536 | 0.150239165 | [0.134722953, 0.167560170] | 22 Oct |
| unsloth UD-IQ3_XXS | 10,934,860,704 | 0.154338009 | [0.136850540, 0.174640700] | 22 Oct |
| ISTA GSQ-RCO-IQ3_XXS | 10,094,357,632 | 0.171362531 | [0.153632617, 0.191704861] | 16 Oct |

B3, our research splice, has the lowest macro KLD among these files at or below AP-IQ3_S's size; the paired comparison is in the [B3 section](#b3--3-bit-research-result-22-october-2026). unsloth UD-Q3_K_XL is 9.2% larger and lower still, on every corpus. RC3Y remains behind AP-IQ3_S on quality. AP-IQ3_S's full file is **3.581464 complete bpw**, so it is not within a strict 3.5-bpw cap. Sizes count the whole artifact, including embeddings, head, metadata, padding and any MTP tensors. No 16-GB capacity claim follows from this quality board.

**22 October rows** use the same frozen teacher, token windows, scorer and llama.cpp commit as the 16 October board, run on an A100 80 GB; AP-IQ3_S re-scored there reproduces its 16 October value (0.105547321 vs 0.105547324). Their intervals were recomputed from the stored per-window arrays with the frozen convention below, which reproduces the 16 October AP-IQ3_S interval.

**Protocol and provenance.** A800 80 GB PCIe; BF16 reference; wiki/web/code/chat, **60 independent windows per domain**, context **2048**, scored positions **1024–2046**, teacher log-p **> -16**. The frozen macro is the equal-weight mean of the four domain KLDs. Intervals use **1000 stratified window bootstrap replicates**, seeds **0/1/2/3** by corpus. Native GGUF source revision: `69f201a2051ea9b9e9b50c3cc56afd9e2ae64414`. Rival files: AP = [agentionai/Qwen3.8-27B-AP-GGUF](https://huggingface.co/agentionai/Qwen3.8-27B-AP-GGUF), UD = [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), bartowski = [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF), ISTA GSQ-RCO = [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF), ByteShape = [byteshape/Qwen3.8-27B-GGUF](https://huggingface.co/byteshape/Qwen3.8-27B-GGUF). q2lite's reference-runtime fidelity remains unresolved. The frozen intervals supersede its original receipt's differently seeded intervals.

Source: `overseer/gsqboard-20261016@2a100b4f`, `evidence/gsqboard_20261016/{REPORT.md,SUMMARY.json,ROWS.json,metadata/PROTOCOL_IDENTITY.json}`. 22 October rows and B3: qboard `cloud-rows/q3sota-*-canonical-original-v1.json` (cohort `canonical-D4-original-frozen-teacher-native-A100-v1`), paired analysis `quant-supervisor-20261022/PAIR_REPORT_B3ALL_2327.txt`. Original RC3Y: `overseer/q3lite-20261016@0d70b131d22e61a8a28755dde278fc97021962b1`, `evidence/q3lite_20261016/REPORT.md`, original Q4_K-head row. Table point displays round our KLD upward and public KLD downward; interval displays round outward. The chart uses the unrounded source values for bar lengths and intervals.

## NQ4MAX-B70 — 4-bit public preview

**Quality:** held-out loss is +0.012 nats over BF16. Every task we ran is within the stated standard errors of
BF16; this is not proof of identical quality. See the [task table](#nq4max-b70-vs-bf16).

**Speed:** on one B70, MTP gives 58–59 t/s for single-stream chat and 60–65 t/s for code, versus 27 t/s without
speculation. Concurrent serving results are in the [speed tables](#speed). MTP is not always better for aggregate throughput.

### Quickstart: vLLM

Requires a vLLM installation with Intel XPU support. This export uses the standard AWQ int4 path
**without Nyquant patches**. Budget for about 18.7 GiB of weights **plus** KV cache and runtime buffers.

```bash
hf download XGeorgeCostanzaX/Qwen3.8-27B-Nyquant-preview --include "NQ4MAX-B70-AWQ/*" --local-dir qwen38-nyquant
vllm serve qwen38-nyquant/NQ4MAX-B70-AWQ --quantization awq
```

Use the base model's recommended sampling settings and the supplied tokenizer/chat template. This is a plain-serving
example, not the exact benchmark or MTP configuration. Validated XPU installation and MTP instructions will follow.

**Limitations:** preview; no like-for-like community 4-bit quality board yet. Task samples are limited,
with no paired significance test or run-to-run variance measured. Vision quality was not separately evaluated;
MTP was evaluated through acceptance and throughput, not a separate quality suite. B70 timings do not establish
performance on other GPUs, and peak context/VRAM use still need release measurements.

## RC3Y — 3-bit P100 public preview

**Quality:** the original RC3Y file with Q4_K head scores macro KLD **0.143354810**, with 95% CI **[0.127917392, 0.160995765]**, at **11,650,719,040 complete bytes**. AP-IQ3_S scores **0.10554732360886131** at **12,040,882,016 bytes**. RC3Y is behind on quality. The original-file result supersedes the older **0.1325 simulation**; it does not qualify a rebuilt or changed-head candidate. The scored file has the same byte count as this upload; a SHA-256 receipt binding the score to this exact file is still to be published. See the frozen board below.

**Speed:** the earlier llama-bench decode measurement was **30.6 t/s**, versus **10.45 t/s** for UD-IQ3_XXS on
upstream llama.cpp. Today's development fork reaches **38.9 t/s with MTP K=2**, versus **29.1 t/s plain** in that
measurement series. These are separate measurements; both need release-build validation. See [Speed](#speed).

**Format:** 11.65 GB (11,650,719,040 bytes; 10.85 GiB), including the MTP layer. The custom `NQ_AR3` / `NQ_AR4`
tensor types require the patched llama.cpp below; being a GGUF does **not** make this file upstream-compatible.

### Quickstart: patch llama.cpp, build, run

**Preview instructions.** The patch set (11 patches in `RC3Y/llama.cpp-p100-patches/`, details in its `BUILD.md`)
applies to upstream llama.cpp b11551, commit `69f201a2`. The CUDA build was tested for compute capability 6.0 with
llama-bench, llama-perplexity and the speculative example; **llama-cli and llama-server have not yet been built or
smoke-tested on this exact patch set.** CUDA toolkit required; these examples use a Bash shell.

```bash
hf download XGeorgeCostanzaX/Qwen3.8-27B-Nyquant-preview --include "RC3Y/*" --local-dir qwen38-nyquant
git clone https://github.com/ggml-org/llama.cpp llama.cpp-p100
cd llama.cpp-p100
git checkout 69f201a2051ea9b9e9b50c3cc56afd9e2ae64414
git am ../qwen38-nyquant/RC3Y/llama.cpp-p100-patches/*.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"
cmake --build build -j --target llama-cli llama-server llama-bench llama-perplexity

export GGUF=../qwen38-nyquant/RC3Y/Qwen3.8-27B-RC3Y-P100.gguf
build/bin/llama-cli -m "$GGUF" -ngl 99 -fa on -c 8192 -p "The capital of France is"
build/bin/llama-server -m "$GGUF" -ngl 99 -fa on -c 8192 --port 8080
```

Use Qwen's recommended sampling settings; the chat template is embedded in the GGUF. The example context is **not**
a validated maximum for 16 GB: reduce it if necessary. These commands illustrate plain decoding; a validated MTP
invocation will follow.

**Limitations:** P100 (compute capability 6.0) is the only measured fast path. P40 correctness and speed are
unverified; do not treat it as supported. Other GPUs are unverified too. About **10.1 GiB** of weights reside on
the GPU, with token embeddings in host memory; allow additional space for KV cache and buffers. Maximum context
on 16 GB remains unmeasured. Task benchmarks, long-context quality, tool calling, vision, LoRA, partial offload and
multi-GPU split need testing. Server/chat sampling paths need smoke tests. CUDA graphs are off on compute capability
6.0. Measurements are from one card in a shared server; run-to-run variance is not established.

## B3 — 3-bit research result (22 October 2026)

**Status:** research result, **provisional** (no independent replication yet). **Not downloadable yet.** Not a best-at-size claim.

**What it is.** B3 is a **splice of public tensors, not a new quantisation format**. It keeps 835 of the 866 tensors of
agentionai's AP-IQ3_S unchanged (92.3% of tensor bytes, values untouched), takes 23 later-layer FFN gate/up tensors from
basecompute's BaseQuant-V1-Small (6.5%), and re-quantises the 8-tensor MTP layer (`blk.64`) to Q2_K (1.2%). The complete
file is **12,035,434,112 bytes**, 5,447,904 bytes smaller than AP-IQ3_S. All tensor types are standard GGUF types, and it
was scored in upstream llama.cpp at commit `69f201a2`.

**Paired comparison with AP-IQ3_S at equal size** (the same 240 windows; B3 minus AP-IQ3_S, 95% paired bootstrap CI):

| Corpus | B3 KLD | AP-IQ3_S KLD | B3 − AP-IQ3_S [95% CI] |
|---|---:|---:|---|
| wiki | 0.0386 | 0.0403 | −0.0018 [−0.0029, −0.0009] |
| fineweb (held-out) | 0.0311 | 0.0336 | −0.0026 [−0.0031, −0.0021] |
| code (held-out) | 0.0203 | 0.0207 | −0.0004 [−0.0008, −0.0001] |
| chat (held-out) | 0.3063 | 0.3274 | −0.0211 [−0.0305, −0.0121] |
| Arithmetic macro | 0.0991 | 0.1055 | −0.0065 [−0.0089, −0.0042] |
| Geometric mean of the four | 0.0523 | 0.0551 | ratio **0.9471** [0.9362, 0.9571] |

B3 is lower on every corpus and every interval excludes zero. That meets our **-Balanced** rule: the upper bound of the
geometric-mean ratio is below 0.99 and no corpus is worse.

**Against other files** (same board, paired windows):

- **unsloth UD-Q3_K_XL** (13,146,393,504 bytes, 9.2% larger) beats B3 on every corpus: geometric ratio 1.2916 [1.2559, 1.3266].
- **Smaller public files** from unsloth, bartowski and agentionai that we scored with per-window arrays (AP-IQ3_XS, AP-IQ3_XXS,
  UD-IQ3_XXS, UD-Q2_K_XL, AP-IQ2_S and bartowski Q2_K, IQ2_M, IQ2_S, IQ2_XXS): B3 is lower on every corpus, with geometric
  ratios from 0.21 to 0.75. These files are 0.5–3.2 GB smaller, so this is expected from size.
- **Not yet scored:** a Hugging Face metadata census on 22 October found 26 more repositories with Qwen3.8-27B GGUFs at or
  below B3's size, for example byteshape IQ3_S-3.23bpw and AtomicChat AD-IQ2_S. Until they are scored there is no
  "best at 12 GB" claim. ISTA GSQ-RCO-IQ3_S is on the board above without a paired test.

**Chat and tool use (-Chat rule):** on our 45-prompt on-policy chat/tool panel, B3 and AP-IQ3_S **tie**: B3 − AP-IQ3_S is
+0.0011 [−0.0035, +0.0061] (mean of six source means; the mean over prompts is 0.03756 vs 0.03758). The panel cannot resolve
differences smaller than about 14%; a larger held-out panel is pending. We make no chat-specific claim.

**Limitations:**

- The MTP (speculative-decoding) layer is Q2_K in B3. The board does not exercise it, so draft acceptance and MTP speed are unmeasured.
- Decode speed, prefill speed and memory use were not measured.
- The D4 windows were used in earlier allocation work in this project. B3's tensor exchange was pre-registered before final
  scoring and screened on separate development windows, but treat the result as provisional until it is replicated on fresh windows.
- A release would ship with a per-tensor provenance MANIFEST and a NOTICE crediting agentionai and basecompute.

## Quality comparison

### NQ4MAX-B70 vs BF16

All rows use the same protocol on the same machine. Q8 and NVFP4 are round-to-nearest evaluation references,
not releases. Task scores and acceptance below are fractions.

| Model | bits/weight (body) | FineWeb NLL (dNLL vs BF16) | WikiText-2 PPL | GSM8K | ARC-e | ARC-c acc / norm | HellaSwag acc / norm | MATH-long | Tool calls | Agentic | MTP acceptance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BF16 | 16 | 1.994 | 7.051 | 0.824 | 0.826 | 0.568 / 0.592 | 0.647 / 0.814 | 0.828 | 0.972 | 0.929 | - |
| Q8 (RTN, reference) | 8 | 1.994 (+0.000) | 7.085 | 0.820 | 0.826 | 0.567 / 0.590 | 0.649 / 0.815 | - | 0.972 | 0.929 | 0.924 |
| NVFP4 (RTN, reference) | about 4.5 | 2.011 (+0.017) | 7.397 | 0.848 | 0.827 | 0.567 / 0.607 | 0.651 / 0.812 | - | 0.972 | 0.929 | 0.926 |
| **NQ4MAX-B70** | **4.30** | **2.006 (+0.012)** | **7.147** | **0.792** | **0.827** | **0.558 / 0.583** | **0.645 / 0.814** | **0.813** | **1.000** | **0.929** | **0.882** |

**Evaluation conditions** (held-out evaluation, not a quantisation recipe):

- FineWeb-Edu NLL: 64 held-out documents × 2048 tokens, not used during quantisation. WikiText-2 uses the same tokenizer.
- GSM8K: first 250 problems, thinking off; standard error about ±0.025. ARC-e/c: full sets; ARC-c SE about ±0.015.
- HellaSwag: fixed subset of 1000. MATH-long: first 64 problems, thinking on, 2048-token limit; SE about ±0.047.
- Tool calls: 36 single-turn cases. Agentic: 14 multi-step episodes.
- MTP acceptance: the model's own head, 2 speculative tokens, vLLM 0.19.1. This is separate from RC3Y's tokens-per-verify metric.
- GSM8K and MATH-long gaps are within the stated SEs; paired bootstrap significance tests have not been run.
- A like-for-like comparison against community 4-bit quants is still pending. These references do not establish a community ranking.

## Speed

### One Tesla P100 16 GB

**Earlier comparison:** decode, llama-bench **tg128**, flash attention on, same card/settings. The RC3Y fork is
based on upstream b11551; the measured run predates the final release build.

| Model | Runtime | tg128 (tokens/s) | GPU memory (weights) | Status |
|---|---|---:|---|---|
| **RC3Y** | Our fork, P100 kernels | **30.6** | 10.1 GiB | **To be re-measured on the release build** |
| UD-IQ3_XXS (unsloth) | Upstream llama.cpp b11551 | 10.45 | 10.2 GiB | Control to be re-measured alongside RC3Y |

That is **2.93×** faster in this comparison. IQ3_S and Q3_K_XL speeds on P100 have **not** been measured.
The P100 kernels' KLD against the exact quantised weights is **1.9e-5**; this measures kernel fidelity, not
quantisation quality against BF16.

**Latest MTP update:** one P100, our **development fork build**, plain decoding versus the included
MTP head at different speculative draft lengths K. These are not the earlier tg128 run above. Exact benchmark
settings and repeated trials must accompany the release measurements; do not treat them as a matched upstream comparison.

| RC3Y mode | Tokens/s | Status |
|---|---:|---|
| Plain (no speculation) | 29.1 | Development build; re-measure on release build |
| MTP K=1 | 37.6 | Development build; re-measure on release build |
| **MTP K=2** | **38.9** | **Development build; re-measure on release build** |
| MTP K=3 | 38.0 | Development build; re-measure on release build |

Mean MTP acceptance is about **3.0 tokens per verify step at K=3**. This is a tokens-per-step metric, not a
percentage. K=2 is the fastest measured setting in this development series; further P100 MTP speedups are on the roadmap.

### One Intel Arc B70

vLLM, AWQ int4 path; test window **2026-10-11**, **one card**, not multi-card throughput. c1/c8 are the reported
concurrency labels. The source provides the following aggregate and per-stream summaries:

| Configuration | c1 chat (t/s) | c1 code (t/s) | c8 aggregate (t/s) | c8 per stream (t/s) |
|---|---:|---:|---:|---:|
| With MTP speculation | 58–59 | 60–65 | 135–173 | 37–41 |
| Without speculation | 27 | 27 | 178–180 | 23 |

The c8 aggregate/per-stream summaries still need reconciliation; do not infer aggregate throughput by
multiplying the per-stream range. Prompt/output lengths, cache settings and the exact serving configuration will
be documented with release measurements. MTP improves single-stream speed here, but not the reported c8 aggregate.

## Method

### 1. Mixed precision by measured sensitivity (NQ4MAX-B70, RC3Y)

Nyquant balances weight precision against measured model sensitivity and a size budget, with runtime support
chosen for the target hardware. Mixed precision preserves more capacity where it matters while reducing weight
storage elsewhere. A small file or fast kernel is not by itself evidence of better model quality; release claims
must be checked on the actual downloadable weights and runtime.

### 2. Measured tensor exchange from the strongest public file (B3, new on 22 October)

For the 12 GB class we did not build a new format. The procedure:

1. **Start from the strongest public file at the size**, here agentionai's AP-IQ3_S, measured against BF16 in the same
   runtime on the four-corpus board.
2. **Keep its tensors and values**, including its importance-matrix rounding; change only a small, pre-registered set of tensors.
3. **Count complete bytes**: every tensor, the MTP layer, metadata and padding. The result never exceeds the starting file.
4. **Move bytes to where they buy the most.** For B3, the MTP layer (which the board does not use) was re-quantised to
   Q2_K, and the freed bytes bought higher-precision payloads for 23 later-layer FFN gate/up tensors, taken from another
   public quantisation (basecompute's BaseQuant-V1-Small). Attention K/V/O and linear-attention (GDN) tensors stay as AP shipped them.
5. **Judge the whole model only:** paired per-window KLD against BF16 on 240 windows across wiki, fineweb, code and chat,
   plus an on-policy chat/tool panel. Block-level proxies never promote a candidate; in our tests they over-predicted
   whole-model gains several-fold.
6. **Two products per size.** A **-Balanced** candidate must lower the geometric mean of the four corpus KLDs (95% CI upper
   bound of the ratio below 0.99) with no corpus worse; a **-Chat** candidate must win on the on-policy chat/tool panel.
   B3 is a -Balanced candidate; its -Chat verdict is open.

The result is a splice: 31 of 866 tensors differ from AP-IQ3_S, and the credit for the underlying tensors belongs to
their publishers. What we add is the measurement and the choice of exchange.

### 3. How we report

- BF16 is the only quality reference. Sizes are complete file bytes. Rivals that beat us stay on the charts.
- Our numbers round against us: our KLD, sizes and interval bounds round up or outward; rival values round down.
- Results without independent replication are labelled provisional.

## Roadmap

- **Further MTP speedups on P100:** validate the improvement on a fresh patched build and publish reproducible commands.
- **B3 (12 GB class):** score the census files that could beat it (for example byteshape IQ3_S-3.23bpw and AtomicChat
  AD-IQ2_S), replicate on fresh windows, measure MTP and speed, then release with a per-tensor provenance MANIFEST and NOTICE.
- **Improved 3-bit and 2-bit formats:** in research; evaluate quality and hardware support before committing to releases.
- **NQ4MAX:** compare with community 4-bit quants of similar size on the same board.
- **RC3Y release checks:** bind the original-file KLD receipt to the selected release artifact; any changed file needs a new score. Complete a fresh patched build, CLI/server smoke tests, repeat decode/prefill benchmarks,
  validated context/memory limits, checksums and packaging. P40 claims require actual hardware testing.

## Licence and acknowledgements

- **Model weights and card:** Apache-2.0, inherited from Qwen/Qwen3.8-27B. `LICENSE` and `NOTICE` are included.
- **llama.cpp patches:** MIT; retain the ggml authors' copyright and licence.
- **Base model:** Qwen Team, [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
- **Quantisation:** Nyquant / XGeorgeCostanzaX.
- **RC3Y reused tensors and metadata:** [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF),
  UD-IQ3_XXS (Apache-2.0).
- **Comparisons:** unsloth, bartowski, agentionai (AP), ISTA-DASLab (GSQ-RCO) and byteshape; links under the quality board.
- **B3 tensors (research build, not distributed here):** [agentionai/Qwen3.8-27B-AP-GGUF](https://huggingface.co/agentionai/Qwen3.8-27B-AP-GGUF)
  (AP-IQ3_S) and [basecompute/Qwen3.8-27B-BaseQuant-V1](https://huggingface.co/basecompute/Qwen3.8-27B-BaseQuant-V1)
  (BaseQuant-V1-Small), both Apache-2.0. Credit, not endorsement.
- **Runtime:** vLLM, llama.cpp and the ggml authors.

## Citation

Formal citation details will follow. For now, cite the Qwen base model and this card as
**“Nyquant Qwen3.8-27B quantisations” by XGeorgeCostanzaX**, recording the chosen variant and repository revision.
