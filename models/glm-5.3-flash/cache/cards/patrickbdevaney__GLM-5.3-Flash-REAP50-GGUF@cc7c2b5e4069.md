---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - llama.cpp
  - glm5_next
  - moe
  - reap
  - expert-pruning
  - imatrix
  - mtp
  - jetson-thor
---

# GLM-5.3-Flash REAP-50 — GGUF

GGUF quantisations of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
with half of its routed experts removed by REAP saliency pruning (144 of 288 kept in every MoE layer).
Five quant levels from 67.2 to 92.5 GiB, plus the vision projector, for people running the model
in llama.cpp on a single 128 GB unified-memory machine. These files need the patched llama.cpp fork
described below; stock llama.cpp cannot load them. The FP8 parent they were built from scores
**0.8425 top-1 agreement** with the unpruned model on 241,516 held-out tokens. The GGUF files
themselves have not been evaluated that way.

[![Base model](https://img.shields.io/badge/base-zai--org%2FGLM--5.3--Flash-blue)](https://huggingface.co/zai-org/GLM-5.3-Flash)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Model overview

| | |
|---|---|
| Base model | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (`Glm5NextForConditionalGeneration`, MIT) |
| GGUF architecture | `glm5-next` |
| Mask | **v2** (pass 2): the same weights as [GLM-5.3-Flash-REAP50-FP8-v2](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2) at the repo root. GGUF `general.name` is `Glm 5.3 Flash Reap50 Fp8 Pass2` |
| Parameters | 164.91 B in total, as reported by `llama-bench` for the IQ3_M file (MTP block included) |
| Routed experts | 144 of 288 per MoE layer; 8 active per token; 1 shared expert; expert FFN width 2048 |
| Layers | 45, plus 1 MTP block (`blk.45`), so the GGUF has 46 blocks. Layers 0–2 are dense FFN. The trunk has 34 KDA linear-attention layers and 11 MLA layers (3, 7, …, 43), with mHC hyper-connections (4 streams) |
| Hidden size / heads | 4096 / 64 |
| Context length | 1,048,576 in the header. Retrieval tested only up to 32,000 tokens (see *Evaluation*) |
| Vocabulary | 154,880 |
| Precision | Standard `llama-quantize` mixes per level (see *Files*). Router, norms, mHC and KDA conv/state tensors are F32; `output.weight` is Q6_K in every file. No tensor is Q8_0 |
| Vision | `mmproj-GLM-5.3-Flash-REAP50-F16.gguf`: 24-block ViT, projector type `glm4v`, F16. Not pruned |
| Licence | MIT, inherited from the base model |

## Which file should I use?

No measurement here ranks the quant levels by quality (see *Evaluation*), so choose by memory.
Every file passed the same load / arithmetic / perplexity gate on a Jetson AGX Thor with
`-ngl 0 --no-repack`.

| File | Size | Bits/weight | imatrix | Notes |
|---|---:|---:|---|---|
| [`GLM-5.3-Flash-REAP50-Q4_K_M.gguf`](GLM-5.3-Flash-REAP50-Q4_K_M.gguf) | 92.51 GiB (99,330,938,048 B) | 4.82 | no | largest; NIAH-tested |
| [`GLM-5.3-Flash-REAP50-Q4_K_S.gguf`](GLM-5.3-Flash-REAP50-Q4_K_S.gguf) | 87.10 GiB (93,526,480,064 B) | 4.54 | no | |
| [`GLM-5.3-Flash-REAP50-IQ4_XS.gguf`](GLM-5.3-Flash-REAP50-IQ4_XS.gguf) | 82.00 GiB (88,048,752,032 B) | 4.27 | yes | ChartQA 17/24 |
| [`GLM-5.3-Flash-REAP50-Q3_K_M.gguf`](GLM-5.3-Flash-REAP50-Q3_K_M.gguf) | 73.37 GiB (78,776,759,712 B) | 3.82 | yes | |
| [`GLM-5.3-Flash-REAP50-IQ3_M.gguf`](GLM-5.3-Flash-REAP50-IQ3_M.gguf) | 67.18 GiB (72,132,392,352 B) | 3.50 | yes | ChartQA 19/24; MTP-tested |

Bits/weight are as reported by `llama-quantize`. Whether the imatrix was used is recorded in each
file's own `quantize.imatrix.*` metadata.

Sibling repos (same v2 mask):

| Repo | Format | Use it for |
|---|---|---|
| [GLM-5.3-Flash-REAP50-FP8-v2](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2) | FP8 safetensors | The parent of these GGUFs. Build your own quants from it |
| [GLM-5.3-Flash-REAP50-NVFP4-v2](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2) | NVFP4 safetensors | NVFP4-capable runtimes |

A later v3 mask (`pass3-hope/`) exists only inside those two repos. **There is no v3 GGUF.**
v3 has never been evaluated end to end, and its healing used a fallback estimator (see *Changelog*).

## Quickstart

### 1. Build the patched llama.cpp

These files declare `general.architecture = glm5-next`. The fork
[patrickbdevaney/llama.cpp](https://github.com/patrickbdevaney/llama.cpp) adds that architecture:
the mHC Sinkhorn operator, the hybrid KDA/MLA graph and the vision tower. All testing on this card
used builds of that fork.

Upstream llama.cpp now also has a `glm5-next` architecture, but it reads different metadata keys.
It expects `glm5-next.hyper_connection.count` and a required `glm5-next.attention.indexer.kpool`;
these files carry `glm5-next.attention.hc.mult` and no `kpool` key. Upstream should therefore refuse
to load them. That conclusion comes from reading upstream source (as of 2026-11-06), not from a test.

```bash
git clone --branch glm5-next-reap50-gguf-v1 https://github.com/patrickbdevaney/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The same changes ship in this repo in two other forms:

- `glm5-next-llama.cpp.patch` is 15 commits on upstream `761797ff`; apply it with `git am`.
- `glm5-next-llama.cpp.bundle` is a self-contained git bundle with no prerequisite commits
  (`git clone --branch glm5_next glm5-next-llama.cpp.bundle llama.cpp`).

The tag sits at patch commit 12. Commits 13–15 (hidden-state dump, MTP draft and recurrent-state
rollback) are on the `glm5_next` branch.

### 2. Run

**Always add the two stop-token overrides.** GLM-5.3 has three end-of-generation tokens
(`154820 <|endoftext|>`, `154827 <|user|>`, `154829 <|observation|>`), but these headers carry only
`eos_token_id = 154820`. Without the overrides, the model ends its answer with `<|user|>` and then
writes a fabricated next turn. With them, llama.cpp lists all three as EOG tokens (verified in the
vision runs).

Image + text, as run for the ChartQA check (Thor, CPU decode):

```bash
./build/bin/llama-mtmd-cli -m GLM-5.3-Flash-REAP50-IQ3_M.gguf \
    --mmproj mmproj-GLM-5.3-Flash-REAP50-F16.gguf --image chart.png -p "Your question" \
    -ngl 0 --no-repack -c 4096 \
    --override-kv tokenizer.ggml.eot_token_id=int:154827 \
    --override-kv tokenizer.ggml.eom_token_id=int:154829
```

Text, as run by the release gate (which ran without the overrides; add them for chat use):

```bash
./build/bin/llama-completion -m GLM-5.3-Flash-REAP50-Q4_K_M.gguf -c 512 -n 160 --temp 0 \
    -ngl 0 --no-warmup --no-repack -p "Q: What is 17 multiplied by 23?
A:"
```

`llama-server` with these files is **untested**.

**On unified-memory machines (Jetson Thor and similar), start with `-ngl 0 --no-repack`.** The file
stays memory-mapped and its pages stay evictable. Offloaded layers become non-evictable device
buffers, and `-ngl 99` on a 93 GiB file ran the 128 GB Thor out of memory (recorded in
`scripts/gguf_gate.sh`).
Partial offload (`-ngl 24`, `-ngl 36`) of IQ3_M ran fine.

## Evaluation

### The parent: teacher-forced agreement with the unpruned model

Measured on the FP8 v2 checkpoint these GGUFs were converted from, not on the GGUFs. Both models
are teacher-forced over 241,516 held-out tokens that calibration never saw (image-placeholder
positions excluded). The metric is how often the pruned model's top-1 token matches the unpruned
model's.

| Domain | Tokens | Top-1 agreement | ΔNLL |
|---|---:|---:|---:|
| math | 66,558 | 0.920 | +0.022 |
| code | 15,821 | 0.919 | +0.052 |
| agentic | 69,202 | 0.873 | +0.137 |
| science | 45,034 | 0.830 | +0.125 |
| finance | 22,215 | 0.755 | +0.181 |
| ballast (general prose) | 22,154 | 0.580 | +0.989 |
| **all** | **241,516** | **0.8425** | **+0.1756** |

Top-k KL against the unpruned model is 0.6503. The vision bucket (532 text tokens) is too small to
report. Source: `artifacts/eval/eval_glm-5.3-flash-reap50-fp8-pass2.json` in the build pipeline
([glm-5.3-reap](https://github.com/patrickbdevaney/glm-5.3-reap); method in
[wiki/97-evaluation.md](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/97-evaluation.md)).
The damage is uneven: code and maths stay above 0.91, while general prose falls to 0.58.

### Measured on these GGUF files

| Check | File | Result |
|---|---|---|
| Arithmetic smoke test (`17 × 23`, greedy) | all five | pass |
| Perplexity tripwire (8 × 512 tokens) | Q4_K_S / Q4_K_M / IQ4_XS / IQ3_M / Q3_K_M | 4.5833 / 4.5959 / 4.6002 / 4.6253 / 4.6502 |
| Needle-in-a-haystack, depths 0.1 / 0.5 / 0.9 | Q4_K_M | 2,000 tok 3/3 · 8,000 tok 3/3 · 32,000 tok 3/3 (median 54 s / 88 s / 236 s) |
| ChartQA relaxed accuracy, 24 held-out questions, with mmproj | IQ3_M | 19/24 = 79.2% (Wilson 95% CI 59.5–90.8%); 1 empty reply counted wrong |
| Same harness | IQ4_XS | 17/24 = 70.8% |

How to read these:

- **The perplexity numbers detect a broken quant and nothing else.** The gate text
  ([`artifacts/ppl_gate.txt`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/ppl_gate.txt))
  is printed PyTorch tensor reprs of token IDs, mostly digits, not natural language. The
  differences between levels are within noise and do not mean Q4_K_S beats Q4_K_M.
- **ChartQA n = 24 is a smoke test.** The confidence interval is wide, and the IQ3_M-vs-IQ4_XS gap
  is inside it. ChartQA years are graded exactly rather than within 5%.
- Sources: `artifacts/ppl_GLM-5.3-Flash-REAP50-*.txt`,
  [`artifacts/niah_GLM-5.3-Flash-REAP50-Q4_K_M.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/niah_GLM-5.3-Flash-REAP50-Q4_K_M.json),
  [`artifacts/vision_characterization.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/vision_characterization.json)
  (IQ3_M). The IQ4_XS ChartQA result is from the shipping log.

**Not measured:** teacher-forced agreement or KL for any GGUF file; the extra cost of 4-bit
quantisation on top of pruning; generative benchmarks (MMLU, HumanEval, tool-calling, etc.);
retrieval beyond 32,000 tokens; any quality ranking between the five levels.

## Hardware and memory

Everything above ran on one NVIDIA Jetson AGX Thor (128 GB unified memory; llama.cpp reports
125,771 MiB of device memory) with fork builds of llama.cpp.

| Measurement | Value |
|---|---|
| IQ3_M decode, `-ngl 24` | 2.99 tok/s (`llama-completion`); 2.99 ± 0.07 tok/s (`llama-bench` tg1) |
| IQ3_M decode, `-ngl 36` | 3.79 tok/s |
| IQ3_M decode, `-ngl 0` (CPU) | ~1 tok/s (ChartQA harness note) |
| MLA KV cache | 22.00 MiB for 2,048 cells over 11 layers (~11 KiB/token, f16) |
| KDA recurrent state | 145.56 MiB, independent of context length |

Long context is limited by compute, not memory. llama.cpp runs the 11 MLA layers as **dense**
attention. The fork's DeepSeek-sparse-attention path exists but is off unless `GLM5_DSA=1`, and
these files' imatrix has no indexer entries. NIAH latency grows from 54 s at 2k tokens to 236 s at
32k on Q4_K_M.

## Speculative decoding with the built-in MTP block

Every file contains the model's native multi-token-prediction block as `blk.45`.
`make_mtp_draft.py` (in this repo) extracts it into a standalone draft GGUF, which needs the
`glm5_next` branch (not the tag) for recurrent-state rollback:

```bash
python make_mtp_draft.py GLM-5.3-Flash-REAP50-IQ3_M.gguf GLM-5.3-Flash-REAP50-IQ3_M-MTP-draft.gguf
./build/bin/llama-speculative-simple -m GLM-5.3-Flash-REAP50-IQ3_M.gguf \
    -md GLM-5.3-Flash-REAP50-IQ3_M-MTP-draft.gguf -ngl 24 --draft-max 3 --draft-min 1
```

Measured on IQ3_M at `-ngl 24` with short (48-token) generations, **it is slower than not
speculating**:

| | Baseline | Draft depth 2 | Draft depth 3 |
|---|---:|---:|---:|
| Acceptance | — | 46.5% (20/43) | 58.5% (24/41) |
| Speed | 2.99 tok/s | 1.73 tok/s | 1.85 tok/s |

The draft itself costs 5.33 ms per token. The target does not get cheaper per token until the batch
reaches 32 tokens (`llama-bench`: pp16 4.30 tok/s, pp32 23.05 tok/s), so verifying 2–4 tokens costs
about as much as decoding them. Make the draft from the same file you serve: final hidden states
differ between quant levels (cosine 0.891 between IQ3_M and IQ4_XS, against 0.999 for a same-quant
control; [`research/MTP_CAPTURE_FINDINGS.md`](research/MTP_CAPTURE_FINDINGS.md)).

## How it was made

1. **Pruning (v2 mask).** REAP saliency scores each routed expert by its router-weighted output norm
   over a calibration mixture. Calibration covered agentic, code, math, science, finance, general
   and image-text data. Licence policy: permissively licensed sources only. The 144 highest-scoring
   of 288 experts were kept in each MoE layer. Pruning deletes FP8 expert tensors and slices the
   router; surviving weights are not changed.
2. **Healing.** A measured per-layer scalar (42 layers; median 0.9102, range 0.8335–0.9495) was
   applied to the routed-expert block scales. A per-expert variant was tried first, measured worse, and
   was reverted on 2026-08-29, before these GGUFs were built.
3. **Conversion.** The FP8 checkpoint went to a Q8_0 GGUF (163.4 GiB, not published) through the
   fork's `convert_hf_to_gguf.py` with `GLM5_KEEP_MTP=1`, then to each level with
   `llama-quantize --allow-requantize`.
4. **Importance matrix.** `imatrix-GLM-5.3-Flash-REAP50.dat` (264,108,030 B; legacy format;
   765 entries; 347 chunks of 512 tokens) was computed on the Q4_K_M file, not on Q8_0, and used
   for IQ4_XS, Q3_K_M and IQ3_M. Its calibration file
   ([`artifacts/imatrix_calib.txt`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/imatrix_calib.txt))
   is, like the perplexity gate file, printed tensor reprs of token IDs rather than natural text.
   The activation statistics therefore come from digit strings. The effect on the three imatrix
   quants has not been measured. Some experts are never routed to by the calibration, so their
   rows fall back to unweighted quantisation (`partial data (99.31%)` in the imatrix log).

Files uploaded 2026-08-30 to 2026-08-31.

## Limitations

- **A patched llama.cpp is required** (see *Quickstart*). Other GGUF runtimes have not been tried.
- **General-prose and trivia knowledge took the most damage** (top-1 agreement 0.58 on the parent).
  Code and maths held best.
- **The quant levels are unranked.** No teacher-forced, KL or benchmark comparison exists between them.
- **The imatrix and perplexity-gate corpora are not natural text** (see *How it was made*).
- **The stop tokens are incomplete in the header**; pass the two `--override-kv` flags.
- **Sparse attention is not used by default**, so attention is dense and long prompts are slow.
  Retrieval is verified only to 32k tokens.
- **MTP speculation is a net slowdown** in llama.cpp today.
- Vision evidence is limited to n = 24 ChartQA questions (IQ3_M and IQ4_XS only) and a synthetic-image smoke test on Q4_K_M (shapes, colours, positions and the number all
  identified correctly).

## Changelog / corrections

- **2026-08-30 – 08-31.** Q4_K_M, Q4_K_S, IQ4_XS, Q3_K_M, IQ3_M, mmproj, imatrix and llama.cpp patch
  uploaded.
- **2026-09-03.** MTP draft support: `make_mtp_draft.py`, recurrent-state rollback, patch and bundle
  regenerated.
- **2026-10-12.** Clarified that all GGUFs are v2 and no v3 GGUF exists.
- **2026-10-24.** Corrections to the v3 description:
  - v3 was healed with the fallback first-moment per-layer scalar (median 0.766), not measured or
    per-expert gains.
  - v3 has not been evaluated end to end.
  - The v3 NVFP4 build also quantises the KDA attention projections.
  - None of this affects the GGUF files, which are v2.
- **2026-11-06.** Card rewritten and verified against repo files and build artifacts; a 2026-10-26
  automated rewrite contained errors.

## License and attribution

MIT, the licence of the base model ([`LICENSE`](LICENSE), Copyright (c) 2026 Z.AI Co., Ltd).
The base model is [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by
[Z.ai](https://huggingface.co/zai-org). Pruning follows REAP
([Lasby et al., 2025, arXiv:2510.13999](https://arxiv.org/abs/2510.13999)). Quantisation uses
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).

## Citation

```bibtex
@misc{devaney2026glm53flashreap50gguf,
  author       = {Patrick Devaney},
  title        = {GLM-5.3-Flash REAP-50 GGUF: 50\% expert-pruned GLM-5.3-Flash for llama.cpp},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF}}
}
```
