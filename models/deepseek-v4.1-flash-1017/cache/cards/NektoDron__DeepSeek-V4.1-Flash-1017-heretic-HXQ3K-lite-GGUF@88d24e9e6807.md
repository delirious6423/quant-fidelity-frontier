---
language:
- en
- ru
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
- pqhaz/DeepSeek-V4.1-Flash-1017-heretic
base_model_relation: quantized
tags:
- gguf
- xolotl
- deepseek-v4.1
- heretic
- mixed-precision
pipeline_tag: text-generation
---

# DeepSeek-V4.1-Flash-1017-heretic · HXQ3K-lite

Mixed **imatrix Q3_K + GPTQ Q2_0** routed experts for [Xolotl](https://github.com/Indras-Mirror/Xolotl).
Base: [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) →
[pqhaz's heretic checkpoint](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic) → this quantization.
MIT. This is a community conversion.

**lite190u:** 190 GiB of routed experts; 214.1 GB / 199.4 GiB total model weights. Gate/up and down precision
are chosen independently by measured sensitivity. **96 massive experts always stay Q3_K.** Dense tensors and
retained Q3_K expert matrices are byte-identical to HXQ3K. Engram tables are external and are not included in
that size. This GGUF uses Xolotl's mixed expert layout and Q2_0 grid: **Xolotl only**.

Use current Xolotl `main` after `248982be` or newer; split model loading was added in `1a80dedb`.
The launch settings below also use the newer arena pipeline and indexer options, so a current main is recommended.

## Quality

| Quant | ds4q top-1 ↑ | NLL ↓ | KL to API ↓ |
|---|---:|---:|---:|
| HXQ3K | 91.90% | 0.363 | 0.110 |
| **HXQ3K-lite190u** | **91.40%** | **0.386** | **0.132** |
| XQ V4 | 88.95% | 0.451 | 0.197 |

XQ V4 uses IndrasMirror's own refusal-direction edit, so that row compares complete model builds.

`ds4q` is a fixed set of **112 cases**, teacher-forced against DeepSeek API continuations and top-20 token
probabilities. Top-1 is agreement with the API's top token, weighted by continuation positions. NLL is for
the API's sampled continuation. KL compares the API top-20 distribution, renormalized to its observed mass,
with the model's full log-softmax; it is not full-vocabulary API KL. All arms use the same cases and exact
routing (`--route-bias 0 --route-bias-file 0`).

Lite190u: **GSM8K 99/100**, **tool calls 21/23**, **JailbreakBench harmful refusals 0/100** (benign 0/100).
Refusals use marker matching in the first 240 characters of responses capped at 96 tokens, greedy, reasoning off.
These are small fixed evaluations of this build; they are not full benchmark-suite results.

## Speed and memory

One **RTX 5090**, **EPYC 9654**, fast NVMe; 64 CPU expert threads, FP4 KV, DSpark Q2_K sidecar, four server
slots. Smaller machines were simulated with cgroups on a host with more RAM. The cgroup limits include
the engine and its charged file cache; these are not runs on separate physical 256/192 GB hosts.

| lite190u | 256 GB profile (`MemoryMax=236G`) | 192 GB profile (`MemoryMax=172G`) |
|---|---:|---:|
| Arena | 194 GiB | 156 GiB |
| Generation (4K / 16K / 32K prompts), tok/s | 49.7 / 47.5 / 47.5 | 39.4 / 45.1 / 44.8 |
| Prefill (4K / 16K / 32K), tok/s | 647 / 1278 / 1626 | 450 / 1012 / 1366 |
| Batch 4 aggregate generation, tok/s | 84.8 | 67.0 |
| Expert computations from file tier | 0.00% | 0.48% |

HXQ3K at the same 172G limit generated 32.0 / 36.9 / 36.1 tok/s (batch 4: 47.3). File-tier counters describe
expert sourcing; “0%” does not include Engram's normal sparse row reads. Throughput depends on CPU memory
bandwidth, PCIe, context, concurrency and DSpark acceptance.

## Files and integrity

The main model is **standard split GGUF**, each shard below 48 decimal GB. Keep all shards together and give
Xolotl the first `-00001-of-00005.gguf` file. **No reassembly is required.**

| File | Purpose |
|---|---|
| `V41-heretic-1017-XQ3K-lite190u-0000{1..5}-of-00005.gguf` | model weights, tensor-boundary split |
| `rank-xq3k.bin` | HXQ3K expert ranking, used by `--rank` to seed VRAM/RAM placement |
| `V41-heretic-1017-XQ3K-lite190u.gguf.map.json` | per-expert gate/up and down precision map |
| `V41-heretic-1017-XQ3K-lite190u.gguf.experts.json` | per-expert types and `{file, offset}` for each projection inside the shards |
| `tensor-manifest.json` | tensor names, shapes, types, shard offsets and payload SHA256 |
| `SHA256SUMS`, `*.gguf.sha256` | file checksums |
| `verification.json` | split-load and logits comparison results |

Every tensor payload was read back from the written shards and checked against its source SHA256. The split
and whole files were then loaded through Xolotl on an RTX 5090 and compared on all 112 ds4q cases; the exact
comparison result is in `verification.json`. The source whole-file checksum is in `source.sha256`.

## Download and run

### 1. Weights and dependencies

The examples put the `Xolotl/` checkout alongside `HXQ3K-lite/`, `XQ-deps/` and `FP8-engram/`.
Run these downloads from that parent directory; run extraction and the engine from the Xolotl checkout.

```bash
hf download NektoDron/DeepSeek-V4.1-Flash-1017-heretic-HXQ3K-lite-GGUF \
  --exclude 'lite175u/*' --local-dir HXQ3K-lite
(cd HXQ3K-lite && sha256sum -c SHA256SUMS)

# DSpark/vision sidecar and the matching tokenizer are hosted by IndrasMirror.
hf download IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored \
  --include 'V41-MTP-Vision-1017.Q2_K.gguf' 'tokenizer/*' --local-dir XQ-deps
```

Sidecar: [V41-MTP-Vision-1017.Q2_K.gguf](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored/blob/main/V41-MTP-Vision-1017.Q2_K.gguf).

### 2. Engram tables (not hosted here)

The heretic checkpoint uses the original model's unchanged Engram tables. Download the two official shards
and extract them using Xolotl, following the [XQ model card](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored),
[Xolotl README](https://github.com/Indras-Mirror/Xolotl#readme) and
[DeepSeek V4.1 documentation](https://github.com/Indras-Mirror/Xolotl/blob/main/docs/DEEPSEEK_V41.md).
Each source shard is about 101.5 GB; allow about 203 GB for extracted tables plus temporary source storage.

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  model.safetensors.index.json config.json --local-dir FP8-engram

# From the Xolotl checkout; choose fast storage for the two output directories.
python3 tools/v41/extract_engram.py ../FP8-engram \
  --to layers.1=../engram-l1 --to layers.14=../engram-l14
```

### 3. Xolotl settings

Build and configure Xolotl using its [setup instructions](https://github.com/Indras-Mirror/Xolotl#readme).
The RTX 5090 measurements here used `CMAKE_CUDA_ARCHITECTURES=120a`; the 89 example on the older XQ card
is for an RTX 4090. The engine arguments for the **256 GB profile** are:

```bash
-m ../HXQ3K-lite/V41-heretic-1017-XQ3K-lite190u-00001-of-00005.gguf \
--backend cuda --experts gpu --rank ../HXQ3K-lite/rank-xq3k.bin \
--engram ../engram-l1:../engram-l14 --ctx 524288 --stop 1 --threads 64 \
--arena-gib 194 --arena-adapt --kv-fp4 --route-bias 0 --route-bias-file 0 \
--prompt-batch 12288 --tier-batch-pcie 0.1 --conversation-cache-mib 8192 --vision
```

Set these in the engine/server environment (paths are relative examples; adjust for your installation):

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export STRATA_NO_LARGEPAGES=1
export STRATA_ARENA_PIN_AFTER=1 STRATA_ARENA_PIPE=1
export STRATA_INDEXER_NVIS=1 GGML_CUDA_MXFP4_WMMA=1
export STRATA_DENSE_REQUANT=q6_K STRATA_DENSE_REQUANT_CACHE=cache/dense-q6k
mkdir -p "$STRATA_DENSE_REQUANT_CACHE"
export STRATA_PRESTAGE_GIB=1.5
export STRATA_SERVE_STEP_MAX=96 STRATA_SERVE_PASS_BIAS=0
export STRATA_SERVE_DSPARK=../XQ-deps/V41-MTP-Vision-1017.Q2_K.gguf
export STRATA_DSPARK_KEEP=tools/v41/serve/dspark_expert_counts.txt:64
export STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_CONF=0.6
export STRATA_SERVE_LOOKUP=14 STRATA_SERVE_DSPARK_EVICT=4096
export STRATA_DSPARK_PREFAULT=1 STRATA_DSPARK_STASH=1
```

In `serve_ds41.py`'s config, use `parallel: 4`, `../XQ-deps/tokenizer` as tokenizer, temperature 1 and top_p 1.0.
The argument block goes in the config's `args` array; use your built `ds41_generate` as `exe`. The shown
`--vision` flag enables image routing; image input also requires Xolotl's vision encoder configuration and
weights (see its setup documentation). A text-only setup can omit `--vision` and the `vision` config section.

For **192 GB**, change to `--arena-gib 156 --conversation-cache-mib 2048` and set
`STRATA_DSPARK_STASH=0`. Keep the other arguments. Leave RAM for the OS and Engram's sparse row cache.

## Quantization code

[Branch `NektoDron/Xolotl:ds41-lite-quant`](https://github.com/NektoDron/Xolotl/tree/ds41-lite-quant/tools/v41/lite)
contains preparation, CPU calibration, allocation, mixed-GGUF writing, comparison and self-tests.
A single teacher pass fits GPTQ Q2_0 and measures the four gate/up + down type combinations per expert on
held-out rows. Allocation uses relative output error weighted by token usage, with massive-expert protection.
The writer copies Q3_K bytes and the HXQ3K dense recipe without another quantization of those tensors.

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model and reference inference;
[pqhaz](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic) for the heretic checkpoint;
[Indras-Mirror](https://github.com/Indras-Mirror/Xolotl) for Xolotl, XQ and the DSpark/vision sidecar;
llama.cpp / ggml contributors for Q3_K and imatrix tooling.

## Split-load verification

lite190u: **112/112 cases bitwise identical** to the whole GGUF at all 6922 continuation positions (full vocabulary); maximum absolute difference 0, whole/split KL 0. Complete raw logit dumps also match in 112/112 cases.

lite175u: **112/112 cases bitwise identical** to the whole GGUF at all 6922 continuation positions (full vocabulary); maximum absolute difference 0, whole/split KL 0. Complete raw logit dumps also match in 112/112 cases.


## lite175u variant

The smaller variant is also available in [`lite175u/`](https://huggingface.co/NektoDron/DeepSeek-V4.1-Flash-1017-heretic-HXQ3K-lite-GGUF/tree/main/lite175u): 175 GiB of experts, 197.98 GB total.
It passed the same 112-case ds4q gate: **90.41% top-1**, NLL 0.413.
Its whole/split comparison is recorded separately in `lite175u/verification.json`; its own throughput and task checks are in `lite175u/README.md`.
