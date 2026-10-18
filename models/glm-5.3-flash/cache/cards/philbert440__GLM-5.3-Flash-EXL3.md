---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- exl3
- exllamav3
- v100
- sm70
---

# GLM-5.3-Flash EXL3 (~2.1 bpw, SM70 tile order)

> **PRs needed to run this on V100 (1Cat-vLLM):** build `main` plus [#1159](https://github.com/1CatAI/1Cat-vLLM/pull/1159) (decode fast paths), [#1166](https://github.com/1CatAI/1Cat-vLLM/pull/1166) and [#1165](https://github.com/1CatAI/1Cat-vLLM/pull/1165) (two more decode fast paths), [#1161](https://github.com/1CatAI/1Cat-vLLM/pull/1161) (MTP startup fix),
> [#1102](https://github.com/1CatAI/1Cat-vLLM/pull/1102) (EXL3 MoE loader and kernels) and
> [#1125](https://github.com/1CatAI/1Cat-vLLM/pull/1125) (fast prefill; it already contains
> [#1092](https://github.com/1CatAI/1Cat-vLLM/pull/1092), the FP32-softmax sparse-MLA kernel).
> [#1091](https://github.com/1CatAI/1Cat-vLLM/pull/1091) (MTP speculative decoding) is already merged into `main`.
> Step-by-step build and serve commands are under [Running it](#running-it-on-v100-1cat-vllm).

An EXL3 quantization of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), made from the
FP8 release. It is built to fit four 32 GB V100 GPUs and to run on the native SM70 EXL3 path in 1Cat-vLLM.

**Parameters: 321B total, about 18B active per token**, the same model as the base. The Hub's "Model size" figure
(~52B) counts stored tensor elements, and an EXL3 tensor is stored as packed int16 trellis words that each hold
16 / K weights. Counted from the tensor shapes, the checkpoint holds 311.8B EXL3 expert weights and 9.6B
unquantized ones.

> **Read this before downloading.** The weights use a V100-friendly tile order, recorded as
> `tile_order: sm70_colmajor` in `quantization_config`. They run in **1Cat-vLLM on V100 with the PRs listed
> under [Running it](#running-it-on-v100-1cat-vllm)**, which load this repo directly. Stock exllamav3 does not
> know this order and will decode the weights wrongly. [exllamav3#475](https://github.com/turboderp-org/exllamav3/pull/475)
> adds the order to the converter, but this checkpoint predates its per-tensor marker, so even with #475
> exllamav3 can't load it. A standard-order (sm80) build is not published.

## Recipe

| Part | Precision |
|---|---|
| Routed experts, most layers | EXL3 2 bpw (K2) |
| Routed experts, layers 4, 5, 7, 8, 10 | EXL3 3 bpw (K3), chosen by a per-layer KL scan |
| MTP layer experts | EXL3 4 bpw |
| Shared experts, attention, dense MLP, eh_proj, lm_head | FP16 |
| Gates, norms, routers, hyper-connections, embeddings, vision tower | FP16 / source dtype |

The overall bitrate excluding the head is 2.12 bpw. The codebook is `mcg`. Calibration used 352 rows of
8192 tokens. Mean per-layer relative reconstruction error over the MoE layers is 0.011 (worst layer 0.043). Expert Hessians blend the full-calibration Hessian with one built from the tokens actually
routed to that expert, at weight 0.5, which cut held-out expert output error by about 14% in our tests.

## Quality

**Fidelity to the FP8 original.** KL divergence and top-1 agreement of next-token predictions against the
unquantized FP8 model, on the same three held-out texts (1,536 tokens each: a README, a design document and a
Python file) for every build we made. Lower KL is better.

| Build | KL README | KL design doc | KL Python | Mean KL | Mean top-1 |
|---|---:|---:|---:|---:|---:|
| First build (all K2) | 0.088 | 0.186 | 0.220 | 0.165 | 85.5% |
| Build 2 | 0.096 | 0.200 | 0.231 | 0.176 | 85.4% |
| Build 3 | 0.086 | 0.181 | 0.187 | 0.151 | 86.4% |
| **This release (build 4)** | **0.070** | **0.139** | 0.188 | **0.132** | **87.8%** |

**Task benchmarks.** Same harness we used for our Qwen3.8-27B quants on V100: gsm8k (200 questions, strict
match, 3,072-token answer budget), 75 factual questions, and 75 questions about things that do not exist
(fabricated drugs, places, papers). Factual and fabrication answers are judged by grok-4.3. Greedy decoding.

| Model | Factual | Fabricated answers (of 75) | gsm8k-200 |
|---|---:|---:|---:|
| **GLM-5.3-Flash EXL3 2.1 bpw (this repo)** | **1.000** | **2** | **0.985** |
| Qwen3.8-27B NVFP4, best of our variants | 0.973 | 27 | 0.965 |
| Qwen3.8-27B NVFP4, Unsloth | 0.973 | 29 | 0.955 |

It answered all 75 real questions correctly and declined 73 of the 75 fictional ones. GLM-5.3-Flash is a much
larger model than a 27B, so the point here is that 2-bit quantization did not cost it that lead, not that it wins.

**Agentic coding.** On a small private set of three real bug-fix tasks graded by the projects' own tests, it
passed two of three, the same as the cloud GLM-5.3-Flash API, at a lower judge score (17 vs 20 of 30 points).
Earlier builds of ours collapsed into repetition loops at 75K-100K tokens of context on these tasks; this
build, served with the attention fix below, did not in our runs (one run per task, so treat this as early evidence).

## Speed

Four V100-SXM2-32GB (NVLink), TP4, one request at a time, FP8 KV cache. Measured on the kernels described below
with a repacked copy of these weights; loading this repo directly through #1102 gives token-identical greedy
output (8/8 prompts, no speculation).

**Decode**, mean of four 700-token chat responses, greedy:

| Speculative decoding | Decode tok/s | Max context |
|---|---:|---:|
| **Built-in MTP head, 3 draft tokens, with #1102 (strip-major kernels) + #1159 + #1166 + #1165 (recommended)** | **~91** | 131,072 |
| Built-in MTP head, 3 draft tokens, with #1102 + #1159 only | ~86 | 131,072 |
| Built-in MTP head, 3 draft tokens, previous kernels | ~76 | 131,072 |
| Built-in MTP head, 4 / 5 draft tokens | 70.4 / 65.9 | 131,072 |
| No speculation | ~48 | 131,072 |

The 86 -> 91 step (2026-10-10, second pass) comes from a profile of the 86 tok/s build: the KDA output norm ran as eight
eager PyTorch kernels per layer because the custom-op dispatch takes the decomposed path and the SM70 decode graph is
not compiled; the Triton gated RMSNorm is bit-identical and six times faster (#1165). The hyperconnection post-mapping
and its logits dot product run as one native kernel for 1-8 tokens instead of two TileLang kernels with an FP32
intermediate (#1166). The 76 -> 86 step (2026-10-10) comes from three decode changes measured one at a time on the same build and
prompts: the expert kernels read the trellis in a strip-major tile order with direct per-lane window loads and
8-deep prefetch (#1102, bitwise-identical output), the hyperconnection final stage runs natively for the 2-8
token verify batch, and the KDA input projection uses cuBLAS above one token (#1159). Greedy output on a
7-prompt set matches the previous kernels exactly on 4 prompts and diverges late (after 250+ characters) on the
rest, the same pattern the previous build shows against its predecessor; 17x23 and the long-prompt summary
are unchanged.

MTP numbers with 4-5 draft tokens are from an earlier build. Three draft tokens is the sweet spot, since each
extra draft costs more step time than it adds in accepted tokens.

**Prefill**, time to first token, unique prompts (no prefix-cache reuse):

| Prompt | with [#1125](https://github.com/1CatAI/1Cat-vLLM/pull/1125) | without |
|---|---:|---:|
| ~1.1K tokens | ~0.8 s (~1,250-1,475 tok/s) | ~1.8 s (~645 tok/s) |
| ~29K tokens | ~21 s (~1,390 tok/s) | ~94 s (~314 tok/s) |
| 70.8K-token needle test | 55 s, passes | 266 s, passes |

## Running it on V100 (1Cat-vLLM)

Hardware: four V100-32GB (SXM2 with NVLink, or PCIe). TP4 needs all four. CUDA 12.8 toolkit, gcc/g++ 14.

### The pieces

| PR | What it does | Status |
|---|---|---|
| [1CatAI/1Cat-vLLM#1091](https://github.com/1CatAI/1Cat-vLLM/pull/1091) | Wires GLM-5.3-Flash's built-in MTP head in as a speculator | merged into `main` |
| [1CatAI/1Cat-vLLM#1102](https://github.com/1CatAI/1Cat-vLLM/pull/1102) | EXL3 routed-expert MoE on SM70: loads this repo **directly** (no repack), decodes the trellis inside the matmul, reads both tile orders | draft |
| [1CatAI/1Cat-vLLM#1092](https://github.com/1CatAI/1Cat-vLLM/pull/1092) | FP32-softmax sparse-MLA kernel over the FP8 KV cache, for decode and speculative verify | draft |
| [1CatAI/1Cat-vLLM#1125](https://github.com/1CatAI/1Cat-vLLM/pull/1125) | Same kernel for prefill, about 4.4x faster long prompts. **Contains #1092.** | draft |
| [1CatAI/1Cat-vLLM#1159](https://github.com/1CatAI/1Cat-vLLM/pull/1159) | Decode fast paths: native hyperconnection final stage for 2-8 token verify batches, cuBLAS for the KDA input projection above one token. ~+5% decode. | draft |
| [1CatAI/1Cat-vLLM#1161](https://github.com/1CatAI/1Cat-vLLM/pull/1161) | One-line startup fix: `main` since 2026-10-09 fails to register the MTP draft model (a JSON report chokes on a set-valued policy field). Needed until merged. | draft |
| [1CatAI/1Cat-vLLM#1166](https://github.com/1CatAI/1Cat-vLLM/pull/1166) | Fused native hyperconnection post-mapping + logits dot for 1-8 tokens (was 8-token-only and off). | draft |
| [1CatAI/1Cat-vLLM#1165](https://github.com/1CatAI/1Cat-vLLM/pull/1165) | KDA output norm through the Triton gated RMSNorm instead of eight eager kernels per layer; bit-identical. ~+6% decode. | draft |
| [turboderp-org/exllamav3#475](https://github.com/turboderp-org/exllamav3/pull/475) | `--tile_order colmajor` in the exllamav3 converter, to make checkpoints like this one | draft |

### Build

```bash
git clone https://github.com/1CatAI/1Cat-vLLM && cd 1Cat-vLLM
git checkout ec60883ad                # tested base (2026-10-10 03:57 +0800); main from d4ce51399 on fails at MTP
                                      # draft-model init (Flash-V100 resources bound only at execution), see #1161
git fetch origin pull/1125/head:pr-1125 pull/1159/head:pr-1159 pull/1161/head:pr-1161 \
  pull/1166/head:pr-1166 pull/1165/head:pr-1165 pull/1102/head:pr-1102
# Cherry-pick, don't merge: #1125, #1159, #1161, #1166 and #1165 are rebased onto a newer main that contains
# d4ce51399, so merging their branches would pull that regression back in. Their own commits apply cleanly here.
git cherry-pick 587fac6eb da13d2d9c   # #1125 (contains #1092)
git cherry-pick 44d292046             # #1159 decode fast paths
git cherry-pick a66fcda49             # #1161 startup fix for MTP
git cherry-pick 74d2cf6da             # #1166 fused mHC post+dot
git cherry-pick 60dd820a7             # #1165 KDA output norm
git merge --no-edit pr-1102           # #1102 is based before d4ce51399; conflicts in CMakeLists.txt and csrc/ops.h
# Both conflicts are two PRs adding lines at the same spot; keep both sides:
sed -i '/^<<<<<<< \|^=======$\|^>>>>>>> /d' CMakeLists.txt csrc/ops.h
git add CMakeLists.txt csrc/ops.h && git commit --no-edit

uv venv --python 3.12 --seed
uv pip install -r requirements/build/cuda.txt --torch-backend=cu128
TORCH_CUDA_ARCH_LIST=7.0 CMAKE_CUDA_ARCHITECTURES=70 MAX_JOBS=10 \
CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14 \
  uv pip install -e . --torch-backend=cu128 --no-build-isolation
```

The full build takes 40-60 minutes. The optional Rust `vllm-server` step needs `protoc` (package
`protobuf-compiler`). Without it that step fails but the wheel still builds.

Check the build (the test deps are not part of the build requirements):
`uv pip install pytest tblib && .venv/bin/python -m pytest tests/kernels/quantization/test_exl3_moe.py tests/models/glm5next/test_sm70_sparse.py` (80 pass).

### Serve

Download the repo, then point `--model` at the local directory (that's the tested path):

```bash
hf download philbert440/GLM-5.3-Flash-EXL3 --local-dir ./GLM-5.3-Flash-EXL3
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0   # else 1.6 GiB is reserved for graph capture and no KV fits at 131K
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model ./GLM-5.3-Flash-EXL3 --served-model-name glm-5.3-flash \
  --trust-remote-code --dtype half --tensor-parallel-size 4 \
  --kv-cache-dtype fp8_e4m3 --max-model-len 131072 --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 4096 --max-num-seqs 1 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy"}' \
  --reasoning-parser deepseek_r1 --tool-call-parser glm47 --enable-auto-tool-choice
```

The startup log should show `EXL3 routed experts: mcg, K=2 ... colmajor tile order` (the loader reorders the
tiles to the kernels' strip-major layout in place, nothing to configure) and
`GLM-5.3-Flash route: SM70 FP16 sparse MLA with packed E4M3FN KV and FP32-softmax decode/verify`.

- **Context:** 131,072 tokens fits at `--gpu-memory-utilization 0.92` with MTP, vision and FP8 KV, with the two
  settings above (`--max-num-seqs`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`); the KV pool is about 1.3 GiB per
  GPU and the weights take about 25.4 GiB.
- **Concurrency:** `--max-num-seqs 4` for up to four users; it still fits at 131K and measured 143-156 tok/s aggregate
  (36-43 per user) with four 512-token greedy requests (measured before #1166/#1165), and 91 tok/s for a single user. Do not leave it unset: vLLM's default of 256 sequences
  makes the MLA prefill workspace one 2,304-token page per sequence (589,824 tokens, 9 GiB at TP4) and the server
  fails to start at 131K context.
- **Prefix cache:** it works in agentic multi-turn use (49-96% hit rates in our service), in 2,304-token
  blocks, a consequence of the hybrid KDA layers. Prompts shorter than one block are never cached.

Verified on our quad: loading this repo through #1102 gives greedy output identical to the repacked reference on
8/8 prompts (no speculation), and the 84K needle test passes. The PRs are drafts under review, so pin the commits you build.
Known-good set (2026-10-10, the one the ~91 tok/s figure was measured on): `main` ec60883ad, #1125 at da13d2d9c
(rebased, contains #1092 as 587fac6eb), #1159 at 44d292046, #1161 at a66fcda49, #1102 at b96d43e5c, #1166 at 74d2cf6da, #1165 at 60dd820a7.
That stack, built as above and served with the command above, measured 91 tok/s decode (1 user, 512-token greedy responses)
on our NVLink quad (re-verified 2026-10-10 from a clean clone with exactly these steps: 91.4-91.9 tok/s, 32.4 ms per pass); averaged over five varied 400-token prompts it does 85 tok/s at 32.7 ms per forward pass with 2.8 accepted
tokens per pass. Under MTP the tok/s of any single prompt depends on its text (draft acceptance), so compare builds by
milliseconds per pass, not by one prompt's tok/s; the same stack without the last two PRs measured 85.7 tok/s, 143-156 tok/s aggregate with four users, and
retrieved a fact planted at the start of a 70K-token prompt (prefill 1,392 tok/s including generation).

## Credits

Quantized with [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT), using the SM70 work in
Frenchy2k1's fork plus local patches. Base model by Z.ai, MIT licensed.
