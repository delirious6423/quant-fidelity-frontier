---
license: cc-by-4.0
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
language:
- en
quantized_by: Anjielon
tags:
- gguf
- llama.cpp
- ternary
- quantization
- mixture-of-experts
- vulkan
- cuda
---

# Thor-T-35B-A3B

<p align="center"><img src="assets/thor-t-logo.png" alt="Thor-T logo" width="168"></p>
<p align="center"><a href="https://huggingface.co/Anjielon/Thor-T-35B-A3B/blob/main/Thor-T-35B-A3B-paper.pdf"><img alt="Paper" src="https://img.shields.io/badge/paper-PDF-b31b1b.svg"></a> <img alt="License CC BY 4.0" src="https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg"> <a href="https://github.com/Anjielon/llama.cpp/tree/thor-t-v1"><img alt="llama.cpp thor-t-v1" src="https://img.shields.io/badge/llama.cpp-thor--t--v1-2a78d6.svg"></a> <a href="https://github.com/Anjielon/fucina"><img alt="Fucina" src="https://img.shields.io/badge/forge-Fucina-555555.svg"></a> <a href="https://huggingface.co/datasets/Anjielon/Thor-T-bench"><img alt="bench data" src="https://img.shields.io/badge/%F0%9F%A4%97%20bench-Thor--T--bench-ffcc4d.svg"></a></p>
<p align="center"><a href="https://huggingface.co/Anjielon/Thor-T-35B-A3B/blob/main/Thor-T-35B-A3B-paper.pdf">Paper (PDF)</a> · <a href="https://github.com/Anjielon/fucina">Fucina (the forge)</a> · <a href="https://github.com/Anjielon/llama.cpp/tree/thor-t-v1">llama.cpp branch <code>thor-t-v1</code></a> · <a href="https://huggingface.co/datasets/Anjielon/Thor-T-bench">Bench data and logs</a></p>

Ternary GGUF files of [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), a 35B hybrid
mixture-of-experts with about 3B active parameters per token, in three sizes plus the root they are built from. Almost
every weight is −1, 0 or +1, stored as one to four ternary planes per tensor, with a **ternary multi-token-prediction
head** inside every file. Built post-training, on one home APU (AMD Ryzen AI Max+ 395, 128 GB), with the open forge
[Fucina](https://github.com/Anjielon/fucina); the method and every number are in the [paper](https://huggingface.co/Anjielon/Thor-T-35B-A3B/blob/main/Thor-T-35B-A3B-paper.pdf).

The files need the llama.cpp fork [`Anjielon/llama.cpp`, tag `thor-t-v1`](https://github.com/Anjielon/llama.cpp/tree/thor-t-v1) (Vulkan, CUDA, CPU);
they do not load in upstream llama.cpp yet.

## Files

| file | GiB | bits / param | full MMLU, % of the Q8_0 teacher | tokens/s, APU, MTP on ¹ | tokens/s, 8 GB laptop, MTP on ² | use it for |
|---|---:|---:|---:|---:|---:|---|
| [`Thor-T-35B-A3B-M.gguf`](#m) | 12.19 | 2.949 | 99.34 | 77.53 | 39.2 | **best quality**: level with Q4_K_M at 60 % of its bytes |
| [`Thor-T-35B-A3B-S.gguf`](#s) | 8.82 | 2.135 | 96.59 | 84.41 | 53.7 | balanced: above IQ2_XXS at about its size |
| [`Thor-T-35B-A3B-XS.gguf`](#xs) | 8.04 | 1.946 | 89.70 | 95.42 | 66.1 | smallest and fastest: every linear layer ternary |
| [`Thor-T-35B-A3B-root.gguf`](#root) | 8.24 | 1.993 | 92.47 | 87.78 | 61.3 | research checkpoint: the frozen multi-plane root the three sizes start from |

¹ Launch engine (Vulkan, AMD APU), one draft token, 5 prompts × 256 tokens, greedy; the root on the same engine, six
repetitions. ² RTX 4060 Laptop (8 GB), CUDA, part of the experts in system RAM (`-ncmoe` XS 14, S 18, M 22), five
repetitions. Details and the runs without the head below.

Download one file only:

```bash
pip install -U huggingface_hub
hf download Anjielon/Thor-T-35B-A3B --include "Thor-T-35B-A3B-M.gguf" --local-dir .
```

`replay/` holds the inputs of the public replay of the recipe (the packed root without the MTP head and the dense PTQ
inputs, read by `fucina replay examples/thor/recipe_c.yaml`); they are not needed to run the models.

## Results at a glance

<p align="center"><img src="assets/frontier.png" alt="Full MMLU as a percentage of the Q8_0 teacher against file size" width="760"></p>

Full MMLU test (14,042 questions, zero-shot, scored by log-probability), every file on the same Vulkan build of our
engine (the *launch engine*, commit `4ed3ec488`, AMD Ryzen AI Max+ 395 APU), as a percentage of our Q8_0 conversion of
the base model (the teacher). Q2_K, Q4_K_M and IQ2_XXS are the public files of the same teacher built with a public
importance matrix; IQ1_M was built by us with the same matrix; the naive TQ1_0 file is one ternary plane per tensor
without our recipe.

| file | GiB | full MMLU | % of teacher | paired on the same engine (points; right only for ours / only for the other; exact McNemar) |
|---|---:|---:|---:|---|
| teacher Q8_0 | 35.21 | 80.59 | 100 | reference |
| Q4_K_M | 20.36 | 80.16 | 99.46 |  |
| **M** | 12.19 | **80.06** | **99.34** | vs Q2_K +2.38 pts (95 % CI +1.77 to +2.98; 1,142/808, p < 1e-10) · vs Q4_K_M −0.10 pts (by-subject CI −1.20 to +1.19; 831/845, p = 0.75) · vs teacher −0.53 pts (CI −1.10 to +0.02; 748/823, p = 0.062) |
| Q2_K | 12.19 | 77.68 | 96.39 |  |
| IQ2_XXS | 9.55 | 71.76 | 89.03 |  |
| **S** | 8.82 | **77.85** | **96.59** | vs IQ2_XXS +6.09 pts (1,768/913, p < 1e-10) · vs IQ1_M (4,267/939, p < 1e-10) · vs teacher −2.75 pts (CI −3.33 to −2.17; 704/1090, p < 1e-10) |
| root (research checkpoint) | 8.24 | 74.53 | 92.47 |  |
| IQ1_M | 8.12 | 54.14 | 67.18 |  |
| **XS** | 8.04 | **72.29** | **89.70** | vs naive TQ1_0 (7453/949, p < 1e-10) · vs teacher −8.30 pts (CI −9.00 to −7.62; 726/1892, p < 1e-10) |
| naive TQ1_0 (one plane) | 7.87 | 25.97 | 32.23 |  |

**Read with the caveat above:** M's and S's development used the MMLU test set, so their full-test scores are not
independent estimates and the intervals and p-values are nominal; there is no equivalence claim (details in
*Read this first* and *Selection on MMLU* on the [M](#m) and [S](#s) cards).

<p align="center"><img src="assets/bytes.png" alt="Tensor bytes by family for XS, root, S and M" width="720"></p>

## How to run it

The release engine is the tag `thor-t-v1` of [Anjielon/llama.cpp](https://github.com/Anjielon/llama.cpp)
(llama.cpp at upstream `785322d17` plus the ternary types, packed multi-plane types and sparse expert planes, with
Vulkan, CUDA and CPU kernels; ROCm compiles the CUDA kernels with HIP). `docs/ternary.md` in that tag lists the types,
the requirements and every environment switch.

```bash
git clone --branch thor-t-v1 https://github.com/Anjielon/llama.cpp
cd llama.cpp

# Vulkan (AMD, Intel; needs VK_KHR_shader_integer_dot_product for the int8 paths)
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
# CUDA:  cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
# ROCm:  cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=<your gfx target> -DCMAKE_BUILD_TYPE=Release
# CPU:   cmake -B build
cmake --build build -j

./build/bin/test-backend-ops test -o MUL_MAT
./build/bin/test-backend-ops test -o MUL_MAT_ID
```

### Recommended settings

```bash
# whole model on a GPU with enough memory
GGML_VK_TQ1R_Q8=1 ./build/bin/llama-server -m Thor-T-35B-A3B-M.gguf -ngl 999 -c 32768 --jinja

# 8 GB GPU: keep the experts of the first 22 layers in host memory
./build/bin/llama-server -m Thor-T-35B-A3B-M.gguf -ngl 999 -ncmoe 22 -c 32768 --jinja

# speculative decoding with the built-in MTP head, one draft token
GGML_VK_TQ1R_Q8=1 ./build/bin/llama-server -m Thor-T-35B-A3B-M.gguf -ngl 999 -c 32768 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 1
```

- **`GGML_VK_TQ1R_Q8=1`** (Vulkan only, off by default) selects int8 activations in the matrix-vector products of
  the rotated types; M was evaluated and benchmarked with it on. It needs `VK_KHR_shader_integer_dot_product`. The
  setting was chosen by decode speed, not by accuracy, and the int8 and float paths are not bit-identical. In one
  session of 7 November (binary `5d2e382a9`) M decoded at 55.16 tokens/s with int8 and 53.15 with float activations.

- **`-ncmoe 22`** is the smallest value we measured that fits M in 8 GB on an RTX 4060 Laptop GPU.
- **Reasoning.** The chat template stored in the file enables the model's reasoning by default; every generative
  number below was measured with reasoning on and `--no-reasoning-preserve`. `--reasoning-budget 0` turns it off.
- **MTP head.** The head was distilled from the trunk of XS and spliced unchanged into M; it was not re-aligned to the trunk of M. On the launch engine (Vulkan, APU; 5 prompts × 256 tokens, greedy, one request, int8
  flag on as released) one draft token gives 77.53 against 68.49 tokens/s (+13.2 %, 13.2–15.0 % on four earlier builds measured the same day) at 70.1 % acceptance; two draft tokens give 73.25 tokens/s at 50.0 %. Outputs with drafts are not bit-identical to plain greedy decoding (the verification
  batch changes the numerics). It is not used by any accuracy number. An earlier CUDA measurement on the RTX 4060 laptop is in the paper's appendix.

### Quick reference: one line per file

| file | Vulkan int8 switch (as evaluated) | 8 GB GPU (smallest measured fit, RTX 4060 Laptop) | MTP head |
|---|---|---|---|
| XS | `GGML_VK_TQ1R_Q8=0` | `-ncmoe 14` | `--spec-type draft-mtp --spec-draft-n-max 1` |
| S | `GGML_VK_TQ1R_Q8=1` | `-ncmoe 18` | `--spec-type draft-mtp --spec-draft-n-max 1` |
| M | `GGML_VK_TQ1R_Q8=1` | `-ncmoe 22` | `--spec-type draft-mtp --spec-draft-n-max 1` |
| root | `GGML_VK_TQ1R_Q8=1` | not measured | `--spec-type draft-mtp --spec-draft-n-max 1` |

The server flags of the 8 GB laptop speed runs of M (one user, MTP head on; CUDA build of the same branch; the
requests were raw completions, so no chat template was involved):

```bash
./build/bin/llama-server -m Thor-T-35B-A3B-M.gguf -ngl 999 -ncmoe 22 -c 4096 -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 1
```

- **Switches are plain ASCII double dashes** (`--spec-type`, `--jinja`); copy them from the code blocks, not from
  rendered prose.
- **Sampling.** Every number on this card is greedy (temperature 0). We did not tune sampling settings; the file
  stores the base model's chat template (use `--jinja`), which turns reasoning on by default.
- **Context.** Commands use `-c 32768` as in *Recommended settings*; the laptop speed runs used `-c 4096` per user.
  Long-context behaviour was not evaluated (see *Limitations*).
- **One user or many.** Turn the MTP head **on** for a single interactive user and **off** for a server shared by
  several users: on the 8 GB laptop it raises the speed of one request and lowers total throughput at 2 and 4
  concurrent requests (measured below).

## Results

### Other likelihood-scored tasks

| task | XS | S | M | Q2_K ¹ | IQ2_XXS ¹ | root ¹ |
|---|---:|---:|---:|---:|---:|---:|
| MMLU-Pro (letter likelihood) | 75.72 | 93.14 | 99.36 | pending | pending | pending |
| HellaSwag | 88.15 | 94.19 | 98.42 | 97.41 | 94.01 | 92.41 |
| Winogrande | 94.78 | 102.61 | 104.20 | 100.34 | 97.17 | 104.08 |
| ARC-Challenge | 98.86 | 105.38 | 104.08 | 92.83 | 90.06 | 99.02 |
| ARC-Easy | 111.11 | 113.33 | 112.28 | 98.40 | 100.37 | 106.54 |
| PIQA | 95.08 | 98.58 | 100.13 | 99.45 | 98.85 | 97.57 |
| BoolQ | 99.56 | 104.94 | 103.12 | 85.72 | 85.35 | 103.53 |
| OpenBookQA | 93.98 | 97.22 | 104.17 | 104.63 | 100.46 | 101.85 |
| TruthfulQA-MC1 | 85.71 | 91.19 | 97.87 | 93.32 | 86.62 | 93.62 |

### The MTP head: speed on the APU

| file | no draft | 1 draft token | gain (four earlier builds) | acceptance | 2 draft tokens (acceptance) |
|---|---:|---:|---:|---:|---:|
| XS | 91.29 | **95.42** | +4.5 % (2.8–5.7 %) | 72.7 % | 89.58 (56.6 %) |
| S | 72.30 | **84.41** | +16.8 % (15.0–17.0 %) | 75.1 % | 83.99 (58.4 %) |
| M | 68.49 | **77.53** | +13.2 % (13.2–15.0 %) | 70.1 % | 73.25 (50.0 %) |

### On an 8 GB laptop GPU

| file | `-ncmoe` | no draft | 1 draft token | gain | acceptance | 2 draft tokens |
|---|---:|---:|---:|---:|---:|---:|
| XS | 14 | 63.05 ± 3.66 | **66.15 ± 3.67** | +4.9 % | 75.4 % | 63.42 ± 2.17 (56.7 %) |
| S | 18 | 46.73 ± 1.86 | **53.69 ± 2.23** | +14.9 % | 73.9 % | 53.10 ± 1.24 (57.4 %) |
| M | 22 | 35.99 ± 1.93 | **39.17 ± 1.13** | +8.8 % | 72.2 % | 35.82 ± 1.06 (51.0 %) |
| root + our head | 18 | 54.48 ± 0.18 | **61.30 ± 0.29** | +12.5 % | 72.6 % | 58.41 ± 0.23 (53.7 %) |

In 8 of the 9 XS/S/M configurations the first repetition is the slowest (likely a cold start); all five repetitions are kept in the means.

## The files in detail

### XS

| file | bytes | GiB | bits / parameter | sha256 |
|---|---:|---:|---:|---|
| `Thor-T-35B-A3B-XS.gguf` | 8,635,663,904 | 8.04 | 1.946 | `86c2684b19be716bbaedb44341b236c64ff0fe71ef4a74fced02fde5bd027d7c` |

- **Base and teacher.** Base model Ornith-1.5-35B-A3B (MIT licence). The teacher used for every distillation step and
  as the reference of every percentage below is our Q8_0 conversion of it (37,802,149,216 bytes, 35.21 GiB, sha256
  `e1425b9cba1c7514989cefe944fa9f4a68c34f480bcded2eec3549f75eea0fe0`), redistributed in
  [Anjielon/Thor-T-bench](https://huggingface.co/datasets/Anjielon/Thor-T-bench).
- **Format.** Every linear layer of the 40 main blocks, except the F32 routers, is **one plane** of ternary codes
  {-1, 0, +1} with fp16 group scales. Two storage types are used: `TQ1_0R_1024` (ours: a fixed block-Hadamard rotation
  of 1024 inputs with fixed signs, one fp16 scale per 128 rotated weights; layout in `ggml/src/ggml-common.h` of the
  release tag) and upstream `TQ1_0` (the routed and shared
  expert down projections, whose rows are shorter than one rotation block, and the two down projections of the MTP
  block). The token embedding and the output head are Q8_0, copied from the teacher; norms, routers and SSM vectors
  are F32. Tensor count by type: 359 `TQ1_0R_1024`, 82 `TQ1_0`, 2 `Q8_0`, 310 `F32`.

- **Where the bytes are** (GiB): embedding + head 1.01 (12.5 % of the tensor bytes), dense projections 0.29, routed
  experts 6.48, MTP block 0.17, float tensors 0.08.
- **What it descends from.** XS is built from the last checkpoint (called C in the paper) of a chain that starts from a frozen
  multi-plane ternary checkpoint whose experts come from a forge and whose whole student, then dense tensors, were trained end to end
  ([Thor-T-35B-A3B-root](https://huggingface.co/Anjielon/Thor-T-35B-A3B)): Hessian-weighted re-quantizations,
  low-rank corrections distilled in the ternary forward pass and folded into the existing codes and scales, on-policy
  distillation, and a final distillation of the deployable parameters. It is distillation on top of a trained
  checkpoint, **not training-free post-training quantization**, and should not be compared with training-free PTQ as
  if it were. XS replaces the earlier MTP block of C with a new multi-token-prediction head; its other 733 tensors are
  byte-identical to C, and the 20 differing tensors are all in block 40.
- **The int8 flag.** All accuracy and generative numbers of XS were measured with `GGML_VK_TQ1R_Q8=0` (float
  activations in the matrix-vector products of the rotated type). See *Recommended settings*.

##### What converting the root to one plane costs

XS is smaller than the root it comes from (8,635,663,904 against 8,846,935,776 bytes for the packed root, 8.04 against
8.24 GiB) and less accurate. On the full MMLU test, both on the Vulkan launch engine, XS scores 72.29 against 74.53 for
the root, −2.24 points (926/1240 questions right only for XS / only for the root, p < 1e-10, exact McNemar, same
engine). On the 2,000 development questions on which XS's line was selected the difference is −1.30 points (131/157,
p = 0.14); on the other 12,042 test questions it is −2.39 points (795/1,083, p < 1e-10). **The root is the more
accurate file; what XS buys, for 2.4 % fewer bytes, is the single-plane format.** The root ran with the int8 flag on
(`GGML_VK_TQ1R_Q8=1`), XS with it off, so the comparison does not isolate the conversion (activation paths differ). In the development history, on
earlier builds, the root scored 74.60 and C (the trunk of XS) 73.15 on the development subset.

##### Bit-exact conversion to PrismML's PTQ1_0

As a type, `TQ1_0R_1024` is numerically equivalent to PrismML's `PTQ1_0` with Hadamard-rotation metadata
([Ternary-Bonsai-2](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf); llama.cpp PRs
[#29672](https://github.com/ggml-org/llama.cpp/pull/29672) and
[#29676](https://github.com/ggml-org/llama.cpp/pull/29676), open at the time of writing): the same alphabet, the same
block-1024 Sylvester-Hadamard rotation with fixed signs, one fp16 scale per 128 rotated weights, 1.75 bits per
weight; the layouts differ. **PTQ1_0 predates our type.** We converted 298 `TQ1_0R_1024` tensors of XS bit-exactly
(23,404,216,320 weights, every block identical in scale times codes): the 290 tensors upstream accepts as rotated, plus
8 tensors of the MTP block, converted but **not** listed as rotated (upstream refuses a file that lists them). The 61
remaining rotated tensors (`ssm_alpha`, `ssm_beta` and the MTP `nextn.eh_proj`) are of kinds upstream does not accept
as rotated; they become dense F32 tensors with the same effective weights, which adds 46.6 MB (+0.54 %) to the file.

What this does and does not mean:
- the stored weights are verified bit-exact; the runs below are paired measurements, not an equivalence test;
- **upstream CPU.** On an upstream CPU build (llama.cpp master with the unmerged PR #29676, which contains #29672) XS
  converted to PTQ1_0 scores 72.80 on MMLU-2000 against 72.90 for XS on the CPU of our release branch (questions right
  only for PTQ1_0 / only for XS 20/22, p = 0.88; same choice on 1,941 of 2,000); its KL divergence to XS on the CPU is
  0.005348 (same top token 96.373 %). PTQ1_0 support remains unmerged upstream, so this run needs the open pull
  request;
- **PrismML's fork.** On that fork (CUDA, RTX 4060 laptop) the converted file scores 72.65 on MMLU-2000 against 73.00
  for XS on our CUDA build on the same laptop (questions right only for PTQ1_0 / only for XS 16/23, p = 0.34; same
  choice on 1,948 of 2,000); re-run alone on the laptop with no other task active, it gave the same answers question by question as a first
  run that had overlapped a scheduled task.
  Its KL divergence to XS is 0.004852 on CUDA and 0.005032 on Vulkan (Radeon 8060S, against
  XS on the launch engine). The fork computes the activation side differently (rotation as a separate graph
  operation, its own activation quantization), so outputs are not bit-identical to ours;

- the fork decodes it more slowly than our kernels decode XS: 41.22 against 64.05 tokens/s on CUDA with the expert
  tensors of 14 layers on the CPU (in the session that overlapped a scheduled task: the clean re-run repeated the MMLU
  only, not the speed; XS itself ranged from
  64.05 to 72.03 tokens/s over four sessions with the same binary), 85.69 against 92.88 on Vulkan;
- **MTP head.** In the converted files the MTP head stays ternary, with the same trits as XS, but PrismML's fork and
  upstream do not yet apply the rotation to the MTP block (its 8 tensors are stored rotated but not listed as rotated,
  because upstream refuses a file that lists them), so those runtimes do not use the head; this is why the conversions
  are not released;
- S and M cannot be converted: their multi-plane tensors are sums of planes, which are not ternary.

The PTQ1_0 conversion script and its verification (`converti_tq1r_ptq1.py`) are published in the dataset Anjielon/Thor-T-bench (scripts/ptq1/).
Those copies are **rewritten** from the scripts that ran: machine paths were replaced by four environment variables,
`THOR_ENGINE_SRC` (source tree of the release engine, for its `gguf-py`), `THOR_ENGINE_BIN` (its build's `bin/` with
`libggml-base.so`), `THOR_UPSTREAM_BIN` (an upstream build's `bin/`, for `verifica_c_upstream.py`) and `THOR_XS_GGUF`
(path of `Thor-T-35B-A3B-XS.gguf`, for the verification scripts). The measured Q2_0 file was reproduced bit-identically
with these shipped copies and the documented command (`python3 -I converti_tq1r_ptq1.py XS.gguf OUT.gguf --target q2_0`,
10 Nov: sha256 `47afdf9a...`, 10,145,049,792 bytes; log `logs/forge/stato/q2_0_upstream_1110/conv_bundle_rigenera.log`).

##### A route on stock llama.cpp: Q2_0 with rotation metadata

A route on stock llama.cpp exists today. Upstream applies the Hadamard rotation in the graph by tensor name for this
architecture, whatever the weight type ([llama.cpp PR #29600](https://github.com/ggml-org/llama.cpp/pull/29600),
merged, release b11556); this is how the rotated `Q2_0` file of Ternary-Bonsai-2 runs. **Two layouts share the name
`Q2_0`:** upstream's has 64 weights per block, one fp16 scale and 2-bit codes for {-1, 0, 1, 2}, 2.25 bits per weight;
PrismML's fork has a different 128-weight, 2.125-bit layout under the same name. This route uses **upstream's**
`Q2_0`, which holds a ternary group exactly with the same scale bits. Therefore:
- the 290 tensors upstream accepts as rotated, and the 8 MTP tensors kept rotated but not listed as rotated, are
  stored bit-exactly as `Q2_0` with the same Hadamard metadata (298 tensors);
- the 61 tensors upstream does not rotate (`ssm_alpha`, `ssm_beta`, the MTP `nextn.eh_proj`) cannot keep their
  rotated-basis trits in any upstream type and are stored dense in F32 (+46.6 MB);
- the `TQ1_0` expert downs are an upstream type and need no conversion. Upstream CUDA has no matrix-product support
  for `TQ1_0` (it is not among the types `ggml-cuda` accepts for `MUL_MAT` at the commit we ran), so on CUDA the loader
  keeps the 80 `TQ1_0` downs in host memory and the CPU backend computes them; our logs do not list the placement,
  this follows from the source.

**The conversion is done:** every block of the 298 converted tensors is bit-identical in d·q, and the file is
10,145,049,792 bytes (9.45 GiB; at 2.25 instead of 1.75 bits per weight on 23.4e9 weights, about 1.36 GiB more than
XS, plus the F32 tensors).

**Reproducing the file.** `python3 scripts/ptq1/converti_tq1r_ptq1.py Thor-T-35B-A3B-XS.gguf
Thor-T-35B-A3B-XS-Q2_0.gguf --target q2_0` (defaults: the 8 MTP tensors stay rotated and unlisted) writes the file
measured, sha256 `47afdf9aecc0fbd8249e8cc54010a349018914673dd6a11e800d8330e408eb57` (reproduced bit-identically with
the shipped copy on 10 Nov); `--verify-only` re-checks every block
(our outputs `scripts/ptq1/verifica_completa_q2_0.out` and `verifica_completa.out` are being regenerated with the shipped
converter, whose version differs from the one that wrote them). The scripts need two environment variables:
`THOR_ENGINE_SRC` (source tree of the release engine, for its `gguf-py`) and `THOR_ENGINE_BIN` (its build's `bin/`
with `libggml-base.so`).

**It runs on stock llama.cpp.** On upstream master `9d340ec` (only a patch that prints the per-question records), with
CUDA on the RTX 4060 laptop (`-ngl 999 -ncmoe 14`): KL divergence 0.004667 against XS on our CUDA build (same top token
96.96 %; perplexity 8.9269 against 8.8664 on 4×512 tokens of WikiText-2); MMLU-2000 72.65 against 73.00 for XS on the
same laptop (questions right only for Q2_0 / only for XS 18/25, p = 0.36; same choice on 1,938 of 2,000; not an
equivalence test); in the same session it decodes at 46.17 tokens/s and prefills at 243.40, against 70.45 and 1016.52
for XS with our kernels, a gap that includes the `TQ1_0` downs computed on the CPU and the larger file (9.45 against 8.04 GiB) at the same expert
placement on an 8 GB GPU. No matched floor of XS against itself was measured at these settings (`-ncmoe 14`, 4 chunks), so the 0.004667 is
not split between the conversion and the activation path. The converter copies the file-type key of XS, so upstream labels the file "TQ1_0 - 1.69 bpw ternary"; the label is
the header's `general.file_type`, while the tensors are read by their own types. The run on stock llama.cpp with
Vulkan on the APU is pending. As in the PTQ1_0 file, the MTP head is not usable in this file (the 8 MTP tensors stay rotated
and unlisted by default; the F16 option is experimental and unmeasured).

Upstream PTQ1_0 support remains unmerged, and the stock route costs 1.41 GiB more than XS; what XS buys today over its root is the
single-plane format, not an upstream deployment advantage.

This repository ships one file:

| file | size | sha256 | runs on |
|---|---|---|---|
| `Thor-T-35B-A3B-XS.gguf` | 8,635,663,904 B (8.04 GiB) | `86c2684b19be716bbaedb44341b236c64ff0fe71ef4a74fced02fde5bd027d7c` | our llama.cpp branch `thor-t-v1` (CPU, Vulkan, CUDA), with its ternary MTP head |

```bash
# git clone --branch thor-t-v1 https://github.com/Anjielon/llama.cpp ; build as its README says; see Recommended settings
./build/bin/llama-server -m Thor-T-35B-A3B-XS.gguf -ngl 999 -c 32768 --jinja
```

##### Conversions (not released)

The trits of XS fit PrismML's `PTQ1_0` and upstream `Q2_0` bit-exactly (every block of the 298 converted tensors
identical in d·q). Measured on MMLU-2000 against XS, paired (not equivalence tests; the CUDA comparisons cross builds):

- `PTQ1_0` (sha256 `8db9e811...`, 8,682,286,272 B, 8.09 GiB): 72.65 vs 73.00 on PrismML's fork with CUDA (16/23,
  p = 0.34); 72.80 vs 72.90 on upstream CPU with the open PR #29676 (20/22, p = 0.88);
- `Q2_0` (sha256 `47afdf9a...`, 10,145,049,792 B, 9.45 GiB): 72.65 vs 73.00 on stock llama.cpp master `9d340ec` with
  CUDA (18/25, p = 0.36).

Neither file is released: every released file must carry our working ternary MTP head, and PrismML's fork and
upstream do not apply the rotation to the MTP block, so in these formats the head is not used. They will be released
when our ternary MTP head runs there. A first step is measured: with the MTP tensors kept rotated and listed (and
`eh_proj` ternary; file 8,650,567,136 B, sha256 `7f45ac12…`) and a two-commit local patch to PrismML's fork (not pushed),
our head works in PTQ1_0 without retraining — on the RTX 4060 in one session, acceptance 75.0 % against 75.4 % for XS
on our engine, one draft token 48.23 against 42.21 tokens/s (1.14×; XS on our engine 1.05×). The shipped converter writes both with the commands below; both files were
regenerated bit-identically with the shipped copy on 10 Nov (logs `logs/forge/stato/q2_0_upstream_1110/conv_bundle_*.log`;
the converter's own comment gives the PTQ1_0 size as 8,682,286,528 B, the file is 8,682,286,272 B):

```bash
python3 scripts/ptq1/converti_tq1r_ptq1.py Thor-T-35B-A3B-XS.gguf Thor-T-35B-A3B-XS-Q2_0.gguf --target q2_0
python3 scripts/ptq1/converti_tq1r_ptq1.py Thor-T-35B-A3B-XS.gguf Thor-T-35B-A3B-XS-PTQ1_0.gguf
```

### S

| file | bytes | GiB | bits / parameter | sha256 |
|---|---:|---:|---:|---|
| `Thor-T-35B-A3B-S.gguf` | 9,474,634,400 | 8.82 | 2.135 | `e222f487397475a7a333d87ac6602984879485cac86e7ee6f887b35382e05a8f` |

- **Base and teacher.** Base model Ornith-1.5-35B-A3B (MIT licence). The teacher of every distillation step and the
  reference of every percentage below is our Q8_0 conversion of it (37,802,149,216 bytes, 35.21 GiB, sha256
  `e1425b9cba1c7514989cefe944fa9f4a68c34f480bcded2eec3549f75eea0fe0`), redistributed in
  [Anjielon/Thor-T-bench](https://huggingface.co/datasets/Anjielon/Thor-T-bench).
- **Format.** A tensor with K planes stores the sum of K ternary code planes {-1, 0, +1}, each with its own fp16
  group scales. S uses:
  - **packed dense types** `TQ1_0R_1024_X3/_X4` and `TQ1_0_X3/_X4` (three or four planes on every dense tensor of the
    main network; the token embedding and the output head in `TQ1_0R_1024_X3`); packing is lossless, the bytes are
    those of the separate planes;
  - **sparse second planes of the routed experts**: separate tensors `ffn_{gate,up,down}_exps2.weight` plus an F32
    slot map, present for 806 of the 10,240 (layer, expert) pairs, in 36 of the 40 layers; experts without a second
    plane cost no arithmetic;
  - single-plane `TQ1_0R_1024` / `TQ1_0` elsewhere, and F32 for norms, routers, SSM vectors, `ssm_alpha` and
    `ssm_beta`.

  Tensor count by type: 161 `TQ1_0R_1024`, 131 `TQ1_0R_1024_X3`, 81 `TQ1_0R_1024_X4`, 78 `TQ1_0`, 18 `TQ1_0_X3`,
  22 `TQ1_0_X4`, 478 `F32`. The header sets `general.file_type` to 36 (`MOSTLY_TQ1_0`), the closest existing label;
  tools that print a parameter count report more than the logical parameters, because they count the second-plane
  slots and the maps.

- **Where the bytes are** (GiB): embedding + head 0.62 (7.1 % of the tensor bytes), dense projections 0.93, routed
  experts 6.99, MTP block 0.17, float tensors 0.10. Against XS (MB): embedding + head −413, dense +688, experts +548:
  the bytes move in both directions, so the difference from XS is not just the price of added planes.

- **What it descends from.** S does **not** go through the single-plane chain of XS. It starts from the same frozen
  multi-plane checkpoint ([Thor-T-35B-A3B-root](https://huggingface.co/Anjielon/Thor-T-35B-A3B), whose experts
  come from a forge and whose whole student, then dense tensors, were trained end to end), re-allocates planes under a byte cap by an estimated change of the KL to the
  teacher, receives an expert correction folded into its planes, a scale factor on its single-plane experts, a
  per-family dense re-solve, and a final distillation of the deployable parameters (100 steps). The MTP head of XS is
  spliced in unchanged. It is distillation on top of a trained checkpoint, **not training-free post-training
  quantization**.
- **Released versus measured file.** The released file differs from the benchmarked one only in its metadata header
  (96 bytes: names, file type, one descriptive key); all 969 tensors are byte-identical. The benchmarked copy (`S-packed-testaXS.gguf`) has sha256
  `14a8517dc7fddae66a9c8d12bf3e974b4f9e3b846f2de5eedd6f4a962f5edb99`; the copy measured for C4 on the laptop has the same sha256.
- **The int8 flag.** All accuracy and generative numbers of S were measured with `GGML_VK_TQ1R_Q8=1`.
- **No PTQ1_0 conversion.** Unlike XS, S cannot be converted to PrismML's `PTQ1_0`: a sum of planes is not ternary.

### M

| file | bytes | GiB | bits / parameter | sha256 |
|---|---:|---:|---:|---|
| `Thor-T-35B-A3B-M.gguf` | 13,088,098,592 | 12.19 | 2.949 | `6f56858975fc3bc847f7fcf1a7bab3ac51764e2ec9c26adb82b4e7884bdd93f8` |

M is 2,318,976 bytes (0.018 %) larger than the public Q2_K file of the same teacher (13,085,779,616 bytes) and has
59.9 % of the bytes of Q4_K_M.

- **Base and teacher.** Base model Ornith-1.5-35B-A3B (MIT licence). The teacher of every distillation step and the
  reference of every percentage below is our Q8_0 conversion of it (37,802,149,216 bytes, 35.21 GiB, sha256
  `e1425b9cba1c7514989cefe944fa9f4a68c34f480bcded2eec3549f75eea0fe0`), redistributed in
  [Anjielon/Thor-T-bench](https://huggingface.co/datasets/Anjielon/Thor-T-bench).
- **Format.** A tensor with K planes stores the sum of K ternary code planes {-1, 0, +1}, each with its own fp16
  group scales. M uses:
  - **packed dense types** `TQ1_0R_1024_X2/_X3/_X4` and `TQ1_0_X2/_X3/_X4` (attention and Gated DeltaNet projections
    with three or four planes, shared-expert projections with two to four; the token embedding and the output head in
    `TQ1_0R_1024_X3`); packing is lossless, the bytes are those of the separate planes;
  - **sparse second planes of the routed experts**: separate tensors `ffn_{gate,up,down}_exps2.weight` plus an F32
    slot map, present for 6,122 of the 10,240 (layer, expert) pairs, in all 40 layers; experts without a second plane
    cost no arithmetic;
  - single-plane `TQ1_0R_1024` / `TQ1_0` elsewhere, and F32 for norms, routers, SSM vectors, `ssm_alpha` and
    `ssm_beta`.

  Tensor count by type: 169 `TQ1_0R_1024`, 9 `TQ1_0R_1024_X2`, 126 `TQ1_0R_1024_X3`, 77 `TQ1_0R_1024_X4`,
  82 `TQ1_0`, 4 `TQ1_0_X2`, 17 `TQ1_0_X3`, 19 `TQ1_0_X4`, 490 `F32`. The header sets `general.file_type` to 36
  (`MOSTLY_TQ1_0`), the closest existing label; tools that print a parameter count report more than the logical
  parameters, because they count the second-plane slots and the maps.

- **Where the bytes are** (GiB): embedding + head 0.62 (5.1 % of the tensor bytes), dense projections 0.93, routed
  experts 10.36, MTP block 0.17, float tensors 0.10.
- **What it descends from.** M does **not** go through the single-plane chain of XS. It starts from the same frozen
  multi-plane checkpoint ([Thor-T-35B-A3B-root](https://huggingface.co/Anjielon/Thor-T-35B-A3B), whose experts
  come from a forge and whose whole student, then dense tensors, were trained end to end), re-allocates planes under a byte cap by an estimated change of the KL to the
  teacher, receives one expert correction folded into its planes, and a final distillation of the deployable
  parameters (50 steps). The MTP head of XS is spliced in unchanged. It is distillation on top of a trained checkpoint,
  **not training-free post-training quantization**.
- **Released versus measured file.** The released file differs from the benchmarked one only in its metadata header
  (96 bytes: names, file type, one descriptive key); all 993 tensors are byte-identical. The benchmarked copy (`Thor-T-M-packed-testaXS.gguf`) has sha256
  `49183e40a2480c56322d5dd999273a143b707854745eefa843d89e20e6d135e3`; the copy measured for C4 on the laptop has the same sha256.
- **The int8 flag.** All accuracy and generative numbers of M were measured with `GGML_VK_TQ1R_Q8=1`.
- **No PTQ1_0 conversion.** Unlike XS, M cannot be converted to PrismML's `PTQ1_0`: a sum of planes is not ternary.

### root

| file | bytes | GiB | bits / parameter | sha256 |
|---|---:|---:|---:|---|
| `Thor-T-35B-A3B-root.gguf` (released: the root with our ternary MTP head; identical to the measured copy `25fb7303…` except `general.file_type` = 36, as in XS/S/M) | 8,851,425,088 | 8.24 | — | `f761a7277e91b270be5abf240c75f60609779c36d61cb968c53b42fbf1defdf5` |
| `Thor-T-R2B3m-packed.gguf` (the root as measured and as replay input; earlier MTP block) | 8,846,935,776 | 8.24 | 1.993 | `e40cffd480091a8b11ee1ea63434e4d8440fc5af80ed8a3b443ebc6f3e4bc9fd` |

- **What it is.** A ternary checkpoint of the base model whose dense tensors carry three ternary planes and whose
  routed experts have one plane, with fp16 group scales. Its routed experts come from a sliding-window forge (to
  25 Sep) whose steps, tokens and hardware are not recorded. After it, the run register records at least **42 hours of
  end-to-end GPU training on the APU** against the Q8_0 teacher: first of the whole ternary student (EB2 and EB3,
  26-28 Sep, at least 2.4 M training tokens), then of its dense ternary tensors after their re-ternarization (TT2,
  28-29 Sep); then a few CPU hours added a second and third dense plane and a three-plane embedding and head (29-30
  Sep). The stages, register rows and hours are in the paper's appendix table "root-cost". It **cannot be regenerated
  from a single configuration**, and its training is closer to quantization-aware training of the whole model than the
  light steps that follow it.
- **Packed.** In our run the root was a GGUF with each plane stored as a separate tensor. That layout does not load on
  the release engine, so it is released *packed* into the release types: every family with K planes is one tensor of
  type `TQ1_0R_1024_X{K}` / `TQ1_0_X{K}`, whose block holds the K plane blocks back to back. Packing is lossless and
  the tensor data are those of the separate-plane original. Tensor count by type: 80 `TQ1_0R_1024`,
  212 `TQ1_0R_1024_X3`, 51 `TQ1_0`, 40 `TQ1_0_X3`, 370 `F32`; no sparse second expert planes. The ungrafted packed
  root and the measured copy with our MTP head carry `general.file_type` 19, which tools print as `IQ2_XXS` (they are
  **not** IQ2_XXS); the released file with our head (sha256 `f761a727…`) carries 36, as XS, S and M do. The tensor
  data are identical.

- **Where the bytes are** (GiB): embedding + head 0.62 (7.6 % of the tensor bytes; three planes), dense projections
  0.86, routed experts 6.48, MTP block 0.17, float tensors 0.10.
- **Our ternary MTP head.** The released file carries our ternary MTP head (the head of XS, also used by S and M),
  grafted onto the root: every projection weight of the head is ternary; norms, the router and the shared-expert gate
  stay F32. The trunk is untouched (all 252 packed trunk tensors byte-identical; on the launch engine its logits
  against the ungrafted root: mean KL divergence ≈ 0, same top token 100.00 %). On the RTX 4060 laptop (our CUDA
  build, `-ngl 999 -ncmoe 14`, one session, 5 prompts × 256 tokens, greedy) one draft token raises decode speed from
  58.24 to 65.42 tokens/s (+12.3 %) at 70.7 % acceptance (two drafts: 63.47 tokens/s, 56.4 %), against 56.38 → 60.76
  (+7.8 %, 75.4 %) for XS in the same session; 10 of 20 drafted completions are identical to plain greedy decoding.
  Speed on the launch engine is queued. The MMLU numbers of this card were measured on the root with its earlier MTP
  block, which scoring does not use.

The forge [Fucina](https://github.com/Anjielon/fucina) unpacks the released file back to the separate-plane layout
that the recorded research code reads. `fucina/replay/unpack_planes.py` is the exact inverse of
`tools/ternary/pack_planes.py` of the release engine; with the recorded tensor order it reproduces the original root
**byte for byte** from the ungrafted packed root (`Thor-T-R2B3m-packed.gguf`, the replay input; the released file
differs from it only in block 40, the MTP block):

| | bytes | sha256 |
|---|---:|---|
| original separate-plane root (not released) | 8,846,968,608 | `56cb5bcdd9813a1015a7cad5364b5a80226f990fb8bd561b7cae4727d70efc43` |

```bash
git clone https://github.com/Anjielon/fucina && cd fucina
# check without writing anything (streams the output through sha256):
python -m fucina.replay.unpack_planes Thor-T-R2B3m-packed.gguf --hash-only \
  --order examples/thor/recipe_c/recorded/root_tensor_order.txt --gguf-py /path/to/llama.cpp/gguf-py
# write the separate-plane file:
python -m fucina.replay.unpack_planes Thor-T-R2B3m-packed.gguf Thor-T-R2B3m.gguf \
  --order examples/thor/recipe_c/recorded/root_tensor_order.txt --gguf-py /path/to/llama.cpp/gguf-py
sha256sum Thor-T-R2B3m.gguf   # 56cb5bcdd9813a1015a7cad5364b5a80226f990fb8bd561b7cae4727d70efc43
```

The replay recipe of the chain from the root to C (the trunk of XS) is `examples/thor/recipe_c.yaml` of Fucina, run by
`fucina replay`; it starts by this unpacking step. **We have not yet completed a run of the whole recipe from the
released inputs**, so the chain is not claimed reproducible from released artifacts end to end. Its final steps were
run with `fucina replay --run` on 12 Nov 2026 from tag `thor-recipe-c-v1.1` and end with `REPLAY VERIFIED`
(MMLU-2000 73.05 against the recorded 73.15, paired McNemar p = 0.87, tensor digest equal to the record); the
from-scratch reproduction of C reported in the paper lands at 72.75. GPU training steps are not bit-reproducible on our
GPU, so any replay is expected to land within run-to-run noise, not on the same bytes.

**S and M** have their own recipes in the same tag of Fucina (`recipes/RECIPE_S.md`, `recipes/RECIPE_M.md`, with
runners and recorded hashes). On 12 Nov 2026 both regenerated the released `Thor-T-35B-A3B-S.gguf` and
`Thor-T-35B-A3B-M.gguf` **bit for bit** from their recorded checkpoints, every recorded artifact of their chains matched
its sha256, and the released M measured 80.65 on the MMLU-2000 bench against the recorded 81.05 of its trunk (paired
McNemar p = 0.185).

##### Other inputs in this repository

Together with the root, this repository holds the inputs of the dense post-training-quantization step of the chain
(`tensori_R2B3/`). The per-expert input statistics read by the expert folds are the dataset
[Anjielon/Thor-T-fold-stats](https://huggingface.co/datasets/Anjielon/Thor-T-fold-stats) (32 files with their sha256
list). The earlier quantized student whose flow defined them (`Thor-v20-downaggr3.gguf`, sha256
`0f65c16227fa5b83641a35619054c4c159c0b50b6ee1201dbc479d97e3964b11`) and the corpus it was run on (`corpus_cot.txt`)
are in the dataset [Anjielon/Thor-T-bench](https://huggingface.co/datasets/Anjielon/Thor-T-bench); with them the
statistics can be re-captured (optional step I9 of the recipe).

<p align="center"><img src="assets/root_heads.png" alt="Decode speed of the root with the original and with our MTP head" width="720"></p>

Launch engine (Vulkan, APU, `GGML_VK_TQ1R_Q8=1`), 5 prompts × 256 tokens, greedy, one request, tokens/s. Our head:
mean of six repetitions on an idle machine (11 Nov); the original head: one run (10 Nov):

| MTP head | no draft | 1 draft token (acceptance) | 2 draft tokens (acceptance) |
|---|---:|---:|---:|
| the base model's original head (ungrafted root) | 75.56 | 38.71 (42.2 %) | 30.06 (24.6 %) |
| our ternary head (the released file) | 77.05 | **87.78** (71.7 %) | 87.83 (57.8 %) |

With the original head, one draft token roughly halves decode speed (38.71 against 75.56 tokens/s at
42.2 % acceptance); with our head the same trunk decodes faster with drafts than without. On the RTX 4060
laptop (CUDA, `-ncmoe 18`, five repetitions, the same protocol as the XS/S/M cards) our head gives
61.30 ± 0.29 against 54.48 ± 0.18 tokens/s,
1.13× at 72.6 % acceptance.

## Limitations

- **Selected on the reported quantity** (see the note at the top): MMLU full test and GSM8K test were used during the
  development of M.
- **One model, one teacher, one GPU family for quality.** Accuracy numbers come from Vulkan on one AMD gfx1151 APU;
  CUDA was checked against it on the 2,000-question development subset only, ROCm with operator tests.
- **Build effects.** Every teacher reference is on the launch engine, like M. Between builds (earlier build against
  the launch build) the same teacher file changes 135 of 2,376 ARC-easy choices, and its Winogrande score moves by
  −1.18 points (93 of 1,267 answers), 2.2 times the gap between M and the teacher on MMLU. On MMLU itself it moves
  from 80.33 to 80.59 (+0.26 points, 650 of 14,042 answers changed, correctness 270/233, p = 0.11), which alone moves
  M's retention from 99.66 % to 99.34 %. Results from other builds of the fork can differ by that much; our intervals
  do not cover this drift.
- **Activations are not full precision.** Likelihood scoring is batched: int8 `q8_1` activations on single-plane and
  expert tensors, f16 on the packed dense tensors; the int8 flag also changes the one-column products of the final
  layer and the head. Not comparable at matched activation precision with weight-only (W1.58A16) results.
- **Batch effects.** Batched evaluation is not bit-deterministic on this engine. The control, ARC-Challenge at
  `-np 5` against the bench's `-np 16` on all 1,172 items, changes 37 choices for XS (51.96 against 51.71; 16/13,
  p = 0.71) and 43 for M (11/16, p = 0.44). A first, interrupted control (`-np 1`, raised to 4 by the scorer, stopped
  at the first five-option question) had covered 836 items.
- **No ablation.** No equal-byte control separates the effect of the extra planes of M from its training history, and
  no simple untrained ternary allocation at this size was measured.
- **Training data.** The last step trains on WikiText-103 rows with a KL term to the teacher and a cross-entropy term.
- **Reproducibility.** The recipes of S and M (Fucina tag `thor-recipe-c-v1.1`) regenerate the released files bit for bit
  from their recorded checkpoints; their GPU training steps are documented with exact commands but are not
  bit-reproducible on this GPU, and a full re-run of their chains from the root has not been done. The root's
  construction cannot be regenerated from one configuration.
- **Not evaluated:** safety and refusal behaviour (inherited from the base model, and quantization can change it),
  long-context behaviour, languages other than English.

## License

The weights in this repository are released under **CC BY 4.0** (decision of the author). They are derived from
Ornith-1.5-35B-A3B, which is released under the MIT licence; the base model's copyright and permission notice is
reproduced in `LICENSE.base-model` and applies to the parts derived from it.

## Citation

```bibtex
@misc{mogavero2026thort,
  title         = {A Ternarization Toolchain for Mixture-of-Experts Models That Run on a Consumer APU: Three Sizes of a 35B Model, a Ternary MTP Head, and llama.cpp Kernels},
  author        = {Mogavero, Angelo},
  year          = {2026},
  howpublished  = {\url{https://huggingface.co/Anjielon/Thor-T-35B-A3B/blob/main/Thor-T-35B-A3B-paper.pdf}},
  note          = {Preprint}
}
```
