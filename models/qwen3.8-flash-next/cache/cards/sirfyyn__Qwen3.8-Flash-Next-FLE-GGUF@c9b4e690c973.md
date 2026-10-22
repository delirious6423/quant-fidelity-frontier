---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
language:
  - en
quantized_by: sirfyyn
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - moe
  - llama.cpp
  - fle
  - mtp
  - speculative-decoding
  - hybrid-quantization
  - kl-divergence
  - mixture-of-experts
  - quantized
  - qwen3.8
  - local-llm
---

# Qwen3.8-Flash-Next, FLE hybrids (GGUF): FLE-IQ3_M, FLE-IQ3_XXS, FLE-IQ2_M and an IQ4_NL MTP head

Three hybrid GGUF quantizations of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (512 routed
experts, 10 active per token) plus the model's own multi-token-prediction (MTP) head with its experts requantized to
IQ4_NL for FLE. We also tried re-training the head; it did not pay off on unseen prompts, so the untrained head is what
we ship (see [MTP head](#mtp-head)). A pruned variant of the first, FLE-IQ3_M-REAP448, is in its own repo:
[sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF).

| Name | Size | In one line |
|---|---|---|
| **FLE-IQ3_M** | 88.87 GB | KL divergence against BF16 within noise of Unsloth UD-IQ4_XS, at 4.81 GB less |
| **FLE-IQ3_XXS** | 79.48 GB | KL divergence equal to Unsloth UD-IQ3_XXS (82.0 GB), at 2.5 GB less |
| **FLE-IQ2_M** | 71.62 GB | KL divergence clearly below the larger Unsloth UD-IQ1_M (74.5 GB) |
| [FLE-IQ3_M-REAP448](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF) | 82.00 GB | FLE-IQ3_M with 64 of 512 experts per layer pruned by REAP saliency (161.84 B parameters), in its own repo |
| **MTP head** | 1.53 GB | the model's own MTP module (Unsloth's shared Q8_0 file), experts requantized to IQ4_NL, not re-trained |

The names follow the expert bit width, as in Unsloth's "UD" mixtures: the `IQ3_M`, `IQ3_XXS` and `IQ2_M` part says how
many bits the routed experts get (see [Bit allocation](#bit-allocation)), and the `FLE-` prefix marks the mixture. These
are not standard llama.cpp file types (`ftype`); every file mixes several tensor types.

They were built and measured for **FLE (Fast Local Experts)**, our llama.cpp fork for running large MoE models on one
GPU plus system RAM: <https://github.com/sirfyyn/FastLocalExperts>. The three quantizations here and FLE-IQ3_M-REAP448 are standard GGUF files and load in
upstream llama.cpp (FLE-IQ3_M-REAP448 simply has 448 instead of 512 experts per layer; its KLD below was measured with
upstream). The speed numbers below, and the MTP head, need the FLE engine (see [Running](#running)).

How the files and the engine were built and measured is in the report, with every bracket and the raw numbers behind
this card: <https://sirfyyn.github.io/FastLocalExperts/> (also as
[PDF](https://sirfyyn.github.io/FastLocalExperts/fle-paper.pdf)).

The hybrids are not new quantization methods. They are **splices of existing public quantizations**: whole expert
layers are taken from donor files that other people made, chosen layer by layer by measured sensitivity at a fixed
file size. All of the quantization work that matters here was done by [Unsloth](https://unsloth.ai) and
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab), see [Credits](#credits-and-licence).

## Making Rhea fast: eight short films

Eight short films (English, subtitles burned in, 33 to 63 s each, 7 min in total) tell how the speed was measured and
what did and did not work. Every number in them comes from our measurement database (6,130 measurements, state of
2026-10-08); where a film and this card differ, the card and the paper are the reference. All eight are also on YouTube
as one playlist: <https://www.youtube.com/playlist?list=PLMYhpsBSEhIQ>.

More from the same project: [website](https://sirfyyn.de/en/fle/), the [YouTube channel](https://www.youtube.com/@sirfyyn)
and [FLE Studio](https://github.com/sirfyyn/fle-studio), a local chat app for llama.cpp and FLE.

<video controls preload="none" width="100%" poster="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-1.png" src="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-1.mp4"></video>

| Film | Topic | Length |
|---|---|---|
| [1 · The race](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-1.mp4) | twelve seconds of decoding: upstream llama.cpp, the previous production and FLE side by side | 0:33 |
| [2 · The bracket](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-2.mp4) | how one paired measurement bracket works, and the two brackets that were void | 0:34 |
| [3 · The staircase](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-3.mp4) | nine steps from about 56 to about 100 tokens/s, each measured on its own | 1:03 |
| [4 · The expert cache](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-4.mp4) | 48 layers × 512 experts, hit rate, slot count and warm start | 0:59 |
| [5 · Draft and verify](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-5.mp4) | the MTP head, why two drafts, and how the re-trained head c2 learned (the release ships the untrained head instead: c2's gain held only on the tasks it was trained on, see [MTP head](#mtp-head)) | 1:00 |
| [6 · What did not work](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-6.mp4) | four ideas that lost: prefetching, n-gram drafts, PCIe copies, a rebase | 0:56 |
| [7 · The rebench](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-7.mp4) | a void first run, the slot fix, and IFBench, GPQA Diamond and LiveCodeBench against Qwen's card | 0:58 |
| [8 · Why we do this](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-8.mp4) | one machine at home, and what this release contains | 0:58 |
| [All eight in one file](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-complete.mp4) | | 6:59 |

## Related models

- Katy, a dense 27B quantization (Swift-1.5, IQ4_XS) on the same engine: https://huggingface.co/sirfyyn/Katy-27B-GGUF (project page: https://sirfyyn.de/en/katy/)
- The engine and the studio: https://github.com/sirfyyn/FastLocalExperts and https://github.com/sirfyyn/fle-studio

## Files

| File | Size | What | SHA-256 |
|---|---|---|---|
| `IQ3_M/Qwen3.8-Flash-Next-FLE-IQ3_M-0000N-of-00002.gguf` (N = 1, 2) | 88.87 GB (88,872,241,728 B) | main hybrid | per part, below |
| `IQ3_XXS/Qwen3.8-Flash-Next-FLE-IQ3_XXS-0000N-of-00002.gguf` (N = 1, 2) | 79.48 GB (79,483,800,128 B) | smaller hybrid | per part, below |
| `IQ2_M/Qwen3.8-Flash-Next-FLE-IQ2_M-0000N-of-00002.gguf` (N = 1, 2) | 71.62 GB (71,617,465,824 B) | smallest hybrid | per part, below |
| `mtp/mtp-Qwen3.8-Flash-Next-FLE-IQ4_NL.gguf` | 1.53 GB (1,528,277,056 B) | MTP head (untrained, IQ4_NL experts), needs FLE | `25217613e5e4659186db0b81ce2be77ab5955d01817732e4597b8f2bf0e292fa` |
| `mtp/fle-mtp-vocab-32k.txt` | 0.28 MB (281,159 B) | 32k draft vocabulary for the MTP head (FLE only) | `8385f0457511fe600986bbd0ecb0b32766cbe31859c12e78a033685c28eef0d8` |
| `media/making-rhea-fast-N.mp4` (N = 1 … 8), `media/making-rhea-fast-complete.mp4` | 43.2 MB / 43.1 MB | the eight short films and all of them in one file, see [Making Rhea fast](#making-rhea-fast-eight-short-films) | `media/SHA256SUMS` |

Each quantization is split into two parts of at most 45 GB, one folder per quantization; give llama.cpp or FLE the first
part and it loads the rest. Sizes are the sums of the two parts. SHA-256 of each part (after the metadata fix of
2026-10-09, see `hf-repo-plan.md` section 4; the same list is in `SHA256SUMS`):

```
624718434053eb01798d20c0f35f4673db58ee629cd01f9aee236e80c9f474dc  IQ3_M/Qwen3.8-Flash-Next-FLE-IQ3_M-00001-of-00002.gguf
7b3b094d2e5f0724720c217633677c450637cdd0212b02aad7a576510ecffa6a  IQ3_M/Qwen3.8-Flash-Next-FLE-IQ3_M-00002-of-00002.gguf
363de45326e29a1d210077cc30340a25df2281514cf04969672b8d5d2abc505f  IQ3_XXS/Qwen3.8-Flash-Next-FLE-IQ3_XXS-00001-of-00002.gguf
271b45f015bc1aa97fa44adf5c957330feb81ea85158b6428445fb441f8763b0  IQ3_XXS/Qwen3.8-Flash-Next-FLE-IQ3_XXS-00002-of-00002.gguf
36fa6072301df9bbc54000164625cc4a7292030f84e834d8a2fe1be65d97cc9c  IQ2_M/Qwen3.8-Flash-Next-FLE-IQ2_M-00001-of-00002.gguf
2fab43ec437fb7f0007686ac5ada3935ce7119fcfd8a3101931ab16485431155  IQ2_M/Qwen3.8-Flash-Next-FLE-IQ2_M-00002-of-00002.gguf
```

FLE-IQ3_M-REAP448 was in the folder `IQ3_M-REAP448/` of this repo until 2026-10-10. It is now in its own repo,
[sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF), with
the same two files and SHA-256, because the Hub's GGUF overview counted it as a second IQ3_M file of this repo.

Vision: this repo has no `mmproj`. Use ISTA's or Unsloth's projector from their repos (not redistributed here):
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
We have not tested the projectors with these hybrids.

## The model

Qwen3.8-Flash-Next is an open-weight model by the Qwen team: a Mixture-of-Experts model with 48 MoE layers of 512
routed experts each (10 active per token) plus a shared expert, hybrid attention with QSA, hyper-connections, and an
MTP module. The architecture is described in Qwen Team, "On the Design of Qwen3.8-Next Architecture",
[arXiv:2608.30320](https://arxiv.org/abs/2608.30320) (2026). The weights are released under the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). All files here are
derivatives of those weights.

## Why these quantizations

The model is 354 GB in BF16 and about 94 GB in the public 4-bit release. On one workstation (32 GB GPU, 120 GB RAM)
that is on the edge: every gigabyte saved is a gigabyte more for the GPU expert cache or for the KV cache.

Public quantizations of this model differ per tensor and per layer in how much damage they do. We measured, for every
MoE layer and every donor file, the relative output error of swapping that layer's expert tensors
(`|y_Q - y_BF16|^2 / |y_BF16|^2` on captured layer inputs, 4096 tokens), and then built files from whole expert layers
(gate, up and down together) of several donors, everything else from one base file:

- **FLE-IQ3_M**: base Unsloth UD-IQ4_XS; the cheapest-to-swap layers come from ISTA GSQ-RCO IQ3_S and IQ3_XXS. Selected as
  largest saving under an error limit.
- **FLE-IQ3_XXS / FLE-IQ2_M**: same tools, but a knapsack: for a fixed target size (80 GB / 72 GB) pick per layer the donor
  with the smallest predicted error (`tools/fle-hybrid/fle_select.py` in the engine repo, together with the splice
  tool `gguf_splice.py` and the provenance check `gguf_provenance.py`; re-running them reproduces FLE-IQ2_M's selection
  and layer provenance exactly).

The prediction is only a selection aid. Every file was measured afterwards against BF16 (next section). Quantizations
that did not pass are listed in the report (our own full quantizations, Q5_0 dense weights, fewer active experts).

Donors and bases:

| Donor | Source |
|---|---|
| Unsloth UD-IQ4_XS, UD-IQ3_XXS, UD-Q2_K_XL | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) ([Unsloth Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)) |
| ISTA-DASLab GSQ-RCO IQ3_S, IQ3_XXS, IQ2_XS | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF); methods GSQ ([arXiv:2604.18556](https://arxiv.org/abs/2604.18556)) and RCO ([arXiv:2605.00649](https://arxiv.org/abs/2605.00649)) |

### Provenance per layer

Layer numbers are 0-based MoE layers (0 to 47). Everything outside the experts (attention, shared expert, embeddings,
the 28.8 GB per-layer token table) comes from the base file. For FLE-IQ3_M the provenance was verified by comparing tensor
data with the source files (all 48 layers match exactly one source).

**FLE-IQ3_M** (base Unsloth UD-IQ4_XS):

| Source | Layers |
|---|---|
| Unsloth UD-IQ4_XS (base) | 1, 8, 10, 12, 13, 15, 19, 20, 21, 23, 24, 26, 36, 39 (14) |
| ISTA GSQ-RCO IQ3_S | 2, 3, 4, 5, 6, 7, 9, 11, 14, 16, 17, 18, 22, 25, 27, 28, 29, 31, 33, 35, 37, 38, 40, 41, 42, 44, 45, 46 (28) |
| ISTA GSQ-RCO IQ3_XXS | 0, 30, 32, 34, 43, 47 (6) |

**FLE-IQ3_XXS** (base ISTA IQ3_XXS; built size 79.48 GB for a target of 80 GB; summed predicted error 3.19 against 4.95 for the base alone; UD-IQ4_XS was offered and not chosen):

| Source | Layers |
|---|---|
| ISTA GSQ-RCO IQ3_XXS (base) | 0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 13, 14, 29, 35, 38, 42, 44, 46 (18) |
| ISTA GSQ-RCO IQ3_S | 8, 9, 11, 15, 18, 19, 22, 23, 24, 26, 28, 31, 32, 33, 34, 37, 39, 41, 43, 45 (20) |
| Unsloth UD-IQ3_XXS | 16, 17, 20, 21, 25, 27, 30, 36, 40, 47 (10) |

**FLE-IQ2_M** (base ISTA IQ2_XS; built size 71.62 GB for a target of 72 GB; summed predicted error 5.90 against 7.36 for the base alone):

| Source | Layers |
|---|---|
| ISTA GSQ-RCO IQ2_XS (base) | 0, 1, 2, 4, 8, 9, 10, 12, 14, 17, 18, 20, 22, 23, 24, 25, 28, 30, 33, 34, 36, 41, 45, 47 (24) |
| ISTA GSQ-RCO IQ3_XXS | 3, 5, 6, 7, 11, 13, 19, 21, 26, 29, 31, 35, 38, 39, 42, 43, 44, 46 (18) |
| Unsloth UD-Q2_K_XL | 15, 16, 27, 32, 37, 40 (6) |

The provenance of the built FLE-IQ3_XXS and FLE-IQ2_M files was verified the same way as for FLE-IQ3_M (tensor data of
every expert layer compared with the source files): every layer matches exactly one source, as listed in the tables
above.

## Bit allocation

The names follow the expert bit width, the way Unsloth's "UD" mixtures are named: `IQ3_M`, `IQ3_XXS` and `IQ2_M` say
how many bits the routed experts get (the part we mix layer by layer). `FLE-` marks the mixture; it is not a standard
llama.cpp `ftype`, and the files contain several tensor types. The file average (total) is higher than the expert
width because the per-layer token embedding table (PLE table, `per_layer_token_embd`) is stored at about 4.5 bits per
weight in every file and makes up 29 % of the parameters (28.84 GB in FLE-IQ3_M); it stays in system RAM.

Bits per weight, measured from the GGUF headers on 2026-10-09 (176.94 B parameters including the PLE table, 125 B
without):

| Name | Total bpw | Experts | PLE table | Attention / shared expert / norms | Embeddings / output |
|---|---|---|---|---|---|
| FLE-IQ3_M | 4.02 | 3.62 | 4.50 | 9.05 | 7.54 |
| FLE-IQ3_XXS | 3.59 | 3.08 | 4.51 | 7.32 | 4.53 |
| FLE-IQ2_M | 3.24 | 2.59 | 4.51 | 6.61 | 4.31 |
| FLE-IQ3_M-REAP448 | 4.05 | 3.62 | 4.50 | 9.00 | 7.54 |

FLE-IQ3_M-REAP448 has 161.84 B parameters (64 of the 512 experts per layer removed), the other three 176.94 B.

## Quality: KL divergence against BF16

Mean KL divergence (KLD) of the quantized model's next-token distribution against BF16, with top-1 agreement. Measured
with **upstream** `llama-perplexity` (llama.cpp commit `a8c9a4e7c`, no FLE code), `-c 2048 -b 2048 -ub 2048 -t 16
-ngl 99 -cmoe`, one GPU, no server running; the BF16 reference logits come from the same build and settings. Three
public texts that anyone can rebuild (script and SHA-256 of the sources are in the report):

- **E**: Python language reference (pydoc topics, PSF license, 63 KB)
- **W**: four German Wikipedia articles at fixed revisions (CC BY-SA 4.0, 81 KB)
- **K**: seven files of the German Kubernetes documentation at commit `4472af456106` (CC BY 4.0, 47 KB)

Values are mean KLD ± standard error. Lower is better. Compare within a column only; the texts differ in difficulty.
Sizes of public files are the totals of their GGUF splits as listed on the Hub.

| Quant | Source | Size (GB) | KLD E | KLD W | KLD K | Top-1 E / W / K (%) |
|---|---|---|---|---|---|---|
| UD-Q6_K_XL (quality ceiling, does not fit 120 GB RAM) | Unsloth | 169.2 | 0.0116 ± 0.0008 | 0.0164 ± 0.0005 | 0.0363 ± 0.0012 | 98.5 / 95.1 / 93.0 |
| 4.27 bpw | AtomicChat | 94.5 | 0.079 ± 0.004 | 0.071 ± 0.002 | 0.207 ± 0.006 | 96.0 / 89.9 / 85.7 |
| UD-IQ4_XS | Unsloth | 93.7 | 0.049 ± 0.003 | 0.046 ± 0.001 | 0.122 ± 0.004 | 96.8 / 91.7 / 88.6 |
| **FLE-IQ3_M** | this repo | **88.9** | **0.051 ± 0.003** | **0.050 ± 0.001** | **0.129 ± 0.004** | 96.9 / 91.5 / 89.1 |
| IQ3_XXS | bartowski | 88.0 | 0.181 ± 0.008 | 0.106 ± 0.003 | 0.292 ± 0.008 | 92.8 / 87.6 / 82.4 |
| 3.84 bpw | AtomicChat | 84.9 | 0.199 ± 0.009 | 0.185 ± 0.004 | 0.447 ± 0.012 | 92.6 / 83.6 / 78.6 |
| IQ3_S | ISTA GSQ-RCO | 83.6 | 0.069 ± 0.004 | 0.061 ± 0.002 | 0.183 ± 0.005 | 96.4 / 90.6 / 86.8 |
| **FLE-IQ3_M-REAP448** | [own repo](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF) | **82.0** | **0.080 ± 0.005** | **0.086 ± 0.003** | **0.185 ± 0.006** | 96.0 / 89.7 / 87.2 |
| UD-IQ3_XXS | Unsloth | 82.0 | 0.091 ± 0.005 | 0.091 ± 0.003 | 0.229 ± 0.007 | 95.5 / 88.3 / 84.6 |
| **FLE-IQ3_XXS** | this repo | **79.5** | **0.090 ± 0.005** | **0.093 ± 0.002** | **0.227 ± 0.007** | 95.7 / 88.1 / 84.2 |
| UD-Q2_K_XL | Unsloth | 78.9 | 0.133 ± 0.007 | 0.125 ± 0.003 | 0.299 ± 0.008 | 94.4 / 85.5 / 81.7 |
| IQ3_XXS | ISTA GSQ-RCO | 75.8 | 0.118 ± 0.006 | 0.112 ± 0.003 | 0.264 ± 0.007 | 95.0 / 87.4 / 83.6 |
| UD-IQ1_M | Unsloth | 74.5 | 0.189 ± 0.008 | 0.186 ± 0.004 | 0.413 ± 0.011 | 92.6 / 83.4 / 79.5 |
| **FLE-IQ2_M** | this repo | **71.6** | **0.160 ± 0.007** | **0.163 ± 0.004** | **0.339 ± 0.008** | 94.0 / 84.7 / 80.9 |
| IQ2_XS | ISTA GSQ-RCO | 68.0 | 0.179 ± 0.007 | 0.191 ± 0.004 | 0.382 ± 0.010 | 93.0 / 83.6 / 80.3 |
| Q2_0 | ISTA GSQ-RCO | 66.4 | 0.223 ± 0.009 | 0.236 ± 0.005 | 0.485 ± 0.012 | 92.4 / 82.3 / 77.7 |

All KLD values, standard errors and top-1 rates were re-checked on 2026-10-07 against the 45 measurement logs (15 files
× texts E/W/K, runs of 2026-10-05/06) and the file sizes against the files themselves; no deviation beyond the
rounding shown. FLE-IQ3_M-REAP448 was measured on 2026-10-08 with the same build, settings and BF16 reference (three
more logs). AtomicChat's and ISTA's own published quality numbers are on different corpora and
benchmarks and are not used here.

How to read it:

- **FLE-IQ3_M against UD-IQ4_XS**: +0.002 / +0.004 / +0.007 KLD on E / W / K (standard errors of the single values are
  0.003 / 0.001 / 0.004), 4.8 GB smaller (88.87 against 93.68 GB file, 60.06 against 64.87 GB resident without the
  per-layer embeddings). The acceptance criterion we fixed before selecting
  (on internal texts F, E, V; V held out): mean KLD on F and E at most about +0.0078 above UD-IQ4_XS, top-1 at most
  0.5 percentage points worse. FLE-IQ3_M gave F −0.0079, E +0.0031, V +0.0005.
- **FLE-IQ3_XXS against UD-IQ3_XXS**: the same KLD (0.090 / 0.093 / 0.227 against 0.091 / 0.091 / 0.229) at 2.5 GB less.
- **FLE-IQ2_M against UD-IQ1_M**: clearly lower KLD (0.160 / 0.163 / 0.339 against 0.189 / 0.186 / 0.413) at 2.9 GB less.
- **FLE-IQ3_M-REAP448 against FLE-IQ3_M**: 64 of the 512 experts per layer removed, 6.9 GB smaller (82.00 against
  88.87 GB), +0.029 / +0.036 / +0.056 KLD. At the size of UD-IQ3_XXS its KLD is lower on all three texts; ISTA IQ3_S,
  1.6 GB larger, is lower on E and W. KLD understates what pruning removes, see the
  [knowledge probe](#knowledge-probe-mkqa).
- **Against the ISTA files**: ISTA's quantizations are excellent for their size. FLE-IQ2_M and FLE-IQ3_XXS lie about on the line
  between ISTA IQ2_XS, IQ3_XXS and IQ3_S; they add size steps for whatever RAM you have, not a better size-quality curve.
- **Floor**: nothing in GGUF goes much below 66 GB for this model. Every low-bit release carries about 32.6 GB of
  non-expert weights, 28.8 GB of which is the per-layer token table `per_layer_token_embd` (rows of 160 values, stored
  as IQ4_NL; only 32-wide block types can store it). A planned 1-bit class (FLE-Q1, 63 GB) could not be built from
  public files and was dropped.

Noise: this model amplifies any change in computation order because expert selection flips. The same file, text and
build reproduce bit-exactly; GPU0 against GPU1 on one internal text gave 0.0905 against 0.0975. We therefore compare
KLD only on the same card, build and settings, and treat differences below about 0.01 on a single text as no finding.

The FLE engine's expert cache adds a small router bias towards cached experts (`FLE_MOE_CACHE_BIAS=0.001`). It changes
outputs slightly; it is not part of the KLD numbers above (those use the neutral upstream meter). On our internal texts
the effect stayed within the acceptance criterion. Set it to 0 for unchanged routing.

## Knowledge probe: MKQA

Three short texts do not show everything: removing experts can remove facts that those texts never touch. So every
file also answered 400 factual questions from MKQA ([Longpre, Lu and Daiber, arXiv:2007.15207](https://arxiv.org/abs/2007.15207);
open-domain export `nthakur/mkqa-open-domain`, answerable questions only), the same 400 questions in German and in
English. The system prompt asks for the answer only, or exactly "I don't know" when unsure. Sampling as on Qwen's card
(temperature 1.0, top-p 0.95, top-k 20, min-p 0), thinking off, up to 256 tokens, one fixed seed per question for every
file, so the comparison is paired. An answer counts as correct when one of MKQA's German or English answer aliases
appears in it (after lower-casing and removing accents and punctuation); "I don't know" counts as not correct. Server:
the production FLE build (Patch 51) with all experts on the CPU (`-cmoe`) and without MTP; measured 2026-10-08.

| File | Size (GB) | Correct, German | Correct, English | Against FLE-IQ3_M, German / English (points, paired 95 %) |
|---|---|---|---|---|
| FLE-IQ3_M | 88.87 | 23.2 % | 33.0 % | – |
| FLE-IQ3_XXS | 79.48 | 21.5 % | 30.8 % | −1.8 [−4.2 .. +0.5] / −2.2 [−5.0 .. +0.2] |
| **FLE-IQ3_M-REAP448** | 82.00 | 21.5 % | 29.8 % | −1.8 [−4.5 .. +0.8] / −3.2 [−6.5 .. +0.0] |
| FLE-IQ3_M, REAP to 384 experts (not released) | 75.13 | 17.5 % | 21.2 % | −5.8 [−9.2 .. −2.5] / −11.8 [−16.0 .. −7.8] |
| FLE-IQ3_M, 384 experts by mean activation norm (not released) | 75.13 | 11.2 % | 15.5 % | −12.0 [−15.8 .. −8.5] / −17.5 [−21.8 .. −13.2] |

The 95 % Wilson interval of each single rate is between ±3 and ±5 points; the differences are paired bootstrap
intervals (2,000 draws) over the same questions and seeds.

How to read it:

- The absolute rates are low by design: the questions come from Google's Natural Questions (many of them US-specific),
  the model is asked to abstain when unsure, and it does (FLE-IQ3_M: 91 of 400 German and 21 of 400 English answers
  were "I don't know"). Compare the rows with each other, not with other benchmarks.
- With 384 experts per layer, REAP's KLD (0.135 / 0.132 / 0.261) is only somewhat worse than FLE-IQ3_XXS
  (0.090 / 0.093 / 0.227), but it answers 5.8 / 11.8 points fewer questions correctly, three to five times the loss of
  FLE-IQ3_XXS. Pruning by REAP saliency clearly beat pruning by the mean activation norm (MAN, no router weight;
  [Liu et al., arXiv:2606.15716](https://arxiv.org/abs/2606.15716), who recommend it for task-agnostic pruning) at the
  same size, so the 448-expert file was built with REAP. One model and one probe; we do not generalize this.
- **FLE-IQ3_M-REAP448** loses about what FLE-IQ3_XXS loses (−1.8 / −3.2 against −1.8 / −2.2 points), the English drop
  at the edge of its interval. Between the two, FLE-IQ3_XXS is 2.5 GB smaller and FLE-IQ3_M-REAP448 has the lower KLD.
- A second probe checks that rare words survive: 60 short messages per language that must repeat an invented, rare
  full name. All five files repeated the full name in all of them.

How the pruning works: `llama-fle-reap` (in the engine repo) runs FLE-IQ3_M over a calibration text (256 sections of
2,048 tokens) and records, per layer and expert, how often it was routed and its REAP saliency, the mean
router-weighted norm of the expert's output over the tokens routed to it
([Lasby et al., arXiv:2510.13999](https://arxiv.org/abs/2510.13999)). The calibration text is FLE-IQ3_M's own answers
to public prompts (German part of OASST, SWE-Gym/SWE-Zero, UltraChat, GSM8K), mixed as 45 % German conversation, 25 %
code and tool calls, 18 % English conversation and 12 % math; neither the KLD texts nor the probe questions are in it.
`reap_prune.py` keeps the 448 experts per layer with the highest saliency, in their original order, and copies their
weights and router rows byte for byte; nothing is re-quantized. On the calibration text the kept experts receive on
average 96.1 % of the summed router weight (lowest layer 90.5 %).

## Task benchmarks: FLE-IQ3_M on FLE against Qwen's card

Three benchmarks from Qwen's own model card, run once each against our production server (FLE-IQ3_M file, the internal
re-trained MTP head c2, FLE build with patch 50, thinking on; the published head is the untrained one, see
[MTP head](#mtp-head)). These check that the production configuration keeps the model's task quality; they
are not a measurement of quantization loss (that is the KLD section above), and we did not run the unquantized model or
the other quantizations on them.

| Benchmark | Tasks | FLE-IQ3_M on FLE, % [Wilson 95 %] | Qwen card, % |
|---|---|---|---|
| IFBench, prompt-level strict | 300 | 80.3 [75.5 – 84.4] | 81.3 (strict or loose not stated) |
| IFBench, prompt-level loose | 300 | 83.0 [78.3 – 86.8] | 81.3 (strict or loose not stated) |
| GPQA Diamond | 198 | 88.9 [83.8 – 92.5] | 91.7 |
| LiveCodeBench v6 (2025-02-01 to 2025-05-31) | 131 | 85.5 [78.5 – 90.5] | 91.9 |

Card values: the language benchmark table of the
[Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, read 2026-10-08. Measured 2026-10-07 and 2026-10-08.

- **Setup**: the card's thinking-mode sampling (temperature 1.0, top-p 0.95, top-k 20, min-p 0, no presence penalty),
  up to 81,920 new tokens, one run, no answer re-drawn or filtered. The card recommends 262,144 reasoning tokens, which
  does not fit into a 131,072-token slot. IFBench with the official evaluator; GPQA Diamond with Qwen's answer format
  (`"answer": "C"`); LiveCodeBench with the official `lcb_runner`, release v6, window 2025-02-01 to 2025-05-31 (taken
  from Qwen's earlier cards; this card names no window).
- **Truncated answers count as wrong.** IFBench: none. GPQA Diamond: 6 of 198 reached the 81,920-token limit while
  still reasoning. LiveCodeBench: 10 of 131 did, 10 of its 19 failures. If all of them had been solved, GPQA would be
  91.9 % and LiveCodeBench 93.1 %; we did not re-run with a larger limit.
- **Reading**: IFBench and GPQA Diamond are level with the card (the card's value lies inside the interval).
  LiveCodeBench is **below** the card by 6.4 points and outside the interval; hard problems 72.1 % (44 of 61), medium
  94.9 % (37 of 39), easy 100 % (31 of 31).
- **Limits**: one run (the spread between runs at temperature 1.0 is not in the intervals); the card names neither its
  output limit nor its run count, so the comparison is not like for like; our harness is not Qwen's; the
  LiveCodeBench window has only 131 problems.

Method, token statistics and an invalid first run (a slot mix-up in the server, fixed by patch 50; both discarded) are
in [Section 5.4 of the report](https://sirfyyn.github.io/FastLocalExperts/#s5-4).

## Speed

Final measurement series of 2026-10-06 (brackets B1e, B3e1 to B3e5 and the Strata comparison S) and bracket B3e6 of
2026-10-09 (FLE-IQ3_M-REAP448), all on the final engine configuration (`p49-ring`: Patches up to 49). The engine in production since 2026-10-07 adds Patch 50, which
changes only which slot a waiting request may touch (a correctness fix for concurrent requests, see the FastLocalExperts
changelog), and Patch 51, a crash report for faults in CPU graphs plus a range check of the expert ids. Neither
changes the computation, and the speed numbers were not re-measured with them. The release tag `v1.0` of the engine
is this production state; it differs only in comments, documentation and tools.

The final series (B1e, B3e1 to B3e6, S) was **measured with the internal head c2**; the published untrained head was
+2.95 % [+1.96 .. +3.93] faster than c2 on unseen prompts (bracket KTG, see [MTP head](#mtp-head)). We did not re-run
the series with the published head and do not extrapolate its numbers.

Hardware: AMD Ryzen 9 9950X3D (16 cores, 32 threads), 120 GB DDR5, one NVIDIA RTX PRO 4500 Blackwell (32 GB), Linux
kernel 7.0, NVIDIA driver 610.43; CPU experts on cores 0-15, 16 threads.

Method: every speed claim is **paired** inside one measurement bracket: reference run, candidate runs, reference run
again, same 24 prompts (code 9, math 6, physics 3, prose 6, greedy, MTP on where the engine supports it) and the same
server settings; the bracket counts only if the two reference runs differ by at most 5 %; 95 % intervals are bootstrap
intervals over the paired requests. Numbers from different brackets are never compared directly. Prefill and
time-to-first-token were measured per arm in the same brackets with three prompt lengths (about 1k, 4k and 8.6k
tokens); they are medians, not paired.

### FLE against upstream llama.cpp (bracket B1e)

Same FLE-IQ3_M file, same card, same 16 threads, `--load-mode none`; 24 paired requests, reference drift +0.44 %. Reference
decode: median 9.9 to 10.1 ms per token, about 100 tokens/s on these prompts, 82 to 86 % of drafted tokens accepted.

| Engine | decode vs FLE | 95 % interval | FLE is |
|---|---|---|---|
| upstream llama.cpp `a8c9a4e7c`, `-ub 1024`, no expert cache, no MTP, fit target 1024 MiB (23.6 GB of experts static on the GPU) | −59.9 % | −61.2 .. −58.5 % | 2.49× as fast |
| the same with `-ub 4096` | −64.0 % | −65.2 .. −62.9 % | 2.78× as fast |

It holds for every prompt type (upstream `-ub 1024`: −59.0 % code, −63.6 % math, −58.6 % physics, −57.9 % prose). The
upstream arm runs without speculative decoding because upstream cannot use the public MTP files (see
[Running](#running)).

**Prefill is where FLE loses.** Prompt processing in tokens/s (median per arm):

| Prompt | FLE | upstream `-ub 1024` | upstream `-ub 4096` |
|---|---|---|---|
| ~1k tokens | 416–442 | 715 | 517 |
| ~4k tokens | 928–934 | 900 | 1,222 |
| ~8.6k tokens | 1,248–1,256 | 967 | 1,416 |

Time to first token for short prompts (20 to 192 tokens): FLE 0.71 to 1.03 s, upstream `-ub 1024` 0.64 to 0.92 s.
FLE is 24 % (4k) and 12 % (8.6k) behind upstream `-ub 4096`, and about 40 % behind upstream `-ub 1024` at 1k (at 4k
and 8.6k it is ahead of `-ub 1024`). Our working assumption, not
measured: upstream keeps 23.6 GB of experts statically on the GPU and needs no copies for them, while FLE streams
experts through its cache and copies more during a large prefill; below 2,048 tokens FLE takes a path that copies all
experts of a layer. Prefill is the next work item on the engine.

### The quantizations against each other on FLE (brackets B3e1 to B3e6)

All candidates run with the final FLE configuration; only the model file differs; the reference in each bracket is FLE-IQ3_M.
The GPU expert cache is sized by one byte budget (17,035 MiB) for every file, so smaller files do not get an advantage
from fixed slot counts. Reference drift −0.30 to +1.16 % in B3e1 to B3e5 and +1.64 % in B3e6, no invalid requests.
Positive means faster than FLE-IQ3_M.

| Quant | Size (GB) | Decode vs FLE-IQ3_M | 95 % interval | Bracket |
|---|---|---|---|---|
| ISTA IQ2_XS | 68.0 | +13.9 % | +11.2 .. +16.7 | B3e3 |
| **FLE-IQ2_M** | 71.6 | **+11.6 %** | +9.8 .. +13.4 | B3e5 |
| Unsloth UD-IQ3_XXS | 82.0 | +8.6 % | +7.4 .. +9.8 | B3e2 |
| ISTA Q2_0 | 66.4 | +8.5 % | +6.2 .. +10.8 | B3e3 |
| Unsloth UD-IQ1_M | 74.5 | +2.3 % | +0.4 .. +3.9 | B3e4 |
| **FLE-IQ3_M-REAP448** | 82.0 | **+0.7 %** | −0.3 .. +1.8 | B3e6 |
| Unsloth UD-Q2_K_XL | 78.9 | +0.6 % | −0.8 .. +1.9 | B3e4 |
| ISTA IQ3_XXS | 75.8 | −0.4 % | −2.2 .. +1.4 | B3e1 |
| **FLE-IQ3_XXS** | 79.5 | **−0.7 %** | −2.1 .. +0.6 | B3e5 |
| bartowski IQ3_XXS | 88.0 | −1.1 % | −2.4 .. +0.2 | B3e2 |
| ISTA IQ3_S | 83.6 | −4.3 % | −5.5 .. −3.2 | B3e1 |
| AtomicChat 3.84 bpw | 84.9 | −6.0 % | −7.3 .. −4.7 | B3e2 |
| AtomicChat 4.27 bpw | 94.5 | −22.5 % | −23.3 .. −21.7 | B3e3 |
| Unsloth UD-IQ4_XS | 93.7 | −34.7 % | −35.4 .. −34.0 | B3e1 |

FLE-IQ3_M-REAP448 was measured three days after the others in its own bracket (B3e6, 2026-10-09, same build, budget and
protocol, 24 paired requests): as fast as FLE-IQ3_M, +0.7 % [−0.3 .. +1.8] started cold and +0.5 % [−0.6 .. +1.7]
with the warm-start profile its cold run had written. Every candidate except the reference starts cold, because the profile
belongs to FLE-IQ3_M; that may cost the candidates up to about 1.5 % (bracket CG, see the flag table under
[Running](#running)). Differences of one to two percent in this table are not a ranking.

The three slowest files did not fit the configuration completely: with the same budget, 48 (AtomicChat 3.84), 47
(AtomicChat 4.27) and 46 (UD-IQ4_XS) of 49 layers were placed on the GPU. FLE-IQ3_M exists because UD-IQ4_XS, the file it is
built from, does not fit this machine well. Not measured for speed: Unsloth UD-Q6_K_XL (does not fit into 120 GB RAM).
Prefill at 8.6k tokens was 1,197 to 1,363 tokens/s for all files that fit completely (FLE-IQ3_M 1,233 to 1,254).

How to read it: a smaller file frees GPU memory for more cache slots, but the dequantization cost of the cheaper types
on the CPU decides whether that pays. FLE-IQ2_M is the fastest of our files; FLE-IQ3_XXS is as fast as FLE-IQ3_M at 9.4 GB less. This
is a speed ordering on one machine for one configuration, not a quality ordering; read the KLD table next to it. In
the greedy speed prompts, the small files more often ran into the length limit (FLE-IQ2_M 3 of 24, ISTA IQ2_XS and Q2_0
2 each, ISTA IQ3_XXS 2, UD-IQ1_M 2, AtomicChat 3.84 1, FLE-IQ3_M 0), a hint of repetition loops at low bit widths.

**Patch 45 (AVX2 dot product for Q2_0) matters for FLE-IQ2_M and FLE-IQ3_XXS.** Upstream llama.cpp computes Q2_0 only with a
scalar kernel on x86; the FLE engine adds an AVX2 kernel (Patch 45), and all B3e numbers include it. Without it, FLE-IQ2_M
was −5.1 % and FLE-IQ3_XXS −6.1 % against FLE-IQ3_M (fixed 173 cache slots, earlier engine state); use these two files with an FLE
build that contains Patch 45. Q2_0 appears only in `ffn_down_exps`: in FLE-IQ3_XXS in 10 of 48 layers (1, 3, 5, 6, 7, 8, 10,
12, 13, 14), in FLE-IQ2_M in 33 of 48 layers; FLE-IQ3_M has none (all its `ffn_down_exps` are IQ4_NL).

### Against Strata (comparison S, not paired)

[Strata](https://github.com/Niko1221/strata) is the engine whose ideas FLE took over last (expert pager, prefill in the
KV slots). Same FLE-IQ3_M file, same machine, Strata with speculative depth 4 on 16 cores; FLE measured before and after
Strata in the same window. These are medians of two separate runs, not a paired bracket.

| | FLE | Strata |
|---|---|---|
| decode, 6 tasks (tokens/s) | 91.2 / 91.3 | 74.0 |
| prefill ~1k / ~4k / ~8.6k (tokens/s) | 443 / 922–933 / 1,179–1,181 | 1,183 / 2,239 / 2,673 |
| time to first token, 20–192 tokens (s) | 0.69–1.04 | 0.39–0.50 |

FLE decodes about 23 % faster; Strata processes prompts 2.3 to 2.7 times as fast. The Strata takeover (W7) narrowed
the prefill gap at 4k and 8.6k from 3.4 and 3.8 times to 2.4 and 2.3 times; at 1k and for the first token it did not
change.

### How FLE got there

Each step against the engine state before it, paired, earlier brackets: expert-cache policy +12.15 % [+10.68 ..
+13.57] (bracket HZ2); overlap, fusion and IQ4_NL MTP head +19.46 % [+18.00 .. +21.03] (ALL2); re-trained MTP head c2
+4.54 % [+2.67 .. +6.75] (KTC, on the very tasks it was trained on; on unseen prompts the untrained head was faster, see
[MTP head](#mtp-head), so this step is not in the release); 32k draft vocabulary +4.14 % [+3.04 .. +5.24] (VK);
16 threads instead of 8 +5.23 % [+3.64 ..
+6.86] (RB2); against the previous production (stock build, UD-IQ4_XS, about 56 tokens/s) +58.53 % [+55.83 .. +61.58]
(END2). The Strata takeovers W2 to W7 (Patches 44 to 49) are in the final configuration; their single steps are in the
report.

## MTP head

Qwen3.8-Flash-Next ships an MTP module. In llama.cpp, MTP is used as draft-and-verify speculative decoding
([Leviathan et al., arXiv:2211.17192](https://arxiv.org/abs/2211.17192);
[Chen et al., arXiv:2302.01318](https://arxiv.org/abs/2302.01318); MTP: DeepSeek-V3,
[arXiv:2412.19437](https://arxiv.org/abs/2412.19437), Gloeckle et al.,
[arXiv:2404.19737](https://arxiv.org/abs/2404.19737)). With the shipped head the conditional acceptance is about
0.86 / 0.78 / 0.78 for the first three drafted positions, which makes two drafted tokens (`n-max 2`) the optimum.

### The head in this repo

`mtp/mtp-Qwen3.8-Flash-Next-FLE-IQ4_NL.gguf` is the model's own MTP module, **not re-trained**:

- **Source**: Unsloth's file `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from
  [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), revision `eb2a07ecb33b`
  (2026-09-01).
- **Change**: only its three expert tensors `blk.48.ffn_down_exps`, `blk.48.ffn_gate_exps` and `blk.48.ffn_up_exps` were
  requantized from Q8_0 to IQ4_NL (`gguf_requant.py`); relative error against the source on the first 8,192 rows
  0.0772 / 0.0785 / 0.0781. The other 29 tensors are unchanged. 32 tensors, 1,528,277,056 bytes.
- **Quantization**: IQ4_XS is not possible because the head's down experts have a row length of 640, which is not a
  multiple of 256. A Q4_0 variant was built and measured: with the FLE overlap build the IQ4_NL head gave +19.46 % and
  the Q4_0 head +16.63 % against the same reference (bracket ALL2).
- **Draft vocabulary**: a short list of 32,768 tokens (`fle-mtp-vocab-32k.txt`, `FLE_MTP_VOCAB_N=32768`) limits the
  vocabulary the head drafts from: +4.14 % [+3.04 .. +5.24] (bracket VK, measured with the internal head c2); a 16k
  list gave +2.9 %. The list holds the tokens that FLE-IQ3_M itself predicted as top-1 on the training captures
  described below, ordered by frequency; the engine takes the first N entries.
- **Not helpful**: `n-max` other than 2 (1 −6.7 %, 3 −1.3 %, 6 −30.7 % with the untrained head, bracket NM2, which is
  formally invalid at drift −5.9 % and only indicative); adaptive `n-max` (−7.7 % to −10.3 %, bracket P42); n-gram
  drafting before MTP (−13.8 % / −23.3 %, bracket NG).

The head is used through `-md` (draft model) with `--spec-type draft-mtp` in FLE. Acceptance of the drafted tokens with
this head on FLE-IQ3_M was 71.00 % on the unseen prompts of bracket KTG (below). The acceptance figures elsewhere in
this card (82 to 86 % in B1e; 77.9 % on FLE-IQ3_XXS, 81.5 % on FLE-IQ2_M and 81.6 % on FLE-IQ3_M in the final bracket
B3e5) were measured with the internal head c2 on the speed prompts; the published head was not measured on FLE-IQ3_XXS
and FLE-IQ2_M.

### Negative result: a re-trained head did not generalize

We re-trained the head to raise acceptance at greater depth. It looked like a gain and was not one.

- **Method**: self-distillation on the model's own answers, with a recursive rollout. The draft head is unrolled to
  depth 7 on its own predictions, and at every depth it is trained against the main model's next-token distribution
  (cross-entropy against the teacher's top-32 logits, renormalized, with per-depth weights), plus an L2 anchor
  (3e-3) towards the shipped head. Teacher and answer generator are the **FLE-IQ3_M file itself** (not BF16), so the head
  learns the distribution it has to predict in operation. Only the 25 non-expert tensors of the head (88.9 M
  parameters) are trained; its experts stay byte-identical to the shipped head.
- **Data**: prompts from OASST (German part), SWE-Gym, SWE-Zero, UltraChat and GSM8K, and, for about 40 % of the
  prompt pool, the author's own assistant conversations (filtered for secrets, never published). FLE-IQ3_M answered them
  (temperature 0.6, top-p 0.95, top-k 20, up to 1,500 tokens; 6,127 answers, about 5.6 M output tokens in two stages);
  the teacher pass stored hidden state and top-32 logits per token at context 2,048. Training run `lauf3`: 3,781
  sections of 2,048 tokens (7.74 M token positions), drawn from four capture sets with weights 6/3/1/1; two of the
  smaller sets are not on-policy (earlier evaluation outputs and project documentation). The training corpus (`m2`)
  contained FLE-IQ3_M's answers to **exactly the 8 fixed tasks that bracket KTC measures with**. This is the mistake
  the rest of this section is about.
- **Training**: 6,000 steps, sequence 1,024, gradient accumulation 2 (about 1.6 epochs), AdamW (betas 0.9/0.98, no
  weight decay), learning rate 3e-5 with 50 warm-up steps and cosine decay to 10 %, bf16 autocast, gradient clipping
  1.0; one GPU (17.8 GiB peak), 3.5 hours. The checkpoint with the best expected tokens per cycle at `n-max 4` on 16
  validation sections was kept: 1.653 before, 2.028 after; greedy acceptance at depth 1 on the same sections 70.3 % →
  78.7 %. "c2" is this head used at `n-max 2`. A longer run (`lauf5`: 1.7 times the sections, plus a fourth capture
  set, started from the shipped head) did not beat c2: +0.28 % [−0.73 .. +1.48] at `n-max 2` (bracket KTE, 48 paired
  requests; acceptance 79.9 % against 80.4 %).
- **First result, on the training tasks** (bracket KTC, 24 paired requests on the same 8 tasks, `n-max 2`, drift
  −1.84 %): head c2 +4.54 % [+2.67 .. +6.75] against the untrained head, acceptance 80.2 % against 77.1 %. An earlier
  figure of +4.92 % could not be reproduced from the raw data and is not used. Because the corpus contained answers to
  these very tasks, this was overfitting to the measurement tasks, not a general gain.
- **Unseen tasks, first try** (bracket KTF): 16 tasks the head had not seen (8 German, 8 English; code, math, prose,
  knowledge), 48 paired requests, reference c2 before and after. The untrained head was **+1.61 %** [+0.88 .. +2.36]
  faster than c2; a third head `pub`, re-trained on public prompts only, was −0.11 % [−1.11 .. +0.86].
  Acceptance c2 71.35 %, untrained 72.40 %, pub 71.73 %. Confounder: every arm had its own warm-start expert profile
  (c2's predicted 72 % cache hits, the other two 80 %).
- **Unseen tasks, clean repeat** (bracket KTG, 2026-10-09): the same 48 requests, but every arm starts from a copy of
  the same production expert profile (79.1 % predicted hits); FLE-IQ3_M, `n-max 2`, bootstrap 95 %. Reference c2
  11.740 → 11.298 ms/token (drift −2.38 %). The untrained head was **+2.95 %** [+1.96 .. +3.93] faster than c2, `pub`
  −1.63 % [−2.35 .. −0.91]. Acceptance: untrained 71.00 %, pub 71.10 %, c2 70.91 %.

Per task type in KTG (speed against c2, n = 6 each):

| Head | de code | de math | de prose | de knowledge | en code | en math | en prose | en knowledge |
|---|---|---|---|---|---|---|---|---|
| untrained (released) | +2.7 % | +2.1 % | +1.6 % | −0.5 % | +3.9 % | +5.1 % | +5.1 % | +3.7 % |
| pub | −1.5 % | −2.0 % | −1.8 % | −3.8 % | −2.7 % | −2.5 % | +1.6 % | −0.3 % |

Acceptance is practically the same for all three heads, so the untrained head's speed lead is **not explained by
acceptance**; its cause is not known. What is clear: the gain of the re-trained head held only on the tasks it was
trained on, and on unseen prompts the untrained head is faster. That is why this repo ships the untrained head and not
c2 or pub. The lesson for anyone trying this: **measurement tasks and training corpus must be disjoint.** The
re-trained heads are not published.

## Running

### Upstream llama.cpp (works as is, no MTP)

FLE-IQ3_M, FLE-IQ3_XXS, FLE-IQ2_M and FLE-IQ3_M-REAP448 (own repo) are standard GGUF files. With current upstream llama.cpp, for example with the experts on the CPU:

```bash
llama-server -m Qwen3.8-Flash-Next-FLE-IQ3_M.gguf -cmoe -ngl 99 -c 32768 --jinja
```

(Upstream is what we measured against in bracket B1e at commit `a8c9a4e7c`, with `--fit-target 1024` and
`--load-mode none` instead of `-cmoe`. The command above is the simple form.)

**The MTP head does not work with upstream** with these files: upstream expects the MTP module inside the target
model (llama.cpp PR #29761), while Unsloth's releases and this repo ship it as a separate file. In our smoke test
(`b1_mtp_probe.sh`), the Unsloth "shared" head fails with `token_embd.weight not found`, and the complete head aborts
at the first compute step (`GGML_ASSERT(buffer)` in `process_ubatch`), on the 16 GB card as well as on the 32 GB card
with the head on the GPU, so it is not a memory problem. For a user of the public files, the arm without MTP is the
realistic default on upstream. This smoke test ran on 2026-10-05 with upstream `a8c9a4e7c`; later upstream versions
may load a separate head.

### FLE (full speed, with MTP head)

Engine: <https://github.com/sirfyyn/FastLocalExperts> (a llama.cpp fork; build instructions are in its README, CUDA and a
CPU with AVX2 are required). The fork is based on llama.cpp `6a1a922d` (2026-09-05) plus upstream PRs #28302 and
#29599 and parts of #29761. Use the release tag
[`v1.0`](https://github.com/sirfyyn/FastLocalExperts/tree/v1.0): the engine state of these measurements plus
Patches 50 and 51 (see [Speed](#speed)).

The command we run in production on the hardware above (configuration of all final speed numbers; production and those
measurements use the internal head c2 where this command names the published head; the variable names are the current
English ones, the production unit still uses the deprecated aliases listed in the FastLocalExperts `docs/fle/ENV.md`):

```bash
FLE_MOE_CACHE_BUDGET_MIB=17035 FLE_MOE_CACHE_BUDGET_FIT=1 \
FLE_MOE_PROFILE=/path/to/expert-profile.bin FLE_MOE_PROFILE_SAVE_SEC=600 \
FLE_MOE_ADMIT_FREQ=2000 FLE_MOE_EVICT_OLDEST=32 FLE_MOE_CACHE_BIAS=0.001 FLE_MOE_OBS_MAXTOK=8 \
FLE_MC_OVERLAP=2 FLE_HC_FUSED=1 FLE_CKPT_TAIL=512 \
FLE_MTP_VOCAB=/path/to/fle-mtp-vocab-32k.txt FLE_MTP_VOCAB_N=32768 FLE_MTP_PENALTIES=1 \
FLE_PREFILL_LEAN=1 FLE_SLOT_EARLY_SAVE=1 FLE_CACHE_GUARD_SLOT=0 \
FLE_UBATCH_LEND=1 FLE_UBATCH_PREFETCH=1 FLE_UBATCH_PREFETCH_SLOTS=3 \
/path/to/FastLocalExperts/build/bin/llama-server \
  -m /path/to/Qwen3.8-Flash-Next-FLE-IQ3_M.gguf \
  -md /path/to/mtp-Qwen3.8-Flash-Next-FLE-IQ4_NL.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.5 --no-spec-draft-backend-sampling \
  -cmoed -c 262144 --parallel 2 --kv-unified --jinja \
  -b 4096 -ub 4096 -ctk q8_0 -ctv q8_0 \
  -fit on --fit-target 1115 --moe-expert-cache-inserts 1 \
  --cache-ram 32768 --load-mode none -t 16 -C 0x0000FFFF --cpu-strict 1 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5
```

The production unit also passes `--moe-expert-cache 173`; with the byte budget set, the slot count per layer is planned
from the budget and that number has no effect, so it is left out here.

What the flags do, in short:

| Flag | Meaning |
|---|---|
| `-cmoed` | experts of the draft (MTP) model stay on the CPU |
| `FLE_MOE_CACHE_BUDGET_MIB=17035`, `FLE_MOE_CACHE_BUDGET_FIT=1` | GPU expert cache sized by one byte budget for all layers (Patch 44); the budget is added to the fit margin, so `--fit-target 1115` is the free memory besides the cache |
| `FLE_MOE_PROFILE` | expert-usage profile: warm start of the cache after a restart (measured +1.49 % [+0.64 .. +2.41] against the same build without budget, bracket CG); the file is written while running |
| `--moe-expert-cache-inserts 1` | at most one expert upload per layer and decode step |
| `FLE_MOE_ADMIT_FREQ`, `FLE_MOE_EVICT_OLDEST`, `FLE_MOE_CACHE_BIAS`, `FLE_MOE_OBS_MAXTOK` | cache policy (admission by frequency, eviction among the oldest, router bias towards cached experts, learn only from batches up to 8 tokens) |
| `-b 4096 -ub 4096` with `FLE_UBATCH_LEND=1` | decode runs with a 1,024-token ubatch; a prefill call of 2,048 tokens or more lends expert-cache layers for the 4,096-token buffer (Patch 48) |
| `FLE_UBATCH_PREFETCH=1`, `FLE_UBATCH_PREFETCH_SLOTS=3` | during a lend, a second stream copies the next expert weights ahead (Patch 49) |
| `FLE_SLOT_EARLY_SAVE=1`, `FLE_CACHE_GUARD_SLOT=0`, `--cache-ram 32768` | prompt cache in RAM; a slot is saved when it is released, and the entries of slot 0 are evicted last |
| `--load-mode none` | expert weights are read once into a pinned host buffer (about 54 GB of RAM that the page cache cannot reclaim; prefill +86 to +91 % and decode +9.1 % in our test with alternating arms, not a paired bracket; drop it if RAM is tight) |
| `--spec-draft-n-max 2` | two drafted tokens per cycle (measured optimum) |
| `-t 16 -C 0x0000FFFF --cpu-strict 1` | one thread per physical core, pinned to cores 0-15 |
| sampling | Qwen's recommended settings for thinking mode in chat; the benchmark runs used the settings of Qwen's model card instead (temperature 1.0, top-p 0.95, top-k 20, min-p 0, no presence penalty) |

The budget and `--fit-target` depend on your GPU memory: they are tuned for a 32 GB card with a 262,144-token context
shared by two slots (`--parallel 2 --kv-unified`). For FLE-IQ3_XXS and FLE-IQ2_M, use the same flags and the same budget (this
is how all B3e numbers were measured); the planner fits more slots into the budget for the smaller files. Run FLE-IQ3_XXS and
FLE-IQ2_M only with a build that contains Patch 45.

### Memory

Measured on 2026-10-09 with the `v1.0` build and the command above (published head, 32k vocabulary): the server loaded
each file from an NFS share and answered one short request, then we read the memory figures.

| File | System RAM taken | of it pinned (`--load-mode none`) | n-gram table, memory-mapped | GPU memory | Load time |
|---|---|---|---|---|---|
| FLE-IQ3_M | 54.3 GiB | 52.2 GiB | 26.8 GiB | 31,315 MiB | 167 s |
| FLE-IQ3_XXS | 46.1 GiB | 43.9 GiB | 26.8 GiB | 31,279 MiB | 241 s |
| FLE-IQ2_M | 38.7 GiB | 36.7 GiB | 26.8 GiB | 31,501 MiB | 127 s |
| FLE-IQ3_M-REAP448 | 48.0 GiB | 46.1 GiB | 26.8 GiB | 31,367 MiB | 127 s |

- *System RAM taken* is the drop of `MemAvailable` from before loading to after the request. Almost all of it is the
  pinned buffer that `--load-mode none` fills with the expert weights and the MTP head.
- The machine's second GPU was visible in this probe and held 0.5 to 1.3 GiB of weights. With one GPU these weights
  stay in system RAM: the production server, FLE-IQ3_M on one GPU, holds 53.3 GiB pinned against 52.2 GiB here.
- Not included: the prompt cache (`--cache-ram 32768`, up to 32 GiB) and the page cache for the n-gram embedding table
  (`per_layer_token_embd`, 27,466 MiB in every file), which stays memory-mapped from the file and is read as the text
  needs it.
- The expert cache fills the 32 GB card by design (the budget plus `--fit-target`).
- We have run these files only on the machine above (120 GB RAM) and have not tested how much less is enough. Without
  `--load-mode none` the experts are memory-mapped like the n-gram table and the page cache can give them back; prefill
  and decode were slower that way in our test (flag table above).

## Limitations

- One model, one machine (a single 32 GB GPU, 120 GB RAM, Linux, CUDA). The cache policies are tuned on this
  hardware; the speed may not transfer to other machines.
- Quality is **KLD against BF16** on three texts, two of them German, a 400-question MKQA knowledge probe per language
  (FLE-IQ3_M, FLE-IQ3_XXS and FLE-IQ3_M-REAP448; FLE-IQ2_M was not probed), plus one run each of IFBench, GPQA Diamond
  and LiveCodeBench v6 on the production configuration (sections above). KLD is not task accuracy; the benchmark runs
  are single runs without a comparison against the unquantized model, and LiveCodeBench is below Qwen's card. Earlier
  task tests from late September did not hold up statistically and are not reported.
- The KLD numbers come from the neutral upstream meter. The FLE engine adds a cache bias (default 0.001) that changes
  outputs slightly (see above) and is not included in them.
- The speed numbers were measured with engine state `p49-ring`; the release tag `v1.0` adds Patches 50 and 51, which do
  not change the computation (see [Speed](#speed)).
- FLE-IQ3_XXS and FLE-IQ2_M are slower than FLE-IQ3_M on FLE without Patch 45 (see Speed).
- Selection used one prediction metric on 4096 captured tokens; it is a selection aid, not a guarantee. The measured
  KLD of the finished files is what counts.
- The MTP head is the model's own, untrained head with requantized experts. The final speed series (B1e, B3e1 to B3e6,
  S), the task benchmarks and the draft-vocabulary step were measured with the internal re-trained head c2 (see
  [MTP head](#mtp-head)). Acceptance
  depends on the type of text (82 to 86 % in the B1e speed prompts with c2, about 71 % for all heads on the unseen
  prompts of KTG). The head works only in FLE.
- No support promise. This is a personal project. Issues and PRs on the FLE repo are welcome; there is no service
  level.

## Credits and licence

These files are derivatives of Qwen3.8-Flash-Next and of the public quantizations named below.

- **Qwen team**: Qwen3.8-Flash-Next, Qwen Community License 1.0 (copy in the `LICENSE` file of this repo), Copyright (c) 2026
  Qwen. Architecture paper: [arXiv:2608.30320](https://arxiv.org/abs/2608.30320).
- **Unsloth**: UD-IQ4_XS (base of FLE-IQ3_M), UD-IQ3_XXS and UD-Q2_K_XL (donors),
  [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
  [Unsloth Dynamic quantization](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs).
- **ISTA-DASLab**: GSQ-RCO IQ3_S, IQ3_XXS and IQ2_XS (donors and base of FLE-IQ3_XXS and FLE-IQ2_M),
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
  GSQ: Dadgarnia et al., [arXiv:2604.18556](https://arxiv.org/abs/2604.18556); RCO: Helcig and Alistarh,
  [arXiv:2605.00649](https://arxiv.org/abs/2605.00649). ISTA's repository metadata names Apache-2.0, while its model
  card says the quantized weights inherit the base model's licence; we treat the weights as covered by the Qwen
  Community License 1.0 (see `NOTICE.md`).
- **llama.cpp / ggml and GGUF** (G. Gerganov et al., MIT): <https://github.com/ggml-org/llama.cpp>. The FLE engine is a
  fork of it and also borrows from ik_llama.cpp (Kawrakow, MIT), FreeToken (Yang et al., code by FlashML-org, Apache-2.0,
  [arXiv:2608.16157](https://arxiv.org/abs/2608.16157)), KTransformers (Apache-2.0) and Strata (Niko1221, MIT); see
  the NOTICE files of the engine repo.
- **Quantization tooling**: importance matrix and IQ types in llama.cpp (I. Kawrakow, PRs #4861, #4897, #5676, #5590);
  KLD in `perplexity` (Kawrakow, PR #5076; Gaessler, PR #6936); KLD as a metric: Dutta et al.,
  [arXiv:2407.09141](https://arxiv.org/abs/2407.09141).
- **MTP and speculative decoding**: DeepSeek-V3 Technical Report [arXiv:2412.19437](https://arxiv.org/abs/2412.19437);
  Gloeckle et al. [arXiv:2404.19737](https://arxiv.org/abs/2404.19737); Leviathan et al.
  [arXiv:2211.17192](https://arxiv.org/abs/2211.17192); Chen et al. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318);
  llama.cpp MTP support #22673 (A. Gupta) and Qwen4Exp MTP #29761; model support for Qwen3.8-Flash-Next in llama.cpp:
  #27742 (D. Han).
- **Hyper-connections** (architecture): Zhu et al. [arXiv:2409.19606](https://arxiv.org/abs/2409.19606); mHC
  [arXiv:2512.24880](https://arxiv.org/abs/2512.24880).
- **MoE offloading and expert caches** (related work): Fiddler [arXiv:2402.07033](https://arxiv.org/abs/2402.07033);
  HybriMoE [arXiv:2504.05897](https://arxiv.org/abs/2504.05897); Mixtral offloading
  [arXiv:2312.17238](https://arxiv.org/abs/2312.17238); MoE-Infinity [arXiv:2401.14361](https://arxiv.org/abs/2401.14361);
  HOBBIT [arXiv:2411.01433](https://arxiv.org/abs/2411.01433); MoE-SpAc
  [arXiv:2603.09983](https://arxiv.org/abs/2603.09983); Wang et al., "Accelerating Mixture-of-Experts Inference by
  Hiding Offloading Latency with Speculative Decoding" [arXiv:2508.21706](https://arxiv.org/abs/2508.21706).
- **Upstream llama.cpp**: the engine is based on ggml-org/llama.cpp commit `6a1a922d` (2026-09-05) and carries three
  later upstream changes: #28302 (checkpoint eviction), #29599 (`llama_prefetch_rows`) and, partly and adapted,
  #29761 (Qwen4Exp MTP). The references above were checked against arXiv and GitHub on 2026-10-07.
- **Expert pruning** (FLE-IQ3_M-REAP448): REAP, Lasby et al., "REAP the Experts: Why Pruning Prevails for One-Shot MoE
  compression", [arXiv:2510.13999](https://arxiv.org/abs/2510.13999); the comparison criterion MAN, Liu et al.,
  [arXiv:2606.15716](https://arxiv.org/abs/2606.15716). Our implementation (`llama-fle-reap`, `reap_prune.py`) is in the
  engine repo.
- **Text sources for the KLD measurement**: Python documentation (PSF license), German Wikipedia (CC BY-SA 4.0),
  Kubernetes documentation (CC BY 4.0).
- **Knowledge probe**: MKQA, Longpre, Lu and Daiber, [arXiv:2007.15207](https://arxiv.org/abs/2007.15207)
  ([apple/ml-mkqa](https://github.com/apple/ml-mkqa), CC BY-SA 3.0), used through the open-domain export
  `nthakur/mkqa-open-domain`. The questions are not redistributed here.

Licence of this repo: the model files are derivatives of Qwen3.8-Flash-Next and are provided under the
**Qwen Community License 1.0** (`license: other`, `license_name: qwen-community-1.0`; the licence text is the `LICENSE`
file of this repo). In short, as we read it: use, copy, modify, distribute and fine-tune are allowed; the copyright
and licence notice must travel with copies; products with more than 100 million monthly active users or US$ 20 million
monthly revenue must display the model name; running a Model-as-a-Service or AI Work Assistant business needs a
separate licence from Qwen. This summary is not legal advice; the licence text decides.

Copyright notice required by the licence: "Copyright (c) 2026 Qwen". Condition 1 of the licence asks that this notice
and the permission notice be included in all copies or substantial portions; the `LICENSE` file of this repo is the
unchanged Qwen licence text and carries both, and `NOTICE.md` repeats the notice.

## Citation

Please cite the report and this repo. The engine release with the report is archived on Zenodo:
<https://doi.org/10.5281/zenodo.23272400>.

```bibtex
@misc{bitzer2026fle,
  author       = {Max Bitzer},
  title        = {{FLE}: Fast Local Experts. An 89 GB Mixture-of-Experts model at ~90 tokens/s on one 32 GB GPU},
  year         = {2026},
  month        = oct,
  doi          = {10.5281/zenodo.23272400},
  note         = {Technical report, version 1.0.2. Engine: https://github.com/sirfyyn/FastLocalExperts
                  (tag v1.0.1, same engine code as v1.0).
                  Model files: https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF and
                  https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF},
  howpublished = {\url{https://sirfyyn.github.io/FastLocalExperts/}}
}
```

## Authors

Max Bitzer (sirfyyn). Quantization recipes, engine patches, measurements and this card were made together with Claude
(Anthropic, Claude Code).
