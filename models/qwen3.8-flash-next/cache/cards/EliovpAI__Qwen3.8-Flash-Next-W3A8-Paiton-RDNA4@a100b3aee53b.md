---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: paiton
pipeline_tag: text-generation
inference: false
tags: [paiton, vllm, amd, rocm, rdna4, gfx1201, int3, 3-bit, w3a8, gptq, hadamard, moe, mtp, speculative-decoding]
---


# Qwen3.8-Flash-Next 3-bit (W3A8) for Paiton on two RDNA4 GPUs

**The fastest way to run Qwen3.8-Flash-Next on two R9700: everything in VRAM, with GSM8K, HumanEval and MMLU-Pro scores that
show no significant difference to the 16-bit model.** Release of 24 Oct 2026 (`B3h-dnld4m`).

| Task | 16-bit model | **This 3-bit** | 4-bit ([W4A8](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4)) |
|---|---:|---:|---:|
| GSM8K (grade-school math, 1,319 questions) | 96.9 % | **97.4 %** | 96.8 % |
| HumanEval (Python coding, 164 problems) | 96.3 % | **94.5 %** | 96.3 % |
| MMLU-Pro (knowledge and reasoning, 1,400 questions) | 84.7 % | **83.6 %** | 83.7 % |
| AI2D (science diagrams with images, 600 questions, emulated) | 85.3 % | **81.8 %** | – |

GSM8K asks school math word problems, HumanEval asks for Python functions that must pass tests, MMLU-Pro asks hard
multiple-choice questions across 14 subjects, AI2D asks questions about science diagrams. No task difference to the 16-bit model
is statistically significant ("within noise"); for scale, the 16-bit model itself moves by up to 1.2 points between two runs
with different seeds. The 4-bit is the accuracy release, this 3-bit the speed release. [How we tested](#how-we-tested).

## Accuracy

### Task benchmarks

| Task | Questions | 16-bit | **This 3-bit** | 4-bit | 3-bit vs 16-bit (paired 95 % CI, exact McNemar p) |
|---|---:|---:|---:|---:|---|
| GSM8K | 1,319 | 96.9 % | **97.4 %** | 96.8 % | +0.5 points [−0.3, +1.3], p 0.26 |
| HumanEval pass@1 | 164 | 96.3 % | **94.5 %** | 96.3 % | −1.8 points [−5.0, +1.3], p 0.45 |
| MMLU-Pro (100 per subject × 14) | 1,400 | 84.7 % | **83.6 %** | 83.7 % | −1.1 points [−2.3, +0.0], p 0.08 |

On the MMLU-Pro questions where neither model reached the 16,384-token limit, this 3-bit scores 91.0 % and the 16-bit model
91.1 %. A confidence interval that contains 0 means the difference can be chance; p above 0.05 means the same.

### Closeness to the 16-bit model

| Weights | bits / weight (routed experts) | KL t+a, free | KL t+a, routing locked | top-1 agreement |
|---|---:|---:|---:|---:|
| bf16 (two correct implementations against each other: the floor) | 16 | 0.0077 | n/a | 96.77 % |
| **INT3, this release** | 3.125 | **0.0550** | **0.0367** | **91.18 %** |
| INT4 / INT3 hybrid as served ([W4A8](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4)) | 3.926 | 0.0306 | 0.0166 | 93.36 % |
| standard MXFP4, round-to-nearest, measured by us on the same experts | 4.25 | 0.0515 | 0.0320 | 91.69 % |

KL t+a is the average full-vocabulary Kullback-Leibler divergence from bf16's next-token distribution to this model's at the
170,884 text and assistant positions of our evaluation corpus: 0 means identical, and two correct bf16 implementations already
differ by about 0.008. "Free" lets the model choose its own experts; "routing locked" gives it bf16's choices and isolates the
rounding error. Top-1 is how often the most likely next token is the same as bf16's.

| Long documents (44 held-out documents of 8K-32K tokens) | bf16 | **This 3-bit** | 4-bit |
|---|---:|---:|---:|
| NLL on assistant text, document level | 1.7574 | **1.7636 (+0.35 %)** | 1.7603 (+0.16 %) |
| KL t+a, free / routing locked | – | **0.0549 / 0.0388** | 0.0307 / 0.0180 |
| top-1 agreement | – | **90.48 %** | 92.75 % |

NLL is how surprised a model is by real text (lower is better); +0.35 % means slightly more surprised than bf16.

**On the running server** (two R9700): top-64 KL against bf16 0.0563 and top-1 90.72 % on 86,672 positions, within +0.003 of the
emulation (checked with 512-token prompt chunks instead of the shipped 2,048). Needle retrieval at 131,072 tokens: 40 of 40.

**Image questions.** AI2D (600 four-option questions, image up to 4K, letter scoring) through an emulation of the served
arithmetic with the bf16 vision tower: **81.8 %** (bf16 85.3 %); without the image 65.8 % (bf16 67.0 %). The released 3-bit modes
take text only.

**Long agentic sessions.** Beyond about 28K tokens of context the 3-bit is more likely than bf16 to end a reply early: at the real
stop points of long agent sessions it stops with probability 0.452 (bf16 0.328); genuine reply ends are kept (0.703, bf16 0.602).
For long coding sessions prefer the 4-bit.

**Speculative decoding.** The MTP layer's experts are INT4 (GPTQ on this target). Expected accepted drafts per step at depth 3
(teacher-forced, greedy, assistant text): 1.774 with the exact head, 1.764 with the shipped int2 draft head (bf16 MTP layer on
the bf16 model 1.789). The launcher's native modes use the distilled drafter that ships in the runtime image.

### How we tested

- **Hardware.** Two AMD Radeon AI PRO R9700 (32 GB each), PCIe 4.0 x16 host with 251 GiB of RAM. The 16-bit reference does not
  fit there (about 360 GB): it is the original checkpoint served with stock vLLM on a larger AMD GPU server.
- **Serving.** Task scores come from the server exactly as the launcher ships it (`--mode decode`), through its OpenAI-compatible
  API; the 16-bit and 4-bit rows are the runs of the [4-bit card](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4#task-benchmarks).
- **Task settings.** Thinking on (the default chat template); temperature 1.0, top-p 0.95, top-k 20; replies capped at 16,384
  tokens, a capped reply counts as wrong; the same prompts and per-question seeds for every model. Full GSM8K (1,319, 0-shot,
  `\boxed{}` answer), HumanEval (164, tests run in a sandbox), MMLU-Pro (1,400: 100 per subject, 14 subjects). Differences are
  paired per question: 95 % confidence intervals and an exact McNemar test.
- **Closeness.** KL, top-1, NLL, AI2D and the MTP numbers come from an emulation of the served arithmetic (the same expert,
  trunk, attention and head arithmetic the runtime executes); the server KL and needles from the running server.
- **Speed.** BetterBench 0.6.0 standard profile (below); LocalMaxxing v0.1.54: one warm-up request, three runs on one server,
  the median run kept.

## The weights

3-bit weights for [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (revision `de4b8e4d`), quantised
with our own GPTQ run and served by Paiton's native RDNA4 runtime on two AMD Radeon AI PRO R9700 cards. The whole language
model is resident in the two cards' VRAM; the n-gram embedding table lives in pinned system RAM.

**GPUs:** 2 × Radeon AI PRO R9700 (`gfx1201`, 32 GB each) · **System RAM:** 128 GB or more · **Size on disk:** 108.5 GiB
(116.5 GB) · **Context:** 262,144 tokens natively; served at 98,304 (speculative decoding on), 200,000 (off) and 262,144 (native
prefix-caching modes) · **Text weights, MTP layer and draft head; bf16 vision tensors included**

Recipe: sequential GPTQ with per-expert down-projection Hessians (Ledoit-Wolf shrinkage to the diagonal, as in the 4-bit)
on 4 million calibration tokens. `common/` is shared with the earlier 3-bit weights.

## Speed

BetterBench 0.6.0 standard profile through the
OpenAI-compatible server on two R9700 (tensor parallel 2), exact wire, unique prompts (no prefix-cache hits), 48 completed
requests per concurrency level; one run on this revision, 24 Oct 2026:

| `--mode decode --fn-worker native` | This revision |
|---|---:|
| Decode, 1 request, weighted over 8 categories | **258.7 tok/s** |
| Aggregate throughput, 8 concurrent requests | **687.5 tok/s** |
| Prefill, 64K-token prompt | **8,433 tok/s** |
| Time to first token, short prompts (p50) | **56 ms** |

The torch-worker default (`--mode decode`) serves 216 tok/s single-request decode and 565 tok/s at 8 requests. For
agents and coding tools the native worker has prefix caching (`--prefix-caching`): a repeated 90K-token prompt reaches its
first token in about 0.3 s instead of 10.5 s. All modes and numbers: the
[launcher README](https://github.com/Eliovp-BV/paiton-vllm-plugin/tree/main/models/Qwen3.8-Flash-Next).

## How to run

The weights are served by the Paiton container through the
[public Paiton plugin repository](https://github.com/Eliovp-BV/paiton-vllm-plugin), which pins this revision and the runtime
image and provides the launcher under `models/Qwen3.8-Flash-Next`. Transformers, stock vLLM and llama.cpp cannot load this
format.

```bash
pip install huggingface_hub
bash models/Qwen3.8-Flash-Next/run-flashnext.sh --fn-worker native                    # fastest: native worker
bash models/Qwen3.8-Flash-Next/run-flashnext.sh --fn-worker native --prefix-caching    # agents and coding tools
bash models/Qwen3.8-Flash-Next/run-flashnext.sh                                        # torch-worker default
python3 models/Qwen3.8-Flash-Next/launch-flashnext.py --mode decode --dry-run          # checks RAM, GTT, GPU links, disk
```

The first start downloads and verifies the weights (SHA256SUMS). On a host with older 3-bit weights, the launcher downloads them into a fresh folder (`~/paiton-models/Qwen3.8-Flash-Next-W3A8-71faf406`), never mixed with the first release, and says once that the old folder can be deleted; an explicit `--weights DIR` with other weights is served as is, with a warning. Native image:
`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261024-pc-rc3`
(digest `sha256:5bca558833ad230ab1e8db7e6a64f538a7396c44dd660b69716a5aec4e714640`). Manual download (the optional
`contract-probe/` directory of 2.2 GiB can be skipped):

```bash
hf download EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4 --local-dir "$PAITON_FLASHNEXT_DIR"
(cd "$PAITON_FLASHNEXT_DIR" && sha256sum -c SHA256SUMS)
```

## Hardware requirements

| Component | Requirement |
|---|---|
| GPUs | Two AMD Radeon AI PRO R9700 (`gfx1201`, 32 GB each) in one host; peak 30.3-31.5 GiB per card |
| System RAM | **128 GB or more.** The n-gram table (48.9 GiB) and other host tables (about 55 GiB over both GPUs) are pinned in host memory; the native worker needs about 60 GiB of free GTT (half the RAM by default). The native 3-bit prefix-caching modes without speculative decoding start from about 64 GB |
| Disk | 108.5 GiB for this repository; `contract-probe/` (2.2 GiB) is optional |
| Host | Linux x86-64, ROCm 10 host driver, Docker with `/dev/kfd` and `/dev/dri` access |

## Format

- **Routed experts.** All 512 experts of the 48 MoE layers: 8-level codebook ±{1, 3, 5, 8} with an E4M3 scale per
  (row, 64 weights) times a per-matrix power of two, 3.125 bits per weight, in a block-diagonal Hadamard-128 basis; expert
  inputs are quantised to fp8 (E4M3) per token after the online Walsh-Hadamard transform. Calibration: sequential GPTQ on
  4 million tokens with per-expert down-projection Hessians (Ledoit-Wolf shrinkage to the diagonal); best of three cuts on a
  pre-registered selection half, reported on the other half.
- **Trunk.** W8A16: E4M3 per output row with bf16 activations.
- **Output head.** int8 per (row, 128) with a signed bf16 scale and fp32 logits.
- **n-gram (PLE) table.** E4M3 with fp32 row scales in pinned system RAM, never on the GPU.
- **MTP layer, draft head, vision.** INT4 MTP experts (re-cut on this target), int2 draft head; vision encoder bf16 as in
  the checkpoint.
- `FORMAT.md` is the byte-level specification, `QUALITY.md` the accuracy record, `manifest.json` (name `FN3-B3h-dnld4m`)
  the per-tensor index, `SHA256SUMS` the checksum of every file.

## Limitations

- The weights are lossy (tables above); the [4-bit](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4)
  is the choice for maximum accuracy.
- Early reply ends in very long agentic sessions (beyond ~28K tokens), more often than bf16 ([above](#closeness-to-the-16-bit-model)).
- Image input is not part of the released 3-bit modes (the AI2D numbers are emulated). Multilingual and tool-calling quality were
  not measured.
- Two R9700 cards only; tied to checkpoint revision `de4b8e4d` and to Paiton's rotation and activation rules.

## License and attribution

The base model [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is released by Qwen under the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE); these weights are a derivative
and are distributed under the same license. The quantisation method builds on GPTQ. Calibration data: a 4-million-token
mixed-domain corpus of our own (prose, code and assistant conversations). Performance measured with
[BetterBench](https://github.com/GGZ14/BetterBench) 0.6.0 by GGZ14.

Compute: this revision took 8.4 GPU-hours on an AMD Instinct MI355X box (7.4 for three cuts, 1.0 for evaluations), on top of
the 169.0 GPU-hours of the original 3-bit quantisation round.

[Paiton](https://eliovp.com/products/paiton) · [Public Paiton plugin](https://github.com/Eliovp-BV/paiton-vllm-plugin)
