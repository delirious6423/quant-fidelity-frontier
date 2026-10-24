---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
language:
- en
tags:
- gguf
- llama.cpp
- quantized
- mixture-of-experts
- qwen3_5_moe
- imatrix
- finance
- accounting
---

# Ornith-1.5-35B-A3B: Q3_K / Q4_K / Q6_K mixed quant (16.9 GB) + Q8_0 vision projector

A GGUF quantization of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (a 35B-parameter, 3B-active mixture-of-experts vision-language model by the Ornith Team) built for the best quality per gigabyte around the 16-17 GB mark, with a calibration and evaluation setup aimed at **finance/accounting text (numbers, GL codes, tables), Python, shell/terminal work and tool calling**.

**Headline numbers** (all measured by us, details and caveats below):

- **16.91 GB** (3.90 bits per weight over the 34.66 B text parameters), text model with the MTP head removed, plus a separate **Q8_0 vision projector** (0.61 GB).
- Mean KL divergence to a Q8_0 reference over eight domain slices: **0.0759** (lower is better; see the comparison for how this ranks against other ~16-20 GB quants).
- GPQA Diamond (198 questions, 2 epochs, 8192-token thinking budget): **87.4% +/- 1.7** on this exact file against 87.6% +/- 1.7 for the Q8_0 reference, with no missing answers. The difference is far inside the noise, so the honest reading is "no measurable loss on this benchmark", not "equal".

## Files

| File | Size | SHA-256 |
|---|---:|---|
| `Ornith-1.5-35B-A3B-Q3K-Q4K8-emb8-nx6-16.9GB.gguf` | 16,910,045,920 B | `6414cb2f2764fb4755d80e6a804fde402ee18479f79802445086559b626701fe` |
| `mmproj-Ornith-1.5-35B-A3B-Q8_0.gguf` (vision projector) | 614,194,400 B | `f9fbe9b82479f7f97d7ef8a91e395c170950e66513c0b39433bfc0f5f30b899c` |

The GGUF is **text-only**. Images need the projector file. GGML has no FP8 tensor type, so the projector is stored as Q8_0 (the converter's `--fp8-as-q8` equivalent).

## What is inside

| Tensor group | Tensors | Type |
|---|---:|---|
| Routed experts, layers 8-39 (gate, up, down) | 96 | Q3_K |
| Routed experts, layers 0-7 (gate, up, down) | 24 | Q4_K |
| Full-attention q/k/v/o (10 layers) | 40 | Q6_K |
| Linear-attention qkv/gate and ssm alpha/beta/out (30 layers) | 150 | Q6_K |
| Shared-expert gate/up/down (40 layers) | 120 | Q6_K |
| Output matrix | 1 | Q6_K |
| Token embedding | 1 | Q8_0 |
| Norms, routers, shared-expert gates, SSM state/conv | 301 | F32 |

Architecture `qwen35moe`: 40 layers (30 linear-attention + 10 full-attention), 256 routed experts with 8 active, 262,144 maximum context. The original checkpoint's multi-token-prediction (MTP) layer was dropped and the metadata corrected (`nextn_predict_layers = 0`), so this file cannot be used for MTP speculative decoding.

## Running it

Verified with the prebuilt llama.cpp release b11829 (commit `184f65360`, CUDA 13.4 build) on Windows:

```
llama-server -m Ornith-1.5-35B-A3B-Q3K-Q4K8-emb8-nx6-16.9GB.gguf --mmproj mmproj-Ornith-1.5-35B-A3B-Q8_0.gguf --no-mmproj-offload -c 131072 -ngl 99 --n-cpu-moe 17 -fa on -ctk q8_0 -ctv q8_0 -ub 512 --jinja --top-k 20 --top-p 0.95 --temp 0.6
```

- `--n-cpu-moe N` keeps the expert tensors of the first N layers in system RAM; everything else is offloaded. Lower N uses more VRAM and runs faster. With N = 17 and a 131,072-token context (q8_0 KV cache) the model, KV cache and compute buffers took **11,495 MiB** of VRAM on our test machine; an image request and a text request both worked at that setting.
- The token embedding (Q8_0, 0.5 GB) stays in system RAM.
- Thinking is on by default (the chat template is embedded). Disable with `--chat-template-kwargs '{"enable_thinking":false}'`. The base model's card recommends temperature 0.6 / top-p 0.95 / top-k 20 for general use.
- llama.cpp prints a warning that Qwen-VL models need at least 1024 image tokens for grounding tasks; we did not test grounding.

Only llama.cpp b11829 was tested. Other builds or engines may behave differently.

## Results

### Test setup in one paragraph

All numbers below were produced on a single machine: NVIDIA RTX 5070 Ti (16 GB), AMD Ryzen 7 7800X3D, 32 GB DDR5, Windows 11, llama.cpp b11829. Quality is measured as **KL divergence to a Q8_0 reference** (`llama-perplexity --kl-divergence`, context 2048, scored on the second half of each window, which is llama.cpp's convention) on eight text slices of 20,480 tokens each (10 windows per slice), plus the "same top token" rate (how often the model's most likely next token equals the Q8_0 model's). The Q8_0 reference was made with `llama-quantize` from the BF16 GGUF; the BF16 weights do not fit this machine's RAM, so BF16 itself was **not** used as the reference (see Limitations). The slices are described under *Data*. A "mix" of 5 windows per slice (40 windows, standard error about 0.002) was used for the design experiments.

### Quality on the eight domain slices (this model)

| Slice | Mean KLD vs Q8_0 | 99th-pct KLD | Same top token | PPL (this model) | PPL (Q8_0) |
|---|---:|---:|---:|---:|---:|
| Finance prose | 0.0838 | 1.107 | 90.7% | 6.724 | 6.471 |
| Finance tables | 0.2098 | 2.612 | 83.4% | 7.779 | 7.528 |
| Synthetic GL | 0.0988 | 1.682 | 86.6% | 3.326 | 3.311 |
| Python | 0.0190 | 0.190 | 96.5% | 1.782 | 1.762 |
| Shell | 0.0254 | 0.311 | 95.3% | 1.954 | 1.948 |
| Tool calls | 0.1178 | 2.303 | 92.8% | 2.419 | 2.490 |
| Reasoning | 0.0170 | 0.202 | 96.6% | 1.681 | 1.666 |
| General web | 0.0351 | 0.287 | 91.4% | 9.506 | 9.390 |
| **Mean of 8 slices** | **0.0759** | | **91.7%** | | |

Perplexity is shown only for context: on some slices the quantized model has a *lower* perplexity than Q8_0 (for example tool calls) while its KL divergence is clearly non-zero. We therefore used KL divergence, not perplexity, to compare builds.

### Comparison with other quants (same slices, same Q8_0 reference, same machine)

| Model | Size (GB) | Mean KLD, 8 slices | Finance + GL (3) | Code/shell/tools (3) | Same top token | tg tok/s (20 layers on CPU) |
|---|---:|---:|---:|---:|---:|---:|
| **This model (Q3_K experts, Q4_K layers 0-7, Q8_0 embedding, Q6_K rest)** | 16.91 | **0.0759** | 0.1308 | 0.0541 | 91.7% | 96.1 |
| Sibling: Q3_K experts, Q6_K attention/output, Q4_K embedding (15.8 GB) | 15.80 | 0.0831 | 0.1412 | 0.0612 | 91.4% | 100.4 |
| Previous stretch build (Q4_K embedding, 16.7 GB) | 16.66 | 0.0780 | 0.1333 | 0.0569 | 91.6% | 94.3 |
| Uniform Q3_K experts (Q8_0 attention/shared, Q4_K embedding/output) | 16.01 | 0.0853 | 0.1426 | 0.0633 | 90.8% | 97.0 |
| mudler APEX-I-Compact | 16.54 | 0.0996 | 0.1703 | 0.0735 | 90.7% | 81.4 |
| Our 3.5-bit baseline (llama.cpp-style mix, 52 Q2_K / 52 Q3_K / 16 Q4_K) | 15.16 | 0.1045 | 0.1676 | 0.0782 | 89.5% | 80.5 |
| pfeifferj GSQ-RCO 3.5-bit | 15.16 | 0.1149 | 0.1884 | 0.0846 | 89.3% | 89.7 |
| Same allocation as pfeifferj, plain imatrix codes (our ablation) | 15.16 | 0.1142 | 0.1824 | 0.0887 | 89.5% | 84.1 |
| Uniform IQ4_XS experts | 19.28 | 0.0595 | 0.1014 | 0.0458 | 92.4% | 89.3 |
| AtomicChat AD-Q4_K-IQ4_XS | 20.13 | 0.0511 | 0.0887 | 0.0402 | 93.7% | 75.2 |
| Uniform Q4_K experts | 20.29 | 0.0558 | 0.0977 | 0.0397 | 92.6% | 91.0 |
| Uniform Q5_K experts | 24.32 | 0.0462 | 0.0799 | 0.0351 | 93.2% | 82.7 |

Notes: KL values from different builds share the same tokens, but we have no paired significance test; per-slice standard errors are roughly 0.002-0.008, and about 0.0015 for the 8-slice mean. Differences below about 0.002-0.003 should be read as noise. Third-party files were downloaded unmodified and scored with the same harness; their own cards report numbers on different corpora and against BF16, so those numbers are not comparable with this table. Speed is generation (tg128) with the experts of the first 20 layers on the CPU, mainline llama.cpp kernels, DDR5.

### What each expert format costs (uniform formats, everything else unchanged)

| Expert format (all 120 routed tensors) | Size (GB) | Mean KLD, 8 slices | Finance + GL | Same top token | tg tok/s (20 layers on CPU) | tg tok/s (all experts on CPU) |
|---|---:|---:|---:|---:|---:|---:|
| Q2_K | 12.74 | 0.1484 | 0.2286 | 87.7% | 93.3 | 65.7 |
| IQ2_S | 12.49 | 0.1545 | 0.2499 | 87.6% | 68.7 | 42.4 |
| IQ3_XXS | 14.50 | 0.1014 | 0.1644 | 89.9% | 81.9 | 40.4 |
| Q3_K | 16.01 | 0.0853 | 0.1426 | 90.8% | 97.0 | 64.2 |
| IQ3_S | 16.01 | 0.1171 | 0.1841 | 89.8% | 78.2 | 49.9 |
| IQ4_XS | 19.28 | 0.0595 | 0.1014 | 92.4% | 89.3 | 57.9 |
| Q4_K | 20.29 | 0.0558 | 0.0977 | 92.6% | 91.0 | 56.7 |
| Q5_K | 24.32 | 0.0462 | 0.0799 | 93.2% | 82.7 | 50.8 |

Q2_K is by far the most expensive step (the 16 GB-class builds that mixed in Q2_K tensors all lost to Q3_K-based builds of the same size, see below). In this llama.cpp build the IQ2/IQ3 formats were also 22-37% slower than the K-quants of similar size on the CPU path.

### GPQA Diamond

Protocol: the 198 GPQA Diamond questions, 2 epochs, answer choices shuffled per question and epoch (following the dataset's own `eval.yaml`, which uses 4 epochs; we ran 2), thinking enabled with a **fixed thinking budget of 8192 tokens** (llama.cpp `--reasoning-budget`, which ends thinking when the budget is used up and forces an answer), temperature 0.6 / top-p 0.95 / top-k 20, q8_0 KV cache, answer taken from a final `Answer: X` line, a missing answer counts as wrong. The standard error is binomial over the 396 requests and slightly understates the uncertainty because epochs repeat the same questions.

| Model | Size (GB) | GPQA Diamond accuracy | Epoch 1 / Epoch 2 | No answer | Avg tokens |
|---|---:|---:|---:|---:|---:|
| Q8_0 (reference) | 37.81 | 87.6% ± 1.7 | 87.4% / 87.9% | 0 | 4333 |
| **This model (16.9 GB)** | 16.91 | 87.4% ± 1.7 | 89.4% / 85.4% | 0 | 4428 |
| Sibling build: Q3_K experts + Q6_K attention/output (15.8 GB) | 15.80 | 86.9% ± 1.7 | 88.4% / 85.4% | 1 | 4298 |
| Our ablation (pfeifferj allocation, plain imatrix codes) | 15.16 | 86.4% ± 1.7 | 86.4% / 86.4% | 0 | 4361 |
| Our 3.5-bit baseline | 15.16 | 85.9% ± 1.8 | 86.4% / 85.4% | 3 | 4338 |
| pfeifferj GSQ-RCO 3.5-bit | 15.16 | 84.8% ± 1.8 | 85.9% / 83.8% | 1 | 4330 |

The Q8_0 file here is our reference file and still contains the unused MTP tensors. The base model card self-reports 89.2 for GPQA Diamond under its own protocol (temperature 1.0, different harness and budget), which is not comparable to these numbers. The 15.8 GB sibling in the table uses Q3_K (not Q4_K) experts in layers 0-7 and a Q4_K (not Q8_0) token embedding, and has a higher KL divergence (0.0831 against 0.0759 here). At this sample size GPQA cannot separate builds that differ by one to two points: this model (87.4%), the sibling (86.9%) and Q8_0 (87.6%) are all within one standard error of each other. It shows that nothing is badly broken; it does not rank the builds.

### Speed and memory (this model, this machine)

llama-bench, 512-token prompt and 128-token generation, two repeats:

| Experts of the first N layers on CPU | Prompt (pp512) tok/s | Generation (tg128) tok/s |
|---|---:|---:|
| 40 (all experts on CPU) | 654 | 64.6 |
| 20 | 1022 | 96.1 |

With `--n-cpu-moe 17`, q8_0 KV cache: prompt 1138 +/- 164 tok/s and generation **100.2 +/- 1.7 tok/s** at depth 0; prompt 1236 +/- 5 tok/s and generation **88.3 +/- 0.6 tok/s** with 32,768 tokens of context already filled. Only 10 of the 40 layers use full attention, which is why generation slows by only about 12% at 32k context. Quality was only evaluated at a context of 2048; the 131,072-token setting was verified to load and answer, not for long-context quality.

## How it was made and what we learned

### Recipe

1. `convert_hf_to_gguf.py` (llama.cpp commit `184f65360`) turned the Hugging Face checkpoint (`ornith-ai/Ornith-1.5-35B-A3B`, revision `10fbf86fed7ecee4a061f8b499a618f46001cac1`) into a BF16 GGUF and, separately, the vision projector.
2. A Q8_0 GGUF (quality reference, also the source of the Q8_0 embedding) was made with `llama-quantize`.
3. For each expert format a full-model GGUF was quantized with `llama-quantize` using an importance matrix and per-tensor type files; the final file was then **assembled by copying the already-quantized tensors byte for byte** (a small script built on `gguf-py`), so each tensor is exactly what llama.cpp's quantizer produced.
4. The importance matrix is [bartowski's](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF) (573 chunks of 512 tokens, chat/tool/prose) accumulated with 120 chunks of 2048 tokens (about 245k tokens) of our domain calibration text computed on a 3.5-bit build with `llama-imatrix`.
5. The attention, linear-attention, shared-expert and output tensors use the Q6_K produced with the same matrix; the token embedding is the Q8_0 tensor from the reference file.

### Design experiments (mean KL divergence on the 40-window mix, lower is better, standard error about 0.002)

Starting point of the design work was an attempt to reproduce [pfeifferj's GSQ-RCO 3.5-bit build](https://huggingface.co/pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF) (Gumbel-softmax code refinement and Riemannian budget-constrained format allocation, see references). Our measurements did not show an advantage for those two steps on our metric, so the final recipe does **not** use GSQ or RCO; it came from the experiments below.

| Build | Size (GB) | Mean KLD (mix) | ± SE |
|---|---:|---:|---:|
| Uniform Q3_K (Q8_0 attention/shared, Q4_K embedding/output; control) | 16.01 | 0.0842 | 0.0021 |
| Uniform Q3_K + attention/ssm/shared experts Q6_K | 15.67 | 0.0847 | 0.0021 |
| Uniform Q3_K + attention/ssm/shared experts Q5_K | 15.48 | 0.0965 | 0.0023 |
| Uniform Q3_K + Q6_K attention/ssm/shared experts + Q6_K output (the 15.8 GB sibling) | 15.80 | 0.0813 | 0.0021 |
| Uniform Q3_K + output Q6_K only | 16.14 | 0.0806 | 0.0021 |
| 15.8 GB sibling with attention back at Q8_0 | 15.87 | 0.0819 | 0.0021 |
| 15.8 GB sibling with linear-attention/ssm back at Q8_0 | 16.04 | 0.0800 | 0.0021 |
| 15.8 GB sibling with shared experts back at Q8_0 | 15.83 | 0.0811 | 0.0021 |
| 15.8 GB sibling with output at Q8_0 | 15.92 | 0.0811 | 0.0021 |
| 15.8 GB sibling with token embedding at Q8_0 | 16.05 | 0.0814 | 0.0021 |
| Uniform Q3_K, Q4_K on layers 0-14 | 17.61 | 0.0690 | 0.0018 |
| Uniform Q3_K, Q4_K on layers 25-39 (same bytes as the layers 0-14 row) | 17.61 | 0.0773 | 0.0022 |
| Uniform Q3_K, Q4_K on every ffn_down | 17.44 | 0.0740 | 0.0020 |
| Integer-programming allocation, K-quants only, 15.0 GB | 15.00 | 0.1019 | 0.0023 |
| Integer-programming allocation, K-quants only, 15.5 GB | 15.50 | 0.0927 | 0.0021 |
| Integer-programming allocation, K-quants only, 16.0 GB | 16.00 | 0.0888 | 0.0020 |
| Integer-programming allocation, K-quants only, 16.5 GB | 16.49 | 0.0806 | 0.0020 |
| Integer-programming allocation, K + IQ3_XXS, 15.0 GB | 15.00 | 0.0937 | 0.0021 |
| Integer-programming allocation, K + IQ3_XXS, 15.5 GB | 15.50 | 0.0895 | 0.0019 |
| Integer-programming allocation, K + IQ3_XXS, 16.0 GB | 16.00 | 0.0854 | 0.0020 |
| Integer-programming allocation, K + IQ3_XXS, 16.5 GB | 16.50 | 0.0832 | 0.0021 |
| Q4_K L0-9 / Q3_K L10-29 / Q2_K L30-39 (+Q6_K base) | 16.05 | 0.0893 | 0.0019 |
| Q4_K L0-12 / Q3_K L13-22 / Q2_K L23-39 (+Q6_K base) | 15.80 | 0.0964 | 0.0021 |
| Q4_K L0-9 / IQ3_XXS L10-39 (+Q6_K base) | 15.74 | 0.0869 | 0.0020 |
| 15.8 GB sibling + Q4_K on layers 0-1 | 16.01 | 0.0812 | 0.0021 |
| 15.8 GB sibling + Q4_K on layers 0-3 | 16.23 | 0.0798 | 0.0020 |
| 15.8 GB sibling + Q4_K on layers 0-7 | 16.66 | 0.0750 | 0.0019 |
| 15.8 GB sibling + Q4_K on layers 4-11 | 16.66 | 0.0748 | 0.0021 |
| 15.8 GB sibling + Q4_K on layers 6-13 | 16.66 | 0.0731 | 0.0020 |
| 15.8 GB sibling + Q4_K on layers 8-15 | 16.66 | 0.0746 | 0.0020 |
| 15.8 GB sibling + Q4_K on layers 0,2,...,14 | 16.66 | 0.0746 | 0.0020 |
| 16.7 GB build (sibling + Q4_K on layers 0-7) + Q8_0 token embedding (THIS MODEL) | 16.91 | 0.0718 | 0.0018 |
| 16.7 GB build (sibling + Q4_K on layers 0-7) + Q8_0 output | 16.78 | 0.0748 | 0.0019 |
| 16.7 GB build (sibling + Q4_K on layers 0-7) + Q8_0 embedding and output | 17.03 | 0.0716 | 0.0018 |
| Uniform Q3_K, larger imatrix | 16.01 | 0.0840 | 0.0021 |
| Uniform IQ3_S, larger imatrix | 16.01 | 0.1088 | 0.0024 |
| Uniform IQ3_XXS, larger imatrix | 14.50 | 0.0997 | 0.0022 |
| 15.8 GB sibling, larger imatrix | 15.80 | 0.0817 | 0.0021 |
| 16.7 GB build (sibling + Q4_K on layers 0-7), larger imatrix | 16.66 | 0.0759 | 0.0020 |
| 15.8 GB sibling + Q4_K on layers 6-13, larger imatrix | 16.66 | 0.0736 | 0.0020 |
| AtomicChat recipe rebuilt (Q4_K down, IQ4_XS gate/up), Q4_K embedding+output | 19.62 | 0.0542 | 0.0017 |
| AtomicChat recipe rebuilt + output Q6_K | 19.75 | 0.0507 | 0.0017 |
| AtomicChat recipe rebuilt + embedding Q8_0 | 19.87 | 0.0493 | 0.0015 |
| AtomicChat recipe rebuilt + embedding and output Q8_0 | 20.13 | 0.0455 | 0.0015 |
| AtomicChat AD-Q4_K-IQ4_XS (their file) | 20.13 | 0.0464 | 0.0016 |
| Uniform IQ2_S experts (Q8_0 attention/shared, Q4_K embedding/output) | 12.49 | 0.1502 | 0.0027 |
| Uniform Q2_K experts (Q8_0 attention/shared, Q4_K embedding/output) | 12.74 | 0.1481 | 0.0027 |
| Uniform IQ3_XXS experts (Q8_0 attention/shared, Q4_K embedding/output) | 14.50 | 0.0989 | 0.0022 |
| Uniform Q3_K experts (Q8_0 attention/shared, Q4_K embedding/output) | 16.01 | 0.0842 | 0.0021 |
| Uniform IQ3_S experts (Q8_0 attention/shared, Q4_K embedding/output) | 16.01 | 0.1092 | 0.0024 |
| Uniform IQ4_XS experts (Q8_0 attention/shared, Q4_K embedding/output) | 19.28 | 0.0552 | 0.0017 |
| Uniform Q4_K experts (Q8_0 attention/shared, Q4_K embedding/output) | 20.29 | 0.0531 | 0.0016 |
| Uniform Q5_K experts (Q8_0 attention/shared, Q4_K embedding/output) | 24.32 | 0.0440 | 0.0015 |

Findings, in order of importance:

- **Avoid Q2_K.** At about 16 GB, every build that put Q2_K tensors in the "insensitive" layers (the 16.0 GB integer-programming build and the two hand-designed Q4_K/Q3_K/Q2_K layer ramps) lost to a Q3_K-based build of the same size. A sensitivity-weighted integer-programming allocation (probe each 5-layer group and projection, solve for the best format per tensor under a byte budget) over-predicted its gains by 0.006-0.011 and never beat uniform Q3_K at 16 GB.
- **Bytes matter more than placement.** Q4_K upgrades land roughly on the straight line between uniform Q3_K and uniform Q4_K. Putting them in early layers (0-14) was clearly better than in late layers (25-39) at equal size (0.0690 vs 0.0773), but which 8 early layers is within noise.
- **Set the non-expert tensors deliberately.** Output Q4_K to Q6_K: -4% KL divergence for 0.13 GB. Attention/SSM/shared experts at Q6_K are essentially free (0.34 GB saved); at Q5_K they cost +15%. Q8_0 for any non-expert class at the 3-bit base made no measurable difference.
- **The token embedding matters once the experts are good enough.** Q8_0 instead of Q4_K: no effect on the 3-bit base, -4% on the 16.7 GB build, about -9% on a 4-bit-expert build. On all eight slices the final file improves over the previous 16.7 GB build from 0.0780 to 0.0759 (7 of 8 slices better) for 0.25 GB. The gain is about what the same bytes buy elsewhere, so it is a modest, consistent improvement, not a breakthrough.
- **A larger importance matrix did not help.** A second matrix (490 windows of 2048 tokens on the Q8_0 model, about 40k tokens per expert against 9k for bartowski's) changed nothing beyond noise on six A/B pairs, and did not revive the IQ formats.
- **AtomicChat's 20 GB file is explained by its Q8_0 embedding and output.** Rebuilding its recipe (Q4_K down, IQ4_XS gate/up) from our own libraries reproduced its score once embedding and output were Q8_0 (0.0455 vs their file 0.0464) and was 19% worse (0.0542) with Q4_K embedding and output.
- **Early wikitext-2 perplexity runs (3.5-bit builds) were misleading,** which is why the work moved to KL divergence on domain text:

| 3.5-bit build | wikitext-2 PPL (ctx 2048, 145 windows) |
|---|---:|
| Our 3.5-bit baseline (native imatrix) | 8.0383 ± 0.057 |
| Same allocation as pfeifferj, plain imatrix codes | 7.8251 ± 0.055 |
| pfeifferj GSQ-RCO 3.5-bit | 8.3379 ± 0.060 |

## Data

Calibration and evaluation text were built from public sources and a rule-based generator, split **by document** (a document goes to evaluation only if a hash of its key falls in one bucket), with a line-overlap check (lines of 120+ characters) against every calibration set we built; the only overlaps are two boilerplate system-prompt lines in the tool-call slice and one generic sentence in the finance-table slice. Eight slices of 20,480 evaluation tokens each:

| Slice | Sources |
|---|---|
| Finance prose | [Josephgflowers/Finance-Instruct-500k](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k) (Apache-2.0) |
| Finance tables | [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench) 10-K excerpts (**CC BY-NC 4.0**, Islam et al. 2023); FLARE mirrors by ChanceFocus of TAT-QA (Zhu et al. 2021), FinQA (Chen et al. 2021) and ConvFinQA (Chen et al. 2022), see PIXIU (Xie et al. 2023); the mirror cards state no license |
| Synthetic GL | Rule-based generator written for this work: charts of accounts, balanced journal entries, trial balances, aging reports, bank reconciliations, Intacct-style XML, worked accounting arithmetic. No real company data |
| Python | Python subset of [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) (MIT; Wei et al. 2023; its card points to OpenAI's usage policy) and [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) |
| Shell / terminal | Shell subset of Magicoder-OSS-Instruct, [westenfelder/NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA) (MIT; Westenfelder et al. 2025), [tldr-pages](https://github.com/tldr-pages/tldr) |
| Tool calls | [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) (Apache-2.0) |
| Reasoning | [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) (Apache-2.0; Guha et al. 2025) |
| General web | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) sample-10BT (ODC-By; Penedo et al. 2024) |

The same sources (different documents) fed the calibration text used for the importance matrix, which therefore included FinanceBench excerpts; anyone with commercial-use concerns about that dataset's license should be aware of it. The weights in this repository contain no dataset text. Early comparison runs also used the test split of WikiText-2 (Merity et al. 2016, CC BY-SA 3.0 / GFDL). GPQA is used under its CC BY 4.0 license and terms; no GPQA question text is reproduced here or stored in our logs.

## Limitations

- **Reference is Q8_0, not BF16.** A two-window check on WikiText-2 found llama.cpp's BF16 and Q8_0 within about 1.5% perplexity per window; a layer-streamed BF16 forward pass in Hugging Face transformers agreed with llama.cpp BF16 to 0.7% cumulative perplexity. Absolute KL numbers against BF16 would be somewhat higher.
- **No task-level accuracy was measured for finance or coding work.** KL divergence and GPQA are proxies. We did not build a ledger/GL-code/JSON-validity task test.
- **GPQA covers two epochs of 198 questions with a fixed 8192-token thinking budget** and cannot resolve one-to-two-point differences; it is a sanity check, not a ranking.
- **Quality was measured at 2048-token windows.** The 131k-token configuration was only checked to load and answer; q8_0 KV-cache quality at long context is untested.
- **Vision was smoke-tested** with a single synthetic image (red circle on blue) using the Q8_0 projector; it described the image correctly, as the F16 projector did. Document, chart and grounding quality are untested.
- **One machine, one engine build.** Speeds depend on CPU memory bandwidth and on llama.cpp's kernels; the ranking of formats on the CPU path (K-quants faster than IQ-quants) was measured with this build only.
- Third-party files were compared as published by their authors under our harness; differences in calibration data, goals and corpora mean this table is not a verdict on those projects.
- The MTP head is removed; speculative decoding with the model's own MTP layer is not possible with this file.

## Credits and references

**Base model.** Ornith Team, *Ornith-1.5: From Self-Scaffolding to Self-Improvement*, 2026, [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), MIT license (its card says Ornith 1.0 was built on Qwen3.5 and Gemma 4 and that 1.5 extends it). All credit for the model belongs to its authors; this repository only re-encodes their weights.

```bibtex
@misc{ornith_1_5,
    title = {{Ornith-1.5}: From Self-Scaffolding to Self-Improvement},
    url = {https://ornith.ai/ornith_1_5.html},
    author = {{Ornith Team}},
    year = {2026}
}
```

**Software.** [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (Georgi Gerganov and contributors): conversion, `llama-quantize`, `llama-imatrix`, `llama-perplexity`, `llama-bench`, `llama-server`. The K-quant formats (Q3_K, Q4_K, Q6_K, ...) and importance-matrix quantization were developed by Iwan Kawrakow and contributors for llama.cpp. Tensor assembly used `gguf-py` from the same repository.

**Third-party quants and importance matrix we used or compared against.**
- bartowski, [Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF): the importance matrix (and its calibration text) used as the base of ours.
- pfeifferj, [Ornith-1.5-35B-A3B-GSQ-RCO-GGUF](https://huggingface.co/pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF): independent GSQ-RCO reproduction; the starting point of this work and a comparison point.
- mudler / LocalAI, [Ornith-1.5-35B-A3B-APEX-GGUF](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-GGUF) (`APEX-I-Compact`): compared.
- AtomicChat, [Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/AtomicChat/Ornith-1.5-35B-A3B-GGUF) (`AD-Q4_K-IQ4_XS`): compared.
- ByteShape, [Qwen3.6-35B-A3B GGUF release notes](https://byteshape.com/blogs/Qwen3.6-35B-A3B/): the idea of choosing quantization types per device and per tensor role.

**Methods we studied.**
- Dadgarnia, Tabesh, Nikdan, Helcig, Kurtic, Kleinegger, Alistarh. *GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling.* [arXiv:2604.18556](https://arxiv.org/abs/2604.18556), code [IST-DASLab/GSQ](https://github.com/IST-DASLab/GSQ).
- Helcig, Alistarh. *Model Compression with Exact Budget Constraints via Riemannian Manifolds* (RCO). [arXiv:2605.00649](https://arxiv.org/abs/2605.00649), code [IST-DASLab/RCO](https://github.com/IST-DASLab/RCO).

**Benchmarks and data.**
- Rein et al. *GPQA: A Graduate-Level Google-Proof Q&A Benchmark.* [arXiv:2311.12022](https://arxiv.org/abs/2311.12022), [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) (CC BY 4.0).
- Islam et al. *FinanceBench: A New Benchmark for Financial Question Answering.* [arXiv:2311.11944](https://arxiv.org/abs/2311.11944).
- Chen et al. *FinQA: A Dataset of Numerical Reasoning over Financial Data.* [arXiv:2109.00122](https://arxiv.org/abs/2109.00122).
- Zhu et al. *TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance.* [arXiv:2105.07624](https://arxiv.org/abs/2105.07624).
- Chen et al. *ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering.* [arXiv:2210.03849](https://arxiv.org/abs/2210.03849).
- Xie et al. *PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance.* [arXiv:2306.05443](https://arxiv.org/abs/2306.05443).
- Wei et al. *Magicoder: Empowering Code Generation with OSS-Instruct.* [arXiv:2312.02120](https://arxiv.org/abs/2312.02120).
- Westenfelder et al. *LLM-Supported Natural Language to Bash Translation.* [arXiv:2502.06858](https://arxiv.org/abs/2502.06858).
- Guha et al. *OpenThoughts: Data Recipes for Reasoning Models.* [arXiv:2506.04178](https://arxiv.org/abs/2506.04178).
- Penedo et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* [arXiv:2406.17557](https://arxiv.org/abs/2406.17557).
- Merity et al. *Pointer Sentinel Mixture Models* (WikiText). [arXiv:1609.07843](https://arxiv.org/abs/1609.07843).

**How this was produced.** Quantization, evaluation tooling and this card were developed by the repository owner with the help of Claude Code (Anthropic), which wrote and ran the scripts; all numbers above are taken from the generated result files. This is an independent community quantization: it is not an official Ornith release and carries no endorsement from the Ornith Team or from any project listed above.

**Change log.** 2026-10-24: first release with GPQA measured on the sibling build only. Same day: GPQA Diamond run on this exact file added (87.4% +/- 1.7); no other number changed.
