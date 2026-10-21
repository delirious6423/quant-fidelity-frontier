---
license: mit
license_link: LICENSE
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- mxfp4
- moe
- mtp
- whirl
- amd
- rdna4
- rdna3.5
- llama.cpp
---

# Ornith-1.5-35B-A3B MXFP4 GGUF

MXFP4 GGUF quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), a mixture-of-experts model with **~35B total and ~3B active parameters per token**, with the **MTP (multi-token prediction) layer kept** for self-speculative decoding, plus an F16 vision projector. Made for [WHIRL](https://github.com/tsaipifong/whirl-llm) 0.2.0, now on AMD Ryzen AI Max+ 395 (Radeon 8060S) and Radeon AI PRO R9700.

**Rebuilt 2026-10-19:** the language-model file was re-quantized with an **imatrix-weighted MXFP4 quantizer** and an importance matrix computed for this exact checkpoint (same tensor layout, same file name). Against the BF16 model its KL divergence is 19% lower than the first release (2026-10-03) and its perplexity is lower too; see [Quality](#quality). The vision projector is unchanged.

> **Unofficial community quantization.** This repository is not affiliated with, endorsed by, or supported by the Ornith team (ornith-ai). For the official model, official GGUF builds and evaluations, see [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) and [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF).

Ornith-1.5-35B-A3B is the Ornith team's coding- and agent-oriented MoE model (architecture `qwen35moe`: Gated DeltaNet + gated attention hybrid, 40 layers, 256 experts with top-8 routing plus a shared expert). Please read the upstream model card for its training approach and benchmark results; nothing in this card re-evaluates the model's benchmark results (the only quality measurements here are KL divergence and perplexity against the BF16 model, see [Quality](#quality)).

> ### Best performance with WHIRL, now on AMD Ryzen AI Max+ 395 (Radeon 8060S) and Radeon AI PRO R9700
> These files were made for **[WHIRL](https://github.com/tsaipifong/whirl-llm)** (Windows HIP Inference for RDNA LLMs), our open-source (Apache-2.0) native-Windows C++/HIP inference engine for the **Radeon 8060S** of the **AMD Ryzen AI Max+ 395** (new in 0.2.0) and the **AMD Radeon AI PRO R9700** — download [**release v0.2.0**](https://github.com/tsaipifong/whirl-llm/releases/tag/v0.2.0): unzip and run, only the AMD driver needed. With this file WHIRL 0.2.0 is faster than the official llama.cpp b11214 (ROCm and Vulkan builds) in every mode and at every measured prompt length (2k to ~250k tokens) on both GPUs ([tables below](#measured-performance)). WHIRL has dedicated MXFP4 expert kernels and MTP + n-gram speculative decoding whose output, in its precise and balance modes, is identical to plain greedy decoding of that mode.
>
> The files are standard GGUF and **also run in llama.cpp**: the first release was tested with llama.cpp **b11214** (ROCm build) — `llama-bench`, `llama-server` (plain and `--spec-type draft-mtp`) and `llama-mtmd-cli` with the projector; the 2026-10-19 rebuild with **b11719** (CPU `llama-perplexity`; Vulkan `llama-server` with `--spec-type draft-mtp` and the projector). Other llama.cpp builds and other GGUF runtimes were not tested.

## Files

| File | Contents | Size (bytes) | SHA256 |
|---|---|---|---|
| `Ornith-1.5-35B-A3B-MXFP4.gguf` | language model + MTP layer (MXFP4, see recipe) | 19,819,767,584 | `8fa77518492831be0c5487fecbb0a04ca10d60d9c342235c59361af4f8cef07a` |
| `mmproj-Ornith-1.5-35B-A3B-F16.gguf` | vision encoder / projector, F16 | 899,283,296 | `7950af74588cfb1acc3dd37974baaa3cceda4acf4d792cc643b4a0adb394e041` |
| `Ornith-1.5-35B-A3B-imatrix.gguf` | importance matrix used for the quantization (llama-imatrix GGUF; not needed to run the model) | 384,359,808 | `21216dbe71161194ee85aa0464754a0d7ac913acb935fc44d4d746570438e703` |

The language-model file is 4.46 bits per weight on average and takes 18.45 GiB of VRAM when loaded (plus KV cache). The projector is needed only for image input.

`SHA256SUMS` lists the same hashes. The language-model file is the 2026-10-19 rebuild; the first-release file (2026-10-03) had the same name, was 448 bytes smaller (metadata only; SHA256 `6735282d…`) and remains in this repository's commit history.

## Usage with WHIRL

Download `whirl-0.2.0-windows-x64.zip` from the [release page](https://github.com/tsaipifong/whirl-llm/releases/tag/v0.2.0), unzip, run; **only the AMD driver is needed** (no HIP SDK, ROCm or Visual C++ runtime). Requirements: Radeon 8060S (AMD Ryzen AI Max+ 395) or AMD Radeon AI PRO R9700, Windows 11 64-bit, AMD Software: Adrenalin Edition 26.8.1 (driver 32.0.31041.1004) or newer (older drivers untested). In PowerShell:

```powershell
# one prompt (MTP + n-gram speculative decoding is on by default)
.\whirl.exe chat C:\models\Ornith-1.5-35B-A3B-MXFP4.gguf "用 Python 寫一個 LRU cache，附上 pytest 測試。" --max-tokens 800

# image input
.\whirl.exe chat C:\models\Ornith-1.5-35B-A3B-MXFP4.gguf "Describe this image." `
  --mmproj C:\models\mmproj-Ornith-1.5-35B-A3B-F16.gguf --image photo.png --max-tokens 400

# OpenAI-compatible server (4 slots, continuous batching, prefix cache with RAM / SSD tiers)
.\whirl-server.exe C:\models\Ornith-1.5-35B-A3B-MXFP4.gguf --port 8080 `
  --mmproj C:\models\mmproj-Ornith-1.5-35B-A3B-F16.gguf
```

WHIRL 0.2.0 has three numerics modes: balance (near-lossless, default), `--precise` (bitwise lossless) and `--fast` (opt-in: maximum speed within gated KL limits, for example 4-bit KV). In precise and balance, MTP + n-gram drafts, batching and cache restores give the same tokens as plain greedy decoding of that mode.

The first run with a new model file tunes the prefill kernels once (about 3 seconds for this file on the R9700; cached afterwards). Every option is in the [usage reference](https://github.com/tsaipifong/whirl-llm/blob/main/docs/usage.md).

## Usage with llama.cpp

Plain decoding (the configuration that was fastest for this model in llama.cpp b11214, see the performance table):

```
llama-server -m Ornith-1.5-35B-A3B-MXFP4.gguf -ngl 99 -c 131072 -np 1 -fa on -ub 2048 -b 2048 \
  --port 8080 --jinja --mmproj mmproj-Ornith-1.5-35B-A3B-F16.gguf
```

The MTP layer can be used with `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-p-min 0.3`; on this MoE model it was slower than plain decoding in b11214 for generation prompts, but faster on file-editing prompts with `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 1`. Without `--spec-*` flags llama.cpp logs the MTP-layer tensors as unused, which is expected. The loader prints `unknown type mxfp4` as a warning; loading succeeds.

Image input:

```
llama-mtmd-cli -m Ornith-1.5-35B-A3B-MXFP4.gguf --mmproj mmproj-Ornith-1.5-35B-A3B-F16.gguf \
  --image photo.png -ngl 99 -c 8192 -fa on -p "Describe this image."
```

Sampling recommended by the upstream card: temperature 0.6, top_p 0.95, top_k 20 for general tasks (temperature 1.0 to reproduce its benchmarks). Thinking is on by default in the chat template.

## Quantization recipe

1. **Convert** the official BF16 checkpoint ([ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) revision `10fbf86fed`, every file checked against its Hugging Face hash, list in `SOURCE_SHA256.txt`) with llama.cpp `convert_hf_to_gguf.py --outtype bf16` (build **b11719**, commit 5aeac37a3). The MTP layer is kept (`nextn_predict_layers = 1`, layer `blk.40`); tensor names match the official GGUF. The tokenizer and chat template are embedded by the conversion script.
2. **Importance matrix**: computed by us for this exact checkpoint (no published imatrix was made from revision `10fbf86fed`: the upstream re-uploaded one weight shard on 2026-08-23, after the community imatrices were made). `llama-imatrix --process-output -c 512` (llama.cpp b11719, Vulkan, Radeon 8060S) on a Q8_0 conversion of the BF16 GGUF, over 982 chunks of 512 tokens of calibration text: bartowski's calibration_datav6 plus C++/HIP source code with Chinese comments and Chinese technical prose, with no overlap with the text used for the KL / perplexity measurement in [Quality](#quality). 511 entries: every matrix of the 40 main layers (all 256 experts covered) and `output`; none for `token_embd` or the MTP layer. Included in this repository as `Ornith-1.5-35B-A3B-imatrix.gguf` (SHA256 `21216dbe71161194ee85aa0464754a0d7ac913acb935fc44d4d746570438e703`); details in [`RECIPE.md`](RECIPE.md).
3. **Quantize** with `llama-quantize` built from llama.cpp **b11719 (commit 5aeac37a3) plus [`mxfp4_wsearch.patch`](mxfp4_wsearch.patch)**, an imatrix-weighted MXFP4 block-scale search. Upstream `llama-quantize` ignores the importance matrix for MXFP4 and always takes the block exponent from the block maximum; the patch tries that exponent and its two neighbours for every 32-value block and keeps the one with the lowest imatrix-weighted squared error. Only quantize-time code changes: the output is the standard MXFP4 block format, so the file loads in unmodified llama.cpp and WHIRL. Same tensor-type file as the first release (first matching rule wins):

```
blk\.40\.ffn_gate_inp=f32
blk\.40\.=q8_0
ssm_alpha=q8_0
ssm_beta=q8_0
token_embd=q8_0
attn_gate=mxfp4
attn_qkv=mxfp4
attn_q\.=mxfp4
attn_k\.=mxfp4
attn_v\.=mxfp4
attn_output=mxfp4
ffn_gate_exps=mxfp4
ffn_up_exps=mxfp4
ffn_down_exps=mxfp4
ffn_gate_shexp=mxfp4
ffn_up_shexp=mxfp4
ffn_down_shexp=mxfp4
ssm_out=mxfp4
```

```
llama-quantize --imatrix Ornith-1.5-35B-A3B-imatrix.gguf --tensor-type-file tt_orn.txt --output-tensor-type q6_k \
  ornith-1.5-35b-bf16.gguf Ornith-1.5-35B-A3B-MXFP4.gguf MXFP4_MOE 24
```

(`llama-quantize` = the patched build. The first release, 2026-10-03, used the same tensor-type file with upstream `llama-quantize` from build 11214, commit 2ebd9ae62, without an importance matrix. The rebuild has the same 753 tensor names and types; only the MXFP4 block scales / codes and the Q6_K head differ.)

| Tensor group | Type |
|---|---|
| Routed experts (`ffn_gate/up/down_exps`), shared expert (`ffn_*_shexp`), and every dense matrix of the 40 main layers (`attn_qkv`, `attn_gate`, `attn_q/k/v`, `attn_output`, `ssm_out`) | MXFP4 |
| `token_embd`; `ssm_alpha`, `ssm_beta`; the whole MTP layer `blk.40` (except its router) | Q8_0 |
| Routers (`ffn_gate_inp`, including `blk.40`), `ffn_gate_inp_shexp`, norms, small SSM tensors | F32 |
| `output` (LM head) | Q6_K |

4. **Vision projector** (unchanged from the first release): `convert_hf_to_gguf.py --mmproj --outtype f16` from the same BF16 weights, no further quantization (the official GGUF repo ships a BF16 projector; this one is F16).

Why these choices: `ssm_alpha` / `ssm_beta` in Q8_0 rather than F32 keep WHIRL's single-token and multi-token (MTP verify) paths on the same fused kernel, so speculative output stays bit-identical to plain greedy decoding; a Q6_K head is the fastest head for WHIRL's draft step; Q8_0 for embeddings and the MTP layer costs a little speed for slightly higher draft acceptance. More detail: [WHIRL quantization guide §6](https://github.com/tsaipifong/whirl-llm/blob/main/docs/guide/en/quantization.md#ornith-mxfp4).

## Measured performance

WHIRL 0.2.0 in all three modes vs the **official llama.cpp b11214**, its **ROCm** and its **Vulkan** build in separate columns (lead columns for balance, the default; precise and fast lead in every cell too). Same GGUF file, same prompts (a long-document question with a unique first line: explain a function quoted in the document and rewrite it, 256 generated tokens, so part of the answer can be copied; [exact prompts](https://github.com/tsaipifong/whirl-llm/blob/main/docs/benchmarks.md#prompts)), greedy, thinking off, one user, tok/s from each server's own timings. WHIRL measured 2026-10-17 with the first-release file; the 2026-10-19 rebuild has the same format and tensor types, and its speed was re-checked against the first-release file (see [Quality](#quality), other checks). llama.cpp decodes this MoE model **without speculation** (its MTP was slower than plain decoding here), WHIRL with MTP: the decode rows compare both engines **as shipped**, not like for like. "~250k" is a prompt of 249,510 tokens. Full methodology: [WHIRL benchmarks](https://github.com/tsaipifong/whirl-llm/blob/main/docs/benchmarks.md). The R9700 in this machine is a USB4 eGPU (affects model loading and host-RAM / SSD restores, not prefill or decode).

Radeon 8060S (max power (80 W), ASUS ROG Flow Z13), WHIRL balance vs llama.cpp ROCm / Vulkan: prefill at 8k 2,244 vs 1,282 / 749.2 tok/s; decode after a 128k-token document 58.4 vs 38.0 / 37.3. R9700: prefill 8,469 vs 3,872 / 3,395 at 32k (+119% / +149%), 4,275 vs 2,307 / 1,922 at 128k; decode after a 2k-token document 238.4 vs 115.4 / 115.9; coding chat (new text) 241.5 vs 118.9 / 115.1.

### AMD Radeon 8060S (Ryzen AI Max+ 395), max power (80 W), ASUS ROG Flow Z13

WHIRL single run per cell; llama.cpp median of 1 to 3 runs.

#### Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active): Prefill tok/s

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 2,044 | **2,411** | 2,401 | 1,259 | 447.7ᶜ | +91% | +439%ᶜ |
| 8k | 1,867 | **2,244** | 2,232 | 1,282 | 749.2 | +75% | +199% |
| 32k | 1,577 | **1,829** | 1,830 | 1,049 | 756.2 | +74% | +142% |
| 64k | 1,299 | **1,463** | 1,466 | 848.9 | 603.6 | +72% | +142% |
| 96k | 1,108 | **1,227** | 1,229 | 710.4 | 500.5 | +73% | +145% |
| 128k | 966.3 | **1,055** | 1,059 | 636.6 | 428.4 | +66% | +146% |
| ~250k | 644.7 | **685.0** | 697.9 | 418.3 | 278.9 | +64% | +146% |

#### Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active): Decode tok/s, document QA (256-token answer: explain a function quoted in the document and rewrite it, so part of the answer can be copied)

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 91.6 | **101.4** | 93.8 | 70.0 (plain) | 70.7 (plain) | +45% | +43% |
| 8k | 87.5 | **94.9** | 93.6 | 64.3 (plain) | 65.2 (plain) | +48% | +46% |
| 32k | 76.3 | **83.8** | 86.1 | 57.9 (plain) | 57.1 (plain) | +45% | +47% |
| 64k | 67.9 | **69.2** | 76.2 | 49.0 (plain) | 48.4 (plain) | +41% | +43% |
| 96k | 58.5 | **63.9** | 67.0 | 42.7 (plain) | 42.2 (plain) | +50% | +51% |
| 128k | 51.1 | **58.4** | 62.7 | 38.0 (plain) | 37.3 (plain) | +54% | +57% |
| ~250k | 38.9 | **40.4** | 48.9 | 26.3 (plain) | 25.8 (plain) | +54% | +57% |

ᶜ 2k prefill on the 8060S, llama.cpp Vulkan: Vulkan prefill is much slower on short prompts on this GPU (well below its own trend at 8k+), so the 2k lead over Vulkan is not used as a headline; headline leads over Vulkan use 8k and longer.

### AMD Radeon AI PRO R9700 (32 GB), single run per cell

#### Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active): Prefill tok/s

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 7,575 | **11,550** | 11,801 | 4,014 | 3,551 | +188% | +225% |
| 8k | 6,846 | **10,601** | 10,501 | 4,487 | 3,912 | +136% | +171% |
| 32k | 5,959 | **8,469** | 8,450 | 3,872 | 3,395 | +119% | +149% |
| 64k | 4,873 | **6,451** | 6,420 | 3,173 | 2,735 | +103% | +136% |
| 96k | 4,098 | **5,156** | 5,143 | 2,667 | 2,280 | +93% | +126% |
| 128k | 3,520 | **4,275** | 4,270 | 2,307 | 1,922 | +85% | +122% |
| ~250k | 2,351 | **2,664** | 2,666 | 1,519 | 1,218 | +75% | +119% |

#### Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active): Decode tok/s, document QA (256-token answer: explain a function quoted in the document and rewrite it, so part of the answer can be copied)

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 181.8 | **238.4** | 208.3 | 115.4 (plain) | 115.9 (plain) | +107% | +106% |
| 8k | 174.6 | **212.9** | 191.3 | 112.3 (plain) | 111.9 (plain) | +90% | +90% |
| 32k | 154.2 | **199.1** | 183.5 | 101.4 (plain) | 100.6 (plain) | +96% | +98% |
| 64k | 138.6 | **169.1** | 174.9 | 90.7 (plain) | 89.6 (plain) | +86% | +89% |
| 96k | 126.4 | **148.1** | 156.2 | 81.9 (plain) | 81.1 (plain) | +81% | +83% |
| 128k | 121.3 | **131.3** | 152.4 | 75.3 (plain) | 73.8 (plain) | +74% | +78% |
| ~250k | 90.0 | **101.7** | 125.3 | 54.1 (plain) | 53.6 (plain) | +88% | +90% |

### Coding chat (new text)

512 tokens; the answer (explanation and code) is all newly written, with nothing in the prompt to copy, a hard case for speculative drafts.

| GPU | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| Radeon 8060S (max power) | 94.5 | **106.0** | 102.8 | 70.6 (plain) | 72.9 (plain) | +50% | +45% |
| Radeon AI PRO R9700 | 191.2 | **241.5** | 219.0 | 118.9 (plain) | 115.1 (plain) | +103% | +110% |

These are speed measurements only. In WHIRL's precise and balance modes the speculative output is identical to plain greedy output of that mode, so the MTP + n-gram speed-up does not change the generated text. WHIRL's fast mode is not fully optimized yet: on this model it decodes slower than balance below about 64k tokens and faster at longer context.

## Quality

KL divergence and perplexity against the **BF16 model** (the unquantized Ornith-1.5-35B-A3B weights converted to a BF16 GGUF), measured with llama.cpp b11719 `llama-perplexity` on the CPU (official release build), `-c 512`, 80 chunks of 512 tokens of a mixed text: a C++/HIP source file with Chinese comments (70,000 characters) plus Chinese technical prose (40,000 characters), the same text and settings as our Swift MXFP4 card. BF16 perplexity on this text: 6.2537 ± 0.1226. Measured 2026-10-20; "first release" = the 2026-10-03 file with the same name. Lower KLD and higher same-top are better.

| File | Mean KLD vs BF16 | 99th-percentile KLD | Same top token as BF16 | PPL | PPL / PPL(BF16) |
|---|---|---|---|---|---|
| **this rebuild** | **0.1316 ± 0.0019** | **1.206** | **85.81%** | **6.5151** | **1.0426 ± 0.0047** |
| first release | 0.1618 ± 0.0023 | 1.449 | 84.61% | 6.6594 | 1.0657 ± 0.0053 |

- Same tensor types in both files (MXFP4 for every large matrix of the 40 main layers, see the recipe); the rebuild differs only in the MXFP4 block scales and codes (imatrix-weighted search) and an imatrix-weighted Q6_K head. KLD is 19% lower and the perplexity gap to BF16 shrinks from 6.6% to 4.3% (paired ratio ± its standard error).
- This layout puts the attention and shared-expert matrices in MXFP4 too, so the loss against BF16 is larger than in our dense Swift file; both builds are 4-bit and stay measurably above BF16 perplexity. One text and one tokenizer; this is not a downstream task evaluation, and these numbers are not comparable with other models.
- (The first release's card compared perplexity with another 4-bit GGUF format on another corpus; that table was replaced by this BF16 comparison.)

Other checks of the rebuilt file (2026-10-19/20, WHIRL 0.2.0 and llama.cpp b11719):

- `whirl selftest` and `whirl seqtest` (multi-request paths vs single request) in precise / balance / fast on the Radeon 8060S, and `whirl seqtest` in all three modes on the R9700: all passed.
- WHIRL MTP + n-gram output was identical to plain greedy output on a Chinese and an English coding prompt (balance mode, Radeon 8060S). WHIRL's KL gate (balance and fast vs precise, Chinese and code text, 8k–128k tokens) passed 20 of 20 cases on each GPU, and WHIRL's golden-hash regression references were re-recorded for the rebuilt file on both GPUs.
- Image input with the F16 projector (WHIRL server on the Radeon 8060S, and llama.cpp b11719 Vulkan `llama-server` with `--spec-type draft-mtp`): a Traditional Chinese / English sign was read correctly ("營業時間：週一至週五", "上午 9:30 - 下午 6:00", "Closed on Sundays", "Tel: 02-2345-6789"; the only deviation is the en dash written as a hyphen) by the rebuilt and the first-release file in both engines; the synthetic image of the first-release card (red square, blue circle, the text "WHIRL 42" above them) was described correctly with the exact text by both files in WHIRL. A second synthetic image with "WHIRL 42" in large bold text below the shapes was misread by both files in both engines (for example "WHILE that" / "WHILE THAT'S HOT"), so this is not a difference between the builds.
- Speed (WHIRL 0.2.0, balance, 2k- and 32k-token document prompts, 3 interleaved rounds per file, plain decoding to remove draft-acceptance noise): no measurable difference between the rebuild and the first release. Radeon 8060S: prefill 2,322 vs 2,334 tok/s at 2k and 1,850 vs 1,850 at 32k, decode 80.3 vs 80.5 and 64.2 vs 64.3 tok/s (medians, with the same prefill tuning table; a fresh tuning run on this GPU gives about 2% lower 2k prefill for either file). R9700: prefill 10,854 vs 10,269 tok/s at 2k (overlapping ranges) and 9,133 vs 9,137 at 32k, decode 171.1 vs 170.1 and 143.4 vs 143.2 tok/s.

For the model's own evaluations, see the upstream card. A 4.46-bpw MXFP4 file is expected to lose some accuracy against BF16; if that matters for your use, compare against the official Q8_0 / BF16 GGUFs.

## License

The upstream model is published under the **MIT License** (declared in the [upstream model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) metadata). These quantized files are a derivative of it and are distributed under the same license:

- [`LICENSE`](LICENSE): the MIT License text. At the time of this release the upstream repository declares `license: mit` but contains no LICENSE file (its `license_link` returns "Entry not found"), so this is the standard MIT License text with the Ornith team named as copyright holder.
- [`NOTICE`](NOTICE): attribution and a description of the changes made in this redistribution.

The upstream card states that the Ornith family was developed on top of Qwen3.5 and Gemma 4; see the upstream card for the provenance of the base weights.

## Credits

- **Ornith team (ornith-ai)** — [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), the model these files are made from.
- **llama.cpp / ggml** contributors — conversion, quantization and inference tooling.
- **[bartowski](https://huggingface.co/bartowski)** — the calibration text calibration_datav6, part of our imatrix calibration data.
- Quantization and measurements: [tsaipifong](https://huggingface.co/tsaipifong). See also our [Swift-1.5-Qwen3.8-27b MXFP4 GGUF](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF) (dense 27B).

Upstream citation:

```bibtex
@misc{ornith_1_5,
    title = {{Ornith-1.5}: From Self-Scaffolding to Self-Improvement},
    url = {https://ornith.ai/ornith_1_5.html},
    author = {{Ornith Team}},
    year = {2026}
}
```

## 繁體中文摘要

- 這是 [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) 的**非官方社群 MXFP4 GGUF 量化**，與 Ornith 團隊無關。MoE 模型，總參數約 35B、每個 token 約 3B 啟用；保留 MTP 層（Q8_0），可做自我推測解碼；另附 F16 視覺投影檔。
- 專為 **[WHIRL](https://github.com/tsaipifong/whirl-llm)**（Apache-2.0，AMD Ryzen AI Max+ 395 的 Radeon 8060S（0.2.0 新增）與 AMD Radeon AI PRO R9700 的原生 Windows C++/HIP 推論引擎，[v0.2.0 下載](https://github.com/tsaipifong/whirl-llm/releases/tag/v0.2.0)：下載 `whirl-0.2.0-windows-x64.zip`、解壓縮即可執行，只需 AMD 驅動 Adrenalin 26.8.1 或更新版）製作；也能在 llama.cpp 執行（已用 b11214 測試：llama-bench、llama-server、llama-mtmd-cli）。
- **2026-10-19 重新量化**：改用 imatrix 加權的 MXFP4 量化器（[`mxfp4_wsearch.patch`](mxfp4_wsearch.patch)，llama.cpp b11719）與我們針對這個確切版本（`10fbf86fed`）算的 imatrix（llama-imatrix --process-output，982×512 token 校準文字，與品質量測文字不重疊）；張量配置與檔名不變，視覺投影檔沒有變。imatrix 檔已放在本 repo（`Ornith-1.5-35B-A3B-imatrix.gguf`）。
- 量化：專家、共享專家與所有稠密矩陣 MXFP4；token_embd、ssm_alpha / beta、整個 MTP 層 Q8_0；router 與 norm 為 F32；輸出頭 Q6_K。平均 4.46 bpw，載入 18.45 GiB。
- WHIRL 0.2.0 三種模式對官方 llama.cpp b11214（ROCm／Vulkan 分開列）量測，完整表格見上方。llama.cpp 在此 MoE 模型不用投機解碼（它的 MTP 較慢），WHIRL 用 MTP，decode 為各自出廠設定的對比：
Radeon 8060S（最大功率（80 W），ASUS ROG Flow Z13），WHIRL balance 對 llama.cpp ROCm／Vulkan：8k prefill 2,244 對 1,282／749.2 tok/s；128k token 文件之後的 decode 58.4 對 38.0／37.3。R9700：32k prefill 8,469 對 3,872／3,395（+119%／+149%），128k 4,275 對 2,307／1,922；2k token 文件之後的 decode 238.4 對 115.4／115.9；寫程式對話（全新撰寫）241.5 對 118.9／115.1。
- 品質（對 BF16，llama.cpp b11719 CPU，程式碼＋中文 80×512 token，BF16 困惑度 6.2537）：本版 KLD **0.1316**、與 BF16 同 top-1 **85.81%**、困惑度 6.5151（比 BF16 高 4.3%）；第一版（2026-10-03）KLD 0.1618、84.61%、6.6594（高 6.6%）→ KLD 低 19%。單一語料、非下游任務評估。
- 重建檔在 WHIRL 0.2.0 的檢查（Radeon 8060S 與 R9700）：selftest／seqtest 三種模式全部通過；MTP + n-gram 與 plain greedy 輸出逐字相同；KL 關卡兩張卡各 20／20 通過；golden 參考值已在兩張卡重錄。看圖：中英招牌新舊兩檔在 WHIRL 與 llama.cpp b11719 都正確讀出（只有長破折號寫成連字號）；第一版卡片的合成圖（WHIRL 42）新舊兩檔都正確；另一張大字合成圖新舊兩檔、兩個引擎都讀錯，非兩版差異。速度（balance、2k／32k、各 3 輪交錯、plain 解碼）與第一版無可量測差異：8060S prefill 2k 2,322 對 2,334、32k 1,850 對 1,850 tok/s，decode 80.3 對 80.5、64.2 對 64.3；R9700 prefill 2k 10,854 對 10,269（範圍重疊）、32k 9,133 對 9,137，decode 171.1 對 170.1、143.4 對 143.2。
- 授權：MIT（上游 metadata 宣告；上游 repo 沒有 LICENSE 檔，本 repo 附標準 MIT 條文）；詳見 LICENSE / NOTICE。
