---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: strata
language:
- en
- zh
tags:
- nvfp4
- q8_0
- gptq
- strata
- moe
- qwen3.8
- flash-next
- mtp
---

# Qwen3.8-Flash-Next — NVFP4 + Q8_0 mix for Strata

Qwen's original [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) with **our NVFP4 GPTQ experts,
and Q8_0 in the layers that move the answers most**, packed for [Strata NVFP4](https://github.com/sergqwer/strata-nvfp4).
It is the most accurate model that fits a **128 GB PC** with every expert in RAM. Its most likely next token is the
Q8_0 model's 95.0% of the time, against 93.6% for the all-NVFP4 GPTQ, and it moves the answers 43% less (KL). It
decodes at about half the GPTQ model's speed.

**Quick start (Windows):** from the [Strata NVFP4](https://github.com/sergqwer/strata-nvfp4) release 0.1.42-nvfp4.2
or newer, run `START-HERE.bat --family qwen-nvfp4-mix`: it checks the PC, downloads this repository (only its experts if
`qwen-nvfp4-gptq` is installed: the other files are the same) and starts the model.

## How it compares

| Model | Same top token as Q8_0 | Decode, tokens/s | 32K prompt | RAM in use | PC for this speed | Get it |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q8_0: every expert Q8_0 (ours, `qwen-q8`) | 100% (the reference) | 47 | 10.3 s | 106 GiB | 128 GB | [Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata) |
| **This model: NVFP4 + Q8_0 mix (ours, `qwen-nvfp4-mix`)** | **95.0%** | **91** | **4.2 s** | **105 GiB** | **128 GB** | [Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata) |
| Unsloth UD-Q4_K_XL (`unsloth`, `--model UD-Q4_K_XL`) | 94.1% | 169 | 4.9 s | 77 GiB | 96 GB | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| NVFP4 GPTQ (ours, `qwen-nvfp4-gptq`, setup's default) | 93.6% | 177 | 3.9 s | 69 GiB | 96 GB | [Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata) |
| NVIDIA NVFP4, converted by us (`qwen-nvidia-nvfp4`) | 92.9% | 167 | 3.9 s | 69 GiB | 96 GB | [Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata) |
| Unsloth UD-Q3_K_XL (`unsloth`, `--model UD-Q3_K_XL`) | 90.7% | 208 | 3.8 s | 58 GiB | 96 GB | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| ISTA-DASLab IQ3_S (`qwen`, `--model IQ3_S`) | 90.0% | 228 | 3.9 s | 52 GiB | 64 GB | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |
| ISTA-DASLab IQ3_XXS (`qwen`, `--model IQ3_XXS`, upstream's pick) | 87.3% | 239 | 3.9 s | 45 GiB | 64 GB | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |
| ISTA-DASLab IQ2_XS (`qwen`, `--model IQ2_XS`) | 85.0% | 265 | 4.2 s | 38 GiB | 48 GB | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |
| ISTA-DASLab Q2_0 (`qwen`, `--model Q2_0`) | 84.1% | 256 | 3.9 s | 37 GiB | 48 GB | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |

- **Accuracy:** how often the model's most likely next token is the Q8_0 model's. Measured on 5,153 tokens of code,
  docs, English and Ukrainian chats and agent transcripts, every model fed the same text. Every model ran with its
  experts on the CPU and the route tail skip off, so only the quantization differs. The Q8_0 model also has Q8_0
  attention, Qwen's BF16 router and Qwen's own FP8 PLE, so it is the original's precision everywhere but the 8-bit
  weights.
- **Speed:** RTX 5090 (32 GB), Ryzen 9 9950X3D, 128 GB of DDR5-5600, Windows 11, Strata NVFP4 0.1.42-nvfp4.1's engine with
  the arguments 0.1.42-nvfp4.2's setup writes for each model. Median of 3 runs: 4 greedy chats (up to 1,500 tokens each) and one
  32,007-token prompt.
- **RAM:** "RAM in use" is the engine and the server at the speed shown; "PC for this speed" is the installed RAM that
  holds it with Windows beside it. Smaller PCs still run the models, slower: the NVFP4 GPTQ and NVIDIA's on a 64 GB PC
  in the low-RAM mode decode ~115 tokens/s and read a 32K prompt in ~9 s (docs/NVFP4.md, "64 GB of RAM"); Unsloth's
  files go into a RAM-budget mode there and read experts from the SSD (UD-Q4_K_XL: 7-8.5 tokens/s at 64 GB,
  upstream's measurement); setup's minimum is 48 GB for them and the ISTA 2-bit files, 60-64 GB for the ISTA 3-bit ones.
- **Unsloth's files keep every expert in RAM where they fit with 16 GiB more** (setup from 0.1.42-nvfp4.2): from ~88 GB
  of RAM for UD-Q4_K_XL and ~68 GB for UD-Q3_K_XL. Below that, setup's RAM-budget mode reads the rest from the GGUF on
  the SSD. Setup before 0.1.42-nvfp4.2 used the budget
  mode on every PC; on 128 GB UD-Q4_K_XL then decoded 118 tokens/s and read the 32K prompt in 30 s (run setup again to
  switch an existing install).
- UD-Q3_K_XL's experts are i-quants despite the name: IQ3_XXS gate/up with IQ4_NL or Q8_0 down.
- The GGUF models run upstream's route tail skip by default: ~10% faster decode, and answers a little further from
  Q8_0 than the table shows.
- The two 128 GB models need a 128 GB PC (setup asks below that). Q8_0 runs there in the low-RAM mode by itself (100 GiB
  of experts locked in RAM); the mix keeps every expert in RAM.

## About this model

- **Chain:** [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (BF16, revision `de4b8e4d`)
  → [our NVFP4 GPTQ](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata) and
  [our Q8_0](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata) → this mix. Each expert's rows are
  byte copies of one of the two; no other model's weights are in it.
- **103.1 GiB of experts** (GPTQ 63.3, Q8_0 119.5). Per layer:

  | layers | gate / up | down |
  | --- | --- | --- |
  | 3, 5-13, 17-30, 32-35 (28) | Q8_0 | Q8_0 |
  | 14, 16, 31, 38, 41-44 (8) | Q8_0 | NVFP4 GPTQ |
  | 1, 2 | NVFP4 GPTQ | Q8_0 |
  | 0, 4, 15, 36, 37, 39, 40, 45-47 (10) | NVFP4 GPTQ | NVFP4 GPTQ |

  `pack/mix.json` has the plan.
- **Everything but the experts is the same as in the NVFP4 GPTQ repository**, byte for byte: the dense GGUF (Q8_0
  projections; router, small tensors and norms BF16 / F32), Qwen's own FP8 PLE table, the BF16 embedding and the MTP
  draft head.
- **The original model, censored, with a switch.** It refuses what Qwen refuses. The fork can remove the refusal
  direction at run time (layers 8-33, the weights untouched): `--uncensored on` at install, the web app's "Disable
  censorship" switch, or `"uncensored": true` in a request. Checked on this pack: 5 of 5 held-out harmful requests
  refused with it off, 0 of 5 with it on.
- **Only for Strata NVFP4.** The files are Strata's runtime formats. They do not load in transformers, vLLM or
  llama.cpp.

## How the layers were chosen

Measured, not guessed: a guess by each layer's own error (the usual metric) put the late layers first, and a pack made
that way removed only ~12% of GPTQ's KL for 18 GiB more.

- **Block sensitivity:** four packs with Q8_0 in one block of 12 layers (GPTQ elsewhere), each compared end to end
  against the Q8_0 model. Per unit of relative error, layers 0-11 moved the answers most (1.0), 12-23 0.45, 24-35 0.50,
  36-47 0.38 (the fitted weights); an error in gate/up moved them ~3.3x as much as the same error in down.
- **A model of the KL** fitted to 9 measured packs (rms log error 0.03) chose, per layer, NVFP4, Q8_0 down, Q8_0
  gate/up or both, under the size budget.
- **The budget:** the whole model in at most 110 GiB of RAM on a 128 GB PC: the pack plus the engine, the image
  encoder and the server measured 106 GiB. 3.5 GiB more of Q8_0 would have cut the KL only by 13%, for the last free
  RAM.

End to end against the Q8_0 model (the same runs as the table; the thinking text was written by the GPTQ model, which
favours it):

| | same top token, plain / thinking text | KL, plain / thinking text | perplexity vs Q8_0 |
| --- | ---: | ---: | ---: |
| **this mix** | **95.0% / 97.4%** | **0.0215 / 0.0058** | **100.0%** |
| NVFP4 GPTQ | 93.6% / 96.5% | 0.0379 / 0.0124 | 100.1% |

## What it needs

- **128 GB of RAM.** Every expert stays in RAM (the resident mode, no `--low-ram`): 106 GiB in use with the image
  encoder and the server, ~10 GiB left to Windows. On a PC with less RAM setup asks first (default no), and from
  0.1.42-nvfp4.2 the engine starts it in the low-RAM mode by itself, reading some experts from the SSD; that was not
  measured.
- Nothing else to set: Strata NVFP4 keeps the route tail skip off for this pack by default (0.1.42-nvfp4.1 and newer).
- **Disk:** 171.6 GB (159.8 GiB). An NVMe drive: every start reads the experts.
- Measured on a 32 GB card; a smaller card caches fewer experts and decodes slower.

## Files

| file | size | Strata flag | what it is |
| --- | ---: | --- | --- |
| `pack/experts.bin` | 103.13 GiB | `--pack pack` | the routed experts, 48 layers x 512, NVFP4 GPTQ and Q8_0 (this repo's point) |
| `pack/native_experts.txt`, `pack/mix.json` | 4.2 KB | | the experts' layout and the plan |
| `pack/dense.bin`, `pack/index.txt` | 1.43 GiB | | weights the engine keeps in its own form: hyper-connections, routers, PLE key/value, indexer, DeltaNet gates |
| `pack/tokenizer/` | 10.1 MB | | the server's tokenizer and chat template |
| `qwen-nvfp4-mix-dense.gguf` | 5.58 GiB | `--native`, `--native-dense-gguf` | the model's metadata and dense projections (Q8_0 / BF16) and head, without the routed experts |
| `ple-fp8.gguf` | 47.68 GiB | `--ple-gguf` | the per-layer n-gram (PLE) table in FP8 E4M3, Qwen's own (Qwen/Qwen3.8-Flash-Next-FP8), read from the SSD |
| `token-embd-bf16.gguf` | 1.18 GiB | `--embd-gguf` | the token embedding in BF16 as shipped |
| `mtp/` | 0.77 GiB | `--mtp mtp` | Qwen's MTP draft head, for speculative decoding |
| `LICENSE`, `SHA256SUMS`, `SIZES.txt` | | | the license, the files' SHA-256 and sizes |

## Checks

Strata NVFP4 0.1.42-nvfp4.1 on an RTX 5090 (32 GB), Ryzen 9 9950X3D, 128 GB of DDR5-5600, the arguments under
[Run it](#run-it).

- **Speed:** 94.8 / 90.1 / 91.5 tokens/s over 4 greedy chats in 3 runs (the GPTQ pack: 177), a 32,007-token prompt in
  4.2 s (3.9 s). The Q8_0 experts are twice the bytes: the VRAM cache holds 5,233 experts instead of 8,535 (84% of
  lookups hit, against 92%), and each miss makes the CPU read more RAM. Windows does not give large pages to an
  arena over ~100 GB, which costs ~5%.
- **No reasoning loop on a long agentic request.** A captured Claude Code request (72K tokens of conversation, 24
  tools, thinking on, greedy decoding), replayed with a fresh engine each time: **5 of 5 OK**, thinking closed after
  2,389, 1,582, 1,744, 1,810 and 2,858 tokens, each run then answering with a tool call (the GPTQ pack: 777-1,774).

## Run it

Build or download [Strata NVFP4](https://github.com/sergqwer/strata-nvfp4) (Windows, NVIDIA driver 580+; its README has
the requirements: page file, large pages). Then:

```bat
hf download Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata --local-dir models\qwen-mix
strata.exe --pack models\qwen-mix\pack ^
  --native models\qwen-mix\qwen-nvfp4-mix-dense.gguf --native-dense-gguf models\qwen-mix\qwen-nvfp4-mix-dense.gguf ^
  --ple-gguf models\qwen-mix\ple-fp8.gguf --embd-gguf models\qwen-mix\token-embd-bf16.gguf ^
  --mtp models\qwen-mix\mtp --spec 6 --spec-min-p 0.7 --prefill auto ^
  --expert-profile data\expert-profile.bin --expert-cache auto ^
  --max-context 262144 --kv int8 --tokens-file prompt.txt --max-new 512 --stop-eos
```

As a server (OpenAI and Anthropic APIs), put the same arguments in the release's `config\strata-nvfp4.json`
(`"args"`) and point `"tokenizer"` at `models/qwen-mix/pack/tokenizer`. With less VRAM, lower `--max-context` as the
fork's README says.

## Reproduce

`pack/experts.bin` is byte ranges of the two source packs, as `pack/mix.json`'s plan says: an NVFP4 projection's rows
and scales from the GPTQ repository's `pack/experts.bin`, a Q8_0 projection's rows from the Q8_0 repository's (each
blob is gate | up | down, plus the NVFP4 scales when a projection is NVFP4; `pack/native_experts.txt` gives each
layer's types, offset and blob size). It is what the fork's `tools/requant.py pack` writes for that plan, except that the
GPTQ rows are the published ones rather than a fresh fit.

## License

The [Qwen Community License 1.0](LICENSE), Qwen3.8-Flash-Next's own LICENSE file. Among its terms: a separate license
from Qwen for commercial Model-as-a-Service or AI Work Assistant businesses. Strata NVFP4 itself is MIT.
