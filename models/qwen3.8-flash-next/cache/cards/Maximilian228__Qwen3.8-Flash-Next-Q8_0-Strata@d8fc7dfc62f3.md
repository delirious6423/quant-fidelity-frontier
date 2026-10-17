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
- q8_0
- strata
- moe
- qwen3.8
- flash-next
- mtp
---

# Qwen3.8-Flash-Next — Q8_0 for Strata

Qwen's original [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) with **every routed expert in
Q8_0**, rounded from the BF16 checkpoint, packed for [Strata NVFP4](https://github.com/sergqwer/strata-nvfp4). It is
the fork's accuracy reference: the other models' accuracy below is measured against it. It needs a PC with **128 GB of
RAM**, and it decodes at about a quarter of the NVFP4 GPTQ model's speed.

**Quick start (Windows):** from the [Strata NVFP4](https://github.com/sergqwer/strata-nvfp4) release 0.1.42-nvfp4.2
or newer, run `START-HERE.bat --family qwen-q8`: it checks the PC, downloads this repository (only its experts if
`qwen-nvfp4-gptq` is installed: the other files are the same) and starts the model.

## How it compares

| Model | Same top token as Q8_0 | Decode, tokens/s | 32K prompt | RAM in use | PC for this speed | Get it |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **This model: Q8_0: every expert Q8_0 (ours, `qwen-q8`)** | **100% (the reference)** | **47** | **10.3 s** | **106 GiB** | **128 GB** | [Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata) |
| NVFP4 + Q8_0 mix (ours, `qwen-nvfp4-mix`) | 95.0% | 91 | 4.2 s | 105 GiB | 128 GB | [Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata) |
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
  → this quantization. No other model's weights are in it.
- **8.5 bits a weight in the experts** (llama.cpp's Q8_0: 32 weights share an FP16 scale). The experts' output is
  0.3% of NVFP4 round-to-nearest's error against BF16 (energy-weighted relative MSE 0.00006; the table in
  [the GPTQ card](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata#error-against-bf16)).
- **Everything but the experts is the same as in the NVFP4 GPTQ repository**, byte for byte: the dense GGUF (Q8_0
  projections; router, small tensors and norms BF16 / F32), Qwen's own FP8 PLE table, the BF16 embedding and the MTP
  draft head.
- **The original model, censored, with a switch.** It refuses what Qwen refuses. The fork can remove the refusal
  direction at run time (layers 8-33, the weights untouched): `--uncensored on` at install, the web app's "Disable
  censorship" switch, or `"uncensored": true` in a request. Checked on this pack: 5 of 5 held-out harmful requests
  refused with it off, 0 of 5 with it on.
- **Only for Strata NVFP4.** The files are Strata's runtime formats. They do not load in transformers, vLLM or
  llama.cpp.

## What it needs

- **128 GB of RAM.** The 119.5 GiB of experts do not all fit next to Windows, so the engine runs in the low-RAM mode
  (by itself from 0.1.42-nvfp4.2; `--low-ram` before): ~100 GiB of experts locked in RAM, the most-used ones in the VRAM cache, and Windows keeps 10 GiB
  (`STRATA_RESIDENT_HEADROOM_GIB=10`). Measured: 106 GiB in use. With less RAM free, more experts come from the SSD
  and it gets slower.
- **The route tail skip off.** Strata NVFP4 0.1.42-nvfp4.2 and newer keep it off for this pack by default. On
  0.1.42-nvfp4.1 set `STRATA_ROUTE_TAIL_SKIP=0`: that release turned it off only for packs with NVFP4 experts, and
  this pack has none, so the engine turned upstream's skip on and gave up the accuracy this model is for (on the GPTQ pack the skip moved the answers by KL 0.035, as much as its
  whole quantization error).
- **Disk:** 189.2 GB (176.2 GiB). An NVMe drive: every start reads the experts.
- Measured on a 32 GB card; a smaller card caches fewer experts and decodes slower.

## Files

| file | size | Strata flag | what it is |
| --- | ---: | --- | --- |
| `pack/experts.bin` | 119.53 GiB | `--pack pack` | the routed experts, 48 layers x 512, Q8_0 (this repo's point) |
| `pack/native_experts.txt` | 1.5 KB | | the experts' layout |
| `pack/dense.bin`, `pack/index.txt` | 1.43 GiB | | weights the engine keeps in its own form: hyper-connections, routers, PLE key/value, indexer, DeltaNet gates |
| `pack/tokenizer/` | 10.1 MB | | the server's tokenizer and chat template |
| `qwen-q8-dense.gguf` | 5.58 GiB | `--native`, `--native-dense-gguf` | the model's metadata and dense projections (Q8_0 / BF16) and head, without the routed experts |
| `ple-fp8.gguf` | 47.68 GiB | `--ple-gguf` | the per-layer n-gram (PLE) table in FP8 E4M3, Qwen's own (Qwen/Qwen3.8-Flash-Next-FP8), read from the SSD |
| `token-embd-bf16.gguf` | 1.18 GiB | `--embd-gguf` | the token embedding in BF16 as shipped |
| `mtp/` | 0.77 GiB | `--mtp mtp` | Qwen's MTP draft head, for speculative decoding |
| `LICENSE`, `SHA256SUMS`, `SIZES.txt` | | | the license, the files' SHA-256 and sizes |

## Checks

Strata NVFP4 0.1.42-nvfp4.1 on an RTX 5090 (32 GB), Ryzen 9 9950X3D, 128 GB of DDR5-5600, the arguments under
[Run it](#run-it).

- **Speed:** 46.1 / 46.5 / 47.0 tokens/s over 4 greedy chats in 3 runs (the NVFP4 GPTQ pack: 177), a 32,007-token
  prompt in 10.2-10.3 s (3.9 s). 65% of the expert lookups hit the VRAM cache (92% with NVFP4): a Q8_0 expert is
  twice the bytes, so the VRAM cache holds 3,970 of them (8,535 NVFP4), and each miss makes the CPU read twice as much
  RAM.
- **No reasoning loop on a long agentic request.** A captured Claude Code request (72K tokens of conversation, 24
  tools, thinking on, greedy decoding), replayed with a fresh engine each time: **5 of 5 OK**, thinking closed after
  1,097, 2,069, 722, 749 and 1,371 tokens, each run then answering with a tool call. The 72K prompt took 24-53 s to
  read.

## Run it

Build or download [Strata NVFP4](https://github.com/sergqwer/strata-nvfp4) (Windows, NVIDIA driver 580+; its README has
the requirements: page file, large pages). By hand (the two `set` lines matter only before 0.1.42-nvfp4.2):

```bat
hf download Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata --local-dir models\qwen-q8
set STRATA_ROUTE_TAIL_SKIP=0
set STRATA_RESIDENT_HEADROOM_GIB=10
strata.exe --pack models\qwen-q8\pack ^
  --native models\qwen-q8\qwen-q8-dense.gguf --native-dense-gguf models\qwen-q8\qwen-q8-dense.gguf ^
  --ple-gguf models\qwen-q8\ple-fp8.gguf --embd-gguf models\qwen-q8\token-embd-bf16.gguf ^
  --mtp models\qwen-q8\mtp --spec 6 --spec-min-p 0.7 --prefill auto ^
  --expert-profile data\expert-profile.bin --expert-cache auto --low-ram ^
  --max-context 262144 --kv int8 --tokens-file prompt.txt --max-new 512 --stop-eos
```

As a server (OpenAI and Anthropic APIs), put the same arguments in the release's `config\strata-nvfp4.json`
(`"args"`), add `"env": {"STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_RESIDENT_HEADROOM_GIB": "10"}`, and point
`"tokenizer"` at `models/qwen-q8/pack/tokenizer`. With less VRAM, lower `--max-context` as the fork's README says.

## Reproduce

With the fork's tools, the BF16 checkpoint (`hf download Qwen/Qwen3.8-Flash-Next`, 360 GB) and the GPTQ repository's
files (its dense GGUF, `pack/dense.bin`, PLE table, embedding and MTP head are this model's too):

```bat
echo {"default": {"gu": "q8_0", "d": "q8_0"}} > plan_q8.json
.venv\Scripts\python tools\requant.py pack --bf16 models\qwen-bf16 --plan plan_q8.json --method rtn ^
  --base models\qwen-gptq\pack --out pack
```

## License

The [Qwen Community License 1.0](LICENSE), Qwen3.8-Flash-Next's own LICENSE file. Among its terms: a separate license
from Qwen for commercial Model-as-a-Service or AI Work Assistant businesses. Strata NVFP4 itself is MIT.
