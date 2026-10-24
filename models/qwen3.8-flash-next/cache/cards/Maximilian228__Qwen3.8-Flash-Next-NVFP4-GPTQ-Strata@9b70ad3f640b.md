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
- gptq
- strata
- moe
- qwen3.8
- flash-next
- mtp
---

# Qwen3.8-Flash-Next — NVFP4 (GPTQ) for Strata

Qwen's original [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) with **every routed expert in
NVFP4, quantized from the BF16 checkpoint by GPTQ**, packed for [Strata NVFP4](https://github.com/sergqwer/strata-nvfp4):
the 125B hybrid MoE on one RTX 20-50 card (12 GB of VRAM or more) and 64 GB of RAM or more.

## How it compares

| Model | Same top token as Q8_0 | Decode, tokens/s | 32K prompt | RAM in use | PC for this speed | Get it |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q8_0: every expert Q8_0 (ours, `qwen-q8`) | 100% (the reference) | 47 | 10.3 s | 106 GiB | 128 GB | [Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata) |
| NVFP4 + Q8_0 mix (ours, `qwen-nvfp4-mix`) | 95.0% | 91 | 4.2 s | 105 GiB | 128 GB | [Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata) |
| Unsloth UD-Q4_K_XL (`unsloth`, `--model UD-Q4_K_XL`) | 94.1% | 169 | 4.9 s | 77 GiB | 96 GB | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| **This model: NVFP4 GPTQ (ours, `qwen-nvfp4-gptq`, setup's default)** | **93.6%** | **177** | **3.9 s** | **69 GiB** | **96 GB** | [Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata) |
| NVIDIA NVFP4, converted by us (`qwen-nvidia-nvfp4`) | 92.9% | 167 | 3.9 s | 69 GiB | 96 GB | [Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata) |
| NVIDIA FP4-Local, retrained by NVIDIA for FP4 (QAD), converted on the PC | 92.2% | 191¹ | 3.7 s¹ | 69 GiB | 96 GB | [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) |
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
- ¹ **NVIDIA FP4-Local** is NVIDIA's retrained 4-bit model (quantization-aware distillation): its experts and n-gram
  table were trained again for FP4, so it is a slightly different model rather than a rounding of the original. Its
  KL to the Q8_0 model is 0.051 (the NVFP4 GPTQ and NVIDIA's NVFP4: 0.038), while its perplexity on these texts is a
  little below Q8_0's. Measured on 2026-10-24 on the same texts against the same reference, with the engine of this
  fork's 0.2.0-beta-nvfp4.3. The NVFP4 GPTQ model gives the same accuracy on that engine (93.65%), and in the same runs
  it decoded 180-191 tokens/s, so the two run at the same speed within the noise. NVIDIA ships ModelOpt safetensors
  (95 GiB) that are converted on the PC (upstream Strata's `tools/nvfp4_local_pack.py prepare`, about 10 minutes,
  99 GiB of files); this fork's setup does not install it. Its license, the NVIDIA Open Model Agreement, rules out the
  censorship switch.

**Quick start (Windows):** from the [Strata NVFP4](https://github.com/sergqwer/strata-nvfp4) release, run
`START-HERE.bat --family qwen-nvfp4-gptq`: it checks the PC, downloads this repository and starts the model.

- **Chain:** [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (BF16, revision `de4b8e4d`)
  → this quantization. No other model's weights are in it.
- **The same format, size and decode cost as NVIDIA's NVFP4** ([`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4),
  converted for Strata as [`Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata`](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata)):
  63.3 GiB of experts, 4.5 bits a weight; other codes and scales. NVIDIA rounds each weight to the nearest NVFP4
  code (with MSE-calibrated scales); GPTQ chooses codes that cancel each other's error on the model's own activations:
  45% of the error of NVIDIA's experts on held-out tokens (table below). End to end the gap is smaller: the most
  likely next token is the Q8_0 model's 93.6% of the time against 92.9%, and the KL is the same (0.038, table
  above).
- **The original model, censored, with a switch.** It refuses what Qwen refuses. The fork can remove the refusal
  direction at run time (layers 8-33, the weights untouched): `--uncensored on` at install, the web app's "Disable
  censorship" switch, or `"uncensored": true` in a request.
- **Only for Strata NVFP4.** The files are Strata's runtime formats (an expert pack, GGUFs the fork reads, the MTP
  draft head). They do not load in transformers, vLLM or llama.cpp.

## Files

| file | size | Strata flag | what it is |
| --- | ---: | --- | --- |
| `pack/experts.bin` | 63.28 GiB | `--pack pack` | the routed experts, 48 layers x 512, NVFP4 by GPTQ (this repo's point) |
| `pack/native_experts.txt` | 1.6 KB | | the experts' layout |
| `pack/dense.bin`, `pack/index.txt` | 1.43 GiB | | weights the engine keeps in its own form: hyper-connections, routers, PLE key/value, indexer, DeltaNet gates |
| `pack/tokenizer/` | 10.1 MB | | the server's tokenizer and chat template |
| `qwen-nvfp4-gptq-dense.gguf` | 5.58 GiB | `--native`, `--native-dense-gguf` | the model's metadata and dense projections (Q8_0 / BF16) and head, without the routed experts |
| `ple-fp8.gguf` | 47.68 GiB | `--ple-gguf` | the per-layer n-gram (PLE) table in FP8 E4M3: Qwen's own FP8 table (Qwen/Qwen3.8-Flash-Next-FP8), read from the SSD |
| `token-embd-bf16.gguf` | 1.18 GiB | `--embd-gguf` | the token embedding in BF16 as shipped |
| `mtp/` | 0.77 GiB | `--mtp mtp` | Qwen's MTP draft head (Q2_0 experts), for speculative decoding |
| `LICENSE`, `SHA256SUMS`, `SIZES.txt` | | | the license, the files' SHA-256 and sizes |

Everything but `pack/experts.bin` is Qwen's model as the fork converts it, the same bytes as in the NVIDIA conversion's
repository wherever a file is shared (`ple-fp8.gguf`, `token-embd-bf16.gguf`, `mtp/`; setup links them instead of a
second download):

- **Dense weights:** `qwen-nvfp4-gptq-dense.gguf` is made from the BF16 checkpoint by the fork's converter policy (the
  projections the engine serves natively as Q8_0, the small ones its kernels read as BF16); all 1,079 of its tensors
  are byte-identical to the NVIDIA conversion's dense GGUF (NVIDIA kept Qwen's BF16 tensors unchanged: 1,432 of 1,432
  shared tensors byte-identical).
- **PLE table:** Qwen ships the n-gram table in FP8 in Qwen/Qwen3.8-Flash-Next-FP8; these are its bytes (taken from
  NVIDIA's checkpoint, which carries them unchanged; checked against Qwen-FP8 by range reads on three shards). On
  524K sampled rows it is 2.66% RMS off the BF16 table.
- **Embedding:** BF16 `embed_tokens` as Qwen ships it.
- **MTP head:** packed by `tools/mtp_extract.py` → `mtp_pack.py --experts q2_0` → `mtp_rt.py` from this checkpoint's
  31 `mtp.*` tensors (rebuilt here and byte-identical).

## How it was quantized

NVFP4 stores a weight as a 4-bit E2M1 code times its 16-value block's FP8 (E4M3) scale times one FP32 scale per
expert matrix. Each expert was quantized from the BF16 checkpoint with `tools/requant.py`'s GPTQ, as the fork's
OrcaRouter quant:

- **Calibration:** the engine's own MoE inputs at every layer (`STRATA_DUMP_MOE_INPUT`, `STRATA_DUMP_MOE_LAYER=all`)
  for 57.5K tokens: Ukrainian and English Wikipedia, llama.cpp and Rust sources, and 16 chat answers
  (`data/requant_calib` in the fork), with the routing of each token. The engine ran NVIDIA's NVFP4 experts (the
  converted repository above) for the dump: the closest quantized model of the same weights there was. The dump only
  shapes the Hessians; no NVIDIA weight is in this repository.
- **Hessians:** for each expert, `H = sum w^2 x x^T` over the tokens routed to it (`w` its routing weight), mixed with
  the layer's average input as 256 pseudo-tokens so rarely used experts get a sensible one. `down`'s Hessian is built
  from `silu(gate x) * up x` of the already quantized gate/up, so down corrects the error gate/up left.
- **GPTQ:** columns left to right in blocks of 128, 1% damping; each column's rounding error is spread over the
  columns not yet quantized through the inverse Hessian.
- **Block scales:** each 16-column group's FP8 scale is chosen on the weights as GPTQ has updated them, among the
  codes for amax -> 6 and amax -> 4 and two neighbours of each, by the least squared error weighted by diag(H). The
  FP32 scale per expert matrix leaves 25% headroom so the updates do not saturate E4M3.
- **Every routed expert is NVFP4** (no 8-bit layers): the pack is as large as NVIDIA's and a decode step costs the same.

### Error against BF16

Measured with the fork's `tools/requant.py analyze` (plus NVIDIA's own experts read from its converted pack):
per layer, on held-out tokens (every 4th calibration token, not used for the Hessians), the routed-weighted squared
error of the experts' whole output `down(silu(gate x) * up x)` against BF16, relative to the output's energy; summed
over the 48 layers weighted by each layer's energy.

48 layers, 57551 tokens per layer (14387 held out)

| method | summed error (rel. MSE x energy) | energy-weighted rel. MSE | of RTN | of NVIDIA's |
| --- | ---: | ---: | ---: | ---: |
| RTN (ModelOpt's recipe: amax scales) | 14161.88 | 0.01684 | 100.0% | 134.4% |
| NVIDIA's checkpoint (ModelOpt, MSE-calibrated scales) | 10534.21 | 0.01252 | 74.4% | 100.0% |
| **GPTQ + block-scale search (this pack)** | 4716.95 | 0.00561 | 33.3% | 44.8% |
| GPTQ gate/up + Q8_0 down (reference, not used) | 1930.52 | 0.00230 | 13.6% | 18.3% |
| Q8_0 everything (reference) | 48.90 | 0.00006 | 0.3% | 0.5% |

- GPTQ below NVIDIA's experts in 48 of 48 layers; GPTQ / NVIDIA per layer: min 0.28, median 0.53, max 0.63.
- NVIDIA's MSE-calibrated scales are below plain RTN in 48 of 48 layers (per layer 0.73-0.76 of RTN): its checkpoint
  is round-to-nearest with better block scales, not GPTQ.

<details><summary>Per layer</summary>

| layer | RTN | NVIDIA | GPTQ | GPTQ / NVIDIA |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00957 | 0.00727 | 0.00204 | 0.28 |
| 1 | 0.01226 | 0.00915 | 0.00378 | 0.41 |
| 2 | 0.01087 | 0.00813 | 0.00347 | 0.43 |
| 3 | 0.01352 | 0.01006 | 0.00448 | 0.45 |
| 4 | 0.01053 | 0.00790 | 0.00222 | 0.28 |
| 5 | 0.01422 | 0.01066 | 0.00498 | 0.47 |
| 6 | 0.01425 | 0.01058 | 0.00512 | 0.48 |
| 7 | 0.01525 | 0.01132 | 0.00571 | 0.50 |
| 8 | 0.01546 | 0.01142 | 0.00604 | 0.53 |
| 9 | 0.01562 | 0.01158 | 0.00596 | 0.51 |
| 10 | 0.01671 | 0.01231 | 0.00690 | 0.56 |
| 11 | 0.01727 | 0.01269 | 0.00729 | 0.57 |
| 12 | 0.01760 | 0.01298 | 0.00759 | 0.58 |
| 13 | 0.01723 | 0.01266 | 0.00722 | 0.57 |
| 14 | 0.01561 | 0.01155 | 0.00614 | 0.53 |
| 15 | 0.01614 | 0.01191 | 0.00473 | 0.40 |
| 16 | 0.01739 | 0.01280 | 0.00597 | 0.47 |
| 17 | 0.01902 | 0.01397 | 0.00787 | 0.56 |
| 18 | 0.01877 | 0.01381 | 0.00809 | 0.59 |
| 19 | 0.02032 | 0.01490 | 0.00895 | 0.60 |
| 20 | 0.02006 | 0.01473 | 0.00876 | 0.59 |
| 21 | 0.01966 | 0.01445 | 0.00902 | 0.62 |
| 22 | 0.01999 | 0.01475 | 0.00923 | 0.63 |
| 23 | 0.02107 | 0.01547 | 0.00896 | 0.58 |
| 24 | 0.01829 | 0.01346 | 0.00751 | 0.56 |
| 25 | 0.01767 | 0.01304 | 0.00724 | 0.55 |
| 26 | 0.01924 | 0.01413 | 0.00858 | 0.61 |
| 27 | 0.01935 | 0.01421 | 0.00874 | 0.61 |
| 28 | 0.02052 | 0.01507 | 0.00954 | 0.63 |
| 29 | 0.01879 | 0.01379 | 0.00836 | 0.61 |
| 30 | 0.01614 | 0.01186 | 0.00617 | 0.52 |
| 31 | 0.01768 | 0.01302 | 0.00569 | 0.44 |
| 32 | 0.01879 | 0.01383 | 0.00709 | 0.51 |
| 33 | 0.01950 | 0.01433 | 0.00843 | 0.59 |
| 34 | 0.01803 | 0.01336 | 0.00728 | 0.54 |
| 35 | 0.01975 | 0.01453 | 0.00829 | 0.57 |
| 36 | 0.01809 | 0.01333 | 0.00657 | 0.49 |
| 37 | 0.01811 | 0.01350 | 0.00654 | 0.48 |
| 38 | 0.01754 | 0.01305 | 0.00704 | 0.54 |
| 39 | 0.01715 | 0.01268 | 0.00553 | 0.44 |
| 40 | 0.01750 | 0.01286 | 0.00592 | 0.46 |
| 41 | 0.01884 | 0.01390 | 0.00739 | 0.53 |
| 42 | 0.01917 | 0.01411 | 0.00724 | 0.51 |
| 43 | 0.01919 | 0.01414 | 0.00774 | 0.55 |
| 44 | 0.01906 | 0.01417 | 0.00700 | 0.49 |
| 45 | 0.01977 | 0.01468 | 0.00692 | 0.47 |
| 46 | 0.02026 | 0.01505 | 0.00656 | 0.44 |
| 47 | 0.01348 | 0.01025 | 0.00284 | 0.28 |

</details>

## Checks

Strata NVFP4 0.1.41-nvfp4.3 (engine `strata.exe` SHA-256 `1217e500...`) on an RTX 5090 (32 GB) with 128 GB of RAM,
the arguments under [Run it](#run-it) (`--spec 6 --spec-min-p 0.7`, expert cache auto), the pack on a PCIe 5 NVMe drive.

- **It answers.** "Explain in detail how a hash table works ... at least 300 words" (no thinking): a correct
  446-token answer that ends by itself (EOS), 185.5 tok/s, 2.36 tokens per speculative round. A 2,099-token prompt (a
  CUDA source file cut off mid-line, no question): prefill 2,903 tok/s; with `--stop-eos` the model completes the cut
  line in the file's pattern (`q2[l` -> `q2[l0 / ...] >> 9);`) and ends the turn after 12 tokens.
- **A decode step costs the same as with NVIDIA's experts.** Three interleaved pairs on the hash-table question, each
  answer to its end, against the NVIDIA conversion (the same GGUF, PLE table, embedding and MTP head; only
  `pack/experts.bin` differs):

  | experts | ms per round | tokens per round | tok/s |
  | --- | ---: | ---: | ---: |
  | NVIDIA's ModelOpt NVFP4 | 13.17 ± 0.14 | 2.310 ± 0.008 | 175.5 ± 1.3 |
  | this GPTQ pack | 13.56 ± 0.17 | 2.431 ± 0.029 | 179.3 ± 2.7 |

  Per generated token 5.58 ms here, 5.70 ms with NVIDIA's experts. This pack's answers averaged 5% more tokens per round
  (more MTP drafts accepted), so its rounds were a little longer and its tok/s a little higher; three pairs cannot tell that from
  the answers' own spread (paired tok/s difference +3.8 ± 1.6). The PC was hashing and uploading this repository's
  files from the same drive at background I/O priority while these ran.
- **No reasoning loop on a long agentic request.** A captured Claude Code request (72K tokens of conversation, 24
  tools, thinking on, greedy decoding) that sent another abliterated Flash-Next build into an endless thinking loop in
  5 of 5 runs, replayed 5 times through the fork's server with a fresh engine each time: **5 of 5 OK**, thinking
  closed after 1,099, 1,628, 1,823, 2,292 and 1,655 tokens, each run then answering with a tool call (NVIDIA's
  conversion: 5 of 5 OK, 856-2,181 tokens).

## Run it

Build or download [Strata NVFP4](https://github.com/sergqwer/strata-nvfp4) (Windows, NVIDIA driver 580+; see its README
for the requirements: RAM, pagefile, large pages). Then:

```bat
hf download Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata --local-dir models\qwen-gptq
```

One-shot (`prompt.txt` holds token ids):

```bat
strata.exe --pack models\qwen-gptq\pack ^
  --native models\qwen-gptq\qwen-nvfp4-gptq-dense.gguf --native-dense-gguf models\qwen-gptq\qwen-nvfp4-gptq-dense.gguf ^
  --ple-gguf models\qwen-gptq\ple-fp8.gguf --embd-gguf models\qwen-gptq\token-embd-bf16.gguf ^
  --mtp models\qwen-gptq\mtp --spec 6 --spec-min-p 0.7 --prefill auto ^
  --expert-profile data\expert-profile.bin --expert-cache auto ^
  --max-context 262144 --kv int8 --tokens-file prompt.txt --max-new 512 --stop-eos
```

`--max-context 262144` fits a 32 GB card; with less VRAM lower it (the fork's README and setup use these):

| card's VRAM | `--max-context` |
| --- | ---: |
| 32 GB | 262144 |
| 24 GB | 131072 |
| 16 GB | 65536 |
| 12 GB | 32768 |

`data\expert-profile.bin` ships with the engine. As a server (OpenAI and Anthropic APIs), put the same arguments in
the release's `config\strata-nvfp4.json` (`"args"`) and point `"tokenizer"` at `models/qwen-gptq/pack/tokenizer`.
**Pictures** need the image encoder (`mmproj-Qwen3.8-Flash-Next-BF16.gguf` from
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), which setup downloads); point the
config's `"vision"` entry's `"model"` at `qwen-nvfp4-gptq-dense.gguf` (the encoder reads only its vocabulary).

## Reproduce

With the fork's tools and the BF16 checkpoint (`hf download Qwen/Qwen3.8-Flash-Next`, 360 GB), starting from the
NVIDIA conversion's files (`Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata`, or the fork's `tools/nvfp4_convert.py`
and `tools/iq_pack.py` on NVIDIA's checkpoint): its dense GGUF, PLE table, embedding and MTP head are this model's too.

```bat
:: the calibration dump: the engine with the NVIDIA conversion, on the four token files in data\requant_calib
set STRATA_DUMP_MOE_INPUT=calib\d
set STRATA_DUMP_MOE_LAYER=all
strata.exe <the arguments above, the NVIDIA conversion's files> --tokens-file data\requant_calib\calib_uk.txt --max-new 1   (and en, code, chat)
:: errors per layer and method, then GPTQ NVFP4 in every layer
.venv\Scripts\python tools\requant.py analyze --bf16 models\qwen-bf16 --calib calib\d --out calib\errors.jsonl --methods rtn,gptq
echo {"default": {"gu": "nvfp4", "d": "nvfp4"}} > calib\plan_nvfp4.json
.venv\Scripts\python tools\requant.py pack --bf16 models\qwen-bf16 --calib calib\d --plan calib\plan_nvfp4.json ^
  --method gptq --base models\qwen-nvidia\pack --out pack
```

This repository's `qwen-nvfp4-gptq-dense.gguf` and `pack/dense.bin` were made from the BF16 checkpoint itself (the fork's
converter policy with the routed experts left out); their tensors are byte-identical to the NVIDIA conversion's, whose
`pack/dense.bin` also carries NVIDIA's 288 small per-expert scale tensors (unused next to `experts.bin`). The GPTQ was
run in groups of layers (`requant.pack_layer`, the same code), ~27 s a layer on the RTX 5090, 22 minutes in all.

## License

The [Qwen Community License 1.0](LICENSE), Qwen3.8-Flash-Next's own LICENSE file. Among its terms: a separate license
from Qwen for commercial Model-as-a-Service or AI Work Assistant businesses. Strata NVFP4 itself is MIT.
