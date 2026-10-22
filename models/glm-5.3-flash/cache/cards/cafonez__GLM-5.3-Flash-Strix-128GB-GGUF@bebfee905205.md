---
license: mit
license_link: LICENSE
base_model: zai-org/GLM-5.3-Flash
pipeline_tag: text-generation
library_name: gguf
quantized_by: cafonez
language:
- en
- zh
tags:
- gguf
- glm
- glm5-next
- llama.cpp
- rocmfpx
- strix-halo
- 128gb
- iq2_s
- iq3_xxs
---

# GLM-5.3-Flash GGUF for 128 GB machines (Strix Halo)

A 3.00 bpw GGUF of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320.8B total parameters, MoE with 288 experts, top-8, 18B active), sized to run fully on the GPU of a 128 GB AMD Ryzen AI Max ("Strix Halo") machine.

- Format: llama.cpp `glm5-next`. Loads in current llama.cpp `master` and in [ROCmFPX](https://github.com/ROCmFPX/ROCmFPX).
- Size: 112.0 GiB in 3 shards. Pass the first shard to `-m`; llama.cpp finds the rest.
- Loaded size: 109.3 GiB of weights. The MTP (next-token prediction) layer adds 2.7 GiB and is only loaded when MTP drafting is enabled.
- Text only: the vision encoder of the original multimodal model is not included.

## Files

| file | size |
|---|---|
| `GLM-5.3-Flash-Strix128-IQ2S-IQ3XXS-00001-of-00003.gguf` | 44.3 GB |
| `GLM-5.3-Flash-Strix128-IQ2S-IQ3XXS-00002-of-00003.gguf` | 44.6 GB |
| `GLM-5.3-Flash-Strix128-IQ2S-IQ3XXS-00003-of-00003.gguf` | 31.4 GB |

## Quantization recipe

The routed experts hold 94% of the bytes, so they get the low-bit types. Which layers keep 3 bits was decided by measurement: with each group of seven layers switched to IQ2_S in turn, the early MoE layers cost 3-4x more KL divergence than the late ones.

| tensors | type | bpw | size |
|---|---|---|---|
| routed expert gate / up, layers 3-24 | IQ3_XXS | 3.06 | 37.9 GiB |
| routed expert gate / up, layers 25-44 | IQ2_S | 2.56 | 28.8 GiB |
| routed expert down, layers 3-44 | IQ3_XXS | 3.06 | 36.2 GiB |
| MTP layer (45) experts | IQ3_XXS | 3.06 | 2.6 GiB |
| attention (MLA), KDA, dense FFN, shared experts | Q5_K | 5.50 | 4.6 GiB |
| indexer, hyper-connections, small projections | Q8_0 | 8.50 | 1.2 GiB |
| output head | Q6_K | 6.56 | 0.5 GiB |
| norms, biases | F32 | | 0.2 GiB |

- All 288 experts are kept in every layer. Pruning the 15% least-used experts was tried and rejected: it raised KL divergence to 0.43.
- The routed experts were quantized directly from the official FP8 weights (E4M3 with 128x128 block scales), one expert at a time, with ggml's own quantizer and a per-expert importance vector. The importance statistics come from 32K calibration tokens of mixed English (WikiText-2), Chinese and Python text.
- The non-expert tensors are a byte copy of [GLM-5.3-Flash-Gorgon-GGUF](https://huggingface.co/cafonez/GLM-5.3-Flash-Gorgon-GGUF).

## Quality

Measured against [GLM-5.3-Flash-Gorgon-GGUF](https://huggingface.co/cafonez/GLM-5.3-Flash-Gorgon-GGUF) (3.84 bpw, 143.5 GiB), with `llama-perplexity --kl-divergence` on 16 x 512-token chunks of English documentation text:

| | this file | 3.84 bpw reference |
|---|---|---|
| mean KLD | 0.146 | 0 |
| perplexity | 3.80 | 3.75 |
| same top-1 token | 87.3% | 100% |

HumanEval and HumanEval+ (164 tasks, greedy, one sample, low reasoning effort, up to 8192 tokens, tests run in a sandbox): this file scored **156/164 (95.1%)** on HumanEval and **149/164 (90.9%)** on HumanEval+ ([EvalPlus](https://github.com/evalplus/evalplus) 0.3.1, HumanEvalPlus v0.1.10), served by the ROCmFPX branch below with n-gram + MTP drafting. The 3.84 bpw reference also scored 156/164 on HumanEval.

## Fitting in 128 GB

Measured peaks on the test machine (192 GB, so nothing was squeezed), ROCm 10.2, q8_0 KV cache:

| engine and settings | GPU memory (GTT) | total memory used |
|---|---|---|
| llama.cpp, 64K context, `-ub 512` | 111.0 GiB | 112.9 GiB |
| ROCmFPX, 64K context, `-ub 512` | 110.5 GiB | 115.6 GiB |
| ROCmFPX, 128K context, `-ub 512` | 111.3 GiB | 113.5 GiB |
| ROCmFPX, 64K context, MTP drafting | 114.3 GiB | 116.5 GiB |

A 128 GB machine has about 125 GiB usable, so every row fits with the GTT limit below; the KV cache of this hybrid-attention model is small, so 128K context costs less than 1 GiB more than 64K. Larger `-ub` values add compute buffers (about 2 GiB at `-ub 4096` with a drafter): keep `-ub 512` to `-ub 2048` on 128 GB.

## How to run

### 1. Let the GPU use the memory

Strix Halo GPUs use system memory through GTT. Keep the BIOS GPU carve-out (UMA frame buffer) at its minimum and raise the GTT limit with kernel parameters, then reboot. For a 128 GB machine, give the GPU 120 GiB (leaves about 8 GiB for the system):

```
ttm.pages_limit=31457280 ttm.page_pool_size=31457280
```

On Ubuntu, add them to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, run `sudo update-grub`, and reboot. Check with `cat /sys/class/drm/card*/device/mem_info_gtt_total`.

Run the machine headless (no desktop session) or with a minimal one: the model leaves only a few GiB for everything else.

### 2. Download

```
hf download cafonez/GLM-5.3-Flash-Strix-128GB-GGUF --local-dir GLM-5.3-Flash-Strix-128GB-GGUF
```

### 3a. llama.cpp

Plain decoding (llama.cpp does not run GLM-5.3 MTP drafting yet). Build with ROCm (HIP); tested with ROCm 10.2 (TheRock nightly 10.2.0a20261021) and llama.cpp `master` of 2026-10-01:

```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server

./build/bin/llama-server \
  -m GLM-5.3-Flash-Strix-128GB-GGUF/GLM-5.3-Flash-Strix128-IQ2S-IQ3XXS-00001-of-00003.gguf \
  -dev ROCm0 -ngl 999 -fa on -lm dio \
  -c 131072 -b 2048 -ub 512 -ctk q8_0 -ctv q8_0 -np 1 \
  --jinja --temp 0.7 --top-p 0.95 \
  --host 127.0.0.1 --port 8080
```

The Vulkan backend was not verified with this file: on the test machine its model load pushed the system into swap.

### 3b. ROCmFPX (ROCm, with MTP speculative decoding)

[ROCmFPX](https://github.com/ROCmFPX/ROCmFPX) runs the MTP layer that is included in this file, which is the fastest speculative decoding that needs no extra download. Build it like llama.cpp above, then:

```
ROCMFPX_DRAFT_VOCAB=16384 ./build/bin/llama-server \
  -m GLM-5.3-Flash-Strix-128GB-GGUF/GLM-5.3-Flash-Strix128-IQ2S-IQ3XXS-00001-of-00003.gguf \
  -dev ROCm0 -ngl 999 -fa on -lm dio \
  -c 131072 -b 2048 -ub 512 -ctk q8_0 -ctv q8_0 -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
  --jinja --temp 0.7 --top-p 0.95 \
  --host 127.0.0.1 --port 8080
```

`ROCMFPX_DRAFT_VOCAB=16384` lets the MTP drafter score only the 16384 most likely tokens (same output, faster drafts).

### 3c. Fastest: the GLM-5.3-Flash branch of ROCmFPX

The branch [`glm-5.3-flash-strix128`](https://github.com/charlie12345/ROCmFPX/tree/glm-5.3-flash-strix128) adds kernels made for this file on gfx1151: LDS-staged IQ2_S/IQ3_XXS expert tiles (up to about 4x the prefill of ROCmFPX `main` on this model: 481 vs 121 t/s on a 4096-token prompt) and a multi-token expert kernel for speculative verification. It is the build behind the "branch" rows in the speed table.

```
git clone -b glm-5.3-flash-strix128 https://github.com/charlie12345/ROCmFPX && cd ROCmFPX
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGGML_HIP_NO_VMM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server

ROCMFPX_DRAFT_VOCAB=16384 ./build/bin/llama-server \
  -m GLM-5.3-Flash-Strix-128GB-GGUF/GLM-5.3-Flash-Strix128-IQ2S-IQ3XXS-00001-of-00003.gguf \
  -dev ROCm0 -ngl 999 -fa on -lm dio \
  -c 65536 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 -np 1 \
  --spec-type ngram-mod,draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
  --jinja --temp 0.7 --top-p 0.95 \
  --host 127.0.0.1 --port 8080
```

- `--spec-type ngram-mod,draft-mtp` stacks n-gram drafting (repeats of text already in the context: code edits, JSON, quoted documents) on the built-in MTP layer. It was the fastest drafting setup in every coding and agent test, with nothing extra to download.
- Memory on 128 GB: these settings stay around 116 GiB. The measurements below used 128K context and `-ub 4096`/`-ub 8192` on the 192 GB test machine, where the GPU peaked at 119-123 GiB; that is too close to a 128 GB machine's limit, so use 64K context with `-ub 2048`, or 128K with `-ub 512`.
- Built and tested with ROCm 7.1 (Ubuntu packages); ROCm 10.2 (TheRock) gave the same speeds except about 12% faster prefill of short prompts.

### Options

- `--jinja` is required: the chat template handles thinking and tool calls. Thinking is on by default; send `"chat_template_kwargs": {"enable_thinking": false}` in a request to turn it off, or cap it with `--reasoning-budget N`.
- `-lm dio` loads the weights with direct I/O. With the default memory-mapped load, the 112 GiB of page cache competes with the model for memory on a 128 GB machine.
- `-c`: the model supports up to 1M tokens of context; see the memory table above for what fits.

Then use any OpenAI-compatible client against `http://127.0.0.1:8080/v1`.

## Speed

ROCm 10.2, decode in tokens/s. "code" and "JSON" are short completions (median of 3), "chat" a 200-word answer, "15K"/"24K" generation after a 15K/24K-token prompt.

| engine | prefill (2K prompt) | prefill (24K prompt) | code | JSON | chat | 15K | 24K |
|---|---|---|---|---|---|---|---|
| llama.cpp | 194 t/s | 136 t/s | 16.3 | 16.3 | 16.3 | 13.2 | 13.2 |
| ROCmFPX `main`, no drafting | 88 t/s | 68 t/s | 16.8 | 16.8 | 16.8 | 14.8 | 14.1 |
| ROCmFPX `main`, MTP (built in) | | | 26.8 | 32.5 | 21.6 | 17.8 | 18.3 |
| branch, no drafting (`-ub 4096`) | 456 t/s | 435 t/s | 18.6 | 18.5 | 18.4 | 16.5 | 18.2 |
| branch, n-gram + MTP (`-ub 4096`) | | 430 t/s | 51.1 | 66.6 | 22.7 | 20.8 | 23.0 |

On the branch at 128K context: prefill 481 t/s at `-ub 4096` and 507 t/s at `-ub 8192` (8K-16K prompts); decode 19.1 t/s with an empty context and 18.0 t/s with 120K tokens of context. With n-gram + MTP: returning an edited 2K-token file 61.9 t/s, a 12K-token generation 24.7 t/s, a 100K-token prompt prefilled at 416 t/s.

These are local measurements on one machine (AMD Ryzen AI Max+ PRO 495 / Radeon 8065S, 192 GB LPDDR5X, Ubuntu 26.04, kernel 7.0), not leaderboard results.

## License and credits

- Model: [Z.ai GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), released under the MIT License, Copyright (c) 2026 Z.AI Co., Ltd. This repository redistributes quantized weights of that model under the same license; the full license text is in [LICENSE](LICENSE).
- Non-expert tensors: from [GLM-5.3-Flash-Gorgon-GGUF](https://huggingface.co/cafonez/GLM-5.3-Flash-Gorgon-GGUF), which was built from [Unsloth](https://huggingface.co/unsloth)'s GLM-5.3-Flash GGUF and importance matrix.
- llama.cpp GLM-5.3 support: [PR #27773](https://github.com/ggml-org/llama.cpp/pull/27773) and its contributors; ggml quantization types by the llama.cpp project (MIT).
- Calibration text includes [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) (CC BY-SA 3.0); only aggregate activation statistics were used, no text is included here.
