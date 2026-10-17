---
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - llama.cpp
  - speculative-decoding
  - mtp
  - draft-model
---

# Qwen3.8-Flash-Next MTP draft head, re-fit to a requantized target (GGUF)

A multi-token-prediction (MTP) draft head for Qwen3.8-Flash-Next, for speculative decoding. The target model
re-checks every drafted token, so it changes speed, never output. Fine-tuned to the **C2T8** requantization of the
model: live draft acceptance 0.651 -> **0.692** against the original head (details below).

Part of **[llama.cpp-lab](https://github.com/routhjim/llama.cpp-lab)**: measured llama.cpp changes for running
Qwen3.8-Flash-Next and Qwen3.8-27B on AMD Strix Halo and Radeon RX 7900 XTX (Vulkan), with every before/after number
and the negative results. Overview: **[routhjim.github.io/llama.cpp-lab](https://routhjim.github.io/llama.cpp-lab/)**.

## What you need

1. **A Qwen3.8-Flash-Next target GGUF.** Any quantization loads; the measured gain is on C2T8, whose recipe
   (built from unsloth's UD-Q4_K_XL plus its imatrix) is in the lab repo. On other quants it still works, unmeasured.
2. **A llama.cpp build with Qwen3.8-Flash-Next MTP, and the matching file:**

   | Build | File |
   |---|---|
   | [llama.cpp-lab](https://github.com/routhjim/llama.cpp-lab) (`main`) | `mtp-Qwen3.8-Flash-Next-Q4DRAFT-refit0930.gguf` |
   | upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) (Qwen4Exp MTP since #29761, 2026-10-01) | `mtp-Qwen3.8-Flash-Next-Q4DRAFT-refit0930-upstream.gguf` |

   The two files hold identical weights; upstream names the drafter's head-mixer tensors
   `blk.48.nextn.hc_head_*` instead of `output_hc_*`, so the upstream copy only renames those three. Checked on
   upstream master `2e7c58c54` (Strix Halo iGPU, Vulkan, greedy, 3 prompts): it loads and drafts with acceptance
   0.776 against 0.801 for the lab build, decode 45.2 vs 46.7 tok/s.

## Build and run

    # llama.cpp-lab (or upstream llama.cpp with the -upstream file and without --spec-coupled)
    git clone https://github.com/routhjim/llama.cpp-lab && cd llama.cpp-lab
    cmake -B build -DGGML_VULKAN=ON -DGGML_VULKAN_MMV_MAX_COLS=32 -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j

    build/bin/llama-server -m Qwen3.8-Flash-Next-<quant>-00001-of-0000N.gguf \
      -md mtp-Qwen3.8-Flash-Next-Q4DRAFT-refit0930.gguf -ngld 99 \
      --spec-type draft-mtp --spec-draft-n-max 3 --spec-coupled \
      -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -lm mmap -lzm auto -c 262144 -np 1 --jinja

- `--spec-draft-n-max 3` (draft depth 3) is what was measured; depth 2 was slightly worse on this model.
- `--spec-coupled` (lab-only) shares sampling noise between drafter and target, so sampled (temperature > 0)
  output still accepts good drafts. Leave it out for greedy decoding.
- Always pass `-lm mmap` for Flash-Next; `auto` can turn mmap off for every device and load the whole file into RAM.
- Tested on AMD (Strix Halo iGPU and Radeon RX 7900 XTX) with Vulkan/RADV. CUDA and ROCm are untested.
- Other layouts (iGPU only, or split over an iGPU and two dGPUs) are in the lab README under "Running Flash-Next".

## Files

| File | For | Size | SHA-256 |
|---|---|---|---|
| `mtp-Qwen3.8-Flash-Next-Q4DRAFT-refit0930.gguf` | llama.cpp-lab | 2.48 GiB | `260e61a89d40b5ae2e5d301f44ef9fc63bea2e808f5dda43897155b29c054020` |
| `mtp-Qwen3.8-Flash-Next-Q4DRAFT-refit0930-upstream.gguf` | upstream llama.cpp | 2.48 GiB | `e4cdcf0695632d27b9c0d9fe6b11da93824e3d81fefae9dfa265d76e74961535` |

Experts q4_K/q5_0, the rest q8_0/f32.

## Why

Requantizing a model shifts its output distribution a little. The MTP head was trained against
the original weights, so after a requant its drafts are accepted less often, mostly at draft
positions 2 and 3. Re-fitting the head to the requantized model's own predictions recovers that
and more.

## Results (AMD Strix Halo, Vulkan, llama.cpp-lab, MTP depth 3, np1)

Live A/B on the same server and 12 real agent requests, only the drafter file differs:

| | original head | this head |
|---|---|---|
| draft acceptance (server counters) | 0.651 (9241 / 14190) | **0.692** (8215 / 11871) |
| decode, time-weighted | 38.7 tok/s | 40.0 tok/s (+3.4%) |
| decode, median request | 40.7 tok/s | 41.6 tok/s (+2.2%) |

Offline, on held-out agent traffic (per draft position, expected acceptance):

| position | 1 | 2 | 3 | accepted per 3-token round |
|---|---|---|---|---|
| original head | 0.839 | 0.781 | 0.741 | 1.979 |
| this head | 0.871 | 0.810 | 0.771 | 2.119 |

The decode gain is within this box's ~3% run-to-run noise; the acceptance gain is not.

## How it was made

2 epochs (about 20 minutes on the Strix Halo iGPU) over ~400k positions of the model's own agent
traffic, rendered through its chat template. The loss is soft cross-entropy against the target's
top-20 distribution at draft depths 1 to 3, unrolled the way the draft KV cache runs. Trained:
attention, the embedding/hidden projection, hyper-connection mixers, norms, router and shared
expert (89M parameters). Frozen: the 512 experts and the embedding/output tables.

Full pipeline and write-up: https://github.com/routhjim/llama.cpp-lab
(`tools/mtp-dump`, `scripts/mtp-refit/`, `docs/flash-next-mtp-refit.md`).

## Caveats

- One live A/B of 12 requests; decode gain inside the noise band.
- Training corpus is mostly terminal-agent traffic (~85%); other workloads were not measured.

## License

Derived from Qwen3.8-Flash-Next and distributed under the Qwen Community License 1.0 (LICENSE).
