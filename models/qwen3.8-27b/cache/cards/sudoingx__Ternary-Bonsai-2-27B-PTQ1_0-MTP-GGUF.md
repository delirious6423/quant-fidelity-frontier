---
license: apache-2.0
base_model:
- prism-ml/Ternary-Bonsai-2-27B-gguf
- Qwen/Qwen3.8-27B
library_name: llama.cpp
pipeline_tag: text-generation
tags:
- gguf
- qwen3_5
- ternary
- ptq1_0
- mtp
- speculative-decoding
- bonsai
- rtx-3060
- rtx-5060-ti
- amd
- radeon
- vulkan
- blackwell
- windows
- linux
- local-ai
---

# Ternary Bonsai 2 27B PTQ1_0 with the MTP head, for Linux and Windows

One line installs it on the card you already own, RTX 20, 30, 40 and 50, and AMD Radeon in beta, with 8 GB or more: it picks the build and the model file for your VRAM, checks every SHA256 and starts it, with the chat page at `http://127.0.0.1:8899` and an OpenAI-compatible API at `/v1` (model `bonsai-2-27b`).

Windows, in PowerShell (measured on native Windows on an RTX 3060 12GB and an RTX 5070 Ti 16GB; RTX 20 in beta):

```
irm https://raw.githubusercontent.com/sudoingX/local-ai-on-your-gaming-gpu/main/install.ps1 | iex
```

Linux or WSL2:

```
curl -fsSL https://raw.githubusercontent.com/sudoingX/local-ai-on-your-gaming-gpu/main/install.sh | bash
```

AMD Radeon (beta): the same two lines, on PrismML's own Vulkan build, no ROCm to install. 8 GB cards get up to 64K, 12 GB about 195K, 16 GB and up the full 262K with vision, all from the files in this repo; the cards and the numbers are in [Radeon (beta)](https://github.com/sudoingX/local-ai-on-your-gaming-gpu#radeon-beta).

Uncensored sibling: [Offrail 27B](https://huggingface.co/sudoingx/Offrail-27B-Uncensored), the same installer and bundles with OS-Software's uncensored weights (`--model offrail` on Linux, `$env:BONSAI_MODEL='offrail'` on Windows).

The installers, the serve lines and the sweeps live in [github.com/sudoingX/local-ai-on-your-gaming-gpu](https://github.com/sudoingX/local-ai-on-your-gaming-gpu).

The 1.75 bpw PTQ1_0 build of Ternary Bonsai 2 27B (PrismML's ternary compression of Qwen 3.8 27B, 5.95 GB, the file that fits 8GB and 12GB cards) with the Qwen 3.8 27B multi-token-prediction head grafted back on as block 64, so the PrismML llama.cpp fork can run `--spec-type draft-mtp` on it.

Three Bonsai 2 MTP grafts already exist, all on the PQ2_0 file (7.21 GB): [ProCreations](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP), [decent-jawfish](https://huggingface.co/decent-jawfish/bonsai-2-27b-mtp), [BoldingBuilds](https://huggingface.co/BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF). BoldingBuilds also tried PTQ1_0 and measured +1.6%: the head drafts fine, the verification does not pay, because on the 1.75 bpw kernels a 3-token batch cost about 3x a single token. This repo is that graft, and [PrismML-Eng/llama.cpp#218](https://github.com/PrismML-Eng/llama.cpp/pull/218) is the kernel that removes the wall: on an RTX 3060 12GB a 3-token verify now costs 1.55x a single token instead of 2.28x, and single-token decode itself goes from 26 to 40 tok/s.

## Download and run

One file: `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf` (7.0 GB). It runs on the PrismML fork release binary as shipped, no patch, and it runs faster on the kernel branch. Every number below is this file.

```
hf download sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf --local-dir .
git clone -b bonsai2 https://github.com/sudoingX/llama.cpp && cmake -S llama.cpp -B build -DGGML_CUDA=ON && cmake --build build -j $(nproc)
GGML_CUDA_BATCH_INVARIANT=1 build/bin/llama-server -m Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf \
  -ngl 99 -fa on -c 131072 -np 1 -ctk q4_0 -ctv q4_0 --jinja --reasoning-effort medium \
  --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8899 --spec-type draft-mtp --spec-draft-n-max 1
```

RTX 50 cards: CUDA 12.8 or newer for the build (`nvcc --version`), and v1.1 of the branch. If you built it before v1.1, clone it again: the branch moved onto PrismML's latest release.

No compiler: extract the prebuilt tarball for your card from this repo, driver only. RTX 30 and 40 take the cuda12.4 sm86-sm89 one and run `serve-12gb-mtp.sh`; RTX 20 take the cuda12.4 sm75 one and run its own `serve-8gb-mtp.sh` (`CTX=49152` on an 8 GB card with no display) or `serve-12gb-mtp.sh`; RTX 50 take the cuda12.8 sm120 one, which adds `serve-16gb-mtp-vision.sh` for 16GB cards. Every script passes `--reasoning-effort medium` for you. On the stock PrismML release binary the same serve line works without the env var, at the +8% row below.

## Numbers

RTX 3060 12GB, one slot, thinking off, 131072 context, q4_0 K/V, client-measured tok/s (`probe.py` in the [repo](https://github.com/sudoingX/local-ai-on-your-gaming-gpu), 3 prompts x 3 runs, medians), this file:

| build | flag | tok/s |
| --- | --- | ---: |
| PrismML release prism-b10685 | off | 25.0 |
| PrismML release prism-b10685 | draft-mtp, n-max 1 | 27.1 (+8%, the wall) |
| #218 kernel branch | off | 39.8 |
| #218 kernel branch, `GGML_CUDA_BATCH_INVARIANT=1` | draft-mtp, n-max 1 | **50.1** (code 53.2, bash 50.1, prose 41.8) |
| #218 kernel branch, `GGML_CUDA_BATCH_INVARIANT=1` | draft-mtp, n-max 2 | 45.4 |
| #218 kernel branch, 41.8K tokens filled | head off / n-max 1 / n-max 2 | 22.23 / 26.86 / 21.08 (release binary, head off: 16.85) |
| #218 kernel branch, 18K / 39K / 77K / 115K filled | head off → n-max 1 | 29.6 → 33.9 / 22.9 → 27.5 / 16.1 → 18.3 / 12.4 → 13.9 (+12% to +20%; n-max 2 below head off at every one of these depths) |

Draft acceptance on this card: 0.85 to 0.95 on Python, 0.73 to 0.81 on bash, 0.45 to 0.68 on prose, 0.65 to 0.73 on a long document at 18K to 120K tokens of context, 0.88 on an image prompt with the vision tower loaded. The head was trained by Qwen against fp16 hidden states and here it reads a ternary trunk; acceptance is lower than on stock Qwen 3.8 (about 0.70) and still well above break-even.

`llama-bench` on the same card, original PTQ1_0 file, `-fa 1 -ctk q4_0 -ctv q4_0 -r 3`: tg128 26.32 → 40.54 tok/s, pp512 269.6 → 268.7 tok/s (prefill untouched). Other owners' rows, 8GB through 5070 Ti, are in the repo's `sweeps/` and on the PR threads.

## RTX 50 series (Blackwell), v1.1

RTX 5060 Ti 16GB, the same probe and file, served with the 16GB line (262144 context, the mmproj loaded) unless stated:

| build | flag | tok/s |
| --- | --- | ---: |
| PrismML release prism-b10743 | off | 42.0 |
| `bonsai2` v1.1 (the cuda12.8 sm120 tarball) | off | 53.4 |
| `bonsai2` v1.1, `GGML_CUDA_BATCH_INVARIANT=1` | draft-mtp, n-max 1 | **67.3** (code 71.0, bash 67.3, prose 59.5) |
| same, 38,774 / 119,489 tokens filled | draft-mtp, n-max 1 | 39.5 / 21.0 (RTX 3060: 27.5 / 13.6) |
| same, four requests at once, 32K each | draft-mtp, n-max 1 | 127.1 aggregate, 31.7 to 36.2 per request (prism-b10743, head off: 58.3) |

On 16GB the head, the vision tower and the full 262,144 window fit together: 15,070 MiB after load, 15,326 MiB after an image request, all six facts of a shop-page screenshot read right (site, headline, product, price, button colour, number of links), 69.6 tok/s on the image answer with 53 of 60 drafts accepted. With `GGML_CUDA_BATCH_INVARIANT=1`, head off and n-max 1 give byte-identical text here too.

v1.1 carries two fixes that matter only on RTX 50 cards (details in the repo's `kernel/blackwell.md`). Before it, a source build of the branch decoded a run of `!` with the head off on these cards, because the PTQ1_0 mat-vec kernel did not wait for its input under programmatic dependent launch; if you are on an older build, put `GGML_CUDA_PDL=0` in front of the server command. And llama.cpp's flash-attention q4_0 V dequant sat on the stack on sm_120, which halved long-context decode with a q4_0 K/V cache: llama-bench tg128 at 64K depth 12.41 tok/s before, 29.40 after, same output. RTX 30 and 40 cards are unaffected by both.

## RTX 20 series (Turing), v1.4

The same v1.1 code built for sm_75. On an RTX 2070 Super 8GB the 8 GB line (the lean file, the head at n-max 1, ubatch 256, q4_0 K/V) decodes at 52.6 tok/s with the head and 42.6 without (live-stream, greedy, 3 x 512 tokens), holds 48K on a card with no display and 16K with a desktop on it (64K does not load on 8 GB), and reads the vision check right at 16K with `--image-max-tokens 1024`. All of the branch's PTQ1_0 test suites pass on sm_75. On the same card the CUDA build decodes 40% faster than PrismML's Vulkan build. Rows: `sweeps/rtx2070super-8gb.md` in the repo.

## Losslessness

With `GGML_CUDA_BATCH_INVARIANT=1` on the kernel branch, greedy output (temperature 0, top_k 1, 300 tokens, 3 prompts) with the head on is byte-identical to greedy output with the head off; the transcripts are in the repo under `results/identity/`. Without that switch, on any build, batched verification changes the floating-point reduction order and the text can differ at near-ties, which is what the other grafts report too. On RTX 20 (Turing) the switch does not cover every kernel: a near tie can flip a token late in an answer (2 of 5 test prompts at n-max 1), so the head is fast there but not byte-identical. Sampling at the model card's `temperature 1.0, top_p 0.95, top_k 20` is unaffected either way; speculative decoding verifies every draft token against the target.

## Files

| file | bytes | needs |
| --- | ---: | --- |
| `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf`, **the file** | 7,012,820,512 | the PrismML fork, prism-b10685 or later, as released. Carries its own unrotated copy of the token embedding for the head, so `--spec-type draft-mtp` starts without any patch. For the speed above, the kernel branch. |
| `bonsai2-small-gpu-linux-x64-cuda12.4-sm86-sm89-285542d.tar.gz` | 514,982,389 | nothing but the NVIDIA driver. `llama-server`, `llama-bench`, `llama-cli` built from the `bonsai2` branch (kernel + Hadamard fix), CUDA runtime bundled, serve scripts for 8GB, 12GB and 12GB with the head. RTX 30 and 40 (Ampere and Ada). |
| `bonsai2-small-gpu-linux-x64-cuda12.8-sm120-ff41412.tar.gz` | 646,541,994 | nothing but the NVIDIA driver, 570 or newer. The same three binaries built from `bonsai2` v1.1 (the kernel and both RTX 50 fixes on PrismML's prism-b10743), the CUDA 12.8 runtime bundled, serve scripts for 8GB, 12GB, 12GB with the head and 16GB with the head and the vision tower. RTX 50 series. |
| `bonsai2-small-gpu-windows-x64-cuda12.4-sm86-sm89-ff41412.zip` | 550,702,580 | Windows 10 or 11 x64 and an RTX 30 or 40 card with NVIDIA driver 551.61 or newer. `llama-server.exe`, `llama-cli.exe` and `llama-bench.exe` built from `bonsai2` v1.1 (ff41412) on GitHub Actions, with the CUDA 12.4 runtime, cuBLAS and the Microsoft C++ runtime inside. |
| `bonsai2-small-gpu-windows-x64-cuda13.3-sm120-ff41412.zip` | 447,095,737 | Windows 10 or 11 x64 and an RTX 50 card with NVIDIA driver 580 or newer. The same three programs from the same commit, with the CUDA 13.3 runtime, cuBLAS and the Microsoft C++ runtime inside. |
| `bonsai2-small-gpu-linux-x64-cuda12.4-sm75-ff41412.tar.gz` | 445,797,836 | nothing but the NVIDIA driver, 525 or newer. The same three binaries built from `bonsai2` v1.1 for sm_75, the CUDA 12.4 runtime bundled, serve scripts for 8GB and 12GB with the head. RTX 20 series (Turing). |
| `bonsai2-small-gpu-windows-x64-cuda12.4-sm75-ff41412.zip` | 473,325,481 | Windows 10 or 11 x64 and an RTX 20 card with NVIDIA driver 551.61 or newer, in beta. The same three programs from the same commit, built on GitHub Actions, with the CUDA 12.4 runtime, cuBLAS and the Microsoft C++ runtime inside. |

Smaller variant, optional: `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf`, 6,297,658,848 bytes, the same graft without the embedding copy, 715 MB smaller (on a 12GB card that is 196K of context instead of 163K). Same output. It needs the 15-line qwen35 MTP Hadamard fix in the build (PrismML-Eng/llama.cpp#205, or the `bonsai2` branch above); the prism-b10685 release binary refuses to start the draft graph on it. #205 is in PrismML's release prism-b10743; the lean file on that release binary is not tested here yet.

`SHA256SUMS` in the repo covers all nine files; each tarball carries its own `SHA256SUMS` for its files. RTX 50 tarball facts: NVIDIA driver 570 or newer, glibc 2.38 or newer with the libstdc++ of GCC 13 or newer (Ubuntu 24.04, Fedora 39 or newer; on Ubuntu 22.04 or Debian 12 build from source), AVX2; extracted and run from a clean directory under `env -i` on the RTX 5060 Ti, greedy output byte-identical with the head on and off and across three sessions, 67.3 tok/s probe median with the head, llama-bench tg128 54.5 tok/s fresh and 29.5 at 64K depth, pp512 1,061.9 tok/s. RTX 30 and 40 tarball facts: NVIDIA driver 525 or newer, glibc 2.35 or newer, AVX2; extracted and run from a clean directory with no toolchain in the environment, identity byte-identical, 48.0 tok/s median, and prompt processing 522.5 tok/s at pp512 against 267.3 on the previous bundle, same card and same session, llama-bench r=3, the gain coming from the PTQ1_0 MMQ tile loader now underneath the branch; the lean file on the same binaries loads at 9,940 MiB at 131072 and 11,732 MiB at 196608, the full file does not fit 196608 on 12 GB (the MTP compute buffer needs 1,040 MiB more).

Parents:

- PrismML `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5,946,648,928 bytes, sha256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`. Every one of its 851 tensors is present with identical bytes and offsets; `tools/merge.py --strip` on the merged file reproduces this sha256.
- unsloth `Qwen3.8-27B-UD-Q4_K_M.gguf`, 16,464,440,224 bytes, sha256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`. Source of the 15 `blk.64.*` tensors (Q6_K, Q8_0, F32, 335 MiB) and, in the full file, of `blk.64.nextn.embed_tokens.weight` (a Q4_K copy of its token_embd, 682 MiB).

Header changes versus Bonsai 2: `qwen35.block_count` 64 → 65, `qwen35.nextn_predict_layers = 1`, plus `graft.donor.name`, `graft.head_blocks`, `graft.head_tensor_count`, `graft.tool` for provenance. Everything else, the `prism.hadamard.*` keys and the tokenizer included, is byte-identical. Without `--spec-type draft-mtp` the file behaves exactly like Bonsai 2, the head is skipped.

## Serving notes

**Set `--reasoning-effort medium`.** The GGUF chat template defaults it to `xhigh`, which injects a "think carefully through the task" system line, and on an RTX 3060 at a 4,096-token client cap that returned **nothing at all** on all three build tasks tested: an SVG, a single-file HTML page and a 100-line Python CLI each spent the whole 4,096 tokens thinking and emitted no answer, in 108 seconds. At `medium` the same three completed in 45 to 58 seconds. Raising the cap does not fix it on its own, the SVG was still empty after 16,384 thinking tokens and 477 seconds. Three tasks, two caps, two runs each, every pair identical; the table is in [`kernel/reasoning_effort.md`](https://github.com/sudoingX/local-ai-on-your-gaming-gpu/blob/main/kernel/reasoning_effort.md). Per request it is the OpenAI `reasoning_effort` field; the serve scripts in the repo already pass it.

VRAM on an RTX 3060 12GB: full file 10,638 MiB at 131072, 11,726 MiB at 163840 (with `-ctkd q4_0 -ctvd q4_0`); lean file 9,956 MiB at 131072, 11,990 MiB at 196608. On an RTX 5060 Ti 16GB: full file 14,218 MiB at 262144, 15,070 MiB with the mmproj added. Stock ggml-org llama.cpp cannot read PTQ1_0 and produces gibberish on any Bonsai 2 file; use the PrismML fork or the branch above.

Sampling per the base card: thinking `temperature 1.0, top_p 0.95, top_k 20`; instruct `temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5`. `--spec-draft-n-max 1` was best on this card at every depth measured, fresh to 120K tokens; n-max 2 only pays on a fresh context and loses to head-off past ~16K. Sweep it on yours.

## Windows

Native since v1.2, no WSL needed, and measured on real cards: 48.6 tok/s with the head on an RTX 3060 12GB and 84.7 on an RTX 5070 Ti 16GB, native Windows 10 (`sweeps/rtx3060-12gb-windows.md` and `sweeps/rtx5070ti-16gb.md` in the repo). The one line at the top picks the zip for your card (RTX 20, in beta, and RTX 30 and 40: CUDA 12.4, driver 551.61 or newer; RTX 50: CUDA 13.3, driver 580 or newer), puts the model next to it, makes a desktop shortcut and opens the chat page. `bench-bonsai.bat` measures your card and copies the results to the clipboard. On 12 GB and up `--fit` sizes the window to the VRAM the desktop leaves free. On a card that is tight on VRAM, set NVIDIA Control Panel, Manage 3D settings, CUDA - Sysmem Fallback Policy to Prefer No Sysmem Fallback, so an overflow fails instead of silently spilling into system RAM. WSL2 still works with the Linux line: the Windows NVIDIA driver, no Linux driver inside WSL, the model files in `~/`, not `/mnt/c`.

## How it was made

15 `blk.64.*` tensors, a full transformer block plus the `nextn` projections, copied verbatim from the unsloth GGUF into the Bonsai GGUF with the two header keys above; the full file also copies the donor's token embedding for the head's own lookup. The residual stream of Bonsai 2 is in the original basis (its RMSNorm weights match stock Qwen 3.8 elementwise), so the head receives the activations it was trained on; only the embedding table is Hadamard-latent, which is what the lean file's 15-line fix inverts. Merge tool, byte-exact strip, identity and batch-numerics probes, 15 tests and `recipe.txt`: `graft/` in the [repo](https://github.com/sudoingX/local-ai-on-your-gaming-gpu). The kernel: `kernel/` there, and #218.

## Credit

PrismML for Bonsai 2 and the fork; Qwen for the head; unsloth for the donor GGUF. decent-jawfish, ProCreations and BoldingBuilds for the PQ2_0 grafts and the Hadamard patch, and BoldingBuilds for publishing the PTQ1_0 negative result this file answers. The graft recipe traces back to [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp).

## Licence

All parents are Apache 2.0 (Qwen/Qwen3.8-27B by Alibaba Cloud, Ternary Bonsai 2 27B by PrismML, unsloth's GGUF export of the former). This repo redistributes their weights unchanged under the same licence. The tarball carries llama.cpp's MIT licence and the NVIDIA CUDA runtime redistribution notice. Not affiliated with PrismML, Qwen or unsloth.
