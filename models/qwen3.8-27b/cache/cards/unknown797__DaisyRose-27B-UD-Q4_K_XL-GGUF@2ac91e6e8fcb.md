---
license: apache-2.0
base_model:
  - RadixArk/DaisyRose-27B
  - Qwen/Qwen3.8-27B
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - qwen3_8
  - daisyrose
  - unsloth
  - ud-q4_k_xl
  - hrx-loom
  - gufo
  - multimodal
---

# DaisyRose-27B-UD-Q4_K_XL-GGUF

A dynamic **`UD-Q4_K_XL`** (`5.17 BPW`) GGUF quantization of **[RadixArk/DaisyRose-27B](https://huggingface.co/RadixArk/DaisyRose-27B)** (based on **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**) matching **[Unsloth's](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q4_K_XL.gguf)** exact 866-tensor dynamic quantization layout, engineered for full compatibility with AMD Strix Halo (`gfx1151`) HRX-Loom and **[Loom Fun](https://huggingface.co/loom-fun)** kernels on **[Gufo](https://gufo.blog)**.

## Key Files
* **`DaisyRose-27B-UD-Q4_K_XL.gguf`** (17.66 GB): Main model weights with 866 preserved tensors and native MTP speculative head.
* **`mmproj-F16.gguf`** (0.93 GB): Multimodal vision-language projector (from [Unsloth's `Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)).
* **`mmproj-DaisyRose-27B-BF16.gguf`** (0.93 GB): Original multimodal vision-language projector for standard `llama.cpp` / LM Studio use, from [RadixArk's `DaisyRose-27B-GGUF`](https://huggingface.co/RadixArk/DaisyRose-27B-GGUF/blob/main/mmproj-DaisyRose-27B-BF16.gguf).

## Benchmarks & Evaluation Status
* **Formal Quality Benchmarks:** *Not yet benchmarked* — standardized accuracy evals (such as MMLU, GPQA, or SWE-bench) have not been run yet. Community evaluations are welcome!
* **Throughput Verified:** Tested locally on Windows 11 AMD Ryzen AI Max+ 395 ("Strix Halo", `gfx1151`) with Adrenalin Edition 26.11.2 via **[Gufo v0.37.0](https://gufo.blog)** with **[Loom Fun](https://huggingface.co/loom-fun)**.

## Quick Setup

Loom - Use provided ([mmproj-F16.gguf](https://huggingface.co/unknown797/DaisyRose-27B-UD-Q4_K_XL-GGUF/blob/main/mmproj-F16.gguf)).

1. **[Gufo](https://gufo.blog) / [Loom Fun](https://huggingface.co/loom-fun):** Fetch the Loom package ([`loom-fun/unsloth-Qwen3.8-27B-UD-Q4_K_XL`](https://huggingface.co/loom-fun/unsloth-Qwen3.8-27B-UD-Q4_K_XL)), place these `.gguf` files into its `parameter/` directory, and run:
   ```bash
   gufo serve --host 127.0.0.1 --port 8080 --sessions 2 llm --served-model-name daisyrose-27b --model /path/to/daisyrose27b-loom --speculative mtp
   ```

Not using Loom - Use RadixArk/DaisyRose-27B original ([mmproj-DaisyRose-27B-BF16.gguf](https://huggingface.co/RadixArk/DaisyRose-27B-GGUF/blob/main/mmproj-DaisyRose-27B-BF16.gguf))

2. **Gufo (without Loom):** Serve the main weights directly with Gufo, using `dflash2` speculative decoding with [`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) (1.14 GB) as the draft model:
   ```bash
   gufo serve --host 127.0.0.1 --port 8080 --sessions 2 llm --served-model-name daisyrose-27b --model /path/to/DaisyRose-27B-UD-Q4_K_XL.gguf --speculative dflash2 --dflash-model /path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --mmproj /path/to/mmproj-DaisyRose-27B-BF16.gguf
   ```

3. **Standard `llama.cpp` / LM Studio:**
   ```bash
   llama-server -m DaisyRose-27B-UD-Q4_K_XL.gguf --mmproj mmproj-DaisyRose-27B-BF16.gguf --temp 1.0 --top-p 0.95 --top-k 20
   ```

## Acknowledgments
Shoutouts to **[Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)** for the dynamic `UD-Q4_K_XL` quantization recipe and `mmproj-F16.gguf`, the **[RadixArk](https://huggingface.co/RadixArk)** and **[Qwen](https://huggingface.co/Qwen)** teams for the base models, the **[Gufo](https://gufo.blog)** and **[Loom Fun](https://huggingface.co/loom-fun)** teams for `gfx1151` runtime support, and special thanks to **Jeremy Fowers** for bringing **HRX-Loom** to my attention.
