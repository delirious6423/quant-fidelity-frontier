---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- llama.cpp
- speculative-decoding
- mtp
- gguf
- qwen
---

# Qwen 3.8 Flash Next - MTP Draft Speculative Model (GGUF)

This repository provides working MTP (Multi-Token Prediction) draft models in GGUF format for **Qwen 3.8 Flash Next** (and experimental architectures with hyper-connections / QSA / Hadamard KV rotations), powered by [cafe-llama.cpp](https://github.com/quimmedes/cafe-llama.cpp).

## Compatible Engine

This MTP model requires the fork implementation supporting hyper-connection hidden states, QSA Hadamard KV rotation, and MTP layer graph generation:

- **Repository**: [https://github.com/quimmedes/cafe-llama.cpp](https://github.com/quimmedes/cafe-llama.cpp)

```bash
git clone https://github.com/quimmedes/cafe-llama.cpp.git
```

### Building from Source

Pick the backend for your hardware, then build. `-j N` sets the number of compile threads — a safe value is the number of physical cores.

**1. NVIDIA CUDA (Windows / Linux)**

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 2
```

**2. Vulkan (Cross-Platform AMD / Intel / NVIDIA)** — requires the Vulkan SDK installed

```bash
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j 2
```

**3. AMD ROCm / HIP (Linux / Windows)**

```bash
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1100;gfx1030"
cmake --build build --config Release -j 2
```

**4. Apple Metal (macOS)**

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j 2
```

**5. CPU Only (AVX2 / AVX-512)**

```bash
cmake -B build -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
cmake --build build --config Release -j 2
```

## Available Files

- `mtp-Qwen3.8-Flash-Next-Q2_0.gguf`: Quantized Q2_0 MTP draft model (~1.10 GB)
- `mtp-Qwen3.8-Flash-Next-Q3_0.gguf`: Quantized Q3_0 MTP draft model (~1.50 GB)
- `mtp-Qwen3.8-Flash-Next-Q4_0.gguf`: Quantized Q4_0 MTP draft model (~2.20 GB) - *Recommended*

Re-quantized from BF16 with llama-quantize (cafe-llama.cpp build c2f5366e3). The previous
Q4_K_M/Q6_K/Q8_0/BF16 files were removed: they were produced from a broken conversion and
did not work with speculative decoding.

## Usage with llama.cpp / llama-server

Run `llama-server` or `llama-cli` with `--spec-type draft-mtp` and point `-md` to your preferred MTP file.
Setting `--spec-draft-n-max 2` provides optimal acceptance rate (~50-70%) for a 1-layer MTP draft head:

```bash
llama-server   -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf   -md mtp-Qwen3.8-Flash-Next-Q4_0.gguf   --spec-type draft-mtp   --spec-draft-n-max 2   -ngl 999   -hmoe   -fa on   -ctk q8_0 -ctv q8_0 -kvu   -c 8192 -b 1024 -ub 128
```

### CLI Example:

```bash
llama-cli   -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf   -md mtp-Qwen3.8-Flash-Next-Q4_0.gguf   --spec-type draft-mtp   --spec-draft-n-max 2   -ngl 999   -hmoe   -fa on   -ctk q8_0 -ctv q8_0   -p "Explain quantum entanglement simply."
```