---
license: mit
base_model: zai-org/GLM-5.3-Flash
pipeline_tag: text-generation
library_name: gguf
quantized_by: cafonez
tags:
- gguf
- glm
- glm5-next
- llama.cpp
- rocmfpx
- vulkan
- strix-halo
- gorgon-halo
- framework-desktop
---

# GLM-5.3-Flash GGUF for the Framework Desktop (Gorgon Halo)

A 3.84 bpw GGUF of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320.8B total parameters, MoE with 288 experts, top-8), sized to run fully on the GPU of a 192 GB AMD Ryzen AI Max 400 ("Gorgon Halo") Framework Desktop.

- Format: llama.cpp `glm5-next` (llama.cpp [PR #27773](https://github.com/ggml-org/llama.cpp/pull/27773)). It loads in current llama.cpp `master` and in [ROCmFPX](https://github.com/ROCmFPX/ROCmFPX).
- Size: 143.5 GiB (154.1 GB) in 4 shards. Pass the first shard to `-m`; llama.cpp finds the rest.
- The MTP (next-token prediction) layer is included, but llama.cpp does not run GLM-5.3 MTP drafting yet, so plain decoding is used.

## Files

| file | size |
|---|---|
| `GLM-5.3-Flash-Q5K-IQ3S-00001-of-00004.gguf` | 44.4 GB |
| `GLM-5.3-Flash-Q5K-IQ3S-00002-of-00004.gguf` | 44.2 GB |
| `GLM-5.3-Flash-Q5K-IQ3S-00003-of-00004.gguf` | 43.9 GB |
| `GLM-5.3-Flash-Q5K-IQ3S-00004-of-00004.gguf` | 21.6 GB |

## Quantization recipe

Routed experts carry almost all of the weights, so they get the low-bit types; everything that every token passes through stays at 5 bits or more.

| tensors | type | bpw | params | size |
|---|---|---|---|---|
| routed expert gate / up (41 layers) | IQ3_S | 3.44 | 198.1B | 79.3 GiB |
| routed expert down (39 layers) | IQ4_XS | 4.25 | 94.2B | 46.6 GiB |
| routed expert down (3 layers) | Q6_K | 6.56 | 7.3B | 5.5 GiB |
| routed experts, remaining layers | IQ4_XS / Q4_K / Q3_K | 3.4-4.5 | 12.1B | 5.6 GiB |
| attention (MLA), KDA, dense FFN, shared experts | Q5_K | 5.50 | 7.2B | 4.6 GiB |
| indexer, hyper-connections, small projections | Q8_0 | 8.50 | 1.2B | 1.2 GiB |
| output head | Q6_K | 6.56 | 0.6B | 0.5 GiB |
| norms, biases | F32 | | | 0.2 GiB |

Built from Unsloth's GLM-5.3-Flash UD-IQ4_XS GGUF and its importance matrix (`glm-imatrix.gguf`), requantizing the attention and dense tensors to Q5_K and the routed experts as above. The file was then converted from the pre-merge `glm5next` layout to the merged `glm5-next` layout: keys renamed, `indexer.kpool_select_tail = true` and `indexer.types` (all `full`) taken from the official `text_config`, and the gate/up experts stored as separate tensors. The conversion is a byte copy of the weights.

## Quality

- KL divergence against Unsloth's UD-IQ4_XS GGUF (the file this mix was made from): mean KLD 0.040, same top-1 token 93.1% (`llama-perplexity --kl-divergence`, 16 x 1024-token chunks, Vulkan).
- Vulkan vs CPU on this exact file (12K tokens, ctx 2048): mean KLD 0.018, same top-1 96.3%. This measures backend numerics, not quantization loss.

## Speed on the Framework Desktop (Gorgon Halo)

AMD Ryzen AI Max+ PRO 495 / Radeon 8065S, 192 GB LPDDR5X, Ubuntu 26.04, kernel 7.0, Mesa 26.0.8 RADV, Vulkan backend. ROCmFPX `main` (GLM-5.3 support from [#34](https://github.com/ROCmFPX/ROCmFPX/pull/34), prefill work from [#35](https://github.com/ROCmFPX/ROCmFPX/pull/35) and [#36](https://github.com/ROCmFPX/ROCmFPX/pull/36)). `llama-bench -fa 1 -b 2048 -ub 2048`, 3 runs:

| test | t/s |
|---|---|
| prefill, 2048 tokens | 256.4 |
| decode | 17.4 |
| prefill, 2048 tokens at 32K context | 95.9 |
| decode at 32K context | 15.0 |

These are local measurements on one machine, not leaderboard results. Plain llama.cpp `master` runs the file at the same decode speed; prefill is lower there because it does not have #35 and #36.

## How to run

### 1. Let the GPU use the memory

The weights need about 144 GiB of GPU-visible memory plus the KV cache. On a 192 GB machine, raise the GTT limit with kernel parameters (this example gives the GPU 176 GiB) and reboot:

```
ttm.pages_limit=46137344 ttm.page_pool_size=46137344
```

On Ubuntu, add them to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, run `sudo update-grub`, and reboot. Check with:

```
cat /sys/class/drm/card*/device/mem_info_gtt_total   # about 189000000000 bytes
```

Keep the BIOS GPU memory carve-out at its minimum: the GPU uses GTT, and memory carved out as VRAM is no longer available to it as GTT.

### 2. Build ROCmFPX (or llama.cpp) with Vulkan

```
sudo apt install build-essential cmake libvulkan-dev glslc mesa-vulkan-drivers
git clone https://github.com/ROCmFPX/ROCmFPX && cd ROCmFPX
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-cli
```

Stock llama.cpp (`https://github.com/ggml-org/llama.cpp`) builds the same way and also runs this file, with slower prefill.

### 3. Download

```
hf download cafonez/GLM-5.3-Flash-Gorgon-GGUF --local-dir GLM-5.3-Flash-Gorgon-GGUF
```

### 4. Serve

```
export GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 GGML_VK_MAX_NODES_PER_SUBMIT=2000 GGML_VK_ALLOW_GRAPHICS_QUEUE=1
./build/bin/llama-server \
  -m GLM-5.3-Flash-Gorgon-GGUF/GLM-5.3-Flash-Q5K-IQ3S-00001-of-00004.gguf \
  -dev Vulkan0 -ngl 999 -fa on \
  -c 131072 -b 2048 -ub 2048 -np 1 \
  --jinja --temp 0.7 --top-p 0.95 \
  --host 127.0.0.1 --port 8080
```

- `--jinja` is required: the chat template handles thinking and tool calls. Thinking is on by default; send `"chat_template_kwargs": {"enable_thinking": false}` in a request to turn it off, or cap it with `--reasoning-budget N`.
- `-c`: 131072 fits comfortably with 176 GiB of GTT. The model supports up to 1M tokens; larger contexts need more memory, and `-ub 1024` lowers the compute buffers if you run short.
- `-ub 2048` gives the best prefill; decode is the same at any `-ub`.
- The three `GGML_VK_*` variables were set for all measurements above.

Then use any OpenAI-compatible client against `http://127.0.0.1:8080/v1`.

## Credits

- Model: [Z.ai GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT).
- Importance matrix and the source GGUF: [Unsloth](https://huggingface.co/unsloth).
- llama.cpp GLM-5.3 support: [PR #27773](https://github.com/ggml-org/llama.cpp/pull/27773) and its contributors.
