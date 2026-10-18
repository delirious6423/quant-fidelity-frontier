---
license: mit
tags:
- unsloth
- deepseek_v4
- deepseek
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
---
# DeepSeek-V4-Flash-0731 REAP-W GGUF

Structurally pruned GGUF variants derived from
[`unsloth/DeepSeek-V4-Flash-0731-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF),
using the `UD-Q2_K_XL` quantization at revision
`fbbb5b93fb787c21338159b0af3318bb3f4d9768`.

## Variants

| Directory | Routed experts per MoE layer | Expert intermediate width | Size |
|---|---:|---:|---:|
| `UD-Q2_K_XL-E256-W1280` | 256 / 256 (unpruned) | 1280 / 2048 | 63.17 GB |
| `UD-Q2_K_XL-E216-W1280` | 216 / 256 (REAP-selected per layer) | 1280 / 2048 | 54.99 GB |

REAP ranks routed experts independently in each MoE layer using calibration-time
router weights and expert output norms. Width channels are selected independently
per expert using calibration-time activation scores. Width-sliced down projections
are requantized with a block size of 256.

Each directory contains a three-part sharded GGUF and a `slice_manifest.json` with
the source revision, slicing parameters, file sizes, and SHA-256 checksums.

These models require `llama.cpp` with the DeepSeek-V4 width-slicing changes from
the [`deepseekv4-width`](https://github.com/mmnga/llama.cpp/tree/deepseekv4-width) branch.

## Usage

```bash
git clone --branch deepseekv4-width --single-branch https://github.com/mmnga/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j

llama.cpp/build/bin/llama-server \
  -m UD-Q2_K_XL-E256-W1280/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf
```

Use `UD-Q2_K_XL-E216-W1280` in the model path to run the E216 variant.
