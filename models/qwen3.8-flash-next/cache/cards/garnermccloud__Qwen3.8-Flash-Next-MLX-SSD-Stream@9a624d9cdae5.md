---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx-serve
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-serve
- qwen4_exp
- moe
- multimodal
- sparse-attention
- ngram-embedding
---

# Qwen3.8 Flash-Next MLX SSD Stream

## Quality-first Flash-Next at 70+ tok/s on a 128 GB Mac

This prepared model prioritizes quality and speed: a BF16 core and verification
path, Q4 routed experts, native MTP, multimodal support, and the full
262,144-token context window. On an M4 Max with 128 GB unified memory, matched
warm workloads measured 71-74 tok/s for code and technical prose.

SSD Stream makes that configuration fit without keeping the model's
51.2 GB lookup table in unified memory. The table streams from SSD
while the GPU is already working, leaving that memory available for model
weights, KV cache, and macOS. A Q6 copy of the vocabulary head proposes MTP
tokens that are always checked by the untouched BF16 verification head.

## What this model is

| Component | Source |
|---|---|
| Transformer and vision weights | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at [`de4b8e4d43b917e7706784d8bb445c9af86a3540`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/commit/de4b8e4d43b917e7706784d8bb445c9af86a3540) |
| FP8 lookup table | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) at [`7b719225242aacd3dbd3f9407468c2ee9a9d2594`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/commit/7b719225242aacd3dbd3f9407468c2ee9a9d2594) |
| Serving engine | [garnermccloud/mlx-serve](https://github.com/garnermccloud/mlx-serve) |
| Tested hardware | Apple M4 Max with 128 GB unified memory |
| Prepared download | Approximately 127 GB |

The prepared repository contains the complete model and a read-only
`ngram_table.bin`; no conversion step or companion draft model is required.
The table payload is source-verified during conversion and remains in its
original FP8 E4M3 format.

## Performance

### M4 Max, 128 GB

| Result | Value |
|---|---:|
| Native context | 262,144 tokens |
| Model allocation before request caches | Approximately 75.3 GB |
| SSD-streamed lookup table | 51.2 GB |
| Complete prepared repository | Approximately 127 GB |
| Long-context prefill | 280.7 prompt tok/s at 248,445 tokens |

The decode comparison used the 262,144-token server context and native MTP at
depth 4. Rates are completion tokens divided by total request wall time,
median of three warm runs after one discarded warmup. Each pair used the same
prompt and greedy sampling.

| Workload | Completion | Native MTP | Serial | Speedup |
|---|---:|---:|---:|---:|
| Python CSV parser module | 600 tokens | **73.8 tok/s** | 33.2 tok/s | 2.22x |
| Technical Rayleigh-fading prose | 600 tokens | **71.2 tok/s** | 33.0 tok/s | 2.16x |
| Numbered unique sentences | 1,024 tokens | **54.4 tok/s** | 32.2 tok/s | 1.69x |

The uncached 248,445-token retrieval test returned its exact buried key and
stopped normally within the native context window. The release also passed 39
end-to-end checks covering text, structured tools, streaming, MTP, long-context
retrieval, images, concurrent decode, and mixed speculative and serial traffic.

## Run it

Build `mlx-serve`, then give it this Hugging Face model ID:

```bash
git clone --recurse-submodules https://github.com/garnermccloud/mlx-serve
cd mlx-serve
brew bundle install --file=Brewfile
./app/build.sh
./zig-out/bin/mlx-serve run garnermccloud/Qwen3.8-Flash-Next-MLX-SSD-Stream
```

The first run downloads the prepared files and then starts the server at
`http://localhost:11234`. Later launches reuse the verified local snapshot.
Native MTP and SSD streaming are selected from the model metadata, so there are
no model-specific serving flags.

## Capabilities

- 262,144-token native context
- text, image, and video input
- reasoning and structured tool calls
- OpenAI Chat Completions, Responses, Anthropic Messages, and Ollama APIs
- native MTP enabled automatically; no companion model or serving flags

## Precision

| tensors | width |
|---|---|
| routed experts, including MTP | Q4 affine, group 64 |
| draft-only vocabulary head | Q6 affine, group 64 |
| core, embeddings, verification head, vision | BF16 |
| n-gram table | source FP8 E4M3, streamed from SSD |

## How SSD Stream works

Flash-Next needs 16 deterministic lookup rows per token. Their addresses are
known before the model reaches the block that consumes them, so `mlx-serve`
starts persistent parallel `pread` operations while the preceding GPU work is
still running. It restores rows in token order, converts FP8 directly to BF16,
and waits only when storage takes longer than the overlapping computation.

The 51.2 GB table is mapped read-only. SSD Stream does not create a second
private copy or reserve a multi-gigabyte cache; macOS may retain useful pages
in its reclaimable filesystem cache.

## Hugging Face metadata

Hugging Face's automatic parameter and precision labels describe the packed
safetensors, not the complete source architecture. The transformer has 125B
parameters with 6B active per token. The architecture also includes the
51.2-billion-element PLE table and MTP weights, bringing its logical total near
180B. The PLE is stored separately in `ngram_table.bin`, so it is not included
in the safetensors parameter count.

## License and attribution

The model retains the
[Qwen Community License](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
Transformer and vision weights come from Qwen. The FP8 table comes from the
RadixArk NVFP4 checkpoint. Serving support and the prepared layout are provided
by [garnermccloud/mlx-serve](https://github.com/garnermccloud/mlx-serve), an
independent fork of [mlx-serve](https://github.com/ddalcu/mlx-serve).
