---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- sushi
- exl3
- apple-silicon
- qwen4_exp
- moe
---

<p align="center"><img src="assets/sushi-horizon.png" alt="sushi" width="100%"></p>

# Qwen3.8-Flash-Next-Sushi-3bpw

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) packed for
[sushi](https://github.com/beamivalice/sushi), a native inference engine for Apple Silicon.
**Target: a 64 GB Mac.** On 96 GB and 128 GB Macs it runs the full 1M-token context.

| part | how it is stored |
|---|---|
| routed experts (48 layers x 512, plus the MTP layer) | EXL3, made with our own converter |
| dense part (attention, GDN, shared experts, hyper-connections, indexer, embedding, lm_head) | 8-bit affine; the same in both packs |
| n-gram embedding table | 4-bit affine, group size 32 (mlx-serve's quantization of the released bf16 table), one file `ngram_table.bin`, read from the SSD and never loaded into GPU memory |
| vision tower (Qwen3-VL ViT) | included: image and video input |
| MTP head | included |
| context | up to 1,048,576 tokens (`config.json` sets YaRN x4 over the native 262,144) |
| disk | ~79 GiB: 49.3 GiB of weights + 29.8 GiB n-gram table |

**sushi only.** This pack needs sushi v1.0.0 or later on Apple Silicon with macOS 26.2 or later. It does not load
in transformers, vLLM, mlx-lm or exllamav3.

## Run it on a 64 GB Mac

```bash
# sushi: the release binary (ad-hoc signed; curl does not quarantine it)
curl -L https://github.com/beamivalice/sushi/releases/latest/download/sushi-bin-macos-arm64.tar.gz | tar xz

# this pack
hf download beamster/Qwen3.8-Flash-Next-Sushi-3bpw --local-dir ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-3bpw

# let the GPU use 59000 MB of the 64 GB (until the next reboot)
sudo sysctl iogpu.wired_limit_mb=59000

# serve on 127.0.0.1:12345
# 1. images, 8-bit KV — the default quality
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-3bpw \
  --mtp --kv-quant 8 --mtp-head-kv-quant --ctx-size 128000 \
  --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --prefix-cache-mem 1GB --temp 1

# 2. images, 4-bit KV — twice the context, at 9% KLD and 0.7 points of next-token agreement
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-3bpw \
  --mtp --kv-quant 4 --mtp-head-kv-quant --ctx-size 256000 \
  --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --prefix-cache-mem 1GB --temp 1
```

- `--kv-quant 8` keeps the KV cache at 8 bits, and `--mtp-head-kv-quant` does the same for the MTP head's own KV.
- `--prefill-chunk 2048` caps the prompt tokens forwarded per step, which keeps the prefill peak small.
- `--prefix-cache-disk 10GB` keeps up to 10 GB of seen prompt prefixes on the SSD, so a repeated prompt skips its
  prefill, across restarts too.
- MTP (the model's own draft head) is on by default.
- A browser download of the binary is quarantined by macOS: clear it with
  `xattr -dr com.apple.quarantine sushi-macos-arm64`.

### Faster sampling: `--mtp-typical 0.2`

By default a draft token is kept only when it is exactly what the model would have sampled. Add `--mtp-typical 0.2` to
also keep drafts the model finds likely enough: sampled decoding gets faster, and greedy (temperature 0) output does not
change at all. Measured on Sushi-4bpw (M5 Max, 16 prompts x 512 tokens, temperature 1.0, top_k 20, top_p 0.95, two
seeds):

| acceptance | decode tok/s | tokens per step | NLL of the generated text |
|---|---|---|---|
| exact (default) | 69.2 / 63.3 | 2.1-2.2 | 0.7805 |
| `--mtp-typical 0.2` | 80.0 / 76.6 | 2.4-2.6 | 0.7819 |

That is 15-20% faster decode. The NLL (the model's own score of what it wrote, lower is better) moves by 0.001 nats,
well inside the 0.1-nat spread between two seeds of the default.

## Memory

Everything below is GPU memory, in GiB (what `sushi run` reports). The n-gram table is read from the SSD and never
enters GPU memory.

| | Sushi-3bpw | Sushi-4bpw |
|---|---|---|
| model weights | 47.51 | 61.58 |
| MTP head | 0.98 | 1.27 |
| vision tower | 0.84 | 0.84 |
| **weights loaded** | **49.33** | **63.68** |
| KV cache, 256k tokens | 4.06 | 4.06 |
| KV cache, 512k tokens | 8.12 | 8.12 |
| KV cache, 1M tokens | 16.25 | 16.25 |
| **total at 256k / 512k / 1M** | **53.4 / 57.5 / 65.6** | **67.7 / 71.8 / 79.9** |

The KV cache is for one request at 8 bits with MTP on and `--mtp-head-kv-quant`: 16,640 bytes per token of context
(attention 13,056, long-context indexer 2,304, MTP head 1,280). Each concurrent request adds its own. Leave room for
the hot prefix cache (`--prefix-cache-mem`, 2 GiB by default) and the prefill buffers, which `--prefill-chunk 2048`
keeps small. macOS gives the GPU only part of the RAM by default; the `sysctl` line in the launch command raises that
limit until the next reboot.

## Quality

<p align="center"><img src="assets/kld-chart.png" alt="KLD vs size" width="100%"></p>

KLD against the bf16 checkpoint: 16 prompts x 512 tokens, scored to the first EOS (7186 positions), KV cache at kv8,
every pack run by the same sushi build. Lower is better. The n-gram table is not counted: it stays on the SSD.

| pack | weights in GPU memory (GiB) | KLD | top-1 agreement |
|---|---|---|---|
| oMLX oQ5e | 83.97 | 0.0625 | 92.40% |
| **Sushi-4bpw** | **63.68** | **0.0632** | **92.99%** |
| mlx-serve [mixed-4-8bit](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit) | 70.13 | 0.0818 | 91.39% |
| **Sushi-3bpw** (4-bit n-gram table) | **49.33** | **0.1047** | **90.34%** |
| oMLX oQ4e | 69.21 | 0.1370 | 88.87% |
| MLX affine q3 (3-bit experts, 8-bit dense) | 54.94 | 0.1444 | 88.05% |
| mlx-serve [iQ-MLX 3.3bpw](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw) | 50.60 | 0.1987 | 86.28% |

At about the same memory, Sushi-4bpw halves oQ4e's KLD (0.063 vs 0.137) and matches oQ5e with 20 GiB less. At about 50 GiB,
Sushi-3bpw has about half the KLD of mlx-serve's iQ-MLX 3.3bpw (0.105 vs 0.199), and beats MLX affine q3 by 27% with
5.6 GiB less. Each pack is measured with the n-gram table it ships: 4-bit g32 in Sushi-3bpw and mlx-serve's packs, 4- or
5-bit in oMLX's, bf16 in Sushi-4bpw and affine q3. With the bf16 table, Sushi-3bpw reads 0.1012. The oMLX and
mlx-serve packs were measured as published; the oMLX n-gram tables were repacked into sushi's file unchanged. Affine
q3 was built for this comparison from the same bf16 checkpoint.

## Choosing the n-gram table

Sushi-3bpw and Sushi-4bpw ship different copies of the same n-gram table, and either copy works with either pack. Use
the 4-bit table to save 66 GiB of disk, or the bf16 table for a small quality gain. The table is read from the SSD, so
GPU memory is the same either way.

| table | size on disk | ships with | KLD with Sushi-3bpw | KLD with Sushi-4bpw |
|---|---|---|---|---|
| 4-bit, group size 32 | 29.8 GiB | Sushi-3bpw | 0.1047 | 0.0666 |
| bf16 | 95.4 GiB | Sushi-4bpw | 0.1012 | 0.0632 |

To swap, download the other pack's `ngram_table.bin` into this pack's folder, replacing the current one:

```
hf download beamster/Qwen3.8-Flash-Next-Sushi-4bpw ngram_table.bin --local-dir <this pack's folder>
```

Then set `ngram_table` in `config.json` to match the file:
`"ngram_table": {"file": "ngram_table.bin", "bits": 16, "group_size": 0}` for the bf16 table.

## Credits and license

- **Qwen team**: the base model, [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). This
  pack is a derivative under the **Qwen Community License 1.0** (`LICENSE`), including its conditions on large
  commercial services.
- **turboderp**: the EXL3 format.
- **ddalcu**: [mlx-serve](https://github.com/ddalcu/mlx-serve). sushi is a fork of it.
- **David Tai**: his work on --mtp-typical
