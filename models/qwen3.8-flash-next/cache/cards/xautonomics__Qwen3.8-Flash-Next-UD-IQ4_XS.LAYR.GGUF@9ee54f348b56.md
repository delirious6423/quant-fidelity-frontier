---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
library_name: llama.cpp
tags:
  - gguf
  - layer-library
  - multi-stage
  - ik_llama.cpp
  - dreamcatcher
  - xautonomics
---
# Qwen3.8-Flash-Next-UD-IQ4_XS.LAYR.GGUF

A **per-layer GGUF library** of Unsloth's `unsloth/Qwen3.8-Flash-Next-GGUF` (`UD-IQ4_XS`). These are the same quantized weights, sliced once into one file per transformer block plus the shared parts, so any contiguous layer window `[A, B)` can be assembled at load time. It's built for [dreamcatcher](https://github.com/xautonomics-inc/dreamcatcher), [xAutonomics](https://xautonomics.com)' fork of ik_llama.cpp, which serves a library either in a single process or split across hosts.

| | |
|---|---|
| base model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| source quant | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97/UD-IQ4_XS) · `UD-IQ4_XS` · pinned revision `38bb39ee97` · 3 shards |
| architecture | `qwen4exp` · 48 blocks |
| files | 51 (embd, layer, other, output) · 87.8 GiB |
| integrity | `manifest.json`: blake2b-128 per whole file and per tensor |
| sliced with | `layer_distribution.slice` (the slicer commit is in the `SLICER_COMMIT` file); nothing was re-quantized |

## Support status

Measured on the dreamcatcher tree (status as of [`c2f8b680`](https://github.com/xautonomics-inc/dreamcatcher/blob/c2f8b680ca48f091c311dc8c10e49ea8dfcda3fa/docs/HF-MODEL-CARDS.md); the linked document is authoritative):

**qwen4exp — CUDA verified** (monolith coherent 23.2 tok/s; library head/tail
ring coherent 17.4 tok/s; library ≡ monolith tensor-wise and forward-pass
byte-identical). Ring token-exactness: `[0,40)+[40,48)` on CPU is
token-identical to the monolith (64/64); other splits and CUDA rings diverge
at near-tie tokens (6/15) from attention-kernel path differences (`meta#98`),
not window metadata — the earlier "tail drops `per_layer_token_embd`"
explanation was wrong (`ple.layers = [1]`, always in the head; `meta#91`
fixed). `llama-server --model-dir` on the library is token-identical to the
monolith (64/64, 1224/1224 hashes; `meta#90` fixed). **Vulkan fixed** (was degenerate
on every vendor — AMD RDNA3/RADV and NVIDIA coopmat1 alike; the cause was a
short `MULTI_ADD` `src0` descriptor range, not flash attention or BF16 tensor
placement): NVIDIA coopmat1 verified coherent and agreeing with CUDA within
rounding; RDNA3 re-measure pending; RDNA4 / Intel ANV not measured for this
architecture (`meta#88`).

Only model files from [huggingface.co/xautonomics](https://huggingface.co/xautonomics) are supported by dreamcatcher. Other GGUFs, including libraries you slice yourself, may load but are unsupported. **Do not rename the files**: the loader expects `blk-NNNNN.gguf` / `parts-*.gguf` / `manifest.json` exactly.

## Layout

```
blk-00000.gguf … blk-00047.gguf   one transformer block each
parts-embd.gguf                          token embedding (every stage loads it)
parts-output.gguf                        output norm + head (every stage loads it)
parts-other.gguf                         remaining non-block tensors (every stage loads it)
manifest.json                            file list, sizes, per-file + per-tensor hashes, source provenance
```

A stage that serves layers `[A, B)` needs `blk-A … blk-(B-1)` plus the `parts-*` files and `manifest.json`.

## Use

Single process (the whole model on one host):

```
llama-server --model-dir /models/Qwen3.8-Flash-Next-UD-IQ4_XS.LAYR.GGUF
```

Multi-host ring (stages exchange hidden states over TCP; each host needs low-latency, direct node-to-node links — see the dreamcatcher README):

```
llama-stage-runner --role tail --listen 8081 --model-dir /models/Qwen3.8-Flash-Next-UD-IQ4_XS.LAYR.GGUF --layers 24,48
llama-stage-runner --role head --connect <tail-host>:8081 --model-dir /models/Qwen3.8-Flash-Next-UD-IQ4_XS.LAYR.GGUF --layers 0,24
```

Plan a split for your hosts (weights-only screen; the proof is a survived prefill), and verify the download before launching:

```
python3 -m layer_distribution.plan /models/Qwen3.8-Flash-Next-UD-IQ4_XS.LAYR.GGUF/manifest.json --node a:16GiB:200GiB:8080 --node b:24GiB:200GiB:8081
python3 -c "from layer_distribution import verify; r = verify('/models/Qwen3.8-Flash-Next-UD-IQ4_XS.LAYR.GGUF', '/models/Qwen3.8-Flash-Next-UD-IQ4_XS.LAYR.GGUF/manifest.json'); print(r.passed, r.hash_verified)"
```

## Provenance and license

Weights are byte-for-byte the tensors of `unsloth/Qwen3.8-Flash-Next-GGUF` at revision `38bb39ee97` (Unsloth's dynamic quantization of `Qwen/Qwen3.8-Flash-Next`). Source shard names and sizes are recorded in `manifest.json` (`source.shards[]`, `source.content_hash`). License follows the base model: **qwen-community-1.0**, see `LICENSE`. Credit to the base-model authors and to Unsloth for the quantization.

## About

Published by [xAutonomics](https://xautonomics.com) for [dreamcatcher](https://github.com/xautonomics-inc/dreamcatcher).
