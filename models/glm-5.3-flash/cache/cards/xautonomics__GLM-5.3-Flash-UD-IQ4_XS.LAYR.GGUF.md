---
license: mit
base_model:
  - zai-org/GLM-5.3-Flash
library_name: llama.cpp
tags:
  - gguf
  - layer-library
  - multi-stage
  - ik_llama.cpp
  - dreamcatcher
  - xautonomics
---
# GLM-5.3-Flash-UD-IQ4_XS.LAYR.GGUF

A **per-layer GGUF library** of Unsloth's `unsloth/GLM-5.3-Flash-GGUF` (`UD-IQ4_XS`). These are the same quantized weights, sliced once into one file per transformer block plus the shared parts, so any contiguous layer window `[A, B)` can be assembled at load time. It's built for [dreamcatcher](https://github.com/xautonomics-inc/dreamcatcher), [xAutonomics](https://xautonomics.com)' fork of ik_llama.cpp, which serves a library either in a single process or split across hosts.

| | |
|---|---|
| base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| source quant | [`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9/UD-IQ4_XS) · `UD-IQ4_XS` · pinned revision `621d456e93` · 5 shards |
| architecture | `glm5next` · 46 blocks |
| files | 48 (embd, layer, nextn, output) · 146.5 GiB |
| integrity | `manifest.json`: blake2b-128 per whole file and per tensor |
| sliced with | `layer_distribution.slice` (the slicer commit is in the `SLICER_COMMIT` file); nothing was re-quantized |

## Support status

Measured on the dreamcatcher tree (status as of [`c2f8b680`](https://github.com/xautonomics-inc/dreamcatcher/blob/c2f8b680ca48f091c311dc8c10e49ea8dfcda3fa/docs/HF-MODEL-CARDS.md); the linked document is authoritative):

**glm5next — CUDA verified** (library head/tail ring coherent, greedy,
7.6 tok/s, experts on CPU; library ≡ monolith 1383/1383 hashes). The
mono-vs-library runtime A/B is done — hidden states md5-identical mono vs
library (the `STAGE_EMIT=hidden` logits abort that blocked it is fixed,
`meta#93`). Vulkan: not measured for this architecture on the published tree
(the RDNA3.5 APU proof was the expert-server path with GLM-5.3-Flash experts,
not a local Vulkan run).

Only model files from [huggingface.co/xautonomics](https://huggingface.co/xautonomics) are supported by dreamcatcher. Other GGUFs, including libraries you slice yourself, may load but are unsupported. **Do not rename the files**: the loader expects `blk-NNNNN.gguf` / `parts-*.gguf` / `manifest.json` exactly.

## Layout

```
blk-00000.gguf … blk-00044.gguf   one transformer block each
parts-embd.gguf                          token embedding (every stage loads it)
parts-output.gguf                        output norm + head (every stage loads it)
parts-nextn-NNNNN.gguf                   NextN/MTP block(s)
manifest.json                            file list, sizes, per-file + per-tensor hashes, source provenance
```

A stage that serves layers `[A, B)` needs `blk-A … blk-(B-1)` plus the `parts-*` files and `manifest.json`.

## Use

Single process (the whole model on one host):

```
llama-server --model-dir /models/GLM-5.3-Flash-UD-IQ4_XS.LAYR.GGUF
```

Multi-host ring (stages exchange hidden states over TCP; each host needs low-latency, direct node-to-node links — see the dreamcatcher README):

```
llama-stage-runner --role tail --listen 8081 --model-dir /models/GLM-5.3-Flash-UD-IQ4_XS.LAYR.GGUF --layers 24,46
llama-stage-runner --role head --connect <tail-host>:8081 --model-dir /models/GLM-5.3-Flash-UD-IQ4_XS.LAYR.GGUF --layers 0,24
```

Plan a split for your hosts (weights-only screen; the proof is a survived prefill), and verify the download before launching:

```
python3 -m layer_distribution.plan /models/GLM-5.3-Flash-UD-IQ4_XS.LAYR.GGUF/manifest.json --node a:16GiB:200GiB:8080 --node b:24GiB:200GiB:8081
python3 -c "from layer_distribution import verify; r = verify('/models/GLM-5.3-Flash-UD-IQ4_XS.LAYR.GGUF', '/models/GLM-5.3-Flash-UD-IQ4_XS.LAYR.GGUF/manifest.json'); print(r.passed, r.hash_verified)"
```

## Provenance and license

Weights are byte-for-byte the tensors of `unsloth/GLM-5.3-Flash-GGUF` at revision `621d456e93e926e4b52f85cff5f634358c1828f9` (Unsloth's dynamic quantization of `zai-org/GLM-5.3-Flash`). Source shard names and sizes are recorded in `manifest.json` (`source.shards[]`, `source.content_hash`). License follows the base model: **mit**, see `LICENSE`. Credit to the base-model authors and to Unsloth for the quantization.

## About

Published by [xAutonomics](https://xautonomics.com) for [dreamcatcher](https://github.com/xautonomics-inc/dreamcatcher).
