---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
tags:
- colibri
- moe
- mxfp4
---

# DeepSeek-V4-Flash-0731-speedy-colibri-mxfp4

MXFP4 container for [colibrì](https://github.com/GriffinPilz/SpeedyColibri), a Rust MoE
inference engine for single-box streaming-expert inference on a DGX Spark (GB10).

**This repository exists so the model can be downloaded and run directly, with no
conversion step.** The routed experts are QAT-native MXFP4 and are copied through
**bit-exact** — a dequant→requantize round trip through another 4-bit format measured
**6.40% rel-RMS of pure loss and 5.9% more bytes**, i.e. worse on both axes.

- **Upstream:** [`unsloth/DeepSeek-V4-Flash-0731`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731)
- **Base model:** [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- **Size:** 145 GiB, 45 shards (from a 167 GB fp8 source)
- **Licence:** MIT, inherited from the base model. The upstream `LICENSE` ships in this repo.

## Architecture

43 layers, **every one MoE** — 256 routed experts top-6 plus 1 shared.

| piece | detail |
|---|---|
| residual | **Hyper-Connections** — 4 copies of the hidden state, `[b,s,4,4096]`, not an ordinary residual stream |
| attention | latent, `head_dim` 512 / 1 KV head, **O-LoRA** output projection (rank 1024, 8 groups), attention sinks |
| context | 128-token sliding window over raw KV; everything older is reachable **only** through the Compressor (41/43 layers, ratios alternating 4/128) |
| sparsity | DSA-family **Indexer** on 21/43 layers, `index_topk` 512 |
| routing | `sqrt(softplus)` scores + `noaux_tc`; the **first 3 layers bypass the router entirely** and select experts from a `tid2eid[token_id]` hash table |
| experts | MXFP4 (block-32 nibbles + E8M0 scales), passed through unquantised; dense weights fp8 e4m3 |

## Measured on one DGX Spark (GB10, 121.7 GiB), 2026-08-09

| | value |
|---|---|
| decode | **4.7 tok/s** end-to-end (4.7 forward-only) |
| prefill | **13.8 tok/s** (37.1 s) |
| serving | **4.3 tok/s** median, 12 diverse prompts over HTTP |
| expert residency | 79% of 11008 routed experts resident; the rest stream from NVMe (measured 2026-08-05, not re-run) |

**Context runs to the model's full 1,048,576 tokens**, at **13.4 KB/token** of KV.

An earlier revision of this card reported a ~40k ceiling at 206.9 KB/token. That was an
engine limitation, not the checkpoint's — the KV cache allocated the full context for the
raw latent tier even though only the 128-token window is ever read — and it has since been
fixed: the raw tier is a ring sized to the window, so those rows are a per-*sequence* cost
rather than a per-token one. V4 is now cheaper per token than a 120B hybrid (13.4 vs 16 KB)
despite caching KV on all 43 layers.

**`decode` is the noisiest column here.** The eight reps behind that median spanned **4.53-4.78 tok/s** forward-only in one session — a 5.5% spread with no code change. Treat a single suite as one draw, and do not read a difference against an earlier revision of this card as a speedup.

Method, caveats and the full fleet table: [SpeedyColibri README](https://github.com/GriffinPilz/SpeedyColibri#1--how-fast-is-it).

## Use

The engine is [github.com/GriffinPilz/SpeedyColibri](https://github.com/GriffinPilz/SpeedyColibri). With it cloned and built, one command downloads this container and one serves it:

```bash
scripts/fetch.sh deepseek-v4-flash
scripts/serve.sh deepseek-v4-flash 8080
```

`serve.sh` fetches the container itself when it is missing, so the first line is optional. To download it by hand instead, mind the directory name — the registry resolves `deepseek-v4-flash` to `-container-v2`, and any other path will serve nothing:

```bash
huggingface-cli download Kanposer/DeepSeek-V4-Flash-0731-speedy-colibri-mxfp4 \
  --local-dir ~/models/DeepSeek-V4-Flash-0731-container-v2
```

## Credits

DeepSeek for the model and its reference implementation; [unsloth](https://huggingface.co/unsloth)
for the fp8 checkpoint this was converted from; [colibrì](https://github.com/pjw1/colibri)
for the original streaming-expert idea this engine grew from.
