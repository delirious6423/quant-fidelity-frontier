---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: image-text-to-text
tags:
  - exl3
  - exllamav3
  - trellis
  - 3.05bpw
  - qwen4_exp
  - moe
  - vision
  - long-context
  - dgx-spark
  - veloGB10
---

# Qwen3.8-Flash-Next — EXL3 3.05 bpw

> **Original model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen
> team, released under the [Qwen Community License 1.0](LICENSE) — **not** an OSI license; see
> [License](#license) below. This artifact contains **modified weights** (EXL3 trellis quantization);
> the original model is © Qwen.

**Direct source:** [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3),
branch/revision **`3.05bpw_h5_ng5`** — 3.05 bpw experts, 5-bit head, n-gram table included.

An **EXL3 (ExLlamaV3 trellis) 3.05 bpw** quantization of **Qwen3.8-Flash-Next**: a 125B-parameter
mixture-of-experts model with about 6B active parameters, plus a 51B hashed n-gram embedding and a 4B
MTP layer, with a vision encoder. The pack is **~85 GB (~79 GiB) against ~360 GB for the original
bf16 release**.

**This quantized artifact exists first and foremost to run with the
[veloGB10](https://github.com/sf-stav/veloGB10) inference engine** (`gb10_inference`) on one, two or
four NVIDIA DGX Spark / GB10 systems — it is the pack we load, validate and serve with veloGB10's
EXL3 kernels. The weights are standard EXL3, so this artifact can be used **for any purpose, with any
framework that reads the format**, and it is fit and proven to work with veloGB10.

The quantization is **turboderp's `3.05bpw_h5_ng5` EXL3 pack** (exllamav3 1.4.4), redistributed here
with these differences from turboderp's repository: a vLLM-compatible `config.json` and a regenerated
tensor index `model.safetensors.index.json` (the pack's originals are kept as the `.native` files),
the `pack_scan.json` record behind the regenerated index, and the **original full-precision (bf16)
vision tower** as a separate file, `vision_tower_bf16.safetensors` (see [Notes](#notes)).

**The weights are the unmodified pack.** No abliteration, fine-tuning or merging is applied — the weight
files (shards, n-gram table, MTP patch) are byte-identical to the pack as distributed.

## Serving

Serve it with the **[veloGB10](https://github.com/sf-stav/veloGB10)** engine (`gb10_inference`),
which reads EXL3 packs directly from `--model-dir` at **TP=1, TP=2 and TP=4**. The engine loads its
PTX kernel artifacts from the working directory, so run it from wherever the binary and `src/ptx/`
live.

### Single node (TP=1)

```bash
./gb10_inference --server \
  --model-dir /path/to/model/Qwen3.8-Flash-Next-exl3-3.05bpw \
  --port 9000 \
  --max-seq-len 262144 \
  --max-batch 1 \
  --max-tokens 65536 \
  --prefix-cache on \
  --mtp=auto
```

### Two nodes (TP=2)

```bash
./gb10_inference --node --port 29500                                    # on the peer
```

```bash
./gb10_inference --server \
  --model-dir /path/to/model/Qwen3.8-Flash-Next-exl3-3.05bpw \
  --tp 2 \
  --nodes <node ip address>:29500 \
  --port 9000 \
  --max-seq-len 262144 \
  --max-batch 1 \
  --max-tokens 65536 \
  --prefix-cache on \
  --mtp=auto
```

### Four nodes (TP=4)

Start `./gb10_inference --node --port 29500` on **each of the three** peer GB10s, then:

```bash
./gb10_inference --server \
  --model-dir /path/to/model/Qwen3.8-Flash-Next-exl3-3.05bpw \
  --tp 4 \
  --nodes <node 1 ip>:29500,<node 2 ip>:29500,<node 3 ip>:29500 \
  --port 9000 \
  --max-seq-len 262144 \
  --max-batch 1 \
  --max-tokens 65536 \
  --prefix-cache on \
  --mtp=auto
```

**The peers need no model copy.** The head plans each rank's shard and ships only that, through the
TP blob cache (`~/.cache/gb10_tp`): roughly **57.5 GiB per node at TP=2** and **46.8 GiB per node at
TP=4**, not the whole pack. Only missing blobs move, so a second start of the same model syncs
nothing. The server is OpenAI-compatible on `http://<head-ip>:9000/v1`.

## Performance

Measured with **VeloBenchmark 0.1.0** (TP=1/TP=2 on 2026-09-30, TP=4 on 2026-10-02) against the
served model, one request at a time, reasoning effort low. Decode includes MTP speculation.

**Pure-code decode** — ANSI C sorting, ~3.2K output tokens:

| | TP=1 | TP=2 | TP=4 |
|---|---:|---:|---:|
| Decode median | 137 tok/s | 186 tok/s | **221 tok/s** |
| Decode min / max | 83.1 / 144 | 122 / 199 | 173 / 240 |
| Decode p50 / p90 / p99 | 137 / 142 / 144 | 186 / 193 / 195 | 221 / 234 / 237 |
| Time per output token (TPOT) | 7.5 ms | 5.4 ms | **4.6 ms** |
| Draft acceptance / depth | 84% / 5.7 | 85% / 6.0 | 84% / 5.6 |

**Prefill** — one measurement per input size:

| Input tokens | TP=1 tok/s | TP=2 tok/s | **TP=4 tok/s** |
|---|---:|---:|---:|
| ~550 | 1,323 | 2,001 | 2,285 |
| ~2.1K | 1,470 | 2,319 | **3,250** |
| ~6.2K | 1,517 | 2,441 | 3,227 |
| ~10.3K | 1,541 | 2,452 | 3,226 |
| ~18.5K | 1,541 | 2,451 | 3,246 |
| ~34.9K | 1,525 | 2,432 | 3,197 |

Time to first token (TP=4): 0.24 s at ~550 tokens, 0.64 s at ~2.1K, 1.92 s at ~6.2K, 3.19 s at
~10.3K, 5.69 s at ~18.5K and 10.90 s at ~34.9K.

Against a single GB10: **TP=2 is ×1.36 on decode and ×1.5–1.6 on prefill; TP=4 is ×1.6 on decode and
×2.0–2.2 on prefill.**

**Concurrency** — every figure above is `--max-batch 1`. With several busy requests the engine shares
one batched step from about **four** requests onward (roughly **×1.45 aggregate at 8 requests and
×1.9 at 16** on one node, code workload; TP=2/TP=4 proportionally). Two and three requests still take
turns, so their aggregate stays at the single-request rate. Greedy output is byte-identical to running
each request alone, at every topology.

## Specifications

| | |
|---|---|
| Base model | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Architecture | `qwen4_exp` — MoE, 125B total / ~6B active, 48 layers, hidden 2560, vocab 248,320 |
| Attention | hybrid 12 × (3 × Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE); GDN 48 V / 16 QK heads (head_dim 128), QSA 24 Q / 2 KV heads (head_dim 256, rope dim 64) with a 4-head MQA indexer |
| MoE | 512 experts, 10 routed per token, expert intermediate 640 |
| Extra parameters | 51B n-gram embedding (20M rows, bigrams/trigrams at layer 2) and a 4B MTP layer |
| Vision | vision encoder included (27 layers, hidden 1152) — image-text-to-text; a 5-bit tower is inside the shards and the **original full-precision (bf16) tower** is included as `vision_tower_bf16.safetensors` |
| Context | 262,144 tokens |
| Quantization | EXL3 trellis, `mul1` codebook — 3.05 bpw average, 5-bit `lm_head`, 5-bit n-gram table, 5-bit vision tower, 3-bit MTP; calibration 250 × 2048 |
| ExLlamaV3 tooling | 1.4.4 |
| Format / size | `exl3` — ~85 GB across 7 shards plus a separate 32.6 GB n-gram table |

## Files

| File | Notes |
|---|---|
| `model-00001..00007-of-00007.safetensors` | the quantized weights (~53 GB) |
| `ngram_embedding.safetensors` | the 32.6 GB n-gram table (5-bit, 128 shards) |
| `mtp_hyper_connection_mixer_patch.safetensors` | the MTP layer's mixer weights |
| `quantization_config.json` | full EXL3 per-tensor metadata |
| `config.json` | vLLM-compatible pack config (EXL3 `quantization_config`, incl. `non_routed_exl3` and the n-gram map) |
| `config.json.native`, `model.safetensors.index.json.native` | the pack's original config and index |
| `model.safetensors.index.json`, `pack_scan.json` | regenerated tensor index and the pack-scan record behind it |
| `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `generation_config.json` | tokenizer and chat template |
| `preprocessor_config.json`, `video_preprocessor_config.json` | vision preprocessors |
| `vision_tower_bf16.safetensors` | the **original full-precision (bf16) vision tower** (0.9 GB, sha256 `cdd69998…c641e`); veloGB10 uses it for image input when present, and otherwise falls back to the 5-bit tower inside the shards |
| `qbench_prompts.json`, `qbench_prompts.md` | the quality-benchmark prompt set that ships with the pack |
| `LICENSE`, `.gitattributes` | Qwen Community License 1.0; LFS rules |

## Notes

- **Quantization detail:** 3.05 bpw on the linear layers, with the `lm_head`, the n-gram table and the
  vision tower at 5 bits and the MTP layer at 3 bits. The n-gram embedding is sharded (128 shards,
  2,500,012 rows each) and is the larger part of the on-disk size.
- **Vision:** **images** are served at TP=1, TP=2 and TP=4; they are resized so the longer side is at
  most `--image-max-edge` (default 1024), and they work past the 2,051-token dense window. veloGB10 runs
  the tower from the included **full-precision bf16 file** (`vision_tower_bf16.safetensors`) when it is in
  the model folder, and falls back to the 5-bit tower inside the shards when it is not. **Video and audio
  parts return `400`** for now.
- **Quality:** the pack ships turboderp's `qbench_prompts` set and a `pack_scan.json` record; use those
  to reproduce the pack-level check on your own hardware.
- **Higher fidelity:** a [4.05 bpw pack](https://huggingface.co/doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw)
  (4-bit experts, 6-bit head) is available for veloGB10 v0.7.2 and newer; it is about 13% slower to decode
  on one or two nodes.
- **Long context is 262,144 tokens** per the config, not the 1M that the hosted Qwen3.8-Flash API
  offers.
- **Engine defaults change output bytes** versus veloGB10 v0.7.0: `--prefill-chunk 4095` (was 2048)
  and `--qsa-key-rope full`. Both have documented escape hatches in the engine's documentation.

## License

The **Qwen Community License 1.0** applies (a copy is included as `LICENSE`). It is a custom license,
not an OSI-approved one. In short, as of this writing:

- You may use, modify, publish and distribute the model and derivative works, provided the copyright
  notice and license are included.
- If you serve the model or a derivative as a **Model-as-a-Service or AI Work Assistant business**,
  you need a separate license from Qwen (internal use is exempt).
- Products with more than 100M monthly active users or US$20M monthly revenue must display the model
  name prominently.

Read `LICENSE` for the binding text. This repository redistributes Qwen's weights in quantized form
and is not affiliated with or endorsed by Qwen.

## Credits

- **Model and license:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) —
  © the Qwen team, Qwen Community License 1.0.
- **EXL3 quantization:** [turboderp](https://huggingface.co/turboderp) — the `3.05bpw_h5_ng5` pack and
  the accompanying [qbench](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) prompts, built
  with [exllamav3](https://github.com/turboderp-org/exllamav3).
- **Engine:** [veloGB10](https://github.com/sf-stav/veloGB10).
