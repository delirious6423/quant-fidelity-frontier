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
  - 4.05bpw
  - qwen4_exp
  - moe
  - vision
  - long-context
  - dgx-spark
  - veloGB10
---

# Qwen3.8-Flash-Next — EXL3 4.05 bpw

> **Original model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen
> team, released under the [Qwen Community License 1.0](LICENSE) — **not** an OSI license; see
> [License](#license) below. This artifact contains **modified weights** (EXL3 trellis quantization);
> the original model is © Qwen.

**Direct source:** [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3),
branch/revision **`4.05bpw_h6_ng6`** (commit `55a732e0c4c3d4614bc42b68493bb930d9b02c0a`) — 4.05 bpw average
(4-bit experts), 6-bit head, 6-bit n-gram table.

An **EXL3 (ExLlamaV3 trellis) 4.05 bpw** quantization of **Qwen3.8-Flash-Next**: a 125B-parameter
mixture-of-experts model with about 6B active parameters, plus a 51B hashed n-gram embedding and a 4B
MTP layer, with a vision encoder. It is the higher-quality sibling of the
[3.05 bpw pack](https://huggingface.co/doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw): about **108 GB on disk**
(68 GB of weights plus a 39 GB n-gram table) against ~360 GB for the original bf16 release.

**This artifact exists first and foremost to run with the
[veloGB10](https://github.com/sf-stav/veloGB10) inference engine** (`gb10_inference`) on one, two or
four NVIDIA DGX Spark / GB10 systems. The weights are standard EXL3, so it can be used **for any purpose,
with any framework that reads the format**.

**This is the unmodified pack.** Every weight, index, tokenizer and config file is byte-identical to
turboderp's `4.05bpw_h6_ng6` revision (checked against the published file hashes). The only addition is
**`vision_tower_bf16.safetensors`** — the original bf16 vision tower of Qwen3.8-Flash-Next
(333 `model.visual.*` tensors, 897,899,504 bytes, sha256
`cdd69998e52e34badced49ef8ec09824b8b9a52e488094f0cadeadc7ebfc641e`) — the **full-precision vision encoder**.
**veloGB10 takes image input from this file and does not read the pack's own 6-bit tower
(`vision_k6.safetensors`)**, so keep it next to the weights if you want images: without it the server still
starts, but is text-only and refuses image requests. Other loaders can ignore it and use `vision_k6.safetensors`.

> **Requires veloGB10 v0.7.2 or newer.** The 4-bit expert and 6-bit dense/head kernels, the 6-bit n-gram
> reader and the per-rank shipping of the n-gram table are new in v0.7.2 (it adds a fourth PTX file,
> `src/ptx/exl3_bench_k6.ptx`, to the deploy set). Earlier engine versions cannot load this pack.

## Serving

Serve it with the **[veloGB10](https://github.com/sf-stav/veloGB10)** engine (`gb10_inference`),
which reads EXL3 packs directly from `--model-dir` at **TP=1, TP=2 and TP=4**. The engine loads its
PTX kernel artifacts from the working directory, so run it from wherever the binary and `src/ptx/`
live.

### Single node (TP=1)

One GB10 holds the 68 GB of weights plus the lane KV, so the **39 GB n-gram table is read from SSD**
(`--ple-ram auto` chooses this by itself when RAM is short; you can force it with `--ple-ram ssd`).
Decode is about 1–3% slower than with the table in RAM (cold prefill a few percent more); output is identical.

```bash
./gb10_inference --server \
  --model-dir /path/to/model/Qwen3.8-Flash-Next-EXL3-4.05bpw \
  --port 9000 \
  --max-seq-len 262144 \
  --max-batch 1 \
  --max-tokens 65536 \
  --prefix-cache on \
  --mtp=auto
```

If a configuration cannot fit (more lanes or context than the memory allows), the server refuses to start
**before loading** and prints the arithmetic and the way out (`--max-batch`, `--max-seq-len`, `--tp 2`).

### Two nodes (TP=2)

```bash
./gb10_inference --node --port 29500                                    # on the peer
```

```bash
./gb10_inference --server \
  --model-dir /path/to/model/Qwen3.8-Flash-Next-EXL3-4.05bpw \
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
  --model-dir /path/to/model/Qwen3.8-Flash-Next-EXL3-4.05bpw \
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
TP blob cache (`~/.cache/gb10_tp`): about **71 GiB per node at TP=2** and **57 GiB per node at TP=4**
(including the n-gram table, which every rank holds). Only missing blobs move, so a second start of the
same model syncs nothing. The server is OpenAI-compatible on `http://<head-ip>:9000/v1`.

### Concurrency

`--max-batch N` serves N requests at once (each lane's KV is allocated up front). With one busy request the
engine speculates exactly as before; with several, it picks per round between speculative rounds and one
shared batched step. Aggregate throughput starts to rise from about 4 busy requests.

## Performance

These are **smoke-test figures from the engine's own 1,000-token generation and 2K-token prefill, single
request, untuned** — they are *not* VeloBenchmark results (those will follow). For reference, the 3.05 bpw
pack measured with the same smoke test gives about 73 tok/s decode / 1,550 tok/s prefill at TP=1,
102 / 2,570 at TP=2 and 122 / 3,400 at TP=4.

| | TP=1 | TP=2 | TP=4 |
|---|---:|---:|---:|
| Decode (tok/s, 1,000 tokens, MTP on) | 63 | 89 | 120 |
| Prefill (tok/s, 2K tokens) | 1,399 | 2,612 | 3,496 |

4.05 decodes about 0.86–0.87× as fast as 3.05 at TP=1 and TP=2 (more bytes per token) and about the same at
TP=4. MTP draft acceptance on this pack is about 52% on prose (2.8 tokens per round).

## Verification

- **Against the exllamav3 reference implementation, bit for bit:** every 4-, 5- and 6-bit weight tensor class
  (15.8 million trellis blocks, including all dense layers, the 6-bit `lm_head`, the 4-bit experts and
  the MTP layer) decodes identically to exllamav3's own reconstruction, and the 6-bit n-gram path (hashing,
  row fetch, decode, end-of-sequence segmentation) matches exllamav3's `NGramEmbedding` on 1,536 sampled
  positions (24,576 row ids and 3.9 million values, zero differences). The engine's CUDA decode was also
  checked bit-exact against that host decoder on the 6-bit `lm_head` (2.5 million blocks).
- **Quality sanity:** next-token loss on a 73K-token corpus equals the 3.05 pack within noise (0.625 vs 0.621
  at 8K windows, 1.579 vs 1.578 at 16K). That test cannot resolve the small KL-divergence advantage
  turboderp reports for 4.05; it is a mis-decode alarm, not a quality ranking.
- **Not covered by an external reference:** the full end-to-end forward pass was not compared against
  exllamav3, only the individual tensors, lookups and the loss check above.

## Specifications

| | |
|---|---|
| Base model | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Architecture | `qwen4_exp` — MoE, 125B total / ~6B active, 48 layers, hidden 2560, vocab 248,320 |
| Attention | hybrid 12 × (3 × Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE); GDN 48 V / 16 QK heads (head_dim 128), QSA 24 Q / 2 KV heads (head_dim 256, rope dim 64) with a 4-head MQA indexer |
| MoE | 512 experts, 10 routed per token, expert intermediate 640 |
| Extra parameters | 51B n-gram embedding (20M rows, bigrams/trigrams at layer 2) and a 4B MTP layer |
| Vision | vision encoder (27 layers, hidden 1152) — image-text-to-text; the **original full-precision (bf16) tower is included** as `vision_tower_bf16.safetensors` (the pack's own 6-bit tower is `vision_k6.safetensors`) |
| Context | 262,144 tokens |
| Quantization | EXL3 trellis, `mul1` codebook — 4.05 bpw average (4-bit experts), 6-bit dense layers, 6-bit `lm_head`, 6-bit n-gram table, 6-bit vision tower file, MTP layer with 4-bit experts and 4–6-bit dense layers; calibration 250 × 2048 |
| ExLlamaV3 tooling | 1.4.4 |
| Format / size | `exl3` — ~68 GB across 9 shards plus a separate 39 GB n-gram table and ~1.5 GB of vision towers |

## Files

| File | Notes |
|---|---|
| `model-00001..00009-of-00009.safetensors` | the quantized weights (~68 GB) |
| `ngram_embedding.safetensors` | the 39 GB n-gram table (6-bit, a single tensor; not listed in the index — veloGB10 loads it by name) |
| `vision_tower_bf16.safetensors` | the **original full-precision (bf16) vision tower** (0.9 GB, sha256 `cdd69998…c641e`); **veloGB10 needs it for image input** |
| `vision_k6.safetensors` | the pack's own 6-bit vision tower (0.56 GB), for other loaders; veloGB10 does not read it |
| `quantization_config.json` | full EXL3 per-tensor metadata |
| `config.json`, `model.safetensors.index.json` | the pack's native config and tensor index |
| `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `generation_config.json` | tokenizer and chat template |
| `preprocessor_config.json`, `video_preprocessor_config.json` | vision preprocessors |
| `qbench_prompts.json`, `qbench_prompts.md` | the quality-benchmark prompt set that ships with the pack |
| `LICENSE` | Qwen Community License 1.0 |

## Notes

- **Vision:** images are served at TP=1, TP=2 and TP=4 (resized so the longer side is at most
  `--image-max-edge`, default 1024), through the **full-precision bf16 vision tower** in
  `vision_tower_bf16.safetensors` — keep that file in the model folder; without it veloGB10 runs text-only.
  **Video and audio parts return `400`** for now.
- **Long context is 262,144 tokens** per the config, not the 1M that the hosted Qwen3.8-Flash API offers.
- **Need less memory or more speed?** The [3.05 bpw pack](https://huggingface.co/doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw)
  is smaller and faster; 4.05 trades about 13% of decode speed on one or two nodes for higher fidelity.

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
- **EXL3 quantization:** [turboderp](https://huggingface.co/turboderp) — the `4.05bpw_h6_ng6` pack and
  the accompanying [qbench](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) prompts, built
  with [exllamav3](https://github.com/turboderp-org/exllamav3).
- **Engine:** [veloGB10](https://github.com/sf-stav/veloGB10).
