---
license: mit
base_model:
  - zai-org/GLM-5.3-Flash
language:
  - en
  - zh
tags:
  - GGUF
  - llama.cpp
  - glm
  - glm-5.3
  - quantization
  - turboquant
  - tq3_4s
  - mtp
  - Mixture of Experts
  - imatrix
  - conversational
pipeline_tag: text-generation
---

![thumbnail](thumbnail.png)

# GLM-5.3-Flash-TQ3_4S

GGUF quantization of [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) using **TQ3_4S** — a TurboQuant 4-bit format (~4.0 BPW nominal) with per-tensor precision overrides — 2-bit routed experts, 6-bit attention & shared experts, with the MTP (NextN) head retained for native speculative decoding.

## Files

Download **all 4 shards into one directory** and point `llama-server` at shard 1.

| File | Description |
|------|-------------|
| `tq3_4s/GLM-5.3-Flash-TQ3_4S.gguf-00001-of-00004.gguf` | Shard 1 (27.6 GiB) |
| `tq3_4s/GLM-5.3-Flash-TQ3_4S.gguf-00002-of-00004.gguf` | Shard 2 (27.8 GiB) |
| `tq3_4s/GLM-5.3-Flash-TQ3_4S.gguf-00003-of-00004.gguf` | Shard 3 (27.8 GiB) |
| `tq3_4s/GLM-5.3-Flash-TQ3_4S.gguf-00004-of-00004.gguf` | Shard 4 (18.5 GiB) |
| **Total** | **101.6 GiB · ~4.0 BPW** |

> **Text-only artifact.** The upstream base is image-text-to-text, but this quant ships **no vision tower** — there is no `mmproj` to load.

## Quantization

MoE experts tolerate aggressive compression because only 8 of 288 are active per token. This quantization exploits that asymmetry:

| Component | Quant | Rationale |
|-----------|-------|-----------|
| Routed expert MLP | `iq2_s` / `q2_k` | 2-bit, MoE-tolerant (8/288 active) |
| Shared experts + attention | `q6_K` | quality anchor |
| Norms / biases | `f32` | precision-sensitive |
| Embeddings + output | `q8_0` | quality anchor |
| `attn_output` | `q8_0` | keep the residual path clean |
| `blk.45` experts | `q2_k` | no imatrix coverage (documented workaround) |
| MTP / NextN head (`blk.45.nextn.*`) | `q6_K` | keeps the draft accurate for speculative decode |

Requantized from [`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) `Q8_0` shards with the Unsloth imatrix (covers `blk.3–44`).

## Runtime Requirement

This model requires the public TurboQuant runtime fork (mainline llama.cpp does not support the `TQ3_4S` tensor type):

* https://github.com/turbo-tan/llama.cpp-tq3   (build ≥ `10403`, CUDA `sm_120`/`sm_121` or CPU)

## Recommended Settings (single GB10 / DGX Spark, 121 GB unified)

```bash
LLAMA_MTP_PROCESS_ONLY=1 llama-server \
  -m tq3_4s/GLM-5.3-Flash-TQ3_4S.gguf-00001-of-00004.gguf \
  --spec-type draft-mtp --spec-draft-n-max 1 \
  -np 1 -c 40960 -ub 8 -b 8 -fa on -lm none -ngl 47 \
  --jinja
```

* `--jinja` is **required** (custom chat template).
* `LLAMA_MTP_PROCESS_ONLY=1` keeps the draft KV correct under deep context (without it, long-context MTP can hang).
* **Context:** native spec is 1M; on one GB10 with `-np 1` we validated serving up to **512K**. `-c 40960` is the benchmarked shipping config.
* The `blk.45.nextn.*` MTP head enables native speculative decoding — no separate draft model needed.

## Performance (NVIDIA GB10 / DGX Spark, 121 GB, full clock)

| Metric | Value |
|--------|------:|
| Decode (prose, temp 0.7, n=256, median of 3) | 19.5 tok/s |
| MTP acceptance (dn=1) | 100% deterministic · 87–90% prose |
| Size | 101.6 GiB |
| BPW | ~4.0 (nominal) |
| ngl | 47 (unified-memory box) |
| Context (shipping) | 40960 (validated to 512K) |

Fits and serves on a single GB10 (121 GB unified) — no multi-GPU split needed.

## Quality

Measured on the hardware above, greedy / temp 0.7 as noted.

| Suite | Score |
|-------|------:|
| Hard86 (unit-test-pass, 86 coding tasks) | 77.9% (67/86) |
| Coding (internal quality suite) | 81.2 (9/12) |
| Toolcall | 90.0 (13/15) |
| Data extraction | 90.8 (12/15) |
| Instruction following | 90.0 (13/15) |
| Reasoning / math | 73.3 (11/15) |
| **Quality composite** | **85.1** |

## Base Model

* [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT)
* Source: [`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) `Q8_0` (credit: Unsloth for the GGUF conversion and calibration/imatrix data)

## License

MIT — inherited from [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash). Copyright (c) 2026 Z.AI Co., Ltd.

## Tool Call Validation

Tested with `--jinja` under both `--reasoning off` and `--reasoning on --reasoning-budget 2048`. Tool-call aggregate **13/15 (90.0)**: correct tool selection from multiple candidates, no spurious calls on plain questions, and tool-response → final-answer without runaway loops.

Verify against your own server:

```bash
# Start the server (see Recommended Settings), then:
curl -s http://127.0.0.1:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-TQ3_4S","messages":[{"role":"user","content":"What is 27*43? Use the calculator tool."}],
       "tools":[{"type":"function","function":{"name":"calculator","description":"Eval math",
       "parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}]}'
```
