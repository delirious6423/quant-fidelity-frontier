---
license: mit
library_name: mlx
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
inference: false
tags:
  - deepseek
  - deepseek-v4.1
  - deepseek-v4.1-flash
  - mixture-of-experts
  - engram
  - dspark
  - speculative-decoding
  - vision
  - reasoning
  - tool-use
  - mlx
  - apple-silicon
  - quantized
  - jang
  - jang-k
  - affine
  - imatrix
quantization_config:
  family: jang
  weight_format: affine
  profile: JANG_2K
  group_size: 64
  routed_avg_bits: 2.47
  routed_down_bits: 3
  routed_gate_up_bits: "3 (layers 0-7) / 2 (layers 8-39)"
  attention_bits: 8
  calibration: imatrix-clip-search
---

<p align="center"><img src="jangq-logo.png" width="300" alt="JANG"/></p>

<p align="center">
  <a href="https://vmlx.net"><img src="vmlx-app.png" alt="vMLX — run JANG models on Apple Silicon" width="820"></a>
</p>

<h3 align="center">⚡ All JANG models are meant to be run in <a href="https://vmlx.net">vMLX</a></h3>

# DeepSeek-V4.1-Flash-1017-JANG_2K

Mixed-affine JANG quantization of
[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
for Apple Silicon, with the release's Engram n-gram tables, its 3-stage **dSpark** speculative
draft and its **vision tower** all included. Built and measured on a 256 GB M5 Ultra.

| | |
|---|---|
| Source | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| Source revision | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| License | MIT, inherited from upstream |
| Architecture | 40 layers, 384 routed experts (top-6) + shared expert, hyper-connections, compressed sparse attention with a shared indexer, Engram n-gram embeddings (layers 1, 14) |
| Format | JANG_2K mixed affine (group 64), imatrix-calibrated routed experts |
| Backbone | 45 weight shards, 2,214 tensors, **~197 GB resident** |
| Engram tables | `engram/`, the two unmodified FP8 release shards (~203 GB); read from SSD by row, never loaded into RAM |
| dSpark draft | `dspark/`, 3 stages, routed experts 4-bit (8.9 GB) |
| Vision | in-bundle ViT + aligner (image input) |
| Repository size | 423.6 GB / 394.5 GiB |
| Context | 1,048,576 tokens (YaRN) |
| Needs | a 256 GB Mac (serving footprint ~207 GB) |

## Quantization (JANG_2K)

`JANG_2K` spends one extra bit on the routed `down_proj` (its output enters the residual stream) and
keeps every non-routed path at 8-bit. The first eight layers also get 3-bit `gate`/`up`.

| Tensor role | Bits | Group |
|---|---:|---:|
| Routed expert `down_proj`, all 40 layers | 3 | 64 |
| Routed expert `gate_proj` / `up_proj`, layers 0–7 | 3 | 64 |
| Routed expert `gate_proj` / `up_proj`, layers 8–39 | 2 | 64 |
| Attention, shared expert, indexer `wq_b`, Engram `wkv` | 8 | 64 |
| Token embedding, LM head | 8 | 64 |
| Norms, router, hyper-connections, compressor | source dtype (F32 kept) | — |
| Engram tables | source FP8 E4M3 + E8M0, unmodified | — |

Routed-expert average: **2.47 bits**. Routed experts were requantized from the FP4 source with an
importance-matrix clip search (activation-x²-weighted, 11 clip ratios per group, plain affine always
among the candidates) using statistics from 694,517 tokens of technical and security-domain text.
MTP weights of the main model are replaced by the bundled dSpark draft.

## Quality

Measured on this exact bundle. Teacher-forced perplexity on 32 held-out rows (6,766 tokens), excluded
from all calibration statistics:

| build | perplexity | NLL / token |
|---|---:|---:|
| plain affine, no calibration | 9.234 | 2.2229 |
| imatrix, held-out domain excluded from calibration | 9.356 | 2.2360 |
| **this bundle** (imatrix, all calibration data) | **9.318** | **2.2319** |

The three builds are within 1.3% of each other, which is within the noise of 6.8k tokens. Top-1 agreement
and KL against the original FP4/FP8 checkpoint are not reported: the source model does not run on Apple
Silicon, so there is no reference on the same hardware yet.

Runtime fidelity checks (real model):
- vMLX speed paths are exact: prompt-suffix pruning gives last-token logits cos 0.99996 / 0.99994 at
  8k / 32k with the same top-5 and 24 of 24 greedy tokens; the fused decode kernels give 256 of 256
  identical greedy tokens; an identical answer with pruning on and off at 924k tokens.
- dSpark output equals greedy decoding up to near-ties (it runs one verify forward per step).
- Long-context retrieval, greedy, thinking off, question at ~54 % depth:
  - one distinctive fact in ~460k / ~845k tokens of natural text: correct both times (top-1 probability 0.99);
  - 46,200 / 92,600 numbered lines (`Doc line N: value V`, random 3-digit values; 461k / 925k tokens): correct
    (0.90 / 0.99);
  - a deliberately hard variant (IDs like `r1m-15617` plus a second number on every line, space-joined) is correct
    up to 300k tokens and misses at 461k / 924k. The miss is the same with cold, RAM and SSD caches, with 512- or
    2048-token prefill chunks, with bf16 cache pools and with the release's own FP8/FP4 activation rounding; and
    putting the release's original FP4 routed experts back in layers 0-19 does not bring the target line into the
    sparse-attention candidate set (it ranks lower, not higher). We read it as a limit of the model's sparse attention
    on near-identical lines, not of this quantization or runtime.

## Sampling defaults

`generation_config.json` sets DeepSeek's recommended sampling (temperature 1.0, top_p 1.0). vMLX (Python and
Swift / Osaurus) applies it when a request sends no sampling parameters; dSpark then runs its exact sampled path.
Send `temperature: 0` for greedy output.

## Speed and memory (M5 Ultra 256 GB, thinking off, unique-text cold prompts)

| | vMLX Python (`vmlx serve`) | Osaurus app (vMLX Swift) |
|---|---|---|
| decode with dSpark (default), code / code2 / structured / prose | 84 / 68 / 119 / 44 tok/s | 68 / 55 / 104 / 42 tok/s |
| plain decode (dSpark off), 2k context | 39–44 tok/s (34 at 100k) | 43–45 tok/s |
| time to first token, short prompt | 0.21–0.59 s | 0.22–0.78 s |
| cold prefill, 1.6k / 6.5k tokens | 845 / 1,325 tok/s | 883 / 1,424 tok/s (dSpark off) |
| cold prefill, 26.5k / 42k / 100k tokens | 1,572 / — / 1,584 tok/s | 1,584 / 1,371 / 1,117 tok/s |
| prefix cache hit, 3.4k-token prompt | 0.36 s (RAM), 0.34 s (SSD, after restart) | 0.29 s (SSD), 0.34 s (SSD, after restart) |
| prefix cache hit, 924k-token prompt (cold 1352 s) | 5.8 s (RAM), 6.5 s (SSD after restart, 849 MB on disk) | — |
| agent tool loop: wait after the last streamed token / next-turn first token | 0.2–0.3 ms / 0.38–0.51 s | 0.0–0.2 ms / 0.29–0.37 s |
| KV cache at 1M tokens | 905 MiB (905 B/token) | 903 MiB at 1.04M (911 B/token) |
| process memory (Activity Monitor) | 207 GB | 206 GB (198 GB with dSpark off) |

Prefill above 2.4k tokens is fast because layers 21–39 only compute the prompt rows the last token can
still reach through their 128-token windows (exact). Decode is fast because the per-layer work runs in a
few fused Metal dispatches. The compressed KV pools are stored 4-bit, as in the reference's FP4 cache.

## Runtime requirement

This bundle needs a DeepSeek-V4.1-aware runtime: Engram tables read from `engram/`, the compressed
sparse attention with shared indexer, hyper-connections, the dSpark draft and the vision tower. It does
not load in stock `mlx-lm`.

- **vMLX (Python):** the `jang_tools.dsv41` runtime (`jjang-ai/jangq`) with vMLX DeepSeek-V4.1 serving
  support: `vmlx serve <bundle>`. dSpark is on by default for greedy requests (`VMLX_DSV41_DSPARK=0`
  turns it off).
- **vMLX Swift / Osaurus:** DeepSeek-V4.1 support in `osaurus-ai/vmlx-swift` (PR #657) and Osaurus
  (PR #3206).

Keep the bundle on a fast internal SSD: every prompt token reads 48 random Engram rows.

## Chat contract

The official release ships `encoding/encoding.py` (no Jinja template); vMLX renders with it. V4.1
reasoning effort is numeric (`Reasoning Effort: N`); `chat_template_kwargs.enable_thinking=false`
gives the non-reasoning mode. Tool calls use V4.1 DSML and are returned as OpenAI `tool_calls`.
Images go in standard OpenAI `image_url` content parts.

## Download

```bash
hf download JANGQ-AI/DeepSeek-V4.1-Flash-1017-JANG_2K \
  --local-dir ~/models/DeepSeek-V4.1-Flash-1017-JANG_2K
```

## Korean summary

공식 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 릴리스를 Apple Silicon용 JANG_2K 혼합 affine(라우티드
전문가 평균 2.47비트, down 3비트, 비라우티드 8비트, imatrix 보정)으로 양자화한 번들입니다. Engram
n-gram 테이블(SSD에서 행 단위로 읽음), 3단계 dSpark 추측 디코딩 드래프트, 비전 타워가 모두
포함되어 있습니다. 256 GB M5 Ultra 기준 vMLX Python에서 dSpark 코드 디코드 약 90 tok/s, 100k 토큰
콜드 프리필 약 1,584 tok/s, 1M 토큰 KV 캐시 약 905 MiB, 메모리 약 207 GB를 측정했습니다. 원본
체크포인트 대비 top-1/KL은 동일 하드웨어 기준 참조가 없어 아직 보고하지 않습니다.

## Contact

eric@jangq.ai
