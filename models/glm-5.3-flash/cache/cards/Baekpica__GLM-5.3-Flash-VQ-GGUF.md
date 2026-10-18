---
license: mit
base_model: zai-org/GLM-5.3-Flash
tags:
  - gguf
  - vector-quantization
  - glm5-next
  - q4_k
  - dgx-spark
  - experimental
pipeline_tag: image-text-to-text
---
# GLM-5.3-Flash VQ GGUF (Q4_K nonexpert v3)

Experimental custom GVQ1 / GGUF v3 conversion of the official
[Z.AI GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/91b69215804f2d2036a8867f6ce44848e278dcda)
(`91b69215804f2d2036a8867f6ce44848e278dcda`), tuned for decode speed on
DGX Spark (GB10). A reader and runtime implementing [FORMAT.md](FORMAT.md)
and the `q4_k_nonexpert_v3` recipe are required:
[ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs) `main` from `c5384b5e`
([#86](https://github.com/Baekpica/ds4-dfm-rs/pull/86)) on; the measurements
below use `3dd964ba` ([#88](https://github.com/Baekpica/ds4-dfm-rs/pull/88)).
Standard llama.cpp/Ollama/Transformers loading is not claimed.

The previous all-native-precision release is the tag `vq-original`
(revision `fe2a446e774624ef8d397136d81524c61db77f65`); its card documents the
VQ conversion and its measured quality loss in full.

## What this revision changes

- **Routed experts are unchanged:** 36,288 matrices of layers 3–44 use
  eight-dimensional VQ, K8192, 12+1-bit indices, one E4M3 codebook per layer
  and F16 row gains (1.625 index bits per weight).
- **316 nonexpert matrices are Q4_K** (256-element / 144-byte blocks, no
  importance matrix), each derived from the original weights (FP8 times its
  block scale rounded to BF16, or BF16): KDA q/k/v/o (136), DSA q_a/q_b/kv_a/o
  (44), dense FFN of layers 0–2 (9), shared experts of layers 3–44 (126) and
  the output head (1). [q4k-selection.json](q4k-selection.json) lists them;
  the manifest keeps every parent payload hash.
- **Kept in source precision:** embedding, DSA kv_b (BF16 latent attention),
  router and its bias, mHC maps, KDA gates, indexer, norms, vision and the
  MTP layer 45.

Decode reads about 7.5 GB of weights per token instead of 9.6 GB.

## Measured on DGX Spark (GB10)

![GLM-5.3 Flash VQ 2K–64K prefill and decode on one DGX Spark](assets/glm53-vq-ds4-dfm-rs-2k-64k-throughput.png)

*Measured on this artifact with ds4-dfm-rs `main` (`3dd964ba`). One warm
session per process, 2,048-token incremental prefill and 128 forced greedy
tokens per frontier from 2K to 64K. Curves are per-frontier medians; bands
are the observed min–max of three fresh runs with two embedded MTP drafts.*

| Workload | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| 2K–64K sweep, two MTP drafts (median of three run means) | 1,163.5 | 35.08 |
| 3K-token code prompt, two MTP drafts (one run) | — | 39.98 |
| 512 tokens at 2K context, MTP off (two runs) | — | 25.9 |
| Whole 8K prompt (7,936 tokens), MTP off (two runs) | 1,343.8 | — |

The sweep and the 8K row ran the `3dd964ba` code. The decode rows ran the
`c5384b5e` kernels; #88 changed prefill kernels only, and the sweep's decode
matches (35.08 against 35.12).

- Incremental prefill falls 7.1% from 4K (1,238.0) to 64K (1,149.9). The 2K
  point is each process's first prefill. `c5384b5e` measured 1,042.2 / 35.12
  on the same protocol.
- MTP gains follow draft acceptance on the generated text: the sweep accepted
  2.32 tokens per verify and a two-draft verify cycle averages 66.2 ms.
- CUDA 13.4, busy SM clock 2,184–2,190 MHz under a 300–2,200 MHz policy.
- Serve with `--cuda --mtp-draft 2 --temp 0`; `ds4` and `ds4-server` default
  to the model's native 1,048,576-token context (throughput measured through
  64K).

## Quality

A six-case screen (Korean, English, code, math, JSON, stopping; 106
teacher-forced targets, greedy answers up to 96 tokens) against the KDA-only
artifact on the same build: paired NLL +0.0033 nat/token. Five task answers
stay correct. **The Korean meeting-summary case degenerates under greedy
decoding** (its first-token distribution is flat, top-5 at 5–15%), as did the
earlier Q4_K screens. The output head is not the cause: keeping it BF16 gives
the same failure. Q4_K was chosen for speed with this loss accepted; this is
not a quality-equivalent replacement of the source model.

The routed-expert VQ loss of the original release still applies (its card
reports held-out prose PPL 4.8555 → 8.0797).
