---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
quantized_by: Avifenesh
pipeline_tag: text-generation
tags:
  - nvfp4
  - gguf
  - speculative-decoding
  - mtp
  - conversational
  - blackwell
  - qwen3_5_moe
  - moe
model-index:
  - name: Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF
    results:
      - task:
          type: text-generation
          name: MTP speculative decode K=3, greedy, 256-token serve probes (code + agentic), same-window interleaved A/B
        dataset:
          name: tiyuvta serve probes (deterministic, per-request usage.spec accounting)
          type: tiyuvta-probe
        metrics:
          - name: draft acceptance, embedded continued-trained head (this artifact)
            type: acceptance-rate
            value: 0.4309
          - name: draft acceptance, vendor head (same mint recipe, superseded revision)
            type: acceptance-rate
            value: 0.3523
          - name: draft acceptance, FR-Spec masked head (top-32768 own-gen ranks)
            type: acceptance-rate
            value: 0.393
        source:
          name: tiyuvta serving engine v0.94.0, serving A/B on RTX PRO 6000 Blackwell
          url: https://tiyuvta.ai
---

# Ornith-1.5-35B-A3B — NVFP4 GGUF with continued-trained MTP head (tiyuvta serving artifact)

NVFP4 (4-bit e2m1, per-16 FP8-e4m3 scales) GGUF of
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), quantized
from the official BF16 GGUF release (token embeddings and output head Q5_K, norms F32).
**71.9 GB BF16 → 20.2 GB.** First NVFP4 quantization of this model in either format at
original publication (2026-08-19, ~34.5 h after the model dropped).

**This revision ships a continued-trained MTP head.** The vendor's multi-token-prediction
head is functional at chain depth 1 and collapses deeper — measured next-token top-1 on
held-out own-generation data: depth 1 **0.80**, depth 2 **0.27**, depth 3 **0.13** (the trunk's
RL loop moved; the 1-layer head did not follow). We continued-trained the head on the trunk's
own generations with depth-3 chain-rollout (teacher-forced tokens, self-recursive hidden
carrier — the exact recurrence speculative decoding runs at serve time). Same held-out
measurement after training: **0.81 / 0.58 / 0.43**. Serve-level draft acceptance (K=3, greedy,
same-window interleaved A/B): **0.431 vs 0.352** for the vendor head; code-generation probes
0.540 vs 0.388. Trunk, embeddings, lm_head and every non-head tensor are unchanged official
bytes; a draft head can never change output — the target verifies every drafted token — it
only moves acceptance and speculative speed.

Built as a serving artifact for the tiyuvta serving engine on RTX Blackwell (sm_120a), with
per-request exactness gates: speculative, graphed, and batched serving are gated
byte-identical to plain decode. A single 96 GB Blackwell card fits these weights with room
for the full 262,144-token native context.

NVFP4 is not an upstream llama.cpp tensor type: this file runs on the serving engine and on the
[NVFP4 branch of avifenesh/llama.cpp](https://github.com/avifenesh/llama.cpp/tree/nvfp4-imatrix-scale-search).
For upstream-llama.cpp/Ollama use, the official
[Q4_K_M–Q8_0 GGUFs](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) are the right pick.

- Context: 262,144 native; chat template embedded (XML tool calling + `<think>` reasoning)
- Vocabulary: 248,320 (text→text serving; the upstream vision tower is not in this artifact)

## Provenance and verification

| | |
|---|---|
| Base | `ornith-ai/Ornith-1.5-35B-A3B` (MIT), official BF16 GGUF (`qwen35moe`, 41 blocks incl. NextN) |
| Head training | `mtp.*` only, continued-trained on the model's own generations (4,044 prompts, ~2.1M sampled tokens at the vendor serving temperature), depth-3 chain-rollout, frozen trunk; recipe + receipts banked in the lab's research record (`ornith15-mtp-train-20260820`) |
| Quantization | `llama-quantize` NVFP4 ftype (imatrix-aware branch), `--output-tensor-type q5_k --token-embedding-type q5_k`; head quantized after training (NVFP4 head measured at zero acceptance cost on this pipeline) |
| Main file | `Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf`, 20,188,038,400 B, sha256 `72ff9600aa2b0de77a5b27041a84448c2ce88c7b2055529fc23b3cd5bf518fd3` |
| Exactness | serving engine batteries, 2026-08-20: `run-spec` K=1..8 self-consistency PASS (spec ≡ plain greedy, token-identical) · chat-templated generation probes coherent · serve-level A/B accounted per-request (`usage.spec`) |

## Masked MTP draft (FR-Spec)

`mtp-Ornith-1.5-35B-A3B-NVFP4-frspec-owngen32768.gguf` (0.9 GB, sha256 `46f0dd4c…73fe899`) —
standalone speculative-decode draft from the trained MTP block: lm head masked to the
top 32,768 of 248,320 rows, ranked by the model's **own generations** (external text used as
prompts only), then requantized in the mask-first order — NVFP4 head, Q4_K_M block. Ranks ship
as `ornith15-ranks-owngen-32768.txt` (drives the serving engine's load-time head mask on
safetensors trunks) and as `.gguf`. A mask can never change output; it moves draft
acceptance (measured 0.393 vs 0.431 embedded at K=3) against a smaller head read.

## Self-trimmed draft head (recommended serving form)

`head-Ornith-1.5-35B-A3B-frspec-owntrim-q5k-32768.gguf` (46 MB, sha256
`d7c47026e232c0d19240e6ecb3ec9702751fc68dec82909bec79b909846b3b4d`) — the top-32,768
own-generation-ranked rows of this artifact's own Q5_K `output.weight`, byte-gathered
(zero requant) plus the `d2t` map. The serving engine consumes it (or the ranks file
directly) at load to trim the DRAFT lm_head from 248,320 to 32,768 rows (~221 → ~29 µs per
draft step). Verify stays full-vocab, so output is unchanged by construction; measured
serve effect on one RTX PRO 6000 Blackwell (ABBA, N=6/shape, 2026-08-22): single-stream
short prompts **+10%**, shared-prefix c8 **+2.9%**.

## Serving posture (serving engine ≥ v0.105.0)

The early spec-off posture is superseded. With the continued-trained head + the self-trim,
the serving engine serves Ornith-1.5 **spec-on by default**, and since v0.105.0 the automatic depth
table serves cached long prompts at **K=5** when a rank-trimmed head is loaded — the
strongest measured cell (single-stream cached-long **326–354 tok/s**; ~273 at the old K=2,
same card). Other measured serve cells on one RTX PRO 6000 Blackwell WS (vendor sampling
T=0.6/top-p 0.95/top-k 20): 8-turn agentic session 11.6–12.9 s total; shared-prefix c8
~915 tok/s aggregate; c16 ~700 tok/s aggregate; warm-turn TTFT 25–40 ms.

**Pair the card with a current-generation host CPU.** This model's architecture forces the
*sampled* speculative path, and that path is host-issue-bound rather than GPU-bound: the
phase profile of one round splits verify-issue 50–58%, host-commit 24–26%, draft 19–25%
and **verify-wait 0.0%** — the GPU is never what the host waits for. Measured consequence,
same GPU / same binary / same artifact: a Zen 3 host (~3.4 GHz) serves **195–230 tok/s**
while a Zen 5 host serves **326–354**. A cold host also reads low until its clocks ramp.
Receipts: lab research record `orndecode-20260822`.

Served on the tiyuvta serving engine, tuned for this model on RTX PRO 6000 Blackwell, with
the ranks file (`ornith15-ranks-owngen-32768.gguf`) loaded as the self-trim.

The standalone masked draft (`mtp-…-frspec-owngen32768.gguf`) remains published for
external-attach engines; on the serving engine the self-trim of the embedded head is the
faster route (the external attach forgoes the resident graph-draft path). Built with
`tools/make-trimmed-draft.sh` + `frspec-owngen`.
