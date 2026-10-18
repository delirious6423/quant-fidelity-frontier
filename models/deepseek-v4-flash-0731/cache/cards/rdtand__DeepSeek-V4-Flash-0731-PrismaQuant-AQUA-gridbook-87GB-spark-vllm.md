---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
library_name: vllm
tags:
- quantization
- mixed-precision
- codebook
- gridbook
- prismaquant
- nvfp4
- fp8
---

# DeepSeek-V4-Flash-0731 — PrismaQuant AQUA (GridBook CB), 87 GB

A mixed-precision, per-Linear quantization of
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(284B-A38B, 43 layers, 256 routed + 1 shared expert, FP8-native source) that
fits the full model body in **87.08 GB** — sized for a single 128 GB
unified-memory machine (NVIDIA GB10 / DGX Spark) with headroom to serve.

Formats are chosen **per Linear** by the PrismaQuant AURA allocator
(KL-adjoint Fisher cost × production-rendered ΔW, with activation-aware
[AQUA] pricing on both weight and activation sides), then the shipped
operating point is selected and measured on **real end-to-end KL against the
BF16 teacher** — not on a local proxy.

## Measured quality (this exact artifact, served)

Gold contract: WikiText, 8 × 512 tokens (4,088 scored positions), teacher =
BF16 source, teacher distribution truncated to **top-8192 logits per
position**; student served by vanilla vLLM with the pinned GridBook plugin
(the same stack documented below), `score_positions=all`.

| metric | value |
|---|---|
| KL(teacher‖student), mean | **1.2221** |
| KL, confident positions (n=2,935) | **1.0293** |
| KL, p99 | 8.805 |
| KL, max | 18.078 |
| WikiText PPL (direct, same serve) | **20.95** |

Internal ablation at the identical 87.4 GB byte budget: against the
weight-only-cost allocation it replaced, the activation-aware (aqua-both)
allocation shipped here measures **−3.4% mean KL / −7.2% confident-KL /
−11.0% p99-KL** — all outside the ±0.7% cross-session repeat envelope —
while direct PPL is statistically unchanged (+1.0%, inside the ~1.5%
spread observed when re-serving identical bytes across sessions).

## Precision map

Body assignment: **2.346 bits per parameter** over the body's assignment
tensor payload (including deduplicated codebook sidecars). Per-serving-unit
format counts:

| format | units |
|---|---|
| FP8_BLOCK_UE8M0_SOURCE (lossless FP8 passthrough) | 125 |
| FP8_CB K48 / K44 / K36 / K28 (codebook FP8) | 90 / 5 / 34 / 69 |
| NVFP4_CB K12 / K14 / K16 / K18 (codebook NVFP4) | 42 / 18 / 2 / 2 |

Activation contracts as served by the pinned runtime: NVFP4_CB units execute
**W4A4** (E2M1 group-16 activations, static UE4M3 global scale) on the 32
routed-MoE stages; FP8_CB and FP8-source units decode to BF16 and run a BF16
GEMM (weight-only). 189 declared passthrough tensors; no orphan scale planes.

### Layer map

![Per-Linear format allocation across all 43 layers](allocation-map.png)

One column per layer (0–42), one row per projection; every cell is one
serving unit, colored by the format the allocator chose, rendered from this
artifact's own `quant_config.json`. Bright mint = verbatim FP8-source
passthrough (lossless); teal = FP8 codebook rungs, brighter with codebook
size K; navy/violet = NVFP4 codebook rungs; hatched = pinned at source
precision and never offered to the allocator (router gates, the attention
compressor, the indexer's compressor and score head — plus embedding and
`lm_head` off-grid). Routed-expert rows are the packed per-layer expert
stack: all 256 routed experts in a layer share one format, a serving
invariant, not a simplification.

How to read it: the projections feeding attention state (`wq_a`, `wkv`,
`wo_a`) ship as lossless FP8 source everywhere; shared experts stay rich
(K48 or source); and the routed-expert mass — the bulk of the bytes — runs
NVFP4-CB K12 through the first ~21 layers while the allocator concentrates
its 8-bit budget on late-layer experts (FP8-CB K28 on layers 22–38, where
the sensitivity card priced flips as most expensive). On the 11 FP8-CB
expert layers gate/up ship as separate stacks and fuse at load; NVFP4-CB
layers ship fused `gate_up`. K sizes the codebook — the weight coding only;
the activation contract is set by the format family (table above).

The same map is browsable cell-by-cell, alongside every other PrismaQuant
artifact, in the [allocation explorer](https://prismaquant.org/explorer.html).

## What this artifact does and does not claim

- **Claimed:** the KL/PPL numbers above, the byte size, and that vanilla vLLM
  with the pinned GridBook plugin loads and serves it (eager and CUDA-graph
  mode) on Blackwell.
- **Not claimed: matched-budget serving *performance* parity.** This is a
  body-only artifact (see below), and the matched-budget performance-parity
  gate structurally requires construction units and telemetry this artifact
  class cannot produce; it is recorded as scoped-out on the shipcard, not
  passed. Decode on NVFP4_CB units routes to the native codebook LUT path.
  Under the gated runtime (`gridbook==0.8.8`) large-batch prefill on codebook
  units rode a grouped dequant-to-BF16 bridge; **GridBook 0.8.9 defaults the
  fused persistent-B decode-in-mainloop kernels on**, so this artifact's 32
  FP4-CB routed expert layers take that lane with no flags set (its 11
  FP8-CB routed layers ship per-role codebooks and keep the bridge, announced
  per layer; dense CB prefill keeps the bridge). Same activation contract
  either way — a default-environment 0.8.9 serve of this exact artifact
  re-measured kl_mean +0.17 % / PPL −0.06 % against the recorded gold, inside
  the ±0.7 % cross-session KL envelope.
- The KL teacher is top-8192-truncated (mass renormalized); tail mass beyond
  rank 8192 is not scored.
- **The MTP / speculative-decode sidecar is not included.** The source's
  `mtp.` namespace (4,705 tensors) is excluded from this body-only artifact
  and recorded in `quant_config.json` provenance.

## Serving

Tested stack: vanilla **vLLM** (Blackwell / SM121, CUDA 13) with the **GridBook**
out-of-tree quantization plugin, `gridbook==0.8.10`.

```bash
pip install gridbook==0.8.10
vllm serve rdtand/DeepSeek-V4-Flash-0731-PrismaQuant-AQUA-gridbook-87GB-spark-vllm \
  --quantization gridbook --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --generation-config vllm \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 10737418240 \
  --max-model-len 131072 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}' \
  --host 0.0.0.0 --port 8000
```

**Every flag above is load-bearing on a 128 GB single-Spark box. The short form
that omits them does not serve this model well, and in two cases does not serve
it at all.**

- `--max-model-len 131072` — the config default is 1,048,576. Left at the
  default, vLLM sizes the KV pool against the machine's entire RAM (GPU and host
  share one pool on a Spark), reaches ~106 GiB of hard allocation on a 121.6 GiB
  box, and the container is OOM-killed after weights have already loaded
  successfully. It looks like a load failure; it is a sizing failure.
- `--kv-cache-memory-bytes 10737418240` — a fixed 10 GiB pool (~637k tokens).
  Passing it also skips memory profiling entirely, which is what removes the
  autotune/warmup spike that triggers the kill above.
- `--compilation-config … FULL_DECODE_ONLY` — decode is launch-overhead-bound on
  this hardware. Capture is what makes decode throughput reproducible; without
  it decode runs materially slower.
- `--default-chat-template-kwargs '{"reasoning_effort":"low"}'` — vLLM defaults
  DeepSeek-V4 to `reasoning_effort="high"` whenever a request omits it, which
  injects a maximum-deliberation system prompt. The model obeys: a default-path
  "what is 2+2" can run for over twenty minutes without terminating. This is
  stock upstream vLLM behaviour, not a quantization artifact. `"none"` disables
  thinking entirely.
- `--enable-auto-tool-choice --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4`
  — required for tool calls and reasoning segments to parse.

**Known issue on newer vLLM — read before serving without the flags above.**
On vLLM 0.27+, whose default `cudagraph_mode` is `FULL_AND_PIECEWISE`, CUDA-graph
capture fails in GridBook's MoE router (`_padded_route` performs a host sync inside
the capture region) and the engine exits during startup rather than serving. Tracked
as [gridbook#47](https://github.com/RobTand/gridbook/issues/47). The explicit
`--compilation-config` above avoids it; `--enforce-eager` also works but costs
substantial decode throughput, since decode on this hardware is launch-overhead-bound.
If you are benchmarking this artifact, check which of those two you are on before
quoting a tokens/sec number.

No forked runtime, no custom vLLM build: GridBook registers the quant method
through vLLM's plugin interface. Weights are a single `model.safetensors` plus
`cb_codebooks.pqcb` (codebook sidecar) and `quant_config.json`.

## Provenance

- Built by [PrismaQuant](https://github.com/RobTand/prismaquant) — probe →
  AURA/AQUA cost → knapsack allocation → codebook export — at commit
  `a24fce2b`, allocation recipe `aqua-both`.
- Artifact `model_sha` `3d3b8bde1343fe33…` (full value and the complete gate
  record in `shipcard.json` inside the repo).
- Quantized from the FP8-native source; FP8 source tensors selected by the
  allocator ship as verbatim lossless copies.

Author: Robert Tand — robert.tand@icloud.com
