---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
quantized_by: Avifenesh
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - modelopt
  - qwen4_exp
  - mtp
  - speculative-decoding
  - conversational
  - blackwell
---

# Qwen3.8-Flash-Next NVFP4, with the MTP head quantized too

> **Run Qwen3.8 today through an API.**
> [tiyuvta](https://inference.tiyuvta.ai/?c=hf-q48fn) serves Qwen3.8-27B now, and this
> Flash-Next artifact is in bring-up for the same endpoint.
> [Get an API key and send the first request.](https://inference.tiyuvta.ai/quickstart?c=hf-q48fn)

NVFP4 (4-bit e2m1 weights and activations, per-16 FP8-e4m3 scales, group 16) of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at revision
`de4b8e4d`, quantized from the BF16 release with NVIDIA TensorRT Model Optimizer
(modelopt 0.46.0). Calibration: 256 samples of 512 tokens, abisee/cnn_dailymail 3.0.0,
seed 17.

What is and is not quantized, and how this differs from other NVFP4 mints of this model:

| surface | precision |
|---|---|
| 48 layers of 512 routed experts (about 120B params) | **NVFP4** |
| the MTP block (4B, full decoder layer + fusion projections) | **BF16, grafted byte-exact from the base release.** transformers carries no MTP module, so a transformers-based PTQ pass never sees `mtp.*`; this artifact grafts the full block back so MTP speculative decoding works on engines that read it. An honestly calibrated NVFP4 MTP needs real activations through the block and lands with the bring-up on the tiyuvta serving engine |
| QSA and GDN attention, router gates, shared experts, gated-residual mixers, the 51B n-gram table, vision tower, embeddings, lm_head | BF16 |

## Mint gate (measured against the BF16 parent, same box, same prompts)

Fixed-probe last-token logits before and after quantization:

| probe | argmax same | top-20 overlap | KL |
|---|---|---|---|
| merge two sorted linked lists | yes | 18/20 | 0.0272 |
| TCP vs UDP in two sentences | yes | 16/20 | 0.0035 |
| `def fib(n):` continuation | yes | 19/20 | 0.0124 |
| capital of Australia | no | 18/20 | 0.0355 |

Chat template is byte-identical to the base release (checked on the exact publish bytes).

## Architecture notes (why this checkpoint is unusual)

125B total, 6B active, plus a 51B n-gram embedding table and a 4B MTP block.
Three of every four layers are Gated DeltaNet; the fourth runs Qwen Sparse Attention
(micro-block indexer, 2048-token budget), so decode reads a bounded KV set at any
context length. The residual stream is four branches wide with learned read and write
gates. The n-gram table is gather-only and host-memory friendly. Native context 262,144,
extensible to 1M with YaRN.

Built as a serving artifact for tiyuvta's serving engine on RTX Blackwell (sm_120a), with
exactness gates. qwen4_exp bring-up state in the serving engine: loader, reference oracle and
the GPU eager path have landed (census-gated tensor contract over all 1,658 tensors; reference
executor within 2e-5 of transformers on the cross-oracle gate; spec-vs-plain byte-identity
gates; the full 262,144 native window runs on one 96 GB card with the MTP draft on a second).

## Serving readiness: NOT ready for multi-session use

This checkpoint is a **single-request bring-up artifact**. The serving engine's qwen4_exp path
is single-sequence (its multi-row programs are one request's speculative-verify columns, not a
batch of requests), it is **not wired into the serving engine's server**, and no multi-session
battery has run. Do not put it behind a shared endpoint yet: concurrent sessions would
serialize, and no throughput, latency or context claim on this card applies to more than one
request at a time. This section will be revised when batched decode, server wiring and the
qualification gates land.

- Vendor sampling defaults honored (thinking: temperature 1.0, top_p 0.95, top_k 20;
  instruct: temperature 0.7, top_p 0.80, presence 1.5); reasoning_effort xhigh, medium, low
- Chat template embedded (tools, thinking controls); eos `[248046, 248044]`
- `PROVENANCE.sha256` in this repo chains every file to the pinned base revision

## License

Qwen Community License 1.0, copied verbatim in [LICENSE](LICENSE) with the copyright
notice retained, as clause 1 requires. Read clause 2 for Model-as-a-Service terms.
