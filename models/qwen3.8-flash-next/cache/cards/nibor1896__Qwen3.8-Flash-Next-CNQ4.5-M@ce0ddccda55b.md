---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
- en
- de
tags:
- nvfp4
- cnq
- quantized
- moe
- vision
- multimodal
- windows
- linux
- rtx-5090
- crow-nest
---

# Qwen3.8-Flash-Next-CNQ4.5-M

> [!Note]
> This repository holds a quantization of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) in the CNQ container format of the [crow-nest](https://github.com/nibor1896/crow-nest) engine. It is not the upstream checkpoint: it does not load in Transformers, vLLM, SGLang, TokenSpeed or llama.cpp, only in crow-nest.
>
> The sections "Highlights", "Model Overview", the upstream half of "Benchmark Results", the sampling rows under "Best Practices" and the BibTeX under "Citation" are quoted verbatim from the upstream card at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` and describe the ORIGINAL BF16 model. Everything else describes this quant and says where its numbers come from.

## About this quant

| item | value |
|---|---|
| base model | `Qwen/Qwen3.8-Flash-Next`, revision of record `de4b8e4d43b917e7706784d8bb445c9af86a3540` (upstream `lastModified` 2026-08-27T05:03:36Z, still `main` on 2026-09-27) |
| quantization | CNQ4.5-M: NVFP4 at 4.5 bpw, round to nearest, calibration free; BF16 keeps for embeddings, `lm_head`, router, shared expert gate, all norms and every 1-D tensor; sub-block scales by `--scales mse` |
| container | one file, `Qwen3.8-Flash-Next-CNQ4.5-M.cnq`, 104,727,179,972 B (about 105 GB), 1,658 tensors, index v1 |
| engine | crow-nest, `https://github.com/nibor1896/crow-nest`: own HTTP server with an OpenAI-style `/v1/chat/completions`, own container format, no GGUF, no transformers |
| vision | the container carries the FULL vision tower, same quant policy as the text tower; since crow-nest v0.5.0 (2026-09-24) `serve` prefers llama.cpp's F16 projector when it finds one (see Vision) |
| platform | Linux (every current number, crow-nest v0.3.0 to v0.7.0) and Windows (crow-nest v0.3.0, 2026-09-17; no Windows run is recorded after v0.3.1), CUDA, NVIDIA Blackwell (`sm_120`); every measured number on this card comes from one RTX 5090 |
| intended use | local inference with crow-nest on one GPU with a pinned host-memory expert tier, as the engine behind the Crow agent; no hosted endpoint is offered |
| licence | model weights: Qwen Community License 1.0 (see License); engine and converter code: Apache-2.0 |

### Files in this repository

| file | size | `sha256` | purpose |
|---|---|---|---|
| `Qwen3.8-Flash-Next-CNQ4.5-M.cnq` | 104,727,179,972 B | `7c058e555667b1c3d8f7d804d4a3393ba3664161dd307305718d85e4f33646b7` | the container |
| `Qwen3.8-Flash-Next-CNQ4.5-M.cnq.sidecar.jsonl` | 436,716 B | `af85905fbf46c12fa9528d22f2932b8785c55731a8c8fb013b2e91eecc6f0692` | verification sidecar: one JSON line per tensor with max and mean error against the sub-block scales, gate 0 of the measurement ladder |
| `hotsets-M-longctx2100-n160.json` | 83,705 B | `4a408907d553518ee4421e59eae09e243db82ac0cd677557596dfdbaa4b099dc` | hot-set manifest of this package, of the self-test and of the engine's gates of record; not a sidecar derived from the container name (engine issue #49). It is NOT the engine default any more: since 2026-09-24 that is `decode_out/hotsets-M-crow0924-n160.json` in the crow-nest repository, carried here as well since 2026-10-07 (next row) |
| `hotsets-M-crow0924-n160.json` | 37,167 B | `f83e210a21391521b145d53d41fb0308f99cf6543bada03860127544a1b6fff5` | the engine's default hot-set manifest since 2026-09-24 (engine issue #106), the same bytes as `decode_out/hotsets-M-crow0924-n160.json` in the crow-nest repository at `f4a3bd8`; CrowSetup fetches it from here (Crow issue #196, 2026-10-07) |
| `selftest/layer0-input.f32` | 327,680 B | `65907fe567547704dd644eea9212c4c71b0b02f2fd954f23ecd28c7f5804d974` | self-test input: the `[8][10240]` f32 layer-0 activation the golden below belongs to (see Self-test) |
| `selftest/layer0-golden-output.f32` | 327,680 B | `e98292a0413d7aaf6825ab23e80cfae52f2424b5524f835624e9d76e90d7c705` | self-test golden: the `[8][10240]` f32 output of the UNQUANTIZED layer 0 on that input |
| `selftest/layer3-attn-input.f32` | 81,920 B | `d2662da06b578f7c2d8a466ea5a1d74b8b13afedd271b25c16bf0237c621cbeb` | self-test input: the `[8][2560]` f32 activation the layer-`3` attention mixer sees (engine issue #69, 2026-09-18) |
| `selftest/layer3-attn-output.f32` | 81,920 B | `76c64e1747466719dd852e6178f03910b4b45f070b9b5d29bf834af40b2fbd69` | self-test golden: the `[8][2560]` f32 output of the UNQUANTIZED layer-`3` attention mixer on that input (engine issue #69, 2026-09-18) |
| `selftest/manifest.json` | 6,224 B | `7e9569e6dd3e8f12bc70593386588ffa09081f9f5b54ba61ef824c00c190bee8` | the self-test's check list, its two gates, the four sums above and the goldens' full provenance |
| `SHA256SUMS` | 879 B | n/a, it carries the sums | the nine sums above; the first three were each read twice when the package was formed (F1, engine issue #57), the layer-`0` self-test lines were added with the self-test (F5, engine issue #64) and the layer-`3` lines with the second check (engine issue #69, 2026-09-18), the `crow0924` hot-set line with that file (Crow issue #196, 2026-10-07) |
| `LICENSE` | 3,235 B | `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08` | Qwen Community License 1.0, verbatim from the upstream revision |

- Verify the download with the first command below, in the package directory; it must report 9 of 9 OK and covers the container, the sidecar, both hot-set manifests and the five self-test files (2026-10-07). The second command checks the self-test files alone, which costs no read of the 105 GB container.

```
sha256sum -c SHA256SUMS
grep ' selftest/' SHA256SUMS | sha256sum -c -
```

### Files from other repositories, needed at run time

| file | from | size | `sha256` | needed for |
|---|---|---|---|---|
| `tokenizer.json` | `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d` | 12,809,320 B | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` | `serve` loads it from `models/Qwen3.8-Flash-Next-original/`; without it `serve` exits 3 at boot |
| `tokenizer_config.json` | same | 17,928 B | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` | the chat template (`chat_template` field) |
| `config.json`, `generation_config.json` | same | 4,745 B, 202 B | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b`, `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` | optional: they arm the metadata gate for this index-v1 container (21 constants checked at boot, engine issue #94); without them the boot logs one WARN and continues |
| `mmproj-F16.gguf` | `unsloth/Qwen3.8-Flash-Next-GGUF` @ `38bb39ee` | 904,004,000 B | `1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980` | optional: the F16 vision projector `serve` prefers over the container's NVFP4 tower (see Vision) |

- The sha256 values were computed on 2026-09-27 from the files crow-nest was measured with and equal the Hub's records that day (tokenizer LFS record and the two small files fetched again at `de4b8e4d`; the projector's LFS record on `unsloth/Qwen3.8-Flash-Next-GGUF` `main` = `38bb39ee`, `lastModified` 2026-09-02).
- The crow-nest README's build line downloads only the container; the tokenizer files above are not in the git clone (`models/` is gitignored), so they have to be fetched once.

### Format, in short

- The container is the CNQ v1 format written by the `converter` in the crow-nest repository: magic `CNQ1`, 12-byte header, a streamed payload, the JSON index as a TRAILER with a `u64` little-endian length in the last 8 bytes; NVFP4 blocks of 64 values with four `ue4m3` sub-block scales and one `f32` global scale per tensor, 36 B per block, 4.5 bpw (`converter/src/main.rs:1-59`, spec section 1 approved 2026-09-02). The full tables are in [`converter/README.md`](https://github.com/nibor1896/crow-nest/blob/main/converter/README.md) of the engine repo.
- BF16 keep set (spec 1.2, 2026-09-02): embeddings and `lm_head`, the router GEMM, `shared_expert_gate`, all norms, every 1-D tensor (biases, `A_log`, `dt_bias`, gates), and any tensor whose length is not a multiple of 64. A latent finding recorded 2026-09-26: `linear_attn.in_proj_a/b` and `linear_attn.conv1d` (108 tensors) fall through to NVFP4 in this container; the container stays unchanged (engine CHANGELOG v0.7.0).
- Section marks: `text` (always loaded), `ple` (`ngram_embedding`, NVFP4, always loaded), `vit` (`model.visual`, optional), `mtp` (carried, not loaded by the engine for this model).
- Scale policy `mse`: per 16-value sub-block the `ue4m3` step minimizing the summed squared error, by analytic pre-selection plus local refinement; clipping above six times the scale is deliberate, so no fixed relative bound holds; the sidecar reports `mse`, `mse_ceil`, `mse_ratio` and `max_abs_clipped` per NVFP4 tensor plus one `section_summary` line per section (`main.rs:32-48`).
- This container's index is version 1. Since 2026-09-26 (Crow #300 C6) the converter writes an index version 2 that carries the checkpoint's `config.json` and `generation_config.json` verbatim, and the engine accepts a version 1 index only for this container, recognised by the sha256 of its index trailer, `a21afc43203d983e46be5dd378ac5fb34581ab31fb5a0f6e8093d4730078c4ba` (473,528 B; engine `docs/architecture.md` 1.7). The container itself is unchanged.

## Highlights

The first open-weight release under this architecture is Qwen3.8-Flash-Next, which introduces:

- **Hybrid Attention with QSA**: The Gated DeltaNet and Gated Attention pairing has been reworked into Gated DeltaNet and Qwen Sparse Attention (QSA). Rather than selecting individual tokens for processing, QSA operates at the micro-block level. This cuts long-context latency significantly, a critical gain as agentic workloads increasingly dominate real-world usage.
- **Gated Residual**: Residual streams with normalization are what make deep LLM training manageable. Gated Residual modulates information flowing through widened residual streams via an element-wise, data-dependent read gate and a per-branch scalar write gate. This brings finer-grained expressiveness across layers while preserving training stability and keeping inference overhead low.
- **N-gram Embedding**: Embeddings provide a unique axis for parameter scaling that requires less computation and is more amenable to offloading than Mixture-of-Experts (MoE). By indexing with short n-grams, this approach makes parameter scaling highly efficient for memory-constrained accelerators without sacrificing quality.
- **Tailored Training Recipe**: The Muon and AdamW optimizers are applied to specific weight categories to maximize efficiency. Guided by refitted scaling laws, we eliminate traditional batch-size warmups and start directly at the target batch size, substantially reducing total optimizer steps while safely supporting larger learning rates for robust convergence.

For more details, please refer to our blog post [Qwen3.8-Flash-Next](https://qwen.ai/blog?id=qwen3.8-flash-next) and [the technical report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf).

We are excited to embark on this next chapter with you and welcome your feedback as we build what comes next.

- crow-nest note: `serve` renders the upstream template, but a request that names neither `chat_template_kwargs.enable_thinking` nor `reasoning_effort` runs NON-thinking; see Quickstart. Video input is not implemented in crow-nest.

## Model Overview

- Type: Causal Language Model with Vision Encoder
- Training Stage: Pre-training & Post-training
- Language Model
    - Number of Parameters: 125B with 6B activated, plus 51B n-gram embedding and 4B MTP
    - Hidden Dimension: 2560
    - Token Embedding: 248320 (Padded)
    - N-gram Embedding: 20,000,000 (bigrams/trigrams at layer 2)
    - Number of Layers: 48
    - Hidden Layout: 12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))
    - Gated DeltaNet:
        - Number of Linear Attention Heads: 48 for V and 16 for QK
        - Head Dimension: 128
    - Qwen Sparse Attention:
        - Number of Attention Heads: 24 for Q and 2 for KV
        - Head Dimension: 256
        - Rotary Position Embedding Dimension: 64
        - Indexer Structure: MQA with 4 Query Heads and 1 Shared Key Head
        - Indexer Head Dimension: 128
        - Budget: 512 blocks or 2048 tokens
    - Mixture Of Experts
        - Number of Experts: 512
        - Number of Activated Experts: 10 Routed + 1 Shared
        - Expert Intermediate Dimension: 640
    - Gated Residual:
        - Number of Branches: 4
        - Bottleneck Rank: 320
    - LM Output: 248320 (Padded)
    - MTP: 1 layer, trained with multi-steps
- Context Length: 262,144 natively and extensible up to 1,000,000 tokens.

What this container carries of it: all 48 layers with the 512 routed experts, the PLE n-gram embedding (NVFP4), the vision tower (NVFP4) and the MTP layer (not used by crow-nest for this model). crow-nest boots it at 200,000 tokens of context with an FP8 e4m3 KV cache by default; YaRN and contexts beyond 262,144 are not available in crow-nest.

## Benchmark Results

### The original BF16 model (upstream, not re-measured here)

The tables below are the upstream card's, verbatim. They were measured by the Qwen team on the BF16 model with their own harnesses. None of them was re-run on this quant.

<style>
.vl-table th{font-size:15px!important;line-height:1.2}
.vl-table td:not(.benchmark-cell):not([colspan]){font-size:15px;line-height:1.2;vertical-align:middle}
.vl-table .benchmark-cell{padding:12px 10px 12px 18px!important;vertical-align:middle}
.vl-table .benchmark-capability{font-size:15px;font-weight:600;line-height:1.22;color:#171717}
.vl-table .benchmark-name{margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B}
.vl-table .metric-stack{display:flex;flex-direction:column;gap:7px;padding:3px 0}
.vl-table .metric-label{font-size:10px;font-weight:400;line-height:1.1;color:#777}
.vl-table .metric-value{margin-top:2px;font-size:15px;line-height:1.15;color:#171717}
.vl-table .metric-pair{white-space:nowrap}
.vl-table .metric-sep{color:#9A9A9A;padding:0 3px}

@media (prefers-color-scheme: dark){
.vl-table th,.vl-table td[colspan]{color:#8DA2FF!important;border-bottom-color:#8DA2FF!important}
.vl-table td[colspan]{background:rgba(10,46,254,0.22)!important}
.vl-table .benchmark-capability,.vl-table .metric-value{color:#E8E8E8!important}
.vl-table .benchmark-name,.vl-table .metric-label{color:#A3A3A3!important}
}
</style>

### Language

<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1200px;margin:0 auto;padding:16px 0">
<table class="vl-table" style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px">
<thead><tr>
<th style="padding:10px 7px;text-align:left;font-weight:600;border-bottom:2px solid #0A2EFE;color:#0A2EFE"></th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;background:rgba(10, 46, 254, 0.08);">Qwen3.8-Flash-Next</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.8-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.7-Plus</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">DeepSeek-V4-Flash-0731</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Claude-Opus-4.6 (Max)</th></tr></thead>
<tbody>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717"># Params</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">125B</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">27B</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">397B</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">284B</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717"># Activated params</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">6B</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">27B</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">17B</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">13B</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717"># N-gram embedding params</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">51B</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">Coding</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Agentic coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">DeepSWE 1.1</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>58.7</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">42.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">16.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">54.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Agentic coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">SWE-bench Pro</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>62.5</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">61.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">55.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">56.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">53.4</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Multilingual software engineering</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">SWE-bench Multilingual</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>81.0</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">73.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">75.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">77.5</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Repo-level code generation</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">NL2Repo-Bench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">48.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">42.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">41.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>54.2</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">47.6</td>
</tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">Agent</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Long-horizon office work</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">CoWorkBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>73.9</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">70.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">65.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">45.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">68.2</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Professional job tasks</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">JobBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>55.7</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">33.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">27.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">41.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">36.6</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Frontier agentic tasks</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">Agents' Last Exam</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@1</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">24.3</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Score</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>51.2</strong></div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@1</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">20.4</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Score</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">42.9</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@1</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">13.2</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Score</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">33.6</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@1</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>25.2</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Score</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">--</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Real-world tool use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">Toolathlon Verified (Pass@1)</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>73.5</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">67.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">50.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">70.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">General</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Instruction following</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">IFBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>81.3</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">79.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">79.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">79.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">62.5</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Scientific reasoning</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">GPQA Diamond</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>91.7</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">89.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">90.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">90.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">91.3</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Multidisciplinary reasoning</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">HLE</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">35.9</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">30.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">34.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">33.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>40.0</strong></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Competitive coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">LiveCodeBench v6</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>91.9</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">90.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">89.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">90.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">88.8</td>
</tr>
</tbody>
</table>
<p style="margin-top:12px;font-size:10px;line-height:1.4;opacity:0.7">1. DeepSWE 1.1: evaluated with the Claude Code and mini-SWE-agent harnesses, temp=1.0, top_p=0.95, 256K context window. We report the highest score across the two harnesses; notably, Qwen3.8-Flash-Next performs best on mini-SWE-agent.<br>2. SWE-bench Pro: except for Claude-Opus-4.6 (Max), for which we report the officially published score, all models are evaluated with the Claude Code harness, temp=1.0, top_p=0.95, 256K context window. Problematic tasks were corrected and all baseline models were re-evaluated on the refined benchmark.<br>3. SWE-bench Multilingual: evaluated with the mini-SWE-agent harness, temp=1.0, top_p=0.95, 256K context window.<br>4. NL2Repo-Bench: evaluated with the Claude Code harness. To prevent reward hacking, we disable Bash commands that attempt to access the specific repository, such as pip download, pip install and git clone.<br>5. CoWorkBench: an in-house cowork benchmark for evaluating long-horizon office and productivity agent tasks across computer science, finance, law, medical and other productivity domains.<br>6. HLE: judged by GPT-4o.<br>7. The best result in each row is shown in bold.<br>8. Empty cells (--): scores are not yet available or are not applicable.</p>
</div>

### Vision Language

<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1000px;margin:0 auto;padding:16px 0">
<table class="vl-table" style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px">
<thead><tr>
<th style="padding:10px 7px;text-align:left;font-weight:600;border-bottom:2px solid #0A2EFE;color:#0A2EFE"></th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:16.00%;background:rgba(10, 46, 254, 0.08);">Qwen3.8-Flash-Next</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:16.00%;">Qwen3.8-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:16.00%;">Qwen3.7-Plus</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:16.00%;">Claude-Opus-4.6 (Max)</th></tr></thead>
<tbody>
<tr><td colspan="5" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">Agentic Multimodal Intelligence</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Multimodal tool use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">ClawEval-MM</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>64.4</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>60.4</strong></div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">57.4</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">56.9</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">57.4</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">60.1</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">52.5</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">54.7</div></div></div></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Application recreation</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">RecreationBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>49.9</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">47.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">30.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Mobile use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">AndroidWorld</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>84.5</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">81.9</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">81.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">62.0</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Computer use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">OSWorld 2.0</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Binary</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>19.4</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Partial</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>52.3</strong></div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Binary</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">19.4</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Partial</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">48.0</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Binary</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">2.8</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Partial</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">21.5</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Visual web development</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">Vision2Web</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>64.0</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">62.9</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">42.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr><td colspan="5" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">General Multimodal Intelligence</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Embodied intelligence</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">ERQA</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>72.3</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">65.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">69.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">40.8</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Long video understanding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">LVBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>76.6</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">72.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">76.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">63.0</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Real-world perception</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">RealWorldQA</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>88.5</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">85.9</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">86.9</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">73.9</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Visual math problem solving</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">MathVision</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>90.6</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>95.7</strong></div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">90.0</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">94.6</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">90.3</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">88.7</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">65.5</div></div></div></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Scientific chart analysis</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">CharXiv (RQ)</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">84.6</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>90.6</strong></div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">83.7</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">90.2</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>85.8</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">85.9</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">66.0</div></div></div></td>
</tr>
</tbody>
</table>
<p style="margin-top:12px;font-size:10px;line-height:1.4;opacity:0.7">1. ClawEval-MM: scores are reported as "pass@3 / average score". Pass@3 measures the percentage passed in at least one of three trials, and the average score is the mean score across the three trials.<br>2. RecreationBench: an in-house long-horizon application-recreation benchmark for evaluating hybrid-agent abilities spanning five platforms — desktop (Ubuntu, macOS, Windows), mobile (Android) and web.<br>3. OSWorld 2.0: scores are reported as "binary / partial". The binary score is the percentage of tasks that receive the full task reward, while the partial score aggregates the partial rewards obtained across all tasks.<br>4. Vision2Web: scores are reported as the average over the frontend, webpage and website categories, using the Claude Code harness and judged by gpt-5.4-2026-03-05.<br>5. MathVision, CharXiv (RQ): scores are reported as "without CI / with CI". A small number of incorrect ground-truth annotations in MathVision were corrected after manual verification. Our model's score is evaluated using a fixed prompt, e.g. "Please reason step by step, and put your final answer within \boxed{}." For other models, we report the higher score between runs with and without the \boxed{} formatting.<br>6. The best result in each row is shown in bold.<br>7. Empty cells (--) indicate scores not yet available or not applicable.</p>
</div>

### This quant: measured

Machine for every row: one RTX 5090, Arch Linux, driver 610.57.04, CUDA 13.3.1, this container, unless a row says Windows. llama.cpp runs the same model on this machine as `Qwen3.8-Flash-Next-UD-Q2_K_XL` (GGUF, 2.4 bpw, 73 GB), so every comparison below is two different weight sets.

**Speed, before the 2026-09-23 engine changes** (the rows of record; not re-measured on these forms since)

| metric | crow-nest CNQ4.5-M | llama.cpp UD-Q2_K_XL | form | source |
|---|---|---|---|---|
| prefill, 16,064 id prompt, cold | **968 / 964 tok/s** | 922.5 tok/s (Windows) | completed-prompt average, two runs | engine commit 1032bc5, 2026-09-17; engine issue #10 |
| decode at 16k context | **42.5 tok/s** (23.52 ms per token) | 44.9 tok/s (Windows pair of 2026-09-11, 22.27 ms per token) | 255 timed steps after the prompt above, at the engine default of 2026-09-18 | engine commit `6c87054`, engine issue #61 (61g), 2026-09-18 |
| decode at 16k context, before the attention lever of 2026-09-18 | 36.8 tok/s (27.19 / 27.22 ms per token) | 41.8 tok/s (Linux, `--load-mode mmap`, `-ncmoe 31 -t 24`, short prompts, 2026-09-16; 36.7 with `--load-mode none`) | 128 steps after the prompt above vs Crow's placement measurement; not a pair | engine commit 1032bc5, 2026-09-17; `nibor1896/Crow` `docs/operating-points.md` |
| prefill, 1024 id prompt, cold | 740 tok/s | n/a | `decode parity`, two runs (92 tok/s before the same-day PLE fix of 2026-09-17) | engine commit 1032bc5, 2026-09-17 |
| warm short turns through `serve` | 228 ms prefill, 247 ms to the first token | n/a | 3,296 id cached prefix, 39 to 101 new ids, mean of six turns, two runs | engine commit 4004e66, 2026-09-17 |
| first token after a restart, 3,928 id prompt in Crow | 819 tok/s prefill | n/a | robin's live session, one reading | engine issue #68 artefacts, 2026-09-17 |

- On 2026-09-23 the engine changed in three ways: the PLE row read was fixed (engine commit `85a48e7`, issue #91), activations gained a per-row pre-scale (`488a840`), and `CROW_QFUSE` fusions became opt-in. The rows above were measured before; the unfused default's decode speed on the 16k form is not measured.

**Speed, after the 2026-09-23 fixes** (2026-09-24 and 2026-09-25, engine v0.5.0 and the v0.6.0 development tree, engine defaults incl. the recalibrated hot set `hotsets-M-crow0924-n160.json`)

| metric | value | form | source |
|---|---|---|---|
| decode, 122k-token held-out prompt | 35.6 to 35.8 tok/s (32.4 to 32.5 with this package's manifest); cold experts per token 180.5 vs 217.7 | `decode run` 64 tokens, N 148 hot experts per layer | engine issue #106, `decode_out/hotset-0924/speed.log`, 2026-09-24 |
| decode in `serve`, live Crow traffic | median 38.0 tok/s at hot-set hit rate 0.72 on 100k to 150k context (n = 50); 37.4 tok/s at 0.71 above 150k (n = 22) | 178 routing events of one session | engine issue #106, 2026-09-24 |
| cold prefill in `serve`, 8,640-token prompt | 8,312 to 9,239 ms | engine default boot, `serve` at `0263990` | engine issue #118, 2026-09-25 |
| warm turn after a parked side request | prefill 28 of 8,668 tokens in 196.6 ms | same | engine issue #118, 2026-09-25 |

- No llama.cpp reading exists on these forms; the llama.cpp rows above are the latest ones for this model (2026-09-11 Windows, 2026-09-16 Linux).
- The 972 tok/s prefill and 42 tok/s decode figures of the engine's decision record are targets (spec section 0.1, approved 2026-09-02) for a session filled to the 200,000-token floor; no reading at that fill exists on Linux, so neither is called met.

**Quality, the ten-task gate** (both arms run different weights: llama.cpp `UD-Q2_K_XL`, `-ncmoe 40`, the arm of `decode_out/srv-59b.log`, 2026-09-11; crow-nest CNQ4.5-M)

| check | crow-nest CNQ4.5-M | llama.cpp UD-Q2_K_XL | reader | source |
|---|---|---|---|---|
| ten tasks, greedy, strict 400 word cap on `t4-prose` | 2 Pass / 5 Partial / 3 Fail of 10 | 2 Pass / 6 Partial / 2 Fail of 10 | same reader for both arms, Rev3 | 2026-09-10, B4, engine issue #11 (comment), `nibor1896/Crow` issue #192 (comment) |
| ten tasks, sampled, six seeds, `temperature` above 0 | gate met in 1 of 6 seeds; 4 Pass / 37 Partial / 19 Fail of 60 answers; degeneration 0 of 60 | the reference is the greedy row above | one reader over all six rows, Rev3 | 2026-09-10, C2, engine issue #44 |

- No sugarcoating: greedy crow-nest is one task below the llama.cpp reference on this reading, and sampling met the gate in 1 of 6 seeds (2026-09-10). These runs predate the 2026-09-23 PLE fix; no ten-task run of the fixed engine exists.
- Oracle KLD against the f32 originals exists only for the engine before the fix (2026-09-19, engine commit `09cb00a`, `docs/oracle-kld.md`): CNQ4.5-M 0.343 (298 rows) and 0.461 (607 rows), same top-1 82.4 %; llama.cpp `UD-Q2_K_XL` 0.408 / 0.557 on the same rows. They describe that engine, not the current one; a container KLD against BF16 is not measured.

**Token corruption in long tool calls** (engine issue #91; multi-site probe, 23 corrupt tool-call sites of one Crow session, 9 with fresh context, teacher-forced, prompts of 18k to 103k tokens, 2026-09-23)

| arm | corrupt token wins | fresh context | mean margin, nats |
|---|---|---|---|
| crow-nest before the fix, bare container | 15 of 23 | 6 of 9 | -1.67 |
| crow-nest before the fix, dense BF16 overlay | 12 of 23 | 3 of 9 | -0.23 |
| crow-nest with the fix (`85a48e7`), dense BF16 overlay | 4 of 23 | 0 of 9 | +8.40 |
| llama.cpp UD-Q2_K_XL through llama-server | 4 of 23 | 1 of 9 | +8.29 |

- Cause: the PLE n-gram rows were read at the wrong container offset since the engine's first commit (2026-09-05); fixed in `85a48e7` (v0.4.0, 2026-09-23). The bare container with the fix has no multi-site number (that boot crashed with `CUDA_ERROR_INVALID_CONTEXT`, 2026-09-23). The 4 sites still corrupt all carry an earlier corrupt spelling in their context; llama.cpp loses three of the same four.
- Live reading (robin's Crow session, evening of 2026-09-23, bare container with the fix): 0 corruption hits in 73,887 characters of tool-call arguments, against 60 in the same amount of text before the fix. One session, counted once.

**Vision** (crow-nest engine)

| item | value |
|---|---|
| section | `vit`, the section mark `model.visual`, optional to load |
| tensors | 333 records: 27 vision blocks with 12 tensors each, plus the patch-embed convolution, the position embedding and the merger; 112 NVFP4 (4.5 bpw, `--scales mse`, per-tensor error stats in the sidecar under `"section":"vit"`) and 221 BF16 keeps, counted in the container index 2026-09-14 |
| geometry | hidden 1152, 16 heads; patch conv 1536 to 1152; pos embed 2304 x 1152; merger fc2 4608 to 2560; `gelu_pytorch_tanh` in the blocks, exact-erf GELU in the merger; read from the container index 2026-09-14 |
| container path | landed 2026-09-14 (engine `#VIT`, issue #66): ViT embeddings against the f32 oracle over the same container weights max_abs 3.43e-06 at cos 1.000000 |
| F16 projector path, the default when the file is found | since 2026-09-24 (engine #108): `mmproj-F16.gguf` from `unsloth/Qwen3.8-Flash-Next-GGUF`; the 112 vision linears run in F16 for +611 MiB of VRAM (about 4.8 hot-set units, taken before the planner sizes the hot set). Each of the 333 container `vit` tensors matches its projector tensor at cosine >= 0.9958 (CPU check, 2026-09-24). Live 2026-09-24: boot line `mode f16`, colour probe 11 of 11, a render described correctly |
| image size | since 2026-09-24 (engine #107) every image reaches the model at 1,024 to 1,280 visual tokens (`CROW_VIT_MIN_TOKENS`, `CROW_VIT_MAX_TOKENS`), the window llama.cpp's `--image-min-tokens 1024` point uses |
| image positions | since v0.7.0 (engine #123, 2026-09-27) image tokens get the mrope H/W positions HF and llama.cpp use; before, the positions followed the tower's patch order (since 2026-09-14). The quality effect on this model was not measured |

## Quickstart

### Requirements

| item | value | source |
|---|---|---|
| engine | crow-nest, `https://github.com/nibor1896/crow-nest`, current release v0.7.0; the server binary is `serve`, started from the engine repo root | engine README |
| OS | Linux (x86_64, tested on Arch Linux, kernel 7.2, driver 610.57.04) and Windows x64 (crow-nest v0.3.0; not re-run since v0.3.1). Linux needs the CUDA 13.3 runtime libraries in `CUDA_LIB` (default `~/.local/share/crow/cuda/lib`) and the container on a path WITHOUT filesystem compression (`chattr +m` on btrfs `compress=` mounts); `tools/serve-linux.sh` starts `serve` inside a memory-bounded `systemd-run` scope | engine README; engine issue #15, closed 2026-09-17 |
| GPU | one NVIDIA Blackwell GPU, `sm_120`, `compute_120a` target; every measured number here is one RTX 5090 (32 GB) | `docs/system-landscape.md` of the engine repo |
| host RAM | 62 to 64 GB class: the engine pins up to 46 GiB of host memory for the cold expert tier, sized at boot from the RAM that can be pinned (on Linux the NVIDIA driver's freed pinned-page pool and the page cache count as free; another live CUDA process makes the gate conservative); measured 2026-09-17. On Linux the cold tier is allocated as `register` by default since 2026-09-23: anonymous memory registered with the driver, charged to the serve scope (engine issue #103, `CROW_PINNED_ALLOC`) | `docs/system-landscape.md` and `docs/architecture.md` 2.1 of the engine repo |
| CUDA | CUDA 13.3. Windows: `nvrtc64_133_0.dll` needs the toolkit bin directory on `PATH`. Linux: `libnvrtc.so.13` and the runtime on `LD_LIBRARY_PATH` (never the `lib/stubs` directory); the driver's own `libcuda.so.1` | `docs/system-landscape.md` of the engine repo |
| context and KV | 200,000 tokens by default (`CROW_CONTEXT`), KV cache FP8 e4m3 (`CROW_KV`) | `docs/env.md` of the engine repo |
| container placement | the engine default path is `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` inside the engine repo, or `CROW_CNQ` names any path | `geo.rs` `DEFAULT_CNQ`, `docs/env.md` row `CROW_CNQ` |
| hot-set manifest | default `decode_out/hotsets-M-crow0924-n160.json` (in the git clone, and in this repository as `hotsets-M-crow0924-n160.json`); `CROW_HOTSETS` names any other file, for example this package's `hotsets-M-longctx2100-n160.json` | `geo.rs` `DEFAULT_HOTSETS`, engine issue #106 |
| tokenizer placement | `models/Qwen3.8-Flash-Next-original/tokenizer.json` and `tokenizer_config.json` beside the repo root, or `CROW_TOKENIZER` names the file | `tokenizer.rs`, `docs/env.md` row `CROW_TOKENIZER` |
| projector placement (optional) | `models/mmproj-F16.gguf` beside the repo root, `$CROW_MODELS`, `~/.local/share/crow/models`, or `CROW_VIT_MMPROJ` names it (`0` = always the container's tower) | `vit.rs` `resolve_mmproj`, `docs/env.md` row `CROW_VIT_MMPROJ` |
| server | binds `127.0.0.1`, default port 8099, one request at a time, one engine per machine via `engine/.engine.lock` | engine README |
| VRAM sharing | while idle `serve` can lend about 1.7 GiB of stateless scratch VRAM to Crow's renderer (`POST /v1/crow/vram/lend`, engine issue #117, 2026-09-25) | engine CHANGELOG v0.6.0 |

### Download and run (Linux)

```bash
git clone https://github.com/nibor1896/crow-nest && cd crow-nest
(cd engine && cargo build --release --bin serve)

# the container (105 GB); add the other files of this repository to verify or self-test
hf download nibor1896/Qwen3.8-Flash-Next-CNQ4.5-M Qwen3.8-Flash-Next-CNQ4.5-M.cnq --local-dir converter

# tokenizer, chat template and config from the upstream model
hf download Qwen/Qwen3.8-Flash-Next tokenizer.json tokenizer_config.json config.json generation_config.json \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --local-dir models/Qwen3.8-Flash-Next-original
# optional: the F16 vision projector
hf download unsloth/Qwen3.8-Flash-Next-GGUF mmproj-F16.gguf \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 --local-dir models

tools/serve-linux.sh --port 8099
```

- Windows: `engine/target/release/serve.exe --port 8099` (engine README); last run of record on Windows is crow-nest v0.3.0.
- The tokenizer and projector lines are this card's reading of where `serve` looks (`tokenizer.rs`, `vit.rs` at crow-nest `d47128c`); the machine the numbers come from has those files in exactly these places.

### Chat Completions API

`POST /v1/chat/completions` takes the OpenAI chat format, streamed or not. Images are base64 `data:` URLs in `image_url` blocks; `http(s)` URLs are refused.

```bash
curl -s http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write a Python function to merge two sorted linked lists."}],
  "chat_template_kwargs": {"enable_thinking": true},
  "stream": true
}'
```

| request field | crow-nest `serve` | upstream card |
|---|---|---|
| thinking | off unless `chat_template_kwargs.enable_thinking: true`, or a top-level / kwargs `reasoning_effort` of `low`, `medium`, `high` or `xhigh` (`high` renders `xhigh`; `none` = off; anything else is a 400) | on by default, `xhigh` |
| `preserve_thinking` | not passed to the template, so the template default applies (preserved) | on by default |
| `temperature`, `top_p`, `top_k`, `min_p` absent | filled from the upstream sampling row of the request's thinking mode (engine issue #111, 2026-09-24) | the rows are a recommendation |
| `temperature <= 0` sent | greedy (the gates and probes) | n/a |
| `presence_penalty` absent | 0, not the card's 1.5 (engine issue #91) | 1.5 in the non-thinking row |
| `max_tokens` | default 8,192, capped at 32,768 | up to 262,144 reasoning + 131,072 final recommended |
| video | not implemented | supported |

- Source: the request table of `engine/src/bin/serve.rs` at crow-nest `d47128c`, identical in v0.7.0 for these fields. The rule "a request without `temperature` runs greedy" of the 2026-09-11 decision (engine issue #1) was replaced by the card rows on 2026-09-24 (engine issue #111).

## Best Practices

Upstream, verbatim:

1. **Sampling Parameters**: We suggest using the following sets of sampling parameters:  
    
    - Thinking Mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
    - Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
    
    For supported frameworks, you can adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetition. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.

What crow-nest `serve` does with them:

- A request that leaves `temperature`, `top_p`, `top_k` and `min_p` out gets exactly these rows by its thinking mode: 1.0 / 0.95 / 20 / 0.0 thinking, 0.7 / 0.8 / 20 / 0.0 non-thinking. The model is never decoded greedy unless the client sends `temperature <= 0` (engine issue #111, 2026-09-24).
- `presence_penalty` defaults to 0 in both modes, not 1.5: crow-nest applies it HF-style over the whole answer, and at 1.5 it pushed long agent answers full of hashes, hex and repeated code off tokens they needed (engine issue #91, 2026-09-22). Send `presence_penalty: 1.5` for the upstream non-thinking row in full.
- Output length is capped at 32,768 tokens per request; YaRN and video (upstream items 3 and 4) are not available in crow-nest.

## Self-test

This package verifies itself. `selftest/` ships goldens that the layer-wise oracle produced from the UNQUANTIZED originals, and the engine compares its own layer outputs against them — no original safetensors, no python, no oracle environment, no network. It is the one numeric check a download can actually run (engine issue #64, first run 2026-09-18; the second check engine issue #69, 2026-09-18).

| item | value |
|---|---|
| what runs | `decode selftest <golden-dir>` of the crow-nest engine: one container load and two checks, 32.0 s wall in total on 2026-09-18, of which the load is 23 s |
| check 1 | the WHOLE text decoder layer `0` — hyper-connection mix, the GDN linear-attention mixer, the second mix, the `352`-expert MoE, both residual injections — on a fixed 8-token input, against the f32 reference of the unquantized weights |
| check 2 | the `self_attn` mixer of layer `3`, the first full-attention layer — q/k/v projections, the two head norms, rotary, the KV store, the sparse-attention indexer and its selection, the attention kernel, the sigmoid gate and `o_proj` — on a fixed 8-token input |
| the goldens | produced 2026-09-02 by the engine repo's `oracle/` chain from `Qwen/Qwen3.8-Flash-Next` at the revision of record: transformers 5.16.1, torch 2.13.0+cpu, f32, real unquantized weights, 24 of 24 layer-`0` tensors and 9 of 9 layer-`3` `self_attn` tensors suffix-matched |
| gate | `max_abs <= 0.125` for layer `0` and `max_abs <= 0.625` for layer `3`, with `NaN == 0`, both bounds inclusive; an output that is identically zero fails whatever the gate says (2026-09-18) |
| measured, this package | layer `0` `max_abs 9.184837e-2`, `rel_L2 1.3671e-2` (73.5 percent of its gate), layer `3` `max_abs 4.473233e-1`, `rel_L2 1.2821e-1`, correlation `0.99179` (71.6 percent of its gate), `NaN 0`, `PASS 2 of 2`, on 2026-09-18, RTX 5090, driver 610.57.04, CUDA 13.3.1, NVRTC 13.3.33, engine v0.3.x; not re-run after the 2026-09-23 engine changes (the activation pre-scale touches both checks) |
| positive control | `test ! -d models`: in a directory that holds the originals the self-test REFUSES before the engine starts, because such a run says nothing about a download; both arms run on 2026-09-18 |
| what it does not cover | two sub-blocks on one 8-token input each; it catches a container that was corrupted, truncated, converted with the wrong scales or loaded by a broken build. It is not the engine's full numeric contract, which is the parity gate of the engine repo |

```
tools/selftest.sh /path/to/this/package
```

## Converter command of record

```
converter --scales mse <model-dir> converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq
wrote converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq: 1658 tensors (843 nvfp4, 815 bf16-keep), payload 104.73 GB, scales mse, violations 9178206, elapsed 5205 s
```

- The first line is the command that produced this container; the second line is the result line of the generation log, verbatim (`converter/converter_mse_vollauf.log:24`).
- The `violations` figure counts elements above the old max-rel bound; under `--scales mse` that is expected, not a defect.
- The converter of today requires `--source-repo` and writes an index v2; rebuilding with it gives a v2 container, which is a different file from this v1 container of record (Crow #300 C6, 2026-09-26).

## Limitations and not measured

| item | status |
|---|---|
| downstream benchmarks (the upstream tables) on this quant | not measured |
| speed and ten-task quality of the engine after the 2026-09-23 fixes on the forms of record | not measured; the post-fix readings are the 2026-09-24 rows above |
| container KLD against the BF16 originals | not measured; the oracle KLD of 2026-09-19 predates the PLE fix |
| self-test after the 2026-09-23 engine changes | not re-run |
| the effect of the pre-v0.7.0 image positions (engine #123) on this model | not measured |
| long agentic sessions | long-context gate 5 of 5 Pass at 104,433 ids and 4 of 5 at 178,553 (greedy, before the PLE fix, 2026-09-18); the #68 degeneration record of a 300-turn session is open, its cause attributed to history content, not the engine |
| Windows after v0.3.1 | no run recorded; the `register` cold tier of engine #103 is Linux only |
| Linux, the numeric contract | the gate values of record were re-set on 2026-09-25 (parity8 `148cb72e8e44`) after the intended 2026-09-23 changes; the v0.7.0 release gate is ALL GREEN on them (2026-09-27) |
| MTP for this model | the `mtp` section is carried but not loaded by crow-nest |
| Ampere and Ada GPUs | the fallback stage is not planned (engine issue #12) |
| sharded or split variants of this container | none exist |
| trademark review of the repo name | not done; the licence regulates no naming for derivatives (see License) |

## Provenance

| item | value |
|---|---|
| base model | `Qwen/Qwen3.8-Flash-Next` on Hugging Face, public, not gated |
| revision of record | `de4b8e4d43b917e7706784d8bb445c9af86a3540`, upstream `lastModified` 2026-08-27T05:03:36Z; the Hub's `main` on 2026-09-27 |
| verification | the upstream licence text was fetched and checked against the Hub on 2026-09-11 at the revision above; the local download was byte-identical |
| projector | `unsloth/Qwen3.8-Flash-Next-GGUF` `mmproj-F16.gguf`, LFS sha256 `1f7b7f0b…a8a980`, revision `38bb39ee` (`lastModified` 2026-09-02); that repository declares `license: other` (Hub API, 2026-09-27), not checked further |

## License

| item | value |
|---|---|
| licence | Qwen Community License 1.0 for the model weights; name and version verified verbatim against the upstream repo on 2026-09-11, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| text | ships as `LICENSE` next to this card: 3,235 B, 15 lines, sha256 `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08`, byte-identical to the upstream file at the revision above, verified 2026-09-11 |
| duty, every copy | keep the copyright notice and the permission notice in all copies or substantial portions (licence section 1, sentence 1) |
| duty, large deployments | above 100,000,000 monthly active users or above US$ 20,000,000 monthly revenue, the model name must be displayed prominently in the product UI (licence section 1, sentence 2; licence verified 2026-09-11) |
| duty, hosted service | a Model as a Service or AI Work Assistant offering needs a separate licence from Qwen before commercial use; internal use without third-party access is exempt (licence section 2) |
| attribution | the licence prescribes no attribution formula and no naming rule for derivatives; this card therefore states provenance plus the upstream BibTeX and invents no formula |
| scope | this package is weights for local inference with the crow-nest engine; no hosted inference endpoint is offered |
| engine code | the crow-nest engine and converter are Apache-2.0, see the `LICENSE` of the engine repo |

## Citation

The upstream card's citation (`README.md:653-673` of the upstream repo at the revision of record), verbatim:

```bibtex
@techreport{qwen2026design,
    title       = {On the Design of {Qwen3.8-Next} Architecture: Evaluation, Efficiency, and Training Stability},
    author      = {{Qwen Team}},
    institution = {Alibaba Group},
    month       = {August},
    year        = {2026}
}

@misc{qwen3.8flashnext,
    title  = {{Qwen3.8-Flash-Next}: A New Architecture, Towards Ultimate Cost-Efficiency},
    author = {{Qwen Team}},
    month  = {August},
    year   = {2026},
    url    = {https://qwen.ai/blog?id=qwen3.8-flash-next}
}
```
