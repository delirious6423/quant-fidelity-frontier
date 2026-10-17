---
license: apache-2.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- nvfp4
- cnq
- quantized
- dense
- vision
- multimodal
- linux
- rtx-5090
- crow-nest
- mtp
---

# Qwen3.8-27B-CNQ4.5

> [!Note]
> This repository holds a quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) in the CNQ container format of the [crow-nest](https://github.com/nibor1896/crow-nest) engine. It is not the upstream checkpoint: it does not load in Transformers, vLLM, SGLang, TokenSpeed or llama.cpp, only in crow-nest.
>
> The sections "Qwen3.8 Highlights", "Model Overview", the upstream half of "Benchmark Results", the sampling rows under "Best Practices" and the BibTeX under "Citation" are quoted verbatim from the upstream card at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and describe the ORIGINAL BF16 model. Everything else describes this quant and says where its numbers come from.

## About this quant

| item | value |
|---|---|
| base model | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (upstream `lastModified` 2026-08-14T15:00:01Z, still `main` on 2026-09-27) |
| quantization | CNQ4.5: NVFP4 (E2M1 values, one `ue4m3` scale per 16 values, one `f32` global scale per tensor), 4.5 bpw, on the MLP, attention, Gated DeltaNet projections and `lm_head`; BF16 / f32 keeps elsewhere (see What differs) |
| scale choice | `--scales diag`: every sub-block tries all 126 finite `ue4m3` steps and keeps the one with the smallest activation-weighted squared error, weights from 491,520 calibration tokens (crow-nest `86cf4b8`, 2026-09-26) |
| quality against the BF16 original | mean KLD 0.223 over 3,470 rows, same top-1 token 86.1 % (was 0.290 / 84.0 % with the calibration-free `--scales mse` build); RTX 5090, 2026-09-26, see This quant: measured |
| container | one file, `Qwen3.8-27B-CNQ4.5.cnq`, 17,844,005,845 B (17.8 GB), 866 tensors, index v2 |
| engine | crow-nest `https://github.com/nibor1896/crow-nest`, v0.7.0 or later: own HTTP server with an OpenAI-style `/v1/chat/completions`, own container format; Rust + CUDA, no Python at run time |
| vision | NOT in the container. The tower runs from the llama.cpp F16 projector `mmproj-F16.gguf` of `unsloth/Qwen3.8-27B-GGUF`, a separate 927,607,488 B download (see Vision) |
| MTP | the upstream MTP head rides along in BF16 (section `mtp`); crow-nest uses it for speculative decoding by default, output byte-identical to plain decoding (crow-nest #95, 2026-09-27) |
| platform | Linux x86_64, CUDA 13.3, one NVIDIA Blackwell GPU (`sm_120`); every number on this card comes from one RTX 5090 (32,607 MiB) on Arch Linux, driver 610.57.04 |
| intended use | local inference with crow-nest on one GPU, as the engine behind the Crow agent; no hosted endpoint is offered |
| licence | Apache-2.0, the upstream licence, shipped as `LICENSE`; crow-nest itself is Apache-2.0 too |

### Files in this repository

| file | size | `sha256` | purpose |
|---|---|---|---|
| `Qwen3.8-27B-CNQ4.5.cnq` | 17,844,005,845 B | `ba014aeb00215cdc8d6e46d84d0b76733906b8003cb98173d9643acb05fa147d` | the container; the Hub's LFS record carries the same sha256 (checked 2026-09-27, upload commit `63a3ea24`) |
| `Qwen3.8-27B-CNQ4.5.cnq.sidecar.jsonl` | 213,231 B | `5fc23c469c7209b3d805892bb9dcab78353d7dbdeb830556ad79a89722936160` | verification sidecar written by the converter in the same run: one JSON line per tensor (866) plus one `section_summary` line; the NVFP4 lines carry `max_abs_err`, `mean_abs_err`, `max_rel_err`, `mse`, `mse_ceil`, `mse_ratio`, `max_abs_clipped`, `violations`, `scales_mode: "diag"` |
| `SHA256SUMS` | 266 B | n/a, it carries the sums | the three sums of this table |
| `LICENSE` | 11,544 B | `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` | Apache License 2.0, byte-identical to `LICENSE` of `Qwen/Qwen3.8-27B` at the revision above (fetched and compared 2026-09-27) |

- The sidecar belongs to this container: its 866 tensor lines match the container's own index in name, dtype, value count and section, 866 of 866; both files carry the same write time (2026-09-26 23:49:29 CEST); its NVFP4 `violations` sum to 11,279,944, the figure the crow-nest CHANGELOG records for the `--scales diag` build; and it is the file committed with that build (crow-nest `86cf4b8`), unchanged since.
- `sha256sum -c SHA256SUMS` in the download directory must report 3 of 3 OK; the container line reads 17.8 GB, the other two cost nothing.

### Files from other repositories, needed at run time

| file | from | size | `sha256` | why |
|---|---|---|---|---|
| `tokenizer.json` | `Qwen/Qwen3.8-27B` @ `1d4bf0f2` | 12,809,320 B | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` | the tokenizer `serve` loads |
| `tokenizer_config.json` | `Qwen/Qwen3.8-27B` @ `1d4bf0f2` | 17,928 B | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` | carries the chat template (`chat_template` field) `serve` renders with |
| `mmproj-F16.gguf` | `unsloth/Qwen3.8-27B-GGUF` @ `4ca72078` | 927,607,488 B | `cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e` | the vision tower (F16); without it the 27B boots text-only |

- All three sha256 values were computed from the files the numbers on this card were measured with and equal the Hub's records on 2026-09-27 (the two tokenizer files fetched again at the revision above; the projector's LFS sha256 on `unsloth/Qwen3.8-27B-GGUF` `main` = `4ca72078`, `lastModified` 2026-08-20).
- `config.json` and `generation_config.json` are NOT needed: the container carries both verbatim in its index (sha256 `191e0af2…` and `e70c136c…`, identical to the upstream files at `1d4bf0f2`), and crow-nest reads them from there. `chat_template.jinja` is not read either; its text equals the `chat_template` field of `tokenizer_config.json` (compared 2026-09-27).

### What differs from the original checkpoint

| part | upstream `Qwen/Qwen3.8-27B` | this container |
|---|---|---|
| MLP `gate_proj` / `up_proj` / `down_proj`, attention `q/k/v/o_proj`, Gated DeltaNet `in_proj_qkv` / `in_proj_z` / `out_proj`, `lm_head` | BF16 | NVFP4, 4.5 bpw, `--scales diag`: 401 tensors |
| token embedding, all norms, `in_proj_a` / `in_proj_b`, `conv1d`, `dt_bias` | BF16 | BF16, unchanged: 402 tensors; the embedding (2.368 GiB) stays in host RAM |
| `A_log` | BF16 in the checkpoint | f32, widened exactly: 48 tensors |
| MTP head (`mtp.*`) | BF16 | BF16, own section `mtp`, optional to load: 15 tensors |
| vision tower (`model.visual.*`) | BF16, 333 tensors, 921,460,192 B | omitted by the recipe (`recipe::omitted`); the separate F16 projector replaces it |
| total | 55,563,006,776 B in 18 safetensors shards | 17,844,005,845 B in one file |

- Recipe `cnq4.5-qwen35-dense` is a whitelist: a tensor without a row is refused by name at conversion (crow-nest CHANGELOG v0.7.0, Crow #300 C6).
- Tensor counts above were read from this container's index on 2026-09-27.

### Format, in short

The CNQ layout (magic `CNQ1`, 12-byte header, streamed payload, JSON index as a trailer with a `u64` little-endian length in the last 8 bytes), the NVFP4 block geometry (64 values, 4 sub-block scales, 36 B per block) and the sidecar are the same as for the Flash-Next container and are described once, in the crow-nest repository: [`converter/README.md`](https://github.com/nibor1896/crow-nest/blob/main/converter/README.md) and `docs/architecture.md` sections 1 and 1.7. What this container adds:

| index key | this container |
|---|---|
| `format_version` | `2` (the Flash-Next container of record is index v1) |
| `recipe` | `cnq4.5-qwen35-dense` |
| `scales` | `diag` |
| `sections` | `text` (always loaded) and `mtp` (optional); no `vit`, no `ple` |
| `model.family` / `model.model_type` | `Qwen35Dense` / `qwen3_5_text` |
| `model.config_json`, `model.generation_config_json` | upstream `config.json` and `generation_config.json` verbatim as strings, sha256 `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` and `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` |
| `model.geo` | 64 layers (48 Gated DeltaNet, 16 full attention), hidden 5120, FFN 17408, 24 Q / 4 KV heads of 256, vocab 248320, 1 MTP layer |
| `model.source` | repo `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, all 18 shards with size and sha256 (`sha256_from: hf-lfs`) |
| index trailer | 250,305 B, sha256 `8c304fdef2825daee0343f532483969e939d6a540e0c9d7ff95a9edfffd85750` (read 2026-09-27) |

- The engine reads the model's config from this block at boot and refuses a container whose stored config no longer matches its sha256, and any other model's config, with a named mismatch table (crow-nest v0.7.0, Crow #300 C7).
- `--scales diag` (crow-nest `adde3bd` + `86cf4b8`): per sub-block, the byte `s` in 1..0x7E minimizing `sum_j d_j (q_j(s) - w_j)^2`, with `d_j` the sum of `x_j^2` of input column `j` over the calibration tokens; the global scale stays max-based as in the other modes, so the file size and the kernels are the same as a `--scales mse` build. Clipping above six times the scale is allowed, which is why the sidecar counts 11,279,944 elements above the old max-relative bound (`--scales mse`: 1,344,706): expected, not a defect.

## Qwen3.8 Highlights

Qwen3.8-27B features the following enhancements:
- **Core Capabilities**: Comprehensive improvements across coding, professional work, research, and long-horizon agentic tasks.
- **Agent Execution**: Stronger autonomous planning and better handling of environment feedback, leading to more reliable end-to-end task completion.
- **Downstream Compatibility**: Broader support for popular harnesses and development tools, making it easier to integrate into your existing stack.
- **Flexible Thinking Control**: Thinking mode is on by default and can be disabled per request; reasoning depth can be tuned with `reasoning_effort`, and reasoning context from historical messages is retained via `preserve_thinking`.
- **Vision-Language Understanding**: Native support for image and video understanding, from STEM diagrams and documents to hour-scale videos.

- crow-nest note: `serve` renders the upstream template, but a request that names neither `chat_template_kwargs.enable_thinking` nor `reasoning_effort` runs NON-thinking; see Quickstart. Video input is not implemented in crow-nest.

## Model Overview

- Type: Causal Language Model with Vision Encoder
- Training Stage: Pre-training & Post-training
- Language Model
    - Number of Parameters: 27B
    - Hidden Dimension: 5120
    - Token Embedding: 248,320 (Padded)
    - Number of Layers: 64
    - Hidden Layout: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
    - Gated DeltaNet:
        - Number of Linear Attention Heads: 48 for V and 16 for QK
        - Head Dimension: 128
    - Gated Attention:
        - Number of Attention Heads: 24 for Q and 4 for KV
        - Head Dimension: 256
        - Rotary Position Embedding Dimension: 64
    - Feed Forward Network:
        - Intermediate Dimension: 17,408
    - LM Output: 248,320 (Padded)
    - MTP (Multi-Token Prediction): trained with multiple steps
- Context Length: 262,144 natively and extensible up to 1,000,000 tokens.

What this container carries of it: the full 64-layer language model and the MTP head. The vision encoder is not in the file; it comes from the separate projector. crow-nest boots the 27B at a 65,536-token context by default (`CROW_CONTEXT` raises it up to the model's 262,144 as far as VRAM allows); YaRN and contexts beyond 262,144 are not available in crow-nest.

## Benchmark Results

### The original BF16 model (upstream, not re-measured here)

The two tables below are the upstream card's, verbatim. They were measured by the Qwen team on the BF16 model with their own harnesses. None of them was re-run on this quant.

### Text Performance
<style>
.vl-table th{font-size:15px!important;line-height:1.2}
.vl-table td:not(.benchmark-cell):not([colspan]){font-size:15px;line-height:1.2;vertical-align:middle}
.vl-table .benchmark-cell{padding:12px 10px 12px 18px!important;vertical-align:middle}
.vl-table .benchmark-capability{font-size:15px;font-weight:600;line-height:1.22;color:#171717}
.vl-table .benchmark-name{margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B}
.vl-table .metric-stack{display:flex;flex-direction:column;gap:7px;padding:3px 0}
.vl-table .metric-label{font-size:10px;font-weight:400;line-height:1.1;color:#777}
.vl-table .metric-value{margin-top:2px;font-size:15px;line-height:1.15;color:#171717}
</style>
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1200px;margin:0 auto;padding:16px 0">
<table class="vl-table" style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px">
<thead><tr>
<th style="padding:10px 7px;text-align:left;font-weight:600;border-bottom:2px solid #0A2EFE;color:#0A2EFE"></th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;background:rgba(10, 46, 254, 0.08);">Qwen3.8-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.6-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.7-Plus</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Muse Glimmer-30B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Opus4.6 Max</th></tr></thead>
<tbody>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">Coding</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Agentic terminal coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">Terminal Bench 2.1 (Terminus)</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">73.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">63.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">64.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">51.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>78.2</strong></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Agentic coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">SWE-bench Pro</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>61.7</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">53.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">57.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">51.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">53.4</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Repo-level code generation</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">NL2Repo-Bench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">42.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">36.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">41.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>47.6</strong></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Agentic coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">DeepSWE 1.1</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>42.2</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">13.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">14.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Software engineering</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">QwenSWEBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>79.0</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">49.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">59.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">63.8</td>
</tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">Agent</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Long-horizon office work</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">CoWorkBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>70.7</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">61.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">65.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">68.2</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Professional job tasks</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">JobBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>33.4</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">21.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">27.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Frontier agentic tasks</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">Agents' Last Exam</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@1</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>20.4</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Score</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>42.9</strong></div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@1</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">10.6</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Score</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">27.3</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@1</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">13.2</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Score</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">33.6</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">General</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Instruction following</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">IFBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>79.5</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">69.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">79.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">77.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">62.5</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Scientific reasoning</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">GPQA Diamond</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">89.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">87.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">90.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">83.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>91.3</strong></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Multidisciplinary reasoning</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">HLE</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">30.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">24.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">34.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">22.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>40.0</strong></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Competitive coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">LiveCodeBench v6</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>90.3</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">83.9</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">89.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">88.8</td>
</tr>
</tbody>
</table>
<div style="margin-top:12px;font-size:11px;line-height:1.5;color:rgba(0,0,0,0.72)">
<ol style="margin:0;padding-left:20px">
<li>SWE-bench Pro: Except for Opus4.6 Max, which uses the officially reported score, all models are evaluated with the Claude Code harness at temp=1.0, top_p=0.95, and a 256K context window. Problematic tasks were corrected, and all baseline models were re-evaluated on the refined benchmark.</li>
<li>NL2Repo-Bench: Evaluated with the Claude Code harness. To prevent reward hacking, we disable Bash commands that attempt to access the specific repository, such as pip download, pip install, and git clone.</li>
<li>DeepSWE 1.1: Evaluated with the Claude Code harness at temp=1.0, top_p=0.95, and a 256K context window.</li>
<li>QwenSWEBench: In-house coding benchmark for evaluating models' software engineering capabilities. Evaluated with the Claude Code harness. Reporting avg@3 with an 8-hour timeout, max_tokens=32,768, temperature=1.0, and a 256K context window.</li>
<li>CoWorkBench: In-house cowork benchmark for evaluating long-horizon tasks across computer science, finance, law, medical, and other productivity domains.</li>
<li>HLE: Judged by GPT-4o.</li>
<li>The best result in each row is shown in bold.</li>
<li>Empty cells (--) indicate that results are not yet available or not applicable.</li>
</ol>
</div>
</div>

### VL Performance
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1200px;margin:0 auto;padding:16px 0">
<table class="vl-table" style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px">
<thead><tr><th style="padding:10px 7px;text-align:left;font-weight:600;border-bottom:2px solid #0A2EFE;color:#0A2EFE"></th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;background:rgba(10, 46, 254, 0.08);">Qwen3.8-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.6-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.7-Plus</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Muse Glimmer-30B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Opus4.6 Max</th></tr></thead>
<tbody>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">Agentic Multimodal Intelligence</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Computer use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">OSWorld-Verified</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>84.3</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">63.9</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">73.3</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">65.9</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">72.7</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Browser use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">WebArena-Verified</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>64.8</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">48.8</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">55.3</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Mobile use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">AndroidWorld</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>81.9</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">70.3</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">81.0</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">62.0</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Application recreation</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">RecreationBench</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>47.1</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">29.8</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">30.2</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Multimodal tool use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">ClawEval-MM</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>57.4</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">56.9</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">42.6</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">50.4</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>57.4</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>60.1</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">52.5</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">54.7</div></div></div></td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Multimodal software engineering</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">SWE-MM</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>38.6</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">25.7</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">30.0</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">27.1</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Visual web development</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">Vision2Web</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>62.9</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">45.0</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">42.1</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td></tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">General Multimodal Intelligence</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Visual math problem solving</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">MathVision</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">90.0</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>94.6</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">85.1</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>90.3</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">65.5</div></div></div></td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">General visual reasoning</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">BabyVision</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>65.7</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>85.6</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">28.9</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">64.7</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">70.4</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">12.6</div></div></div></td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Scientific chart analysis</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">CharXiv (RQ)</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">83.7</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>90.2</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">78.4</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>85.8</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">85.9</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">78.8</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">66.0</div></div></div></td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Document intelligence</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">OmniDocBench 1.5</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">91.1</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">89.4</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>91.4</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">75.8</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">86.6</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Real-world perception</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">RealWorldQA</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">85.9</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">84.1</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>86.9</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">73.9</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Embodied intelligence</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">ERQA</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">65.5</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">62.5</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>69.8</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">40.8</td></tr>
</tbody>
</table>
<div style="margin-top:12px;font-size:11px;line-height:1.5;color:rgba(0,0,0,0.72)">
<ol style="margin:0;padding-left:20px">
<li>MathVision, BabyVision, and CharXiv (RQ): Where both settings are available, cells report “Without CI” and “With CI” separately; otherwise, only the available setting is shown. A small number of incorrect ground-truth annotations in MathVision and CharXiv (RQ) were corrected following manual verification, and all reported scores on those benchmarks were computed using the corrected annotations.</li>
<li>MathVision: Qwen3.8-27B is evaluated using the fixed prompt: “Please reason step by step, and put your final answer within <code>\boxed{}</code>.” For the remaining models, we report the higher score from two prompt variants—one with and one without the <code>\boxed{}</code> formatting requirement.</li>
<li>WebArena-Verified: Scores are computed with the official WebArena-Verified grader under the OSWorld scaffold.</li>
<li>RecreationBench: An in-house, long-horizon application-recreation benchmark designed to evaluate hybrid-agent capabilities across five platforms: desktop (Ubuntu, macOS, and Windows), mobile (Android), and the web.</li>
<li>ClawEval-MM: Scores are reported as “Pass@3 / average score.” Pass@3 is the percentage of tasks passed in at least one of three trials; the average score is the mean benchmark score across the three trials.</li>
<li>Vision2Web: Scores are averaged across the frontend, webpage, and website categories. Evaluations use the Claude Code harness and are judged by <code>gpt-5.4-2026-03-05</code>.</li>
<li>SWE-MM: Scores are evaluated on the Claude Code harness using the public dev split of SWE-bench Multimodal, with the modifications described in Appendix 8.3 of the Claude Opus 4.7 system card.</li>
<li>Empty cells (--) indicate that results are not yet available or not applicable.</li>
</ol></div>
</div>

### This quant: measured

Machine for every row: one RTX 5090, Arch Linux, driver 610.57.04, CUDA 13.3, crow-nest `serve` / `decode` with `CROW_MMA=1 CROW_GRAPH=1` (the `serve` defaults). Reference for KLD: the upstream BF16 weights through crow-nest's f32 oracle (`oracle/ref_qwen35_logits.py`); metric `tools/oracle-kld.py`, KL(reference ‖ arm) in nats, mean ± SE.

**Quality against the BF16 original** (crow-nest Crow #300 phase 2, PREREGs `decode_out/p2-kld/PREREG.md` and `decode_out/p2-lh/PREREG.md` written before any row; 3,470 rows = 905 code + 2,565 prose; 2026-09-26)

| arm | mean KLD | median | p99 | same top-1 | code rows | prose rows |
|---|---|---|---|---|---|---|
| this container, f32 math | **0.2233 ± 0.0190** | 0.0325 | 2.92 | 86.05 % | 0.4435 | 0.1456 |
| this container in the engine, BF16 KV | 0.2230 ± 0.0189 | 0.0327 | 2.92 | 86.17 % | 0.4423 | 0.1456 |
| the earlier `--scales mse` build (not published) | 0.290 ± 0.022 | 0.049 | n/a | 84.0 % | 0.572 | 0.191 |

- Paired gain of `diag` over `mse`: +0.067 ± 0.015 (4.5 SE, 23 %), measured on the simulation; the real container then read 0.2233, the simulation's value, and 379 of 1,599,897,600 scale bytes differ from the simulation (0.24 ppm, near ties of the f32 summation order).
- Engine minus container, paired: −0.0003 ± 0.0007 (the engine adds nothing measurable on these short rows).
- Published 4-bit KLD figures for the Qwen3.x 27B family (0.02 to 0.03, Unsloth, for Qwen3.6-27B) are on an unknown corpus and are not comparable to these rows; this card makes no claim against them.
- Where the error comes from (2026-09-26, on the `mse` build, 1,026 rows, one group NVFP4 at a time, the rest BF16): MLP 0.050, Gated DeltaNet 0.021, `lm_head` 0.0105, attention 0.0075, all together 0.084.

**Engine against its own references** (2026-09-26 / 2026-09-27)

| check | result |
|---|---|
| sub-block goldens against HF `qwen3_5` (`decode p2golden`, thresholds pre-registered) | ALL PASS on goldens re-exported from this container, 2026-09-26 |
| end to end, 63-token prompt + 4 decode steps (`decode parity`) | argmax 67 / 67; mean KL 5.5e-7 with BF16 KV, 1.3e-4 with FP8 KV (2026-09-26, on the `mse` build; the same check was not repeated after the rebuild) |
| KV cache: FP8 against BF16, long context (last 64 rows at six anchors of one 178k agent session) | FP8 fails the pre-registered bound (KL ≤ 0.073) at 5 of 6 anchors (1.05, 1.99, 0.51, 1.54, 4.42, 0.042), 2026-09-26; BF16 KV is therefore the 27B default |

**Speed** (tok/s, decode = generated tokens only; crow-nest numbers with this container and BF16 KV unless stated)

| form | crow-nest, plain | crow-nest, MTP on (default) | llama.cpp reference, 27B `UD-Q4_K_XL` | source |
|---|---|---|---|---|
| decode, greedy, 512 tokens, short prompt (apollo) | 73.2 / 72.2 / 72.3 | 120.2 / 120.3 / 120.0 | 66.5 plain / 123.05 with its MTP | crow-nest `decode_out/p2-mtp/RESULTS.md`, `22bea05`, 2026-09-27; llama.cpp: Crow #118, 2026-08-21 |
| decode, greedy, 512 tokens, 8k Crow-session window | 67.2 | 96.2 | same reference | crow-nest CHANGELOG v0.7.0, 2026-09-27 |
| decode, greedy, 256 tokens after a 30k prompt | 58.9 / 59.0 / 58.9 | 73.1 / 72.1 / 71.9 | not measured at 30k | `RESULTS.md` S2, 2026-09-27 |
| decode, sampled, upstream non-thinking row, seeds 1-3 | 72.5 / 71.9 / 71.9 | 103.2 / 87.7 / 95.4 | not measured | `RESULTS.md` S3, 2026-09-27 |
| decode, sampled, upstream thinking row, seeds 1-3 | 72.2 / 71.8 / 71.9 | 109.4 / 100.4 / 99.6 | not measured | `RESULTS.md` S3, 2026-09-27 |
| `serve`, one 288-token prose answer | 72.5 | 86.5 | not measured | CHANGELOG v0.7.0, 2026-09-27 |
| prefill, 8k prompt | 2,124 | 2,039 | 2,262.96 (Crow's 27B operating point, cited in Crow #142, 2026-08-27) | CHANGELOG v0.7.0, 2026-09-27 |
| prefill, 30k prompt | 1,279 (FP8 KV, `mse` build, same kernels and bytes per weight) | n/a | not measured | CHANGELOG v0.7.0, 2026-09-26 |

- The llama.cpp numbers are not paired with the crow-nest rows: they come from one tool-heavy 11-round Crow turn on the same RTX 5090 (Crow #118, 2026-08-21), not from these prompts. At short context crow-nest decodes faster than llama.cpp without MTP (73 vs 66.5) and about as fast with MTP (120 vs 123.05); llama.cpp's prefill reference is higher than crow-nest's 8k figure.
- MTP head acceptance on the 27B's own greedy text: 0.80 at the first draft position (teacher-forced reference, 2026-09-27); text the head predicts badly gains little (dracula prose: 79.8 vs 71.9 plain).

**VRAM** (`serve` boot, RTX 5090 with 32,607 MiB)

| boot | used | free after load | date |
|---|---|---|---|
| default: 65,536 context, BF16 KV (4,096 MB), MTP on, text only | 21.60 GiB | 9.77 GiB | 2026-09-27 |
| the same with the F16 projector loaded (968 MiB weights + 298.1 MiB scratch) | n/a | **8.48 GiB** | 2026-09-27 |
| context 100,000, BF16 KV, MTP on / off | 23.85 / 22.07 GiB | n/a | 2026-09-27 |
| context 131,072, BF16 KV, no MTP | 23.12 GiB | n/a | 2026-09-26 |
| context 262,144, BF16 KV | refused by the planner (16.83 GiB needed, 16.49 free) | n/a | 2026-09-26 |

**Vision** (crow-nest #122, PREREG `decode_out/p3-vit27b/PREREG.md` written before any result, 2026-09-27)

| check | threshold | measured | verdict |
|---|---|---|---|
| P0, projector against the HF BF16 tower | 333 tensors mapped, cos ≥ 0.9999 | 333 / 333, worst cos 0.99999999999999, max_abs 3.0e-8 | pass |
| P1, tower output against the f32 HF oracle on the same F16 weights, 3 images | cos_min ≥ 0.99999, max_abs ≤ 1e-4 | cos_min 0.9999997; max_abs 2.2e-3 / 3.3e-3 / 1.3e-2 (merger channel 3994, values −36 to −82, relative error 6e-5 to 1.6e-4) | FAIL on max_abs; the bound was carried over from Flash-Next's container path; accepted by the maintainer 2026-09-27 |
| P2, image-prompt logits against the f32 reference, 1,057 rows | argmax ≥ 99 %, top-5 ≥ 4.5, no NaN | argmax 98.3 % (1,039 / 1,057), top-5 4.93, no NaN; all 18 flips are `image_pad` rows (never sampled) at top-2 margins 0.007 to 0.52; text rows 25 / 25 | FAIL on argmax; accepted by the maintainer 2026-09-27 |
| P3, live `serve` | vision on, projector named at boot, colour probe 11 / 11, image order right, ≥ 7.8 GiB free | all met; a Qwen-Image render described correctly | pass |

## Quickstart

crow-nest serves this container through its own server. The steps below are for Linux; the 27B was not run on Windows.

### Requirements

| item | value | source |
|---|---|---|
| engine | crow-nest v0.7.0 (the release that added the dense 27B, MTP and the projector) or later | crow-nest CHANGELOG |
| GPU | one NVIDIA Blackwell GPU, `sm_120`; measured only on an RTX 5090 with 32 GB. The default boot uses 21.60 GiB before the projector; a smaller card was not tried | crow-nest `docs/system-landscape.md`, CHANGELOG v0.7.0 |
| CUDA | CUDA 13.3 runtime libraries (`libnvrtc.so.13` and the runtime) in `CUDA_LIB`, default `~/.local/share/crow/cuda/lib`; the driver's own `libcuda.so.1` | `tools/serve-linux.sh` |
| host | Linux x86_64 (measured on Arch Linux, 64 GB RAM); `tools/serve-linux.sh` runs `serve` in a memory-bounded `systemd-run --user` scope. The 27B's own host-RAM need was not measured separately; its token embedding holds 2.368 GiB of it | `tools/serve-linux.sh`, CHANGELOG v0.7.0 |
| build | a Rust toolchain and `cargo build --release --bin serve` in `engine/` | crow-nest README |
| disk | 17.8 GB container + 0.93 GB projector + 12.8 MB tokenizer | this card |
| file placement | the tokenizer and the projector are looked up in `models/Qwen3.8-27B/` of the directory ABOVE the container's directory, i.e. the crow-nest repo root when the container sits one level below it (`models/<model>`, `<model>` = the source repo name in the index) | `engine/src/vit.rs` `resolve_mmproj`, `engine/src/tokenizer.rs` |

- Tokenizer, v0.7.0: `serve` reads `tokenizer.json` and `tokenizer_config.json` from `models/Qwen3.8-Flash-Next-original/` for every container unless `CROW_TOKENIZER` names a file (`CROW_TOKENIZER_CONFIG` defaults to the sibling `tokenizer_config.json`). The Flash-Next and 27B files are byte-identical (compared 2026-09-25 and 2026-09-27), so either source gives the same ids; the command below sets `CROW_TOKENIZER` so that no Flash-Next directory is needed.
- Tokenizer, after v0.7.0 (crow-nest `d47128c`, #121, on `main` locally, not yet released): `serve` finds `models/Qwen3.8-27B/tokenizer.json` beside the container on its own and names the choice on the boot line `[serve] tokenizer source: ...`; checked 2026-09-27 with the Flash-Next directory moved away (boots, answers). `CROW_TOKENIZER` still wins when set.

### Download and run

```bash
git clone https://github.com/nibor1896/crow-nest && cd crow-nest
(cd engine && cargo build --release --bin serve)

# this repository: container, sidecar, SHA256SUMS, LICENSE (17.8 GB)
hf download nibor1896/Qwen3.8-27B-CNQ4.5 --local-dir cnq-27b
(cd cnq-27b && sha256sum -c SHA256SUMS)

# tokenizer and chat template from the upstream model, projector from unsloth
hf download Qwen/Qwen3.8-27B tokenizer.json tokenizer_config.json \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir models/Qwen3.8-27B
hf download unsloth/Qwen3.8-27B-GGUF mmproj-F16.gguf \
  --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 --local-dir models/Qwen3.8-27B

CROW_CNQ=cnq-27b/Qwen3.8-27B-CNQ4.5.cnq \
CROW_TOKENIZER=models/Qwen3.8-27B/tokenizer.json \
  tools/serve-linux.sh --port 8099
```

- The documented command of the crow-nest README is `CROW_CNQ=converter/Qwen3.8-27B-CNQ4.5.cnq tools/serve-linux.sh --port 8099` with the container in `converter/`; any directory one level below the repo root works the same way. The block above as a whole (download to `cnq-27b/`, `CROW_TOKENIZER` on v0.7.0) was not run end to end; its parts were: the serve command with `CROW_CNQ` (2026-09-27), the automatic tokenizer lookup (#121, 2026-09-27) and the projector lookup in `models/Qwen3.8-27B/` (#122, 2026-09-27).
- Expected boot lines (from the 2026-09-27 vision run): `[boot] container index v2 (family Qwen35Dense, recipe cnq4.5-qwen35-dense, Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0): config from its model block ...`, `[boot] kv cache dtype bf16 (default of family Qwen35Dense, CROW_KV unset)` and `[vit] visual tower loaded: mode f16 (llama.cpp projector models/Qwen3.8-27B/mmproj-F16.gguf, 112 F16 linears) ...`.
- `serve` binds `127.0.0.1:8099`, answers one request at a time and allows one engine per machine (`engine/.engine.lock`). `GET /props` reports `vision: true` when the projector loaded.
- Switches (crow-nest `docs/env.md`): `CROW_CONTEXT` (default 65,536 for this model), `CROW_MTP=0` (MTP off), `CROW_KV=fp8` (FP8 KV, fails the long-context bound above), `CROW_VIT_MMPROJ=<path>` (projector elsewhere).

### Chat Completions API

`POST /v1/chat/completions` takes the OpenAI chat format, streamed or not.

```bash
# non-thinking (the serve default when the request names no thinking switch)
curl -s http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write a Python function to merge two sorted linked lists."}],
  "max_tokens": 1024
}'

# thinking on, as the upstream card uses the model by default
curl -s http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write a Python function to merge two sorted linked lists."}],
  "chat_template_kwargs": {"enable_thinking": true},
  "stream": true
}'

# image input: serve takes images as base64 data URLs only, not as http(s) URLs
IMG=$(base64 -w0 photo.png)
curl -s http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}},
    {"type": "text", "text": "Where is this?"}]}]
}'
```

| request field | crow-nest `serve` | upstream card |
|---|---|---|
| thinking | off unless `chat_template_kwargs.enable_thinking: true`, or a top-level / kwargs `reasoning_effort` of `low`, `medium`, `high` or `xhigh` (`high` renders `xhigh`; `none` = off; anything else is a 400) | on by default, `xhigh` |
| `preserve_thinking` | not passed to the template, so the template default applies (preserved) | on by default |
| `temperature`, `top_p`, `top_k`, `min_p` absent | filled from the upstream sampling row of the request's thinking mode (see Best Practices) | the rows are a recommendation |
| `temperature <= 0` sent | greedy | n/a |
| `presence_penalty` absent | 0, not the card's 1.5 (see Best Practices) | 1.5 in the non-thinking row |
| `max_tokens` | default 8,192, capped at 32,768 | up to 262,144 reasoning + 131,072 final recommended |
| images | `image_url` data URLs; each image resized into 1,024 to 1,280 visual tokens | URLs and data URLs, video too |
| video | not implemented | supported |

- Source: `engine/src/bin/serve.rs` request table and `map_reasoning_effort` at crow-nest `d47128c`; identical in v0.7.0 for these fields.

## Best Practices

Upstream, verbatim:

1. **Sampling Parameters**: We suggest using the following sets of sampling parameters:  
    
    - Thinking Mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
    - Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
    
    For supported frameworks, you can adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetition. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.

What crow-nest `serve` does with them:

- A request that leaves `temperature`, `top_p`, `top_k` and `min_p` out gets exactly these rows by its thinking mode: 1.0 / 0.95 / 20 / 0.0 thinking, 0.7 / 0.8 / 20 / 0.0 non-thinking (crow-nest #111). An absent `temperature` samples; greedy happens only when the client sends `temperature <= 0`.
- `presence_penalty` defaults to 0 in both modes, not 1.5: crow-nest applies it over the whole answer HF-style, and at 1.5 it pushed long agent answers full of hashes, hex and repeated code off tokens they needed (crow-nest #91, 2026-09-22). Send `presence_penalty: 1.5` to get the upstream non-thinking row in full. `repetition_penalty` 1.0 is the default already.
- Output length: `max_tokens` is capped at 32,768 per request, and the default context is 65,536 tokens, so the upstream advice of 262,144 + 131,072 output tokens cannot be followed in crow-nest.
- Ultra-long texts (upstream item 3) and long video (item 4) do not apply: YaRN is not available and video is not implemented.

## Converter command of record

```
converter --scales diag --diag-stats decode_out/p2-lh/diag-stats.json \
  --source-repo Qwen/Qwen3.8-27B models/Qwen3.8-27B converter/Qwen3.8-27B-CNQ4.5.cnq
```

- Reconstructed, not copied from a log: no generation log of this run was kept. The flags are those the crow-nest CHANGELOG (v0.7.0, "The 27B container carries calibrated scales") records; `--source-repo` is required by the converter; the revision `1d4bf0f2` in the index came from the Hugging Face cache in the model directory, the converter's default. Build time 659 s, 2026-09-26, container written 23:49:29 CEST.
- Converter source: `86cf4b8` (the `--scales diag` search of `adde3bd`, spread over every core; the build ran with that change before it was committed nine minutes later, per the commit times). The per-sub-block choice is independent of the core count, so the result does not depend on the machine.
- The calibration statistics file `decode_out/p2-lh/diag-stats.json` (35,697,752 B, sha256 `084a386563d344773507024ddbb1d86b80bdb1305023e34b322aac93321f0d3a`, per-column `sum x^2` over 491,520 tokens, built by `oracle/calib_qwen35_stats.py` + `oracle/export_diag_stats.py`) is NOT published, and neither is its calibration text: the maintainer's Crow agent sessions (5 files, 229,376 tokens), crow-nest and Crow source code (131,072), private Markdown notes (65,536) and crow-nest `docs/*.md` (65,536); calibration and evaluation rows share no 64-gram (checked 2026-09-26).

**Can a reader reproduce this container?** Not byte for byte. The converter and the two oracle scripts are public, but the statistics file and the calibration text are not, and `converter/README.md` at v0.7.0 does not document `--scales diag` yet (only the converter's `--help` text in `converter/src/main.rs` lists it). What a reader can do: (a) build the calibration-free `--scales mse` container from the same upstream shards (same size and layout, different scale bytes, KLD 0.290 instead of 0.223, see above), or (b) run `oracle/calib_qwen35_stats.py` + `oracle/export_diag_stats.py` over their own calibration text and pass that file to `--diag-stats` (a different container; its quality was not measured). The sha256 values in `SHA256SUMS` and the sidecar identify this build; the shard sha256 values in the index identify its input.

## Limitations and not measured

| item | status |
|---|---|
| downstream benchmarks (the upstream tables) on this quant | not measured |
| agentic quality (the crow-nest ten-task gate, long Crow sessions) on the 27B | not measured |
| KLD with `--scales diag` at long context | not measured; the 3,470 KLD rows are short and mid-length (code, prose up to 512 rows per book) |
| KLD after the mrope fix for image rows | not measured; the effect of the image positions before crow-nest #123 (fixed in v0.7.0) on answer quality was not measured either |
| vision quality against another engine (llama.cpp with the same projector) | not measured; P1 and P2 above missed their pre-registered bounds and were accepted by the maintainer |
| FP8 KV | fails the long-context bound (above); BF16 KV is the default |
| MTP | speed only, never output: the head's own KV is not saved with a parked conversation or a slot file, so drafts are poor right after an unpark or slot restore; VRAM +1.78 GiB at 100k context, over the pre-registered 1.7 GiB (crow-nest #95) |
| MTP C1 on one row | the MTP forward flipped one row of 297 at a top-2 margin of 0.0133 under `CROW_MMA=1` (attributed to the main model's activation quantization of the prompt path, not the head); output identity with plain decoding (C2) passed |
| GPUs other than the RTX 5090, Windows, cards under 32 GB | not measured; Ampere and Ada are not planned in crow-nest (crow-nest #12) |
| host RAM need of the 27B | not measured separately |
| contexts above 131,072 | not booted on 32 GB with BF16 KV (262,144 refused by the planner); YaRN not available |
| byte reproduction of this container | not possible without the unpublished statistics file (see Converter command of record) |
| trademark review of the repo name | not done |

## Provenance

| item | value |
|---|---|
| base model | `Qwen/Qwen3.8-27B` on Hugging Face, public, not gated (API, 2026-09-27) |
| revision of record | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, upstream `lastModified` 2026-08-14T15:00:01Z; the Hub's `main` on 2026-09-27 |
| input shards | 18 safetensors files, 55,563,006,776 B, each sha256 from the Hub's LFS record, stored in the container index (`model.source.shards`) |
| conversion | crow-nest converter `86cf4b8`, `--scales diag`, 2026-09-26, RTX 5090 machine above (the converter itself runs on the CPU) |
| acceptance | crow-nest `86cf4b8`: scale bytes against the simulation, KLD against BF16, sub-block goldens (`decode_out/p2-lh/accept-*.txt`) |
| projector | `unsloth/Qwen3.8-27B-GGUF` `mmproj-F16.gguf`, LFS sha256 `cbb841a9…b4e43e`, revision `4ca72078` (`lastModified` 2026-08-20); the file measured here was downloaded 2026-08-27 and equals that record |
| upload | Hub commit `63a3ea248f8ca3be987b0f9fc3b20f1eed9512c0`, 2026-09-27, container only; the Hub's LFS sha256 equals the local file |

## License

| item | value |
|---|---|
| licence | Apache License 2.0, the licence of `Qwen/Qwen3.8-27B` (card front matter `license: apache-2.0` and `LICENSE` at revision `1d4bf0f2`) |
| text | `LICENSE` next to this card: 11,544 B, 201 lines, sha256 `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`, byte-identical to the upstream file at `1d4bf0f2` (compared 2026-09-27); it carries "Copyright 2026 Alibaba Cloud" |
| modification notice | this container is a modified form of the upstream weights (Apache-2.0 section 4(b)): the listed tensors are quantized to NVFP4, `A_log` is widened to f32, and the vision tower is omitted, by the crow-nest converter on 2026-09-26 |
| projector | `mmproj-F16.gguf` is not part of this repository; `unsloth/Qwen3.8-27B-GGUF` declares `license: apache-2.0` (Hub API, 2026-09-27) |
| engine | crow-nest (engine and converter) is Apache-2.0, see its `LICENSE` |

## Citation

The upstream card's citation, verbatim:

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```
