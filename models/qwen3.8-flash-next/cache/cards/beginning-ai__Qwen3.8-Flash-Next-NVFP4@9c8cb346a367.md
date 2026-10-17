---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - fp8
  - modelopt
  - sglang
  - blackwell
---

# Qwen3.8-Flash-Next-NVFP4 (beginning-ai)

A mixed-precision quantization of
[**Qwen/Qwen3.8-Flash-Next**](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(base revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`), made to run on
**one 96 GB Blackwell GPU** with SGLang.

- 120 GiB on disk. The base is 336 GiB in BF16.
- About 78 GB of weights on the GPU. The 47.7 GiB n-gram PLE table stays in host
  RAM or on NVMe.
- On one RTX PRO 6000, single-stream decode runs at about **265 tok/s at 2K–64K
  context and about 235 tok/s at 200K**, with a **770K-token KV cache** and **524K
  context**.
- DeepSWE: **19/30 (63%)** on a 30-task subset. See [Results](#results).

Nothing is removed: the vision tower and the multi-token-prediction (MTP) draft
layer are kept. Every quantized weight was computed from the official BF16 base. No weight
was copied from another quantized release.

## How this differs from the base

| Part | Base | This checkpoint | Size here |
|---|---|---|---|
| Routed experts (48 layers × 512) `mlp.experts.*.{gate,up,down}_proj` | BF16 | **NVFP4** (4-bit weights and activations, group 16) | 68.0 GiB |
| Shared experts `mlp.shared_expert.*` | BF16 | **NVFP4 weights, BF16 activations** (`W4A16_NVFP4`, group 16) | 0.13 GiB |
| Linear-attention projections (36 layers) `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` | BF16 | **FP8**, 128×128 block scales, dynamic activations (`FP8_PB_WO`) | 1.93 GiB |
| Sparse-attention projections (12 layers) `self_attn.{q,k,v,o}_proj` | BF16 | **FP8**, 128×128 blocks (`FP8_PB_WO`) | 0.65 GiB |
| MTP draft layer routed experts `mtp.layers.0.mlp.experts` | BF16 | **NVFP4** (group 16) | 1.49 GiB (whole MTP layer) |
| N-gram PLE table | BF16 | **FP8**, copied byte-for-byte from [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | 47.7 GiB |
| Output head, embeddings, router, shared-expert gate, norms, linear-attention state parameters, hyper-connections, rest of the MTP layer, vision tower | BF16 | **BF16, unchanged** (byte-for-byte) | 4.6 GiB (plus the MTP layer's BF16 part) |

The full per-layer list is in `hf_quant_config.json` (`quant_algo: MIXED_PRECISION`).
Byte counts and calibration details are in `MANIFEST.json`.

### How it was made

- **Quantizer.** NVIDIA ModelOpt 0.47.0 tensor quantizers (`NVFP4QTensor`,
  `FP8QTensor`), applied to the official BF16 weights.
- **Shared scale for gate and up.** SGLang runs each expert's `gate_proj` and
  `up_proj` as one fused GEMM with one global scale. So both halves are quantized
  against one shared `weight_scale_2`, as ModelOpt does for fused layers. A build
  check confirms that all 24,624 fused pairs, plus the 512 MTP experts, have equal
  scales.
- **Activation scales (routed experts).** We calibrated one value per layer on
  128 × 512-token prefills of `abisee/cnn_dailymail`. Each expert's scale is the
  larger of our value and the per-expert value in
  [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).
  RadixArk's value is the larger one for `gate_proj`/`up_proj` in 4 of 48 layers,
  and for `down_proj` in 1,566 of 24,576 experts. So a scale is never smaller than
  either measurement, and it does not clip an activation that either one saw.
- **MTP experts.** Their activation scales come from 117K tokens of MTP-layer
  inputs captured on real coding-agent prefills.
- **No quantization of the output head.** It stays BF16, as in NVIDIA's and
  RadixArk's releases. A 4-bit head changed the top token at about 20% of
  positions in our tests.

## Results

### Test machine

| | |
|---|---|
| GPU | 1 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB, SM120), driver 580 |
| CPU | AMD Ryzen 9 9950X3D (16 cores) |
| Host RAM | 64 GB. The PLE table does not fit, so it streams from NVMe. |
| Storage | 1 × PCIe Gen4 NVMe (2 TB) for the weights, the PLE table and the KV disk cache |
| Software | SGLang at commit `f3fe753475` with our patch series (see [Serving](#serving)); FP8 KV cache; MTP speculative decoding (3 steps, 4 draft tokens) with rejection sampling and a 64K-token draft vocabulary |

### Coding agent: DeepSWE

[DeepSWE](https://deepswe.datacurve.ai/) v1.1, `mini-swe-agent` harness (no
history compaction), reasoning effort `medium`, 524K context, 2 tasks at a time,
**one attempt per task** on a fixed 30-task subset of the 113 tasks.

| | Score |
|---|---|
| This checkpoint (30-task subset, 1 attempt per task) | **19/30 = 63.3%** |
| Qwen's reported score for the BF16 base (113 tasks, high effort) | 58.7% |

These two scores are **not directly comparable**. Ours uses 30 of the 113 tasks
and one attempt per task, so the 95% range is about ±17 points. The official
leaderboard uses all 113 tasks with 4 attempts each. What the result shows is that
this quantization stays in the base model's range on long agentic coding tasks.
It does not show that it is better than the base.

The verifier also reports partial credit. On average, the agent's patches passed
95.7% of the tests for the new feature and 97.7% of the existing tests. One run
reached the 524K context limit after it had committed a passing solution.

<details>
<summary>Per-task results</summary>

| Task | Solved |
|---|---|
| abs-stepped-slices | 0 |
| arktype-json-schema-refs-dependencies | 0 |
| claude-code-by-agents-recursive-delegation | 0 |
| dynamodb-toolbox-lazy-recursive-schemas | 1 |
| effect-sse-httpapi-streaming | 0 |
| etree-xml-diff-patch | 1 |
| fd-deterministic-multi-key-sorting | 1 |
| go-genai-streamed-function-args | 1 |
| httpx-deterministic-cookie-store | 1 |
| ink-grid-box-layout | 0 |
| ipython-session-bundle-replay | 1 |
| koota-composite-trait-aspects | 0 |
| koota-deferred-mutation-buffer | 1 |
| koota-entity-snapshot-rollback | 1 |
| obsidian-linter-link-format-conversion | 0 |
| opa-template-string-reconstruction | 1 |
| pebble-durability-wait-apis | 0 |
| prometheus-transactional-reload-status | 1 |
| quill-shared-toolbar-focus | 0 |
| returns-validated-error-accumulation | 1 |
| scriggo-method-declarations | 1 |
| sql-formatter-bigquery-pipe-formatting | 1 |
| sqlfmt-create-table-ddl-formatting | 0 |
| sqlite-utils-safe-import-checkpoints | 1 |
| superjson-error-stack-serialization | 1 |
| testem-bail-on-test-failure | 0 |
| true-myth-iterable-collection-combinators | 1 |
| valibot-recursive-schema-composition | 1 |
| wasmi-trap-coredumps | 1 |
| yaegi-go-embed-directives | 1 |

</details>

Other checks:
- **Conversation replay:** 120 of 120 real agent turns, at 8K–200K context,
  produce well-formed tool calls.
- **Decode compared with prefill:** scoring the same tokens in decode mode and in
  prefill mode gives no severe differences over 6,357 tokens.

### Speed

All numbers are for one stream on the test machine above. The prompts are real
coding-agent prompts. Each run generates 1,024 tokens with temperature 1.0,
top_p 0.95 and top_k 20.

| Context | Decode (tok/s) |
|---|---|
| 2K | 268 |
| 64K | 262 |
| 200K | 233–240 |

Decode is the median of 4 seeds at 2K and 64K. At 200K it is the total over 4
prompts × 4 seeds, in two separate runs. MTP accepts about 3.0 tokens per
verification step.

| Prompt | Time to first token |
|---|---|
| 2K tokens, not cached | 0.17 s |
| 32K tokens, not cached | 2.5 s |
| 128K tokens, not cached | 11.0 s |
| Agent turn: 2K new tokens after a cached 128K prefix | 0.22 s |

| Capacity | |
|---|---|
| KV cache on the GPU (FP8) | 770,368 tokens |
| Context length | 524,288 tokens (YaRN factor 2 over the native 262,144) |

## Serving

The results above were measured with this exact SGLang build:

- **Commit [`93540334d2`](https://github.com/beginning-ai/sglang/commit/93540334d2ce687f0677bebb42dbd252b218e92f)**
  (tag [`qwen38-fn-nvfp4-2026-10-10`](https://github.com/beginning-ai/sglang/tree/qwen38-fn-nvfp4-2026-10-10),
  branch [`qwen38-flash-next-nvfp4`](https://github.com/beginning-ai/sglang/tree/qwen38-flash-next-nvfp4)
  in [beginning-ai/sglang](https://github.com/beginning-ai/sglang)).
- It is upstream SGLang
  [`f3fe753475`](https://github.com/sgl-project/sglang/commit/f3fe753475) plus 9
  commits
  ([compare](https://github.com/beginning-ai/sglang/compare/f3fe753475...qwen38-fn-nvfp4-2026-10-10)).

| Commit | What it adds |
|---|---|
| 1 | HiCache: Mamba state in `buffer_only` host memory mode (backport of the open upstream PR [sgl-project/sglang#36345](https://github.com/sgl-project/sglang/pull/36345)) |
| 2 | An NVMe backend that streams the PLE n-gram table from local storage |
| 3 | A working MoE runner for `modelopt_mixed` on SM120 GPUs |
| 4 | HiCache NIXL: Mamba state saved to disk |
| 5 | Responses API: reasoning and tool calls kept in the correct assistant turn |
| 6 | Decode: FP8 weights per row and small-batch GEMV kernels (`SGLANG_FP8_ROW_WEIGHTS`) |
| 7 | Small-batch NVFP4 MoE decode kernels (`SGLANG_NVFP4_DECODE_MOE`) |
| 8 | FlashInfer's GDN verify kernel for MTP on SM120 |
| 9 | Rejection sampling with an FR-Spec draft vocabulary (`--speculative-use-rejection-sampling`) |

```bash
pip install -e "git+https://github.com/beginning-ai/sglang@93540334d2ce687f0677bebb42dbd252b218e92f#egg=sglang&subdirectory=python"
```

We have not tested this checkpoint on unpatched SGLang.

```bash
python3 -m sglang.launch_server \
  --model-path beginning-ai/Qwen3.8-Flash-Next-NVFP4 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --context-length 262144 \
  --linear-attn-decode-backend flashinfer \
  --linear-attn-prefill-backend flashinfer \
  --mamba-radix-cache-strategy extra_buffer \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

Notes:
- **PLE table on NVMe.** If the 47.7 GiB PLE table does not fit in host RAM, as on
  our 64 GB machine, add `--ple-offload-backend nvme` (patched SGLang).
- **524K context.** Add `--context-length 524288` and set
  `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`. Also pass
  `--json-model-override-args` with a YaRN `rope_parameters` block: `rope_type:
  "yarn"`, `factor: 2.0`, `original_max_position_embeddings: 262144`, and the base
  config's `mrope_*` / `rope_theta` / `partial_rotary_factor` values.
- **Decode kernels.** `SGLANG_FP8_ROW_WEIGHTS=1` (FP8 output head and FP8 draft
  projections, converted at load time) and `SGLANG_NVFP4_DECODE_MOE=1` turn on the
  patched decode kernels.

## Files

| File | Content |
|---|---|
| `layer-*-experts-*.safetensors` | Routed experts (NVFP4) |
| `model-quant-00000.safetensors` | FP8 attention projections, NVFP4 shared experts |
| `model-mtp-experts-nvfp4.safetensors` | MTP draft-layer routed experts (NVFP4) |
| `model-ple-*.safetensors` | N-gram PLE table (FP8, from Qwen's FP8 release) |
| `model-bf16-*.safetensors` | All BF16 tensors |
| `MANIFEST.json` | Provenance, byte accounting, calibration report |
| `VALIDATION.txt` | Layout and fused-scale checks run at build time |

## License and credits

- **License:** the Qwen Community License 1.0, the same as the base model (see
  `LICENSE`).
- **Base model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
  by the Qwen team. The PLE table is from
  [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8).
- **Activation scales:** the per-expert activation scales that we used as a lower
  bound are from
  [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).
- **Quantization:** NVIDIA TensorRT Model Optimizer (ModelOpt).
