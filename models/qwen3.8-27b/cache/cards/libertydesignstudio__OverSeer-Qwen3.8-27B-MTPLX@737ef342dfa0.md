---
language:
- en
library_name: mlx
pipeline_tag: text-generation
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- mtplx
- qwen3.8
- speculative-decoding
- apple-silicon
---

# OverSeer Qwen 3.8 27B MTPLX

A mixed-precision MLX build of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with its native MTP draft head preserved for speculative decoding through MTPLX.

The release target is a 24 GB Apple Silicon workstation: approximately the memory footprint of a Q3_K_M GGUF while materially improving decode speed.

## Measured result

Tested on a base Apple M4 with 24 GB unified memory, macOS 26.5.2, MLX 0.32.0, mlx-lm 0.31.3, and MTPLX 2.7.2.

| Measurement | Result |
|---|---:|
| Loaded model weights | 13.4 GB |
| Forge D2 verification | 13.79 tok/s |
| OpenAI server, two 300-token prompts | 11.23–14.01 tok/s |
| Long reasoning samples | 10.95–13.08 tok/s |
| Same-artifact AR baseline | 7.41 tok/s |
| Prior LM Studio Ridge baseline | 5.26–5.79 tok/s |

The server measurements used `performance-cold`, MTP depth 2, one active request, a one-entry/1 GB session bank, and extended background warmup disabled. Generation speed excludes prompt prefill; end-to-end latency still depends on prompt length.

## Quality checks

The release candidate was checked on three paired tasks:

- Levenshtein edit-distance implementation: executable code passed five deterministic cases; stated `O(m·n)` time and space.
- TCP versus UDP explanation: correct connection, reliability, ordering, and analogy requirements.
- First-occurrence binary-search repair: executable code passed seven boundary and duplicate cases.

Known limitation: an optional illustrative edit-distance table contained incorrect intermediate cells, although both generated implementations, the complexity analysis, and the final distance were correct. The debugging task needed a 4,096-token reasoning allowance to reach its correct final answer; 1,800 tokens ended during reasoning. This is a small release check, not a broad quality benchmark.

## Quantization

The body defaults to 2-bit affine quantization with higher precision on quality-sensitive paths:

- embeddings and LM head: 6-bit
- linear-attention QKV/Z/out: 4-bit
- linear-attention A/B: 8-bit
- full-attention Q/K/V: 5-bit; output: 6-bit
- all MLP down projections: 3-bit
- first and last four MLP gate/up projections: 3-bit
- native MTP sidecar: BF16

The complete machine-readable recipe is in `quantization_recipe.json`.

## Forge verification patch

The included MTPLX 2.7.2 patch is needed to reproduce Forge's MTP-depth verification for this 2-bit body. The recommended server intentionally leaves `MTPLX_NAX_VERIFY` unset: enabling that experimental dispatcher reduced decode on an 8.1K-token OMP prompt from `10.23` to `2.94 tok/s`. Stock MLX quantized matmul is the validated server lane.

```bash
python -m pip install 'mtplx==2.7.2'
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
patch -p1 -d "$SITE_PACKAGES" < mtplx-2.7.2-q2-verify.patch
```

The patch adds numerically checked 2-bit affine verify kernels for MTP row counts 2 through 6. Synthetic BF16 comparisons against stock MLX measured maximum absolute differences from `0.00225` to `0.00384` across those row counts.

## Recommended text server

```bash
MTPLX_SESSION_BANK_MAX_ENTRIES=1 \
MTPLX_SESSION_BANK_MAX_BYTES=1G \
MTPLX_SESSION_BANK_PER_SESSION_BYTES=1G \
MTPLX_SESSION_BLOCK_PREFIX_RESTORE=0 \
MTPLX_MEMORY_PRESSURE_GUARD=0 \
MTPLX_WARMUP_EXTENDED=0 \
mtplx serve \
  --model ./OverSeer-Qwen3.8-27B-MTPLX \
  --profile performance-cold \
  --depth 2 \
  --scheduler-mode serial \
  --batching-preset solo \
  --max-active-requests 1
```

`performance-cold` is the measured fast lane for text requests under 8K prompt tokens (10.95–14.01 tok/s). Native OpenAI function calling works on a compact 604-token direct request (one valid call at 10.96 tok/s), but the full OMP tool/rule prompt expands to roughly 23.7K tokens: `performance-cold` correctly blocks that unsafe prefill, while `sustained` did not complete the end-to-end OMP agent turn within 15 minutes. This release is therefore recommended as a fast text/coding model, not as the default full-tool agent. Upstream model metadata retains 262K context; the qualified local OMP text route advertises 16K.

## Files

- mixed-precision MLX language-model shards
- `mtp.safetensors`: native BF16 MTP head
- `model-vision.safetensors`: restored vision tower
- `mtplx_runtime.json`: runtime contract and measured verification evidence
- `quantization_recipe.json`: reproducible Forge recipe
- `mtplx-2.7.2-q2-verify.patch`: optional Forge/tuning verification patch

## License

Apache 2.0, matching the upstream Qwen3.8-27B release. This derivative is not affiliated with or endorsed by the Qwen team.
