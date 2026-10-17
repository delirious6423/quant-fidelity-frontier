---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
license: apache-2.0
tags:
  - mlx
  - omlx
  - qwen4_exp
  - qwen3.8-flash-next
  - moe
  - quantized
  - ternary
  - imatrix
  - apple-silicon
---

> ⚠️ **Runtime requirement.** This checkpoint only runs on the
> [Qwen48 oMLX fork](https://github.com/fuutott/omlx): the ternary expert banks need its native
> Bonsai Metal kernel. Stock oMLX, mlx-lm and mlx-vlm cannot load it. See [Running it](#running-it).

# Qwen3.8-Flash-Next for a 48 GB Mac (MLX, ternary experts + Q3 down, imatrix)

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
that runs on a **48 GB Apple Silicon Mac** with the
[Qwen48 oMLX fork](https://github.com/fuutott/omlx).
The 360 GB source model is brought down to about **35.6 GiB resident** by storing the routed
experts' gate/up projections as ternary weights (oMLX's Bonsai T5 packing, ~1.875 bits/weight),
the routed down projections as affine Q3, and the 53.6 GiB PLE n-gram embedding table as Q8 that
is memory-mapped from SSD instead of loaded. All low-bit tensors are fitted with the
[Unsloth importance matrix](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) for this model.

Measured on an M3 Max 48 GB: **38 tok/s decode, 210–380 tok/s prefill**, 64K-token chat requests
completed, and a 129K-token prefill verified by oMLX's automatic context sizing. Quality sits
between the previous ternary builds and Unsloth's Q4_K_XL GGUF, which does not fit this machine
(numbers below). This is an experimental research build, not an official Qwen or oMLX release.

## What is in the checkpoint

| Component | Precision |
| --- | --- |
| Routed expert gate/up (48 layers) | Bonsai T5 ternary, group 128, imatrix-weighted least-squares scale fit |
| Routed expert down | Affine Q3, group 128, imatrix-weighted range search |
| PLE n-gram embeddings (128 shards, 53.6 GiB) | Affine Q8, group 32, mmapped from SSD at run time |
| Shared experts / shared-expert gate | Affine Q8, group 128 / 64 |
| Token embeddings, LM head | Affine Q6, group 64 |
| Attention and DeltaNet projections | Affine Q5, group 64 (QSA `o_proj`: Q4, group 64) |
| Other eligible matrices | Affine Q4, group 64 |
| Vision tower Linear layers | Affine Q8, group 64 |
| Routers, norms, convolutions, recurrent state, vision non-Linear | BF16 (source precision) |
| Multi-token prediction (MTP) head | Removed |

Every affine tensor whose input channels could be mapped onto the GGUF importance matrix is
fitted with it (routed and shared experts, attention, DeltaNet, hyper-connection projections),
including the DeltaNet `out_proj`, whose imatrix entry llama.cpp stores in a tiled V-head order
that the converter undoes. The T5 packing stores five ternary digits per byte with a BF16 scale
per group of 128; it is a packing format, not the T5 language-model family, and it is not
Tencent AngelSlim's STQ1_0 layout, although the weighted-least-squares fit was inspired by it.

| Size | Bytes |
| --- | ---: |
| All safetensors shards (131 files) | 97,059,902,320 (90.4 GiB) |
| of which PLE n-gram shards, SSD-mmapped, not resident | 57,600,276,480 (53.6 GiB) |
| Weights that must be resident | 39,459,625,840 (36.8 GiB) on disk, about 35.6 GiB in memory after load |

Architecture is unchanged from the source: `qwen4_exp` (hyper-connections, Gated DeltaNet and
QSA attention layers, 512-expert MoE with top-10 routing, one PLE n-gram embedding layer,
vision tower). It is not Qwen3.5; loaders that treat it as Qwen3.5 will fail.

## Measured on the target machine

MacBook Pro M3 Max, 48 GB unified memory, macOS 27.0, fork commit
`c85c35ff` built with `OMLX_WITH_CUSTOM_KERNEL=1`, MLX 0.32.2, one model loaded, one request at
a time, 16-bit KV cache, MTP and thinking off, greedy decoding, context 8192 unless stated.

| Request | Prefill tok/s | Decode tok/s | Physical peak |
| --- | ---: | ---: | ---: |
| 20-prompt pilot set, 8K context | 211–215 | 38.4 | 41.5 GiB |
| 4,096-token prompt, 512 output | 381 | 36.0 | – |
| 32,768-token prompt, 256 output | 272 | 33.3 | 41.4 GiB |
| 65,536-token prompt, 256 output | 211 | 30.7 | 42.3 GiB |
| 98,304-token prompt | rejected by oMLX's prefill memory guard (43.05 + 0.41 GiB > 43.32 GiB cap) | – | 43.4 GiB |
| 129,024-token prefill, automatic context sizing benchmark | 61 | – | 43.5 GiB (whole benchmark) |

- MLX allocator peak at 8K context: 39.45 GiB. Loading pages in 1.8–2.5 GiB of swap once;
  no swap growth was observed during generation at 8K, 32K or 64K.
- The 129K figure is oMLX's own context-sizing benchmark (one output token); the largest
  chat-style request completed with a real answer was 64K + 256 tokens. Treat ~64K as the
  practical context on 48 GB and the 129K result as a prefill-capacity measurement.
- Decode speed on this hardware is bound by kernel-launch overhead rather than memory
  bandwidth, so the ternary experts buy memory, not speed.

## Quality

Three independent measurements, all with thinking disabled and greedy decoding.

### KL divergence against a Q8_0 teacher

wikitext-2 test set, 40 windows of 2048 tokens (40,920 scored positions), llama.cpp's
`--kl-divergence` convention. Teacher: `lmstudio-community/Qwen3.8-Flash-Next-GGUF` Q8_0
(perplexity 3.297). The MLX checkpoints were scored under MLX 0.32.2 on CUDA after losslessly
re-packing the ternary groups to affine Q2, so all rows use the same stock kernels.

| Checkpoint | Resident weights | Mean KLD | Median KLD | Top-1 agreement | PPL (× teacher) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **This checkpoint**: T5 + imatrix gate/up, Q3 down | 35.6 GiB | **0.489** | 0.187 | 76.1 % | 4.35 (× 1.32) |
| Affine Q2 + imatrix, all expert projections (stock-oMLX variant) | 35.6 GiB | 0.477 | 0.186 | 76.4 % | 4.31 (× 1.31) |
| T5 + imatrix gate/up, Q2 down | 30.9 GiB | 0.573 | 0.237 | 74.0 % | 4.67 (× 1.42) |
| T5 weight-only gate/up (no imatrix), Q2 down | 30.9 GiB | 0.690 | 0.327 | 71.3 % | 5.27 (× 1.60) |
| Unsloth UD-Q4_K_XL GGUF (reference, does not fit 48 GB) | – | 0.036 | 0.007 | 93.9 % | 3.31 (× 1.005) |

At this memory budget the total number of expert bits sets the KLD, not where they are placed:
this checkpoint and the all-affine Q2 build are a statistical tie (per-position difference 0.012,
standard error 0.003). The imatrix alone is worth 0.12 KLD over the weight-only ternary fit.

### Knowledge and reasoning, 700 fixed cases

Stratified random subsets (seed 42) of MMLU (300 items, 5-shot), GSM8K (100), TruthfulQA MC1
(200) and CMMLU (100, 0-shot), served over the OpenAI-compatible API with 8192-token context,
128-token answers for multiple choice and 512 for GSM8K. Truncated, unparseable and errored
answers count as wrong. The Q4_K_XL column was served by LM Studio (llama.cpp) on the identical cases.

| Suite | n | This checkpoint | Previous ternary build (weight-only, Q2 down) | Unsloth UD-Q4_K_XL |
| --- | ---: | ---: | ---: | ---: |
| MMLU (5-shot) | 300 | 249 (83.0 %) | 237 (79.0 %) | 269 (89.7 %) |
| GSM8K | 100 | 90 (90.0 %) | 85 (85.0 %) | 92 (92.0 %) |
| TruthfulQA MC1 | 200 | 180 (90.0 %) | 178 (89.0 %) | 192 (96.0 %) |
| CMMLU (0-shot) | 100 | 68 (68.0 %) | 78 (78.0 %) | 81 (81.0 %) |
| **Total** | 700 | **587 (83.9 %)** | 578 (82.6 %) | 634 (90.6 %) |

### Code generation (pass@1)

oMLX's bundled HumanEval and MBPP tasks, sandboxed execution, one attempt per problem.

| Benchmark | n | This checkpoint | Unsloth UD-Q4_K_XL |
| --- | ---: | ---: | ---: |
| HumanEval | 164 | 152 (92.7 %) | 157 (95.7 %) |
| MBPP | 500 | 390 (78.0 %) | 429 (85.8 %) |

Paired by problem: HumanEval both pass 150, only this 2, only Q4 7, neither 5; MBPP both 381,
only this 9, only Q4 48, neither 62. Failures are mostly genuinely wrong code, not formatting.

## Known issues and limitations

- **Chinese instruction following regressed.** On CMMLU the model often writes an option-by-option
  analysis in Chinese instead of answering with the requested letter; 24 of the 100 CMMLU cases
  hit the 128-token cap that way (the previous ternary build did this 6 times). The analyses are
  usually correct, so it is a compliance loss rather than a knowledge loss, but it costs real
  points on strict-format Chinese tasks. The likely cause is the English/code-centric importance
  matrix now weighting the dense projections; a bake with the imatrix restricted to the experts
  is the planned bisect.
- The KL divergence was measured on English text only. Chinese, other languages and code were
  not scored that way.
- Vision inputs were not tested. The vision tower is present (its Linear layers at Q8), but no
  image prompt has been validated against the source model.
- Thinking mode was disabled for every measurement here; tool calling and multi-request
  batching were not exercised.
- Loading on a 48 GB machine briefly swaps 1.8–2.5 GiB. Keep other memory users closed, keep the
  memory guard on, and start with an 8K context before growing it.
- The multi-token-prediction head is not included; speculative decoding gave little on this
  hardware and cost several GiB.

## Running it

You need Apple Silicon, macOS 15 or newer, full Xcode (the Metal toolchain, not just Command
Line Tools), `uv` and the Hugging Face CLI. Build the fork in its own checkout and environment;
it is not a drop-in for an existing oMLX install.

```bash
git clone https://github.com/fuutott/omlx.git omlx-qwen48
cd omlx-qwen48
uv venv --python 3.12 .venv
OMLX_WITH_CUSTOM_KERNEL=1 uv pip install -e .
uv run python -c "from omlx.custom_kernels import native_kernel_status; print(native_kernel_status())"
# the 'bonsai' entry must report available: True
```

Download the checkpoint outside the Git checkout, verify it, and serve it:

```bash
hf download fuutott/Qwen3.8-Flash-Next-MLX-t5-imatrix-q3down-ple8 \
  --local-dir /path/to/checkpoints/Qwen3.8-Flash-Next-MLX-t5-imatrix-q3down-ple8

uv run python tools/quantize_qwen4_flash_next_t5.py \
  --verify-only /path/to/checkpoints/Qwen3.8-Flash-Next-MLX-t5-imatrix-q3down-ple8

uv run omlx serve --model-dir /path/to/checkpoints --memory-guard safe --memory-guard-gb 48
```

Put the checkpoint on a fast internal SSD: the 53.6 GiB PLE table is read through `mmap` on
every token. In the oMLX admin UI confirm the model is detected as `qwen4_exp` and that Qwen4
PLE SSD offload is enabled (it is forced automatically at a 48 GiB ceiling). Keep MTP off, load
one model at a time, and use context 8192 for the first run. The checkpoint validates with the
converter's `--verify-only` (schema, packing, layout and the SSD-offload representation).

## Reproducing the bake

The full recipe, environment lock and converter live in the fork's README under
[How to bake the cake](https://github.com/fuutott/omlx#how-to-bake-the-cake).
This checkpoint was produced on Windows with CUDA from the pinned source revision and the Unsloth
importance matrix (`imatrix_unsloth.gguf_file`, SHA-256
`a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154`) with:

```text
tools/quantize_qwen4_flash_next_t5.py --model <Qwen3.8-Flash-Next @ de4b8e4d> --output <dir>
  --expert-format t5 --t5-fitter prefix --expert-down-bits 3
  --imatrix <imatrix_unsloth.gguf_file> --imatrix-strict --imatrix-scope safe --allow-experimental-imatrix
  --clip-search --ple-bits 8 --vision-bits 8 --device cuda:0 --chunk-rows 4096
```

If you want the same size class without the fork's kernels, `--expert-format affine` produces the
all-affine Q2 variant from the KLD table, which loads on stock oMLX.

## Files

- `model-*.safetensors` (131 shards) and `model.safetensors.index.json`
- `config.json`, `generation_config.json`, tokenizer and preprocessor files, `chat_template.jinja`
  from the source model; `config.json` carries the `omlx_t5` loader marker and declares no MTP layers
- `omlx_conversion.json`: recipe, source identity, imatrix statistics and the per-tensor precision record
- `omlx_conversion_manifest.json`: the converter's identity manifest (source, code and environment
  hashes, chunking) whose fingerprint is stamped into every shard's metadata
- `SHA256SUMS.txt`: hashes of every file in this repository; check with `shasum -a 256 -c SHA256SUMS.txt`
- `LICENSE`: the source model's Apache-2.0 license

Source: `Qwen/Qwen3.8-Flash-Next` at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
Converter: fork commit `c85c35ffbf24f83a085538266f3942eb68c43c1a`.

## License and credits

Apache-2.0, inherited from the source model; the base model's card and license conditions apply
to these weights. Thanks to the Qwen team for the model, to [oMLX](https://github.com/jundot/omlx)
for the runtime, the Bonsai ternary format and the Qwen4 PLE SSD offload this build depends on, to
[Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) for the importance matrix, to
[LM Studio](https://huggingface.co/lmstudio-community/Qwen3.8-Flash-Next-GGUF) for the Q8_0 teacher
GGUF, to [llama.cpp](https://github.com/ggml-org/llama.cpp) for the KL-divergence tooling, and to
Tencent's AngelSlim work for the weighted ternary fitting idea.
