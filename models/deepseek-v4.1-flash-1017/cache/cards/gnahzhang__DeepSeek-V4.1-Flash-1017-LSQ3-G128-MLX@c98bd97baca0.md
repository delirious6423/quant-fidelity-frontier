---
license: mit
library_name: mlx
pipeline_tag: image-text-to-text
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- mlx
- deepseek
- deepseek-v4.1
- quantized
- mixed-precision
- lsq
- vision
- tool-calling
---

# DeepSeek-V4.1-Flash-1017 — LSQ3-G128 MLX

A mixed-precision MLX conversion of [DeepSeek-AI's DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170), retaining the vision encoder, Engram embedding tables, and all three MTP/DSpark draft stages. The source is fixed to revision **`1bae0e1088993b9b82de08f59de5be5a34e42170`**.

This repository uses the custom `omlx_deepseek_v41` checkpoint format and ships a pinned [MLX/oMLX runtime recipe](runtime-bundle/README.md). Use that recipe for the qualified serving profile. Stock Transformers and `mlx_lm` loading are not established for this format.

**No standardized task-quality benchmark or perplexity evaluation has been run for this 1017 quantization.** Upstream benchmark scores are not results for this derivative.

## Parameters and storage

The checkpoint retains **763,205,315,794 logical model parameters** (763.205B). This count excludes quantization scales and biases, accounts for packed FP4/3-bit representations, and was independently checked against both the current source headers and the converted tensor shapes.

| Parameter family | Logical parameters |
| :--- | ---: |
| Language backbone, including its non-embedding Engram projections | 551,880,835,184 |
| Engram embedding tables | 196,613,849,600 |
| Three retained MTP/DSpark stages | 14,225,362,530 |
| Vision encoder and aligner | 485,268,480 |
| **Total** | **763,205,315,794** |

| Files | Exact bytes | GiB |
| :--- | ---: | ---: |
| 128 converted safetensors shards | 239,961,225,877 | 223.481307 |
| 2 original Engram shards | 203,073,077,576 | 189.126541 |
| **All 130 checkpoint safetensors files** | **443,034,303,453** | **412.607848** |
| 120 exact-metadata safetensors files | 8,503,892,112 | 7.919867 |
| **Checkpoint plus exact-metadata tensor files** | **451,538,195,565** | **420.527715** |

GiB means 2³⁰ bytes. These totals cover the listed tensor files; configuration, tokenizer, metadata sidecars, provenance, and runtime files add further bytes. Reserve additional storage for downloading, native builds, and runtime caches.

The two Engram shards preserve the original source files, including FP8 embedding values and their scales. They are accessed through the runtime's SSD Engram path. Parameter counts, download size, resident tensor payload, and measured process memory are different quantities.

## Mixed-precision policy

| Component | Representation |
| :--- | :--- |
| Main routed experts: 120 projections across 40 layers | Affine **3-bit / group 128**, BF16 scales and biases |
| Token embedding and output head | Affine **8-bit / group 64** |
| 312 native FP8 projections | Source values/scales losslessly repacked as MLX MXFP8, group 32 |
| 9 MTP routed-expert projections | Source values/scales losslessly repacked as MLX MXFP4, group 32 |
| Engram embeddings | Original F8_E4M3 values and F8_E8M0 scales |
| Other dense tensors | BF16/F32 as appropriate for the port |

Here **LSQ** means least-squares optimization of weight reconstruction error. The exporter starts from standard MLX affine3 quantization, searches seven clipping seeds, and refines scale/bias for up to five iterations. A group keeps its baseline quantization if no candidate improves reconstruction error. The main routed experts are dequantized from the official FP4 representation to BF16 before this conversion. This procedure uses no activation calibration dataset and performs no model training.

All 120 reconstruction spot checks passed against standard MLX affine3. Each check uses the first 16 rows of the first expert in a projection; these checks do not measure downstream model accuracy.

The `exact-metadata/` overlay dictionary-encodes the routed experts' scale/bias metadata. All 120 projections passed full exact round-trip/readback checks. Quantization codes are unchanged. This reduces resident tensor payload by **8,483,059,248 bytes (7.900465 GiB)**, from 223.481091 GiB to 215.580627 GiB, before runtime allocations. The published launcher requires this overlay.

## Run on Apple Silicon

Read [runtime-bundle/README.md](runtime-bundle/README.md) for the pinned toolchain, native build requirements, wired-memory configuration, authentication, and profiles. The recipe reconstructs the custom MLX/oMLX sources in a separate work directory.

Download the complete repository, including `engram/`, `exact-metadata/`, and `runtime-bundle/`:

```sh
hf download gnahzhang/DeepSeek-V4.1-Flash-1017-LSQ3-G128-MLX \
  --local-dir ./DeepSeek-V4.1-Flash-1017-LSQ3-G128-MLX
cd DeepSeek-V4.1-Flash-1017-LSQ3-G128-MLX

export MODEL_DIR="$PWD"
export WORK_DIR="$HOME/flash-1017-runtime"
python3 runtime-bundle/runtime.py --help
uv run --no-project --python 3.13.15 python runtime-bundle/runtime.py prepare \
  --work-dir "$WORK_DIR"
"$WORK_DIR/venv/bin/python" runtime-bundle/runtime.py test \
  --work-dir "$WORK_DIR"
```

After following the runtime guide's wired-memory setup:

```sh
"$WORK_DIR/venv/bin/python" runtime-bundle/runtime.py serve \
  --work-dir "$WORK_DIR" --model-dir "$MODEL_DIR" \
  --profile cached --port 8011
```

The cached profile provides the local API model alias `deepseek-v41-lsq`, DSpark, SSD Engram offload, exact metadata, eight request slots, a 3 GiB persistent prefix-cache budget, and a configured 240 GiB process limit. The guide explains the locally generated API key.

The required `attention_output_quantization: true` metadata selects the 1017 attention-output arithmetic. The bundled runtime also includes the corresponding tool/image encoding and completion-whitespace updates. An older runtime can omit required 1017 behavior.

## Validation and limits

The original local production runtime passed functional and serving checks on an **Apple M5 Ultra with 256 GiB memory**, macOS 27.0.1, Python 3.13.15, and custom MLX `0.32.3.dev20261001+59d600b5e`.

- **Functional:** arithmetic, forced function call with correct arguments, tool-result continuation, a synthetic color-block image, and a 128-token DSpark generation sample.
- **Context:** one 32,768-token raw input completed successfully with one output token. The architecture config specifies 1,048,576 positions; full-context quality, retrieval, and memory behavior were not validated. The local acceptance profile configured 16,384 maximum output tokens.
- **Prefix cache:** 14 scenarios before restart and three after restart passed API-content and generated-token-ID parity checks. Nine prefix checkpoints were restored.
- **Concurrency:** eight identical streams passed before restart, each using 8,242 input tokens, 8,192 cached prefix tokens, and 64 output tokens. After restart, **one restored stream** checked parity.
- **Memory:** the highest sampled physical footprint was **225.43 GiB**, with samples every 0.5 seconds. Peaks between samples were not established. This observation applies to the stated custom runtime, dictionary overlay, SSD offload, and cache profile; it is not a universal minimum-memory guarantee.

Separately, the portable runtime bundle passed **68 focused CPU and small Metal GPU regression checks** after reconstructing the pinned patch series. A fresh native rebuild and full-checkpoint replay of the relocated public bundle were not performed. Its source/regression validation and the original local full-model qualification are distinct evidence.

See [evidence/model-card-facts.json](evidence/model-card-facts.json) for the factual ledger. Stage-specific conversion and metadata records can retain earlier `inference_validated: false` flags; later final local acceptance passed. Those historical flags do not constitute a new evaluation of this public package. The exact-metadata external index is a custom runtime artifact, not a standalone standard llama.cpp model.

## License and credits

The original model and weights are released by **DeepSeek-AI under the [MIT License](LICENSE)**. This repository is an independent quantized conversion and retains that license for the model weights. Credit also goes to the tacos8me DeepSeek V4.1 MLX/oMLX port and the [public optimization recipe](https://github.com/gnahz/omlx-flash-optimizations).

Runtime wrapper code and documentation have separate notices; see [runtime-bundle/THIRD_PARTY_NOTICES.md](runtime-bundle/THIRD_PARTY_NOTICES.md) and its included licenses.
