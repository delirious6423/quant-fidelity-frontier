---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- ds4
- apple-metal
- quantized
- qwen
- imatrix
- image-text-to-text
---

# Qwen3.8-Flash-Next DS4 Q2 (imatrix)

The default Q2 release is now **44.81 GB (41.73 GiB)**, including MTP.
It uses **IQ2_XXS gate/up experts and padded Q2_K down projections**, saving
**5.16 GiB (11%)** over the previous MXFP4-down build with essentially
unchanged measured speed. This is the smaller option for memory-constrained
Macs, including 64 GB systems.

The **external Q4_1 PLE sidecar is required** and reused from the
[Q4 repository](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4).
Both files together occupy **76.81 GB (71.53 GiB) on disk**. The PLE table is
CPU-mapped and demand-paged; resident pages still consume RAM. Start with 8K
context and a 1,024-token prefill chunk on a 64 GB Mac. Context buffers, host
allocations, and other applications also affect memory use. The measurements
below ran on an M3 Ultra with 512 GiB; they do not establish a physical 64 GB fit.

## Model structure

- **48 trunk layers, 512 routed experts per layer:** the 96 gate/up tensors
  use imatrix-calibrated **IQ2_XXS**, unchanged from the previous release.
- **48 trunk down tensors:** imatrix-calibrated **Q2_K**, quantized directly
  from the original BF16 weights. Each 640-column logical input row is
  zero-padded to **768 physical columns** for three 256-value Q2_K blocks.
  Metal kernels use the padded weight stride and only the 640 real inputs;
  the model's architecture and activation width remain unchanged.
- **Dense GDN/QSA projections, shared experts and output:** **Q8_0**,
  preserved from the previous model.
- **Token embeddings:** **BF16**. Routers, norms, hyper-connections and
  control tensors retain their existing F16/F32 precision.
- **Embedded MTP block:** all 32 tensors preserved, including **Q4_K gate/up**
  and **MXFP4 down** experts. Ordinary decoding and optional `--mtp`
  speculation use the same GGUF.
- **Required PLE n-gram sidecar:** **Q4_1**, approximately 32.00 GB / 29.80 GiB
  on disk, shared with the Q4 release. It is not included in the main GGUF.

Only 48 tensor payloads changed. The other **1,207 tensors are byte-identical**
to the previous release, and all **1,255 output tensors** were read back and
hash-verified. The `.gguf.json` file records output hashes, down-weight source
hashes, calibration provenance, and the physical padding geometry.

This is a DS4-targeted `qwen4exp` GGUF. It requires the padded-down runtime
support in **commit `5bd8796` or later** on the
[`qwen3.8-flash-next` branch](https://github.com/ivanfioravanti/ds4-metal/tree/qwen3.8-flash-next).
Older builds reject the new down-tensor layout. Compatibility with other
runners is not claimed.

## Download and run

Update the DS4 checkout to the current `qwen3.8-flash-next` branch, then build
with `make -j8 ds4 ds4-server`. Run from that checkout's root:

```sh
./download_model.sh qwen38-q2
./ds4 --metal \
  --ple gguf/Qwen3.8-Flash-Next-PLE-Q4_1.gguf \
  --ctx 8192 --prefill-chunk 1024
```

The script downloads this model and the PLE sidecar, then updates
`ds4flash.gguf` to the new main model. For a custom directory, use:

```sh
QWEN_Q2_DIR="$HOME/models/qwen38-ds4-iq2"
hf download ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2 \
  Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf \
  --local-dir "$QWEN_Q2_DIR"
hf download ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4 \
  Qwen3.8-Flash-Next-PLE-Q4_1.gguf --local-dir "$QWEN_Q2_DIR"

./ds4 --metal \
  -m "$QWEN_Q2_DIR/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf" \
  --ple "$QWEN_Q2_DIR/Qwen3.8-Flash-Next-PLE-Q4_1.gguf" \
  --ctx 8192 --prefill-chunk 1024
```

An existing Q4_1 PLE sidecar can be reused without downloading it again.
Add `--mtp` for speculation; use `--mtp-exact-sampling` when preserving the
target sampling distribution is required. The older main-model filename has
been replaced, so update any explicit `-m` paths.

For optional vision, run `./download_model.sh qwen38-vision` and add
`--vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf`. The encoder is separate
and unchanged. Earlier vision smoke scores belong to the superseded model;
the tests below cover text, kernel correctness, and a short MTP check.

## Measured tradeoff

On **99 BF16-reference continuations / 2,376 tokens**, the old MXFP4-down
model versus this Q2_K-down release measured:

- BF16 top-token agreement: **91.33% → 90.32%**.
- Target NLL: **0.290735 → 0.303570** (4.4% higher; lower is better).
- First-token matches: **69/99 → 65/99**.

On the matched **12-question hard-smoke suite**, with a 2,048-token answer
budget, temperature 0, seed 123, 32K context and no retries or MTP:

- Previous release: **6 correct, 0 wrong, 6 incomplete**.
- Current release: **5 correct, 0 wrong, 7 incomplete**.

Incomplete means no gradeable final answer. These limited tests show extra
quantization drift and do not establish equal reasoning quality.

Two benchmark passes in opposite model order used identical prompts,
1,024-token prefill chunks and 128 teacher-forced decode tokens per frontier:

- **4K:** decode **47.63 → 47.31 tok/s**; prefill **962.72 → 973.17 tok/s**.
- **32K:** decode **47.09 → 46.94 tok/s**; prefill **926.06 → 935.95 tok/s**.

The 32K prefill processes the additional 28,672 tokens after 4K. A separate
128-token greedy smoke test produced identical output with and without MTP.
Kernel tests cover padded decode and prefill, and the one-token CPU/GPU check
agrees on top-1. See [full results](qwen38-q2down.md) and
[machine-readable measurements](qwen38-q2down-results.json).

## Provenance and integrity

Source: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
BF16 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
Calibration uses the pinned
[Unsloth imatrix](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
revision `c8b5954a88c2775c546b92593eda40ea041d3176`, SHA-256
`a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154`.
It contains 45 chunks of 18,432 tokens. Sixteen gate/up and eight down entries
with zero activation counts use the deterministic per-expert weight-energy
fallback; the remaining entries use measured activation importance.

Build tool and recipe:
[`qwen4_iq2.py --projection down`](https://github.com/ivanfioravanti/ds4-metal/blob/5bd8796/gguf-tools/README.md#experimental-padded-q2_k-down-projections).

Main GGUF SHA-256:
`341c8d79468384a05e22998ae834a489145f6c385e276dcf79170a6b0d0ffd2d`.
`SHA256SUMS` covers the published release files except itself and the Hub's
`.gitattributes`. The previous weights and their text/vision reports remain
available in [repository history](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2/tree/672c52bbea7865352c8f0aa766c43939acf0f0d5).

Derived from Qwen's original model, with calibration from Unsloth and the
DS4 runtime/build tools. This derivative carries the **Qwen Community
License 1.0**; the original copyright and full terms are in [LICENSE](LICENSE).
