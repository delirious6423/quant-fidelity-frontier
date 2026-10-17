---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
  - gguf
  - qwen3_5
  - multimodal
  - mtp
  - rocm
  - rocmfpx
  - strix-halo
  - gfx1151
  - int4
---

# Qwen3.8-27B ROCmI4 + embedded MTP (GGUF)

This private repository contains the tested Qwen3.8-27B artifact used with the
experimental ROCmI4/IU4 path in
[charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX). It is a
conversion/quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
not a newly trained model.

The main GGUF includes the model's one-layer NextN/MTP head. You do **not**
need a separate draft model. `mmproj-BF16.gguf` is included for image/video
input and is optional for text-only use.

## Files and identity

| File | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `Qwen3.8-27B-Q4_0_ROCMI4.gguf` | 27B language model, ROCmI4 weights, embedded MTP | 14,534,384,384 | `ad51ecbee55930ababc629c377943a7bbe50f8aafcf32340c7cdae6fa6b7bb3c` |
| `mmproj-BF16.gguf` | BF16 multimodal projector | 931,146,432 | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` |

Verified GGUF metadata:

- architecture: `qwen35`
- parameters: approximately 27.32B
- native context metadata: 262,144 tokens
- tensors: 866
- quantization type: `Q4_0_ROCMI4` (GGML type 108)
- embedded MTP: `qwen35.nextn_predict_layers = 1`
- MTP tensors: `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`, and
  `nextn.shared_head_norm`

## Required ROCmFPX build

Stock llama.cpp builds are not expected to understand this custom ROCmI4 type
or the qualified IU4/MTP path. Use ROCmFPX commit
[`c49ebdbd5c9f01ec242369f9e7f7967855f80cba`](https://github.com/charlie12345/ROCmFPX/commit/c49ebdbd5c9f01ec242369f9e7f7967855f80cba)
or a later compatible commit.

The fast W4A4 path is opt-in and currently qualified for AMD Strix Halo
`gfx1151`. Build HIP-only: a combined HIP/Vulkan scheduler can attempt to send
ROCmI4 tensors to Vulkan, which has no shader for this custom type.

```bash
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX
git checkout c49ebdbd5c9f01ec242369f9e7f7967855f80cba

cmake -S . -B build-rocmi4-w4a4 \
  -DGGML_HIP=ON \
  -DGGML_VULKAN=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DGGML_HIP_ROCMI4_W4A4=ON \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build-rocmi4-w4a4 \
  --target llama-cli llama-server llama-bench -j 16
```

If ROCm does not identify Strix Halo natively:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```

At startup, verify that the log says `ROCmI4 W4A4: enabled`. If the option is
off, ROCmFPX retains the exact int8 MMQ fallback. W4A4 is faster but adds lossy
four-bit activation quantization.

## Download from this private repository

You must be granted access and logged in:

```bash
hf auth login
hf download cafonez/Qwen3.8-27B-ROCmI4-MTP-GGUF \
  --local-dir ./Qwen3.8-27B-ROCmI4-MTP-GGUF
```

Verify the download:

```bash
cd Qwen3.8-27B-ROCmI4-MTP-GGUF
sha256sum -c checksums.sha256
```

## Quick MTP command-line test

`llama-cli` is interactive, so feed `/dev/null`, use `-st`, and cap `-n` in
automated tests:

```bash
timeout 180 ./build-rocmi4-w4a4/bin/llama-cli \
  -m ./Qwen3.8-27B-ROCmI4-MTP-GGUF/Qwen3.8-27B-Q4_0_ROCMI4.gguf \
  -dev ROCm0 -ngl 999 -c 4096 -b 512 -ub 256 -t 16 -fa on \
  -ctk f16 -ctv f16 -st --temp 0 -n 128 \
  --spec-type draft-mtp --spec-mtp-strict-qwen \
  --spec-draft-device ROCm0 --spec-draft-ngl all \
  --spec-draft-type-k f16 --spec-draft-type-v f16 \
  --spec-draft-n-max 16 --spec-draft-n-min 0 \
  --spec-draft-p-min 0.60 --spec-draft-backend-sampling \
  -p "Write a function to reverse a linked list." </dev/null
```

Read throughput from the final `Generation: ... t/s` line, not from generated
text.

## OpenAI-compatible server

Text-only users may omit `--mmproj`. Reduce `-c` if the host does not have
enough unified memory for a 262K target and draft context.

```bash
./build-rocmi4-w4a4/bin/llama-server \
  -m ./Qwen3.8-27B-ROCmI4-MTP-GGUF/Qwen3.8-27B-Q4_0_ROCMI4.gguf \
  --mmproj ./Qwen3.8-27B-ROCmI4-MTP-GGUF/mmproj-BF16.gguf \
  --host 127.0.0.1 --port 8116 --alias qwen38-27b-rocmi4 \
  -dev ROCm0 -ngl 999 -np 1 -c 262144 \
  -b 512 -ub 256 -t 16 -tb 32 -fa on \
  -ctk f16 -ctv f16 --jinja \
  --spec-type draft-mtp --spec-mtp-strict-qwen \
  --spec-draft-device ROCm0 --spec-draft-ngl all \
  --spec-draft-type-k f16 --spec-draft-type-v f16 \
  --spec-draft-n-max 16 --spec-draft-n-min 0 \
  --spec-draft-p-min 0.60 --spec-draft-backend-sampling
```

Test the endpoint:

```bash
curl http://127.0.0.1:8116/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen38-27b-rocmi4",
    "messages": [{"role": "user", "content": "Say hello."}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

### Tuning MTP

The qualified PR #94 profile is `n_max=16`, `p_min=0.60`, strict Qwen MTP,
and backend sampling. It is the profile used by the live service from which
this artifact was verified. MTP gains depend strongly on content: predictable
code, structured data, and lists tend to accept longer drafts than creative
prose. If acceptance or stability is poor, lower `--spec-draft-n-max` to 4 or
6 and remeasure the same workload.

Strict MTP preserves the selected target path's greedy decisions; it does not
remove the approximation introduced by optional W4A4 activation quantization.

## What ROCmI4, INT4, IU4, and W4A4 mean

- **ROCmI4** is the GGUF weight format: signed four-bit weight codes packed two
  per byte with block scales.
- **INT4** is the generic four-bit integer width; it does not identify a file
  format or kernel by itself.
- **IU4** is AMD's `v_wmma_i32_16x16x16_iu4` matrix instruction spelling on
  `gfx1151`. It is an execution path, not another model format.
- **W4A4** means four-bit weights and four-bit activations during accelerated
  matrix multiplication. The weight file does not change, but activations are
  quantized to a signed four-bit grid.
- **INT8** has a wider integer range. The exact ROCmI4 MMQ fallback expands the
  packed weights and uses int8 computation.
- **FP8/FP4** are floating-point encodings with exponent and mantissa behavior;
  they are not interchangeable with ROCmI4/INT4.

Plain one-token decode remains memory-bandwidth-bound and uses MMVQ. MTP
proposes multiple tokens and verifies them as a batch, allowing the W4A4 IU4
MMQ kernel to improve end-to-end decode when enough drafts are accepted.

## Measured results

Measurements below are specific to a Ryzen AI MAX+ 395 / Radeon 8060S
(`gfx1151`) system and the pinned ROCmFPX build. They are not guarantees for
other prompts or hosts.

| Qualification | Result |
|---|---:|
| 10-task HumanEval pilot, exact ROCmI4 MMQ, strict MTP-16 | 41.63 tok/s mean |
| 10-task HumanEval pilot, W4A4 IU4, strict MTP-16 | **49.40 tok/s mean** |
| Full 164-task HumanEval, W4A4 IU4 | **44.39 tok/s mean**, 45.23 tok/s median |
| Full HumanEval / HumanEval+ pass@1 | 94.5% / 91.5% |
| Non-speculative `tg128` | about 13.8 tok/s |

The local live-service smoke for this exact GGUF on 2026-08-22 returned the
requested control string with embedded MTP active. Longer recent workloads
ranged from about 29 to 43 tok/s depending on draft acceptance and prompt
shape.

Quality checks measured perplexity 6.3068 for this ROCmI4 artifact versus
5.9455 for a local Q4_K_M reference (about +6.1%). Separately, enabling the
experimental W4A4 activation path increased a 25-chunk perplexity sample by
about 5.4% over the exact ROCmI4 path. Treat W4A4 as an explicit speed/quality
tradeoff.

## Limitations

- Native IU4 W4A4 acceleration is currently qualified only on `gfx1151`.
- `Q4_0_ROCMI4` and the MTP flags require the ROCmFPX fork; stock llama.cpp may
  reject the file or omit the optimized path.
- The 262,144-token value is model metadata and a tested server allocation,
  not a claim that every task maintains equal quality across the full window.
- Multimodal use requires `mmproj-BF16.gguf` and significantly more memory.
- Quantized models can differ from the upstream BF16 model. Validate quality,
  safety, and task-specific correctness before production use.
- Use the model according to the upstream model card and Apache-2.0 license.

## Provenance and license

This distribution is licensed under **Apache License 2.0**, matching the base
model. The complete, unmodified upstream license text is included as
`LICENSE`. Attribution and modification details are also recorded in
`NOTICE`.

The GGUF metadata identifies `Qwen/Qwen3.8-27B` as the base model, Unsloth as
the original GGUF quantizer, and Apache-2.0 as the license. The local
ROCmI4 file was produced by requantizing the corresponding ROCmFP8 GGUF with
ROCmFPX while preserving the embedded NextN/MTP tensors. The multimodal
projector matches the one published in
[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).

Source revisions used for this release:

- base model: `Qwen/Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- GGUF/projector lineage: `unsloth/Qwen3.8-27B-GGUF` at
  `4ca720788d1e01f1bff70c033e0d0028fd02e502`
- conversion/runtime implementation: `charlie12345/ROCmFPX` at
  `c49ebdbd5c9f01ec242369f9e7f7967855f80cba`

Changes from the upstream distribution are limited to GGUF conversion and
ROCmI4 quantization, preservation of the embedded NextN/MTP tensors, and this
ROCmFPX-specific documentation/launcher. No claim is made that Qwen or
Unsloth endorses this derivative. Upstream copyrights and attribution remain
in force.

See the upstream [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
for intended use, capabilities, and model-level limitations. See the
[ROCmFPX ROCmI4 documentation](https://github.com/charlie12345/ROCmFPX/blob/c49ebdbd5c9f01ec242369f9e7f7967855f80cba/ggml/rocmfpx/ROCMI4.md)
for implementation details and rollback instructions.
