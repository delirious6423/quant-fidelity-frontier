---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags: [gguf, rocmfpx, strix-halo, mixed-precision, quantization]
---

# DeepSeek-V4-Flash-0731 for Strix Halo

**82/92 quality in one 98.29 GB GGUF file.**

Built for 128 GB AMD Strix Halo systems. It loads unsplit on the Radeon 8060S
iGPU and needs no sidecar files.

## Quality

| Test | Score |
|---|---:|
| Full 92-question evaluation | **82/92** |
| COMPSEC-17 | **17/17** |

The full score matches the published reference. This file averages 2.766 bits
per model weight, about 4% less than the 2.88-bit reference.

None of the 92 test questions were used while preparing this file. The
published reference used 75 of them during its own preparation. Both results
used the same grader and reasoning allowance.

## Quality or speed

| Mode | Options | Decode speed | COMPSEC-17 |
|---|---|---:|---:|
| **Quality** (default) | No extra flags | 18.1 tok/s | **17/17** |
| **Faster** | `--ds4-expert-top-k 4 --ds4-fused-decode` | **22.3 tok/s** | 16/17 |

Quality mode is the recommended setting. Faster mode is 23% quicker but misses
one additional COMPSEC question. The full 82/92 evaluation was run only in
quality mode.

The current DSpark helper model makes this file slower overall, so it is not
recommended yet.

## Download

`DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf`

- Size: 98,294,917,184 bytes
- One file, no sidecars
- SHA-256: `7c0789d190fdd2acad93255825822ca276f29d13f9410f2ac65f5f7a542b0a38`

## Run

```bash
dflash_server DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf \
  --target-device hip:0 \
  --max-ctx 8192
```

Use the normal automatic memory settings on Strix Halo. If the machine also
has a discrete GPU, expose only the iGPU with `HIP_VISIBLE_DEVICES`.

Until support reaches the main dflash release, use the
`feat/qtype106-down-surface` branch of `GeometricAGI/lucebox-hub`.

Artifact and evaluation by Geometric-AI. Mirrored byte-for-byte by Lucebox.

## Vision: DeepSeek-V4-Flash-Vision-Exp

The same Strix Halo format for
[deepseek-ai/DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp),
which reads images as well as text. It is a separate model, not a patch on the
file above: every decoder weight differs from 0731, and it carries the image
router biases.

| File | Size | SHA-256 |
|---|---:|---|
| `DeepSeek-V4-Flash-Vision-Exp-ROCMFPX-MIX-STRIX.gguf` | 99,714,248,320 bytes | `7acd91500a4f6eb3e4892411c15a06bb0d0904cb02fe7696ce810874c14dfdae` |
| `DeepSeek-V4-Flash-Vision-Exp-mmproj-BF16.gguf` (vision projector) | 932,805,568 bytes | `a9a264120ea61d86aa58ba2d5d6f0a05c29629fc50a0116a525ad1455cb9f280` |

Built with the recipe of the 0731 file above: routed gate and up experts in
fp2, down experts in fp2 on the same 15 layers and fp3 elsewhere, dense
projections in ROCmFP4, the token embedding in Q6_K, codebooks inside the file.
Calibrated with the published llama.cpp importance matrix for this checkpoint
(812 chunks of 512 tokens), applied expert by expert.

### Quality

Measured on one Strix Halo, all six routed experts.

| Test | This file | Community Q2_K_S |
|---|---:|---:|
| KL divergence to the MXFP4 reference, 8,176 wikitext-2 tokens (mean) | **0.464** | 0.511 |
| Top-1 agreement with the reference | **78.4%** | 77.9% |
| Perplexity (reference 2.82) | **4.14** | 4.22 |
| AI2D, 100 questions | 84 | 85 |
| ChartQA, 120 questions (relaxed) | 94 | 98 |

The reference is ggml-org's MXFP4 conversion, which holds the model's native
FP4 experts exactly. Image answers on the sanity set and on requests with one
to four images were all correct. The 92-question text evaluation of the 0731
file has not been run on this one yet.

### Run

Image input needs the projector and a build that includes
[Luce-Org/lucebox#722](https://github.com/Luce-Org/lucebox/pull/722) until it
reaches the main release.

```bash
LUCE_DS4_SPEC=1 \
LUCE_DS4_DRAFT=/path/to/DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf \
LUCE_DS4_SPARSE_DECODE_FLASH=1 \
luce_server DeepSeek-V4-Flash-Vision-Exp-ROCMFPX-MIX-STRIX.gguf \
  --mmproj DeepSeek-V4-Flash-Vision-Exp-mmproj-BF16.gguf \
  --target-device hip:0 --max-ctx 131072 --chunk 8192 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --ds4-fused-decode --ds4-fused-verify-f16-kv \
  --ds4-expert-top-k 6 --ds4-prefill sparse
```

With the DSpark drafter from
[Lucebox/DeepSeek-V4-Flash-0731-DSpark-GGUF](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-DSpark-GGUF),
text decodes at 25 to 37 tok/s on 256-token answers (30 on average). Image
requests decode without the drafter, at about 22 tok/s. Send images as base64
JPEG or PNG in OpenAI chat-completion `image_url` parts, up to four per request.

Vision artifact and evaluation by Lucebox.
