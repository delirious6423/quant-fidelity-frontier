---
license: mit
library_name: gguf
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
  - gguf
  - deepseek-v4
  - dspark
  - speculative-decoding
  - draft-model
  - auxiliary-model
---

# DeepSeek V4 Flash 0731 DSpark Drafter GGUF

This repository contains only the **DSpark auxiliary drafter** extracted from
DeepSeek V4 Flash 0731. It is not a standalone language model and does not
contain the target model weights.

The GGUF is built from the official
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
checkpoint using the standalone Python scripts included here. The complete
source manifest, build recipe, structural parser, numeric verifier, dependency
pin, and build results are published alongside the artifact.

## Artifact

| File | Bytes | SHA-256 |
|---|---:|---|
| `DeepSeek-V4-Flash-0731-DSpark-Drafter-MXFP4-Q8_0.gguf` | 10,897,110,272 | `5c8a1ce75e4797c170e22ac65080281a1b694c77f2e4fa7a703fbf09c5de0e33` |
| `DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0.gguf` | 6,971,241,728 | `a3f1156e1e4d2f46e1d4367b1bd09e6923adee61aecd41e880e233df33d30a7b` |
| `DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0-dflash.gguf` | 6,971,243,008 | `3e2be643b7881ac61e49c9907a963bdbbfcffe89c4d15c5f0e99e827e0305914` |

The first two artifacts use GGUF v3 with the custom architecture
`deepseek_v4_flash_dspark_draft` and contain the same 81 tensors. They differ
only in the encoding of the nine routed-expert tensors:

| Output type | Tensors | Transformation |
|---|---:|---|
| F32 | 45 | BF16 expanded exactly to F32, or source F32 bytes preserved |
| F16 | 2 | BF16 Markov heads rounded to F16 |
| Q8_0 | 25 | FP8 E4M3 weights with E8M0 block scales dequantized, then quantized to Q8_0 |
| MXFP4 | 9 | Routed experts in the `mxfp4-q8_0` artifact: FP4 nibbles and E8M0 scales repacked without requantization |
| Q2_K | 9 | Routed experts in the `q2_k-q8_0` artifact: source MXFP4 decoded, then quantized to Q2_K |

The MXFP4 transformation is bit-preserving. Q2_K, Q8_0, and F16
transformations are lossy and are checked against explicit numeric error
bounds. This repository does not claim that either complete conversion is
lossless.

## Pinned source

- Repository: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Input: only safetensors shards 46, 47, and 48
- Total source size: 10,863,342,388 bytes

Exact file sizes and SHA-256 values are in
[`manifest/source.json`](manifest/source.json). The downloader and converter
refuse any source that does not match them.

## Reproduce

The reference build used Linux aarch64, Python 3.14.6, NumPy 2.5.1, uv 0.11.29,
and Hugging Face CLI 1.24.0. The NumPy wheel is pinned by hash in
`requirements-linux-aarch64-py314.lock`. The upstream checkpoint is public, so
a Hugging Face token is not required for source download.

```bash
uv venv --python 3.14.6 .venv
uv pip install --python .venv/bin/python \
  --require-hashes --only-binary=:all: \
  -r requirements-linux-aarch64-py314.lock

.venv/bin/python scripts/download_sources.py --destination sources
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/reproduce.py \
  --sources sources \
  --recipe q2_k-q8_0 \
  --manifest-dir manifest \
  --repeat-check
sha256sum -c SHA256SUMS
```

Use `--recipe mxfp4-q8_0` to reproduce the MXFP4/Q8_0 artifact instead.

`--repeat-check` performs a second clean conversion and requires its complete
GGUF SHA-256 to match the verified first build byte for byte. The converter
memory-maps the source shards and processes FP8 tensors in bounded bands. It
does not require PyTorch, CUDA, safetensors, a GGUF package, or a GPU.

## Verification

The included verifier checks pinned source hashes and tensor inventories, GGUF
structure and metadata, exact F32/F16 bytes, exact MXFP4 repacking, Q2_K
reconstruction error, and Q8_0 encoding plus reconstruction error against a
fresh source decode.

Reference-build results:

| Check | Result |
|---|---:|
| Tensor count | 81 |
| Exact F32/F16 tensors | 47 |
| Exact MXFP4 tensors | 9 |
| Q2_K tensors | 9 |
| Maximum Q2_K block-relative error | 0.49072265625 |
| Q2_K limit | 0.5 |
| Maximum Q8_0 block-relative error | 0.00396728515625 |
| Q8_0 limit | 0.01 |
| Repeat build | byte-identical |

Machine-readable observations are in the MXFP4/Q8_0
[`validation`](manifest/validation.json) and [`build`](manifest/build.json)
manifests, and the Q2_K/Q8_0
[`validation`](manifest/validation-q2_k-q8_0.json) and
[`build`](manifest/build-q2_k-q8_0.json) manifests.

## Standardized dflash migration candidate

`DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0-dflash.gguf` is the
llama.cpp-standardized migration candidate for ns4 issue #205. It uses GGUF
v3, `general.architecture=dflash`, canonical llama.cpp metadata and tensor
names, and canonical one-based `dflash.target_layers=[41, 42, 43]`.

The schema is pinned to official ggml-org/llama.cpp commit
`3581ba0cf591b3f772fbb002de0f70e294bc0396`, incorporating PR #25784 and
the separate-DSpark converter in PR #26452. The official loader at that commit
recognized the artifact as `dflash`, instantiated all 81 tensors, and accepted
the complete layout. Its exact type inventory is F32 45, F16 2, Q8_0 25,
Q2_K 9, and BF16 0.

The standardized file preserves every tensor payload from the existing custom
Q2_K/Q8_0 artifact byte for byte through an explicit checked name mapping. It
is not a new quantization. The legacy artifacts and their stable tags remain
supported and are not deprecated pending Studio runtime validation in ns4
issue #205. BF16 is intentionally excluded from this artifact and remains a
separate later phase.

Reproduce and verify the standardized candidate with:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/dflash.py \
  --sources sources \
  --manifest-dir manifest \
  --repeat-check \
  --legacy-reference DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0.gguf
sha256sum -c SHA256SUMS
```

Dedicated build, validation, payload-comparison, schema, and official-loader
manifests are published under `manifest/`.

## Compatibility and limitations

- This is an auxiliary drafter, not a complete checkpoint.
- The custom files require the checked-in `deepseek_v4_flash_dspark_draft` and
  `dspark.*` dialect. The standardized file requires llama.cpp `dflash` support.
- Official llama.cpp model-loading acceptance is verified for the standardized
  file. No Studio runtime validation or universal compatibility claim is made.
- The drafter must be paired with a compatible DeepSeek V4 Flash 0731 target
  model. No universal quality, acceptance-rate, or speed claim is made across
  target quantizations or runtimes.
- This is a community conversion, not an official DeepSeek release or
  endorsement.

## DSpark configuration

The pinned source configuration defines block size 5, target layers 40, 41,
and 42, Markov rank 256, hidden size 4096, and vocabulary size 129,280. These
values are embedded in the GGUF metadata and validated during the build.

## License and references

The source-derived weights remain under DeepSeek's MIT terms in
[`LICENSE`](LICENSE). The newly authored conversion and verification scripts
are under [`LICENSE.code`](LICENSE.code). The pinned checkpoint's
[upstream license](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/9e165c30e2704aec5d9d593cce3eebd58bbef1cb/LICENSE)
and the DS4/GGML attribution in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) provide the applicable notices.

- [Official DeepSeek V4 Flash 0731 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [DeepSeek V4 technical report](https://arxiv.org/abs/2606.19348)
- [DeepSpec](https://github.com/deepseek-ai/DeepSpec)
