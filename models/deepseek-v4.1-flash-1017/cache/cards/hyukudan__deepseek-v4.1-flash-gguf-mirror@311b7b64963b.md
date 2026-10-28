---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- gguf
- deepseek
- mirror
---

# DeepSeek V4.1 Flash 1017 GGUF (kyuz0), mirror

Mirror of [kyuz0/deepseek-v4.1-flash-gguf](https://huggingface.co/kyuz0/deepseek-v4.1-flash-gguf)@[`20408f699619df7071aa7cda6034e9cef5bf804b`](https://huggingface.co/kyuz0/deepseek-v4.1-flash-gguf/tree/20408f699619df7071aa7cda6034e9cef5bf804b) for reproducibility of [github.com/hyukudan/strix-halo-engine-lab](https://github.com/hyukudan/strix-halo-engine-lab).

The files are exact copies of that upstream revision: same paths, same bytes. They were not re-quantized or modified. All credit belongs to the original authors below; please use and cite the upstream repositories. This mirror only pins the revision our engines were built and measured against.

## This mirror vs. our modified builds

This repository contains **only unmodified upstream files** (the subset our engines use; see below). Our own changes are never mixed into a mirror. They are published as separate repositories:

- [hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6): our re-quantized dense weights (attention, shared expert, output head), an overlay the engine applies on top of the Q2 GGUF in this mirror. Not a standalone model.
- [hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3): our trellis-quantized packs used by the two-node Strix Halo route.

Engine and measurements: [strix-halo-engine-lab](https://github.com/hyukudan/strix-halo-engine-lab).

## Original authors

- [kyuz0](https://huggingface.co/kyuz0): GGUF conversion, imatrix and quantization (Q2 experts, Q8_0 dense) and the DSpark MXFP4 GGUF
- [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): original model DeepSeek-V4.1-Flash-1017 (MIT)
- Base model: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)@`1bae0e1088993b9b82de08f59de5be5a34e42170`

## License

MIT. The upstream `LICENSE` file (DeepSeek's MIT notice, "Copyright (c) 2023 DeepSeek") is included verbatim; the upstream repo states that the upstream MIT license applies to the GGUF files.

## How our engines use it

- `DeepSeek-V4.1-Flash-1017-Q2.gguf` is the target for every DeepSeek route in the engine: the two-node Strix Halo pair (routed experts from this file; dense weights replaced by our [6.5-bit overlay](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6) and [trellis packs](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3)), the RTX PRO pipeline split, and the standalone RTX PRO 6000 route (IQ2_XXS gate/up, Q2_K down, Q8_0 dense, all from this file). Its sha256 is pinned in the engine (`deepseek41_model.hpp`).
- `DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf` is the DSpark speculative-decoding drafter on all routes (target-verified, output identical to greedy).
- See the [engine README](https://github.com/hyukudan/strix-halo-engine-lab#deepseek-v41-flash) and [production defaults](https://github.com/hyukudan/strix-halo-engine-lab/blob/main/docs/models/deepseek-v4.1/production-defaults.md).

## Files

8 files, 373,680,008,833 bytes (373.68 GB). `README.md` is this card; the upstream model card is kept verbatim as `README.upstream.md`. Large-file sha256 values equal the upstream LFS hashes and were recomputed from our local copies before upload (any mismatch would have stopped the upload). `SHA256SUMS` lists every file.

| File | Bytes | sha256 |
|---|---:|---|
| `DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf` | 7,966,294,016 | `d3efe0b1f6b6fea54f1c58c5cbeb3fded5967b15b64476aed00c95806ad3e3e9` |
| `DeepSeek-V4.1-Flash-1017-Q2.gguf` | 365,713,686,528 | `5365629079214ff28becbbab7c0900ef9865b79c0f0c64d951924bb362ae5b67` |
| `LICENSE` | 1,084 | `f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040` |
| `README.upstream.md` (upstream `README.md`) | 11,686 | `f735bc321810cb8fc12a2f2ba21e4afba5e290d8026900388bdd80cf097b8b49` |
| `calibration/1017-provenance.patch` | 2,016 | `a55092d8f700c20518c2b4f977ac1eb509cf98156b37916bdc51e935934572da` |
| `calibration/DeepSeek-V4.1-Flash-1017-imatrix.json` | 8,485 | `d1957870122919c042db91b1d6fb9ecf39dcdef809e568320b07de7a8c367212` |
| `calibration/README.md` | 3,617 | `6727d55dd5ca5bdab3f094622ca0bdc000c2e2f3b75491fb5a818d4eed860916` |
| `compatibility/vision-checkpoint-1017.patch` | 1,401 | `c4713aea20d9054144d85640b3e553cb29779002f036aa0af1e7d3e9c603d915` |

Only the files our engines use are mirrored. Not mirrored: the non-1017 `DeepSeek-V4.1-Flash-Q2.gguf`, the Q4 parts, the vision GGUFs and `calibration/DeepSeek-V4.1-Flash-1017-imatrix.dat`; get them from the upstream repo.
