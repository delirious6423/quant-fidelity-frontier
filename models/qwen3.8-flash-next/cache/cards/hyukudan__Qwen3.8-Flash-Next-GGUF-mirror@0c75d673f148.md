---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- gguf
- qwen
- unsloth
- mirror
---

# Qwen3.8-Flash-Next GGUF (Unsloth UD-Q4_K_XL + MTP), mirror

Mirror of [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)@[`38bb39ee97821de2c9009abb7e93950eec396e66`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66) for reproducibility of [github.com/hyukudan/strix-halo-engine-lab](https://github.com/hyukudan/strix-halo-engine-lab).

The files are exact copies of that upstream revision: same paths, same bytes. They were not re-quantized or modified. All credit belongs to the original authors below; please use and cite the upstream repositories. This mirror only pins the revision our engines were built and measured against.

## Original authors

- [Unsloth](https://huggingface.co/unsloth): UD-Q4_K_XL dynamic quantization and the MTP GGUF sidecars
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): original model Qwen3.8-Flash-Next (Qwen Community License 1.0)
- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)@`de4b8e4d43b917e7706784d8bb445c9af86a3540`

## License

Qwen Community License 1.0. The upstream repo declares `qwen-community-1.0` but ships no LICENSE file, so `LICENSE` here is copied verbatim from the base model repo [Qwen/Qwen3.8-Flash-Next@de4b8e4d](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE). Note its conditions: a separate license from Qwen is needed before commercial use by a "Model as a Service" or "AI Work Assistant" business, and the model name must be displayed above 100M MAU or US$20M monthly revenue.

## How our engines use it

- The engine's production Qwen target is a single file made from these four shards with `llama-gguf-split --merge` (llama.cpp 552f18f9): `Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf`, 111,334,654,400 bytes, sha256 `8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2`. The merge changes only the container: all 1224 tensors keep their names, shapes, types and bytes. We mirror the original shards, not our merged file. For this upload the shards were re-created from our merged file with `llama-gguf-split --split --split-max-size 50G --no-tensor-first-split`; all four came out byte-identical to Unsloth's (sha256 equal to the upstream LFS hashes below), which also shows the merge was lossless.
- `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` is the MTP draft head for speculative decoding (windows 2/4/8, bit-identical to serial greedy).
- See the [engine README](https://github.com/hyukudan/strix-halo-engine-lab#qwen38-flash-next).

## Files

8 files, 114,121,291,364 bytes (114.12 GB). `README.md` is this card; the upstream model card is kept verbatim as `README.upstream.md`. Large-file sha256 values equal the upstream LFS hashes and were recomputed from our local copies before upload (any mismatch would have stopped the upload). `SHA256SUMS` lists every file.

| File | Bytes | sha256 |
|---|---:|---|
| `LICENSE` (added by this mirror) | 3,235 | `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08` |
| `MTP/README.md` | 5,906 | `18a8c2d894bb0b63e31cf8aec0fceeee30564fa3e8115fee81e94ec73027fd63` |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2,786,568,256 | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |
| `README.upstream.md` (upstream `README.md`) | 59,183 | `9538227ff778ba7d8ba61b42cde3927487bf4c9c647ce18ac31d08a0b42998e6` |
| `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | 10,946,624 | `4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082` |
| `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` | 49,859,583,136 | `3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9` |
| `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` | 49,376,141,504 | `56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3` |
| `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` | 12,087,983,520 | `753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a` |

Only the UD-Q4_K_XL shards and the shared-Q8_0 MTP sidecar are mirrored; all other quantizations are on the upstream repo.
