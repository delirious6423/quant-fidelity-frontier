---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- qwen
- moe
- lucebox
---

# Qwen3.8-Flash-Next GGUF for Lucebox

The Qwen3.8-Flash-Next files Lucebox serves: the UD-Q4_K_XL quantization and the multi-token-prediction (MTP) drafter used for speculative decoding. They are byte-identical copies of [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) at revision `766911a6b7369840a91dbcd95f9f997acaab6cd6`, kept here so Lucebox installs download from one place.

- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwen.
- Quantization and MTP export: Unsloth.

## Files

| File | Size (bytes) | sha256 |
| --- | --- | --- |
| `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | 10,946,624 | `4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082` |
| `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` | 49,859,583,136 | `3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9` |
| `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` | 49,376,141,504 | `56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3` |
| `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` | 12,087,983,520 | `753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a` |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2,786,568,256 | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |

## Performance on Lucebox

Measured with the Lucebox engine: UD-Q4_K_XL, greedy decoding with the MTP drafter, 131K context.

| | Strix Halo iGPU | Strix Halo + Radeon AI PRO R9700 |
| --- | --- | --- |
| Decode, short reply | 40 tok/s | 75 tok/s |
| Decode after a 16K prompt | 24-28 tok/s | 44-45 tok/s |
| Prefill, 16K prompt | 1,300 tok/s | 1,820 tok/s |
| Same 16K prompt sent again | | 2.7 s instead of 10.6 s (prefix cache) |

On the two-GPU box the R9700 keeps the dense weights, the KV cache, the drafter and the most-used experts, and the Strix Halo runs the other experts. Speculative decoding does not change the output: greedy text is identical with the drafter on or off.

## Use with the Lucebox engine

Point `luce_server` at the first shard; the other shards load from the same folder, and the engine finds the drafter in `MTP/` on its own. `--draft PATH` picks a drafter explicitly and `--verify-width 1` turns speculative decoding off.

```bash
luce_server UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
```

## License

Qwen community license 1.0; the full text is in [LICENSE](LICENSE) and applies to every file here.
