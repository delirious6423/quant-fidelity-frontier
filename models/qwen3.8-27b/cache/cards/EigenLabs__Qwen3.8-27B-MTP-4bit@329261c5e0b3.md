---
library_name: mlx
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
tags:
- mlx
- mtp
- qwen3.8
- speculative-decoding
---

# EigenLabs/Qwen3.8-27B-MTP-4bit

This is a **non-standalone, one-layer native MTP proposal head** for the pinned target backbone [`EigenLabs/Qwen3.8-27B-4bit@301e9e2767fd0efcfab7883004720ba3c9a552a1`](https://huggingface.co/EigenLabs/Qwen3.8-27B-4bit/tree/301e9e2767fd0efcfab7883004720ba3c9a552a1). It proposes tokens only. The target backbone remains authoritative for language tensors, embeddings, the LM head, token acceptance, and final output.

The official model lineage is [`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0). This artifact was derived locally from the exact EigenLabs BF16 MTP artifact [`EigenLabs/Qwen3.8-27B-MTP-bf16@26a328e070875b0314d652a039b6b59902690f03`](https://huggingface.co/EigenLabs/Qwen3.8-27B-MTP-bf16/tree/26a328e070875b0314d652a039b6b59902690f03), never by copying a third-party quantized checkpoint.

## Artifact contract

- `model_type`: `qwen3_5_mtp`
- MTP layers: 1 (`layers.0` only)
- `block_size`: 3
- 31 tensor keys, all bare (no `mtp.` prefix)
- Eight 2-D matrices stored as canonical `.weight`, `.scales`, and `.biases` tensors
- Seven 1-D normalization tensors retained in BF16
- No tokenizer, target embeddings, target LM head, target-backbone weights, or vision weights

This repository must be paired with the immutable target revision above. Updating or downloading this repository alone does not register it in a runtime catalog or enable speculative decoding.

## Quantization

Quantization used MLX 0.31.2 directly:

```python
weight, scales, biases = mlx.core.quantize(
    source_bfloat16, group_size=64, bits=4, mode="affine"
)
```

The packed weights are `uint32`; affine scales and biases remain BF16. MLX-LM 0.31.3, safetensors 0.7.0, and huggingface-hub 1.7.2 were installed in the conversion environment. Each matrix was independently dequantized with `mlx.core.dequantize(..., group_size=64, bits=4, mode="affine", dtype=mlx.float32)` and compared with its exact BF16 source.

| Matrix | Source shape | Packed shape | Scale/bias shape | Max abs error | Mean abs error |
|---|---:|---:|---:|---:|---:|
| `fc.weight` | 5120×10240 | 5120×1280 | 5120×160 | 0.041015625 | 0.0008614016 |
| `layers.0.mlp.down_proj.weight` | 5120×17408 | 5120×2176 | 5120×272 | 0.0546875 | 0.0010076930 |
| `layers.0.mlp.gate_proj.weight` | 17408×5120 | 17408×640 | 17408×80 | 0.017578125 | 0.0009606532 |
| `layers.0.mlp.up_proj.weight` | 17408×5120 | 17408×640 | 17408×80 | 0.0341796875 | 0.0010548841 |
| `layers.0.self_attn.k_proj.weight` | 1024×5120 | 1024×640 | 1024×80 | 0.0166015625 | 0.0012780037 |
| `layers.0.self_attn.o_proj.weight` | 5120×6144 | 5120×768 | 5120×96 | 0.09765625 | 0.0016031306 |
| `layers.0.self_attn.q_proj.weight` | 12288×5120 | 12288×640 | 12288×80 | 0.02734375 | 0.0011431416 |
| `layers.0.self_attn.v_proj.weight` | 1024×5120 | 1024×640 | 1024×80 | 0.021484375 | 0.0023364164 |

All source, packed, scale, bias, and dequantized values passed finite-value checks. After the local derivation was complete, all 31 local tensor payloads were compared with the independent pinned reference [`mlx-community/Qwen3.8-27B-MTP-4bit@b643c01b6d3b094e325edb6ebd832e16c486c575`](https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit/tree/b643c01b6d3b094e325edb6ebd832e16c486c575); names, dtypes, shapes, and raw tensor bytes matched exactly.

These checks establish artifact structure and quantization fidelity only. They are not end-to-end generation, acceptance-rate, quality, or speedup claims.
