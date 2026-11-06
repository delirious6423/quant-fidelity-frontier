---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w4a16
- compressed-tensors
- vllm
---

# Qwen3.8-27B-W4A16-HGPTQ

All-W4A16 Cholesky GPTQ (`group_size=64`, `actorder="static"`) with online block-Hadamard input rotation (`head_dim=64`, `QuIPModifier` / $R_4$-style + `HadaCore`) of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) across **Text (`400` linears), Native MTP (`7` linears), and Vision (`110` linears)** (`17.33 GiB` total safetensors payload).

- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (`8c0b3d30f3ae96f5c7f774de704bf0ada83e422a`)
- **Text & MTP calibration:** 999 samples × 4096 tokens from [nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) pinned in [warp-3090-quant-lab-frozen-data](https://huggingface.co/datasets/nchapman/warp-3090-quant-lab-frozen-data); Native MTP calibrated on captured post-norm target hidden states and shifted token embeddings
- **Vision calibration:** 64 ChartQA + TextVQA calibration images (strictly disjoint from the 48-image evaluation fixture); `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ (`vision_fc2_channel`) + `head_dim=16` online Hadamard (`v_vision_fc2`)
- **Chat template:** [froggeric Qwen-Fixed-Chat-Templates (v22.5)](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) ships as `chat_template.jinja` (`sha256: e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2`) — the artifact's only template
- **Default sampling** (`generation_config.json`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`)
- **Recipe and tooling:** [warp-3090-model-lab](https://github.com/nchapman/warp-3090-model-lab)

`linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `embed_tokens` / `lm_head`, `mtp.fc`, `visual.patch_embed.proj`, norms, and biases stay BF16.

## Measured Fidelity vs. `Qwen3.8-27B-W4A16-GPTQ` (`g128-h0`)

| Metric | `Qwen3.8-27B-W4A16-GPTQ` (`g128-h0`) | **`Qwen3.8-27B-W4A16-HGPTQ` (this repo)** | Change |
|---|---:|---:|---:|
| Safetensors payload | 18.117 GiB (19,452,784,112 B) | **17.334 GiB (18,612,467,944 B)** | **−801.4 MiB (−4.3%)** |
| `probe-v4` weightedKL (`cascade90-ml10`) ↓ | 0.1369 *(legacy: 0.1264)* | **0.115968** *(legacy: 0.1047)* | **−15.3%** |
| `probe-v4` token mean KL / p99.9 ↓ | 0.357 / 24.66 | **0.301646 / 24.2052** | **−15.5% mean KL** |
| `probe-v4` PPL Δ% vs BF16 (`8.5438`) ↓ | +1.00% | **−0.292% (`8.5189`)** | **Zero PPL degradation** |
| `probe-guard` token mean KL / PPL Δ% (76 rows) ↓ | 0.0867 / −0.52% | **0.063923 / −6.828%** | **−26.3% mean KL** |
| Native MTP isolated KL / Top-1 agreement (12,339 pos) ↑ | *(BF16 in old repo)* | **0.010930 / 96.22% (11,873/12,339)** | **−521.4 MiB MTP shard** |
| Native MTP end-to-end KL / Top-1 agreement ↑ | — | **0.213082 / 87.14%** | — |
| Vision ViT feature Rel-RMSE / Cosine similarity (48 images) | *(BF16 in old repo)* | **0.1143 / 0.9940** | **−605 MiB Vision** |
| Vision isolated answer KL / Top-1 agreement (157 tokens) ↑ | — | **0.015734 / 97.45% (153/157)** | **98.67% ChartQA** |

## Inference Server & vLLM Implementation Notes

1. **Online Input Hadamard Transform (`transform_config`)**:
   - All 517 quantized linear layers (`510` in `model.safetensors` + `7` in `model_mtp.safetensors`) have weights pre-rotated offline along the input dimension ($W' = W H_b / \sqrt{b}$, where $H_b$ is the deterministic $b \times b$ Sylvester Walsh-Hadamard matrix; $b=64$ for `v` and $b=16$ for `v_vision_fc2` on `visual.blocks.*.mlp.linear_fc2`) prior to Cholesky GPTQ quantization (`actorder="static"`, unpermuted before packing so no `weight_g_idx` tensor is emitted).
   - At runtime, apply the normalized block-diagonal Walsh-Hadamard transform to the last dimension of input activations before the W4A16 linear GEMM:
     $$x' = \text{unflatten}(x, (-1, b)) \cdot \frac{H_b}{\sqrt{b}}$$
   - Because $H_b$ is deterministic (`randomize: false`), no transform weight tensors are stored in `.safetensors` (`0` storage overhead).

2. **vLLM V1 (`v0.31.0`) 2-line fix for Hadamard-rotated Vision layers**:
   - Text (`400` linears) and Native MTP (`7` linears) load out of the box in vLLM.
   - For Vision (`model.visual.*`), `vllm/model_executor/layers/quantization/compressed_tensors/transform/linear.py` needs two lines adjusted for biased linears and unfused `QKVParallelLinear`:
     ```python
     # 1. In CompressedTensorsLinearTransformMethod.apply (line 159):
     # Input transforms commute with additive bias: y = (x H)(W H)^T + b
     assert bias is None or self.output_transform is None

     # 2. In CompressedTensorsLinearTransformMethod._validate_tfm_schemes (line 177):
     # Iterate over populated input_tfms keys (matching output_tfms below it)
     for part_index in self.input_tfms:
         if self.input_tfms[part_index] != self.input_tfms[0]:
             raise ValueError("Must have same input")
     ```
