---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- compressed-tensors
- qwen3_5
- int6
- autoround
- humming
---

# Qwen3.8-27B INT6-Mixed AutoRound

Yes, you heard right. Looks like vLLM supports INT5-7 weights.

This is a **7.30bpw** mixed INT5/6/7/8 quantization of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), built with [AutoRound](https://github.com/intel/auto-round) and stored in the compressed-tensors `pack-quantized` format.

This model is for people who have 48GB of total VRAM and use vLLM. You can get **max** context(16bit unquantized) with vision and MTP enabled with this model.

> **This checkpoint does not load on stock vLLM.** It needs a two-line patch to `qwen3_5.py` / `qwen3_5_mtp.py`, supplied in `vllm-patch/`. See [Running it](#running-it) before downloading this.

## Measurements

KL divergence against the BF16 base model, WikiText-2 test, 240 sequences x 512 tokens, scored on the first 511 position of each sequence for 240 × 511 = 122,640 scored next-token positions, `KL(P_base || Q_quant)` in nats. Same reference distribution and the same token sequence as every other row, and the tokenization was verified byte-identical against the previous build's stored token array.
# **KL divergence cannot be compared across models, datasets, or evaluation methods, and is only comparable if all of them are the same. You *CANNOT* compare these KLD values to others' KLD reports!**

# **If other quantized models report a very low / high KLD compared to mine in their READMEs, it is typically due to differences in the evaluation methodology.**

| Metric | INT6-Mixed | BF16 base |
|---|---|---|
| mean KL | 0.001980 (±SE 0.000051) | 0 |
| median | 0.001135 | — |
| p90 | 0.003683 | — |
| p95 | 0.005642 | — |
| p99 | 0.014824 | — |
| p99.9 | 0.055416 | — |
| max | 4.638205 | — |
| top-1 agreement | 97.905 % | 100 % |
| Perplexity | 7.9183 | 7.9112 |

## Layout

64 language-model layers, of which 48 use gated-delta linear attention and 16 use full attention.

The MLP tiers are split by depth. Call layers 0-9 and 54-63 the edge layers (20 of them) and the remaining 44 the middle.

| Group | Scheme | Tensors |
|---|---|---|
| `mlp.{gate,up}_proj`, the 44 middle layers | INT6 symmetric, group_size 64 | 88 |
| `mlp.{gate,up}_proj`, the 20 edge layers | INT7 symmetric, group_size 64 | 40 |
| `mlp.down_proj`, the 44 middle layers | INT7 symmetric, group_size 64 | 44 |
| `mlp.down_proj`, the 20 edge layers | INT6 symmetric, group_size 64 | 20 |
| `self_attn.{q,k,v,o}_proj`, the 16 full-attention layers | INT7 symmetric, group_size 64 | 64 |
| `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, 48 layers | INT8 symmetric, group_size 128 | 144 |
| MTP block (`self_attn.*`, `mlp.*`) | INT5 symmetric, group_size 128 | 7 |
| `embed_tokens`, `lm_head` | INT8 symmetric, group_size 128 | 2 |
| `linear_attn.{in_proj_a,in_proj_b}`, 48 layers | BF16 | 96 |
| Vision tower (27 blocks, merger, pos_embed) | BF16 | 111 |
| `mtp.fc` | BF16 | 1 |
| All norms | BF16 | — |

The layout looks kinda random and this is because I wanted to retain the model's quality as high as possible.

## How to use

I recommend using vLLM docker image as it's the easiest way to use this model.

### 1. A vLLM with the humming kernel

The INT6 and INT7 tiers dispatch to `HummingLinearKernel` through `CompressedTensorsWNA16`. You need a build where:

- `WNA16_SUPPORTED_TYPES_MAP` covers 5/6/7 bits (vLLM PR #46389, merged 2026-06-24), 
- and the `humming-kernels` package is installed. It is in `requirements/cuda.txt`, so a stock CUDA wheel or image has it.

Check both in one line:

```bash
python -c "import importlib.metadata as m; print(m.version('humming-kernels'));
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 \
import WNA16_SUPPORTED_TYPES_MAP as M; print(sorted(M))"
# humming-kernels 0.1.12
# [2, 3, 4, 5, 6, 7, 8]
```

`HummingLinearKernel.get_min_capability()` is 75, so sm86 (RTX 3090) is fine. Verified against vLLM `0.26.1rc1.dev542+gb22afe45a` and `0.27.2rc1.dev122+g8efa13b70`.

### 2. Patch vLLM for the quantized embedding

vLLM has a working quantized-embedding implementation (`CompressedTensorsEmbeddingWNA16Int`) that Qwen3.5's model definition never reaches, because `models/qwen3_5.py` builds

```python
self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)
```

with neither `quant_config` nor `prefix`. `vllm-patch/apply.sh` pulls the two files out of your image, applies the diffs, and writes the bind-mount flags:

```bash
cd vllm-patch && ./apply.sh <your-vllm-image>
podman run ... $(cat mounts.txt) <your-vllm-image> --model /model ...
```

On a checkpoint with unquantized embeddings the layer falls back exactly as before, so it is safe to leave mounted for other models. 


### 3. Serve

```bash
vllm serve /path/to/model \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.97 \
  --max-model-len 262144 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Measured on 2x RTX 3090 (sm86), TP2, vision and MTP both enabled:

```
Using HummingLinearKernel for CompressedTensorsWNA16
Model loading took 12.19 GiB memory and 86.27 seconds
Available KV cache memory: 8.97 GiB
GPU KV cache size: 263,672 tokens, Maximum concurrency for 262,144 tokens per request: 1.01x
Auto-fit max_model_len: full model context length 262144 fits in available GPU memory
Actual usage is 12.36 GiB for consumed memory (weights + non-torch),
1.53 GiB for peak activation, and 0.04 GiB for CUDAGraph memory.
```

Full context fits, but only by **1,528 tokens** out of 263,672. That is 0.6 % of headroom, so anything that eats KV will push you under 262,144: a third GPU-resident process, a larger CUDA graph capture, or a vLLM build whose activation peak differs from the one measured here. If you do not need 262,144 exactly, there is a lot of slack; if you do, measure your own build before relying on it.

## Findings

* `embed_tokens` cannot be 5, 6 or 7 bits. 
vLLM's embedding path is not the Linear path. `CompressedTensorsEmbeddingWNA16Int` computes `pack_factor = 32 // num_bits` and its Triton dequant-gather kernel indexes with `packed_idx = col // PACK_FACTOR; shift = (col % PACK_FACTOR) * NUM_BITS`, i.e. it assumes values never straddle a 32-bit word. compressed-tensors ≥0.18 writes the dense layout, `ceil(in_features * bits / 32)`. At six bits those disagree (`5120 // (32 // 6) = 1024` against `ceil(5120*6/32) = 960`) and the weight loader fails with a shape mismatch. Only widths that divide 32 work, such as 2, 4 and 8. Fixing this means rewriting that kernel, not adding a keyword argument. 

* Asymmetric quantization is unavailable above four bits.
`WNA16_ZP_SUPPORTED_TYPES_MAP` covers 4 and 8 only, so every 5/6/7-bit group here is symmetric. The previous build's four-bit tier was asymmetric and got roughly a quarter-bit of its accuracy from that; this one does not have the option and does not need it.

* **MoE is not supported on this path.** 
Not relevant to this model, but `CompressedTensorsWNA16MoEMethod` still computes `32 // num_bits` and raises for anything but int4/int8, so the same layout cannot be applied to a Qwen3.5-MoE checkpoint through compressed-tensors. The native `humming` checkpoint format has a fused MoE path. This format does not.


## Reproducing this model

AutoRound refuses 5/6/7-bit export to the `llm_compressor` format out of the box. Two bit checks in `auto_round/export/formats/backends/llm_compressor.py` reject them before any work happens; the packing itself is delegated to compressed-tensors and has handled 1-8 bits for a while. The patch is in `auto-round-patch/`:

```diff
-        if scheme.bits not in [4, 8, 16]:
+        if scheme.bits not in [4, 5, 6, 7, 8, 16]:
-        if scheme.data_type == "int" and scheme.bits not in [4, 8]:
+        if scheme.data_type == "int" and scheme.bits not in [4, 5, 6, 7, 8]:
```

plus `W5A16` / `W6A16` in `support_schemes`. With that applied, `pack_layer` produces exactly the shapes vLLM expects. Verified at 4, 5, 6, 7 and 8 bits against `ceil(in_features * bits / 32)`, and end-to-end through a vLLM load.

Calibration: `iters=500`, `nsamples=768`, `seqlen=2048`, `batch_size=2`, `gradient_accumulate_steps=4`, dataset `NeelNanda/pile-10k` (256) plus `codeparrot/github-code-clean` (768).

`layer_config.json` is included: three keyword exclusions (`embed_tokens`, `visual`, `lm_head` at 16 bits) followed by all 400 language-model projections named in full. The projections are spelled out rather than pattern-matched because a pattern like `mlp.gate_proj` also hits `mtp.layers.0.mlp.gate_proj` and the vision tower, and AutoRound resolves the longest match last. Anything the file does not mention falls through to the default scheme, so the dry run's "unexpected 16-bit layers" column has to be empty before committing five hours to it.

If you are building something similar, note why the embedding and head targets are regexes. vLLM rewrites quantization targets through the model's `WeightsMapper`, but `apply_vllm_mapper` treats anything without a dot as a class name and passes it through untouched. A target spelled `lm_head` never reaches `language_model.lm_head`. And `model.language_model.embed_tokens` reaches the main model but not the MTP draft, whose module sits at `mtp.embed_tokens`. `re:` targets skip the rewrite and are matched as regular expressions, so they hit every copy.


## Files

```
model-0000{1..9}-of-00009.safetensors   weights
model-mtp.safetensors                   MTP block
config.json                             includes the compressed-tensors config
layer_config.json                       the allocation, as fed to AutoRound
vllm-patch/                             the two-line vLLM embedding patch
auto-round-patch/                       the 5/6/7-bit export gate patch
```

## Acknowledgements

- [Alibaba / Qwen team](https://huggingface.co/Qwen) for the base [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) model
- [Intel AutoRound](https://github.com/intel/auto-round) team for the quantization framework
- [vLLM project](https://github.com/vllm-project/vllm) for the inference engine and Qwen3_5 MTP support
- [lued](https://huggingface.co/lued) for the quantized dflash2 drafter model
- [z-lab](https://huggingface.co/z-lab) for the dflash2 drafter model
