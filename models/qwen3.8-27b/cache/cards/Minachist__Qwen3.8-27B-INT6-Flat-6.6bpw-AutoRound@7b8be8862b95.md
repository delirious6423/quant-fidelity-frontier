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

# Qwen3.8-27B INT6-Flat 6.6bpw AutoRound

A **6.60 bpw** quantization of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), built with [AutoRound](https://github.com/intel/auto-round) and stored in the compressed-tensors `pack-quantized` format.

This model is for people with **2x24GB** who want 262,144 context with vision and MTP enabled and room to spare. It leaves 301,121 tokens of KV cache, 39k more than the context length needs.

> **This checkpoint does not load on stock vLLM.** It needs a two-line patch to `qwen3_5.py` / `qwen3_5_mtp.py`, supplied in `vllm-patch/`. See [How to use](#how-to-use) before downloading this.

## Measurements
KL divergence against the BF16 base model, WikiText-2 test, 240 sequences x 512 tokens, scored on the first 511 position of each sequence for 240 × 511 = 122,640 scored next-token positions, `KL(P_base || Q_quant)` in nats. Same reference distribution and the same token sequence as every other row, and the tokenization was verified byte-identical against the previous build's stored token array.
# **KL divergence cannot be compared across models, datasets, or evaluation methods, and is only comparable if all of them are the same. You *CANNOT* compare these KLD values to others' KLD reports!**

# **If other quantized models report a very low / high KLD compared to mine in their READMEs, it is typically due to differences in the evaluation methodology.**

| Metric | INT6-Flat 6.6bpw | BF16 base |
|---|---|---|
| mean KL | 0.003051 (±SE 0.000047) | 0 |
| median | 0.001652 | — |
| p90 | 0.005841 | — |
| p95 | 0.009019 | — |
| p99 | 0.025704 | — |
| p99.9 | 0.095758 | — |
| max | 2.923414 | — |
| top-1 agreement | 97.471 % | 100 % |
| Perplexity | 7.9294 | 7.9112 |

## Layout

64 language-model layers, of which 48 use gated-delta linear attention and 16 use full attention.

| Group | Scheme | Tensors |
|---|---|---|
| `mlp.{gate,up}_proj`, all 64 layers | INT6 symmetric, group_size 64 | 128 |
| `mlp.down_proj`, all 64 layers | INT6 symmetric, group_size 64 | 64 |
| `self_attn.{q,k,v,o}_proj`, the 16 full-attention layers | INT6 symmetric, group_size 64 | 64 |
| `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, 48 layers | INT6 symmetric, group_size 64 | 144 |
| MTP block (`self_attn.*`, `mlp.*`) | INT5 symmetric, group_size 128 | 7 |
| `embed_tokens`, `lm_head` | INT8 symmetric, group_size 128 | 2 |
| `linear_attn.{in_proj_a,in_proj_b}`, 48 layers | BF16 | 96 |
| Vision tower (27 blocks, merger, pos_embed) | BF16 | 111 |
| `mtp.fc` | BF16 | 1 |
| All norms | BF16 | — |

This time the layout looks lazy. I actually used dynamic programming to test around 18000 different configs for bit allocation. At a 21.3 gb budget the best setup is actually just flat. The script just put every body projection on int6 gs64 by itself. I didnt force it to be flat it just came out that way.



## How to use

I recommend using the vLLM docker image as it's the easiest way to use this model.

### 1. A vLLM with the humming kernel

The INT6 tier dispatches to `HummingLinearKernel` through `CompressedTensorsWNA16`; the INT8 embedding/head tier uses Marlin. You need a build where:

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

`HummingLinearKernel.get_min_capability()` is 75, so sm86 (RTX 3090) is fine.

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

> If your vLLM image is newer than the patch, `apply.sh` may fail to apply cleanly — upstream has been moving this file. Diff against the copy inside your image before assuming the patch is current.

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
Model loading took 10.95 GiB memory and 51.17 seconds
Available KV cache memory: 10.25 GiB
GPU KV cache size: 301,121 tokens, Maximum concurrency for 262,144 tokens per request: 1.15x
Auto-fit max_model_len: full model context length 262144 fits in available GPU memory
Actual usage is 11.10 GiB for consumed memory (weights + non-torch),
1.51 GiB for peak activation, and 0.03 GiB for CUDAGraph memory.
```

As shown, it has 38,977 tokens of headroom over the full 262,144 context. My INT6-Mixed build fits the same context with only 1,528 to spare, but this one gives you room to load the Dflash-2 drafter with (almost) full context.

## Findings

* Dropping `embed_tokens` to INT4 is not worth it. It saves 0.57 GiB and costs 33 % more KL. Taking the same 0.57 GiB out of the body costs less.

* You should stop at group size 64. Going from gs128 to gs64 is pretty cheap for the bits you get but going from gs64 to gs32 is way too expensive. Just always use gs64 and dont bother with gs32.

* Trying to mix levels is a bad idea for lower quants. Saving a gigabyte by shrinking just one specific tier hurts the kl about twice as much as just shrinking the whole body evenly. If you need a smaller model just drop the whole thing down a level instead of shaving off specific parts.

* **Fused groups must be uniform.** `qkv_proj`, `gate_up_proj` and `in_proj_qkvz` are fused at load time, and vLLM silently applies the *first* member's scheme to the whole fusion. A per-layer config that splits a fused group produces a checkpoint that loads without complaint and is wrong. Everything here is flat, so it cannot happen — but if you build a mixed layout, verify the fusion groups against the actual packed tensors rather than trusting the config.

## Reproducing this model

AutoRound refuses 5/6/7-bit export to the `llm_compressor` format out of the box. Two bit checks in `auto_round/export/formats/backends/llm_compressor.py` reject them before any work happens; the packing itself is delegated to compressed-tensors and has handled 1-8 bits for a while. The patch is in `auto-round-patch/`:

```diff
-        if scheme.bits not in [4, 8, 16]:
+        if scheme.bits not in [4, 5, 6, 7, 8, 16]:
-        if scheme.data_type == "int" and scheme.bits not in [4, 8]:
+        if scheme.data_type == "int" and scheme.bits not in [4, 5, 6, 7, 8]:
```

plus `W5A16` / `W6A16` in `support_schemes`. With that applied, `pack_layer` produces exactly the shapes vLLM expects. Verified at 4, 5, 6, 7 and 8 bits against `ceil(in_features * bits / 32)`, and end-to-end through a vLLM load.

Calibration: `iters=500`, `nsamples=768`, `seqlen=2048`, `batch_size=2`, `gradient_accumulate_steps=4`, dataset `NeelNanda/pile-10k` (256) plus `codeparrot/github-code-clean` (768). 5 h 53 m on 2x RTX 3090 with data-parallel calibration across both cards.

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
