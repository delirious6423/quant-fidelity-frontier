---
base_model: Qwen/Qwen3.8-Flash-Next
library_name: transformers
license: other
license_name: qwen-community-1.0
license_link: LICENSE
tags:
- qwen3.8
- qwen4
- moe
- gptq
- quantized
- 4-bit
- vllm
- vision
- multimodal
- mtp
- ngram
---

# Qwen3.8-Flash-Next GPTQ 4-bit

GPTQ 4-bit (W4, group size 32) quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), the Qwen4-architecture preview: an ultra-sparse multimodal MoE with **125B parameters (6B active)** plus a separate **51B n-gram embedding table** and a 4B MTP head. This checkpoint quantizes the routed experts, shared experts and full-attention projections to INT4 and lands at **+0.58% perplexity** over BF16 — while shrinking the GPU-resident body from ~250 GB to **80 GB**, so it serves on 4×32 GB GPUs with the n-gram table in host memory.

Includes the full vision encoder, the MTP (Multi-Token Prediction) module, and the complete n-gram table in the original sharded BF16 layout.

## Model Overview

- **Architecture**: `Qwen4ExpForConditionalGeneration` (model_type `qwen4_exp`; multimodal: text + vision)
- **Parameters**: 125B total / 6B active (MoE) + 51B n-gram table + 4B MTP = 180B in the checkpoint
- **Layers**: 48, all MoE — 36 linear-attention (Gated DeltaNet) + 12 full-attention (Qwen Sparse Attention with a top-k indexer), repeating 3:1
- **MoE**: 512 routed experts per layer, top-10 + 1 shared expert, expert intermediate size 640
- **N-gram table (PLE)**: 20M-row hashed bigram/trigram embedding memory injected at decoder layer 1 (`ple_layer_ids=[2]`, 1-based), 128 shard tensors
- **Hyper-connections**: 4 parallel residual streams with learned gated mixing around every block
- **Context length**: 262,144 tokens natively (1M with YaRN)
- **Vision encoder**: 27-block ViT, BF16 (333 tensors)
- **MTP module**: 1-layer speculative decoding head, BF16 (31 tensors)

## Quantization Details

| Component | Precision | Notes |
|-----------|-----------|-------|
| `mlp.experts.{i}.{gate_proj, up_proj, down_proj}` | INT4 GPTQ | 512 experts × 48 layers (73,728 modules), stored per-expert |
| `mlp.shared_expert.{gate_proj, up_proj, down_proj}` | INT4 GPTQ | All 48 layers |
| `self_attn.{q,k,v,o}_proj` | INT4 GPTQ | 12 full-attention layers |
| `linear_attn.*` (Gated DeltaNet) | BF16 | All 36 linear-attention layers kept at full precision |
| `self_attn.indexer.*` | BF16 | Sparse-attention top-k selector — kept exact |
| `mlp.gate`, `mlp.shared_expert_gate` | BF16 | Routers |
| Hyper-connection weights (`*_hyper_connection.*`, `hyper_connection_mixer.*`) | BF16 | |
| N-gram table + PLE glue (`layers.1.ple.*`) | BF16 | 128 shard tensors, bit-identical to the original |
| Vision encoder (`model.visual.*`) | BF16 | 333 tensors |
| MTP module (`mtp.*`) | BF16 | 31 tensors |
| Embeddings, LM head, norms | BF16 | |

**GPTQ configuration:**
- **Bits**: 4
- **Group size**: 32
- **Symmetric**: Yes
- **desc_act**: No
- **true_sequential**: Yes
- **mse**: 2.0 (activation-weighted MSE for outlier handling)
- **Fallback**: RTN at 0.5% calibration-coverage threshold (7.6% of expert modules — the rarely-routed tail of the 512-expert distribution)

### Calibration

- **Dataset**: Mixed — evol-codealpaca-v1 (code) + C4 (general English text)
- **Samples**: 2048, binned uniformly across context lengths 256–2048 tokens (~2.4M tokens)
- **Quantizer**: [GPTQModel](https://github.com/modelcloud/gptqmodel) v7.3.5 with a custom `qwen4_exp` model definition (see below)
- Note: this is general-purpose calibration. Calibrating on wikitext directly would yield lower wikitext perplexity but worse out-of-distribution performance; we optimized for the latter.

The exact quantization script is included in this repo as `quantize.py`.

## Model Size

| Version | Size | Notes |
|---------|------|-------|
| BF16 (original) | ~360 GB | ~250 GB body + 102 GB n-gram table + 8 GB MTP |
| **GPTQ 4-bit** | **188 GB** | **80 GB body** (INT4) + 102 GB n-gram table (BF16) + BF16 keeps |

Only the body is GPU-resident at serve time: **~80 GB, about 20 GB per GPU at tensor-parallel 4** — the n-gram table lives in host RAM (see Usage).

The table itself is *not* quantized: it is a pure lookup with no matmul, there is no activation path to calibrate against, engines load it by row gather, and keeping it bit-identical to the original avoids any interaction with the model's hash-collision gating.

## Perplexity

Evaluated on wikitext-2-raw-v1 (test set), seq_len=2048, stride=512, 64 windows (32,768 scored tokens), through the transformers forward with identical windows for both models:

| Model | Perplexity | Degradation |
|-------|------------|-------------|
| BF16 (original)     | 3.1206 | — |
| **GPTQ 4-bit** (this) | **3.1386** | **+0.58%** |

> Note on absolute values: this model's n-gram memory has Wikipedia close to memorized (per-window perplexity runs from ~1.3 on early wikitext-2 articles to ~4 on later ones), so its absolute perplexity is not comparable to other model families — only the BF16-vs-quantized delta is meaningful.

## Fidelity vs BF16

Beyond perplexity, we compared full output distributions of BF16 and this W4 checkpoint on identical token windows (per-token full-vocab log-softmax through the transformers forward). KL is KL(BF16 ‖ W4) per token; top-1 agreement is how often both models rank the same token first (an upper-bound proxy for greedy-decoding divergence); top-5 overlap is the mean fraction of shared top-5 candidates.

| Suite | BF16 PPL | W4 PPL | Δ | KL mean | KL p99 | Top-1 agree | Top-5 overlap |
|-------|----------|--------|---|---------|--------|-------------|---------------|
| wikitext-2 (2k ctx, 64×512 tok) | 3.1206 | 3.1386 | +0.58% | 0.071 | 0.89 | 91.3% | 84.6% |
| wikitext-2 (4k ctx, 8×1024 tok) | 2.2919 | 2.2822 | −0.43% | 0.093 | 1.23 | 91.7% | 83.9% |
| Python code (2k ctx, 32×512 tok)¹ | 2.1801 | 2.1796 | −0.02% | 0.027 | 0.39 | 95.8% | 86.3% |

¹ Fixed slice of `codeparrot/codeparrot-clean-valid` (~65k scored tokens).

Takeaways: no degradation at longer context or on code (deltas within noise; code is the model's strongest domain and also its most faithful — KL 0.027, 95.8% top-1); the +0.58% wiki delta remains the only measurable cost. Mean KL below 0.1 nat across all suites is typical of a healthy W4 quantization; the p99 tail shows occasional token-level distribution shifts, which is expected at 4-bit and consistent with the unchanged downstream perplexity.

## Usage

### vLLM

Qwen3.8-Flash-Next support landed in vLLM main in September 2026 (before that it shipped only in the dedicated `vllm/vllm-openai:qwen38-flash-next` image). The n-gram table must be kept out of GPU memory: use the PLE host-memory offload (`VLLM_PLE_CPU_OFFLOAD=1`) or the memory-mapped table mode (`VLLM_PLE_MMAP=1`), which reads rows directly from this repo's `ple-*.safetensors` shard files.

```bash
VLLM_PLE_MMAP=1 vllm serve btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Notes:
- `--dtype bfloat16` — the sparse-attention implementation requires BF16 activations.
- Requires ≥ 100 GB of free host RAM for the table (or page cache in mmap mode).
- The MTP head is included; drop `--speculative-config` to serve without speculative decoding.
- Served wikitext-2 perplexity on our reference deployment (vLLM, TP4, compiled with full CUDA graphs) is **3.1362**, matching the transformers forward above within noise — a quick perplexity spot-check on your own hardware/kernel build is still recommended, since W4 kernel paths for this architecture are young.
- **ROCm gfx908 (MI100) note**: the upstream AMD Triton kernel for the sparse-attention layers miscompiles at TP4 on gfx908 (its per-rank head-group tile size of 8 produces garbage output; tiles of 16+ are exact), which shows up as fluent-but-degraded generations (perplexity ~2.3× worse). A one-line fix (clamp the tile size to ≥16) is in [btbtyler09/vllm-gfx908](https://github.com/btbtyler09/vllm-gfx908) branch `qwen38-flash-next`.

### GPTQModel / transformers

GPTQModel does not yet ship a `qwen4_exp` definition. The definition used to produce and load this checkpoint (including the n-gram table handling for load-back) is on the `qwen4-exp-support` branch of [btbtyler09/GPTQModel](https://github.com/btbtyler09/GPTQModel); with it installed:

```python
from gptqmodel import GPTQModel
model = GPTQModel.load("btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit", device_map=<4-GPU layer map>)
```

The loader keeps the n-gram table on CPU (memory-mapped from the shard tensors) and places the INT4 body across GPUs. Loading via plain `transformers` GPTQ integration also works but is very slow for the 73,920 quantized modules.

## Technical Notes

- The checkpoint stores routed experts as separate per-expert tensors (`mlp.experts.{i}.{gate_proj,up_proj,down_proj}` with standard GPTQ `qweight/qzeros/scales/g_idx`), which is the layout vLLM's fused-MoE loader consumes.
- The n-gram table keeps the original `ple.ple_embedding.ngram_embedding.shard_{0..127}.weight` layout (in `ple-*.safetensors` files referenced from the index); a fused single-tensor form is deliberately not shipped, as no serving loader consumes it.
- `config.json` is the original config verbatim plus `quantization_config`; no KV-cache scale fields are emitted.
- Quantizing this model required several GPTQModel changes that are worth knowing about if you reproduce it: balanced multi-GPU placement of expert Hessians (`moe_vram_strategy="balanced"` — the default parks all 512 experts' Hessians on one GPU), a CPU-side calibration cache (the 4-stream hyper-connections make cached activations ~4× wider than usual), and forcing `use_cache=False` on the nested text config.

## Credits

- **Base Model**: [Qwen](https://huggingface.co/Qwen) — Qwen3.8-Flash-Next
- **Quantization**: GPTQ via [GPTQModel](https://github.com/modelcloud/gptqmodel) v7.3.5
- **Quantized by**: [btbtyler09](https://huggingface.co/btbtyler09)

## License

This model inherits the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) from the base model (included as `LICENSE`).
