---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- awq
- w4a16
- int4
- compressed-tensors
- llm-compressor
- vllm
- qwen3_5_moe
- moe
---

# Ornith-1.5-35B-A3B — AWQ W4A16 (compressed-tensors)

**Unofficial community quantization.** This is a 4-bit AWQ (weight-only int4, group size 128, symmetric)
derivative of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).
It was **not made, reviewed or endorsed by Ornith AI**. It was produced for use with vLLM on a single
48 GB NVIDIA card (e.g. RTX 6000 Ada).

> **Status: NOT yet tested on vLLM or on any NVIDIA GPU.** The checkpoint was produced and checked on
> AMD hardware only (Radeon 8060S, PyTorch ROCm on Windows): static config/tensor checks, and perplexity,
> top-1 agreement and greedy generations against the BF16 original using `transformers` with the int4
> weights dequantized from this checkpoint. Please report back if `vllm serve` fails.

## What is in here

| | |
|---|---|
| Base | ornith-ai/Ornith-1.5-35B-A3B (BF16, `Qwen3_5MoeForConditionalGeneration`: 40 decoder layers = 30 Gated-DeltaNet + 10 full-attention (gated), MoE with 256 routed experts (top-8, intermediate 512) + 1 shared expert per layer, 1 MTP layer, 27-block vision encoder) |
| Scheme | W4A16, symmetric int4, group size 128, `pack-quantized` compressed-tensors format (vLLM: `CompressedTensorsWNA16` for dense layers, `CompressedTensorsWNA16MarlinMoEMethod` / Marlin MoE for the experts on Ampere/Ada/Hopper) |
| Algorithm | AWQ (activation-aware scaling, `duo_scaling=True`, `n_grid=10`) followed by round-to-nearest int4 with min-max group scales |
| Tool | [llm-compressor](https://github.com/vllm-project/llm-compressor) 0.14.0 (Apache-2.0), compressed-tensors 0.19.0, transformers 5.17.0, torch 2.11.0+rocm7.13.0 |
| Size | 22.0 GB on disk (BF16 original: 71.9 GB): int4 decoder weights (packed + scales) 17.3 GB, BF16 `lm_head` + embeddings 2.0 GB, vision encoder 0.9 GB, MTP head 1.7 GB (`model_mtp.safetensors`), router/norms/misc 0.05 GB |

### Quantized vs. kept in BF16

Quantized to int4 (all 40 decoder layers):
- routed experts: `mlp.experts.{0..255}.{gate,up,down}_proj` (stored per expert, the layout vLLM's FusedMoE loader expects)
- shared expert: `mlp.shared_expert.{gate,up,down}_proj`
- full attention: `self_attn.{q,k,v,o}_proj`
- Gated DeltaNet: `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`

Kept in BF16 (`ignore` list in `config.json`):
- MoE router `mlp.gate` and `mlp.shared_expert_gate` (vLLM creates both without a quant config)
- `linear_attn.in_proj_a` / `linear_attn.in_proj_b` (32-row outputs; vLLM fuses them as `in_proj_ba`)
- `lm_head`, `embed_tokens`
- the whole vision encoder and merger (`model.visual.*`)
- the MTP head (`mtp.*`, copied unchanged from the BF16 checkpoint, including its 256 experts)
- all norms, `conv1d`, `A_log`, `dt_bias` (not Linear layers)

AWQ smoothing mappings (see `recipe.yaml`):
- `input_layernorm → q/k/v_proj` (full-attention layers), `input_layernorm → in_proj_qkv/z/b/a` (linear-attention layers)
- `post_attention_layernorm → all experts' gate/up_proj + shared expert gate/up_proj + shared_expert_gate + router (mlp.gate)`
- `up_proj → down_proj` for the shared expert and for **each routed expert separately**, using only the tokens routed to that expert

Two MoE-specific details:
- **Router compensation.** llm-compressor's built-in Qwen3.5-MoE mapping does not include the router (`mlp.gate` is a bare
  parameter in transformers), so scaling `post_attention_layernorm` would silently change the routing logits. For this
  checkpoint the router was wrapped as an `nn.Linear` and added as an AWQ balance layer, so its (BF16) weight absorbs the
  inverse scale and the routing logits are preserved (checked: effective router weight relative error ≤0.33% per layer,
  i.e. BF16 rounding only).
- **Expert coverage.** Calibration used `moe_calibrate_all_experts=False` (each expert sees only its own routed tokens).
  With 256 calibration samples every one of the 40 × 256 = 10,240 routed experts received tokens (minimum 186, median ≈5,000 routed tokens per expert; 4 experts got fewer than 256 and 128 fewer than 1,000). Had an expert received none, AWQ would skip its `up→down` smoothing and that expert would fall back to plain round-to-nearest (no NaN risk); this did not happen.

### Calibration data

256 chat-formatted samples (rendered with the model's own chat template, thinking disabled), truncated to 2048 tokens,
200,251 tokens in total (193,952 after length-sorted batching with batch size 8):

| Source | License | Samples | Content |
|---|---|---|---|
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (test_sft) | MIT | 64 | English multi-turn chat |
| [shareAI/ShareGPT-Chinese-English-90k](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k) (computer_zh) | Apache-2.0 | 64 | Chinese programming / technical chat |
| shareAI/ShareGPT-Chinese-English-90k (common_zh) | Apache-2.0 | 32 | Chinese general chat |
| [theblackcat102/evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) | Apache-2.0 | 64 | English coding instructions |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Apache-2.0 | 32 | tool / function calling (converted to the model's tool-call format) |

## Quality check (AMD / transformers, not vLLM)

24 held-out chat samples from the same five sources (not used for calibration), each ≤2048 tokens. The quantized model was
evaluated by loading the BF16 model in `transformers` and overwriting every tensor with the one stored in this checkpoint (int4
weights dequantized exactly as `q × scale` per 128-group; AWQ-rescaled BF16 tensors copied), i.e. the same int4 values vLLM's kernels use.
"Assistant-only" scores only the 10,706 tokens of the assistant turns (what the model generates); "full sequence" also scores the
user turns (17,416 tokens). Baseline: data-free round-to-nearest (RTN) W4A16 g128 with the identical ignore list.

| Metric | BF16 original | **AWQ (this repo)** | RTN baseline |
|---|---|---|---|
| Assistant-only PPL (24 samples) | 2.944 | **3.000 (+1.9%)** | 3.064 (+4.1%) |
| Median per-sample change in assistant NLL (nats/token) | 0 | **+0.003** | +0.026 |
| Samples where AWQ loss ≤ RTN loss | | **17 / 24** | |
| Top-1 next-token agreement with BF16, assistant tokens | 100% | **90.8%** | 90.1% |
| Top-1 next-token agreement with BF16, full sequence | 100% | 76.0% | 75.1% |
| Full-sequence PPL (incl. user turns) | 19.70 | 20.59 | 19.25 |

Assistant-only PPL by source (BF16 → AWQ / RTN): English chat (UltraChat) 3.274 → 3.394 / 3.398; Chinese coding 2.745 → **2.820** / 2.907;
Chinese general 3.379 → **3.462** / 3.610; English code 1.972 → **1.964** / 2.004; tool calling 2.744 → **2.454** / 2.475.

Caveats, stated plainly:
- English chat (UltraChat) is where AWQ helps least: 3 of its 5 samples are slightly worse with AWQ than with RTN (largest: +0.064 vs +0.023 nats/token on a 387-token answer).
- Both quantized models get *lower* loss than BF16 on the 5 tool-calling samples (short answers, 494 tokens in total); treat that as noise, not as an improvement.
- Full-sequence PPL (scoring user-written text) is slightly worse for AWQ than for RTN; AWQ is calibrated to preserve the layer outputs on chat data, and predicting user text is not the use case, but the number is reported for completeness.
- 24 samples / ~10.7k assistant tokens is a sanity check, not a benchmark. Please run your own task evaluations.

Greedy generations (thinking disabled, 256 new tokens) for a Chinese coding prompt (`merge_intervals` with Chinese comments) and an English
one (log parser with type hints): BF16, AWQ and RTN all produced well-formed, on-topic code. Small blemishes in the AWQ outputs: a garbled
section heading in the Chinese answer ("演思思路"; the BF16 answer's heading was also garbled, "解思路解") and a wrong type hint
`Counter[tuple[str, str], int]` in the English answer (`Counter` takes one type parameter).

## vLLM usage (single 48 GB card, e.g. RTX 6000 Ada) — untested

```bash
vllm serve tsaipifong/Ornith-1.5-35B-A3B-AWQ \
  --served-model-name Ornith-1.5-35B-A3B-AWQ \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

- The weights take about 20 GB (about 21 GB with the MTP head loaded for speculative decoding) of VRAM, leaving most of the 48 GB for the KV / linear-attention state cache. Only 10 of
  the 40 layers keep a growing KV cache (2 KV heads × 256 dim), so long contexts are cheap; the full 262,144 should fit
  too (`--max-model-len 262144`). Add `--language-model-only` to skip the vision encoder if you do not need images.
- The parser flags follow the upstream model card (`qwen3_xml` tool parser, `qwen3` reasoning parser).
  Upstream recommends `temperature=0.6, top_p=0.95, top_k=20` for general use.
- MTP speculative decoding (**to be verified**; the MTP head is included in BF16):
  `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` (newer vLLM versions may accept `"method":"mtp"`).
- vLLM should detect the format automatically (`quant_method: compressed-tensors`); do **not** pass `--quantization awq`.
  The upstream card asks for vLLM ≥ 0.19.1.

## How it was made

```
python scripts/build_calib.py            # builds calib.jsonl / eval.jsonl from the five datasets above
python scripts/quantize_moe.py <bf16-dir> <out> --n 256 --seqlen 2048 --grid 10 --bs 8 --act-offload cuda --device-map auto
python scripts/validate_moe.py <out> <bf16-dir>     # static config / tensor / router-compensation checks
python scripts/eval_moe.py <bf16-dir> bf16.json                             # BF16 reference
python scripts/eval_moe.py <bf16-dir> awq.json --qdir <out> --ref bf16.json # this checkpoint, dequantized
python scripts/eval_moe.py <bf16-dir> rtn.json --rtn --ref bf16.json        # RTN baseline
```
(`scripts/dist_shim.py` only adds single-process stubs because the Windows ROCm build of PyTorch has no `torch.distributed`.)
llm-compressor `oneshot` with the sequential (layer-by-layer) pipeline on a Radeon 8060S; recipe =
`AWQModifier(duo_scaling=True, n_grid=10, mappings=[...])` + `QuantizationModifier(targets=["Linear"], scheme="W4A16", ignore=[...])`.
Runtime: 3.9 h for `oneshot` on a Radeon 8060S (40 layers, ~5.7 min per layer, most of it the 256 per-expert grid searches). In that run llm-compressor's own MTP-copy step crashed on Windows after the shards were written (a memory-mapped tensor read after its file was closed); `quantize_moe.py` now skips that step and `scripts/finalize.py` copies the MTP tensors and tokenizer files instead (it was run on this checkpoint the same way). Note: llm-compressor converts Qwen3.5's `(1 + w)` RMSNorm weights to the plain convention during calibration and back when saving, which re-rounds the BF16 norm weights by up to one BF16 ulp; this affects both AWQ and the RTN baseline checkpoints made with it.

## License and attribution

- Original model: **Ornith-1.5-35B-A3B by Ornith AI** ([ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)),
  released under the **MIT License**. The upstream model card states that the Ornith models were developed on top of Qwen3.5.
- This quantized derivative is distributed under the same MIT License; see `LICENSE`. The upstream repository declared MIT in its
  metadata but did not ship a LICENSE file at the time of quantization, so the standard MIT text is included with a note.
- "Ornith" is the name of the original authors' model and is used here only to describe the origin of the weights. No Ornith logos
  or other upstream assets are included.

Changes relative to the original: `model*.safetensors` and `model.safetensors.index.json` (AWQ rescaling + int4 quantization;
BF16 tensors such as norms, router, `shared_expert_gate`, `in_proj_a/b` carry the folded AWQ scales), `config.json` (added
`quantization_config`), `recipe.yaml`, `README.md`, `LICENSE` and `scripts/` were created or modified by this quantization.
`model_mtp.safetensors` contains the original BF16 MTP tensors unchanged. Tokenizer, chat template, generation and processor
configs are copied unchanged from the original repository.
