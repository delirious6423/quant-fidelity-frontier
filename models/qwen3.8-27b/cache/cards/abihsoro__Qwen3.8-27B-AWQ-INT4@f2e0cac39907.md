---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- quantized
- awq
- w4a16
- compressed-tensors
- vllm
---

# Qwen3.8-27B-AWQ-INT4

Self-quantized AWQ INT4 (W4A16, group_size=128, symmetric) build of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) via `llm-compressor`, since no AWQ/GPTQ quant existed yet at release time. At matching throughput, it scores meaningfully higher on GSM8K than a same-scheme INT4 quant of the prior-generation `Qwen3.6-27B` (96% vs. 79% flexible-extract, 89% vs. 78% strict-match) — see [Benchmarks](#benchmarks) below.

- **Text-only — no vision tower.** The base `Qwen/Qwen3.8-27B` is vision-capable; this quant is not. It was loaded through `AutoModelForCausalLM`, which resolves to the text-only class variant (`"model_type": "qwen3_5_text"` in `config.json`), so the vision encoder was never included — image and video inputs are unsupported. Two reasons: the target workload is text-only, and `llm-compressor` currently crashes calibrating the vision encoder’s position-embedding interpolation (`NotImplementedError: Cannot copy out of meta tensor`, [llm-compressor#1939](https://github.com/vllm-project/llm-compressor/issues/1939)). The `re:.*visual.*` entries in the recipe below are vestigial and match nothing.
- **Format**: `compressed-tensors`, W4A16, group_size=128, symmetric, single-file `model.safetensors` (~17.6 GB)
- **Calibration**: `HuggingFaceH4/ultrachat_200k`, 256 samples
- **Known limitation**: the model's 48 GDN/linear-attention layers received plain group quantization rather than full AWQ activation-aware smoothing, due to an `llm-compressor` crash calibrating `Qwen3_5GatedDeltaNet` layers via the dynamic hybrid-attention mapping builder. Attention/MLP `Linear` layers got standard AWQ smoothing per the recipe below.
- **Chat template**: patched to alias the `effort: "high"` reasoning-effort tier to `"xhigh"` (this model's template only defines `"xhigh"` as its top tier; some clients still send `"high"`).


## VRAM requirements and context length

This model uses a hybrid attention architecture — only 16 of 64 layers do standard attention (with a KV cache that grows with context); the other 48 are GDN/linear-attention layers with a fixed-size recurrent state that scales with **concurrent sequences** (`--max-num-seqs`), not context length. That makes VRAM scaling non-obvious: it's not a straight proportional relationship between GPU size and usable context.

Measured KV cache cost: ~32 KiB/token (fp8 kv cache, GQA with 4 KV heads, head_dim 256, 16 full-attention layers). Weights: ~17.6 GB / 16.4 GiB.

| GPU VRAM | `--max-model-len` | `--max-num-seqs` | Status |
|---|---|---|---|
| 32 GB | 98,304 | 64 | **Measured** (production config, `--gpu-memory-utilization 0.90 --kv-cache-dtype fp8`) |
| 24 GB | ~40,000–60,000 | 8 | Estimated, untested — lower `--max-num-seqs` is required; the recurrent-state cache is sized per concurrent sequence and otherwise leaves almost no room for KV cache on a 24GB card |
| 20 GB | ~16,000–32,000 | 1–2 | Estimated, untested — tight fit; expect to need aggressive tuning (minimal `--max-num-seqs`, possibly `--enforce-eager` to skip CUDA-graph memory) |

If you're on a smaller card, lower `--max-num-seqs` first (it doesn't cost you anything unless you're serving multiple concurrent requests) — vLLM's own memory profiler will tell you the exact fitting KV cache size, or fail loudly if `--max-model-len` doesn't fit. Reports of real numbers on 24GB/20GB cards welcome via a repo discussion.

## Benchmarks

Evaluated against production `cyankiwi/Qwen3.6-27B-AWQ-INT4` on the same 5090, via `lm-evaluation-harness` (local-chat-completions, `max_gen_toks=2048`, 100 samples) and direct throughput testing:

| Metric | Qwen3.6-27B-AWQ-INT4 | Qwen3.8-27B-AWQ-INT4 (this repo) |
|---|---|---|
| pp512 | ~3,300 tok/s | 3,349 tok/s |
| tg128 | 77.1 tok/s | 74.9 tok/s |
| GSM8K flexible-extract | 79% | **96%** |
| GSM8K strict-match | 78% | **89%** |
| KV cache @ 98,304 ctx | 188,059 tok, 1.91x concurrency | 237,924 tok, 2.42x concurrency |

## Quantization recipe

```yaml
default_stage:
  default_modifiers:
    AWQModifier:
      mappings:
      - smooth_layer: re:.*layers\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\.input_layernorm$
        balance_layers: ['re:.*self_attn.q_proj$', 're:.*self_attn.k_proj$', 're:.*self_attn.v_proj$']
      - smooth_layer: re:.*post_attention_layernorm$
        balance_layers: ['re:.*gate_proj$', 're:.*up_proj$']
      - smooth_layer: re:.*up_proj$
        balance_layers: ['re:.*down_proj$']
      duo_scaling: true
      n_grid: 20
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head, 're:.*visual.*', 're:model[.]visual.*', 're:.*mlp.gate$', 're:.*mlp.shared_expert_gate$']
      scheme: W4A16
```

## Serving (vLLM)

```
vllm serve abihsoro/Qwen3.8-27B-AWQ-INT4 \
  --max-model-len 98304 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --trust-remote-code
```
