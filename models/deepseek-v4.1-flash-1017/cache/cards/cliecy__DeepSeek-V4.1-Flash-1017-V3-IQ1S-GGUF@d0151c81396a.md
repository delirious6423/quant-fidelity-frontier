---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- deepseek
- moe
- expert-pruning
- llama.cpp
- experimental
---

# DeepSeek-V4.1-Flash-1017 · V3 · 384/384 experts per layer

All 384 experts, all routed experts IQ1_S (quantization-only step of the ladder).

Part of the expert-pruning / extreme-quantization test for running DeepSeek-V4.1-Flash-1017 on 2×RTX 4090 48G.
**Full report, all measurements, tools and data: [cliecy/DeepSeek-V4.1-Flash-1017-pruning-eval](https://huggingface.co/datasets/cliecy/DeepSeek-V4.1-Flash-1017-pruning-eval)** (kept up to date).
This is an experimental checkpoint, published so every step of the test can be reproduced; read its results first.

## How it was made

Base: mrfakename Q8_0. Tensor types: all routed experts (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) **IQ1_S** (has a CUDA MMQ kernel; IQ1_M does not); everything else as V0 (engram Q4_K, attention/shared experts Q6_K, router BF16, output Q6_K, token embedding Q8_0). Quantized from mrfakename Q8_0 (experts stored as MXFP4) with `LLAMA_QUANT_DITHER_ZEROS=1` (IQ1 added to the zero-dither patch) and a new imatrix computed on V0 with calib-all (zh/en/ja/code/chat, 452k tokens, ctx 2048).

```
see tools/quant_v3.sh:
LLAMA_QUANT_DITHER_ZEROS=1 llama-quantize --allow-requantize --imatrix NEW_IMATRIX --exclude-weights attn_output_a --tensor-type-file tools/types-iq1s.txt --output-tensor-type q6_k --token-embedding-type q8_0 Q8_0-00001-of-00010.gguf OUT.gguf IQ1_S 20
```

- `prune_experts.py` cuts, per layer and along the expert axis, `ffn_{gate,up,down}_exps`, the router `ffn_gate_inp`
  and the selection biases `exp_probs_b(_vl)`, and sets `deepseek41.expert_count` = 384 (llama.cpp has one global
  expert count, so every layer keeps the same number: its own top-384). A pruned expert can never be selected.
- Rankings come from `llama-reap` on V0 (routing count and REAP score = mean of g_e·‖f_e‖ over routed tokens), on the
  mixed calibration `calib-all` (452k tokens: EN/zh text, V0-generated chats, Japanese, C++) or the Chinese-coding
  calibration `calib-codezh` (1.0M tokens: GLM-5.3 generated Chinese coding chats + code in 8 languages). Per-layer
  dropped expert lists are in the eval repo under `prune-reports/`.
- OLD_IMATRIX = mrfakename Q2_K / calib-mix / ctx 512; NEW_IMATRIX = V0 / calib-all / ctx 2048.

## Results

Copied from the eval report (KLD / top-1 agreement vs V0; per-domain tables: eval-all = wiki/ja/code/chat, eval-v6 = code/codezh/wiki/ja/chat):

| ver | content | file | KLD mean | KLD 99 % | top-1 | Q&A | loops | verdict |
|---|---|---|---|---|---|---|---|---|
| V3 | all experts IQ1_S, 384, new imatrix | 224 GB | 0.446 | 4.07 | 77.8 % | pending | pending | **fail**: quantization alone costs this much |

| ver (PPL / KLD / top-1) | wiki | ja | code | chat |
|---|---|---|---|---|
| V3 | 3.701 / 0.633 / 75.7 % | 19.43 / 0.622 / 63.2 % | 1.457 / 0.231 / 91.9 % | 2.479 / 0.298 / 80.6 % |


## Files

Split with `llama-gguf-split --split-max-size 45G`; point llama.cpp at the first shard.

- `DeepSeek-V4.1-Flash-1017-V3-IQ1S-00001-of-00006.gguf`
- `DeepSeek-V4.1-Flash-1017-V3-IQ1S-00002-of-00006.gguf`
- `DeepSeek-V4.1-Flash-1017-V3-IQ1S-00003-of-00006.gguf`
- `DeepSeek-V4.1-Flash-1017-V3-IQ1S-00004-of-00006.gguf`
- `DeepSeek-V4.1-Flash-1017-V3-IQ1S-00005-of-00006.gguf`
- `DeepSeek-V4.1-Flash-1017-V3-IQ1S-00006-of-00006.gguf`

## Running

Needs the DeepSeek-V4.1 llama.cpp build used for [cliecy/DeepSeek-V4.1-Flash-1017-UD-IQ2XS-GGUF](https://huggingface.co/cliecy/DeepSeek-V4.1-Flash-1017-UD-IQ2XS-GGUF) (its README has the patch and
prebuilt binaries). Settings used for all tests (2 GPUs, engram table on CPU / page cache):

```
llama-server -m <first shard> -ngl 999 -fa on -sm layer -ts 48,52 -ot engram_embd.weight=CPU -lm none -lzm off -t 16 \
  -c 32768 -np 1 --jinja --chat-template-file deepseek-ai-DeepSeek-V4.1-Flash.jinja
```

Eval settings: KLD and top-1 agreement vs V0 logits with `llama-perplexity --kl-divergence`, ctx 2048, 16384 tokens per
domain; Q&A temperature 0.6; HumanEval-zh greedy, max 6000 tokens, every 2nd problem (82).
