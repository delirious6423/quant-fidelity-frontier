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

# DeepSeek-V4.1-Flash-1017 · V2 · 288/384 experts per layer

General model, 25 % of experts pruned by REAP score on the mixed calibration.

Part of the expert-pruning / extreme-quantization test for running DeepSeek-V4.1-Flash-1017 on 2×RTX 4090 48G.
**Full report, all measurements, tools and data: [cliecy/DeepSeek-V4.1-Flash-1017-pruning-eval](https://huggingface.co/datasets/cliecy/DeepSeek-V4.1-Flash-1017-pruning-eval)** (kept up to date).
This is an experimental checkpoint, published so every step of the test can be reproduced; read its results first.

## How it was made

Base: V0 (fixed UD-IQ2XS). V0 tensor types (UD-IQ2XS, zero-dither requant): routed experts `ffn_gate_exps`/`ffn_up_exps` IQ2_XS, `ffn_down_exps` IQ2_S; `engram_embd` Q4_K; attention (`attn_q_a/q_b/kv/output_a/output_b`) and shared experts Q6_K; `ffn_gate_inp` (router), `attn_compressor_*`, `indexer.*` BF16; `hc_*_fn` F32; output Q6_K; token embedding Q8_0. imatrix: mrfakename Q2_K on calib-mix (EN + Chinese), ctx 512.

```
python3 tools/prune_experts.py GGUF_PY V0 OLD_IMATRIX 288 OUT.gguf OUT-imatrix.gguf report.json routing/reap-calib-all.json
```

- `prune_experts.py` cuts, per layer and along the expert axis, `ffn_{gate,up,down}_exps`, the router `ffn_gate_inp`
  and the selection biases `exp_probs_b(_vl)`, and sets `deepseek41.expert_count` = 288 (llama.cpp has one global
  expert count, so every layer keeps the same number: its own top-288). A pruned expert can never be selected.
- Rankings come from `llama-reap` on V0 (routing count and REAP score = mean of g_e·‖f_e‖ over routed tokens), on the
  mixed calibration `calib-all` (452k tokens: EN/zh text, V0-generated chats, Japanese, C++) or the Chinese-coding
  calibration `calib-codezh` (1.0M tokens: GLM-5.3 generated Chinese coding chats + code in 8 languages). Per-layer
  dropped expert lists are in the eval repo under `prune-reports/`.
- OLD_IMATRIX = mrfakename Q2_K / calib-mix / ctx 512; NEW_IMATRIX = V0 / calib-all / ctx 2048.

## Results

Copied from the eval report (KLD / top-1 agreement vs V0; per-domain tables: eval-all = wiki/ja/code/chat, eval-v6 = code/codezh/wiki/ja/chat):

| ver | content | file | KLD mean | KLD 99 % | top-1 | Q&A | loops | verdict |
|---|---|---|---|---|---|---|---|---|
| V2 | REAP-25 %, 288 | 240 GB | 0.134 | 1.98 | 88.8 % | 6/6 | zh ok; ja/en see note | fails V1/V2 KLD line; code passes |

| ver (PPL / KLD / top-1) | wiki | ja | code | chat |
|---|---|---|---|---|
| V2 | 2.709 / **0.247** / 87.2 % | 13.33 / 0.115 / 82.6 % | 1.248 / **0.046 / 97.5 %** | 2.048 / 0.127 / 87.7 % |


## Files

Split with `llama-gguf-split --split-max-size 45G`; point llama.cpp at the first shard.

- `DeepSeek-V4.1-Flash-1017-V2-reap288-IQ2XS-00001-of-00006.gguf`
- `DeepSeek-V4.1-Flash-1017-V2-reap288-IQ2XS-00002-of-00006.gguf`
- `DeepSeek-V4.1-Flash-1017-V2-reap288-IQ2XS-00003-of-00006.gguf`
- `DeepSeek-V4.1-Flash-1017-V2-reap288-IQ2XS-00004-of-00006.gguf`
- `DeepSeek-V4.1-Flash-1017-V2-reap288-IQ2XS-00005-of-00006.gguf`
- `DeepSeek-V4.1-Flash-1017-V2-reap288-IQ2XS-00006-of-00006.gguf`

## Running

Needs the DeepSeek-V4.1 llama.cpp build used for [cliecy/DeepSeek-V4.1-Flash-1017-UD-IQ2XS-GGUF](https://huggingface.co/cliecy/DeepSeek-V4.1-Flash-1017-UD-IQ2XS-GGUF) (its README has the patch and
prebuilt binaries). Settings used for all tests (2 GPUs, engram table on CPU / page cache):

```
llama-server -m <first shard> -ngl 999 -fa on -sm layer -ts 48,52 -ot engram_embd.weight=CPU -lm none -lzm off -t 16 \
  -c 32768 -np 1 --jinja --chat-template-file deepseek-ai-DeepSeek-V4.1-Flash.jinja
```

Eval settings: KLD and top-1 agreement vs V0 logits with `llama-perplexity --kl-divergence`, ctx 2048, 16384 tokens per
domain; Q&A temperature 0.6; HumanEval-zh greedy, max 6000 tokens, every 2nd problem (82).
