---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags: [vllm, rtx-3090, speculative-decoding, gptq]
---
# Qwen3.8-27B "fast" single-user extras for one RTX 3090

Companion files for https://github.com/syv-ai/qwen38-27b-rtx3090 (single-user mode, ~114 tok/s at
default sampling / ~124 tok/s greedy on an RTX 3090 with vLLM 0.27.1). Drop them over a local copy of
the W4A16 AutoRound checkpoint prepared by that repo (`quant_embed.py` + shards 1-6 unchanged):

- `model-00007-of-00007.safetensors`: lm_head requantized to int4 (group 128, GPTQ calibrated on 300k of the
  model's own hidden states; +0.6% perplexity vs bf16, GSM8K unchanged) plus the other tensors of shard 7
- `model_extra_tensors.safetensors`: the MTP draft module requantized to int4 (GPTQ, calibrated) and a
  40,960-row int4 draft head (`mtp.draft_lm_head.*`)
- `mtp_draft_vocab_ids.pt` / `draft_vocab_ids.json`: the draft vocabulary, counted over 5.4M tokens of the
  model's own outputs (97.5% coverage of what it generates; 96% on code)
- `config.json`, `model.safetensors.index.json`: matching quantization config / weight map

Needs the vLLM patches from the repo (`patches/qwen3_5-mtp-draft-vocab.patch` at least). See the repo
README, section "Single-user: 124 tok/s", for what each piece buys and how the files were produced
(`drafter/`).
