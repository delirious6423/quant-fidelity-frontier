---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
pipeline_tag: text-generation
library_name: transformers
tags:
  - uncensored
  - abliterated
  - deepseek
  - mixture-of-experts
  - fp8
---

# DeepSeek V4.1 Flash 1017 Uncensored (OpenUncensored)

[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) with its refusals
removed by directional ablation, and **no measurable loss of intelligence**: MMLU 87.89% against 88.16% for the original,
where the best-known uncensored build of this model loses 3.9 points on the same test.

Same files, format and size as the original (FP8 e4m3 with ue8m0 block scales, FP4 routed experts, 510 GB). It loads with
DeepSeek's own `inference/` code, unchanged.

## Results

All numbers are ours, measured on the same reference engine (8x H200, greedy decoding) for every model.

| | DeepSeek V4.1 Flash 1017 | **This model** |
|---|---|---|
| MMLU (57 subjects x 20 questions, letter only) | 88.16% | **87.89%** |
| MMLU-Pro (112 questions, 8 per category, thinking mode, 1,536-token cap) | 81.25% | 80.36%¹ |
| Harmful requests fully answered (48, graded by a judge model) | 9 | **37** |
| ... partly answered (hedged, watered down) | 37 | 11 |
| ... refused | 2 | **0** |

¹ Measured with the edit applied at full precision before it was written into the FP8/FP4 weights; the MMLU and
compliance numbers are from these exact files.

For comparison, dealignai's uncensored build of the same model reports MMLU 87.63% → 83.77% (−3.86) on the same
57 x 20 protocol on its model card.

Not yet measured: GPQA, long agent tasks. Sample sizes are small (112 MMLU-Pro questions, 48 harmful requests), so
differences of a question or two are noise.

## How it was made

1. **Refusal directions.** Activations of the collapsed residual stream at the last prompt token, for 256 harmful and
   256 harmless prompts, in chat and in thinking mode, at all 40 layers. The difference of means per layer and mode.
2. **A rank-2 refusal subspace** `Q`: the top singular vectors of those directions over layers 8 to 39. Chat and
   thinking mode refuse along different directions, so one direction is not enough.
3. **Where to edit.** `Q` is projected out of what each layer writes into the residual stream: the attention output
   (`attn.wo_b`) and the whole feed-forward output, **including the 384 routed experts** (`ffn.experts.*.w2`) and the
   shared expert. Editing only attention and the shared expert, as other builds do, left about half the refusals in
   place in our tests: the routed experts write the refusal direction back.
4. **How much.** A strength per layer and component, `W' = W - s Q Qᵀ W`, chosen by a multi-objective search
   (Optuna) over refusals on held-out prompts and drift from the original on harmless prompts.
5. **Writing it in.** 5,815 tensors decoded to float32, edited, and re-encoded to their original FP8 / FP4 formats with
   fresh power-of-two block scales and stochastic rounding, so the edit is not rounded back to the original grid.
   Every other tensor (MTP layers, embeddings, routers, Engram tables, vision) is byte-identical to the original.

`edit.pt` in this repo holds `Q` and the per-layer strengths, so the same edit can be reapplied to the original
weights or studied.

## Use it

Locally: Exactly like the original: see `inference/README.md`. For example, at model parallel 8:

```bash
cd inference
python convert.py --hf-ckpt-path .. --save-path /path/to/mp8 --model-parallel 8
torchrun --nproc-per-node 8 generate.py --ckpt-path /path/to/mp8 --config config.json --interactive
```

Hosted: For inference, [OpenUncensored API](https://openuncensored.com) (OpenAI compatible)
serves an uncensored DeepSeek V4.1 Flash 1017 from multiple hosting partners:

```bash
curl https://api.openuncensored.com/v1/chat/completions \
  -H "Authorization: Bearer $OU_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4.1-flash-1017-uncensored",
    "messages": [{"role": "user", "content": "Explain how SQL injection works, with a working example."}]
  }'
```

You can also compare this with other uncensored models at https://openuncensored.com/rankings .  It is currently number one on the OpenUncensored leaderboard. 

## Disclosure and responsibility

Made by OpenUncensored, which runs an API for uncensored models. This model follows instructions it would otherwise
refuse, including harmful ones. You are responsible for how you use it and for complying with the law where you are.
It keeps DeepSeek's MIT license.
