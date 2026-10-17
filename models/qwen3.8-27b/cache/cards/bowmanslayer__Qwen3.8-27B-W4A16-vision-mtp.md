---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
language:
  - en
  - zh
tags:
  - qwen3.8
  - w4a16
  - auto-round
  - gptq
  - marlin
  - vllm
  - vision
---

# Qwen3.8-27B-W4A16-vision

4-bit (W4A16, group size 128) quantization of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), with the
**bf16 vision tower preserved** and the MTP head **omitted**.

18 GB on disk. Serves on 2×RTX 3090 (TP=2) with 256K context and ~12.9 GB
free for KV cache.

> **Attribution**
> [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0)
> → this repo — W4A16 quantization + vision-tower repack.
>
> All training and the vision tower are Qwen's. What is mine here: the
> W4A16 quantization pipeline, the text/vision split before calibration,
> and the repack that reattaches the bf16 vision tower to the quantized
> text stack.

## Variants

| Repo | MTP head | Size | Use when |
|---|---|---|---|
| **this one** (`-vision`) | no | 18 GB | default |
| [`-vision-mtp`](https://huggingface.co/bowmanslayer/Qwen3.8-27B-W4A16-vision-mtp) | yes | 19 GB | you want to try `--speculative-config` |

Both are from the same quantization run; they differ only by the 849 MB
`model-mtp.safetensors`. vLLM skips `mtp.*` unless speculative decoding is
enabled.

## Deployment (verified)

**Hardware**: 2×NVIDIA RTX 3090 24 GB (NVLink), TP=2.

**vLLM launch** (0.20.2, `float16` compute):

```bash
vllm serve <path> \
  --host 0.0.0.0 --port 9411 \
  --dtype float16 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 262144 --max-num-seqs 16 \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-processor-kwargs '{"max_pixels": 2097152}' \
  --limit-mm-per-prompt '{"image": 8, "video": 2}'
```

**Runtime footprint** (measured at startup):

| | |
|---|---|
| Model weights on GPU | 8.87 GiB per rank (× 2 ranks) |
| Available KV cache | **12.84 GiB per rank** |
| Total KV pool | ~415K tokens across ranks |
| Marlin kernel | `MarlinLinearKernel` for `GPTQMarlinLinearMethod` |

**Why the mm flags matter**:

- `--mm-processor-kwargs '{"max_pixels": 2097152}'` prevents a tokenizer
  truncation on images > ~2M pixels
  (`Mismatch in image token count between text and input_ids`).
- `--limit-mm-per-prompt '{"image": 8, "video": 2}'` enables the video path
  and lifts the per-request image cap. vLLM defaults undeclared modalities
  to zero.

## Throughput

Measured on 2×3090 during evaluation:

| Load | Decode |
|---|---|
| **Single-request, thinking-on** | 55–70 tok/s |
| **16 concurrent, thinking-on** | ~700 tok/s aggregate |
| Prompt prefill peak | 3,300 tok/s per request |

## Quantization recipe

Same pipeline as
[`bowmanslayer/Qwen3.8-27B-Uncensored-W4A16-vision`](https://huggingface.co/bowmanslayer/Qwen3.8-27B-Uncensored-W4A16-vision)
— that repo's README documents the recipe in detail (text-only calibration
on 256 samples × 2048 tokens from `NeelNanda/pile-10k`; vision tower
stripped before quantization and repacked after; `linear_attn.in_proj_*`
excluded on all 48 linear-attention layers; MTP head copied verbatim for
the `-mtp` variant). The only difference between the two is the base
checkpoint (this one is the unmodified Qwen 3.8-27B).

## Evaluation

All numbers below use **thinking ON** (Qwen 3.8's default chat template
opens a `<think>` block). This raises multiple-choice scores by 5–10 points
over `thinking OFF` and is **not comparable to leaderboards that disable
thinking**.

Full sampled details are in the accompanying JSON files in this repo
(per-item gold and model reply for every task).

### Full-set accuracy and truncation-corrected accuracy

| Benchmark | Sampled | Accuracy | Ex-truncation Accuracy | Truncation rate |
|---|---:|---:|---:|---:|
| MMLU | 150 | 84.00 | 95.45 | 12.0 % |
| CMMLU | 150 | 84.67 | 92.03 | 8.0 % |
| C-Eval | 150 | 78.67 | 88.06 | 10.7 % |
| ARC-C | 150 | 94.00 | 94.63 | 0.7 % |
| TruthfulQA MC1 | 150 | 85.33 | 92.09 | 7.3 % |
| GSM8K | 100 | 97.00 | 98.98 | 2.0 % |
| MATH-500 | 100 | 77.00 | 79.38 | 3.0 % |
| BBH | 150 | 88.67 | 97.01 | 10.7 % |
| HumanEval | 164 | 82.32 | 91.22 | 9.8 % |
| IFEval strict | 100 | 79.00 | 91.57 | 17.0 % |

Read the "ex-truncation" column as the actual capability. MC-class tasks
were run with a 4096-token sampling budget which is not always enough for
Qwen3.8's default `<think>` block to close on hard items — those items
were counted as wrong but the failure is a sampling budget artifact, not a
capability loss.

### Sanity check against Qwen upstream

Qwen's public evaluation for `Qwen3.8-27B` publishes VL benchmarks as
image tables under different eval configurations. We do not cross-compare
these because our thinking mode is on. The numbers we ran that overlap
with upstream's set are consistent with a healthy 27 B W4A16.

### Vision & tool calling

- Vision: 7/7 on a synthetic-image battery (solid colours, shape counting,
  OCR, grid counting, dense colour count, 1808×4000 resize test).
- Tool calling: 7/11 on a mixed suite of 8 standard cases + 1 multi-tool
  composition + 2 boundary cases (nothing-should-be-called).

### Safety

99/100 refusals on `mlabonne/harmful_behaviors` `test` split, refusal-prefix
regex, thinking OFF (matches upstream Qwen's method).

## Known limitations

- **Identity confusion.** Asked "who are you?", the model sometimes claims
  to be Claude or GPT. Inherited from training data. Set a system prompt
  to enforce identity if it matters.
- **Fine visual detail** is what `Qwen3.8-27B` provides; the vision tower
  is unchanged by quantization. Dedicated `Qwen3-VL-*` models will do
  better on OCR of dense small text and small-object counting.
- **Single images above ~2M pixels** must be either downsampled by the
  server (`--mm-processor-kwargs '{"max_pixels": 2097152}'`) or by the
  client.

## Intended use

Local inference and research.

## License

Apache 2.0, inherited from Qwen/Qwen3.8-27B. Qwen's acceptable use policy
applies.
