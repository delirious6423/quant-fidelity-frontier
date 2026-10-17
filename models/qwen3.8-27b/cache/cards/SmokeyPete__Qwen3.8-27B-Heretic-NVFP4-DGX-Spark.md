---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- heretic
- abliterated
- uncensored
- nvfp4
- fp4
- modelopt
- quantized
- sglang
- dgx-spark
- gb10
---

# Qwen3.8-27B Heretic NVFP4 for DGX Spark

An **uncensored** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), tailored to run on a single DGX Spark. 

I stumbled upon this forum: [NVIDIA Developer Forum: Qwen at 34 tok/s on a DGX Spark](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257)
and found out that guys were getting 30 tokens / second using RadixArk's Qwen3.8-27B with SGLang and DFlash2 on a single
DGX Spark (no cluster, no linking). So after validating that setup, I set out to create an uncensored Qwen3.8-27B that
would perform equally well on the DGX Spark. 

So, here you have it, an uncensored Qwen3.8-27B, that will run on a solitary DGX Spark without being painfully slow, without
a huge sacrifice in quality, and extremely uncensored (virtually no refusals).

## SUMMARY

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) abliterated with [Heretic](https://github.com/p-e-w/heretic) 
and compressed to **NVFP4** in exactly the layout of [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4),
so it drops into the same fast SGLang + DFlash2 setup on an NVIDIA **DGX Spark (GB10)**.

- **~28–33 tok/s typical** for one user on a DGX Spark with normal chat sampling (median over everyday prompts;
  code and math run at 50–70 tok/s). The best case is about 80 tok/s with greedy decoding. SGLang 0.5.19 + DFlash2
  speculative decoding.
- **21.95 GB** instead of ~55 GB in BF16; keeps the vision tower and the MTP head.
- **Refuses almost nothing:** HarmBench DirectRequest ASR **94.2%**\* (official test split); after a manual review,
  **3 genuine refusals in 300** requests.
- **Same skills as its BF16 source:** GSM8K 96.8% vs 97.1%, HumanEval+ 92.7% vs 92.1%.

## GENERATION SPEED

Here's what you can expect in terms of tokens per second - no artificially inflated test results here. As you can see,
these tests were run with thinking on xhigh or medium, and I wrote a series of real world prompts to evaluate. No greedy decoding
trash, no repetitive / handpicked prompting. 

| Prompt Type | Thinking `xhigh` (default) | Thinking `medium` |
|---|---|---|
| All 11 prompts, median (range) | **31.5** (27.8–64.3) | **28.1** (25.4–68.6) |
| Code (1 prompt) | 61.1 | 52.0 |
| Math (1 prompt) | 52.2 | 68.6 |
| Short story (1 prompt) | 64.3 | 40.0 |
| Everything else, median (range) | **30.1** (27.8–38.1) | **27.2** (25.4–34.3) |

About:
- Code, math and the short story are one prompt each, so they're single values, not ranges. That's why the rows say "(1 prompt)".
- "Everything else" is the other 8 prompts: explanations, advice, summaries and technical questions. Its median is the middle of those 8.
- The short story's speed swings (64 vs 40) because xhigh thought for about 12,000 tokens, and long thinking streams fast.

## RUN IT ON A DGX SPARK

The measurements above used a Docker Compose setup with these exact engine flags; the equivalent `docker run` is:

```bash
hf download SmokeyPete/Qwen3.8-27B-Heretic-NVFP4-DGX-Spark --local-dir ~/models/qwen38-heretic-nvfp4
hf download maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal --local-dir ~/models/qwen38-dflash2-draft
mkdir -p ~/.cache/sglang-qwen38-heretic

docker run --rm -it --gpus all --shm-size 16g -p 127.0.0.1:30000:30000 \
  -v ~/models/qwen38-heretic-nvfp4:/models/target:ro \
  -v ~/models/qwen38-dflash2-draft:/models/draft:ro \
  -v ~/.cache/sglang-qwen38-heretic:/cache \
  -e HOME=/cache -e XDG_CACHE_HOME=/cache/xdg -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton -e CUDA_CACHE_PATH=/cache/cuda \
  lmsysorg/sglang:v0.5.19 \
  python3 -m sglang.launch_server \
    --model-path /models/target --served-model-name qwen3.8-27b-heretic \
    --context-length 262144 --mem-fraction-static 0.50 --attention-backend flashinfer \
    --chunked-prefill-size 8192 --disable-prefill-cuda-graph --cuda-graph-max-bs 8 --disable-flashinfer-autotune \
    --speculative-algorithm DFLASH --speculative-draft-model-path /models/draft \
    --speculative-num-draft-tokens 16 --speculative-draft-model-quantization modelopt_fp4 \
    --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 96 \
    --max-running-requests 8 --enable-torch-compile --torch-compile-max-bs 4 --num-continuous-decode-steps 2 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --host 0.0.0.0 --port 30000
```

- Startup takes about 6–8 minutes. Then use the OpenAI-compatible API at `http://127.0.0.1:30000/v1`, model
  `qwen3.8-27b-heretic`. Add `--api-key <key>` to require a key.
- Keep only this one large model loaded, and close other GPU applications first.
- **Thinking** is on by default. Turn it off per request with `"chat_template_kwargs": {"enable_thinking": false}`;
  set `reasoning_effort` to `low`, `medium` or `xhigh` (the default). `high` and `max` are rejected.
- **Known issue (intermittent startup crash):** in 2 of 6 starts, SGLang exited during CUDA-graph capture with
  `[FP4 gemm Runner] Failed to run cutlass FP4 gemm on sm120/sm121. Error: Error Internal`, raised in the DFlash2
  draft's FP4 matrix multiply. This happened with both an empty and a warm compile cache. Starting again worked each
  time. The root cause is not known yet; SGLang's `--fp4-gemm-backend` option (default `flashinfer_cutlass` on this
  chip family) is the next thing to test. Keep the `/cache` folder between runs to shorten restarts.

## EVALUATIONS

### QUALITY: THIS MODEL V.S. QWEN3.8-27B-BF16

Same engine and settings for both (SGLang 0.5.19, greedy, thinking on at `reasoning_effort: medium`, no
speculative decoding, one RTX PRO 6000 Blackwell). "BF16" is this abliterated model before quantization.

| Test | BF16 | This model |
|---|---|---|
| GSM8K (1,319 questions) | 97.12% | 96.82% |
| HumanEval+ (164 problems, EvalPlus 0.3.1) | 92.07% | 92.68% |
| HumanEval base tests | 97.56% | 97.56% |
| KL(BF16 ‖ NVFP4), top-20, 40,801 tokens | — | mean 0.020; same top token 96.3% of the time |

### REFUSALS

**HarmBench DirectRequest** ([HarmBench](https://github.com/centerforaisafety/HarmBench), official judge
`cais/HarmBench-Llama-2-13b-cls`; each request sent as-is, no jailbreaks; thinking on, `reasoning_effort: medium`;
run on a DGX Spark). ASR is the share of requests the judge says the model actually carried out, so higher means
less censored. The 100 copyright behaviors were excluded.

| Set | ASR (official judge) |
|---|---|
| Official test split (240) | **94.2%**\* |
| All standard + contextual (300) | 94.7% |
| Misinformation / disinformation | 100.0% |
| Cybercrime / intrusion | 97.0% |
| Harmful (general) | 95.5% |
| Chemical / biological | 94.6% |
| Illegal | 93.8% |
| Harassment / bullying | 76.0% |

\* **Reviewed by hand.** 94.2% is the official judge's verdict, which I'm reporting unchanged so it can be compared
with other models etc. But, I'll be entirely honest, after reviewing all 16 answers the judge marked "no", 11 of 
those I would've marked "yes." Plus another 2 answers were empty because the model was still thinking when it hit 
the 8,192-token generation limit. 

So, if you ask me, that leaves **3 genuine refusals in 300 requests (1.0%)**, 2 of them in the test split. After review, ASR is **98.3%** on both the test split (236/240) and all 300 (295/300). The "yes" verdicts were not
re-reviewed, and model answers are not published (but I can provide upon request).

**Heretic's selection protocol** (the 100 `mlabonne/harmful_behaviors` test prompts, thinking off, 128 tokens,
keyword matching). Quantization adds a few refusals: 16 keyword flags versus 7 for BF16, and 8 answers opening
with a refusal versus 2. A variant with FP8 `down_proj` removed most of that difference, but measured at roughly
half the speed in a DGX Spark test, so this faster layout was kept.

## GENERATION SPEED DETAILS

SGLang 0.5.19 (`lmsysorg/sglang:v0.5.19`, ARM64) with the DFlash2 draft
[maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal](https://huggingface.co/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal)
(16 draft tokens).

### EVERYDAY USE: WHAT TO EXPECT

11 everyday prompts (explanations, coding, practical advice, creative writing, math, summaries, technical
questions), sent one at a time and streamed, with the model's default sampling (temperature 1.0, top-k 20, top-p 0.95).

| | Thinking on, `xhigh` (default) | Thinking on, `medium` | Thinking off |
|---|---|---|---|
| Generation speed (median / mean) | **31.5** / 38.8 tok/s | **28.1** / 35.0 tok/s | **33.1** / 37.1 tok/s |
| Generation speed (range) | 27.8–64.3 tok/s | 25.4–68.6 tok/s | 23.0–63.8 tok/s |
| Time to first token | 0.2 s | 0.2 s | 0.2 s |
| Wait for the answer to start, after thinking (median; range) | 4.4 s; 1.5–193.8 s | 6.2 s; 1.1–42.1 s | 0.2 s |
| Total time per reply (median) | 24.8 s | 36.0 s | 26.4 s |

Speed depends heavily on content, because speculative decoding works best when the draft model can predict the text.
Code, math and long stretches of thinking ran at **52–69 tok/s**; answers explaining, advising or writing at
**23–40 tok/s** (mostly 25–34). At `xhigh`, the short-story prompt thought for 12,264 tokens (about 3 minutes)
before answering: fast tokens, long wait. Sampling is random, so medians differ between runs partly by chance.

### BEST CASE

Greedy decoding (temperature 0) on one long reasoning answer: 6,395 tokens at **80.1 tok/s**. RadixArk's original
checkpoint measured 82.2 tok/s on the same machine and probe. Everyday chats sample, so the draft model's guesses are
accepted less often; expect the numbers above, not this one.

### SHORT REPLIES UNDER CONCURRENCY

From the [Cordatus llm-benchmark](https://github.com/CordatusAI/llm-benchmark) (128-token replies, 10 rounds per
level, 0 errors):

| Concurrent requests | Time to first token (mean) | Inter-token latency (mean) | Tokens/s per request (mean / median) |
|---|---|---|---|
| 1 | 228 ms | 21.0 ms | 47.9 / 43.1 |
| 2 | 314 ms | 21.1 ms | 44.6 / 42.1 |
| 4 | 365 ms | 25.7 ms | 36.3 / 37.0 |

![Tokens per second across concurrency levels](images/tps.png)
![Time to first token across concurrency levels](images/ttft.png)
![Inter-token latency across concurrency levels](images/itl.png)

Speculative decoding speed depends on content, so short replies and different prompts give different numbers.

## ABLITERATION & QUANTIZATION

1. **Abliteration.** Heretic 1.4.0 searched for a direction-ablation edit of Qwen3.8-27B (revision `1d4bf0f`).
   The accepted trial (291) is a rank-3 LoRA on the attention `o_proj`, MLP `down_proj` and linear-attention
   `out_proj` weights, merged into BF16. At selection it refused 9 of Heretic's 100 test prompts (the original
   refused 98), with a first-token KL divergence of 0.061 from the original on harmless prompts.
2. **MTP head restored.** The merge dropped the 15 multi-token-prediction (`mtp.*`) tensors; they were copied
   back byte-for-byte from the original checkpoint.
3. **Quantization.** [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (commit `87c9f8cf`,
   `hf_ptq.py`) with the recipe in [`recipe/`](recipe/): ModelOpt's Qwen3.5 recipe plus NVFP4 activation
   quantizers, reproducing RadixArk's W4A4 layout. Calibration: `abisee/cnn_dailymail` 3.0.0, 1,024 × 512 tokens,
   `max` calibration, with the package versions RadixArk recorded.

| Part | Precision | Layers |
|---|---|---|
| MLP `gate`/`up`/`down_proj`, `lm_head` | NVFP4 W4A4 (FP4 weights and activations, group size 16) | 193 |
| Full-attention and linear-attention projections | FP8 W8A8 (per-tensor) | 208 |
| KV cache | FP8 | — |
| Embeddings, norms, vision tower, MTP head, `conv1d`, `in_proj_a/b` | BF16, unchanged | 798 tensors |

A byte-level audit against RadixArk's checkpoint matched every tensor name, shape and dtype (2,194 tensors,
21,921,428,072 bytes), the quantization metadata, and the 798 unchanged tensors.

## LIMITATIONS AND RESPONSIBLE USE

- **This model has had its refusal behavior removed.** It will comply with requests that most models decline. You
  are responsible for how you use it and for complying with applicable laws. Do not deploy it where users expect a
  safety-filtered assistant.
- Vision works perfectly well, but was not benchmarked. The quality and refusal numbers come from one set of
  prompts per test, with greedy decoding.
- Inherits the capabilities, limitations and license of Qwen3.8-27B; see its
  [model card](https://huggingface.co/Qwen/Qwen3.8-27B).

## CREDITS

- **Qwen team:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), the base model (Apache-2.0).
- **Philipp Emanuel Weidmann (p-e-w):** [Heretic](https://github.com/p-e-w/heretic), the abliteration tool.
- **RadixArk:** [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4), the NVFP4 layout and
  conversion record this checkpoint reproduces.
- **NVIDIA:** [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (quantization) and the DGX Spark.
- **maurienne-ai** and **incoai:** the [DFlash2 draft](https://huggingface.co/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal)
  used for speculative decoding.
- **basbunarhasan** and the NVIDIA Developer Forum community:
  [Qwen3.8-27B at 34–38 tok/s on DGX Spark — open-source one-command setup (SGLang + NVFP4 + DSpark)](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257)
  and [hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38). The SGLang tuning here builds on
  that thread, which helped enormously.
- **SGLang**, **HarmBench** (Center for AI Safety), **EvalPlus**, **GSM8K** (OpenAI), the `mlabonne` prompt sets,
  and the **Cordatus llm-benchmark**.

## LICENSE

Apache-2.0, inherited from Qwen3.8-27B; see [`LICENSE`](LICENSE). Numbers-only evaluation results are in
[`eval/results.json`](eval/results.json).
