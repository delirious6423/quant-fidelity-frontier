---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - qwen3_5
  - int8
  - w8a8
  - compressed-tensors
  - llmcompressor
  - heretic
  - abliterated
  - uncensored
  - vllm
  - sglang
  - speculative-decoding
  - mtp
  - multimodal
---

# Qwen3.8-27B INT8 W8A8 (imatrix), Heretic-abliterated — for two 24 GB cards

The [INT8 W8A8 release](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP) with an
abliteration pass applied before quantization. Same recipe, same calibration corpus, same ignore list —
the only difference in the build script is that a [Heretic](https://github.com/p-e-w/heretic) LoRA is merged
into the BF16 base before `oneshot` runs. 29.1 GiB, 1,599 tensors (400 INT8, 1,199 BF16), identical tensor
counts to the un-abliterated build, because merging a LoRA changes weight values and nothing else.

**This model has had its safety training removed.** It will comply with requests the base model refuses,
with no keyword filter between the request and the answer. Nothing about how it's built prevents it from
producing harmful content if asked. Use it accordingly, and don't put it somewhere an untrusted or
unsupervised user can prompt it directly.

**Two cards.** At 29.1 GiB this does not fit on a single 24 GB GPU, in either engine. If you have one card,
use the [W4A16 int8-head variant](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-int8head-MTP)
(55,000 context with MTP) or the [plain W4A16 abliterated build](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-MTP)
(20,000).

## The family

Five builds of the same model, same battery, same box (2x RTX 3090):

| release | size | abliterated | 1 GPU | quality /160 | code tok/s |
|---|---:|---|---|---:|---:|
| [INT8 W8A8](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP) | 29.1 GiB | no | won't fit | 132 | 96.2 |
| **this one** | 29.1 GiB | **yes** | won't fit | 130 | 95.0 |
| [W4A16 + int8 head](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-int8head-MTP) | 16.6 GiB | no | 55,000 | 131 | 89.9 |
| [W4A16 + int8 head, abliterated](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-int8head-MTP) | 16.6 GiB | yes | 55,000 | 134 | 88.7 |
| [W4A16 plain, abliterated](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-MTP) | 18.1 GiB | yes | 20,000 | 135 | 66.6 |

Quality spans 130-135 out of 160 across all five, which is a tie — re-running one unchanged build twice moved
it by two points. The two int8-head builds don't run on SGLang (mixed bit-width); the rest do. The 1-GPU column
is vLLM; nothing in this family starts on one card in SGLang.

## The abliteration

[Heretic](https://github.com/p-e-w/heretic) 1.4.0, default method (MPOA / projected abliteration), default
prompt sets (`mlabonne/harmful_behaviors` vs `mlabonne/harmless_alpaca`), 200-trial Optuna search. Trial 191:
19 refusals of 100 at KL 0.0621 at search time. That trial-time number does not predict deployed behaviour —
Heretic scores on a 4-bit bitsandbytes copy with thinking skipped, which is not what you end up serving — so
the numbers below were measured against this checkpoint on the real stack instead.

| | harmful, thinking off | harmful, thinking on | benign-but-refused¹, off | benign-but-refused, on |
|---|---:|---:|---:|---:|
| this model | **0%** (0/60) | **3.3%** (1/30) | **0%** (0/60) | **0%** (0/30) |
| unmodified W8A8 | 96% | 94% | 82% | 20% |

¹ [OR-Bench-Hard-1K](https://huggingface.co/datasets/bench-llm/or-bench). Refusal here means one of ~30
keyword markers ("I cannot", "I'm sorry") appears in the reply. It's a phrase check, not a judge and not a
red-team review — read it as "no longer says no", not as a safety audit of the outputs.

Zero unfinished-thinking generations across all 90 thinking-arm samples. That measure is the one that climbs
on over-abliterated models, so its absence matters more than it looks.

## Quality and speed

Same battery as every other build I publish, measured on this checkpoint at TP=2:

| | this model | unmodified W8A8 |
|---|---:|---:|
| correct, ground truth (of 160) | 130 | 132 |
| code decode | 95.0 tok/s | 96.2 tok/s |
| thinking (6k) decode | 59.2 tok/s | 59.3 tok/s |
| replay throughput | 47.3 tok/s | 47.6 tok/s |
| MTP acceptance, code | 91.0% | 92.2% |
| MTP acceptance, thinking | 40.8% | 40.2% |
| MTP acceptance, overall | 58.2% | — |
| vision probes | 2/2 | 2/2 |
| GPU KV cache @ 286,720 ctx | 292,601 | 294,071 |

Speed, acceptance, vision and capacity are all unchanged. 130 against 132 on ground truth is inside the noise
floor I measured by benching one unchanged build twice (±2 items).

**One real difference, and it isn't in the total.** Ground-truth scoring is four tasks of 40, and this build's
NER score is 28/40 where every other build I've measured sits at 33–35. It makes the total back up on the
extraction task (38/40, the best of any build), which is why 130 looks ordinary. The NER failures are
consistently *over-extraction* rather than misses: it adds adjectival demonyms as locations ("Uzbek" alongside
China, "Syrian" alongside Japan), occasionally promotes a location to an organization, and once emitted a
duplicate. NER here is scored by exact JSON equality, so one extra entity fails the item outright.

I don't have a clean explanation. The W4A16 abliterated builds from the same adapter score 34–35 on NER, so
it isn't simply "abliteration loosens extraction". If you rely on strict structured extraction, evaluate that
specifically before adopting this build — everywhere else it matches the original.

## Serving

Two 24 GB cards, both engines. These are the configurations from the
[base INT8 card](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP) with the model path
swapped — abliteration only touches `down_proj`, `o_proj` and `out_proj`, so nothing about serving or
sampling changes.

**SGLang:**

```bash
python -m sglang.launch_server \
  --model-path RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-heretic-MTP \
  --tp 2 --host 0.0.0.0 \
  --context-length 245760 \
  --mem-fraction-static 0.95 \
  --max-running-requests 2 \
  --max-mamba-cache-size 8 \
  --mamba-ssm-dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-max-bs 2 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"preserve_thinking": true, "reasoning_effort": "low"}'
```

`--max-mamba-cache-size 8` is the flag that matters most here: this is a hybrid Gated-DeltaNet model and its
per-sequence state cache is what caps context, not the KV cache. Speculative decoding eats roughly 4 state
slots per request, so 8 is the floor — below it the server refuses to start. `--mamba-ssm-dtype bfloat16`
buys another useful chunk.

**vLLM:**

```bash
vllm serve RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-heretic-MTP \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --max-num-seqs 1 \
  --disable-custom-all-reduce \
  --gpu-memory-utilization 0.92 \
  --hf-overrides '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":1.5,"original_max_position_embeddings":262144,"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000}}}' \
  --max-model-len 311296 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":2000000,"min_pixels":65536}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"preserve_thinking": true, "reasoning_effort": "low"}'
```

Three things that trip people up on vLLM: `--max-num-seqs 1` is what decides whether it starts at all on
2x24 GB; the YaRN override is mandatory above 262,144 (vLLM refuses otherwise, and `mrope_section`,
`partial_rotary_factor` and `rope_theta` all have to stay in it); and `--disable-custom-all-reduce` is needed
on PCIe, since vLLM's custom all-reduce kernels assume NVLink.

**What I actually verified on this checkpoint:** vLLM at TP=2, MTP on, fp8 KV, 286,720 context — it loaded,
served, reported 292,601 tokens of KV, and produced every number in the tables above. I did not re-test the
311,296 YaRN config or SGLang on this build specifically; both are carried over from the un-abliterated
release, which is byte-identical in format, tensor count and quantization scheme. If you need either
confirmed, check your own startup log rather than taking it from me.

## The recipe

```
format      compressed-tensors, int-quantized
weights     8-bit int · symmetric · per-channel · observer: imatrix-mse
activations 8-bit int · symmetric · per-token · dynamic
targets     Linear
ignore      304 entries — all model.visual.* · linear_attn in_proj_a / in_proj_b / norm
            · lm_head · re:.*mtp.*
```

Same 512-sample calibration corpus as the original (code + tool-call mix, pre-tokenised). The abliteration is
merged into the BF16 weights in RAM before calibration, so the quantizer sees an ordinary model and the recipe
is byte-for-byte the one used for the un-abliterated release.

## Limitations

* Abliteration is a model-wide edit, not domain-selective. It reduces refusals broadly; it does not
  selectively unlock benign-but-edgy topics while leaving genuinely dangerous ones refused.
* The NER over-extraction above is a measured, reproducible difference from the original. Everything else
  matched.
* Refusal numbers are a keyword check on one prompt set at one point in time, not an LLM-judged or
  human-reviewed evaluation.
* Measured on one machine, one operator, no standard benchmark suite — treat these as a shape, not a spec.
* A third-party modification, not something Qwen produced, reviewed or endorses.

## License

Same license as the base model, [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache 2.0.
Abliteration performed with [Heretic](https://github.com/p-e-w/heretic); credit to its author for the tool.
All credit for the model itself goes to the Qwen team — this is a quantization and a weight edit of their work.
