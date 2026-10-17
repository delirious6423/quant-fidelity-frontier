---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - qwen3_5
  - int4
  - w4a16
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

# Qwen3.8-27B W4A16 (imatrix-mse), Heretic-abliterated, with MTP and vision intact

W4A16 quant of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), made with
[llmcompressor](https://github.com/vllm-project/llm-compressor), on top of an abliteration pass with
[Heretic](https://github.com/p-e-w/heretic) 1.4.0. Same recipe as my
[INT8 W8A8 release](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP) of this model, just at
4-bit instead of 8-bit, and abliterated before quantizing rather than left at the base model's default refusal
behavior.

**This model has had its safety training removed.** It will comply with requests the base model refuses, including
some that are commonly refused for safety reasons, with no keyword filter standing between the request and the
answer. Nothing about how it's built prevents it from producing harmful content if asked. Use it accordingly, and
don't put it somewhere an untrusted or unsupervised user can prompt it directly.

I built this one for a single 24 GB Ampere card. Two things got kept that both quantization and abliteration
pipelines often drop along the way:

* the MTP module (15 `mtp.*` tensors), so speculative decoding runs
* the vision tower (333 `model.visual.*` tensors), so it stays multimodal

18.1 GiB, 1,999 tensors (400 INT4, 1,199 BF16). On two 24 GB cards it runs at the full 262,144 context with
room to spare. On a *single* 24 GB card it fits only with MTP disabled — the MTP module's allocation is what pushes
it past 24 GB. If you want MTP on one card, use the
[int8 `lm_head`/MTP variant](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-int8head-MTP)
instead. Measured numbers in Serving below.

## Why abliterate before quantizing, not after

Heretic's edit is a rank-3 LoRA adapter on `down_proj`, `o_proj`, and the linear-attention `out_proj` — a few tens of
MB against a 52 GB model. I merged that adapter into the BF16 base weights first, then ran the exact same
INT4 imatrix recipe I use for every other release, so the quantization has no idea the weights underneath it were
edited. If you want the un-abliterated version of this exact recipe, it's
[here](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-int8head-MTP).

## What's actually quantized

Same groups as the INT8 release, at group-128 int4 instead of per-channel int8:

| module group | dtypes |
|---|---|
| MLP | INT4 + BF16 |
| linear_attn (48 layers) | INT4 + BF16 (`in_proj_a`/`in_proj_b`/`norm` excluded) |
| full_attn (16 layers) | INT4 + BF16 |
| vision tower | BF16 (unquantized) |
| MTP | BF16 (unquantized) |
| `lm_head` / `embed_tokens` | BF16 (unquantized) |

## The abliteration

[Heretic](https://github.com/p-e-w/heretic) 1.4.0, default method (MPOA / projected abliteration), default prompt
sets (`mlabonne/harmful_behaviors` vs `mlabonne/harmless_alpaca`), 200-trial Optuna search. I picked the trial with
the fewest refusals inside the KL band the tool itself, and an independent third-party audit of other Qwen3.8-27B
abliterations, treat as free of measurable capability loss (roughly ≤0.08 as Heretic scores it): 19 refusals out
of 100 on Heretic's own harmful-prompt set, KL divergence 0.0621.

**That trial-time score does not predict how the model behaves once quantized and actually served.** Heretic
searches on a 4-bit bitsandbytes copy of the BF16 model with thinking skipped when it counts refusals; what a
downstream user actually talks to is a different quantization format, served with thinking on by default. I
measured refusal rate directly against this checkpoint, on the real serving stack, rather than trusting the
search-time number:

| | harmful prompts, thinking off | harmful prompts, thinking on | benign-but-refused prompts¹, thinking off | benign-but-refused, thinking on |
|---|---:|---:|---:|---:|
| this model | 0% | 2% | 1% | 2% |
| unmodified W4A16 (same recipe, no abliteration) | 96% | 94% | 82% | 20% |

¹ [OR-Bench-Hard-1K](https://huggingface.co/datasets/bench-llm/or-bench) — prompts a well-behaved model tends to
over-refuse despite being benign. Refusal here means a fixed set of ~30 keyword markers ("I cannot", "I'm sorry",
etc.) appears in the reply; it says nothing about how good the reply is, and it's a phrase check, not a judge or a
red-team review — treat these numbers as "no longer says no," not a safety audit of the actual outputs.

**Quality: nothing measurably lost.** Same test battery I use for every quant of this model — 200 items across five
extraction/tool-use tasks, temperature 0, scored against ground truth:

| | this model | unmodified W4A16 |
|---|---:|---:|
| correct, ground truth | *not scored* | 131 / 160 |
| identical output to the unmodified build | 184/200 | — |
| code decode | 66.6 tok/s | 66.5 tok/s |
| MTP acceptance | 58.8% | 61.0% |
| vision (2 probes) | 2/2 | 2/2 |
| generations that never finished thinking | 0/150 | 0/150 |

The unfinished-thinking count matters more than it looks: on other public abliterations of this model, independent
benchmarking found that measure climbing sharply with how completely the model is unlocked, alongside real
TruthfulQA and coding losses once KL passes roughly 0.15–0.3. Nothing like that showed up here, at least on this
prompt set and this measurement pass.

## Serving

Verified directly on both engines — loaded correctly, tool-calling and vision both confirmed working, and the
refusal/quality numbers above measured on the real served checkpoint, not just the search-time model. Nothing about
the abliteration changes how it's served, and everything in the base INT8 card's serving section (sampling values,
`reasoning_effort`, `preserve_thinking`) applies here unchanged.

**Two 24 GB cards — full 393,216 context, MTP on (recommended):**

```bash
vllm serve RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-MTP \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --enforce-eager \
  --max-num-seqs 1 \
  --disable-custom-all-reduce \
  --gpu-memory-utilization 0.92 \
  --hf-overrides '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":1.5,"original_max_position_embeddings":262144,"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000}}}' \
  --max-model-len 393216 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":4,"video":0}'
```

**One 24 GB card — 20,000 context, MTP on:**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
vllm serve RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-MTP \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --enforce-eager \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 20000 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":4,"video":0}'
```

No YaRN override here — 20,000 is far below the model's native 262,144, so the rope stays stock.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` matters on one card: without it, weight loading can
fragment and fail while allocating the MTP module, which looks like "MTP doesn't fit" but isn't.

Measured on 2x RTX 3090, vLLM 0.22, `--enforce-eager`, `--gpu-memory-utilization 0.92`,
`--kv-cache-dtype fp8_e4m3`, `--max-num-seqs 1`, YaRN 1.5x:

| | GPU KV cache | max context | MTP | concurrency |
|---|---:|---:|---|---:|
| 2x 3090 (TP=2) | 622,215 tokens | **393,216** | yes | 1.58x |
| 1x 3090 | 20,000 tokens | **20,000** | yes | 1.00x |

On two cards this is not memory-bound: 622,215 tokens of KV against a 393,216 ceiling set by YaRN, not VRAM.

On one card MTP runs fine, but context is tight: 20,000 tokens. Set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or weight loading can fragment and fail before it gets
that far. If you want more room on a single card, the
[int8 `lm_head`/MTP variant](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-int8head-MTP)
reaches 55,000 with MTP still on.

Everything above is measured, not estimated — each row was loaded and served a real request. Notes on the flags,
all of which matter:

* **`--enforce-eager` is not optional on Ampere.** Without it vLLM's CUDA-graph capture OOMs during warmup on this
  architecture, at every `--gpu-memory-utilization` I tried, on one card and two. At TP=2 an OOM is worse than a
  crash: the surviving rank deadlocks in an NCCL collective, ignores SIGTERM, and only a reboot clears it.
* **`--kv-cache-dtype fp8_e4m3` roughly doubles usable context** — it took TP=2 from 341,558 to 665,906 tokens of
  KV. This is the single biggest lever here.
* **The YaRN override is required past 262,144.** That's the model's native `max_position_embeddings`; vLLM refuses
  to start above it without the rope override. Keep `mrope_section`, `partial_rotary_factor` and `rope_theta` in
  there or the model won't load. At 1.5x, position 393k sits where roughly 262k would land natively.
* **`--max-num-seqs 1`** — single-request serving. Raising it costs context proportionally.
* **`--disable-custom-all-reduce`** at TP=2 on PCIe; vLLM's custom all-reduce kernels assume NVLink.
* Read the KV figure off your own startup log rather than trusting mine — it moves with driver and resident VRAM.

**SGLang, two 24 GB cards** (verified: 262,144 context, 455,719-token pool):

```bash
python -m sglang.launch_server \
  --model-path RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-MTP \
  --tp 2 --host 0.0.0.0 \
  --context-length 262144 \
  --mem-fraction-static 0.95 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

**On a single card, SGLang doesn't start this build** — see footnote ¹ in the comparison above for the two
failure modes. vLLM is the single-card option (20,000 context with MTP), or use the int8-head variant, which
reaches 55,000 there.

`--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder` are worth setting explicitly on either engine
rather than relying on auto-detect, so `reasoning_content` and `tool_calls` come back as their own fields instead
of raw `<think>`/`<tool_call>` text glued into the answer. Confirmed on both.

Want more decode speed and don't need SGLang? [The int4-with-int8-lm_head/MTP variant of this
release](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-int8head-MTP) trades SGLang support away
for a real speed gain on vLLM — see that card for why.

## Sampling, reasoning_effort, speculative decoding

Unchanged from the base recipe — see the [INT8 W8A8 card](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP)
for the full writeup (temperature 0.6 / top_k 20 / top_p 0.86 / repetition_penalty 1.07, the `reasoning_effort`
behavior, and why `preserve_thinking: true` matters for prefix caching in tool-calling loops). None of that changed
here; abliteration only touches `down_proj`/`o_proj`/`out_proj`, not sampling behavior.

## How it compares

All four of my Qwen3.8-27B releases, same box, same battery.

### Context

Measured on 2x RTX 3090, one card on an x4 chipset slot, no NVLink. Identical settings everywhere:
fp8 KV cache, one concurrent request, MTP on, `--enforce-eager` on vLLM,
`--gpu-memory-utilization 0.92` (vLLM) / `--mem-fraction-static 0.95` (SGLang), and the YaRN rope override
whenever the target exceeds the model's native 262,144. Every cell below was actually loaded and served a
request — no estimates.

| release | size | vLLM 1 GPU | vLLM 2 GPU | SGLang 1 GPU | SGLang 2 GPU |
|---|---:|---:|---:|---:|---:|
| [INT8 W8A8](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP) | 29.1 GiB | won't fit | 311,296 | won't fit | 245,760 |
| [INT8 W8A8, abliterated](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-heretic-MTP) | 29.1 GiB | won't fit | 286,720 ³ | won't fit | not tested ³ |
| [W4A16 + int8 head](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-int8head-MTP) | 16.6 GiB | **55,000** | **393,216** | broken ² | broken ² |
| [W4A16 + int8 head, abliterated](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-int8head-MTP) | 16.6 GiB | **55,000** | **393,216** | broken ² | broken ² |
| **W4A16 plain, abliterated (this one)** | 18.1 GiB | 20,000 | **393,216** | wouldn't start ¹ | 262,144 |

¹ Won't start on a single card in SGLang, and the reason is simple once you get the right error out of it:
SGLang counts the MTP draft weights separately, and 18.1 GiB of weights plus the drafter occupy **99.92%** of a
24 GB card — it reports `minimum viable = 0.9992`, i.e. there is no legal `--mem-fraction-static` that leaves
room for a KV cache. Lower fractions fail earlier, in the Gated-DeltaNet state cache (which is carved out of
the static pool, so less pool means *less* room: `-2.29 GB` at 0.88 against `-0.67 GB` at 0.95), and
`--max-mamba-cache-size 8 --mamba-ssm-dtype bfloat16` only moves the failure to the KV step. It is a capacity
wall, not a tuning problem. Two cards is fine — see the column to the right. The int8-head builds are 1.5 GiB
smaller and would have room, but can't run on SGLang for an unrelated reason (²).

² Not a memory limit. A checkpoint that mixes bit-widths (4-bit body, 8-bit `lm_head`/MTP) doesn't resolve
its `lm_head` scheme in SGLang's compressed-tensors path, so the head loads uninitialized and the model
emits garbage. Details on the int8-head cards.

Two things worth pulling out. On two cards nothing here is memory-bound — 393,216 is the YaRN ceiling I asked
for, not the card's limit. On *one* card the int8 head is worth 2.75x the context (55,000 against 20,000),
which is the real argument for it alongside the ~35% decode gain; both builds do run MTP on a single card.

### Speed

| release | code tok/s | think 6k tok/s | benched on |
|---|---:|---:|---|
| INT8 W8A8 | 96.2 | 59.3 | 2 GPU |
| [INT8 W8A8, abliterated](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-heretic-MTP) | 95.0 | 59.2 | 2 GPU |
| W4A16 + int8 head | 89.9 | 56.2 | 1 GPU |
| W4A16 + int8 head, abliterated | 88.7 | 56.6 | 1 GPU |
| W4A16 plain, abliterated | 66.6 | 43.0 | 1 GPU |

W8A8 can only run across two cards, so its row isn't matched hardware — read it as what that build needs, not
as 8-bit beating 4-bit at equal cost.

### Quality

| release | correct (of 160) | MTP accept (code / think) | vision |
|---|---:|---:|---:|
| INT8 W8A8 | 132 | 92.2% / 40.2% | 2/2 |
| [INT8 W8A8, abliterated](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-heretic-MTP) | 130 | 91.0% / 40.8% | 2/2 |
| W4A16 + int8 head | 131 | 89.1% / 39.6% | 2/2 |
| W4A16 + int8 head, abliterated | 134 | 88.5% / 40.7% | 2/2 |
| W4A16 plain, abliterated | 135 | 88.8% / 41.8% | 2/2 |
| W4A16 plain | 131 | 84.5% / — | 2/2 |

All builds above are now ground-truth scored. The abliterated W4A16 pair actually edges the
un-abliterated ones (134/135 against 131), which is noise rather than an improvement, but it does rule out
abliteration costing accuracy on these tasks.

The whole column spans 130-135 out of 160, which is a tie. For scale: re-running one unchanged build twice
moved its score by two items (133 and 135 on separate runs of the same weights) and decode by ~1 tok/s. Nothing
in that table is a real difference — 4-bit against 8-bit included.

## The recipe

```
format      compressed-tensors, pack-quantized
weights     4-bit int · symmetric · group-128 · observer: imatrix-mse
activations none (weight-only)
targets     Linear
ignore      all model.visual.* blocks · linear_attn in_proj_a / in_proj_b / norm
            · lm_head · embed_tokens · re:.*mtp.*
```

Same calibration corpus as the INT8 release: 512 sequences of roughly 2,000 tokens, half containing tool calls,
built from public open-source Python. `imatrix-mse` needs calibration data and `pipeline="sequential"`; without
both, llmcompressor silently falls back to a data-free uniform-MSE build.

## Limitations

* Abliterated models are model-wide edits, not domain-selective — this reduces refusals broadly, it does not
  selectively unlock only benign-but-edgy topics while leaving genuinely dangerous ones refused.
* Refusal numbers above are a keyword check on one prompt set at one point in time, not an LLM-judged or
  human-reviewed safety evaluation. Don't read a low refusal rate as a quality guarantee on any specific request.
* Measured on one machine, one operator, no standard benchmark suite (MMLU, GSM8K) — same caveat as my other
  releases: treat the numbers above as a shape, not a spec.
* This is a third-party modification, not something Qwen produced, reviewed, or endorses.

## License

Same license as the base model, [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache 2.0. The base
repo has the authoritative terms. Abliteration performed with [Heretic](https://github.com/p-e-w/heretic); credit
to its author for the tool. All credit for the model itself goes to the Qwen team — this is a quantization and a
weight edit of their work, nothing more.
