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
  - speculative-decoding
  - mtp
  - multimodal
---

# Qwen3.8-27B W4A16 + int8 lm_head/MTP, Heretic-abliterated — **vLLM only**

Combines two things I've released separately: the abliteration from
[Qwen3.8-27B-W4A16-imatrix-heretic-MTP](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-MTP) and
the int8 `lm_head`/MTP requantization from
[Qwen3.8-27B-W4A16-imatrix-int8head-MTP](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-int8head-MTP).
16.6 GiB, same base recipe as both.

## Read this before downloading: it does not work on SGLang

I tested it there and it doesn't load correctly. SGLang's compressed-tensors integration doesn't resolve which
quantization scheme applies to `lm_head` when a checkpoint mixes two different bit-widths — the main weights at
4-bit, `lm_head` at 8-bit. It falls back to treating `lm_head` as unquantized, the actual weight tensors on disk
never get mapped onto it, and the result is a model with an uninitialized output head. I confirmed this with a
live request: garbage output (repeated punctuation), not a subtle quality loss. **vLLM has no such issue** — this
loads and runs correctly there, which is the whole reason this exists as a separate release rather than a variant
of the plain abliterated build.

If you need SGLang, use [the plain abliterated build](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-MTP)
instead — no mixed bit-width there, confirmed working on both engines.

**This model has had its safety training removed.** Same abliteration as the plain build above — see that card for
the method and the honest discussion of what the refusal numbers below do and don't mean. Use accordingly.

## Why this exists

Requantizing `lm_head` and the MTP draft module to int8 on top of the 4-bit main weights brought single-request
decode from ~67 to ~89 tok/s on one 3090 in my testing, with no measurable cost — see the numbers below. That's a
meaningful difference for interactive use on one card, if you're staying on vLLM anyway.

## Quality and refusal rate: identical to the plain build

Compared directly against the plain abliterated build (same weights, same abliteration, no `lm_head`/MTP
requantization), same 200-item five-task battery and the same refusal eval, all measured on the actual served
checkpoint on vLLM:

| | this model | plain abliterated build |
|---|---:|---:|
| correct, ground truth | *not scored* | 131 / 160 |
| identical output to the plain build | 183/200 | — |
| code decode | **89 tok/s** | 67 tok/s |
| thinking decode | **56 tok/s** | 43 tok/s |
| MTP acceptance | 58.5% | 58.8% |
| vision (2 probes) | 2/2 | 2/2 |
| harmful prompts, thinking off | 0/60 | 0/60 |
| harmful prompts, thinking on | 1/30 (3.3%) | 1/30 (3.3%) |
| benign-but-refused prompts¹, thinking off | 0/60 | 0/60 |
| benign-but-refused, thinking on | 0/30 | 0/30 |

¹ [OR-Bench-Hard-1K](https://huggingface.co/datasets/bench-llm/or-bench). Refusal counted by keyword marker, not a
judge — see the plain build's card for the full caveat on what this measures and doesn't.

Every refusal and quality number matches the plain build within noise. Quantizing `lm_head`/MTP to int8 doesn't
touch the sampled output in a way that shows up on this battery — `lm_head` only decides which token the already-
correct logits pick, and MTP drafts are verified exactly against the real head regardless of the drafter's own
precision.

## Serving (vLLM only — see above for why)

**Two 24 GB cards — full 393,216 context, MTP on:**

```bash
vllm serve RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-int8head-MTP \
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

**One 24 GB card — 55,000 context, MTP on:**

```bash
vllm serve RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-int8head-MTP \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --enforce-eager \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.92 \
  --hf-overrides '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":1.5,"original_max_position_embeddings":262144,"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000}}}' \
  --max-model-len 55000 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":4,"video":0}'
```

Measured on 2x RTX 3090, vLLM 0.22, `--enforce-eager`, `--gpu-memory-utilization 0.92`,
`--kv-cache-dtype fp8_e4m3`, `--max-num-seqs 1`, MTP on, YaRN 1.5x:

| | GPU KV cache | max context | concurrency |
|---|---:|---:|---:|
| 2x 3090 (TP=2) | 665,906 tokens | **393,216** | 1.69x |
| 1x 3090 | 55,000 tokens | **55,000** | 1.00x |

On two cards this is not memory-bound — 665,906 tokens of KV against a 393,216 ceiling set by YaRN, not by VRAM.

On one card the int8 head buys 2.75x the context: 55,000 tokens against 20,000 for the plain
(BF16 `lm_head`) build, both with MTP enabled. On two cards neither is memory-bound, so the difference
disappears — pick on decode speed there, not capacity.

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

Needs **vLLM ≥0.23** to load the quantized `lm_head` for this architecture
([PR #42124](https://github.com/vllm-project/vllm/pull/42124)) — on 0.22 and earlier it will load without error but
the quantized `lm_head` tensors won't actually be used correctly, and you'll get degraded output with no warning.
Check your version before assuming this works.

Everything else — sampling values, `reasoning_effort`, `preserve_thinking` — is unchanged from the
[INT8 W8A8 card](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP).

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
| **W4A16 + int8 head, abliterated (this one)** | 16.6 GiB | **55,000** | **393,216** | broken ² | broken ² |
| [W4A16 plain, abliterated](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-MTP) | 18.1 GiB | 20,000 | **393,216** | wouldn't start ¹ | 262,144 |

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
format         compressed-tensors, pack-quantized
main weights   4-bit int · symmetric · group-128 · observer: imatrix-mse
lm_head, MTP   8-bit int · symmetric · group-128 · observer: minmax (round-to-nearest)
activations    none (weight-only)
ignore         all model.visual.* blocks · linear_attn in_proj_a / in_proj_b / norm · embed_tokens · mtp.fc
```

Abliteration applied to the BF16 base before quantizing — see the plain build's card for the full method (Heretic
1.4.0, MPOA, rank-3 LoRA on `down_proj`/`o_proj`/`out_proj`) and search parameters. `lm_head`/MTP requantization is
round-to-nearest, not imatrix-calibrated — measured round-trip error 0.64% (`lm_head`), 0.66–0.77% (MTP linears).

## Limitations

* **Broken on SGLang** — stated above, repeating it here because it's the thing most likely to bite someone who
  skips to the bottom. Use the plain build for SGLang.
* Same limitations as the plain abliterated build: model-wide edit not domain-selective, refusal numbers are a
  keyword check on one prompt set not a safety audit, measured on one machine with no standard benchmark suite,
  third-party modification not endorsed by Qwen.

## License

Same license as the base model, [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache 2.0.
Abliteration performed with [Heretic](https://github.com/p-e-w/heretic). All credit for the model itself goes to
the Qwen team.
