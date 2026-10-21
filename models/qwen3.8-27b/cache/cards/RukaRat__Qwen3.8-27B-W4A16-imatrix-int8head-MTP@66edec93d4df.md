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
  - vllm
  - speculative-decoding
  - mtp
  - multimodal
---

# Qwen3.8-27B W4A16 + int8 lm_head/MTP (imatrix-mse), for one 24 GB card — **vLLM only**

**Read this first: this checkpoint does not work on SGLang.** I tested it directly and it loads without error but
produces garbage output — repeated punctuation, not a subtle quality loss. SGLang's compressed-tensors integration
doesn't resolve which scheme applies to `lm_head` when a checkpoint mixes two bit-widths (4-bit main weights,
8-bit `lm_head`/MTP); it silently treats `lm_head` as unquantized, the actual weight tensors on disk never map
onto it, and the result is an uninitialized output head. This is not a version gate — I confirmed it on SGLang
0.5.18 despite that version already wiring `quant_config` into the relevant constructor; the bug is downstream of
that, in scheme resolution. If you need SGLang, the [INT8 W8A8 release](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP)
of this model has no mixed bit-width and works on both engines.

W4A16 quant of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), same recipe and calibration corpus
as my [INT8 W8A8 release](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP), at 4-bit instead of
8-bit, plus one extra step: the `lm_head` and the MTP draft module are also requantized to int8, on top of the
4-bit main weights. Everything else — vision tower, embeddings, the `linear_attn` `in_proj_a`/`in_proj_b`/`norm`
excluded projections — stays exactly as the base W4A16 build leaves them.

16.6 GiB, built for a single 24 GB Ampere card with room to spare for context and MTP.

## Why bother with `lm_head` and MTP specifically

Every published W4A16 quant of this model I could find — including the plain version of this one, without the
extra step — leaves two more tensors in bf16: `lm_head` (2.4 GiB, a 248k-row output head) and the MTP draft module
(0.8 GiB). At single-request decode, `lm_head` gets read once for the real forward pass and once per accepted
speculative draft, so on an MTP-heavy model like this one it's read up to four times a step. Halving both brought
decode from 66.5 to 89.9 tok/s on one 3090, and freed enough VRAM to roughly double usable context at matched
settings (34,560 vs 16,000 tokens measured at an otherwise-identical config). This mirrors the approach
[syv-ai's rtx3090 build](https://github.com/syv-ai/qwen38-27b-rtx3090) takes for the same reason, on the same
model family.

## Quality: identical, not just close

Same 200-item, five-task test battery, temperature 0, scored against ground truth and against the plain W4A16
build's own output:

| | this model | plain W4A16 |
|---|---:|---:|
| correct, ground truth | *not scored* | 131 / 160 |
| identical output to the plain build | **200/200** | — |
| code decode | **89.9 tok/s** | 66.5 tok/s |
| thinking (6k) decode | **56.2 tok/s** | 42.6 tok/s |
| MTP acceptance | 57.6% | 61.0% |
| vision (2 probes) | 2/2 | 2/2 |

Every one of the 200 replayed answers matched the plain build's word for word. `lm_head` only decides which token
gets sampled from logits the rest of the network already computed correctly, and MTP drafts are verified exactly
against the real head regardless of the drafter's own precision — quantizing either changes speed, never the
sampled output.

If you want this same speed gain on an abliterated (uncensored) build instead, the numbers carry over unchanged:
[the Heretic-abliterated int8-head variant](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-int8head-MTP)
measured 89 tok/s code decode and 58.5% MTP acceptance against its own plain-abliterated baseline —
within noise of the numbers here. Requantizing `lm_head`/MTP doesn't interact with abliteration; either can be
applied independently and the other's numbers hold.

## Serving (vLLM only — see above)

**Two 24 GB cards — full 393,216 context, MTP on:**

```bash
vllm serve RukaRat/Qwen3.8-27B-W4A16-imatrix-int8head-MTP \
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
vllm serve RukaRat/Qwen3.8-27B-W4A16-imatrix-int8head-MTP \
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

Needs **vLLM ≥0.23** to load correctly: Qwen3.5-family (`qwen3_5`) model code didn't pass `quant_config` through to
`ParallelLMHead` before that ([vLLM PR #42124](https://github.com/vllm-project/vllm/pull/42124)). On vLLM 0.22 and
earlier, the quantized `lm_head` tensors will be read but never actually dequantized/used correctly — check your
vLLM version before assuming this loads.

Everything else about serving this — sampling values, `reasoning_effort`, `preserve_thinking`, MTP flags — is
unchanged from the [INT8 W8A8 card](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP); nothing
about requantizing `lm_head`/MTP touches any of that.

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
| **W4A16 + int8 head (this one)** | 16.6 GiB | **55,000** | **393,216** | broken ² | broken ² |
| [W4A16 + int8 head, abliterated](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-heretic-int8head-MTP) | 16.6 GiB | **55,000** | **393,216** | broken ² | broken ² |
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
format      compressed-tensors, pack-quantized
main weights   4-bit int · symmetric · group-128 · observer: imatrix-mse
lm_head, MTP   8-bit int · symmetric · group-128 · observer: minmax (round-to-nearest)
activations    none (weight-only)
ignore         all model.visual.* blocks · linear_attn in_proj_a / in_proj_b / norm · embed_tokens
```

Same 512-sample calibration corpus as the INT8 release (code + tool-call mix). The `lm_head`/MTP requantization
is round-to-nearest, not imatrix-calibrated — measured round-trip error 0.64% (`lm_head`) and 0.66–0.77% (MTP
linears), well inside the identical-output result above.

## Limitations

* **Broken on SGLang** — stated at the top, repeating it here because it's the thing most likely to bite someone
  who skips to the bottom.
* Measured on one machine, one operator, no standard benchmark suite — treat the numbers above as a shape, not a
  spec, same as my other releases.
* Embeddings are not quantized here; a `qwen3_5`-aware embedding path exists upstream in newer vLLM but the model
  code doesn't wire it up yet, so there's nothing to gain there without a patch.

## License

Same license as the base model, [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache 2.0. All
credit for the model goes to the Qwen team — this is a quantization of their work, nothing more.
