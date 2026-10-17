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
  - sglang
  - vllm
  - speculative-decoding
  - mtp
  - multimodal
---

# Qwen3.8-27B INT8 W8A8 (imatrix-mse), with MTP and vision intact

INT8 W8A8 quant of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
made with [llmcompressor](https://github.com/vllm-project/llm-compressor) in
compressed-tensors format.

I built this for a **dual RTX 3090 rig** (2x24 GB, tensor parallel across both
cards, no NVLink), which is what shaped most of the decisions here. It should
suit anyone running two 24 GB Ampere cards, and it'll work on a single 48 GB card
as well. Two things got kept that quant pipelines often drop along the way:

* the MTP module (15 `mtp.*` tensors), so speculative decoding runs
* the vision tower (333 `model.visual.*` tensors), so it stays multimodal

29.1 GiB, 1,599 tensors (400 INT8, 1,199 BF16). It fits across two 3090s with
enough headroom left for a fairly deep context.

I serve it with **SGLang**, with MTP as NEXTN, vision and `qwen3_coder` tool
calls all working. It runs under vLLM too, and there's a short section on that
below.

## Why INT8

FP8 needs compute capability 8.9 or higher, so on Ampere cards like the 3090,
A6000 and A100 it isn't available and INT8 W8A8 is the practical choice. Worth
checking what you're on:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

If you're on Ada or newer, an FP8 build will probably serve you better than this
one.

## What's actually quantized

I read this off the tensor dtypes rather than trusting the config:

| module group | dtypes | size |
|---|---|---|
| MLP | 192 INT8 + 192 BF16 | 15.94 GiB |
| linear_attn (48 layers) | 432 BF16 + 144 INT8 | 5.21 GiB |
| norms / misc | 130 BF16 | 2.37 GiB |
| `lm_head` | BF16 | 2.37 GiB |
| full_attn (16 layers) | 96 BF16 + 64 INT8 | 1.56 GiB |
| vision tower | 333 BF16 | 0.86 GiB |
| MTP | 15 BF16 | 0.79 GiB |

The MLP and full-attention projections carry most of the quantization. The vision
tower stays BF16 throughout, and the Gated DeltaNet `linear_attn` layers are only
partly quantized, since `in_proj_a`, `in_proj_b` and `norm` are excluded. That's
where most of that 5.21 GiB sits. `lm_head` and MTP are excluded too.

It's a fairly conservative recipe. You could get this model smaller, though in my
experience you tend to give up speculative decoding or vision along the way, and
for a two card setup the tradeoff didn't seem worth it.

## Read this first: the per-sequence state

One thing dominates memory planning on this model, and it isn't the KV cache.

48 of the 64 layers are Gated DeltaNet linear attention. Each *concurrent
sequence* needs its own recurrent state, and unlike a KV cache that state is a
fixed cost that doesn't shrink for short prompts:

```
48 value heads x 128 key dim x 128 value dim x 4 bytes (fp32)  =  3 MiB per layer
3 MiB x 48 linear-attention layers                             =  144 MiB per sequence
at TP=2                                                        =  ~72 MiB per GPU per sequence
```

Both engines default to a concurrency well above what a single-user setup needs,
and on 2x24 GB that reservation is the difference between starting and not:

* **SGLang** auto-sizes this from `mamba_full_memory_ratio` (default 0.9), which
  handed roughly half the cache arena to DeltaNet state on my box. Setting
  `--max-mamba-cache-size 8` freed most of it.
* **vLLM** `--max-num-seqs` defaults to 128, which is several GB per card gone
  before a single KV token is allocated, on top of a ~15 GiB weight share. Set it
  to `1` or `2`.

If you hit an OOM, or get a much smaller context than you expected, look here
first — before `--gpu-memory-utilization`.

## Running it in SGLang

```bash
python -m sglang.launch_server \
  --model-path RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP \
  --tp 2 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --context-length 245760 \
  --mem-fraction-static 0.95 \
  --max-running-requests 2 \
  --max-mamba-cache-size 8 \
  --mamba-ssm-dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-max-bs 2 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"preserve_thinking": true, "reasoning_effort": "low"}'
```

Out of the box I was capped at a much smaller context than the card advertises.
Four flags got me to 245,760, tested one at a time:

* `--max-mamba-cache-size 8` with `--kv-cache-dtype fp8_e4m3` — by far the
  biggest win, and it's the per-sequence state described above. Speculative
  decoding consumes roughly 4 state slots per request, so 8 is the floor. Fewer
  slots also shrink the mamba radix prefix cache, which is what gives you warm
  starts on multi-turn sessions, so don't go below it.
* `--mem-fraction-static 0.95` — I stress-tested vision at this setting (1024px
  image at deep context) and it held, with well under a gigabyte free. Don't go
  higher.
* `--mamba-ssm-dtype bfloat16` — worth a useful chunk of context. Needle recall
  stayed perfect at 105k, 215k and 240k tokens, and generation stayed coherent.
  Those are light probes rather than a parity proof, so if you see long-session
  weirdness, revert this one first.

Two things that didn't help: `--speculative-num-draft-tokens 3` gained nothing,
and YaRN is moot here, since the pool sits below the model's native 262,144
window. The cap is memory-bound, not rope-bound.

The first request after boot takes the better part of a minute for CUDA graph
capture, and the first request at any new depth also runs slow. Discard it when
benchmarking.

## It also runs in vLLM

I've moved to SGLang and don't run vLLM day to day any more, so treat this as
last known-good rather than current — it was last measured on vLLM 0.22.0. What
it buys you is context, 311,296 against SGLang's 245,760, and better prefill at
short depth. SGLang was faster on decode at every depth I compared, and a quality
A/B on identical weights came out a dead heat.

```bash
vllm serve RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP \
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

Three things trip people up:

1. **`--max-num-seqs 1`.** See the per-sequence state section above. This is the
   flag that decides whether the server starts at all on 2x24 GB.
2. **The YaRN override is not optional** past 262,144. Without it vLLM refuses to
   start, because `--max-model-len` exceeds `max_position_embeddings`. Keep
   `mrope_section`, `partial_rotary_factor` and `rope_theta` in there or the model
   won't load. Extending this way also seemed to help deep-context behaviour
   rather than hurt it: at 1.5x, position 250k sits where roughly 167k would land
   natively, comfortably inside the trained range.
3. **`--disable-custom-all-reduce` on PCIe.** vLLM's custom all-reduce kernels
   assume NVLink.

That config reported around 345,620 tokens of KV cache on 2x24 GB, and I set
`--max-model-len` to 311,296 to leave room, since MTP spends cache on the drafter
too. Read the figure off your own startup output rather than trusting mine, and
set `--max-model-len` just under it. `--max-num-batched-tokens 4096` beat 8192 and
16384 for me, both of which cost KV cache without buying prefill back — vLLM's own
startup warning telling you to raise it is wrong on this hardware, at least when
serving one request at a time.

## Vision

The vision tower has 2,304 position embeddings and each merged token covers
1,024 px, which works out to roughly 2.36 MP per image. Above that — anything
much over 1536x1536 — you get:

```
Mismatch in `image` token count between text and `input_ids`
```

On vLLM, `--mm-processor-kwargs '{"max_pixels":2000000,"min_pixels":65536}'` caps
it, and setting it also gives you back a chunk of KV cache, since vLLM otherwise
reserves room for a worst-case image.

## reasoning_effort

The chat template defaults to `xhigh`, the maximum. I default to `low` instead and
raise it per request when a task calls for it. On a self-grading build task —
the model writes a unittest suite and the harness runs it — `low` came out close
to `xhigh` on tests passed while taking a fraction of the time and a small
fraction of the reasoning tokens.

Two things worth knowing:

* `medium` is not a middle setting. The template injects no instruction for it,
  whereas `low` explicitly says "move directly to the conclusion". So `low` gets a
  clear directive and follows it, while `medium` drifts, and it scored worst of
  the three for me.
* `xhigh` can talk itself out of the right answer. On that build task it rejected
  the stdlib `zoneinfo` module every time and hand-rolled a `datetime.tzinfo`
  subclass that then crashed; `low` and `medium` imported `zoneinfo` every time.
  The `xhigh` instruction says to "consider plausible alternatives", which on a
  problem the stdlib already solves is a liability. I also tried it as an agentic
  default and reverted the same day — the think-to-output ratio made sessions
  unworkable.

Either way, give it plenty of `max_tokens`. This model thinks at length, and if it
runs out mid-thought you get reasoning back with empty content.

## preserve_thinking, and prefix caching

The chat template hides historical reasoning *before* the last real user query and
keeps it after. That boundary moves every time you send a message, so assistant
turns that rendered with their reasoning last turn render without it this turn.
Prefix caching needs append-only prompts, so this invalidates the cache from the
first rewritten block onward.

It only bites during tool-call loops, where several assistant messages sit between
two user messages. Plain turn-taking never moves the boundary, which is why a
chat-shaped test shows nothing and a tool-shaped one reproduces it.

`preserve_thinking: true` keeps reasoning always, which makes prompts append-only
and measurably improved cache hit rate, re-prefill and wall-clock time on a
multi-turn tool-calling workload. It costs context, since retained reasoning is
carried forward, so if you hit the ceiling on long sessions, drop this first.

The repo ships Qwen's official 3.8 `chat_template.jinja`, so you shouldn't need to
pass a template at all. Don't substitute the custom Qwen3.6 one if you have it
lying around.

## Sampling

```
temperature 0.6 · top_k 20 · top_p 0.86 · repetition_penalty 1.07
presence_penalty 0.0 · frequency_penalty 0.0
```

I arrived at these by scoring 8-gram repetition at depth rather than by feel.
Worth flagging that Qwen's documented anti-repetition advice
(`presence_penalty 1.5`) was the worst setting I tested — it repeated more, and it
suppressed natural stopping so replies ran to the token cap.

Avoid temperature 0. Qwen notes that greedy decoding degrades thinking mode, and
that matched what I saw: reasoning that loops with no output at the end.

## Speculative decoding

3 draft tokens is my default on both engines, and MTP is a large decode win across
the whole usable range. Acceptance did not decay with depth in my testing.

Acceptance is **task**-dependent, not context-dependent. Generating code drafts
well. A needle test measured near zero, because no drafter can predict an
arbitrary random string. A low acceptance reading means the workload is
unpredictable, not that MTP has broken.

Qwen3.8 ships one trained MTP layer, so there's no adding capacity through
configuration. Acceptance also seems bounded by the architecture: those 48 Gated
DeltaNet layers carry a compressed recurrent state, so a rejected draft can't be
rolled back the way you'd truncate a KV cache suffix. Qwen3.6-27B reached
noticeably higher acceptance on the same harness, and I couldn't close the gap —
I tried quantization strategy, draft sampling, N of 2/3/5, mamba state dtypes, KV
dtypes, and a nightly meant to address this directly.

### Counting tokens, not chunks

Before you benchmark any of this: accepted draft tokens arrive **batched into a
single SSE chunk**. A harness that does `ntok += 1` per chunk undercounts
speculative decoding by roughly the acceptance factor, and will tell you MTP made
things slower when it didn't. Every "MTP is slower on my machine" result I had
turned out to be this. Count tokens server-side.

## The recipe

```
format      compressed-tensors, W8A8
weights     8-bit int · symmetric · per-channel · static · observer: imatrix-mse
activations 8-bit int · symmetric · per-token · dynamic
targets     Linear
ignore      all model.visual.* blocks · linear_attn in_proj_a / in_proj_b / norm
            · lm_head · re:.*mtp.*
```

Calibration was 512 sequences of roughly 2,000 tokens each, half of them
containing tool calls, built from public open source Python (the vLLM and ComfyUI
trees). I'm not shipping the corpus itself, but the description above should be
enough to rebuild something equivalent.

`imatrix-mse` weights the quantization error by how much each activation actually
matters, rather than minimizing average weight error uniformly. It needs
calibration data and `pipeline="sequential"`. If either is missing, llmcompressor
falls back to a data-free pipeline, logs `imatrix_mse: no importance data
available. Falling back to uniform MSE`, and finishes normally while producing
something different from what you asked for. Worth checking the log for that line.

I built the same base model five ways on one machine and harness and compared MTP
acceptance. `imatrix-mse` and `minmax` came out about even at the top, W8A16 and
SmoothQuant behind them, and plain `mse` last — it clips the activation outliers
the drafter depends on. So: don't "fix" this build to plain `mse`.

### Three things to watch for if you rebuild this

1. Load with `AutoModelForImageTextToText`. With `AutoModelForCausalLM` you get
   the language model on its own and no vision tower, and there's no error to
   tell you.
2. llmcompressor drops the MTP module, so graft the 15 `mtp.*` tensors back from
   the base checkpoint afterwards. They stay BF16 with no scales, which is
   expected.
3. Add `re:.*mtp.*` to `quantization_config.ignore`. Without it vLLM reads those
   BF16 tensors as INT8 and speculative decoding sits at 0% acceptance without
   complaining, or throws `KeyError: 'weight_scale'`. After fixing it, clear
   `~/.cache/vllm/torch_compile_cache`, otherwise vLLM reuses the graph it
   compiled under the broken config and the fix looks like it did nothing. That
   one cost me an hour.

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
| **INT8 W8A8 (this one)** | 29.1 GiB | won't fit | 311,296 | won't fit | 245,760 |
| [INT8 W8A8, abliterated](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-heretic-MTP) | 29.1 GiB | won't fit | 286,720 ³ | won't fit | not tested ³ |
| [W4A16 + int8 head](https://huggingface.co/RukaRat/Qwen3.8-27B-W4A16-imatrix-int8head-MTP) | 16.6 GiB | **55,000** | **393,216** | broken ² | broken ² |
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

## Versions

Version drift is real in this area and flag names keep moving between releases.
Everything above was on Ubuntu 22.04, driver 580 (CUDA 13), 2x RTX 3090 at TP=2
with no NVLink, one card on an x4 chipset slot:

| | SGLang | vLLM |
|---|---|---|
| engine | 0.5.18, sglang-kernel 0.4.6.post1 | 0.22.0 |
| torch | 2.13.0+cu130 | 2.11.0+cu130 |
| flashinfer | 0.6.17 | 0.6.11.post2 |
| transformers | 5.12.1 | 5.9.0 |

Both are hand-built venvs rather than container images. Check flag names against
your own version rather than pasting mine.

One host-side note if you're near the RAM line: building the MTP drafter re-reads
the whole 29 GiB checkpoint, so a box with ~31 GB of RAM will spike hard enough
for `systemd-oomd` to kill the load. Running the server in its own scope with
`ManagedOOMPreference=omit` fixed that for me. The kernel OOM killer still
applies, which is the point.

## Limitations

* Built with Ampere in mind, and a dual 3090 setup in particular. On Ada, Hopper
  or Blackwell you'll likely do better with FP8 or NVFP4.
* Not compressed as far as it could be. Vision, `lm_head`, MTP and a good chunk of
  `linear_attn` are still BF16.
* I measured throughput, speculative acceptance, repetition at depth and a small
  code suite rather than standard benchmarks, so there's no MMLU or GSM8K
  comparison against the base model here.
* Everything here comes from one machine, with one card on an x4 chipset slot and
  no NVLink, and the run-to-run noise on it is high enough that I'd treat any of
  it as a shape rather than a spec.

## License

Same license as the base model,
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), which is Apache 2.0.
The base repo has the authoritative terms. All the credit goes to the Qwen team,
this is just a quant of their work.
