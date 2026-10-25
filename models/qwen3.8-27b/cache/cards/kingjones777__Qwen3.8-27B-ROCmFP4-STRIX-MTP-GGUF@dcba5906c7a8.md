---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - rocm
  - amd
  - rocmfp4
  - rocmfpx
  - strix-halo
  - amd-strix-halo
  - gfx1151
  - ryzen-ai-max
  - ryzen-ai-max-395
  - radeon-8060s
  - mtp
  - speculative-decoding
  - reasoning
  - qwen3.5
  - quantized
---

# Qwen3.8-27B — ROCmFP4 for AMD Strix Halo (gfx1151)

## New file (2026-10-25): `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX-imatrix-headQ6K.gguf`

The ROCmFP4 files already in this repository (STRIX, STRIX_LEAN and FAST) are built without importance weighting. This file keeps the STRIX recipe but adds importance-matrix weighting from an agentic calibration imatrix, and raises the output head and the token embedding to Q6_K. The numbers below pair it with the DFlash2 draft head rather than the in-repo MTP head.

**How it was built:**
- Tool: `gufo-quant`, recipe STRIX, weighted with the agentic imatrix.
- Tensor overrides: `output.weight` and `token_embd.weight` at Q6_K; census 866 tensors (82 at type 100, 421 at type 101).
- The exact command line was not recorded.

**Measured on Super ROC** (AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4; one request at a time, temperature 0, cold prompt cache):

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 513.6 tok/s |
| Generation after that prompt (up to 256 tokens) | 27.6 tok/s |
| Prompt processing, 13384-token prompt | 556.6 tok/s |
| Generation after that prompt | 21.3 tok/s |
| Prompt processing, 26821-token prompt | 534.6 tok/s |
| Generation after that prompt | 21.2 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 50.6 tok/s (45.4-54.9) |
| Drafted tokens accepted on those prompts | 1106/1578 |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

**Quality** (measured with Super ROC's perplexity tool on a 6K-token long-text corpus, context 4096, and against BF16 logits on wikitext-2, 16 x 512 tokens): perplexity 5.2830, KL divergence 0.033983, top-1 agreement 91.078 %. The stock Unsloth UD-Q4_K_XL file measures 5.1989, 0.006348 and 95.858 % on the same tests.

**Run it with Super ROC** (https://github.com/kingjones30/superroc):
```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX-imatrix-headQ6K.gguf --served-model-name qwen3.8-27b --context 32768 --think off --speculative dflash2 --dflash-model /path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```


<!-- LEAD-CLAIM:START -->

> ✅ **the only public ROCmFP4 build of this model published with measured perplexity and an MTP
> tuning curve**
>
> *Checked 2026-08-14 against all four other public ROCmFP4/ROCmFPX builds of Qwen3.8-27B. None
> publishes perplexity (one lists it as "TBD") and none publishes a `--spec-draft-n-max` sweep.
> Card-content comparison only — no third-party build was run or benchmarked here.*

<!-- LEAD-CLAIM:END -->

Three ROCmFP4 quantisations of **Qwen3.8-27B** for AMD Ryzen AI Max+ 395 / Radeon 8060S / gfx1151,
bundled with the multi-token-prediction draft head. ROCmFP4 is a runtime tensor format that exists
only in the [ROCmFPX](https://github.com/charlie12345/ROCmFPX) fork of llama.cpp.

The headline here is not the quantisation on its own — it is **what the MTP head is worth once it is
tuned**, which turns out to be more than the quantisation itself.

| Metric | Result |
| --- | ---: |
| Quantization | ROCmFP4 (ggml types 103 / 105 / 106), 3 variants |
| Model size | 13.33 – 13.75 GiB |
| Effective BPW | 4.25 – 4.39 (measured, not advertised) |
| Tested hardware | AMD Ryzen AI Max+ 395 (Strix Halo), 128 GB unified |
| GPU | Radeon 8060S, gfx1151 |
| ROCm version | 7.2.4 |
| **8K decode, MTP tuned** | **30.30 tok/s** (vs 10.70 for Q4_K_M — **2.83×**) |
| 8K decode, no speculation | 13.46 tok/s (vs 10.70 — 1.26×) |
| 32K decode, no speculation | 12.58 tok/s (vs 10.12 — 1.24×) |
| Prompt processing, 8K | 317.6 tok/s (vs 306.0 for Q4_K_M) |
| Perplexity (wikitext-2, held out) | **5.8877 ± 0.068** vs **5.8926 ± 0.069** for Q4_K_M — parity |
| Cold load, 13.7 GiB with `-dio` | 6.3 s (vs 9.4 s for Q4_K_M) |
| MTP acceptance | 0.926 at `n-max 4` |
| Context validated | 65536 — see *Not yet measured* |
| Vision | ✅ **4/4 on spatial ground truth, and works with `-fa on`** |
| Tool calling | ✅ **7/7 — and 7/7 again with thinking enabled** |

## Why this build?

- **2.83× the decode of a standard Q4_K_M** on the same box, same binary, same flags — 30.30 vs
  10.70 tok/s, once the MTP head is tuned
- **Perplexity parity at 22% smaller**: 5.8877 vs 5.8926, a 0.08% difference against ±1.2% error
  bars. No measurable quality cost for the missing bits.
- **The MTP depth curve is published.** llama.cpp's default `--spec-draft-n-max` is **16**, which on
  this model gives roughly *half* the achievable throughput. The optimum is **3–4**.
- **Prefill improves rather than regressing** (+3.8%), which is unusual for a ROCmFP4 build
- Cold load in **6.3 s**
- Every number below came off a quiesced machine with a **0.08% measured noise floor**

## Which file should I use?

**Start with `STRIX`.** All three variants measured *identical* in speed, so there is no speed reason
to take fewer bits.

Ryzen AI Max+ 395, ROCm 7.2.4, ctx 65536, batch 1, greedy (`temp 0`, `top_k 1`), `ignore_eos` so every
arm generates exactly 256 tokens, unique nonce per request and `cache_prompt: false` so the prefix
cache never serves a hit (`cached_tokens = 0` verified on every run). Median of 3–5 reps.

| Build | ftype | Size | BPW | TG 8K (no spec) | TG 8K (MTP) | PPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **`ROCmFP4-STRIX`** | 105 | 13.75 GiB | 4.39 | 13.46 | **30.30** | 5.8877 |
| `ROCmFP4-STRIX_LEAN` | 106 | 13.59 GiB | 4.34 | 13.46 | — | 5.8871 |
| `ROCmFP4-FAST` | 103 | 13.33 GiB | 4.25 | 13.44 | 30.13 | 5.9233 |
| *`Q4_K_M`* (reference, ggml-org) | — | 17.67 GiB | — | 10.70 | — | 5.8926 |

> **The variants are speed-equivalent, and that is the finding.** 13.44 / 13.46 / 13.46 tok/s is a
> **0.15% spread** — smaller than most measurement noise, and this instrument's noise floor is 0.08%.
> The ftype names imply a size/speed trade that does not materialise on this architecture: decode
> here is not purely weight-bandwidth-bound, so shaving 3% off the file buys nothing. Take the bits.

**`FAST` (103) is therefore dominated on this model** — the same speed as `STRIX` but the highest
perplexity of the three, and the only one above the Q4_K_M reference. It is published for
completeness and for anyone who wants the smallest file, not because it is a good default. (On other
architectures `FAST` genuinely is faster; on this one it is not.)

`STRIX` and `STRIX_LEAN` are indistinguishable in perplexity (5.8877 vs 5.8871, ±0.068) and that gap
should **not** be read as ranking them — `STRIX` is recommended on bit-count. Note also that
perplexity measures next-token prediction on prose; it is not a proxy for code-execution
correctness, where the extra bits in `STRIX` have mattered on other models.

`ROCmFP4-BASE` (ftype 100) is **not** published here: it is larger and, on every axis measured, not
better.

## Quick start

```bash
hf download kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF --local-dir qwen38
```

```bash
llama-server \
  -m qwen38/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf \
  --spec-type draft-mtp \
  --model-draft qwen38/mtp-Qwen3.8-27B-Q4_0.gguf \
  --spec-draft-ngl 99 --spec-draft-device ROCm0 \
  --spec-draft-n-max 4 --spec-draft-n-min 0 --spec-draft-p-min 0.0 \
  -ngl 999 -fa on -dio --jinja -fit off --parallel 1 -dev ROCm0 \
  -c 65536 --host 127.0.0.1 --port 8080
```

The ROCm environment prefix is mandatory on a machine that also has a Vulkan llama.cpp build,
to avoid a soname clash:

```bash
env LD_LIBRARY_PATH=/path/to/ROCmFPX/build/bin:/opt/rocm/lib \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 GGML_HIP_ENABLE_UNIFIED_MEMORY=1 llama-server ...
```

Requires a llama.cpp built with ROCmFP4 support (ggml types 100–106). Stock llama.cpp rejects these
tensor types.

### Four flags that matter more than which file you pick

| Flag | Why |
| --- | --- |
| `--spec-draft-n-max 4` | **The default is 16**, which lands far down the far side of the curve — 19.17 tok/s at `n-max 12` versus 30.30 at 4. This single flag is worth more than the quantisation choice. |
| `--spec-draft-ngl 99` | Without it the draft head can be placed on CPU and the speculative gain disappears entirely. |
| `--jinja` | Required, or `chat_template_kwargs` are **silently ignored** — including the thinking controls below. |
| `-fit off` | llama.cpp's autofit reads `MemAvailable` on integrated GPUs, and that is at its lowest right after a model unload. Leaving it on can silently shrink context or push tensors to CPU. |

`-dio` is also worth setting: it cold-loads the 13.7 GiB file in 6.3 s.

> ### ⚠️ If the MTP draft command crashes on your build
> The separate-model `draft-mtp` path has known bugs in the legacy `charlie12345/ROCmFPX`
> line — reported on **Windows 11 / gfx1151 / HIP SDK 7.2** (five stacked bugs, ending in an
> h-row width mismatch in the draft's embedding buffer).
>
> **Fix: build the official repo instead — no patches needed.**
>
> ```bash
> git clone https://github.com/ROCmFPX/ROCmFPX.git
> ```
>
> The MTP path was reworked there (unified `n_embd_out` row widths, `t_h_nextn` reset in
> `llm_graph_result::reset()`, `ctx_other` wired centrally), which covers the whole chain.
> Reported and verified on that configuration: 24–31 tok/s, coherent output, tool calling working.
> Legacy-line patch: [PR #109](https://github.com/charlie12345/ROCmFPX/pull/109).
>
> Linux builds on the legacy line are not known to be affected.


## Speculative decoding (MTP)

Qwen3.8-27B ships a **multi-token-prediction head** — `nextn_predict_layers = 1`, published
separately by [ggml-org](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) as an 18-tensor draft
model (`block_count = 65`: the 64 base layers plus one nextn layer, with its own embeddings and
output head). The quants in this repository contain the 64 base layers only; the head is a separate
`--model-draft` file and is bundled here.

| `--spec-draft-n-max` | decode @8K | vs Q4_K_M | acceptance |
| ---: | ---: | ---: | ---: |
| off | 13.46 | 1.26× | — |
| 1 | 21.74 | 2.03× | 0.881 |
| 2 | 27.79 | 2.60× | 0.917 |
| 3 | 30.13 | 2.82× | 0.875 |
| **4** | **30.30** | **2.83×** | **0.926** |
| 5 | 27.52 | 2.57× | 0.844 |
| 6 | 27.55 | 2.58× | 0.900 |
| 8 | 21.87 | 2.04× | 0.875 |
| 12 | 19.17 | 1.79× | 0.845 |

Classic rise / knee / decline: past `n-max 4` the draft head's cost grows linearly while acceptance
compounds downward, and wasted draft work dominates.

**⚠️ Use the Q4_0 draft head, not Q8_0.** The head's per-step cost is dominated by its output
projection over a 248K-token vocabulary, so the cheaper head is *faster end to end* at equal or
better acceptance — and the advantage widens with depth: **+6.0%** at `n-max 1`, **+10.2%** at
`n-max 2`. Both are bundled so you can check this yourself.

### A faster path exists — external sidecar + custom build (~45 tok/s)

The MTP numbers above are what **this repository's own files** do under a ROCmFPX build. A different
combination measures **~50 % faster**, at the cost of two things this repo does not ship:

| config | code | prose | reason | json | accept |
|---|---:|---:|---:|---:|---:|
| this repo, MTP `n-max 4`, gate open (below) | 30.30 | — | — | — | 0.544–0.93 |
| **`FAST` target + DFlash2 sidecar, adaptive drafting** | **45.44** | **48.26** | **46.85** | **46.03** | **0.95+** |

At **~21k context** the same config sustains **32.3 tok/s** with full-length generations.

Verified independently at 9 reps/workload: **0 truncations in 36 generations**, every one reaching the
full 400 tokens. The mechanism is acceptance — adaptive draft length holds **0.95+** where fixed `n`
settles around 0.54.

**What you need beyond this repo:**
1. The **`Qwen3.8-27B-Q4_0_ROCMFP4_FAST.gguf`** target — already here.
2. A **DFlash2 draft sidecar**, third-party:
   [`agentionai/Qwen3.8-27B-DFlash2-ROCmFP4-FAST-GGUF`](https://huggingface.co/agentionai/Qwen3.8-27B-DFlash2-ROCmFP4-FAST-GGUF)
   (0.96 GiB, derived from [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)).
   **Not our work** — credit to those authors.
3. A **source build** of [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp),
   which adds `--spec-draft-adaptive`. Its prebuilt Linux asset did not run on our box
   (`OPENSSL_3.3.0 not found`, `GLIBC_2.43 not found` against OpenSSL 3.0.13 / glibc 2.39) — build it.

```bash
llama-server \
  -m Qwen3.8-27B-Q4_0_ROCMFP4_FAST.gguf \
  -md Qwen3.8-27B-DFlash2-Q4_0_ROCMFP4_FAST.gguf \
  --spec-type draft-dflash --spec-draft-adaptive \
  --spec-draft-n-min 3 --spec-draft-n-max 7 --spec-draft-ngl 99 \
  -ngl 999 -fa on --no-mmap -fit off -b 2048 -ub 1024 -t 16 -c 32768
```

> ### ⚠️ Two flags are load-bearing — omit either and it breaks
>
> **`--spec-draft-p-min 0.10`** — this build defaults to **`0.00`**, i.e. *no* confidence gate on the
> draft. Without it a low-probability EOS proposed at position 0 is accepted unchallenged and the request
> returns **exactly 1 token, empty, with no error and `/health` still 200**. We measured **6/28** such
> failures ungated at short context, and **every** request failing at ~21k context. With the gate:
> **0/40 truncations** shallow and full-length generation at 21k. It costs nothing — 45.61 vs 45.63 tok/s.
>
> **`--chat-template-file`** (the model's own template, extractable from the GGUF's
> `tokenizer.chat_template`) — without it this build does not render tool definitions and the model
> replies *"I don't have any weather tools available"*. With it: `finish_reason: tool_calls`.

> Backend note: this path runs on **Vulkan**. That contradicts our own earlier finding that Vulkan
> loses on this model and cannot do MTP — true of the ROCmFPX Vulkan backend, not of this fork's.

**MTP costs about 6% of prefill** (326 → 293 tok/s at 8K). That is the honest trade against ~2.2×
decode.

For scale: the memory-bandwidth ceiling for a *dense* 27B at ~13.7 GiB on this hardware is roughly
**18.6 tok/s**. Speculative decoding clears it because several tokens are produced per weight read.
This is a dense model — all 27B parameters are read on every token — which is why the ceiling is low
and why MTP matters so much more here than it would on a sparse MoE.

## Tool calling

**7/7, and 7/7 again with thinking enabled.** Run against this build through llama-swap, scored on
whether llama.cpp populates **native `tool_calls`** — raw XML appearing in `content` counts as a
failure, since that is the known breakage when the parser does not match the template's format.

| Case | thinking (`reasoning_effort: low`) | non-thinking |
| --- | --- | --- |
| multi-arg (string / int / bool) | ✅ | ✅ |
| nested object argument | ✅ | ✅ |
| enum constraint | ✅ | ✅ |
| correctly declines (no spurious call) | ✅ | ✅ |
| multi-turn tool-result follow-up | ✅ | ✅ |
| streaming tool-call deltas | ✅ | ✅ |
| **two parallel calls in one turn** | ✅ | ✅ |
| **Total** | **7/7** | **7/7** |

⭐ **Tool calling and reasoning are not mutually exclusive on this model.** Both arms scored 7/7, so
you can run tools with `reasoning_effort` active — no need to disable thinking for agent work.

⭐ **Parallel calls work.** This is worth calling out because it is a common failure point: several
open-weight models emit only the first call when asked for two in one turn. This one returns both.

Requires **`--jinja`**. The template emits the XML-style Qwen3-Coder tool format
(`<tool_call><function=name><parameter=arg>…`), and llama.cpp parses it into native `tool_calls` —
verified with nested objects surviving as real JSON objects rather than flattened strings.

Raw example (nested object, as returned):

```json
{"name": "book_flight",
 "arguments": {"passenger": {"name": "Alice Smith", "age": 34},
               "route": "LHR-JFK", "cabin": "business"}}
```

## Vision

Works, and is verified for **spatial correctness** rather than plausible-sounding output. A
four-quadrant colour image is scored on whether each colour lands in the *right* corner — a
misapplied attention mask names colours confidently but places them wrongly, so this distinguishes
a working projector from a fluent-but-broken one.

| `-fa` | Result |
| --- | ---: |
| `on` | **4/4** |
| `off` | **4/4** |

```bash
llama-server -m Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf \
  --mmproj mmproj-Qwen3.8-27B-BF16.gguf \
  -ngl 999 -fa on -dio --jinja -c 16384
```

**Vision does not require `-fa off` on this model.** Both settings scored 4/4, so text and images
can be served from one endpoint with flash attention enabled — there is no accuracy-versus-speed
split to manage here.

## Architecture note

Qwen3.8-27B is **dense with hybrid attention**: 64 layers, `full_attention_interval = 4`, so 16
full-attention layers and 48 linear/SSM layers.

Two practical consequences:

- **KV cache is cheap.** Only 16 of 64 layers grow with context, so decode barely decays with length
  — 13.46 tok/s at 8K versus 12.58 at 32K, a 6.5% drop over 4× the context.
- **Prompt caching does not work.** llama.cpp reports `forcing full prompt re-processing due to lack
  of cache data (SWA or hybrid/recurrent memory)` and reprocesses the entire prompt every turn. This
  is an upstream limitation of hybrid-memory models, not a property of these quants. Budget for full
  prefill on each turn in multi-turn use.

## Sampling

Qwen publishes two profiles. Note `repetition_penalty` stays at 1.0 in both — the intended lever is
`presence_penalty`.

| | thinking | instruct (non-thinking) |
| --- | ---: | ---: |
| `temperature` | 1.0 | 0.7 |
| `top_p` | 0.95 | 0.80 |
| `top_k` | 20 | 20 |
| `min_p` | 0.0 | 0.0 |
| `presence_penalty` | 0.0 | 1.5 |
| `repetition_penalty` | 1.0 | 1.0 |

`presence_penalty` above 1.5 can cause language mixing.

### ⛔ The reasoning-effort trap

Thinking is **on by default at `reasoning_effort: "xhigh"`**. With a small `max_tokens` the whole
budget goes to reasoning and `content` comes back **empty** — the model looks broken when it is not.

The chat template accepts **only** `xhigh`, `medium`, `low`. Anything else raises:

```
Unexpected reasoning effort {...}. Supported types are xhigh (default), medium, and low.
```

In particular **`reasoning_effort: "none"` is not valid and will throw.** To disable thinking, use
the separate flag:

```
--chat-template-kwargs '{"enable_thinking":false}'
```

To keep thinking but bound it, use `'{"reasoning_effort":"low"}'`.

## Quantization methodology

```bash
# 1. measure the real BPW first — the help text is not reliable
llama-quantize --dry-run Qwen3.8-27B-BF16.gguf /tmp/x.gguf Q4_0_ROCMFP4_STRIX 8

# 2. quantize (~150 s for 27B at 16 threads)
llama-quantize Qwen3.8-27B-BF16.gguf Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf Q4_0_ROCMFP4_STRIX 16
```

Source: `Qwen3.8-27B-BF16.gguf` (50.11 GiB, 851 tensors) from
[ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF). Architecture is
`qwen35`, already supported by the ROCmFPX fork — **no porting was required for this model.**

The dry-run step is not ceremony: the help text advertises STRIX at ~4.49 BPW and the real figure
here is **4.39**.

These were produced on the quantizer's **unweighted reference path**. The fork also has an
importance-matrix path (`rocmfp4_quantize_row_*_weighted`, with an exhaustive weighted scale search)
which is not used here — see *Not yet measured*.

## Files

| File | Size | Role |
| --- | ---: | --- |
| `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf` | 13.75 GiB | model — **recommended** |
| `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf` | 13.59 GiB | model — equivalent to STRIX on every measure taken |
| `Qwen3.8-27B-Q4_0_ROCMFP4_FAST.gguf` | 13.33 GiB | model — smallest, but no faster and slightly worse PPL |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | 1.56 GiB | MTP draft head (ggml-org's, unmodified) — **use this** |
| `mtp-Qwen3.8-27B-Q8_0.gguf` | 2.95 GiB | MTP draft head, higher precision — measurably slower |
| `mmproj-Qwen3.8-27B-BF16.gguf` | 0.87 GiB | vision projector (ggml-org's, unmodified) — verified 4/4 |

The draft heads are redistributed unmodified from
[ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) so the repo is
self-contained.

## Not yet measured

Listed so nobody mistakes absence for a pass:

| Test | Status |
| --- | --- |
| Context scaling beyond 65536 (native is 262144) | ❓ only 8K and 32K prompts measured |
| Vision beyond the spatial smoke test (OCR, charts, video) | ❓ only 4-quadrant placement verified |
| Multi-step agentic loop (plan → act → observe → finish) | ❓ not run; the 7 tool cases are single-step plus one follow-up |
| Imatrix-weighted ROCmFP4 build | ❓ not built — the fork supports it; these use the reference path |
| KL divergence vs BF16 | ❓ not measured (perplexity is) |
| MMLU-Pro, GPQA, GSM8K, HumanEval+ | ❓ not run |
| Long-context needle retrieval | ❓ not run |
| Sustained multi-thousand-token generation | ❓ not measured |
| Independent reproduction | ❓ none yet |

**Perplexity caveat.** 40 chunks at ctx 2048 on held-out wikitext-2 (never used for any calibration
here). It is a regression guard strong enough to say *no measurable quality loss versus Q4_K_M*; it
is not a capability benchmark and says nothing about coding, tool use or long-context behaviour.

## Known issues

1. **Vulkan / CUDA / CPU cannot load these files** — ROCmFP4 is a ROCm-only tensor format.
2. **Prompt caching is disabled by the architecture** — every turn reprocesses the full prompt.
3. **Small `max_tokens` returns empty `content`** — the budget goes to reasoning. Allow several
   hundred tokens, or set `enable_thinking: false`.
4. **`reasoning_effort: "none"` throws** — it is not a valid value despite appearing in some
   third-party documentation. Use `enable_thinking: false`.
5. **`--spec-draft-n-max` defaults to 16**, roughly halving throughput on this model.

## Independent results

None yet. If you run this build, please open a discussion with hardware, GPU, ROCm version, runtime
commit, exact command, context, prompt-processing tok/s, generation tok/s and peak RAM. Independent
reproductions will be listed separately from author benchmarks and carry more weight.

## License and attribution

Base model and MTP draft head are the Qwen team's, under Apache 2.0. ROCmFP4 quantisation types are
from the ROCmFPX fork of llama.cpp. This repository contains the quantised weights and the
measurements above.

<!-- CREDITS:START -->

## Acknowledgements

**[ROCmFPX](https://github.com/charlie12345/ROCmFPX) — maintained by
[`charlie12345`](https://github.com/charlie12345)**
The ROCmFP4 / ROCmFPX tensor formats (ggml types 100–106) exist only in this fork. Every ROCmFP4
file here was produced with its `llama-quantize` and runs on its runtime. Licensed MIT, based on
upstream llama.cpp.

**[llama.cpp](https://github.com/ggml-org/llama.cpp) — ggml-org and contributors**
The inference engine, GGUF format and conversion tooling everything here is built on, and the source
of the BF16 and MTP GGUFs used.

**[AMD ROCm](https://github.com/ROCm/ROCm)** — the compute platform these builds target, ROCm 7.2.4
on gfx1151 / Radeon 8060S.

**[Qwen team](https://huggingface.co/Qwen)** — the base model, its MTP head and its licence are
theirs. This repository contributes quantisation and measurement only.

If you use these files, please credit ROCmFPX alongside this repository.

<!-- CREDITS:END -->

## 🩹 Prompt caching with the MTP draft head — fixed

**Reported by a user of this repo:** with `--spec-type draft-mtp` loaded, `llama-server` disabled
prefix caching entirely. Every agentic turn reprocessed the whole prompt. Reproduced here on an
8045-token stable prefix:

| config | prompt_n | cache_n | prompt_ms |
|---|---|---|---|
| no draft head | 519 | 7526 reused | 1 908 |
| **draft head (the defect)** | 8045 | **0** | **27 948** |
| **draft head + this patch** | **4** | **5101** | **100** |

**279× less prompt processing per turn**, with MTP still drafting.

### Root cause
The saved speculative state is the **MTP boundary** — the target model's pre-norm hidden row at the
cached prompt's *exact end position*. Any partial-prefix reuse would leave it describing a position
that no longer exists, so the server demanded an exact full-prefix match and otherwise reprocessed
cold, erasing its own context checkpoints on the way.

### The fix
`patches/mtp-prompt-cache-fix.patch` (4 files, applies to `2809dc5`) captures the speculative
boundary **inside the context checkpoint** (`common_prompt_checkpoint::data_spec`).
`create_checkpoint` runs between decode batches — exactly where that boundary is valid — so exact
state is saved and restored **together with the KV**, never rebuilt.

⛔ Two approaches were tried first and **rejected**: rebuilding the boundary from a zero-fill
changed the model's output (deterministically, 3/3), and truncating the KV back to the reuse point
is impossible here — the bounded rollback window is **4 tokens** against the 333 a real turn needs.
Exact state restore is the only shape that preserves output.

### Verification
Independently gated **10/10**: same prompt cold vs warm, `temperature 0`, byte-identical every run,
with the cache genuinely engaged (`cache_n=5101`, not a vacuous pass). The output hash also matches
the **unpatched** build, so behaviour is unchanged. Fails closed — an unreachable rollback logs
`reason=spec-checkpoint-missing` and cold-reprocesses rather than guessing.

### Related upstream
This is the same family as open llama.cpp issues
[#20225](https://github.com/ggml-org/llama.cpp/issues/20225),
[#19794](https://github.com/ggml-org/llama.cpp/issues/19794) and
[#24055](https://github.com/ggml-org/llama.cpp/issues/24055) — checkpoints being invalidated on
hybrid/recurrent models. This patch is **not** upstreamed; it is offered here as-is.
