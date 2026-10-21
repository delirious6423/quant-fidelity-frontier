---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: coreml
pipeline_tag: text-generation
tags:
- coreml
- core-ml
- apple-silicon
- on-device
- agent
- tool-calling
- speculative-decoding
- qwen
- quantized
---

# Qwen3.8 27B — Core ML agent bundle for Apple Silicon Macs

**Qwen3.8 27B**, converted to a native **Core ML** stateful graph that runs on the GPU of an Apple
Silicon Mac, packaged for a **web-browsing agent**: a 16,384-token context, int8 weights, and a
built-in multi-token-prediction head for lossless self-speculative decoding. With the companion Swift
runtime it searches the web, reads pages, writes notes and answers with sources — fully offline
except for the pages it fetches.

Measured on one machine (Apple M4 Max, 128 GB, macOS 26.6.2): **16.7 tok/s** median decode,
**55.7 GB** peak memory, and **48 of 48** agent runs completed on a frozen 12-task benchmark.
Details and conditions are in [Key numbers](#key-numbers).

> A Core ML conversion of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
> (Apache 2.0) — see [License & attribution](#license--attribution).

## Who this is for

- **Yes** — you have a Mac with **96 GB or more** of unified memory and want a 27B agent that runs
  locally through Core ML, with reproducible numbers.
- **Yes** — you are building a Core ML LLM runtime and want a reference for a **hybrid** model
  (48 gated-delta-rule linear-attention layers + 16 full-attention layers) on `MLState`, static verify
  functions for lossless speculation, and text-prefix KV reuse across agent turns.
- **No** — 64 GB Macs: the measured peak footprint is 55.7 GB, which leaves no room for the OS.
  Everything here was measured on a 128 GB Mac; 96 GB Macs have the headroom on paper but were not
  tested. An int4 build that fits 64 GB was measured but is not published (see
  [Limitations](#limitations)).
- **No** — iPhone / iPad: the graph needs the Mac GPU and 30 GB of weights.
- **No** — long documents beyond 16K tokens: this is a 16,384-token context build.

**Requires:** an Apple Silicon Mac with 96 GB+ unified memory on macOS 26+, about 30 GB of disk, and
the companion Swift runtime (below).

## Quick start

```bash
hf download okayuji/Qwen3.8-27B-coreml-agent --local-dir ./qwen38-27b-agent
```

This repository is **weights + config only**. The stateful chain, the speculation loop and the agent
(search, page reading, notes, tool calls) are the companion Swift runtime and demo app at
**https://github.com/oka-yuji/coreml-llm-samples**:

```bash
git clone https://github.com/oka-yuji/coreml-llm-samples
cd coreml-llm-samples
swift run -c release --package-path CoreLLMKit corellm-chat --model ../qwen38-27b-agent --stats \
  --prompt "List three fruits, one per line."
```

The first load on a machine compiles the graph for the GPU (94 s in the CLI when measured); later
loads take 8–9 s in the CLI and about 47 s per process in the demo app (see [Load time](#load-time)). To run the agent, open the
demo app (`CoreMLSamples.xcodeproj` at the repository root), choose the bundle under **Models**, press **Load in Chat**, then switch
to **Agent**. If you already ran `hf download`, the runtime repository's README (section **Agent**)
explains how to register that folder so the app does not download it again. Speculative decoding is
on by default and changes only the speed: the output is byte-identical with it off (`--no-mtp`).

## Sample output

Demo app headless driver (`DemoApp --agent-e2e`), Apple M4 Max 128 GB, macOS 26.6.2, GPU, greedy,
reasoning effort `low`, page budget 10,000 characters, 2026-09-13; this is task S3, round 1, of the
published benchmark.

Task:

```
Search for what Apple's MLX framework is and summarise it in three lines.
```

The agent called `web_search` once, then answered (shown verbatim):

```
**MLX** is an open-source array framework for machine learning on Apple silicon, developed by Apple's machine learning research team. It offers a NumPy-like Python API (plus C++, C, and Swift bindings) and is optimized for the unified memory architecture of Apple chips, supporting both CPU and GPU. It features lazy computation, composable function transformations, and is used for everything from numerical computing to running large language models.

Sources:
- https://github.com/ml-explore/mlx
- https://mlx-framework.org/
- https://developer.apple.com/videos/play/wwdc2025/315/
```

| Step | Prompt tokens | Reused from KV | Generated | TTFT | Decode |
|---|---|---|---|---|---|
| 2 (answer) | 1,355 | 945 | 153 (eos) | 3.45 s | 16.0 tok/s |

Whole task: 25.6 s after a 47.0 s bundle load. The runtime checks every URL in a `Sources:` section
against the pages and search results the agent saw during the task; here all three were seen.

## Key numbers

All numbers below were measured on an Apple M4 Max (Mac16,6) with 128 GB unified memory,
macOS 26.6.2 unless stated, GPU compute units, greedy decoding, one process at a time. The agent
benchmark, its task list, grading script and per-run summaries are published in the runtime
repository at `docs/results/2026-09-13-qwen38-agent-benchmark.md`. The conversion-time measurements
(CLI speed, load time, reference agreement) are in the author's lab notes, which are not published;
they are reproduced here with their conditions.

### Agent benchmark (2026-09-13, demo app headless driver, runtime tag `qwen38-27b-agent-v1`)

12 frozen tasks (4 search-and-summarise in Japanese and English, 3 read-this-URL, 3 look-up-and-save-a-note,
2 date-dependent), each graded on answer content, a `Sources:` section with retrieved URLs, the note
file when one was requested, and completion without a step limit or token cap.

| Condition | Runs | Passed |
|---|---|---|
| reasoning effort `low`, 3 rounds | 36 | 36 |
| reasoning effort `off`, 1 round | 12 | 12 |
| **All** | **48** | **48 (100%)** |
| `low` with file operations enabled (4 tools), 1 round | 12 | 12 |

Across the 148 generations of the 48 runs: decode median **16.65 tok/s** (12.4–19.7), time to first
token median **6.9 s** (0.7–46.4; the maximum is the first prefill of a session), peak footprint
median **55,662 MB** (max 55,716 MB), speculation acceptance median 0.86, every generation ended at
EOS, zero token-cap cuts, zero full re-prefills after the first step.

### Decode speed by speculation (CLI `corellm-chat`, 2026-09-03, macOS 26.5.2)

| Prompt | Speculation off | Speculation on (S=4, k=3) |
|---|---|---|
| Japanese | 169.5 ms/tok (5.9 tok/s) | 91.7 ms/tok (10.9 tok/s) |
| English | 172.6 ms/tok (5.8 tok/s) | 72.8 ms/tok (13.7 tok/s) |
| Code | 166.8 ms/tok (6.0 tok/s) | 59.4 ms/tok (16.8 tok/s) |

Output is byte-identical between the two columns: on 2026-08-18 the three probe prompts (Japanese
383 B, English 565 B, code 232 B) matched byte for byte with speculation off and on, and a 2026-08-26
smoke prompt matched by SHA-256 (author's lab records). Peak footprint in the CLI was 53.6–54.6 GB
in the 2026-09-03 session; the demo app measures 55.7 GB because it also holds the UI and the web
view.

### Load time

| Path | First load on a machine | Later loads |
|---|---|---|
| Demo app, per process (bundle + tokenizer; the MTP head loads at the first generation) | not timed separately (the OS compiles the graph once and caches it) | 47.4 s median, 46.6–49.9 s (48 runs, 2026-09-13) |
| CLI `corellm-chat` (2026-09-03, macOS 26.5.2) | 94 s | 8–9 s |

## What's inside

- **Hybrid Qwen3.8 27B**: 64 layers, 48 of them gated-delta-rule linear attention (recurrent
  `conv_state` / `rec_state` per layer) and 16 full attention (KV `MLState` of shape
  `[4, 16384, 256]`), hidden size 5,120, 24 query heads, 4 KV heads.
- **Four chunks** of 16 layers each, compiled as `.mlmodelc` with three functions per chunk:
  `main` (prefill, variable width), `decode` (one token) and `verify` (a static 4-row function used
  for speculation and, in the same shape, for lossless one-token decoding).
- **int8 weights** (per-block symmetric, block size 32) for the 2-D linear weights of the chunks and
  the LM head; the linear-attention projections, convolution taps, decay parameters and RMSNorm
  weights stay fp16. The embedding table ships as fp16.
- **MTP head** (`mtp.mlmodelc`, int8, 451 MB): the model's own multi-token-prediction head drafts up
  to three tokens per round; the `verify` function accepts the greedy-correct prefix and writes the
  accepted linear-attention state back, so speculation is lossless by construction.
- **Context 16,384** tokens, addressed with a one-hot position write; the recurrent layers are
  context-length independent.
- **Compute units** `cpuAndGPU`: the multi-`MLState` design needs the GPU; the Neural Engine is not
  used.

## Input format

From `manifest.json` (ChatML with Qwen's thinking block):

```
promptPrefix:         "<|im_start|>user\n"
promptSuffix:         "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   (thinking off)
promptSuffixThinking: "<|im_end|>\n<|im_start|>assistant\n<think>\n"                 (thinking on)
assistantSuffix:      "<|im_end|>\n"
bos: none            eos: 248044, 248046
```

The demo app's agent renders a system turn with the tool definitions and reads tool calls in Qwen's
`<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` form; the exact
prompt and the byte-for-byte re-rendering rules are in
`docs/agent-demo.md` of the runtime repository. Multi-turn KV reuse works by text prefix: the runtime
tokenises only the new suffix of a conversation, so a generated turn is never re-tokenised.

## Files

| File | Role | Size |
|---|---|---|
| `chunk_0_16.mlmodelc` … `chunk_48_64.mlmodelc` | four 16-layer chunks, `main` / `decode` / `verify` (each `weights/weight.bin` is 6,475,094,400 bytes; the last chunk 6,475,104,704) | 6.48 GB each |
| `lmhead.mlmodelc` | LM head, int8 | 1.35 GB |
| `mtp.mlmodelc` | multi-token-prediction head, int8 | 451 MB |
| `embed_fp16.bin` | embedding table | 2.54 GB |
| `convert_config_v2int4.json` | chain configuration read by the runtime (`ChainConfigV2`); despite the file name the chunks are int8 — the `quant` block inside is authoritative and the name cannot change because the loader probes a fixed file list | 22 KB |
| `manifest.json` | bundle manifest: format `coreml-stateful-chain-v2`, prompt template, EOS ids, compute units | 1 KB |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | tokenizer | 23 MB |
| `config.json`, `generation_config.json`, `chat_template.jinja` | upstream configuration, kept for reference | 14 KB |
| `LICENSE` | Apache License 2.0 (from the base model repository) | 12 KB |

35 files, 30,285,202,520 bytes in total (30.3 GB, 28.2 GiB).

## Requirements

- Apple Silicon Mac, macOS 26 or later, **96 GB or more** unified memory (measured peak 55.7 GB;
  macOS needs the rest for itself and for the graph compilation).
- About 30 GB of disk for the bundle (30.3 GB, 28.2 GiB), plus the OS's compiled-model cache.
- The runtime repository builds with Xcode 26 / Swift 6.2. The demo app must be built for `arm64`.

## Operational notes & troubleshooting

- **First load takes minutes.** The OS compiles the four chunks for the GPU on first use and caches
  the result; a later load in the demo app takes about 47 s per process, in the CLI about 8–9 s.
- **Memory pressure.** If macOS shows memory pressure in yellow while loading, close other large
  processes; the bundle needs its 55.7 GB resident.
- **Memory figures on macOS 27.** The 55.7 GB peak is the OS's footprint metric on macOS 26.6.2.
  On macOS 27.0 the same load keeps about 50 GiB resident (RSS 50.2 GiB measured after loading)
  while the footprint metric reports only a few GB, because the weights are mapped as clean
  file-backed pages; the demo app's peak figure and `metrics.jsonl` therefore read far lower on
  macOS 27 although the bundle occupies the same memory. Do not compare peak figures across OS
  versions.
- **The agent's tools.** `web_search` uses DuckDuckGo's HTML endpoint by default (a Brave Search API
  key is optional); `fetch_page` reads pages in a web view with a 20 s deadline and refuses local and
  private addresses; `write_note` saves Markdown under the app's notes folder and never overwrites;
  `move_note` is off unless enabled in Settings and only moves notes into Desktop, Documents or
  Downloads.
- **Changing settings mid-conversation** (reasoning effort, file operations) changes the system prompt,
  so the next generation re-prefills the whole conversation instead of reusing the cache.
- **`cannot rewind from N to M`** in the log means the runtime had to re-prefill: hybrid recurrent
  state cannot be rolled back. With the current runtime this no longer happens across normal agent
  turns (0 of 148 generations in the benchmark); it still happens after a stop or a settings change.

## Limitations

- **Memory.** 96 GB+ Macs only. The int4 build that would fit a 64 GB Mac (30.5 GB peak) was
  converted and measured: about the same speed (0.93–1.20× of int8 on the shipped path) but 11 hard
  top-1 flips in 640 positions against the bf16 reference, versus 0 for int8. It is not published.
- **Context.** 16,384 tokens. In the benchmark the longest prompt reached 9,309 tokens; two or three
  long pages in one task fill the window, after which the agent drops the oldest tool output.
- **Speed.** 16–17 tok/s is a 27B model on a single GPU with int8 weights. An 8K-context build of
  the same weights was measured 1.19–1.21× faster on the same path with byte-identical output on
  18 of 18 prompts (`docs/results/2026-09-03-qwen38-ctx-tax.md`); it halves the context and is not
  published.
- **Search quality.** DuckDuckGo's HTML endpoint is a scraper target and can change; results vary.
- **Thinking is in English** even for Japanese tasks; only the answers are in the user's language.
- **Model identity.** Without an explicit instruction the model describes itself as "Qwen" and
  declines to name its version; the demo app's system prompt tells it to answer with the bundle's
  description.
- Measured on one machine and one OS version; the numbers are that machine's.

## Verification

- **Reference agreement (2026-09-03, author's lab record, not published).**
  Against the bf16 reference implementation, top-1 agreement on 10 prompts × 64 positions was
  633/640 (98.9%) with **0 hard flips** (the 7 disagreements are near-ties within a 0.6 logit
  margin); a 32-token teacher-forced sequence matched exactly (32/32).
- **Lossless one-token decode through the 4-row verify function**: hidden state, `conv_state` and
  `rec_state` at position 0 are bit-equal (100.0000%, max abs diff 0) over 200 positions regardless of
  the filler in rows 1–3; the accepted-prefix write-back after speculation is bit-equal for every
  accepted length 1–4.
- **Speculation off = speculation on**: 3 of 3 probe prompts (Japanese, English, code) produced
  byte-identical text on 2026-08-18, and the accepted-prefix write-back is bit-equal for every
  accepted length, so the property holds by construction (author's lab record, not published; the
  gating method is described in the runtime repository's `docs/verification.md`).
- **Agent benchmark**: 48 of 48 runs on the frozen 12-task set, with the task list and grading
  script frozen by SHA-256 and published with the per-run summaries
  (`docs/results/2026-09-13-qwen38-agent-benchmark.md` in the runtime repository).
- **GUI end-to-end**: a driver inside the demo app (`DemoApp --gui-e2e <dir>`, part of the runtime
  repository) walks the Agent screen through 14 checks (load, tool calls, note save and move, stop
  and resume, screen switching, unload) and captures screenshots; 14 of 14 passed on 2026-09-16 on
  macOS 27.0 at runtime tag `qwen38-27b-agent-v1`.

## License & attribution

The weights in this repository are a quantized Core ML conversion of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (revision `1d4bf0f`), released by the
Qwen team under the **Apache License 2.0**. The same license applies to this conversion; the license
text is included as `LICENSE`. Changes from the original: weights quantized to int8 (per-block,
block size 32) and split into four Core ML chunks with static verify functions, an MTP head compiled
to Core ML, and a 16,384-token context configuration. The companion Swift runtime is licensed
separately in its own repository. Qwen is a model family of Alibaba Cloud; this conversion is an
independent work and is not affiliated with or endorsed by Alibaba Cloud.
