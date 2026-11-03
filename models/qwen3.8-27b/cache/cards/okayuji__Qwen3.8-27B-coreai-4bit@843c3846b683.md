---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- coreai
- apple
- macos
- qwen3.8
- 4-bit
- on-device
library_name: coreai
---

# Qwen3.8 27B for Core AI: local text generation on Mac

Run the text part of Qwen3.8 27B on an Apple silicon Mac with 4-bit weights. The public Swift sample includes command-line chat and a Foundation Models `LanguageModel` adapter for `LanguageModelSession`, plus an app that searches the web, reads pages and saves notes.

**For research and verification.** Agreement with the same 4-bit weights in PyTorch fp32 **failed the pre-set criteria**: maximum probability difference 0.778 against a limit of 0.0875. This is not a validated production model; [the complete results and conditions](#verification) remain below.

[Swift sample app](https://github.com/oka-yuji/coreai-samples) · [Chat and Work guide](https://github.com/oka-yuji/coreai-samples/blob/main/docs/qwen3.8-27b.md) · [Runtime guide](https://github.com/oka-yuji/coreai-samples/blob/main/docs/runtime.md) · [Verification](#verification)

**For:** Swift developers evaluating a 27B local model or checking a Core AI runtime. **Not for:** iPhone/iPad, image input, Transformers, MLX or llama.cpp; this is a text-only macOS `.aimodel` export with dynamic shapes.
**Requires:** macOS 27, Xcode 27 with Metal Toolchain, and 16.85 GB of disk plus about the same for each program's prepared copy. Only M4 Max 128 GiB was measured; loading reached 34.2 GB (31.9 GiB) resident, and the smallest usable Mac is unknown.

## Quick start

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), then run:

```sh
git clone https://github.com/oka-yuji/coreai-samples.git
cd coreai-samples
hf download okayuji/Qwen3.8-27B-coreai-4bit --revision v1.0.0 --local-dir ./models/qwen3.8-27b-coreai-4bit
cd CoreAISamplesKit
swift build -c release
.build/release/coreai-chat --model ../models/qwen3.8-27b-coreai-4bit --max-tokens 2048 --stats \
  --prompt "What is the capital of France? Answer with one word."
```

This repository holds the bundle only; the runtime, command-line chat and app are in [coreai-samples](https://github.com/oka-yuji/coreai-samples). The tag `v1.0.0` marks the measured release, and the companion app pins its commit. The seven bundle files are the ones measured. Thinking prints under `[reasoning]`, the answer under `[response]`, and statistics on stderr.

To use the app, open `CoreAISamples.xcodeproj` at the repository root and run the **CoreAISamples** scheme on **My Mac**. Choose **Models → Open Folder…**, select `models/qwen3.8-27b-coreai-4bit`, press **Load**, then use **Chat** or **Work**. First load took 86.87 s in the app, 82.84 s preparing the model for the GPU; later loads took 3.6–10.3 s under the [recorded conditions](#key-numbers). Enable **Search and sources → Web** for web research.

<details>
<summary>Swift integration with Foundation Models</summary>

From Swift, with `CoreAIBackend` from the companion package:

```swift
import CoreAIBackend
import Foundation
import FoundationModels

let model = try await LocalLanguageModel(bundle: URL(filePath: "/path/to/qwen3.8-27b-coreai-4bit"))
let session = LanguageModelSession(model: model, instructions: "You are a concise assistant.")
for try await snapshot in session.streamResponse(
    to: "What is the capital of France? Answer with one word.",
    options: GenerationOptions(maximumResponseTokens: 2048)
) {
    print(snapshot.content)
}
```

</details>

## Sample output

The companion app's headless driver (`--agent-e2e`), reasoning off, greedy, on the live web, October 12, 2026, Apple M4 Max with 128 GiB, macOS 27.0 (26A428). This run used `--instructions cite-fetched --citation-guard on`; the app's default is the same instructions with the guard off.

Task:

```
Read https://developer.apple.com/documentation/createml and list three key points.
```

The answer and the two notes the app adds under it, verbatim:

```
Here are three key points from Apple's Create ML documentation:

1. **Train custom ML models on your Mac with familiar tools.** Create ML lets you use Swift and macOS playgrounds to create and train machine learning models locally. You train a model by showing it representative samples (e.g., many images of dogs to recognize dogs), then test it on unseen data and evaluate performance before integrating it into your app via Core ML.

2. **Broad model types are supported.** The framework covers image models (image classifier, object detector, hand pose classifier), video models (action classifier, hand action classifier, style transfer), text models (text classifier, word tagger, word embedding, gazetteer), sound classifiers, motion/activity classifiers, and tabular models (classifiers, regressors, recommenders).

3. **It leverages Apple's built-in ML infrastructure.** Because Create ML builds on the machine learning infrastructure used in Apple products like Photos and Siri, your image classification and natural language models are smaller and train much faster. It also provides built-in tools for evaluating model accuracy (metrics for classifiers, regressors, word taggers, recommenders, and object detectors) and controlling training sessions (MLJob, MLTrainingSession, checkpoints).

**Platform availability:** iOS 15.0+, iPadOS 15.0+, Mac Catalyst 15.0+, macOS 10.14+, tvOS 16.0+, visionOS 1.0+.

Sources:
https://developer.apple.com/documentation/createml
--- note Citations
1/1 cited URLs were retrieved in this task · mean 8-gram overlap 0.38
--- note Citation guard
Cited sources fetched · 1/1 cited URLs were fetched in this task
```

| Step | Prompt tokens | Reused | Generated | First token | Decode |
|---|---|---|---|---|---|
| 1 (`fetch_page` call) | 877 | 0 | 34 | 7.97 s | 23.8 tok/s |
| 2 (answer) | 2,189 | 911 | 317 | 4.95 s | 23.0 tok/s |

The page fetch took 6.3 s and returned 5,993 characters. Whole task: 33.3 s after a 4.5 s load from the prepared copy.

## Key numbers

Measured on one machine (Apple M4 Max with 128 GiB, macOS 27.0): **22.74 tok/s** median decode, **33.8–34.2 GB** (31.5–31.9 GiB) peak resident memory while loading, and **48 of 48** agent runs passed on a frozen 12-task benchmark. Agreement with the same 4-bit weights run in PyTorch fp32 did **not** meet the criteria fixed before it was measured. Conditions and details are in [Key numbers](#key-numbers) and [Verification](#verification).

MacBook Pro (Mac16,6) with an Apple M4 Max and 128 GiB of unified memory (137,438,953,472 bytes), macOS 27.0 (26A428), Xcode 27.0 (27A266a), these seven bundle files, the companion runtime, GPU, greedy decoding, one model process at a time, October 11–12, 2026.

### Agent

| Measurement | Result | Conditions |
|---|---|---|
| Frozen 12-task benchmark | 48/48 passed the frozen grader; answer quality 48/48; no unintended re-render; no run that failed to stop | 12 tasks × 4 rounds on the live web (DuckDuckGo), `--instructions frozen --citation-guard off`, reasoning low (36 runs) and off (12) |
| Decode speed | 22.74 tok/s, median of 144 generation steps | the same 48 runs |
| Time to first token | 5.54 s to read the prompt, 6.46 s to the first text on screen (medians) | the same 144 steps |
| Task time | 50.4 s median | the same 48 runs |
| With the citation guard on | 41/42 tasks achieved, 1 held; the guard verified 35, repaired 6, held 1; median task time 56.3 s, p95 138.5 s; decode 22.90 tok/s; first token 5.33 s / 6.46 s (151 steps) | 21 tasks × reasoning low and off on the live web, `--instructions cite-fetched --citation-guard on` |
| Default settings | not measured with this bundle | cite-fetched instructions with the guard off |

### Load, memory and longer use

| Measurement | Result | Conditions |
|---|---|---|
| First load in the app | 82.84 s preparing for the GPU, 86.87 s in all | no prepared copy |
| Later loads in the app | 3.6–10.3 s, median 4.7 s | 48 runs |
| Peak resident memory | 33.8 GB (31.5 GiB) at a load from the prepared copy; 34.2 GB (31.9 GiB) at the first load | `/usr/bin/time -l`, maximum resident set size of the app process |
| Resident memory while running | 16.0–16.7 GiB during Agent steps; 17.26–17.36 GB in 100 CLI conversations | the app's `metrics.jsonl`; `coreai-verify chat` |
| Prepared copy | 16.86 GB (15.70 GiB) per program | `~/Library/Caches/coreai-cache/<OS build>/<program>/` |
| 100 new conversations | decode 22.69 tok/s at the 5th, 20.93 tok/s at the 100th (92.2%); 20.11–21.62 tok/s from the 41st on, except the first question after each reload; resident memory +0.28% from the 10th to the 100th; periodic reloads 4.31–4.77 s | `coreai-verify chat`, a new conversation per prompt, up to 256 tokens each, reload every 10 conversations |
| RAG first text on screen | 15.53 s median of 3 (15.36–15.60 s), 2,950 source tokens, 3,120 input tokens | the app's `--selftest --rag`, 3,000-token source budget, local documents only |

## What's inside

- **Model**: Qwen3.8 27B text decoder, 64 layers (48 gated-delta linear-attention layers and 16 full-attention layers), hidden size 5,120, 24 query heads and 4 key-value heads, vocabulary 248,320 (from the original `config.json`).
- **Conversion**: Apple's converter `coreai.llm.export` from [apple/coreai-models pull request #321](https://github.com/apple/coreai-models/pull/321) at commit `36e9350925b24637a39e52b51c45a68afbec8916`, which is not part of the 1.0.0 tag. Preset `qwen3_5_cfg`: 4-bit weights, bfloat16 compute. One change to the recipe: in each linear-attention layer, the inputs of the state update (`gated_delta_update`) are cast to fp32 and its outputs cast back. macOS export with dynamic shapes; text only. Tools: coreai-torch 0.4.3, coreai-core 1.0.0b3, coreai-opt 0.3.0, torch 2.9.0, transformers 5.12.1. Converted on 2026-10-06 (UTC).
- **Why the fp32 state update**: an earlier export of the same recipe without it had larger differences from the PyTorch reference on the same inputs (99th percentile 0.04901 against 0.02796; top-1 agreement 98.79% against 99.21%).
- **Tokenizer**: the converter re-saves `tokenizer.json` with a different pre-tokenizer pattern; this bundle carries the original file instead. On 2,257 strings from the measurements, the companion runtime's Swift tokenizer produced the same IDs with either file; Python's `tokenizers` differed on two Hindi and Thai texts.

## Input format

- The chat template is `tokenizer/chat_template.jinja`, byte-identical to the original. Messages use `<|im_start|>` (248045) and `<|im_end|>` (248046); the companion runtime stops at `<|im_end|>` or `<|endoftext|>` (248044).
- Thinking appears between `<think>` (248068) and `</think>` (248069). The template reads `reasoning_effort` (`low`, `medium`, `xhigh`) and `enable_thinking`; in the companion runtime, `ContextOptions.reasoningLevel` maps `.light` / `.moderate` / `.deep` to those values and `.custom("off")` sets `enable_thinking` to false.
- Tool calls are `<tool_call>` blocks (248058, 248059) in JSON or in the `<function=...>` / `<parameter=...>` form.
- `metadata.json` declares `max_context_length` 262,144. The companion app uses 16,384 tokens; longer contexts were not checked.
- The original `generation_config.json` suggests sampling (temperature 1.0, top-k 20, top-p 0.95). The measurements here are greedy, which is the companion runtime's behavior when no sampling options are given.

## Files

| File | Role | Size (bytes) |
|---|---|---|
| `qwen3_8_27b_qwen3_5_cfg_gdnfp32_dynamic.aimodel/main.mlirb` | the model | 16,839,684,298 |
| `qwen3_8_27b_qwen3_5_cfg_gdnfp32_dynamic.aimodel/main.hash` | SHA-256 of `main.mlirb`; also names the prepared copy | 32 |
| `qwen3_8_27b_qwen3_5_cfg_gdnfp32_dynamic.aimodel/metadata.json` | asset metadata written by the converter | 394 |
| `metadata.json` | bundle metadata: vocabulary, maximum context, source model | 611 |
| `tokenizer/tokenizer.json` | the original Qwen3.8-27B tokenizer | 12,809,320 |
| `tokenizer/chat_template.jinja` | the original chat template | 8,952 |
| `tokenizer/tokenizer_config.json` | written by the converter; differs from the original | 1,124 |
| `SHA256SUMS` | SHA-256 of the seven bundle files above | 742 |
| `LICENSE` | Apache License 2.0, the original repository's file, unchanged | 11,544 |
| `NOTICE.md` | the origin, the conversion and the changed files | 2,031 |
| `README.md` | this card | — |

The seven bundle files total 16,852,504,731 bytes; with this card and the other small files the download is about 16.85 GB (15.70 GiB). Check a download with `shasum -a 256 -c SHA256SUMS` in its folder.

## Requirements

- Apple silicon Mac, macOS 27, GPU. Measured only on an M4 Max with 128 GiB; plan for about 34 GB (31.9 GiB) resident while loading.
- Disk: 16.85 GB for the bundle, and a prepared copy of about the same size for each program that loads it (the demo app and `coreai-chat` make separate copies).
- Xcode 27 with the Metal Toolchain component to build the companion runtime.

## Operational notes & troubleshooting

- **The first load takes over a minute.** Core AI prepares the model for the GPU and keeps the result under `~/Library/Caches/coreai-cache/<OS build>/<program>/<main.hash>/`. The demo app spent 82.84 s of an 86.87 s first load preparing; `coreai-chat` took 105.6 s to prepare on a busy Mac (1-minute load average 20–29). Another program, or a new OS build, prepares its own copy.
- **Removing a prepared copy**: **Clear Prepared Copy** in the demo app's Models screen, `Engine.deleteSpecializationCache(bundle:)`, or delete the folder.
- **Slow downloads**: if `hf download` stays below 1 MB/s, `HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY=32 HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY=32` raised the speed from 0.2–0.5 MB/s to 18.91 MB/s (891 s for this bundle) in one measurement (hf-xet 1.5.2, huggingface_hub 0.36.2, October 11, 2026).
- **Thinking uses tokens.** With a small `--max-tokens`, the reply can end while the model is still thinking and the answer is empty (16 tokens were all thinking for "Say hi."). Leave room, for example 2,048 tokens.
- **When the agent's context fills up**, the runtime drops the oldest tool outputs; if the prompt still does not fit, the task ends with `Context window is full`.

## Limitations

- **Agreement with the original weights** did not meet the criteria fixed before measuring ([Verification](#verification)): at a small share of positions Core AI's probabilities differ from the same weights in PyTorch, most of all in English prose and in chat-rendered tool results and sources.
- **Measured on one machine**, an M4 Max with 128 GiB on macOS 27.0 (26A428). Other Macs, other OS builds, iOS and contexts beyond 16,384 tokens were not measured.
- **Default agent settings** of the companion app (cite-fetched instructions, citation guard off) were not measured with this bundle.
- **Citations and the citation guard.** The app's **Citations** note counts a cited URL as retrieved when it appeared in the task's search results or in a fetched page, so `4/4 cited URLs were retrieved in this task` was shown for a task that opened no page. The citation guard (off by default) checks against the pages `fetch_page` delivered, but not whether the answer agrees with them; it reports failed fetches only as a count, and reads letters that follow a URL without a space, such as Japanese text, as part of the URL. With the guard on, 1 of 42 live-web runs ended held: its repair turn fetched the page the first answer had cited without opening, but the repaired answer stopped at the token limit, so the app showed the first answer, whose check lists that page as not fetched, with a note saying so.
- **Pages can steer the agent.** Of 10 runs over 5 pages written with instructions in them (citation guard on), 2 saved a note a page asked for; none fetched a URL a page asked for or wrote outside the notes folder. Without **Allow file operations** the tools write only into the notes folder.
- **Some pages fail to load**: `fetch_page` stops with `error: fetch_page needs an absolute http or https URL` when an embedded frame, often an advertisement, navigates to a non-web address (4 of the 48 benchmark runs).
- **Step limit, historical and current behavior**: the Agent build measured here stopped with "Step limit reached" and left no answer. Current Chat and Work make one extra answer attempt without tools after the research budget is exhausted; the attempt can still fail or exhaust the context. Its answer quality with this Qwen bundle has not been measured. See the [current budget contract](https://github.com/oka-yuji/coreai-samples/blob/main/docs/specs/conversation-step-budget.md).
- **One session per model at a time** unless the companion runtime's simultaneous-session checks pass for your setup.
- **Converter revision**: the export comes from a pull request of apple/coreai-models that is not part of the 1.0.0 tag.
- **Quantization was measured on token probabilities only**, on five short inputs ([Verification](#verification)); task quality with and without 4-bit quantization was not compared.

## Verification

The public [Qwen model guide](https://github.com/oka-yuji/coreai-samples/blob/main/docs/qwen3.8-27b.md#verification) links the companion runtime behavior and keeps these historical results separate from later app changes.

All on the machine in [Key numbers](#key-numbers), on October 11–12, 2026 unless a bullet gives another date.

- **Runtime agreement with Apple's engine.** `coreai-verify compare --max-tokens 1024` fed four prompts (Japanese, English, code and a long generation) to the companion runtime and Apple's sequential engine on the same bundle. On the matching `prefill-function` path every generated token was the same at every step (43, 41, 293 and 1,024 steps), the maximum logit difference was 0 and no value was non-finite. This checks the runtime, not the conversion.
- **Agreement with the same 4-bit weights in PyTorch.** Reference: the same quantized weights evaluated by PyTorch in fp32 on the CPU, with the original tokenizer. Inputs: 40 texts not used before (Japanese, English, Hindi, Thai and code: 6 prose texts per language, plus one Agent-shaped and one RAG-shaped chat render), 512 positions each, read by Core AI in one 512-token call. Measure: the absolute difference in the probability each side gives the actual next token.

  | Inputs | Positions | Top-1 agreement | Mean | p99 | Above 0.03 | Max |
  |---|---|---|---|---|---|---|
  | Japanese | 4,096 | 99.07% | 0.002026 | 0.01576 | 11 (0.27%) | 0.778 |
  | English | 4,096 | 98.97% | 0.003584 | 0.04431 | 104 (2.54%) | 0.2137 |
  | Hindi | 4,096 | 99.41% | 0.001848 | 0.01799 | 8 (0.20%) | 0.1002 |
  | Thai | 4,096 | 99.44% | 0.002012 | 0.02007 | 22 (0.54%) | 0.2524 |
  | Code | 4,096 | 99.15% | 0.002706 | 0.028 | 37 (0.90%) | 0.6672 |
  | **All** | **20,480** | **99.21%** | **0.002435** | **0.02796** | **182 (0.89%)** | **0.778** |

  No value was NaN or infinite. Criteria fixed before the measurement: p99 at most 0.02 (not met), at most 0.5% of positions above 0.03 (not met), the top token unchanged at every position above 0.03 (not met: 26 differ), top-1 agreement at least 99% (met), mean at most 0.01 (met), no non-finite value (met), maximum at most 0.0875 (not met). By shape, positions above 0.03 were 106 of 15,360 in prose (1 of 9,216 in Japanese, Hindi and Thai prose), 44 of 2,560 in Agent-shaped renders and 32 of 2,560 in RAG-shaped renders. The largest differences fall on copied URLs, version numbers and the probability of ending a message. Two longer inputs of 8,192 and 15,360 tokens showed no growth toward their ends (mean difference in the last quarter 1.19× and 0.91× the first).
- **What the outliers mean in generation.** Core AI's probabilities depend on how the prompt is split into calls: reading the 10 Agent- and RAG-shaped inputs in 128-token calls instead of one 512-token call changed the top token at 77 of 5,120 positions, and moved the outliers rather than removing them. Greedy generation continued from 31 outlying positions stayed coherent in the assistant's own turns; 4–5 continuations of pasted source text went wrong (a stray Latin letter in Japanese text, broken code, an end of message inside code).
- **Effect of 4-bit quantization.** Reference: the original bfloat16 model run in PyTorch on the GPU (MPS), on five earlier inputs (the Universal Declaration of Human Rights in four languages and Swift source, 512 positions each), October 7, 2026. Measure as above, plus the mean KL divergence where it was computed.

  | Input | This model file in Core AI: top-1 / mean | Same 4-bit weights in PyTorch bfloat16: top-1 / mean / KL |
  |---|---|---|
  | Japanese | 85.74% / 0.06562 | 85.74% / 0.06668 / 0.14561 |
  | English | 98.63% / 0.01555 | 98.63% / 0.01572 / 0.03921 |
  | Hindi | 91.80% / 0.03528 | 90.62% / 0.03598 / 0.05595 |
  | Thai | 87.50% / 0.04842 | 87.50% / 0.04876 / 0.09982 |
  | Swift source | 93.75% / 0.02543 | 93.36% / 0.02594 / 0.02875 |
  | **All, 2,560 positions** | **91.48% / 0.03806** (maximum 0.9990) | **91.17% / 0.03862 / 0.07387** (maximum 0.9990) |

  On these inputs, Core AI and PyTorch running the same 4-bit weights in fp32 differed by a mean of 0.0019 (top-1 agreement 99.41%), and PyTorch's own fp32 and bfloat16 runs of those weights by 0.00395. These are reference values with no pass/fail criterion. Task quality with and without quantization was not compared.
- **Task results.** The benchmark results in [Key numbers](#key-numbers) measure finished tasks; they do not show numerical agreement with the original model. Two graders (the same AI model with the same written guide, not independent people) read the 48 frozen-instruction answers against the text the agent had received: 22 supported, 15 partly supported, 11 not supported; all 48 met the request. Ten of the 11 unsupported answers relied on search snippets without opening a page.
- **State and sampling.** The processed-state check passed in all 9 checked tool-loop runs, and seeded top-k and top-p sampling repeated the same tokens 5 of 5 times, with a different seed changing them.
- **Download check.** A fresh download of an earlier private revision of this repository, which holds the same seven bundle files with a placeholder card, passed `shasum -a 256 -c SHA256SUMS` for all 7 bundle files, and `coreai-chat` generated from that copy (October 11, 2026).

## License & attribution

The weights in this repository are a quantized Core AI conversion of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) at revision [`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), developed by the Qwen team; its files carry **Copyright 2026 Alibaba Cloud** and the **Apache License 2.0**, included unchanged as [LICENSE](LICENSE). Changes from the original: the text model converted with Apple's converter and its weights quantized to 4 bits, the gated-delta state update computed in fp32, the vision encoder left out, and `tokenizer/tokenizer_config.json` written by the converter; [NOTICE.md](NOTICE.md) lists the details. The same license applies to this conversion; retain `LICENSE` and `NOTICE.md` when redistributing. The companion Swift code is MIT-licensed in its own repository and does not cover these files.

Converted by okayuji. Qwen and Alibaba Cloud are named to identify the original model, and Apple, Mac, macOS and Core AI to identify the platform; this independent conversion is not affiliated with or endorsed by them.
