---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
tags:
- qwen
- qwen3.8
- gguf
- llama.cpp
- function-calling
- tool-calling
- vision
- mtp
model-index:
- name: Qwen3.8-27B-Artemis-GGUF
  results:
  - task:
      type: text-generation
      name: Task results
    dataset:
      type: gorilla-llm/Berkeley-Function-Calling-Leaderboard
      name: BFCL
    metrics:
    - type: tool_calls_right
      value: 87
      name: Tool calls (of 111)
    source:
      name: TurinTech
      url: https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF#results
  - task:
      type: text-generation
      name: Task results
    dataset:
      type: openai/gsm8k
      name: GSM8K test
    metrics:
    - type: gsm8k_right
      value: 180
      name: Correct (of 200)
    source:
      name: TurinTech
      url: https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF#results
  - task:
      type: text-generation
      name: Closeness to the original
    dataset:
      type: mixed
      name: Mixed texts
    metrics:
    - type: same_top_p
      value: 68.6
      name: Same next token (%)
    source:
      name: TurinTech
      url: https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF#results
---

<!-- opening:start -->
<p><a href="https://turintech.ai"><img src="https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/assets/turintech-logo.png" width="140" alt="TurinTech"></a><br><em>TurinTech models, built with Artemis</em></p>

Qwen3.8 27B for local tool calling and agents. Optional companion files add image input (a vision projector) and faster generation (an MTP file).

- 7.83 GB download for the default file, `Qwen3.8-27B-Artemis-IQ2_XXS.gguf`.
- GPU memory 9.0 GiB for text on an NVIDIA A100 80 GB, and 11.1 GiB on an RTX 5090 with the projector and the MTP file loaded.
- 87 of 111 parallel tool-calling tasks right, against 63 for the closest public file, AtomicChat's AD-IQ1_M (8.21 GB). These tasks also helped choose the file.
<!-- opening:end -->

## Run it

Needs llama.cpp b11617 or newer for text; the projector and the MTP file were tested on b11889.
`-hf` also downloads the vision projector and loads it; add `--no-mmproj` for text only, which saves about 1.3 GiB of
GPU memory.

```bash
llama-server -hf turintech/Qwen3.8-27B-Artemis-GGUF:IQ2_XXS --port 8080 -c 8192 --jinja
```

<details>
<summary>Request example, images, MTP, Ollama and LM Studio</summary>

The tool-calling results use greedy decoding with thinking off. Set `chat_template_kwargs.enable_thinking` in each
request; the chat template turns thinking on by default.

```bash
hf download turintech/Qwen3.8-27B-Artemis-GGUF Qwen3.8-27B-Artemis-IQ2_XXS.gguf --local-dir .
llama-server -m Qwen3.8-27B-Artemis-IQ2_XXS.gguf --port 8080 -c 8192 --jinja
```

`-c 8192` sets an 8,192-token context and `--jinja` uses the chat template stored in the file. Leave the server
running. In another terminal, wait for `curl -s http://localhost:8080/health` to return `{"status":"ok"}`, then send a
tool request with thinking off (it uses the server's default sampling, not the greedy settings of the evaluation):

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the weather in Paris right now, in celsius?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}],
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

In our test run (this file, NVIDIA A100, llama.cpp b11617 and b11871) the reply's `tool_calls` held `get_weather`
with `{"city":"Paris","unit":"celsius"}`. Asked for the weather in Paris and Tokyo, it returned both calls in one
reply. With thinking on, the server returns the thinking in `message.reasoning_content` and the answer in
`message.content`; allow enough output tokens.

### Images

The vision projector is a separate file, converted from the original weights. `mmproj-F16.gguf` is the same
projector in F16.

```bash
hf download turintech/Qwen3.8-27B-Artemis-GGUF mmproj-BF16.gguf --local-dir .
llama-server -m Qwen3.8-27B-Artemis-IQ2_XXS.gguf --mmproj mmproj-BF16.gguf --port 8080 -c 8192 --jinja
```

Send the image as a data URL in an `image_url` part of the user message. For a short answer, say so in the prompt
and allow enough output tokens: this file often describes a diagram before it answers (see [Image questions](#image-questions)).

### Faster generation with the MTP file

`MTP/mtp-Qwen3.8-27B-Artemis-Q4_0.gguf` holds the original's multi-token prediction layer, which drafts the next
tokens for the main file to check. It does not run on its own. llama.cpp's `-hf` does not fetch it from the `MTP/`
folder, so name it:

```bash
hf download turintech/Qwen3.8-27B-Artemis-GGUF MTP/mtp-Qwen3.8-27B-Artemis-Q4_0.gguf --local-dir .
llama-server -m Qwen3.8-27B-Artemis-IQ2_XXS.gguf -md MTP/mtp-Qwen3.8-27B-Artemis-Q4_0.gguf --spec-type draft-mtp \
  --port 8080 -c 8192 --jinja
```

On an RTX 5090 it raised generation from 112.6 to 189.6 tokens per second on our 30 prompts (see
[MTP](#mtp)). Greedy replies are not always word-for-word the same with MTP on: 20 of 30 matched, and all 10
tool-call replies did. `--mmproj` and `-md` can be combined.

### Ollama

```bash
ollama run turintech/qwen3.8-27b-artemis:iq2_xxs
```

`latest` is the same package. It holds the main file only: its model layer is this repository's file
(7,830,620,736 bytes, the SHA-256 below). The package uses Ollama's `qwen3.5` renderer and tool parser, an
8,192-token context, and temperature 1, top_k 20, top_p 0.95 and presence_penalty 1.5. `ollama show` lists the
quantization as Q4_K_M because of the header field explained under File name.

For tool calls, use Ollama's native chat API with thinking off:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "turintech/qwen3.8-27b-artemis:iq2_xxs",
  "stream": false,
  "think": false,
  "messages": [{"role": "user", "content": "What is the weather in Paris right now, in celsius?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]
}'
```

With Ollama 0.40.11 the reply's `message.tool_calls` held `get_weather` with Paris and celsius, from a local import of
this file and from the published package. Through Ollama's OpenAI-compatible endpoint, one of our tool requests
returned only thinking and no tool call, so we have not tested that path enough to recommend it for tools.

For images in Ollama, build a local package with the projector as a second `FROM` line, next to the downloaded files:

```
FROM ./Qwen3.8-27B-Artemis-IQ2_XXS.gguf
FROM ./mmproj-F16.gguf
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER temperature 1
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER presence_penalty 1.5
PARAMETER num_ctx 8192
```

With Ollama 0.40.11, `ollama show` listed the vision capability, images sent to the native `/api/chat` got answers
(7 of the first 10 ChartQA questions right, against 8 with llama.cpp), and the tool request above still returned `get_weather` with Paris and
celsius. Ollama does not use the MTP file.

### LM Studio

The same main file loaded and answered in LM Studio's headless `llmster` 0.0.25 with its CUDA llama.cpp engine 2.41.0.
That was a short load-and-answer check; the card's numbers come from llama.cpp. The projector and the MTP file were
not tested in LM Studio.

Questions, problems or a model to consider for a smaller version? Reply in our [feedback thread](https://huggingface.co/spaces/turintech/README/discussions/1).

</details>

<!-- choose:start -->
## Choose this model

- Best for tool calling and agents on a GPU with 10 GB or more (about 12 GiB with the projector and the MTP file), with charts as an occasional image input.
- Consider another file when maths, closeness to the original or diagram questions matter more: Unsloth's UD-IQ2_S (8.37 GB) is closer to the original on every kind of text, and Unsloth's UD-IQ2_XXS got 115 of 150 AI2D diagram questions against our 38.
<!-- choose:end -->

## Files

<!-- files:start -->
Every file we publish for this model. Sizes are download sizes, not memory.

| File | Size | Use it for | Ollama tag |
| --- | --- | --- | --- |
| [`Qwen3.8-27B-Artemis-IQ2_XXS.gguf`](https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/Qwen3.8-27B-Artemis-IQ2_XXS.gguf) | 7.83 GB | Default | `iq2_xxs` (latest) |
| [`mmproj-BF16.gguf`](https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/mmproj-BF16.gguf) | 931 MB | Vision companion (mmproj): pass it with `--mmproj` for image inputs | n/a |
| [`mmproj-F16.gguf`](https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/mmproj-F16.gguf) | 928 MB | Vision companion (mmproj): pass it with `--mmproj` for image inputs | n/a |
| [`MTP/mtp-Qwen3.8-27B-Artemis-Q4_0.gguf`](https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/MTP/mtp-Qwen3.8-27B-Artemis-Q4_0.gguf) | 1.34 GB | MTP companion: the multi-token prediction layer, loaded next to the main file | n/a |

- With llama.cpp or a Hugging Face pull, add the file's quant label as the tag: `-hf turintech/Qwen3.8-27B-Artemis-GGUF:IQ2_XXS`.
- The Ollama tag runs our package, which holds the same file: `ollama run turintech/qwen3.8-27b-artemis:<tag>`.
- LM Studio has every file in its Hub entry [`turintech/qwen3.8-27b-artemis`](https://lmstudio.ai/turintech/qwen3.8-27b-artemis).
<!-- files:end -->

<details>
<summary>How the files compare</summary>

| File | Size | Same next token | Tool calls right (of 111) | Choose for |
| --- | --- | --- | --- | --- |
| [`Qwen3.8-27B-Artemis-IQ2_XXS.gguf`](https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/Qwen3.8-27B-Artemis-IQ2_XXS.gguf) | **7.83 GB** | **68.6%** | **87** | Tool calling and agents on a GPU with 10 GB or more. |

GB means decimal download size, not RAM or VRAM. "Same next token" is how often its most likely next token matches
the BF16 original's over 250 mixed texts. The label is a size class; tensor types and the checksums are below. The
companions are optional. A projector adds image input; the MTP file speeds up generation, though greedy replies
can change wording with it (see [MTP](#mtp)).

</details>

<!-- results:start -->
## Results

Our file against AtomicChat's AD-IQ1_M (closest public file by size), Unsloth's smaller UD-IQ2_XXS and the original. The tool and maths tasks also helped choose this file; the image questions were fixed in advance.

| File | Size | Tool calls (of 111) | GSM8K (of 200) | ChartQA (of 150) | AI2D (of 150) |
| --- | --- | --- | --- | --- | --- |
| **Ours, IQ2_XXS** | 7.83 GB | 87 | 180 | 118 | 38 |
| AtomicChat AD-IQ1_M | 8.21 GB | 63 | 186 | 26 | 37 |
| Unsloth UD-IQ2_XXS | 7.27 GB | 75 | 180 | 106 | 115 |
| BF16 original | 54.66 GB | 93 | 188 | 130 | 139 |

Among compressed files, ours leads on tool calls and charts, is level or behind on maths, and trails Unsloth's on diagrams, which it often describes instead of answering. [How it was tested](#how-it-was-tested).
<!-- results:end -->

<details>
<summary>How it was tested</summary>

## How it was tested

### Text

The reference is the BF16 GGUF conversion of Qwen/Qwen3.8-27B (revision `1d4bf0f`, 54.66 GB with the MTP layer,
which llama.cpp does not use for ordinary generation). Every text run used llama.cpp builds of commit `4c3c581` with
CUDA on NVIDIA RTX 5090 GPUs, and every file got the same requests and the same prompt tokens (checked item by item).

- Tool calling: 111 BFCL Python tasks (11 parallel, 100 parallel-multiple) through llama-server's OpenAI-style
  `tools` API, scored with BFCL's own AST checker (`bfcl_eval` 2026.3.23). Greedy decoding, thinking off, up to
  1,024 output tokens. A reply with no tool call counts as wrong.
- Maths: 200 GSM8K test problems, greedy, thinking off, up to 768 output tokens. A reply cut off at the limit counts
  as wrong (14 of ours, 9 of AtomicChat's, 8 of the original's; none of them had the right answer).
- Closeness to the original: 250 texts, 50 each of chat (UltraChat test), coding, long documents (2,048 tokens),
  multilingual (Aya test) and tool-call conversations (BFCL, not the scored tasks). For each text we compare the
  next-token probabilities with the original's: "same next token" is how often the most likely token matches, and KL
  is llama.cpp's approximate KL divergence (lower is closer). Per-kind and "all 250" figures are averaged over
  texts, each text weighted equally.

| File | Size | Tool calls right (of 111) | GSM8K right (of 200) |
| --- | --- | --- | --- |
| BF16 original | 54.66 GB | **93** | **188** |
| Artemis IQ2_XXS | 7.83 GB | 87 | 180 |
| AtomicChat AD-IQ1_M | 8.21 GB (8.50 GB download with its MTP layer) | 63 | 186 |
| Unsloth UD-IQ2_XXS | **7.27 GB** | 75 | 180 |
| Unsloth UD-IQ2_S | 8.37 GB | 81 | 183 |

Bold marks the best displayed value for each metric among the rows shown, ties included; it is not a significance
claim. The quantized files' sizes are the text model without the MTP layer, so they compare like for like; the
original's includes it.

| Comparison (exact McNemar, two-sided, uncorrected) | Tool calls (111) | GSM8K (200) |
| --- | --- | --- |
| Artemis IQ2_XXS vs AtomicChat AD-IQ1_M | 34 / 10, p=0.0004 | 5 / 11, p=0.21 |
| Artemis IQ2_XXS vs Unsloth UD-IQ2_XXS | 17 / 5, p=0.017 | 9 / 9, p=1.00 |
| Artemis IQ2_XXS vs Unsloth UD-IQ2_S | 13 / 7, p=0.26 | 7 / 10, p=0.63 |
| Artemis IQ2_XXS vs BF16 original | 4 / 10, p=0.18 | 4 / 12, p=0.077 |

Each cell gives the items only our file got right, the items only the other file got right, and p. Against
AtomicChat's file the maths difference is 3 points lower, with a 95% interval (paired bootstrap) of 7 points lower to
1 point higher. These samples are small; a test that does not separate two files does not show they are equivalent.

| File | Same next token, chat | Same next token, coding | Same next token, long documents | Same next token, multilingual | Same next token, tool text | Same next token, all 250 |
| --- | --- | --- | --- | --- | --- | --- |
| Artemis IQ2_XXS | 72.1% | 67.6% | 67.6% | 69.1% | 66.5% | 68.6% |
| AtomicChat AD-IQ1_M | 70.5% | 61.6% | 65.8% | 68.9% | 68.7% | 67.1% |
| Unsloth UD-IQ2_XXS | 75.2% | 72.0% | 79.8% | 68.4% | 71.3% | 73.4% |
| Unsloth UD-IQ2_S | **77.4%** | **77.2%** | **94.7%** | **75.0%** | **73.3%** | **79.5%** |

| File | KL, chat | KL, coding | KL, long documents | KL, multilingual | KL, tool text | KL, all 250 |
| --- | --- | --- | --- | --- | --- | --- |
| Artemis IQ2_XXS | .928 | 1.514 | 1.095 | .694 | 2.723 | 1.391 |
| AtomicChat AD-IQ1_M | .703 | 2.045 | 1.365 | .745 | 2.396 | 1.451 |
| Unsloth UD-IQ2_XXS | .752 | 1.350 | .645 | .844 | 2.517 | 1.222 |
| Unsloth UD-IQ2_S | **.576** | **1.071** | **.210** | **.483** | **2.126** | **.893** |

Against AtomicChat's file, paired over the same texts (ours minus theirs, 95% paired-bootstrap interval): ours is
closer on coding (same next token +6.0 points [+4.1, +8.0], KL −.53 [−.64, −.42]) and on long documents (KL −.27
[−.43, −.12]; same next token +1.8 [−1.5, +5.0]); on multilingual text both intervals include zero (+0.2 [−1.2, +1.6], KL −.05
[−.18, +.04]); ours is further on tool text (−2.2 [−3.5, −0.9], KL +.33 [+.20, +.45]) and on chat KL (+.22 [+.04,
+.43]; same next token +1.6 [−1.1, +4.4]). Unsloth's `UD-IQ2_S` is closer than ours on every kind of text, and its
`UD-IQ2_XXS` on every kind except multilingual (KL −.15 in our favour; the same-next-token interval includes zero;
on chat only the same-next-token interval excludes zero). Long documents show
the largest gap: 67.6% same next token for ours against 94.7% for `UD-IQ2_S`. Both Unsloth files had higher next-token
agreement on the 2,048-token document texts; document-answer accuracy and longer contexts were not tested.

What to keep in mind about these numbers:

- They are not an independent test. The tool and maths tasks are the final set of a 20 October comparison of 15
  candidate sizes against Unsloth's files, and the 250 texts are the panel frozen on 21 October for the follow-up
  comparisons. We reused both here and picked this file for release after seeing the results.
- Our task scores were measured on 20 October on the build of this file that still had the MTP layer inside (8.10 GB,
  not used for generation). This file is that build with the MTP layer removed: its 851 tensors are byte-identical,
  and 12 fixed packaging prompts, each run twice, gave 24 matching responses. The original's and Unsloth's task scores are from the same
  20 October run. AtomicChat's file was scored on 25 October with the same llama-server binary, scorer, chat template
  and requests, on one GPU of the same machine instead of a two-GPU pair; our file was not re-scored in that setup.
- The closeness numbers were measured on 25 October, all four files against the same reference output for each
  text, so the comparison between files is paired. Running on one 32 GB GPU instead of two forced five changes from
  the 21 October protocol: (1) the 54.66 GB reference ran with 28 layers on the GPU instead of all; (2) the server
  that renders the chat template ran on the CPU, and its output was checked against the frozen text and token hashes;
  (3) every model file was hashed in full before and after the run instead of being locked; (4) the reference was
  checked against itself once per text, as in the earlier runs (every text passed, with at least 99.2% same next
  token and KL at most .000004); (5) for the two texts whose tokens end exactly at the context limit, the unscored
  padding after the text was generated without its leading newline, and the scored tokens were checked against the
  frozen panel.
- Public files: AtomicChat/Qwen3.8-27B-GGUF revision `ca10ebc` (`Qwen3.8-27B-AD-IQ1_M.gguf`, run as published with
  its MTP layer, which llama.cpp skips for ordinary generation); unsloth/Qwen3.8-27B-GGUF revision `4ca7207`
  (`Qwen3.8-27B-UD-IQ2_XXS.gguf`, `Qwen3.8-27B-UD-IQ2_S.gguf`, main files only).

### Image questions

300 public questions with one image each: 150 AI2D test diagrams (four options, answer with the letter) and 150
ChartQA test charts (75 human-written, 75 generated questions, answer with a word or number). The items, prompts
(lmms-eval's wording) and scoring were fixed and hashed before the first image run, and every file ran them once.
llama.cpp b11889 (`d854d98`) with CUDA on one NVIDIA RTX 5090 (the original on two), greedy, thinking off, up to 32
output tokens. AI2D counts the first standalone letter A to D in the reply; ChartQA uses relaxed accuracy (a number
within 5% of the answer, otherwise an exact match ignoring case). A reply with no answer counts as wrong.

| File | Size | Projector | ChartQA right (of 150) | AI2D right (of 150) | Both (of 300) |
| --- | --- | --- | --- | --- | --- |
| BF16 original | 54.66 GB | ours, F16 | **130** | **139** | **269** |
| Artemis IQ2_XXS | 7.83 GB | ours, F16 | 118 | 38 | 156 |
| Artemis IQ2_XXS with mmproj-BF16 | 7.83 GB | ours, BF16 | 119 | 42 | 161 |
| AtomicChat AD-IQ1_M | 8.21 GB | AtomicChat's, F16 | 26 | 37 | 63 |
| Unsloth UD-IQ2_XXS | **7.27 GB** | Unsloth's, F16 | 106 | 115 | 221 |

| Comparison (exact McNemar, two-sided, uncorrected) | ChartQA (150) | AI2D (150) |
| --- | --- | --- |
| Artemis IQ2_XXS vs Unsloth UD-IQ2_XXS | 18 / 6, p=0.023 | 7 / 84, p<0.0001 |
| Artemis IQ2_XXS vs AtomicChat AD-IQ1_M | 92 / 0, p<0.0001 | 24 / 23, p=1.00 |
| Artemis IQ2_XXS vs BF16 original | 4 / 16, p=0.012 | 3 / 104, p<0.0001 |

Cells as in the text table above. On charts our file is ahead of Unsloth's (8 points, 95% interval 2 to 15). On
diagrams it is far behind, mostly because it ignores the instruction to answer with the letter: 112 of its 150 AI2D
replies were still describing the diagram at the 32-token limit, against 8 for Unsloth's file and for the original.
In a check we added after seeing this, outside the fixed protocol, the same 150 AI2D questions with up to 512 output
tokens and the last letter in the reply counted gave 115 for ours, 120 for Unsloth's and 145 for the original.
AtomicChat's file ran past the limit on most questions of both kinds (110 AI2D, 85 ChartQA).

The two projectors gave nearly the same results with our file (3 / 8 items differ, p=0.23 over all 300). Our F16
projector's 334 tensors are byte-identical to Unsloth's and AtomicChat's F16 projectors, and our BF16 projector's to
Unsloth's BF16 one, so the gaps above come from the main files.

### MTP

30 fixed prompts (10 chat, 10 code, 10 tool calls), each run twice after two warm-up requests, on one NVIDIA RTX 5090
with llama.cpp b11889: greedy, thinking off, up to 256 output tokens, one request at a time, `-ngl 99 -ngld 99 -np 1
--cache-ram 0` and llama.cpp's default of 3 draft tokens. Speed is generation tokens per second from llama-server's
own timings (prompt processing excluded), over both runs; acceptance is the share of drafted tokens the main file kept.

| Main file | MTP file | Chat (tok/s) | Code (tok/s) | Tool calls (tok/s) | All (tok/s) | Accepted | Speed-up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Artemis IQ2_XXS | none | 112.7 | 112.2 | 113.2 | 112.6 | N/A | 1.00 |
| Artemis IQ2_XXS | ours, 1.34 GB | 174.6 | 192.4 | 231.9 | 189.6 | 70.4% | **1.68** |
| Artemis IQ2_XXS | Unsloth's, 1.37 GB | **175.2** | 191.0 | 228.9 | 189.0 | **70.7%** | **1.68** |
| Unsloth UD-IQ2_XXS | none | 117.9 | 117.8 | 119.6 | 118.1 | N/A | 1.00 |
| Unsloth UD-IQ2_XXS | Unsloth's, 1.37 GB | 167.2 | **203.4** | **237.5** | **189.9** | 69.3% | 1.61 |

Acceptance was 61.7% on chat, 72.1% on code and 94.6% on tool calls for ours. Greedy output with MTP on was the same
as with it off for 20 of our 30 prompts, including all 10 tool-call replies; the other 10 change wording partway
through. The same 10 prompts changed with Unsloth's MTP file and with a larger Q4_0 file we tried, which points to
llama.cpp's batched checking of drafts rather than the draft itself; we did not test the cause. Unsloth's main file
behaved the same way (17 of 30 identical).

Our MTP file is the original's layer quantized with stock
`llama-quantize`: Q4_0 for the layer, Q3_K for its embedding and output tables, no importance matrix. A plain Q4_0
version (2.01 GB, Q6_K output table) accepted slightly more drafts (71.3%) but was slower (181.3 tok/s), so we did
not keep it. Before quantization, the BF16 export's 18 tensors (the MTP layer with its embedding, output and norm
tables) were byte-identical to the same tensors of the BF16 reference.

### Limitations

Tested with thinking off; thinking mode was not evaluated. The tool tasks cover parallel calls in Python only.
Nothing here measures long-context accuracy, code correctness, video input or speed on other GPUs. The image check is
300 questions of two kinds; documents, screenshots and OCR were not tested. The calibration data and the importance
matrix are not distributed.

</details>

<!-- closeness:start -->
## Closeness to the original

The closeness numbers from the tables above, in one fixed layout with the protocol behind each table.

<details>
<summary>Protocols and standard tables</summary>

- Mean KLD: the mean KL divergence between the reference's and the file's next-token distributions, computed and averaged as each protocol's Tool line says. Lower is closer.
- 99th-percentile KLD: the 99th percentile of the per-token values. Lower is closer.
- Same top p: llama.cpp's name for top-1 agreement, the share of scored tokens where the file and the reference pick the same most likely token. Higher is closer.
- PPL and PPL ratio: the file's perplexity on the text, and that perplexity divided by the reference's (1 means the same). They measure prediction of the text, not closeness.

n/a means the card's tables or our run records do not have that value.

### Protocol `rtx5090-2026-10-25`

One NVIDIA RTX 5090, 25 October 2026.

- Reference: BF16 GGUF conversion of Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, with the MTP layer (54,657,734,112 bytes, SHA-256 fe268c65113629812ac2a2231aa7cbebf1b26acaf52442eb9c97515143577f2b); its saved log-probabilities for each text are the reference for every file
- Text `chat`: Chat; dataset: HuggingFaceH4/ultrachat_200k at `8049631c405ae6576f93f445c6b8166f76f5505a`; split: test_sft; chunks: 50 (one per text); context (tokens): 256; scored tokens: 6350
- Text `coding`: Coding; chunks: 50 (one per text); context (tokens): 256; scored tokens: 6350
- Text `long_document`: Long documents; chunks: 50 (one per text); context (tokens): 2048; scored tokens: 51150
- Text `multilingual`: Multilingual; dataset: CohereForAI/aya_dataset at `f9ea04583f02a8f86404ff6c58bf75fe637df8a2`; split: test; chunks: 50 (one per text); context (tokens): 256; scored tokens: 6350
- Text `tooltext`: Tool text (BFCL conversations, not the scored tasks); chunks: 50 (one per text); context (tokens): 256; scored tokens: 6350
- Text `all`: All 250 texts, each text weighted equally; dataset: the five sets above; chunks: 250 (one per text); context (tokens): 256 (2048 for long documents); scored tokens: 76550
- Tool: `llama-perplexity --kl-divergence` against the reference's saved log-probabilities, one window per text, run by our harness. Mean KLD and Same top p are computed for each text and then averaged over the texts with equal weight per text, so in the All row a 2,048-token long document counts the same as a 256-token chat text. llama.cpp scores the second half of each window: 127 tokens per 256-token text and 1,023 per 2,048-token text
- llama.cpp build: commit 4c3c5810f87efe177949f1b85ae509d38a2a311f; scoring binary SHA-256 bebf1cb1f97c1747ef41b05be47db18304ab595be62fc4ad3b0e8bd9cf5597fd
- Command: `llama-perplexity -m <file.gguf> --kl-divergence-base <reference.kld> --kl-divergence -c <256 or 2048> -b <same> -ub <same> -ngl 99 -t 8` for each file and text; the reference logs were written with `llama-perplexity -m <BF16.gguf> -f <text> --kl-divergence-base <reference.kld> -c <256 or 2048> --chunks 1 -b <same> -ub <same> -ngl 28 -t 8`
- Hardware: one NVIDIA RTX 5090 (32 GB); the BF16 reference ran with 28 layers on the GPU
- Settings: texts rendered with Qwen's chat template and checked against frozen text and token hashes; the reference was checked against itself on every text (at least 99.2% same next token, KL at most 0.000004); flash attention and KV cache type not set on the command line
- KL unit: nats (natural log)
- Measured by: TurinTech downloaded the AtomicChat and Unsloth files and measured all four files in one run, against the same saved reference output for each text, with the same binary, settings and GPU. AtomicChat's file was run as published, with its MTP layer, which llama.cpp skips in ordinary generation
- Selection: the 250 texts are the panel frozen on 21 October for follow-up comparisons, and this file was picked for release after seeing these results, so they are not an independent test
- Source on this card: the table columns "KL, all 250", "Same next token, all 250", "KL, chat", "Same next token, chat", "KL, coding", "Same next token, coding", "KL, long documents", "Same next token, long documents", "KL, multilingual", "Same next token, multilingual", "KL, tool text", "Same next token, tool text"
- Comparison files: AtomicChat: `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188`, `Qwen3.8-27B-AD-IQ1_M.gguf`, SHA-256 e2a85d6cddd908cc66320148291d8fbc8ef2e41d5a4c4a2a9095a9002f141b59; Unsloth: `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e`, `Qwen3.8-27B-UD-IQ2_XXS.gguf`, SHA-256 e792d8fb3142fe6d9171876d6da0f71f05a71028718debc72dbec93ff645e67d; Unsloth: `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e`, `Qwen3.8-27B-UD-IQ2_S.gguf`, SHA-256 7897d2c5a5cee46aef50895141b2c8a0803c1185f3d03c4fda4cd137a7ad77fe

Protocol `rtx5090-2026-10-25`, text `chat` (Chat):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 GGUF of Qwen/Qwen3.8-27B with its MTP layer | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 54.66 | 54657734112 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `a710497a6269` | TurinTech | 7.83 | 7830620736 | 0.928 | 72.1 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 0.703 | 70.5 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 0.752 | 75.2 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 0.576 | 77.4 |

Protocol `rtx5090-2026-10-25`, text `coding` (Coding):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 GGUF of Qwen/Qwen3.8-27B with its MTP layer | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 54.66 | 54657734112 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `a710497a6269` | TurinTech | 7.83 | 7830620736 | 1.514 | 67.6 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 2.045 | 61.6 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 1.350 | 72.0 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 1.071 | 77.2 |

Protocol `rtx5090-2026-10-25`, text `long_document` (Long documents):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 GGUF of Qwen/Qwen3.8-27B with its MTP layer | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 54.66 | 54657734112 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `a710497a6269` | TurinTech | 7.83 | 7830620736 | 1.095 | 67.6 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 1.365 | 65.8 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 0.645 | 79.8 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 0.210 | 94.7 |

Protocol `rtx5090-2026-10-25`, text `multilingual` (Multilingual):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 GGUF of Qwen/Qwen3.8-27B with its MTP layer | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 54.66 | 54657734112 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `a710497a6269` | TurinTech | 7.83 | 7830620736 | 0.694 | 69.1 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 0.745 | 68.9 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 0.844 | 68.4 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 0.483 | 75.0 |

Protocol `rtx5090-2026-10-25`, text `tooltext` (Tool text (BFCL conversations, not the scored tasks)):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 GGUF of Qwen/Qwen3.8-27B with its MTP layer | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 54.66 | 54657734112 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `a710497a6269` | TurinTech | 7.83 | 7830620736 | 2.723 | 66.5 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 2.396 | 68.7 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 2.517 | 71.3 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 2.126 | 73.3 |

Protocol `rtx5090-2026-10-25`, text `all` (All 250 texts, each text weighted equally):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 GGUF of Qwen/Qwen3.8-27B with its MTP layer | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 54.66 | 54657734112 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `a710497a6269` | TurinTech | 7.83 | 7830620736 | 1.391 | 68.6 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 1.451 | 67.1 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 1.222 | 73.4 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 0.893 | 79.5 |

</details>
<!-- closeness:end -->

## Memory

Peak memory of `llama-server` with the text-only command (`--port 8080 -c 8192 --jinja`, all layers on the GPU)
answering five short prompts on an NVIDIA A100 80 GB: 9.0 GiB of GPU memory and 7.5 GiB of process memory (RSS) with
llama.cpp b11617, and 8.8 GiB and 7.7 GiB with b11871. RSS counts the parts of the memory-mapped model file that were
read. The command keeps llama-server's default 8 GiB prompt cache, so system memory can grow in longer sessions; add
`--cache-ram 0` to limit it. A longer context needs more memory.

GPU memory in use with the companions, on an NVIDIA RTX 5090 with llama.cpp b11889 (`-ngl 99 -c 8192 -np 1
--cache-ram 0`): 8.5 GiB for the main file alone, 9.8 GiB with `mmproj-F16.gguf` after the 300 image questions,
9.8 GiB with the MTP file, and 11.1 GiB with both. With the commands above as written, which leave GPU layers and
request slots at llama-server's defaults, the MTP command used 11.6 GiB and the projector command 10.2 GiB.

<details>
<summary>File name, size and SHA-256 checksum</summary>

## File name

The file name ends in a standard llama.cpp quant label, so llama.cpp, Ollama and LM Studio can pick it by tag. The
label is a size class, not the tensor recipe: no tensor type accounts for 90% of the weight bytes, and the file
averages 2.33 bits per weight. IQ2_XXS is the smallest label among bartowski's stock `llama-quantize` files of the
same model (8.64 GB of text model, about 2.57 bits per weight), and we use those files as the naming ladder. Publishers use the same label for different sizes: Unsloth's `UD-IQ2_XXS` is 7.27 GB.

| File | Bits per weight | Main tensor types (share of bytes) |
| --- | --- | --- |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | 2.33 | IQ2_XS 54%, IQ2_XXS 20%, IQ2_S 8%, IQ3_S 7%, Q2_K 6% |

The header's `general.file_type` reads 15 (Q4_K_M), which is what `ollama show` and some other tools display, and
`general.name` reads "Parent". Neither changes how the file runs. The companions follow Unsloth's layout: the
projectors at the top level and the MTP file in `MTP/`. The MTP file's label names the type of its layer weights;
its embedding and output tables are Q3_K, as in Unsloth's file.

## File checksum

| File | Bytes | sha256 |
|---|---|---|
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | 7,830,620,736 | `6b41a0716f4222f364abbfa88e718db2501bdd173de186f0ba45dd490587602c` |
| `mmproj-BF16.gguf` | 931,146,176 | `b56aca76c51d2ba0e7b5c34d4fab84d50ad71dc7b12f17c0faa9f57e816a5df5` |
| `mmproj-F16.gguf` | 927,607,232 | `7a47ce71686d9a26eb90a9aaa73da3e9a9fc6f3648d9646b1b06959e7e83bb6b` |
| `MTP/mtp-Qwen3.8-27B-Artemis-Q4_0.gguf` | 1,342,556,960 | `5331d11b17004975ef6844ef04f595d0deb439ed8e22f20496f3c3a7c31f3246` |

</details>

## Artemis

Teams use Artemis to test and improve code against their own objectives. [See Artemis](https://turintech.ai).

## Licence

Apache-2.0, the licence of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE); see
`LICENSE` and `NOTICE`.

This is an independent TurinTech conversion, not affiliated with or endorsed by Alibaba Cloud or the Qwen team. The
files are standard GGUF and run on stock llama.cpp. Changes from the original: the weights are converted to GGUF, and
the text weights and the MTP layer are quantized for llama.cpp. See `NOTICE`.

When serving these files to others, for example through an API, an inference provider or OpenRouter, we'd
appreciate a mention of TurinTech (turintech.ai) as the source of the quantization.
