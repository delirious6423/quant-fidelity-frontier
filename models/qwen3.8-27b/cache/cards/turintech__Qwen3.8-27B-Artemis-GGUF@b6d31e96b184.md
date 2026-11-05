---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
tags:
- turintech
- artemis
- qwen
- qwen3.8
- gguf
- llama.cpp
- quantized
- function-calling
- tool-calling
- vision
- mtp
model-index:
- name: Qwen3.8-27B-Artemis-GGUF
  results:
  - task:
      type: text-generation
      name: Text generation
    dataset:
      type: gorilla-llm/Berkeley-Function-Calling-Leaderboard
      name: BFCL live
    metrics:
    - type: tool_calls_right
      value: 222
      name: Tool calls (of 258)
    source:
      name: TurinTech
      url: https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF#results
  - task:
      type: text-generation
      name: Text generation
    dataset:
      type: openai/gsm8k
      name: GSM8K, last 250
    metrics:
    - type: gsm8k_right
      value: 234
      name: Correct (of 250)
    source:
      name: TurinTech
      url: https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF#results
  - task:
      type: text-generation
      name: Text generation
    dataset:
      type: TIGER-Lab/MMLU-Pro
      name: MMLU-Pro sample
    metrics:
    - type: mmlu_pro_right
      value: 128
      name: Correct (of 280)
    source:
      name: TurinTech
      url: https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF#results
  - task:
      type: text-generation
      name: Text generation
    dataset:
      type: google/IFEval
      name: IFEval
    metrics:
    - type: ifeval_followed
      value: 399
      name: Followed (of 539)
    source:
      name: TurinTech
      url: https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF#results
---

<!-- v2-header:start -->
<div align="center">

<a href="https://www.turintech.ai"><img src="https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/assets/turintech-logo-v2.png" width="360" alt="TurinTech"></a>

<em>TurinTech models, built with Artemis</em>

### Qwen3.8 27B for tool calling and agents on a 10 GB GPU, compressed with Artemis

<div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 6px;">
<a href="https://www.turintech.ai"><img src="https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/assets/buttons/artemis.png" width="84" alt="Artemis"></a>
<a href="https://huggingface.co/turintech"><img src="https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/assets/buttons/other-models.png" width="122" alt="Other TurinTech models on Hugging Face"></a>
<a href="https://ollama.com/turintech/qwen3.8-27b-artemis"><img src="https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/assets/buttons/ollama.png" width="78" alt="Run it with Ollama"></a>
<a href="https://lmstudio.ai/turintech/qwen3.8-27b-artemis"><img src="https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/assets/buttons/lm-studio.png" width="99" alt="Run it in LM Studio"></a>
<a href="https://www.linkedin.com/company/turintechai"><img src="https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/assets/buttons/turintech.png" width="98" alt="TurinTech on LinkedIn"></a>
</div>

| Download | GPU memory | Tool calls right, BFCL live |
| :---: | :---: | :---: |
| **7.83 GB** | **8.5 GiB** (original 50.2 GiB) | **222 of 258** (original 223) |

</div>
<!-- v2-header:end -->

## Run it

```bash
llama-server -hf turintech/Qwen3.8-27B-Artemis-GGUF:IQ2_XXS --no-mmproj -np 1 --port 8080 -c 8192 --jinja
```

<!-- v2-apps:start -->
Ollama: `ollama run turintech/qwen3.8-27b-artemis` &nbsp;·&nbsp; LM Studio: [`turintech/qwen3.8-27b-artemis`](https://lmstudio.ai/turintech/qwen3.8-27b-artemis)
<!-- v2-apps:end -->

Needs llama.cpp b11617 or newer for text (the Results table was measured with b11617, the release runs with commit
`4c3c581`); the projector and the MTP file were converted and checked with b11889. Without `--no-mmproj`, `-hf` also
downloads the vision projector and loads it, about 1.3 GiB more GPU memory; `-np 1` keeps one slot. The 8.5 GiB of GPU
memory on this card was measured that way, text only with one slot. The chat template turns thinking on by default; every
result below was measured with it off, so send `"chat_template_kwargs": {"enable_thinking": false}` in the request, or
allow enough output tokens for the thinking, which the server returns in `message.reasoning_content`.

<details>
<summary>Request example, images, the MTP file, Ollama and LM Studio</summary>

To run a downloaded file:

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
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    }, "required": ["city"]}
  }
  }],
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

In our test run (this file, NVIDIA A100, llama.cpp b11617 and b11871) the reply's `tool_calls` held `get_weather`
with `{"city":"Paris","unit":"celsius"}`. Asked for the weather in Paris and Tokyo, it returned both calls in one
reply. With thinking on, the server returns the thinking in `message.reasoning_content` and the answer in
`message.content`; allow enough output tokens.

Images: the vision projector is a separate file, converted from the original weights; `mmproj-F16.gguf` is the same
projector in F16. Send the image as a data URL in an `image_url` part of the user message. For a short answer, say so
in the prompt and allow enough output tokens: this file often describes a diagram before it answers ([results
page](RESULTS.md), Image questions).

```bash
hf download turintech/Qwen3.8-27B-Artemis-GGUF mmproj-BF16.gguf --local-dir .
llama-server -m Qwen3.8-27B-Artemis-IQ2_XXS.gguf --mmproj mmproj-BF16.gguf --port 8080 -c 8192 --jinja
```

Faster generation: `MTP/mtp-Qwen3.8-27B-Artemis-Q4_0.gguf` holds the original's multi-token prediction layer, which
drafts the next tokens for the main file to check. It does not run on its own. llama.cpp's `-hf` does not fetch it
from the `MTP/` folder, so name it:

```bash
hf download turintech/Qwen3.8-27B-Artemis-GGUF MTP/mtp-Qwen3.8-27B-Artemis-Q4_0.gguf --local-dir .
llama-server -m Qwen3.8-27B-Artemis-IQ2_XXS.gguf -md MTP/mtp-Qwen3.8-27B-Artemis-Q4_0.gguf --spec-type draft-mtp \
  --port 8080 -c 8192 --jinja
```

On an RTX 5090 it raised generation from 112.6 to 189.6 tokens per second on 30 fixed prompts ([results
page](RESULTS.md), MTP). Greedy replies are not always word for word the same with MTP on: 20 of 30 matched, and all
10 tool-call replies did. `--mmproj` and `-md` can be combined.

Ollama: the package `turintech/qwen3.8-27b-artemis` (tag `iq2_xxs`, also `latest`) holds the main file only; its
model layer is this repository's file (7,830,620,736 bytes, the sha256 below). The package uses Ollama's `qwen3.5`
renderer and tool parser, an 8,192-token context, and temperature 1, top_k 20, top_p 0.95 and presence_penalty 1.5.
`ollama show` lists the quantization as Q4_K_M because of the header field explained on the results page. For tool
calls, use Ollama's native chat API with thinking off:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "turintech/qwen3.8-27b-artemis:iq2_xxs",
  "stream": false,
  "think": false,
  "messages": [{"role": "user", "content": "What is the weather in Paris right now, in celsius?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    }, "required": ["city"]}
  }
  }]
}'
```

With Ollama 0.40.11 the reply's `message.tool_calls` held `get_weather` with Paris and celsius, from a local import of
this file and from the published package. Through Ollama's OpenAI-compatible endpoint, one of our tool requests
returned only thinking and no tool call, so that path was not tested enough to recommend it for tools.

For images in Ollama 0.40.11 or newer, pull straight from Hugging Face; the pull brings the projector along:

```bash
ollama run hf.co/turintech/Qwen3.8-27B-Artemis-GGUF:IQ2_XXS
```

On 26 October 2026, with that pull, `ollama show` listed vision and the projector, chart questions sent to `/api/chat`
got answers, and the tool request above returned `get_weather` with Paris and celsius. A local package built from the
downloaded main file and `mmproj-F16.gguf` (two `FROM` lines, the renderer, parser and parameters above, `num_ctx
8192`) also listed vision, answered 7 of the first 10 ChartQA questions through the native `/api/chat` (against 8 with
llama.cpp) and returned the same tool call. Ollama does not use the MTP file.

LM Studio: the same main file loaded and answered in LM Studio's headless `llmster` 0.0.25 with its CUDA llama.cpp
engine 2.41.0. That was a short load-and-answer check; the card's numbers come from llama.cpp. The projector and the
MTP file were not tested in LM Studio.

</details>

<!-- v2-results:start -->
## Results

![Tool calls right on 258 BFCL live_simple tasks for our file, the original and seven public files of the same model, with download sizes.](assets/comparison.png)

| File | Size | Tool calls (of 258) | GSM8K (of 250) | MMLU-Pro (of 280) | IFEval (of 539) |
| --- | --- | --- | --- | --- | --- |
| [**TurinTech IQ2_XXS**](https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/Qwen3.8-27B-Artemis-IQ2_XXS.gguf) | 7.83 GB | 222 | 234 | 128 | 399 |
| Unsloth UD-IQ2_XXS | 7.27 GB | 193 | 231 | 92 | 386 |
| Unsloth UD-IQ2_S | 8.37 GB | 212 | 234 | 126 | 415 |
| AtomicChat AD-IQ1_M | 8.50 GB | 204 | **240** | 65 | 393 |
| bartowski IQ2_S | 9.68 GB | 214 | 233 | 162 | 416 |
| BF16 original | 53.81 GB | **223** | **240** | **187** | **446** |

The record of how the file was chosen does not mention the tool-call, MMLU-Pro and IFEval item sets; the 250 GSM8K problems share 47 with the 200 the file was chosen with ([How it was tested](#how-it-was-tested)). Bold marks the best in each column. After correcting for 16 planned comparisons, our file is ahead of Unsloth's UD-IQ2_XXS (7.27 GB) on tool calls (222 against 193) and MMLU-Pro (128 against 92) and of AtomicChat's AD-IQ1_M (8.50 GB) on MMLU-Pro (128 against 65), and behind the original on MMLU-Pro (128 against 187) and IFEval (399 against 446); no other test separated the files. MMLU-Pro allowed 8 output tokens for the answer letter; a reply that began working the problem instead was cut off and counts as wrong, 58 times for ours, 12 for the original, 115 for Unsloth's UD-IQ2_XXS and 179 for AtomicChat's file. [Full results, memory and method](RESULTS.md).
<!-- v2-results:end -->

<!-- v2-files:start -->
## Files

| File | Size | Peak GPU memory | Use it for |
| --- | --- | --- | --- |
| [`Qwen3.8-27B-Artemis-IQ2_XXS.gguf`](https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/Qwen3.8-27B-Artemis-IQ2_XXS.gguf) | 7.83 GB | 8.5 GiB | Default, Ollama tag `iq2_xxs` (latest) |
| [`mmproj-BF16.gguf`](https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/mmproj-BF16.gguf) | 931 MB |  | Image input, with `--mmproj` |
| [`mmproj-F16.gguf`](https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/mmproj-F16.gguf) | 928 MB |  | Image input, with `--mmproj` |
| [`MTP/mtp-Qwen3.8-27B-Artemis-Q4_0.gguf`](https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/MTP/mtp-Qwen3.8-27B-Artemis-Q4_0.gguf) | 1.34 GB |  | Faster generation (MTP companion) |

The vision projector and the MTP file are optional; `-hf` downloads the projector, and the MTP file is named by path. Peak GPU memory measured on NVIDIA RTX 5090, llama.cpp b11617, all layers on the GPU, 8,192-token context, one slot, prompt cache off (--cache-ram 0), first 25 items of each task, without the projector or the MTP file; host RAM includes the parts of the memory-mapped model file that were read. CPU only: AMD Ryzen 9 5900X, 12 threads, llama.cpp b11617 CPU build (g++ 9.4.0), -c 8192 --jinja --cache-ram 0, peak resident memory over three 200-token chat completions, the memory-mapped file included.
<!-- v2-files:end -->

<!-- v2-pick:start -->
## Pick this model

- **Good for** tool calls and agents with thinking off on a GPU with 10 GB, in a 7.83 GB download.
- **Pick another file** when knowledge questions or instruction following matter more: the original got 187 of 280 MMLU-Pro questions and 446 of 539 IFEval prompts against our 128 and 399, and Unsloth's UD-Q4_K_XL (17.56 GB) got 187 and 426. Among files under 10 GB, bartowski's IQ2_S (9.68 GB) got 162 on MMLU-Pro and 416 on IFEval, comparisons outside the planned tests, and Unsloth's UD-IQ2_S (8.37 GB) 415 on IFEval, a gap to ours that the planned test did not separate.
<!-- v2-pick:end -->

## How it was made

Built with [Artemis](https://www.turintech.ai/product/overview), TurinTech's optimisation platform, from the BF16
conversion of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The file is a per-tensor mix of standard
llama.cpp quant types averaging 2.33 bits per weight, so its label is a size class (IQ2_XXS is the smallest label among
bartowski's stock `llama-quantize` files of this model, and those files are the naming ladder; the tensor types are on
the [results page](RESULTS.md)). Stock llama.cpp, Ollama and LM Studio run it. The original's multi-token prediction
layer is quantized separately as the MTP file, and the vision projector is converted from the original weights; the
projector is not changed otherwise.

## How it was tested

Every file in the Results table ran on one NVIDIA RTX 5090 (the BF16 original over two) with llama.cpp b11617, the same chat template, greedy
decoding and thinking off. [The first card](https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/blob/89a6492a2160c56da4ac236b509d0e4f4c9791a0/README.md)
records that this file was chosen on 111 BFCL parallel tool-calling tasks, 200 GSM8K problems and closeness to the
original on 250 texts, all seen before the choice. The Results table uses other item sets: BFCL live_simple, a new
MMLU-Pro sample and IFEval, which that record does not mention, so it neither shows nor rules out a part for them in
the choice; and GSM8K test items 1069 to 1318, 47 of which were among the 200 problems, so that column is not an
independent test. IFEval is scored on 539 of its 541 prompts because two ask for a count of a punctuation mark that
the upstream checker scores at random. The [results page](RESULTS.md) has the method, the full tables, the runs the
card was first published with, the image, MTP and memory checks of the companion files, and the memory of the file on
a CPU alone.

<!-- closeness:start -->
## Closeness to the original

The closeness numbers from the tables on the [results page](RESULTS.md), in one fixed layout with the protocol behind each table.

<details>
<summary>Protocols and standard tables</summary>

- Mean KLD: the mean KL divergence between the reference's and the file's next-token distributions, computed and averaged as each protocol's Tool line says. Lower is closer.
- 99th-percentile KLD: the 99th percentile of the per-token values. Lower is closer.
- Same top p: llama.cpp's name for top-1 agreement, the share of scored tokens where the file and the reference pick the same most likely token. Higher is closer.
- PPL and PPL ratio: the file's perplexity on the text, and that perplexity divided by the reference's (1 means the same). They measure prediction of the text, not closeness.

n/a means the card's tables or our run records do not have that value.

### Protocol `rtx5090-2026-10-27`

One NVIDIA RTX 5090, 27 and 28 October 2026.

- Reference: BF16 GGUF converted with llama.cpp b11617 from Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, text only (--no-mtp; 53,808,282,144 bytes, SHA-256 21bbc32d5d8a4b22f981685138d026951307b09b9715e317470d60f618798c16); its saved log-probabilities for each text are the reference for every file
- Text `wiki`: WikiText-2 test; dataset: Salesforce/wikitext at `b08601e04326c79dfdd32d625aee71d232d685c3`; config: wikitext-2-raw-v1; split: test; chunks: 60; context (tokens): 512; text SHA-256: bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a
- Text `chat`: OpenAssistant validation conversations (English threads as User/Assistant text); dataset: OpenAssistant/oasst1 at `fdf72ae0827c1cda404aff25b6603abec9e3399b`; split: validation; chunks: 60; context (tokens): 512; text SHA-256: 0a3499a7eaead8f4a40655a868e7faee1ee664e258d09a77cd8f3d97216fc8b7
- Tool: `llama-perplexity --kl-divergence` against the reference's saved log-probabilities, chunks of 512 tokens; Mean KLD and Same top p are llama.cpp's own summary lines
- llama.cpp build: b11617 (commit 14310190cd0a0c831fe949ac8cbedd8abdc90695)
- Command: `llama-perplexity -m <file.gguf> -f <text> --kl-divergence-base <reference-text.kld> --kl-divergence -c 512 --chunks 60 -t 16 -ngl 99`; the reference logs were written with the same command without `--kl-divergence`, with the reference over two GPUs
- Hardware: one NVIDIA RTX 5090 (AMD EPYC 7V13 host); the BF16 reference over two RTX 5090s
- Settings: 16 threads, all layers on the GPU, beginning-of-sequence token off or absent in every file as in the original; the KL command sets no flash attention or KV cache flags, so llama-perplexity's defaults applied (the flash attention and F16 KV settings in the results page are for the task runs)
- KL unit: nats (natural log)
- Measured by: TurinTech downloaded the Unsloth, AtomicChat and bartowski files at the revisions below, checked each by SHA-256 and measured every file in this table with the same reference, binary and machine. AtomicChat's file was run as published, with its MTP layer, which llama.cpp skips in ordinary generation
- Selection: these two texts are not mentioned in the record of how this file was chosen, which used 250 other texts; they are not the selection panel
- Source, on the results page: the table columns "KL wiki / chat", "Same next token, wiki / chat"
- Comparison files: Unsloth: `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e`, `Qwen3.8-27B-UD-IQ2_XXS.gguf`, SHA-256 e792d8fb3142fe6d9171876d6da0f71f05a71028718debc72dbec93ff645e67d; Unsloth: `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e`, `Qwen3.8-27B-UD-IQ2_S.gguf`, SHA-256 7897d2c5a5cee46aef50895141b2c8a0803c1185f3d03c4fda4cd137a7ad77fe; AtomicChat: `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188`, `Qwen3.8-27B-AD-IQ1_M.gguf`, SHA-256 e2a85d6cddd908cc66320148291d8fbc8ef2e41d5a4c4a2a9095a9002f141b59; bartowski: `bartowski/Qwen3.8-27B-GGUF` at `87410e7`, `Qwen3.8-27B-IQ2_XXS.gguf`, SHA-256 9f8327b475beeb313977f9e4ca51507f60d61f0e4d0248037e0b524a92bbbe7f; bartowski: `bartowski/Qwen3.8-27B-GGUF` at `87410e7`, `Qwen3.8-27B-IQ2_S.gguf`, SHA-256 9fb5b33adb08cdf37d3ea5531552118266370ff78ffa8fea6c32aa4ebc7b6870; Unsloth: `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e`, `Qwen3.8-27B-UD-Q4_K_XL.gguf`, SHA-256 3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e; Unsloth: `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e`, `Qwen3.8-27B-Q8_0.gguf`, SHA-256 a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348

Protocol `rtx5090-2026-10-27`, text `wiki` (WikiText-2 test):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| `Qwen3.8-27B-BF16-nomtp.gguf` | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 53.81 | 53808282144 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `89a6492a2160` | TurinTech | 7.83 | 7830620736 | 0.2631 | 78.1 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 0.2096 | 80.5 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 0.1280 | 85.2 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 0.3280 | 75.8 |
| `Qwen3.8-27B-IQ2_XXS.gguf` | `bartowski/Qwen3.8-27B-GGUF` at `87410e7` | TurinTech | 8.88 | 8881268448 | 0.2819 | 78.3 |
| `Qwen3.8-27B-IQ2_S.gguf` | `bartowski/Qwen3.8-27B-GGUF` at `87410e7` | TurinTech | 9.68 | 9684043488 | 0.2053 | 81.7 |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 17.56 | 17559178144 | 0.0056 | 96.9 |
| `Qwen3.8-27B-Q8_0.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 29.05 | 29047086048 | 0.0007 | 98.9 |

Protocol `rtx5090-2026-10-27`, text `chat` (OpenAssistant validation conversations (English threads as User/Assistant text)):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| `Qwen3.8-27B-BF16-nomtp.gguf` | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 53.81 | 53808282144 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `89a6492a2160` | TurinTech | 7.83 | 7830620736 | 0.3384 | 79.6 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 0.3462 | 79.8 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 0.2194 | 84.0 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 0.3378 | 79.5 |
| `Qwen3.8-27B-IQ2_XXS.gguf` | `bartowski/Qwen3.8-27B-GGUF` at `87410e7` | TurinTech | 8.88 | 8881268448 | 0.3103 | 80.8 |
| `Qwen3.8-27B-IQ2_S.gguf` | `bartowski/Qwen3.8-27B-GGUF` at `87410e7` | TurinTech | 9.68 | 9684043488 | 0.2959 | 82.6 |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 17.56 | 17559178144 | 0.0231 | 96.1 |
| `Qwen3.8-27B-Q8_0.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 29.05 | 29047086048 | 0.0050 | 98.8 |

### Protocol `rtx5090-2026-10-25`

One NVIDIA RTX 5090, 25 October 2026 (the 250-text panel the card was first published with).

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
- Source, on the results page: the table columns "KL, all 250", "Same next token, all 250", "KL, chat", "Same next token, chat", "KL, coding", "Same next token, coding", "KL, long documents", "Same next token, long documents", "KL, multilingual", "Same next token, multilingual", "KL, tool text", "Same next token, tool text"
- Comparison files: AtomicChat: `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188`, `Qwen3.8-27B-AD-IQ1_M.gguf`, SHA-256 e2a85d6cddd908cc66320148291d8fbc8ef2e41d5a4c4a2a9095a9002f141b59; Unsloth: `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e`, `Qwen3.8-27B-UD-IQ2_XXS.gguf`, SHA-256 e792d8fb3142fe6d9171876d6da0f71f05a71028718debc72dbec93ff645e67d; Unsloth: `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e`, `Qwen3.8-27B-UD-IQ2_S.gguf`, SHA-256 7897d2c5a5cee46aef50895141b2c8a0803c1185f3d03c4fda4cd137a7ad77fe

Protocol `rtx5090-2026-10-25`, text `chat` (Chat):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 original | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 54.66 | 54657734112 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `a710497a6269` | TurinTech | 7.83 | 7830620736 | 0.928 | 72.1 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 0.703 | 70.5 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 0.752 | 75.2 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 0.576 | 77.4 |

Protocol `rtx5090-2026-10-25`, text `coding` (Coding):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 original | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 54.66 | 54657734112 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `a710497a6269` | TurinTech | 7.83 | 7830620736 | 1.514 | 67.6 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 2.045 | 61.6 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 1.350 | 72.0 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 1.071 | 77.2 |

Protocol `rtx5090-2026-10-25`, text `long_document` (Long documents):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 original | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 54.66 | 54657734112 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `a710497a6269` | TurinTech | 7.83 | 7830620736 | 1.095 | 67.6 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 1.365 | 65.8 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 0.645 | 79.8 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 0.210 | 94.7 |

Protocol `rtx5090-2026-10-25`, text `multilingual` (Multilingual):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 original | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 54.66 | 54657734112 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `a710497a6269` | TurinTech | 7.83 | 7830620736 | 0.694 | 69.1 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 0.745 | 68.9 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 0.844 | 68.4 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 0.483 | 75.0 |

Protocol `rtx5090-2026-10-25`, text `tooltext` (Tool text (BFCL conversations, not the scored tasks)):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 original | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 54.66 | 54657734112 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `a710497a6269` | TurinTech | 7.83 | 7830620736 | 2.723 | 66.5 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 2.396 | 68.7 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 2.517 | 71.3 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 2.126 | 73.3 |

Protocol `rtx5090-2026-10-25`, text `all` (All 250 texts, each text weighted equally):

| File | Repository | Measured by | Size (GB) | Size (bytes) | Mean KLD (nats) | Same top p (top-1 agreement, %) |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 original | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff60` | reference | 54.66 | 54657734112 | reference | reference |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | `turintech/Qwen3.8-27B-Artemis-GGUF` at `a710497a6269` | TurinTech | 7.83 | 7830620736 | 1.391 | 68.6 |
| `Qwen3.8-27B-AD-IQ1_M.gguf` | `AtomicChat/Qwen3.8-27B-GGUF` at `ca10ebceb188` | TurinTech | 8.50 | 8498107872 | 1.451 | 67.1 |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 7.27 | 7266070528 | 1.222 | 73.4 |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | `unsloth/Qwen3.8-27B-GGUF` at `4ca720788d1e` | TurinTech | 8.37 | 8371970048 | 0.893 | 79.5 |

</details>
<!-- closeness:end -->

<!-- v2-checksums:start -->
<details>
<summary>File checksums</summary>

| File | Bytes | sha256 |
| --- | --- | --- |
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | 7,830,620,736 | `6b41a0716f4222f364abbfa88e718db2501bdd173de186f0ba45dd490587602c` |
| `mmproj-BF16.gguf` | 931,146,176 | `b56aca76c51d2ba0e7b5c34d4fab84d50ad71dc7b12f17c0faa9f57e816a5df5` |
| `mmproj-F16.gguf` | 927,607,232 | `7a47ce71686d9a26eb90a9aaa73da3e9a9fc6f3648d9646b1b06959e7e83bb6b` |
| `MTP/mtp-Qwen3.8-27B-Artemis-Q4_0.gguf` | 1,342,556,960 | `5331d11b17004975ef6844ef04f595d0deb439ed8e22f20496f3c3a7c31f3246` |

</details>
<!-- v2-checksums:end -->

<!-- v2-footer:start -->
---

<div align="center">

[Artemis](https://www.turintech.ai/product/overview) is TurinTech's optimisation platform: teams use it to make their own models smaller and faster, or to optimise their code.

[All our models](https://www.turintech.ai/models) · [LinkedIn](https://www.linkedin.com/company/turintechai) · [Questions and requests](https://huggingface.co/spaces/turintech/README/discussions/1)

<sub>Apache-2.0, as released by Qwen (Alibaba Cloud) ([licence](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE)); see `LICENSE` and `NOTICE`. An independent TurinTech release, not affiliated with or endorsed by Qwen (Alibaba Cloud). When serving these files to others, for example through an API, an inference provider or OpenRouter, we'd appreciate a mention of TurinTech (turintech.ai) as the source of the quantization.</sub>

These files keep improving.

</div>
<!-- v2-footer:end -->
