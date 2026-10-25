---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
- qwen
- qwen3.8
- gguf
- llama.cpp
- function-calling
- tool-calling
---

# Qwen3.8 27B Artemis GGUF

**Start here.** `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` (7.83 GB) is a small Qwen3.8 27B file for tool calling and
agents. On 111 BFCL parallel tool-calling tasks it got 87 right. AtomicChat's `AD-IQ1_M`, the closest public file we
checked by text-model size (8.21 GB without its MTP layer; ours is 4.6% smaller), got 63 (paired test p=0.0004), and Unsloth's smaller
`UD-IQ2_XXS` (7.27 GB) got 75 (p=0.017). It gives up some maths and closeness for that. It solved 180 of 200 GSM8K maths problems against
186 for AtomicChat's file (lower, though the paired test did not separate them), and on most kinds of text its
next-token probabilities are further from the original's than both Unsloth files at 7.27 and 8.37 GB. Text only.
[Details](#how-it-was-tested).

```bash
llama-server -hf turintech/Qwen3.8-27B-Artemis-GGUF:IQ2_XXS --port 8080 -c 8192 --jinja
```

<div>
  <a href="https://turintech.ai"><img src="https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/assets/turintech-logo.png" width="180" alt="TurinTech"></a>
  <p style="margin: 4px 0;"><em>TurinTech models, built with Artemis</em></p>
</div>

Qwen3.8 27B is the Qwen team's 27B model for chat, reasoning, coding and tool use
([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)). This repository has one text-only GGUF of it, about a
seventh of the size of the BF16 weights, for a GPU with 10 GB or more (9.0 GiB peak measured). The vision encoder and the
multi-token prediction (MTP) layer are not included.

This is an independent TurinTech conversion, not affiliated with or endorsed by Alibaba Cloud or the Qwen team. The
file is standard GGUF and runs on stock llama.cpp. Changes from the original: the text weights are converted to GGUF
and quantized for llama.cpp. See `NOTICE`.

## Why use this file

- Run locally so inputs can stay on the same machine.
- Work offline after downloading the model and runtime.
- Use Qwen3.8 27B for tool calling and agents with a 7.83 GB download: 87 of 111 BFCL parallel tool calls right, against 63 for the closest public file we checked by size (AtomicChat AD-IQ1_M, 8.21 GB of text model).
- Plan a GPU setup with the measured 9.0 GiB peak GPU memory (A100, 8,192-token context); maths and closeness trade-offs are listed below.

## Files

| File | Size | Same next token | Tool calls right (of 111) | Choose for |
| --- | --- | --- | --- | --- |
| [`Qwen3.8-27B-Artemis-IQ2_XXS.gguf`](https://huggingface.co/turintech/Qwen3.8-27B-Artemis-GGUF/resolve/main/Qwen3.8-27B-Artemis-IQ2_XXS.gguf) | **7.83 GB** | **68.6%** | **87** | Tool calling and agents on a GPU with 10 GB or more. |

GB means decimal download size, not RAM or VRAM. "Same next token" is how often its most likely next token matches
the BF16 original's over 250 mixed texts. The label is a size class; tensor types and the checksum are below.

## Before choosing

Pick this file when tool calls matter most and the budget is about 8 GB. If maths or closeness to the original,
especially on long documents, matter more, a larger file is the safer choice: Unsloth's `UD-IQ2_S` (8.37 GB) was closer to
the original on every kind of text we measured and got 183 of 200 maths problems, and on tool calls the paired test
did not separate it from this file (81 against 87 of 111).

The task results come from a final set of tasks that we had already used to compare our candidate files, and we
chose this file after seeing them, so they are not an independent test. See
[How it was tested](#how-it-was-tested).

## Run it

The tool-calling results use greedy decoding with thinking off. Set `chat_template_kwargs.enable_thinking` in each
request; the chat template turns thinking on by default.

```bash
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
`message.content`; allow enough output tokens. The `hf` command (`pip install -U huggingface_hub`) downloads the file
on its own:

```bash
hf download turintech/Qwen3.8-27B-Artemis-GGUF Qwen3.8-27B-Artemis-IQ2_XXS.gguf --local-dir .
```

The same file also loaded and answered in Ollama 0.40.11 (imported from a local Modelfile with Ollama's `qwen3.5`
renderer and parser; chat and a tool call through its native `/api/chat` with `"think": false`) and in LM Studio's headless `llmster` 0.0.25 with its CUDA llama.cpp
engine 2.41.0. These were short load-and-answer checks; the card's numbers come from llama.cpp.

Questions, problems or a model to consider for a smaller version? Reply in our [feedback thread](https://huggingface.co/spaces/turintech/README/discussions/1).

<details>
<summary>Evaluation details and limitations</summary>

## How it was tested

The reference is the BF16 GGUF conversion of Qwen/Qwen3.8-27B (revision `1d4bf0f`, 54.66 GB with the MTP layer,
which llama.cpp does not use for ordinary generation). Every run used llama.cpp builds of commit `4c3c581` with CUDA on
NVIDIA RTX 5090 GPUs, and every file got the same requests and the same prompt tokens (checked item by item).

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

Limitations: text only, no vision. Tested with thinking off; thinking mode was not evaluated. The tool tasks cover
parallel calls in Python only, and nothing here measures long-context accuracy, code correctness or speed. The
calibration data and the importance matrix are
not distributed.

</details>

## Memory

Peak memory of `llama-server` with the command above (`--port 8080 -c 8192 --jinja`, all layers on the GPU) answering
five short prompts on an NVIDIA A100 80 GB: 9.0 GiB of GPU memory and 7.5 GiB of process memory (RSS) with
llama.cpp b11617, and 8.8 GiB and 7.7 GiB with b11871. RSS counts the parts of the memory-mapped model file that were
read. The command keeps llama-server's default 8 GiB prompt cache, so system memory can grow in longer sessions; add
`--cache-ram 0` to limit it. A longer context needs more memory.

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
`general.name` reads "Parent". Neither changes how the file runs.

## File checksum

| File | Bytes | sha256 |
|---|---|---|
| `Qwen3.8-27B-Artemis-IQ2_XXS.gguf` | 7,830,620,736 | `6b41a0716f4222f364abbfa88e718db2501bdd173de186f0ba45dd490587602c` |

</details>

## Artemis

Teams use Artemis to test and improve code against their own objectives. [See Artemis](https://turintech.ai).

## Licence

Apache-2.0, the licence of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE); see
`LICENSE` and `NOTICE`.

When serving this file to others, for example through an API, an inference provider or OpenRouter, we'd
appreciate a mention of TurinTech (turintech.ai) as the source of the quantization.

These files keep improving.
