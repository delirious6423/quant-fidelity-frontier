---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- nace
- quantized
- qwen
---

# Qwen3.8-27B · Nace GGUF

GGUF file of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for llama.cpp, LM Studio and Ollama. `Q4_K_XL` (17.29 GB, mean KL 0.005) is as close to the BF16 model as Unsloth UD-Q4_K_XL (17.30 GB, 0.005), a file of the same size or larger, in 512-token text, and closer in 128K-token text (KL gap −0.0002, 95% interval −0.0002 to −0.0001), counting only the model's own reply tokens in chat text.

Built from Unsloth's Qwen3.8-27B UD-Q4_K_XL GGUF (unsloth/Qwen3.8-27B-GGUF, Apache-2.0) with our own output-layer and FFN precision changes; base model Qwen/Qwen3.8-27B (Apache-2.0).

![Mean KL from BF16 at 128K context, averaged over prose, chat and code, against language-model size: the Nace files on this card, highlighted, and GGUF and EXL3 files of Qwen3.8-27B from other publishers in the same size range, each at its measured size and mean KL. Chat counts only the model's own reply tokens.](assets/kl-vs-size-128k.png)

The chart is at 128K context. Mean KL values in the text and tables are at 512 tokens unless they say 128K.

## Pick a file

| File (tag) | LM GB | Mean KL | All tokens | p99.9 KL | Top-1 accuracy % | Baseline (GB, mean KL) | KL gap (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| **`Q4_K_XL`** | 17.29 | 0.005 | 0.012 | 0.172 | 97.4 | Unsloth UD-Q4_K_XL (17.30, 0.005) | −0.0001 (−0.0003 to −0.0000) |

KL is the divergence from the BF16 model per token, so lower is closer. The gap is ours minus the baseline's mean KL; a negative gap means ours is closer, and the interval is a paired 95% bootstrap over windows. Top-1 accuracy is the share of tokens where the file's most likely next token is the same as the BF16 model's.

![Mean KL from BF16 at 512 tokens against language-model size: the Nace files on this card, highlighted, and GGUF and EXL3 files of Qwen3.8-27B from other publishers in the same size range, each at its measured size and mean KL](assets/kl-vs-size.png)

File names mark the size tier, not a single llama.cpp type: each file mixes tensor types from 3.4 to 8.5 bits per weight, all in stock llama.cpp. By share of file bytes, `Q4_K_XL` is 52% Q5_K, 21% IQ4_XS, 13% Q6_K, 10% Q4_K and 4% other types.

At 512 tokens, `Q4_K_XL` has a lower mean KL than 5 of the 6 GGUF files from other publishers within 0.98 to 1.06 times its size: mradermacher i1-Q4_1 (17.07 GB), bartowski IQ4_NL (17.19 GB), bartowski Q4_K_M (17.19 GB), Unsloth Q4_1 (17.24 GB) and bartowski Q4_1 (17.58 GB).

Counting every token, `Q4_K_XL` is also closer than Unsloth UD-Q4_K_XL at 128K (−0.003, −0.006 to −0.001).

Test text: English Wikipedia articles, chat and tool-call data, and GitHub code, all created after 2026-09-20; separate from the text used to fit and develop the files; used for release evaluation. 64 windows, 32,704 tokens, the same llama.cpp build (0a94a3f2) for every GGUF file. Size is language-model bytes, vision and MTP excluded, the same rule for every file. In chat and tool-call text, KL on this card counts only the model's own reply tokens (assistant turns), at 512 and 128K tokens; the All tokens values beside it count every token, user and tool turns included. Prose and code count every token.

## 128K context

KL from BF16 on 128K-token windows, same reference and same build.

**prose**

| File | LM GB | Mean KL | p99.9 KL | Top-1 accuracy % |
| --- | ---: | ---: | ---: | ---: |
| **Q4_K_XL** | 17.29 | 0.005 | 0.218 | 97.1 |

**chat and tool calls**

| File | LM GB | Mean KL | All tokens | p99.9 KL | Top-1 accuracy % |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Q4_K_XL** | 17.29 | 0.005 | 0.707 | 0.443 | 97.8 |

**code**

| File | LM GB | Mean KL | p99.9 KL | Top-1 accuracy % |
| --- | ---: | ---: | ---: | ---: |
| **Q4_K_XL** | 17.29 | 0.004 | 0.162 | 98.2 |

At 128K, over all three kinds of text, `Q4_K_XL` has a lower mean KL than all 6 GGUF files from other publishers within 0.98 to 1.06 times its size that we scored at that length: mradermacher i1-Q4_1 (17.07 GB), bartowski IQ4_NL (17.19 GB), bartowski Q4_K_M (17.19 GB), Unsloth Q4_1 (17.24 GB), Unsloth UD-Q4_K_XL (17.30 GB) and bartowski Q4_1 (17.58 GB).

### Task check

Accuracy against the baseline file named in the table above, scored with llama.cpp 0a94a3f2 `llama-perplexity` (questions: MMLU 13,943, ARC-Challenge 299, Winogrande 1,267). Each cell gives ours against the baseline: better or worse only when the paired 95% bootstrap interval over questions excludes 0, then the gap in percentage points with that interval. Each answer is scored by its likelihood, with no chat template or few-shot prompt, so the accuracies are not comparable to published scores: compare the files with each other, not with a leaderboard. All three ran with `-np 16`: one ARC-Challenge question has 5 answers, and the default of 4 parallel sequences stops the run there.

| File (tag) | Baseline | MMLU | ARC-Challenge | Winogrande |
|---|---|---|---|---|
| `Q4_K_XL` | Unsloth UD-Q4_K_XL | about the same, −0.0 (−0.2 to +0.1) | about the same, −1.0 (−2.7 to +0.3) | about the same, +0.2 (−0.4 to +0.8) |

## Quickstart

The examples use `Q4_K_XL`; swap in any file from the table.

### llama.cpp

Install a current [llama.cpp](https://github.com/ggml-org/llama.cpp), then start a local chat server. It downloads the file on first use:

```bash
llama-server -hf nace-ai/Qwen3.8-27B-GGUF:Q4_K_XL -c 8192 --port 8080
```

Open http://127.0.0.1:8080 to chat, or call the OpenAI-compatible API:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "What is 17 * 23?"}]
}'
```

The model thinks before it answers, and the API returns the thinking in `reasoning_content`. A small `max_tokens` can run out inside the thinking and leave the answer empty, so leave the cap off or turn thinking off: add `"chat_template_kwargs": {"enable_thinking": false}` to the request, or `--reasoning off` to `llama-cli`.

### LM Studio

Update LM Studio and its llama.cpp runtime. Search for `nace-ai/Qwen3.8-27B-GGUF` under Discover, download the `Q4_K_XL` file, and load it in Chat. If you already have the file, add it with `lms import --user-repo nace-ai/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_XL.gguf`.

### Ollama

```bash
ollama run hf.co/nace-ai/Qwen3.8-27B-GGUF:Q4_K_XL
```

Add `--think=false` to get the answer without the thinking.

### Kapitsa

On a Mac with [Kapitsa](https://github.com/nace-ai/kapitsa), download the file (`hf` comes with `pip install -U huggingface_hub`) and start it. The first start converts the file into Kapitsa's own format; then open http://127.0.0.1:8000 to chat, or send the curl request above to port 8000:

```bash
hf download nace-ai/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_XL.gguf --local-dir .
kapitsa serve --model ./Qwen3.8-27B-Q4_K_XL.gguf
```

Kapitsa stores the tensor types it has no kernels for at a higher bit width, so the converted model is larger than the file, and the measurements on this card, taken in llama.cpp, don't apply to it exactly.

## Files

| File | Download |
| --- | ---: |
| [`Qwen3.8-27B-Q4_K_XL.gguf`](https://huggingface.co/nace-ai/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_XL.gguf) | 17.65 GB |
| [`mmproj-bf16.gguf`](https://huggingface.co/nace-ai/Qwen3.8-27B-GGUF/resolve/main/mmproj-bf16.gguf) | 0.93 GB |

Download sizes, not memory use: the context and the engine need more. `mmproj-bf16.gguf` is the vision encoder, for image input.

Original model and license: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
