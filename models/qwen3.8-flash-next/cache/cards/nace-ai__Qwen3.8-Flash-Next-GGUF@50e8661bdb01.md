---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- nace
- quantized
- qwen
---

# Qwen3.8-Flash-Next · Nace GGUF

GGUF file of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for llama.cpp, LM Studio and Ollama. `IQ4_XS` (93.44 GB, mean KL 0.063) is closer to the BF16 model than Unsloth UD-IQ4_XS (93.66 GB, 0.073), a file of the same size or larger, counting only the model's own reply tokens in chat text.

![Mean KL from BF16 at 128K context, averaged over prose, chat and code, against language-model size: the Nace files on this card, highlighted, and GGUF and EXL3 files of Qwen3.8-Flash-Next from other publishers in the same size range, each at its measured size and mean KL. Chat counts only the model's own reply tokens. 8 files with a much higher mean KL are off the scale and named under the chart.](assets/kl-vs-size-128k.png)

The chart is at 128K context. Mean KL values in the text and tables are at 512 tokens unless they say 128K.

## Pick a file

| File (tag) | LM GB | Mean KL | All tokens | p99.9 KL | Top-1 accuracy % | Baseline (GB, mean KL) | KL gap (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| **`IQ4_XS`** | 93.44 | 0.063 | 0.087 | 2.126 | 91.3 | Unsloth UD-IQ4_XS (93.66, 0.073) | −0.010 (−0.013 to −0.008) |

KL is the divergence from the BF16 model per token, so lower is closer. The gap is ours minus the baseline's mean KL; a negative gap means ours is closer, and the interval is a paired 95% bootstrap over windows. Top-1 accuracy is the share of tokens where the file's most likely next token is the same as the BF16 model's.

![Mean KL from BF16 at 512 tokens against language-model size: the Nace files on this card, highlighted, and GGUF and EXL3 files of Qwen3.8-Flash-Next from other publishers in the same size range, each at its measured size and mean KL](assets/kl-vs-size.png)

File names mark the size tier, not a single llama.cpp type: each file mixes tensor types from 2.2 to 8.5 bits per weight, all in stock llama.cpp. By share of file bytes, `IQ4_XS` is 54% IQ4_NL, 29% IQ4_XS, 10% IQ3_XXS and 7% other types.

At 512 tokens, `IQ4_XS` has a lower mean KL than all 13 GGUF files from other publishers within 0.98 to 1.06 times its size: bartowski IQ3_XS (91.93 GB), bartowski Q3_K_M (91.95 GB), bartowski Q3_K_L (93.18 GB), bartowski IQ3_M (93.22 GB), Unsloth UD-IQ4_XS (93.66 GB), mradermacher Q3_K_M (94.13 GB), mradermacher i1-Q3_K_M (94.13 GB), AtomicChat AD-4.27bpw-Q4_K_M-M64 (94.51 GB), mradermacher Q3_K_L (96.65 GB), mradermacher i1-Q3_K_L (96.65 GB), mradermacher i1-IQ4_XS (97.46 GB), bartowski IQ4_XS (97.67 GB) and mradermacher IQ4_XS (98.41 GB).

Counting every token, `IQ4_XS` is also closer than Unsloth UD-IQ4_XS at 128K (−0.007, −0.011 to −0.003).

EXL3 (ExLlamaV3, CUDA GPUs only): `IQ4_XS` (93.44 GB, mean KL 0.063) is closer to the BF16 model than turboderp 3.18 EXL3 (104.68 GB, 0.078), a larger file.

Test text: English Wikipedia articles, chat and tool-call data, and GitHub code, all created after 2026-09-20; separate from the text used to fit and develop the files; used for release evaluation. 64 windows, 32,704 tokens, the same llama.cpp build (f7b886fe) for every GGUF file. Size is language-model bytes, vision and MTP excluded, the same rule for every file. In chat and tool-call text, KL on this card counts only the model's own reply tokens (assistant turns), at 512 and 128K tokens; the All tokens values beside it count every token, user and tool turns included. Prose and code count every token.

## 128K context

KL from BF16 on 128K-token windows, same reference and same build.

**prose**

| File | LM GB | Mean KL | p99.9 KL | Top-1 accuracy % |
| --- | ---: | ---: | ---: | ---: |
| **IQ4_XS** | 93.44 | 0.041 | 2.530 | 92.5 |

**chat and tool calls**

| File | LM GB | Mean KL | All tokens | p99.9 KL | Top-1 accuracy % |
| --- | ---: | ---: | ---: | ---: | ---: |
| **IQ4_XS** | 93.44 | 0.112 | 0.412 | 5.602 | 90.6 |

**code**

| File | LM GB | Mean KL | p99.9 KL | Top-1 accuracy % |
| --- | ---: | ---: | ---: | ---: |
| **IQ4_XS** | 93.44 | 0.043 | 1.946 | 94.7 |

At 128K, over all three kinds of text, `IQ4_XS` has a lower mean KL than all 8 GGUF files from other publishers within 0.98 to 1.06 times its size that we scored at that length: bartowski IQ3_XS (91.93 GB), bartowski Q3_K_M (91.95 GB), bartowski IQ3_M (93.22 GB), Unsloth UD-IQ4_XS (93.66 GB), AtomicChat AD-4.27bpw-Q4_K_M-M64 (94.51 GB), mradermacher i1-IQ4_XS (97.46 GB), bartowski IQ4_XS (97.67 GB) and mradermacher IQ4_XS (98.41 GB).

### Task check

Accuracy against the baseline file named in the table above, scored with llama.cpp f7b886fe `llama-perplexity` (questions: MMLU 13,943, ARC-Challenge 299, Winogrande 1,267). Each cell gives ours against the baseline: better or worse only when the paired 95% bootstrap interval over questions excludes 0, then the gap in percentage points with that interval. Each answer is scored by its likelihood, with no chat template or few-shot prompt, so the accuracies are not comparable to published scores: compare the files with each other, not with a leaderboard. All three ran with `-np 16`: one ARC-Challenge question has 5 answers, and the default of 4 parallel sequences stops the run there. The stock tool stops on this model, whose n-gram embeddings refuse tokens shared by several sequences, so we patched it to give each answer its own copy of the shared question tokens; both files ran the same patched build. The patch is [scoring/patch_mc_seq.py](scoring/patch_mc_seq.py) in this repo; it differs from the copy we ran only in its comments and the comment marker it checks for. The Winogrande gap is 23 of 1,267 questions net; on Qwen3.5-9B, which the stock tool can score, the patch changed 16 of 1,267 answers.

| File (tag) | Baseline | MMLU | ARC-Challenge | Winogrande |
|---|---|---|---|---|
| `IQ4_XS` | Unsloth UD-IQ4_XS | about the same, −0.1 (−0.5 to +0.2) | about the same, −0.3 (−3.7 to +2.7) | better, +1.8 (+0.2 to +3.5) |

## Quickstart

The examples use `IQ4_XS`; swap in any file from the table.

### llama.cpp

Install a current [llama.cpp](https://github.com/ggml-org/llama.cpp), then start a local chat server. It downloads the file on first use:

```bash
llama-server -hf nace-ai/Qwen3.8-Flash-Next-GGUF:IQ4_XS -c 8192 --port 8080
```

Open http://127.0.0.1:8080 to chat, or call the OpenAI-compatible API:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "What is 17 * 23?"}]
}'
```

The model thinks before it answers, and the API returns the thinking in `reasoning_content`. A small `max_tokens` can run out inside the thinking and leave the answer empty, so leave the cap off or turn thinking off: add `"chat_template_kwargs": {"enable_thinking": false}` to the request, or `--reasoning off` to `llama-cli`.

### LM Studio

Update LM Studio and its llama.cpp runtime. Search for `nace-ai/Qwen3.8-Flash-Next-GGUF` under Discover, download the `IQ4_XS` file, and load it in Chat. If you already have the file, add it with `lms import --user-repo nace-ai/Qwen3.8-Flash-Next-GGUF IQ4_XS/Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf`.

### Ollama

Ollama doesn't load split files, so download the parts and merge them into one file first. The merge needs another 93 GB of free disk; `hf` comes with `pip install -U huggingface_hub`, `llama-gguf-split` with llama.cpp:

```bash
hf download nace-ai/Qwen3.8-Flash-Next-GGUF --include "IQ4_XS/*" --local-dir .
llama-gguf-split --merge IQ4_XS/Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf Qwen3.8-Flash-Next-IQ4_XS.gguf
echo "FROM ./Qwen3.8-Flash-Next-IQ4_XS.gguf" > Modelfile
ollama create qwen3.8-flash-next-iq4_xs -f Modelfile
ollama run qwen3.8-flash-next-iq4_xs
```

Add `--think=false` to get the answer without the thinking.

## Files

| File | Download |
| --- | ---: |
| [`IQ4_XS/Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf`](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-GGUF/resolve/main/IQ4_XS/Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf) | 93.45 GB |

Download sizes, not memory use: the context and the engine need more. Split files: pass the first part (`-00001-of-...`) to llama.cpp and keep the parts in one folder.

Original model and license: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
