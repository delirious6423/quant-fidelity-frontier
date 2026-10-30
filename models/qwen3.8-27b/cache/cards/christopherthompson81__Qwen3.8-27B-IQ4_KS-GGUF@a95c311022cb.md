---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - gguf
  - ik_llama.cpp
  - imatrix
  - qwen3.8
---

# Qwen3.8-27B IQ4_KS GGUF

A 4-bit GGUF of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) built on ik_llama.cpp's IQ4_KS type. Its
quality (KL divergence against BF16) is better than Unsloth's IQ4_XS, and the file is 0.56 GiB smaller. It was made
and tuned with [inference.rs](https://github.com/christopherthompson81/inference.rs), a Rust LLM inference engine that
runs it on CUDA with speculative decoding; the speed figures below come from it.

| file | size | mean KLD vs BF16 | same top token |
|---|---|---|---|
| `Qwen3.8-27B-IQ4_KS.gguf` | 14.06 GiB | 0.0169 | 94.44% |
| for reference: Unsloth IQ4_XS | 14.62 GiB | 0.0175 | 94.1% |

## Compatibility: not mainline llama.cpp

The bulk of the weights are **IQ4_KS**, a quantization type that exists only in
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp). **Mainline llama.cpp, and the apps built on it (LM Studio,
Ollama, ...), cannot load this file.** It runs in:

- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- [inference.rs](https://github.com/christopherthompson81/inference.rs) (CUDA, including a tensor-core path for IQ4_KS
  in speculative decoding)

If you need a mainline-compatible file, use an IQ4_XS or Q4_K_M quant instead.

This file holds the text model, including its built-in MTP layer. It has no vision projector.

## Recipe

Unsloth's IQ4_XS tensor recipe, with IQ4_KS as the bulk type and the linear-attention projections lowered from Q5_K
to IQ4_KS. A per-tensor sensitivity sweep showed that change costs almost nothing in KLD at this bulk type.

| tensors | type |
|---|---|
| FFN (`ffn_up`, `ffn_gate`, `ffn_down`), full attention `q`/`k`/`output`, `attn_gate` | IQ4_KS |
| linear attention `attn_qkv`, `ssm_out` | IQ4_KS |
| full attention `attn_v` | Q5_K |
| `token_embd` | Q4_K |
| `output` | Q6_K |
| `ssm_alpha`, `ssm_beta`, norms | F32 |
| MTP layer (`blk.64`) | Q4_K, `attn_v` Q5_K, `nextn.eh_proj` Q8_0 |

Made with ik_llama.cpp (build c446f1e0) `llama-quantize` from the BF16 conversion, with
[Unsloth](https://huggingface.co/unsloth)'s importance matrix for this model. The rules, first match wins:

```
llama-quantize --imatrix imatrix_unsloth.dat \
  --custom-q "ssm_out\.weight=iq4_ks,attn_qkv\.weight=iq4_ks,blk\.64\.nextn\.eh_proj\.weight=q8_0,blk\.64\.attn_v\.weight=q5_K,blk\.64\..*\.weight=q4_K,ssm_alpha\.weight=f32,ssm_beta\.weight=f32,attn_v\.weight=q5_K,^token_embd\.weight=q4_K,^output\.weight=q6_K" \
  Qwen3.8-27B-BF16.gguf Qwen3.8-27B-IQ4_KS.gguf IQ4_KS
```

## How it was measured

`llama-perplexity` (ik_llama.cpp) on WikiText-2 test, `-c 512 --chunks 40`, against BF16 reference logits. Other
candidates in the same recipe, for context:

| bulk type | size | mean KLD | same top token |
|---|---|---|---|
| IQ4_KS, linear attention at Q5_K | 14.64 GiB | 0.0162 | 94.48% |
| **IQ4_KS, linear attention at IQ4_KS (this file)** | **14.06 GiB** | **0.0169** | **94.44%** |
| IQ4_XS (reproduces Unsloth's file) | 14.63 GiB | 0.0174 | 94.10% |
| IQ4_KT | 14.05 GiB | 0.0208 | 93.36% |
| IQ4_KSS | 14.05 GiB | 0.0221 | 93.41% |
| IQ3_K | 12.72 GiB | 0.0474 | 90.77% |

## Running it on a 24 GB GPU

On an RTX 3090 in inference.rs, this file with a 3-bit KV cache fits the full 262K context alongside the
[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) drafter (quantized to Q4_K at load,
7 drafts). The server takes about 19.9 GiB of the card's 23.6 GiB at load once inference.rs keeps the drafter's
prefix checkpoints in host RAM (the default from
[#723](https://github.com/christopherthompson81/inference.rs/issues/723)); the speed figures below were measured
before that, at 20.5 GiB, peaking near 21.2 GiB at a 199K-token request. Without the drafter it takes about 18.3 GiB:

```
inference serve --format gguf -m <dir> -f Qwen3.8-27B-IQ4_KS.gguf \
  --pa-context-len 262144 --pa-cache-type q3 \
  --mtp-model incoai/Qwen3.8-27B-DFlash2 --mtp-n-predict 7 --mtp-draft-isq q4k
```

### Speed by context length

Measured with inference.rs (master at
[#722](https://github.com/christopherthompson81/inference.rs/pull/722)) on an RTX 3090 at the settings above: full
262K window, q3 KV cache. Requests use the model's default sampling (temperature 1.0, top_k 20, top_p 0.95) with
thinking on, as a coding agent runs it. Two workloads, 1,024 generated tokens each:

- **Agentic coding:** the context is filled with `read_file` tool results from a real Rust codebase, then the model is
  asked to write unit tests for one of the files. It reads more files and writes the code over several turns.
- **Design review:** the same context, with an open-ended review question. Free prose is the hard case for
  speculative decoding, so this is the low end.

| context | agentic coding | design review | step time | no speculation | prefill at this depth |
|---|---|---|---|---|---|
| short | 151 tok/s (4.2) | 138 tok/s (3.8) | 27.9 ms | 52 tok/s | 1,780 tok/s |
| 10K | 151 tok/s (4.3) | 130 tok/s (3.6) | 28.4 ms | | 1,520 tok/s |
| 16K | 143 tok/s (4.0) | 114 tok/s (3.1) | 28.1 ms | | 1,340 tok/s |
| 35K | 146 tok/s (4.3) | 105 tok/s (3.0) | 29.3 ms | 48 tok/s | 1,090 tok/s |
| 46K | 140 tok/s (4.2) | 115 tok/s (3.3) | 30.3 ms | | 905 tok/s |
| 96K | 126 tok/s (4.1) | 123 tok/s (3.9) | 33.0 ms | 43 tok/s | 660 tok/s |
| 143K | 119 tok/s (4.2) | 81 tok/s (2.8) | 35.0 ms | | 470 tok/s |
| 199K | 139 tok/s (5.2) | 70 tok/s (2.6) | 37.5 ms | 37 tok/s | 350 tok/s |

Decode with the DFlash 2 drafter, with tokens per step (accepted drafts plus the one the target adds) in parentheses.
Context is the prompt's size in tokens. Step time is one draft-and-verify step in the coding run. No speculation is
plain decode over the same contexts with the review question, 512 generated tokens; without a drafter the task
doesn't change the speed.

Prefill at this depth is the rate for the tokens between the row above and this one, from cold prefills of each size
(a cold 199K prompt averages 530 tok/s; the short row is a cold 384-token prompt); prefills on top of a cached
context agree within 10%. The GPU never
throttled (SM clock 1.63-1.74 GHz).

Decode speed is tokens per step over step time: 28 ms a step when short, 37.5 ms at 199K. Tool calls and code that
reuse names from the context take 4-5 tokens a step, open prose 2.6-3.9, so a coding agent sees 120-150 tok/s across
the window. Prefill slows with context because each token attends to all earlier ones in the 16 full-attention
layers; the 48 linear-attention layers cost the same at any length.

### Comparing with other published numbers

Most published speeds are measured differently, and read higher:

- **Greedy decoding.** Speculative results are usually greedy (temperature 0). This model loops under greedy
  decoding, so those figures don't describe usable output, and speedups that only work greedily vanish with sampling.
  Everything here is sampled at the model's defaults.
- **Empty context.** `llama-bench`'s `pp512` and `tg128` start from nothing. The short row is the match: 1,780 tok/s
  prefill, 52 tok/s plain decode.
- **Prefill averages.** A cold average over a long prompt hides how slow its end is. The prefill column is the speed at
  that depth, what each turn of a long session pays.

Decode figures include thinking tokens, over 1,024 tokens a request.

At a 199K-token request the full-window server left about 1.8 GiB of the card for other programs (a desktop, a
browser); a GPU that also drives a display needs that headroom.
