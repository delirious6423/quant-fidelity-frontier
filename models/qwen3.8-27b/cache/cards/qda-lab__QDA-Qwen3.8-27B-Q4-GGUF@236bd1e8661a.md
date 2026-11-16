---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags: [qwen, qwen3.8, qda, quantized, gguf, llama.cpp, q4_k]
---

# QDA-Qwen3.8-27B-Q4-GGUF

A GGUF of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for llama.cpp, a mix of 4- and 5-bit types
(`QDA-Qwen3.8-27B-Q4.gguf`, 17.6 GB). Runs in stock llama.cpp.

The file's header field `general.file_type` says 15 (Q4_K_M), the label QDA's GGUF writer gives every file it writes;
it was left as is so that the file is byte for byte the one evaluated below. llama.cpp loads every tensor by its own
stored type, so the label changes nothing at run time; the actual mix is given under How it was made.

## Evaluation

<table>
<thead><tr>
<th>Benchmark</th>
<th>QDA</th>
<th>unsloth UD-Q4_K_XL</th>
</tr></thead><tbody>
<tr><td>AIME 2025</td><td><strong>89.6</strong></td><td>87.9</td></tr>
<tr><td>HMMT Feb 2026</td><td><strong>75.8</strong></td><td>75.4</td></tr>
<tr><td>GPQA-Diamond</td><td><strong>83.2</strong></td><td>82.1</td></tr>
<tr><td>IFBench</td><td>68.7</td><td><strong>69.8</strong></td></tr>
<tr><td>MMLU-Pro</td><td>83.1</td><td><strong>83.2</strong></td></tr>
<tr><td>LiveCodeBench</td><td><strong>83.0</strong></td><td>82.6</td></tr>
<tr><td><b>Average</b></td><td><strong>80.6</strong></td><td>80.2</td></tr>
</tbody></table>

Thinking mode, pass@1 averaged over 1–8 seeds, llama.cpp. Same sampling, token caps and seeds for both files.
Differences from the reference quantization are within run-to-run noise on every benchmark.

## Usage

llama.cpp:

```bash
llama-server -hf qda-lab/QDA-Qwen3.8-27B-Q4-GGUF
```

## How it was made

The file stores each tensor in one of llama.cpp's own formats: by weight count 39 % Q5_K, 30 % Q4_K, 12 % Q6_K,
12 % IQ4_XS, 5 % Q3_K (the token embedding) and 2 % Q8_0 (norms and other small tensors F32). The per-tensor mix was
chosen by QDA's automated search. In the 64 transformer layers, 126 of the Q4_K tensors are written from the build:
round-to-nearest on llama.cpp's Q4_K grid, fitting each group's scales with llama.cpp's importance-weighted fit,
weighted by the Hessian diagonal measured on the calibration data. Every other quantized tensor is written from the
bf16 model by llama.cpp's own quantizer, weighted by an importance matrix measured on the same calibration
conversations; the Q8_0 tensors, the token embedding and the multi-token-prediction block (kept) are quantized without
it.

## Calibration data (public)

All 2,048 rows of [qda-lab/qda-calibration-pools](https://huggingface.co/datasets/qda-lab/qda-calibration-pools),
config `chat` (`chat.jsonl`, sha256 `bebad4681835efefafca220335ac42f4dbcc915708e93eb96ca46c7314feaef1`) at revision
`cab86344e6acd38fb8a679d466982be49f93d270`: multi-turn conversations from HuggingFaceH4/ultrachat_200k. The build read
the same 2,048 rows, re-serialised: `chat-2048.jsonl` in the recipe inputs holds the same rows in the same order
(only the JSON string escaping differs, so its sha256 differs; see `chat-filter.json`). No row was removed.

As built: 597 samples of 4,096 tokens, seed 0.

The importance matrix (`chat-27b.imatrix.gguf` in the recipe inputs) was measured by llama.cpp's `llama-imatrix` on
the same 2,048 conversations as plain text (`chat-27b-imatrix.txt`: each conversation's messages joined by newlines,
followed by a blank line), 128 chunks of 4,096 tokens, output head included.

The calibration file keeps its own licence: the `chat` rows come from ultrachat_200k (MIT); qda-calibration-pools as a
whole is CC BY-NC-SA 4.0.

## Reproduce

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Qwen3.8-27B-Q4-GGUF/*" --local-dir inputs
git clone https://github.com/ggml-org/llama.cpp llama.cpp && git -C llama.cpp checkout 9e2a04a439edc49f08c5acdafefae19dfa25a658
cd inputs/QDA-Qwen3.8-27B-Q4-GGUF
QDA_LLAMA_CPP=../../llama.cpp qda build qwen3.8-27b-q4-gguf.yaml --set out=QDA-Qwen3.8-27B-Q4-GGUF-rebuild
```

Needs the QDA library 0.2.0 or later; the recipe was checked with it. It also needs a llama.cpp checkout at the commit
above (its converter writes the bf16 template; build it, since the importance-weighted fits use its library). Recipe
and inputs:
[`QDA-Qwen3.8-27B-Q4-GGUF/`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/tree/main/QDA-Qwen3.8-27B-Q4-GGUF)
in `qda-lab/qda-recipe-inputs` (sha256 of every file in its `SHA256SUMS`).

## License

Apache-2.0, the same as the base model.

## Citation

TBD.
