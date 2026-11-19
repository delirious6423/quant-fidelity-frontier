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

<table>
<thead><tr>
<th>Evaluation setting</th>
<th>AIME 2025</th>
<th>HMMT Feb 2026</th>
<th>GPQA-Diamond</th>
<th>IFBench</th>
<th>MMLU-Pro</th>
<th>LiveCodeBench</th>
</tr></thead><tbody>
<tr><td>Max new tokens per item</td><td>81,920</td><td>81,920</td><td>32,768</td><td>32,768</td><td>32,768</td><td>81,920</td></tr>
<tr><td>Repeats (seed 17 + j for repeat j = 0, 1, …)</td><td>8 (seeds 17–24)</td><td>8 (seeds 17–24)</td><td>4 (seeds 17–20)</td><td>4 (seeds 17–20)</td><td>1 (seed 17)</td><td>1 (seed 17)</td></tr>
<tr><td>Sampling</td><td colspan="6">temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1.0</td></tr>
<tr><td>Thinking</td><td colspan="6">on; reasoning ends at <code>&lt;/think&gt;</code></td></tr>
<tr><td>chat_template_kwargs</td><td colspan="6"><code>{"reasoning_effort": "medium"}</code></td></tr>
<tr><td>MTP / speculative decoding</td><td colspan="6">off</td></tr>
</tbody></table>

Settings from the frozen eval config.

## Usage

llama.cpp:

```bash
llama-server -hf qda-lab/QDA-Qwen3.8-27B-Q4-GGUF
```

The file contains the model's MTP (multi-token prediction) head; add `--spec-type draft-mtp` to
llama-server to use it for speculative decoding.

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

Install QDA 0.1.0 (Linux, an NVIDIA GPU and [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/humanfia/qda.git && cd qda
uv venv -p 3.12 --managed-python && . .venv/bin/activate
uv pip install -c constraints.txt "./packages/qda-quant[hf,gguf]" "./packages/qda-kl[vllm]" ./packages/qda-agent
```

Then:

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Qwen3.8-27B-Q4-GGUF/*" --local-dir inputs
git clone https://github.com/ggml-org/llama.cpp llama.cpp && git -C llama.cpp checkout 9e2a04a439edc49f08c5acdafefae19dfa25a658
cd inputs/QDA-Qwen3.8-27B-Q4-GGUF
QDA_LLAMA_CPP=../../llama.cpp qda build qwen3.8-27b-q4-gguf.yaml --set out=QDA-Qwen3.8-27B-Q4-GGUF-rebuild
```

Needs QDA 0.1.0. Original model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, the revision the recipe builds from. A rebuild took 1381 s on 1× H100 (80 GB). It also needs a llama.cpp checkout at the commit
above (its converter writes the bf16 template; build it, since the importance-weighted fits use its library). Recipe
and inputs:
[`QDA-Qwen3.8-27B-Q4-GGUF/`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/tree/main/QDA-Qwen3.8-27B-Q4-GGUF)
in `qda-lab/qda-recipe-inputs` (sha256 of every file in its `SHA256SUMS`).

## License

Apache-2.0, the same as the base model.

## Citation

[QDA: Closed-Loop LLM Quantization Design with Agents](https://arxiv.org/abs/2611.23388)

```bibtex
@article{li2026qda,
  title   = {QDA: Closed-Loop LLM Quantization Design with Agents},
  author  = {Li, Xingyang and Zhao, Tianlang and Han, Song and Zhu, Ligeng},
  journal = {arXiv preprint arXiv:2611.23388},
  year    = {2026}
}
```
