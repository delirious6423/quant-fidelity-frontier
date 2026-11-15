---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags: [qwen, qwen3.8, qda, quantized, gguf, llama.cpp, q6_k]
---

# QDA-Qwen3.8-27B-Q6-GGUF

A GGUF of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for llama.cpp, mostly Q6_K
(`QDA-Qwen3.8-27B-Q6.gguf`, 22.0 GB). Runs in stock llama.cpp.

The file's header field `general.file_type` says 15 (Q4_K_M), a label carried over from the build template; it was
left as is so that the file is byte for byte the one evaluated below. llama.cpp loads every tensor by its own stored
type, so the label changes nothing at run time; the actual mix is given under How it was made (mostly Q6_K).

## Evaluation

<table>
<thead><tr>
<th>Benchmark</th>
<th>QDA</th>
<th>unsloth UD-Q6_K</th>
</tr></thead><tbody>
<tr><td>AIME 2025</td><td><strong>89.6</strong></td><td>85.4</td></tr>
<tr><td>HMMT Feb 2026</td><td><strong>73.9</strong></td><td>71.6</td></tr>
<tr><td>GPQA-Diamond</td><td><strong>82.1</strong></td><td>81.1</td></tr>
<tr><td>IFBench</td><td><strong>68.8</strong></td><td>66.9</td></tr>
<tr><td>MMLU-Pro</td><td><strong>83.4</strong></td><td><strong>83.4</strong></td></tr>
<tr><td>LiveCodeBench</td><td>84.4</td><td><strong>84.8</strong></td></tr>
<tr><td><b>Average</b></td><td><strong>80.3</strong></td><td>78.9</td></tr>
</tbody></table>

Thinking mode, pass@1 averaged over 1–8 seeds, llama.cpp. Same sampling, token caps and seeds for both files.
Differences from the reference quantization are within run-to-run noise on every benchmark.

## Usage

llama.cpp:

```bash
llama-server -hf qda-lab/QDA-Qwen3.8-27B-Q6-GGUF
```

## How it was made

The file stores each tensor in one of llama.cpp's own formats: by weight count 75 % Q6_K, 19 % Q5_K, 5 % Q8_0 and
1 % Q4_K (norms and other small tensors F32). The per-tensor mix was chosen by QDA's automated search. In the 64
transformer layers, GPTQ (damping 0.01, static activation order) solves the codes of the Q5_K and Q6_K tensors
directly on llama.cpp's grids, fitting each group's scales with llama.cpp's importance-weighted fit, weighted by GPTQ's
Hessian diagonal. The token embedding, the output head, the multi-token-prediction block (kept) and
the tensors stored as Q8_0 are written from the bf16 model by llama.cpp's own quantizer.

## Calibration data

One file of the base model's own answers, published with the recipe:

- `kd-traces-1234.jsonl` (1,234 rows, sha256 `db975fe59cf0b8d509f01a90a4ed8d7ebe7211e1355e8f6a95f24823663ce6b4`):
  each row holds the token ids of one prompt and of the base model's (Qwen/Qwen3.8-27B) own answer, generated with
  vLLM (bf16) at temperature 1.0, top-p 0.95, top-k 20, seed 17 + the row number, at most 8,192 new tokens.
- The prompts, as text: `kd-prompts-1234.jsonl` (sha256
  `f694463e359576b3355f85546a0865a19953f083044369c7ad3e5e985e714723`): math, code and science problems from
  NuminaMath-CoT, code_contests and ARC. The build's contamination filter removed 8 rows that overlap evaluation sets
  (row numbers in `kd-filter.json`), leaving 1,226. 1,225 of the 1,226 used prompts are rows of
  [qda-lab/qda-calibration-pools](https://huggingface.co/datasets/qda-lab/qda-calibration-pools) (config
  `reasoning`, revision `cab86344e6acd38fb8a679d466982be49f93d270`); one is a code_contests problem (CC BY 4.0) the
  public pool's later filter removed.

As built: 597 samples of 4,096 tokens, seed 0.

The calibration files keep their own licences: the prompts come from NuminaMath-CoT (Apache-2.0), code_contests
(CC BY 4.0) and ARC (CC BY-SA 4.0).

## Reproduce

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Qwen3.8-27B-Q6-GGUF/*" --local-dir inputs
git clone https://github.com/ggml-org/llama.cpp llama.cpp && git -C llama.cpp checkout 9e2a04a439edc49f08c5acdafefae19dfa25a658
cd inputs/QDA-Qwen3.8-27B-Q6-GGUF
QDA_LLAMA_CPP=../../llama.cpp qda build qwen3.8-27b-q6-gguf.yaml --set out=QDA-Qwen3.8-27B-Q6-GGUF-rebuild
```

Needs the QDA library 0.2.0 or later; the recipe was checked with it. It also needs a llama.cpp checkout at the commit
above (its converter writes the bf16 template; build it for the fast quantizer). Recipe and inputs:
[`QDA-Qwen3.8-27B-Q6-GGUF/`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/tree/main/QDA-Qwen3.8-27B-Q6-GGUF)
in `qda-lab/qda-recipe-inputs` (sha256 of every file in its `SHA256SUMS`).

## License

Apache-2.0, the same as the base model.

## Citation

TBD.
