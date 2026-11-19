---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags: [qwen, qwen3.8, qda, quantized, int4, w4a16, compressed-tensors, vllm, sglang]
---

# QDA-Qwen3.8-27B-W4A16-dsh

4-bit weight quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), found by the same QDA
search as [QDA-Qwen3.8-27B-W4A16](https://huggingface.co/qda-lab/QDA-Qwen3.8-27B-W4A16) with a different search
agent backend (QDA's `dsh` backend).

## Evaluation

<table>
<thead><tr>
<th>Benchmark</th>
<th>BF16</th>
<th>QDA W4A16</th>
<th>QDA W4A16-dsh</th>
</tr></thead><tbody>
<tr><td>AIME 2025</td><td>88.0</td><td>89.4</td><td><strong>90.8</strong></td></tr>
<tr><td>HMMT Feb 2026</td><td>74.1</td><td>75.0</td><td><strong>76.2</strong></td></tr>
<tr><td>GPQA-Diamond</td><td>82.0</td><td>81.9</td><td><strong>82.4</strong></td></tr>
<tr><td>IFBench</td><td>67.7</td><td>67.6</td><td><strong>68.4</strong></td></tr>
<tr><td>MMLU-Pro</td><td>83.7</td><td>82.8</td><td><strong>83.2</strong></td></tr>
<tr><td>LiveCodeBench</td><td>85.8</td><td><strong>86.9</strong></td><td>86.5</td></tr>
<tr><td><b>Average</b></td><td>80.2</td><td>80.6</td><td><strong>81.2</strong></td></tr>
</tbody></table>

Thinking mode, pass@1 averaged over 1–32 seeds. Same sampling, token caps and seeds for every model. Bold marks the
higher of the two QDA columns; on every benchmark the difference between them is within run-to-run noise (a
question-paired comparison resolves no difference). The BF16 column is the one on the
[QDA-Qwen3.8-27B-W4A16](https://huggingface.co/qda-lab/QDA-Qwen3.8-27B-W4A16) card.

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
<tr><td>Repeats (seed 17 + j for repeat j = 0, 1, …)</td><td>32 (seeds 17–48)</td><td>32 (seeds 17–48)</td><td>16 (seeds 17–32)</td><td>16 (seeds 17–32)</td><td>1 (seed 17)</td><td>4 (seeds 17–20)</td></tr>
<tr><td>Sampling</td><td colspan="6">temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1.0</td></tr>
<tr><td>Thinking</td><td colspan="6">on; reasoning ends at <code>&lt;/think&gt;</code></td></tr>
<tr><td>chat_template_kwargs</td><td colspan="6"><code>{"reasoning_effort": "medium"}</code></td></tr>
<tr><td>MTP / speculative decoding</td><td colspan="6">off</td></tr>
</tbody></table>

## Usage

vLLM:

```bash
VLLM_DISABLED_KERNELS=MacheteLinearKernel vllm serve qda-lab/QDA-Qwen3.8-27B-W4A16-dsh
```

On H100 and H200, without a CUDA 13 toolkit on the machine, the kernel's compile at start fails (`failed to open
libnvrtc-builtins.so.13.0`). Put the venv's NVIDIA libraries on the loader path first:

```bash
lib="$(python -c 'import glob, sysconfig as s; p = s.get_paths(); print(":".join(sorted({d for k in ("purelib", "platlib") for d in glob.glob(p[k] + "/nvidia/*/lib")})))')" &&
  export LD_LIBRARY_PATH="$lib:$LD_LIBRARY_PATH"
```

SGLang:

```bash
python -m sglang.launch_server --model-path qda-lab/QDA-Qwen3.8-27B-W4A16-dsh
```

## How it was made

Weights are 4-bit integers in groups of 128, quantized with GPTQ on a Hadamard-rotated copy of the model. GPTQ weights
each calibration token by a Fisher-trace score (the original model's uncertainty over the next 16 positions, floored
at a quarter of the mean). The rotation is folded into the weights, so there is no runtime transform. The recipe was
chosen by QDA's automated search.

## Calibration data

A mix of two public files, 400 samples of 2,048 tokens (weight 3 and weight 2, as in the recipe):

1. `own-answers-768.jsonl` (768 rows, sha256 `c17e206c1dcd78693a2b92da9cffb9000084fc2db570c5eb4e3edd6ff2b3fa26`):
   768 prompts from [qda-lab/qda-calibration-pools](https://huggingface.co/datasets/qda-lab/qda-calibration-pools)
   (config `reasoning`, revision `cab86344e6acd38fb8a679d466982be49f93d270`), each with the base model's own answer
   (vLLM, temperature 1.0, top-k 20, top-p 0.95, seed 17, at most 4,096 new tokens).
2. `d1-filtered.jsonl` (2,011 rows, sha256 `f5d64911da06d424254799720630f8e3b5962ea39ed15432ff31b059131bd44b`):
   qda-calibration-pools config `d1` (`d1.jsonl`, 2,048 rows, sha256
   `d7bce0abbf2e6e1eac37f136ddfeb6bd77cba1e9a1854376850846e32bf69901`, same revision) with 37 rows removed for overlap
   with evaluation sets (34 with MMLU-Pro test, then 3 sharing a 20-word window with an evaluation set); the row
   numbers are in `d1-filter.json`, and every other row is kept in order, byte for byte.

The calibration files keep their own licences: `d1-filtered.jsonl` is CC BY-NC-SA 4.0, as qda-calibration-pools; the
prompts in `own-answers-768.jsonl` come from NuminaMath-CoT (Apache-2.0), code_contests (CC BY 4.0) and ARC
(CC BY-SA 4.0).

## Reproduce

Install QDA 0.1.0 (Linux, an NVIDIA GPU and [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/humanfia/qda.git && cd qda
uv venv -p 3.12 --managed-python && . .venv/bin/activate
uv pip install -c constraints.txt "./packages/qda-quant[hf,gguf]" "./packages/qda-kl[vllm]" ./packages/qda-agent
```

Then:

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Qwen3.8-27B-W4A16-dsh/*" --local-dir inputs
cd inputs/QDA-Qwen3.8-27B-W4A16-dsh
qda build qwen3.8-27b-w4a16-dsh.yaml --set out=QDA-Qwen3.8-27B-W4A16-dsh-rebuild
```

Needs QDA 0.1.0. Original model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, the revision the recipe builds from. Recipe and inputs:
[`QDA-Qwen3.8-27B-W4A16-dsh/`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/tree/main/QDA-Qwen3.8-27B-W4A16-dsh)
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
