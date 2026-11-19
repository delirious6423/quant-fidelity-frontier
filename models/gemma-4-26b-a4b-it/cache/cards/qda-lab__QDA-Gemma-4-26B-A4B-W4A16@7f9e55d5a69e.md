---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags: [gemma, gemma4, qda, quantized, int4, w4a16, compressed-tensors, vllm]
---

# QDA-Gemma-4-26B-A4B-W4A16

4-bit weight quantization of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).
**3.1× smaller than BF16 (16.7 GB vs. 51.6 GB), keeps 99.0% of BF16 accuracy.**

## Evaluation

<table>
<thead><tr>
<th>Benchmark</th>
<th>BF16<br>51.6 GB</th>
<th>QDA W4A16<br>16.7 GB</th>
<th>cyankiwi AWQ-4bit<br>17.2 GB</th>
<th>ParoQuant<br>16.6 GB</th>
</tr></thead><tbody>
<tr><td>AIME 2025</td><td>85.4</td><td><strong>86.6</strong></td><td>79.9</td><td>74.3</td></tr>
<tr><td>HMMT Feb 2026</td><td>74.5</td><td><strong>72.3</strong></td><td>65.8</td><td>59.7</td></tr>
<tr><td>GPQA-Diamond</td><td>81.1</td><td><strong>79.5</strong></td><td>77.3</td><td>76.6</td></tr>
<tr><td>IFBench</td><td>80.6</td><td>80.0</td><td><strong>80.2</strong></td><td>76.7</td></tr>
<tr><td>MMLU-Pro</td><td>83.8</td><td><strong>83.9</strong></td><td>82.9</td><td>82.6</td></tr>
<tr><td>LiveCodeBench</td><td>81.2</td><td><strong>79.6</strong></td><td>78.3</td><td>75.8</td></tr>
<tr><td><b>Average</b></td><td>81.1</td><td><strong>80.3</strong></td><td>77.4</td><td>74.3</td></tr>
</tbody></table>

Thinking mode, pass@1 averaged over 1–32 seeds. Same sampling, token caps and seeds for every model; the ParoQuant column ran on its own serving stack (vLLM 0.19.1 with our MoE adapter and a group-size patch, one GPU).

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
<tr><td>Max new tokens per item</td><td>81,920</td><td>81,920</td><td>81,920</td><td>81,920</td><td>32,768</td><td>81,920</td></tr>
<tr><td>Repeats (seed 17 + j for repeat j = 0, 1, …)</td><td>32 (seeds 17–48)</td><td>32 (seeds 17–48)</td><td>16 (seeds 17–32)</td><td>16 (seeds 17–32)</td><td>1 (seed 17)</td><td>4 (seeds 17–20)</td></tr>
<tr><td>Sampling</td><td colspan="6">temperature 1.0, top-p 0.95, top-k 64, min-p 0, presence penalty 0, repetition penalty 1.0</td></tr>
<tr><td>Thinking</td><td colspan="6">on; reasoning ends at <code>&lt;channel|&gt;</code></td></tr>
<tr><td>chat_template_kwargs</td><td colspan="6">none</td></tr>
<tr><td>MTP / speculative decoding</td><td colspan="6">off</td></tr>
</tbody></table>

Speed on one H100 80GB, 512 generated tokens per request (tok/s = generated tokens / wall time, mean of 2 runs): 198 tok/s at batch 1 and 2,330 at batch 16 on vLLM 0.33.0; ParoQuant 133 and 1,652 on vLLM 0.19.1 with our MoE adapter.

## Usage

Runs on NVIDIA H100, A100, RTX Pro 6000 and B200 GPUs.

vLLM:

```bash
VLLM_DISABLED_KERNELS=MacheteLinearKernel vllm serve qda-lab/QDA-Gemma-4-26B-A4B-W4A16
```

On H100 and H200, without a CUDA 13 toolkit on the machine, the kernel's compile at start fails (`failed to open
libnvrtc-builtins.so.13.0`). Put the venv's NVIDIA libraries on the loader path first:

```bash
lib="$(python -c 'import glob, sysconfig as s; p = s.get_paths(); print(":".join(sorted({d for k in ("purelib", "platlib") for d in glob.glob(p[k] + "/nvidia/*/lib")})))')" &&
  export LD_LIBRARY_PATH="$lib:$LD_LIBRARY_PATH"
```

## How it was made

The routed experts are 4-bit integers in groups of 64, quantized with GPTQ; all other linear layers are 8-bit.
MoE routers and the vision tower stay in BF16. The recipe was chosen by QDA's automated search.

## Reproduce

Install QDA 0.1.0 (Linux, an NVIDIA GPU and [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/humanfia/qda.git && cd qda
uv venv -p 3.12 --managed-python && . .venv/bin/activate
uv pip install -c constraints.txt "./packages/qda-quant[hf,gguf]" "./packages/qda-kl[vllm]" ./packages/qda-agent
```

Then:

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Gemma-4-26B-A4B-W4A16/*" --local-dir inputs
cd inputs/QDA-Gemma-4-26B-A4B-W4A16
qda build gemma-4-26b-a4b-w4a16.yaml --set out=QDA-Gemma-4-26B-A4B-W4A16-rebuild
```

Needs QDA 0.1.0. Original model: `google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, the revision the recipe builds from. Recipe: [`gemma-4-26b-a4b-w4a16.yaml`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/blob/main/QDA-Gemma-4-26B-A4B-W4A16/gemma-4-26b-a4b-w4a16.yaml); inputs
(calibration sample and distillation data): [`QDA-Gemma-4-26B-A4B-W4A16/`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/tree/main/QDA-Gemma-4-26B-A4B-W4A16) in `qda-lab/qda-recipe-inputs`.
Built on 1× H100 (80 GB). Rebuilt weights are not bit-identical to this checkpoint.

## License

Apache-2.0, the same as the base model ([Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license)).

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
