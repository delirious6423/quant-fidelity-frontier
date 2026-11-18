---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags: [qwen, qwen3.8, qda, quantized, int3, w3a16, compressed-tensors, vllm]
---

# QDA-Qwen3.8-27B-W3A16

3-bit weight quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
**3.8× smaller than BF16 (14.5 GB vs. 55.6 GB), on par with BF16 accuracy.**

## Evaluation

<table>
<thead><tr>
<th>Benchmark</th>
<th>BF16<br>55.6 GB</th>
<th>QDA W3A16<br>14.5 GB</th>
<th>ISTA-DASLab GSQ 3-bit<br>--</th>
</tr></thead><tbody>
<tr><td>AIME 2025</td><td>88.0</td><td><strong>92.8</strong></td><td>91.0</td></tr>
<tr><td>HMMT Feb 2026</td><td>74.1</td><td><strong>81.9</strong></td><td>75.6</td></tr>
<tr><td>GPQA-Diamond</td><td>82.0</td><td><strong>79.8</strong></td><td>78.2</td></tr>
<tr><td>IFBench</td><td>67.7</td><td><strong>67.4</strong></td><td><strong>67.4</strong></td></tr>
<tr><td>MMLU-Pro</td><td>83.7</td><td><strong>82.4</strong></td><td>82.1</td></tr>
<tr><td>LiveCodeBench</td><td>85.8</td><td><strong>84.6</strong></td><td>82.4</td></tr>
<tr><td><b>Average</b></td><td>80.2</td><td><strong>81.5</strong></td><td>79.4</td></tr>
</tbody></table>

Thinking mode, pass@1 averaged over 1–32 seeds, same settings for all models.

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

Runs on vLLM only (SGLang does not support this 3-bit format yet), on NVIDIA H100, A100, L40S, RTX Pro 6000 and B200 GPUs.

vLLM:

```bash
vllm serve qda-lab/QDA-Qwen3.8-27B-W3A16
```

## How it was made

Built on ISTA-DASLab's public 3-bit GSQ checkpoint of Qwen3.8-27B: its 3-bit integer weights (groups of 128) are kept unchanged,
and QDA re-tuned the group scales and some BF16 tensors; the output head and token embeddings are 4-bit. The recipe was chosen by QDA's automated search.

## Reproduce

Install QDA 0.1.0 (Linux, an NVIDIA GPU and [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/Radioheading/qda.git && cd qda
uv venv -p 3.12 --managed-python && . .venv/bin/activate
uv pip install -c constraints.txt "./packages/qda-quant[hf]" "./packages/qda-kl[vllm]" ./packages/qda-agent
```

Then:

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Qwen3.8-27B-W3A16/*" --local-dir inputs
cd inputs/QDA-Qwen3.8-27B-W3A16
qda build qwen3.8-27b-w3a16.yaml --set out=QDA-Qwen3.8-27B-W3A16-rebuild --set "devices=[0,1]"
```

Needs QDA 0.1.0. Original model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, the revision the recipe builds from. Recipe: [`qwen3.8-27b-w3a16.yaml`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/blob/main/QDA-Qwen3.8-27B-W3A16/qwen3.8-27b-w3a16.yaml); inputs
(the 3-bit start checkpoint in `start/`, 14.6 GB, and the distillation data): [`QDA-Qwen3.8-27B-W3A16/`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/tree/main/QDA-Qwen3.8-27B-W3A16) in `qda-lab/qda-recipe-inputs`.
Built on 2× H100 (80 GB). The rebuild matches this checkpoint's KL to BF16 within 5 %; weights are not bit-identical.

The recipe starts from the 3-bit GSQ checkpoint [ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ) by IST Austria's DAS Lab (Apache-2.0), with its embedding dequantized to BF16 and Qwen3.8-27B's BF16 MTP block added; the dataset folder holds that start (`start/`, with ISTA's licence and our changes in its README) and `derive_start.py`, which rebuilds it byte for byte from ISTA's release:

```bash
hf download ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ --revision b5ce0b76f60020a875dee4f6ec9d934cca4121e4 --local-dir ista
hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --include "model.safetensors.index.json" "model-00018-of-00018.safetensors" --local-dir qwen
python inputs/QDA-Qwen3.8-27B-W3A16/derive_start.py ista qwen start
```

## License

Apache-2.0, the same as the base model.

## Citation

TBD.
