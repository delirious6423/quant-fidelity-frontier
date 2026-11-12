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
<tr><td>LiveCodeBench</td><td>82.9</td><td><strong>81.4</strong></td><td>79.1</td></tr>
<tr><td><b>Average</b></td><td>79.7</td><td><strong>81.0</strong></td><td>78.9</td></tr>
</tbody></table>

Thinking mode, pass@1 averaged over 1–32 seeds, same settings for all models.

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

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Qwen3.8-27B-W3A16/*" --local-dir inputs
qda build inputs/QDA-Qwen3.8-27B-W3A16/qwen3.8-27b-w3a16.yaml --set "devices=[0,1]"
```

Needs the QDA library (the `qda` command). Recipe: [`qwen3.8-27b-w3a16.yaml`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/blob/main/QDA-Qwen3.8-27B-W3A16/qwen3.8-27b-w3a16.yaml); inputs
(the 3-bit start checkpoint in `start/`, 14.6 GB, and the distillation data): [`QDA-Qwen3.8-27B-W3A16/`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/tree/main/QDA-Qwen3.8-27B-W3A16) in `qda-lab/qda-recipe-inputs`.
Takes about 4.5 h on 2× H100 (measured). The rebuild matches this checkpoint's KL to BF16 within 5 %; weights are not bit-identical.

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
