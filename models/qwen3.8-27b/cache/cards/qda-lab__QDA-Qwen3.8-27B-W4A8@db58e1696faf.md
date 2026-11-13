---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags: [qwen, qwen3.8, qda, quantized, int4, fp8, w4a8, compressed-tensors, vllm]
---

# QDA-Qwen3.8-27B-W4A8

4-bit weight, 8-bit (FP8) activation quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
**2.9× smaller than BF16 (19.5 GB vs. 55.6 GB), on par with BF16 accuracy.**

## Evaluation

<table>
<thead><tr>
<th>Benchmark</th>
<th>BF16<br>55.6 GB</th>
<th>QDA W4A8<br>19.5 GB</th>
</tr></thead><tbody>
<tr><td>AIME 2025</td><td>88.0</td><td><strong>89.3</strong></td></tr>
<tr><td>HMMT Feb 2026</td><td>74.1</td><td><strong>75.6</strong></td></tr>
<tr><td>GPQA-Diamond</td><td>82.0</td><td><strong>81.9</strong></td></tr>
<tr><td>IFBench</td><td>67.7</td><td><strong>66.9</strong></td></tr>
<tr><td>MMLU-Pro</td><td>83.7</td><td><strong>82.6</strong></td></tr>
<tr><td>LiveCodeBench</td><td>85.8</td><td><strong>86.8</strong></td></tr>
<tr><td><b>Average</b></td><td>80.2</td><td><strong>80.5</strong></td></tr>
</tbody></table>

Thinking mode, pass@1 averaged over 1–32 seeds, same settings for all models.

## Usage

Runs on vLLM on NVIDIA Hopper GPUs only (H100 / H200, sm90): the W4A8 kernel in vLLM requires compute capability 9.0.

vLLM:

```bash
vllm serve qda-lab/QDA-Qwen3.8-27B-W4A8
```

## How it was made

Weights are 4-bit integers in groups of 128 and activations are 8-bit floating point (dynamic), quantized with GPTQ on a Hadamard-rotated copy of the model; the rotation is folded
into the weights, so there is no runtime transform. The recipe was chosen by QDA's automated search.

## Reproduce

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Qwen3.8-27B-W4A8/*" --local-dir inputs
cd inputs/QDA-Qwen3.8-27B-W4A8
qda build qwen3.8-27b-w4a8.yaml --set out=QDA-Qwen3.8-27B-W4A8-rebuild
```

Needs the QDA library (the `qda` command). Recipe: [`qwen3.8-27b-w4a8.yaml`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/blob/main/QDA-Qwen3.8-27B-W4A8/qwen3.8-27b-w4a8.yaml); inputs
(calibration sample and distillation data): [`QDA-Qwen3.8-27B-W4A8/`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/tree/main/QDA-Qwen3.8-27B-W4A8) in `qda-lab/qda-recipe-inputs`.
Takes about 17 min on 1× H100 (measured). The rebuild matches this checkpoint's KL to BF16 within 5 %; weights are not bit-identical.

## License

Apache-2.0, the same as the base model.

## Citation

TBD.
