---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags: [qwen, qwen3.8, qda, quantized, int4, w4a16, compressed-tensors, vllm, sglang]
---

# QDA-Qwen3.8-27B-W4A16

4-bit weight quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
**2.8× smaller than BF16 (19.5 GB vs. 55.6 GB), on par with BF16 accuracy.**

## Evaluation

<table>
<thead><tr>
<th>Benchmark</th>
<th>BF16<br>55.6 GB</th>
<th>QDA W4A16<br>19.5 GB</th>
<th>cyankiwi AWQ-INT4<br>21.0 GB</th>
<th>RedHat INT4<br>19.5 GB</th>
<th>ParoQuant<br>18.8 GB</th>
</tr></thead><tbody>
<tr><td>AIME 2025</td><td>88.0</td><td><strong>89.4</strong></td><td>86.6</td><td>85.1</td><td>86.9</td></tr>
<tr><td>HMMT Feb 2026</td><td>74.1</td><td><strong>75.0</strong></td><td>73.2</td><td>70.6</td><td>66.0</td></tr>
<tr><td>GPQA-Diamond</td><td>82.0</td><td><strong>81.9</strong></td><td>81.6</td><td>81.3</td><td>81.0</td></tr>
<tr><td>IFBench</td><td>67.7</td><td>67.6</td><td><strong>68.4</strong></td><td>68.0</td><td>67.5</td></tr>
<tr><td>MMLU-Pro</td><td>83.7</td><td>82.8</td><td><strong>83.1</strong></td><td>82.7</td><td><strong>83.1</strong></td></tr>
<tr><td>LiveCodeBench</td><td>85.8</td><td><strong>86.9</strong></td><td>83.6</td><td>86.4</td><td>79.5</td></tr>
<tr><td><b>Average</b></td><td>80.2</td><td><strong>80.6</strong></td><td>79.4</td><td>79.0</td><td>77.3</td></tr>
</tbody></table>

Thinking mode, pass@1 averaged over 1–32 seeds. Same sampling, token caps and seeds for every model; the ParoQuant column ran on its own serving stack (vLLM 0.19.1 with paroquant 0.1.16).

Speed on one H100 80GB, 512 generated tokens per request (tok/s = generated tokens / wall time, mean of 2 runs): 107 tok/s at batch 1 and 1,377 at batch 16; ParoQuant 61 and 840 (both on vLLM 0.19.1).

## Usage

Runs on NVIDIA H100, A100, RTX Pro 6000 and B200 GPUs.

vLLM:

```bash
vllm serve qda-lab/QDA-Qwen3.8-27B-W4A16
```

SGLang:

```bash
python -m sglang.launch_server --model-path qda-lab/QDA-Qwen3.8-27B-W4A16
```

## How it was made

Weights are 4-bit integers in groups of 128, quantized with GPTQ on a Hadamard-rotated copy of the model; the rotation is folded
into the weights, so there is no runtime transform. The group scales were then fine-tuned by distillation. The recipe was chosen by QDA's automated search.

## Reproduce

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Qwen3.8-27B-W4A16/*" --local-dir inputs
qda build inputs/QDA-Qwen3.8-27B-W4A16/qwen3.8-27b-w4a16.yaml --set "devices=[0,1]"
```

Needs the QDA library (the `qda` command). Recipe: [`qwen3.8-27b-w4a16.yaml`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/blob/main/QDA-Qwen3.8-27B-W4A16/qwen3.8-27b-w4a16.yaml); inputs
(calibration sample and distillation data): [`QDA-Qwen3.8-27B-W4A16/`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/tree/main/QDA-Qwen3.8-27B-W4A16) in `qda-lab/qda-recipe-inputs`.
Takes about 23 min of GPTQ on 1× H100, then about 2.5 h of distillation on 2× H100 (measured). The rebuild matches this checkpoint's KL to BF16 within 5 %; weights are not bit-identical.

## License

Apache-2.0, the same as the base model.

## Citation

TBD.
