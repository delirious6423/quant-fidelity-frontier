---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags: [gemma, gemma4, qda, quantized, int4, w4a16, compressed-tensors, vllm, sglang]
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

Speed on one H100 80GB, 512 generated tokens per request (tok/s = generated tokens / wall time, mean of 2 runs): 198 tok/s at batch 1 and 2,330 at batch 16 on vLLM 0.33.0; ParoQuant 133 and 1,652 on vLLM 0.19.1 with our MoE adapter.

## Usage

Runs on NVIDIA H100, A100, RTX Pro 6000 and B200 GPUs.

vLLM:

```bash
vllm serve qda-lab/QDA-Gemma-4-26B-A4B-W4A16
```

SGLang:

```bash
python -m sglang.launch_server --model-path qda-lab/QDA-Gemma-4-26B-A4B-W4A16
```

## How it was made

The routed experts are 4-bit integers in groups of 64, quantized with GPTQ; all other linear layers are 8-bit.
MoE routers and the vision tower stay in BF16. The recipe was chosen by QDA's automated search.

## Reproduce

```bash
hf download qda-lab/qda-recipe-inputs --repo-type dataset --include "QDA-Gemma-4-26B-A4B-W4A16/*" --local-dir inputs
qda build inputs/QDA-Gemma-4-26B-A4B-W4A16/gemma-4-26b-a4b-w4a16.yaml
```

Needs the QDA library (the `qda` command). Recipe: [`gemma-4-26b-a4b-w4a16.yaml`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/blob/main/QDA-Gemma-4-26B-A4B-W4A16/gemma-4-26b-a4b-w4a16.yaml); inputs
(calibration sample and distillation data): [`QDA-Gemma-4-26B-A4B-W4A16/`](https://huggingface.co/datasets/qda-lab/qda-recipe-inputs/tree/main/QDA-Gemma-4-26B-A4B-W4A16) in `qda-lab/qda-recipe-inputs`.
Takes about 11 min on 1× H100 (measured). The rebuild matches this checkpoint's KL to BF16 within 5 %; weights are not bit-identical.

## License

Apache-2.0, the same as the base model ([Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license)).

## Citation

TBD.
