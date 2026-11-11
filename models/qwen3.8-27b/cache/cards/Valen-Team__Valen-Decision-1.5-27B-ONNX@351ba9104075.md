---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: transformers
tags:
- valen
- image-text-to-text
- decision-making
- decision-model
- image-text-to-typed-output
- structured-output
- shared-state
- qwen3.8
- custom-code
---

<p align="center">
  <img src="https://raw.githubusercontent.com/Valen-Team/Valen/main/assets/branding/valen-mark-white.png" alt="Valen logo" width="112">
</p>
<h1 align="center">Valen-Decision-1.5-27B-ONNX</h1>

**Valen-Decision-1.5-27B-ONNX** helps you classify customer messages, check whether images meet your requirements, and identify actions in videos. Try Valen-Decision-1.5 instantly through our **[free API and web demo](https://valen-team.github.io/)** (0.8B, 2B and 4B), with no installation, model download, or GPU required. We also release the **[training code](https://github.com/Valen-Team/Valen)**, so you can fine-tune Valen-Decision on your own data. The training data of Valen-Decision-1.5 is not public.



## ONNX files and usage

This repository holds the ONNX Runtime export of [Valen-Decision-1.5-27B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-27B): backbone, embedding, vision, decision and grounding graphs under `onnx/`, listed with their SHA256 in [`valen_onnx.json`](valen_onnx.json). Load it with the Valen [Python SDK](https://github.com/valen-team/valen-sdk-python) or [JavaScript SDK](https://github.com/valen-team/valen-sdk-js); inference runs on ONNX Runtime without PyTorch or Transformers. The SDK downloads only the variant you select and reuses the cache on later runs.

### Variants

| Variant | Use | Precision | Download |
| --- | --- | --- | ---: |
| `fp32` | CPU default | FP32 throughout | 104.49 GB |
| `q8-mlp` | Smaller CPU download | 8-bit weight-only MLP layers; FP32 attention, embedding, vision and heads | 55.29 GB |
| `cuda-fp16` | NVIDIA GPU through `CUDAExecutionProvider` | FP16 backbone and embedding; FP32 vision and heads | 53.24 GB |

Download sizes add up the files each variant lists in `valen_onnx.json`, including the vision graph.

### Python

```bash
python -m pip install "valen-sdk[onnx]==0.5.1"
```

```python
from valen_sdk import Valen

with Valen.local("Valen-Team/Valen-Decision-1.5-27B-ONNX", variant="fp32") as client:
    result = client.decisions(input="Please send my order to my new address.", questions=[
        {"name": "change_address", "type": "predicate",
         "instructions": "Is the customer requesting a delivery address change?"},
    ])
    print(result.by_name("change_address").probability)
```

Use `variant="q8-mlp"` for the smaller CPU download.

For an NVIDIA GPU, install the CUDA extra in a separate environment and select `cuda-fp16`:

```bash
python -m pip install "valen-sdk[cuda]==0.5.1"
```

```python
with Valen.local(
    "Valen-Team/Valen-Decision-1.5-27B-ONNX", variant="cuda-fp16",
    providers=[("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"],
) as client:
    ...
```

### JavaScript / Node.js

```bash
npm install @valen-team/sdk@0.4.0
npm install @huggingface/tokenizers@0.2.0 @huggingface/jinja@0.5.10 @huggingface/transformers@4.3.1 onnxruntime-node@1.30.0
```

```js
import { Valen } from '@valen-team/sdk';

const client = await Valen.local('Valen-Team/Valen-Decision-1.5-27B-ONNX', { variant: 'fp32' });
try {
  const result = await client.decisions({
    input: 'Please send my order to my new address.',
    questions: [{ name: 'change_address', type: 'predicate',
      instructions: 'Is the customer requesting a delivery address change?' }],
  });
  console.log(result.answers[0].probability);
} finally {
  await client.close();
}
```

### Bbox questions

A `bbox` question locates what its instructions describe in an image and returns `bbox` `[x1, y1, x2, y2]`, `point` `[x, y]` and `confidence`, normalized to 0–1. The grounding graph answers it in the same request as predicate, choice and score questions: `{"name": "mug", "type": "bbox", "instructions": "The blue mug.", "image": 0}`. See [bbox questions](https://valen-team.github.io/docs/valen/onnx/#bbox-questions) for a full example and measured output.

### Comparison with the native model

Each variant was compared with native Valen-Decision-1.5-27B run in FP32 on CPU, from [`validation.json`](validation.json):

| Variant | Device | Requests | Result |
| --- | --- | --- | --- |
| `fp32` | CPU | Text, image and bbox requests | Max difference: text 1.3e-06, image 2.7e-07, bbox 2.6e-07 |
| `q8-mlp` | CPU | Text, image and bbox requests | Max difference: text 0.00058, image 1.1e-05, bbox 0.0006 |
| `cuda-fp16` | NVIDIA H200, CUDA | Text, image and bbox requests, 44 numbers | Decision agreement 100.00%; probability MAE 3e-05, max 0.00016; box max 0.0001 |

The [ONNX guide](https://valen-team.github.io/docs/valen/onnx/) covers all sizes, variants, caching, offline use and options.

## Complete application tutorials

Complete workflows are maintained in the [official Cookbook](https://valen-team.github.io/cookbook/): [API basics](https://valen-team.github.io/cookbook/api-quickstart/), [local inference basics](https://valen-team.github.io/cookbook/local-inference/), [ticket routing](https://valen-team.github.io/cookbook/ticket-routing/), [product-image screening](https://valen-team.github.io/cookbook/product-screening/) and [video tagging](https://valen-team.github.io/cookbook/video-tagging/). Pages import executable source and include actual outputs, uncertainty policies, media provenance and licensing notes.

Valen-Decision-1.5 adds block-bidirectional attention and a bbox grounding head: one forward pass returns probabilities for predicate, choice and score questions and a box for each bbox question.


## Which version should I choose?

For a first try, use the [web experience or free HTTP API](https://valen-team.github.io/), which serves Valen-Decision-1.5 0.8B, 2B and 4B without downloading weights.

Valen-Decision-1.5 comes in five sizes: [0.8B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-0.8B), [2B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-2B), [4B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-4B), [9B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-9B) and [27B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-27B). Larger sizes score higher on Decision Index (see [Evaluation](#evaluation)) and need more memory and compute. Each size is published in three formats: native weights for Transformers and fine-tuning ([Valen-Decision-1.5-27B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-27B)), ONNX ([Valen-Decision-1.5-27B-ONNX](https://huggingface.co/Valen-Team/Valen-Decision-1.5-27B-ONNX)) and GGUF for the Valen llama.cpp runtime 0.2.0 ([Valen-Decision-1.5-27B-GGUF](https://huggingface.co/Valen-Team/Valen-Decision-1.5-27B-GGUF)). The native weights store the backbone in BF16 and the decision and grounding heads in FP32.


## Examples

Image and video decision demonstration of Valen-Decision-1.5 from the [Valen repository](https://github.com/Valen-Team/Valen#demos): boxes, probabilities, choices and scores from one forward pass. Click the preview to watch the full demonstration.

<p align="center">
  <a href="https://raw.githubusercontent.com/Valen-Team/Valen/main/assets/demos/valen-decision-1.5-demo.mp4"><img src="https://raw.githubusercontent.com/Valen-Team/Valen/main/assets/demos/valen-decision-1.5-demo.gif" alt="Valen-Decision-1.5 demo: boxes, probabilities, choices and scores from one forward pass" width="85%"></a>
</p>

## Quick start

Valen supports two decision API formats: **SystemOne**, used by [Jev](https://docs.typesafe.ai/api/system-one), and **Decisions**, used by [OpenAI](https://developers.openai.com/api/docs/guides/decisions). Both work with the same Valen model.

Start with **Decisions** if you are new to Valen. If your application already uses the Jev/SystemOne format, use **SystemOne**. The examples below include everything you need to get started.

See the [Valen Transformers documentation](https://valen-team.github.io/docs/valen/quickstart/) for complete request formats, image/video inputs, and inference parameters.

### Install and load

The Python example below runs locally on an NVIDIA GPU. If you prefer to skip local setup, try the **[free web demo or API](https://valen-team.github.io/)**.

```bash
python -m pip install torch torchvision transformers pillow av
```

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Valen-Team/Valen-Decision-1.5-27B", trust_remote_code=True,
    dtype="auto", attn_implementation="sdpa",
).to("cuda").eval()
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False
```

### Minimal protocol calls

After installation/loading above, verify the path with one predicate. Complete multi-question workflows and image/video tutorials are maintained in Cookbook.

```python
result = model.predict({"input": "A cat is on the sofa.", "questions": [
    {"name": "cat", "type": "predicate", "instructions": "A cat is present."},
]})
print(result["answers"][0]["probability"])
legacy = model.predict({"state": "A cat is on the sofa.", "questions": {
    "cat": {"type": "noul", "instructions": "A cat is present."},
}})
print(legacy["answers"]["cat"]["noul"])
```

[Cookbook](https://valen-team.github.io/cookbook/) · [API](https://valen-team.github.io/docs/valen/api/)

## Try it free: the API

Try the [interactive demo](https://valen-team.github.io/) or call the free API directly. No API key is required. The free API serves Valen-Decision-1.5 under the model IDs `valen-0.8b`, `valen-2b` and `valen-4b`. 27B is not on the free API, so the example uses `valen-4b` (Valen-Decision-1.5-4B).

```bash
curl https://api.valen-team.workers.dev/v1/decisions \
  -H 'Content-Type: application/json' \
  -H 'Origin: https://valen-team.github.io' \
  -d '{
    "model": "valen-4b",
    "input": "Please send my order to my new address.",
    "questions": [{"name": "change_address", "type": "predicate",
                   "instructions": "Is the customer requesting a delivery address change?"}]
  }'
```

Example response (captured from the free API; timing may vary):

```json
{
  "answers": [
    {
      "type": "predicate",
      "name": "change_address",
      "probability": 0.9985537506636069
    }
  ],
  "model": "Valen-Decision-1.5-4B",
  "model_id": "valen-4b",
  "revision": "local",
  "usage": {
    "input_tokens": 79,
    "output_tokens": 0
  },
  "internal_usage": {
    "compute_tokens": 79
  },
  "timing": {
    "total_ms": 311.15,
    "image_ms": 0.0,
    "compile_ms": 5.62,
    "forward_ms": 304.32,
    "queue_ms": 0.01
  },
  "images": []
}
```

The predicate result is in `answers[0].probability`. See the [API reference](https://valen-team.github.io/docs/valen/api/) for image inputs, both protocols, and [available model IDs](https://valen-team.github.io/docs/valen/api/list-models/).

## SGLang / vLLM deployment

Install the corresponding Valen plugin in a compatible official engine environment. Use Linux, Python 3.11+, and an NVIDIA GPU; the SGLang example uses FA3 on H100/H200-class GPUs. The two alternatives below expose `/v1/systemone` and `/v1/decisions` for text and image requests.

### SGLang

```bash
python -m pip install "git+https://github.com/Valen-Team/Valen.git#subdirectory=serving/sglang"
valen-sglang --model-path Valen-Team/Valen-Decision-1.5-27B \
  --served-model-name valen --host 127.0.0.1 --port 30000 \
  --attention-backend fa3 --dtype bfloat16
```

### vLLM

```bash
python -m pip install "git+https://github.com/Valen-Team/Valen.git#subdirectory=serving/vllm"
valen-vllm --model-path Valen-Team/Valen-Decision-1.5-27B \
  --served-model-name valen --host 127.0.0.1 --port 8000
```

Once the chosen server is ready, send a local request. Use port `8000` for vLLM or `30000` for SGLang:

```bash
curl http://127.0.0.1:8000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen","input":"Please send my order to my new address.",
       "questions":[{"name":"change_address","type":"predicate",
                     "instructions":"Is the customer requesting a delivery address change?"}]}'
```

For Docker images, offline deployment kits, and complete setup instructions, see the [SGLang guide](https://valen-team.github.io/docs/valen/sglang/) and [vLLM guide](https://valen-team.github.io/docs/valen/vllm/).

## Fine-tuning

Use the complete native main revision for LoRA or full SFT, retaining the trained Mixer. Start with complete Valen-Decision-1.5-0.8B and LoRA for a small domain run; ONNX/GGUF are inference formats. Methods, data, resume and export are maintained in the [website training guide](https://valen-team.github.io/docs/valen/training/); the complete executable split/train/compare recipe lives in [Cookbook](https://valen-team.github.io/cookbook/domain-finetuning/). A short smoke does not establish domain improvement.

## Evaluation

![Valen-Decision-1.5-27B compared with other decision models on Decision Index and JevalKit benchmarks](assets/bench_chart.png)

### Jev Decision Index

[Decision Index 0.4.1](https://huggingface.co/spaces/multimodalart/jev-decision-index) public Text and Vision tracks, evaluated with [JevalKit](https://github.com/Valen-Team/JevalKit). These are normalized index scores (0–100), not accuracy percentages. Best scores are bold.

| Model | Public Text ↑ | Public Vision ↑ |
| --- | ---: | ---: |
| Valen-Decision-1.5-0.8B | 54.71 | 58.84 |
| Valen-Decision-1.5-2B | 61.89 | 70.54 |
| Valen-Decision-1.5-4B | 69.01 | 77.85 |
| Valen-Decision-1.5-9B | 70.61 | 79.93 |
| Valen-Decision-1.5-27B | **74.66** | **82.91** |

Per-size metrics, official scorer outputs and run manifests: [Valen-Decision-Index-Results](https://huggingface.co/datasets/Valen-Team/Valen-Decision-Index-Results/blob/main/Valen-Decision-1.5-main16k/RESULTS.md).

### JevalKit

[JevalKit benchmarks](https://github.com/Valen-Team/JevalKit). Values are accuracy (%), higher is better.

| Benchmark | Valen-Decision-1.5-27B |
| --- | ---: |
| JevBench | 91.77 |
| Typed Decisions | 79.40 |
| VisualDecisionBench — Image | 89.94 |
| VisualDecisionBench — Video | 89.40 |
| DecisionBench | 84.13 |
| Procedural Typed Decisions | 97.19 |
| JevAlt-Bench | 95.22 |
| zh-decision-bench | 91.65 |

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

If you use Valen-Decision in your work, please cite this repository:

```bibtex
@misc{valen,
  author       = {{Valen Team}},
  title        = {Valen-Decision},
  howpublished = {GitHub - Valen-Team/Valen: A multimodal System One decision model for text, images and video.},
  note         = {GitHub repository},
  year         = {2026},
  url          = {https://github.com/Valen-Team/Valen}
}
```

User concepts, methods and references are maintained only on the website: [Documentation](https://valen-team.github.io/docs/) · [Probabilities](https://valen-team.github.io/docs/valen/probabilities/).
