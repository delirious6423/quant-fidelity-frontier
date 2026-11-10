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
<h1 align="center">Valen-Decision-1.5-27B-GGUF</h1>

**Valen-Decision-1.5-27B-GGUF** helps you classify customer messages, check whether images meet your requirements, and identify actions in videos. Try Valen-Decision-1.5 instantly through our **[free API and web demo](https://valen-team.github.io/)** (0.8B, 2B and 4B), with no installation, model download, or GPU required. We also release the **[training code](https://github.com/Valen-Team/Valen)**, so you can fine-tune Valen-Decision on your own data. The training data of Valen-Decision-1.5 is not public.



## GGUF files and usage

This repository holds [Valen-Decision-1.5-27B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-27B) converted for the Valen custom llama.cpp runtime **0.2.0** ([release](https://github.com/Valen-Team/Valen/releases/tag/gguf-runtime-v0.2.0)). The runtime serves text and image requests with `choice`, `score`, `noul` and `bbox` questions through `/v1/systemone` and `/v1/decisions`. The Q8_0 text file and the matching vision file are provided.

| File | Use | Size |
| --- | --- | ---: |
| `Valen-Decision-1.5-27B-Q8_0.gguf` | Text file; start here | 25.43 GiB |
| `vision-native-limits-F32.gguf` | Vision file for image and bbox requests | 1.72 GiB |

Text-only serving needs just the text file. SHA256 values are in [`SHA256SUMS`](SHA256SUMS) and [`MANIFEST.json`](MANIFEST.json).

### Serve

Get the Linux x86-64 CPU runtime (Ubuntu 22.04+, glibc 2.35+, AVX2/FMA/F16C/BMI2):

```sh
base=https://github.com/Valen-Team/Valen/releases/download/gguf-runtime-v0.2.0
curl -fLO "$base/valen-gguf-runtime-0.2.0-linux-x86_64.tar.gz"
curl -fLO "$base/SHA256SUMS"
sha256sum -c SHA256SUMS
tar -xzf valen-gguf-runtime-0.2.0-linux-x86_64.tar.gz
cd valen-gguf-runtime-0.2.0-linux-x86_64
```

Download the Q8_0 text file and the vision file, then start the server:

```sh
hf download Valen-Team/Valen-Decision-1.5-27B-GGUF Valen-Decision-1.5-27B-Q8_0.gguf vision-native-limits-F32.gguf --local-dir .
./serve -m "$PWD/Valen-Decision-1.5-27B-Q8_0.gguf" --mmproj "$PWD/vision-native-limits-F32.gguf" \
  --load-mode none --device none -ngl 0 \
  --host 127.0.0.1 --port 8080 -c 4096 -b 4096 -ub 4096 \
  -t 4 --parallel 1 --no-warmup
```

In another terminal, send a text request:

```sh
curl --fail http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":{"messages":[{"role":"user","content":"The film was excellent."}]},"questions":{"sentiment":{"type":"choice","instructions":"Classify the sentiment.","criteria":{"positive":"Positive","negative":"Negative"}}}}'
```

The response holds candidate probabilities for each question. The whole request is evaluated in one pass; raise `-c`, `-b` and `-ub` together for larger requests. `/v1/decisions` takes the native Decisions schema and returns the same numbers. Image and bbox requests, conversion from the native weights and source provenance are in the [runtime guide](RUNTIME_GUIDE.md).

### Comparison with the native model

The runtime 0.2.0 `llama-server` answered six requests (text, image and four bbox requests) on Linux x86-64 CPU, compared with native `model.predict` in FP32. Token counts were equal for every request. Maximum absolute differences, from `MANIFEST.json`; the release gate is a probability difference of at most 0.08:

| File | Probability | Box | Bbox confidence |
| --- | ---: | ---: | ---: |
| Q8_0 | 0.00252 | 0.0553 | 0.0113 |

The [GGUF guide](https://valen-team.github.io/docs/valen/gguf/) covers all sizes, the runtime release and bbox requests.

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
  --attention-backend fa3
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
