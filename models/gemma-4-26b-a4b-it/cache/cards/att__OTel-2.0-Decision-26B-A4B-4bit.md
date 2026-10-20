---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
language:
- en
base_model:
- google/gemma-4-26B-A4B-it
base_model_relation: quantized
tags:
- telecom
- telecommunications
- open-telco-ai
- classification
- typed-decisions
- decision-engine
- gemma4
- mixture-of-experts
- awq
- w4a16
- compressed-tensors
- 4-bit
- amd
- rocm
- vllm
pipeline_tag: text-generation
---

# OTel-2.0-Decision-26B-A4B-4bit

> **Versioning notice:** This card describes the current OTel-2.0-Decision-26B-A4B-4bit checkpoint. The model weights may be updated over time. For reproducible evaluation or production deployment, pin a specific model revision, checkpoint hash, or release tag rather than relying on the floating latest version.

Details on the training data, training method and configuration, data-processing pipeline, and training infrastructure will be added soon.

**OTel-2.0-Decision-26B-A4B-4bit** is a 4-bit W4A16 build of Google's Gemma 4 26B-A4B-it for typed decisions: classification, Yes/No questions, and ordered levels. Questions are supplied at request time, and decisions are read from answer-label probabilities rather than generated prose.

## Intended Use

Classify messages, call transcripts, tickets, and documents, or ask multiple questions about the same text. Use label probabilities for confidence thresholds and routing. This release is not intended for open-ended chat, retrieval, or reranking.

## Evaluation

Benchmarks use the typed-decision readout (`level=L0`, no prior, shared-prefix execution, adaptive Yes/No ordering with margin 4.0). All runs reported zero errors and `answer_mass` of 1.00.

| Benchmark | AMD MI350P | AMD Ryzen AI MAX+ PRO 395 (Strix Halo) |
|---|---:|---:|
| JevBench public (231 tasks) | **86.58–87.45%** (200–202/231) | **87.45%** (202/231) |

JevBench by tier: 100% on easy, 94.44–95.83% on original, and 74.77–76.58% on hard tasks. The MoE kernel is not bit-deterministic, so results vary on a few low-margin tasks across runs. The Strix Halo run used vLLM 0.26.0 on ROCm 7.13 (48/48 easy, 71/72 original, 83/111 hard).

Use a system message that asks for a single label (see *Prompting and Label Probabilities*). Without one, the model sometimes answers hard tasks in prose instead of a label, which lowers hard-tier accuracy by about 6 points.

## Inference Recipe

Serve on one AMD MI350P with vLLM 0.30.0. Both options use the same weights and API.

| Option | Image | Throughput |
|---|---|---:|
| 1. Official vLLM | `vllm/vllm-openai-rocm:v0.30.0` | 22.7k tok/s |
| 2. Prebuilt image | `farbodatdocker/otel-decision-vllm-rocm:v0.30.0` | 32.3k tok/s (42% more) |

Throughput figures are peak aggregate output tokens per second. The highest throughput is reached only when the GPU runs near maximum utilization, with many concurrent requests in flight. A single request runs much slower, so batch or parallelize requests to reach these numbers.

### Download

```bash
pip install -U huggingface_hub
hf download att/OTel-2.0-Decision-26B-A4B-4bit \
    --local-dir ./OTel-2.0-Decision-26B-A4B-4bit
```

### Option 1: Official vLLM

```bash
docker run --rm --name otel_decision \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc=host --security-opt seccomp=unconfined -p 8000:8000 \
  -v "$PWD/OTel-2.0-Decision-26B-A4B-4bit:/model:ro" \
  vllm/vllm-openai-rocm:v0.30.0 \
    /model \
    --served-model-name OTel-2.0-Decision-26B-A4B-4bit \
    --max-model-len 16384 \
    --enable-prefix-caching \
    --max-logprobs 64 \
    --port 8000
```

With vLLM 0.30.0 already installed, the equivalent is:

```bash
vllm serve ./OTel-2.0-Decision-26B-A4B-4bit \
  --served-model-name OTel-2.0-Decision-26B-A4B-4bit \
  --max-model-len 16384 --enable-prefix-caching --max-logprobs 64 --port 8000
```

### Option 2: Prebuilt Image

The official image with a corrected Gemma 4 MoE kernel and MI350P tile configuration.

```bash
docker run --rm --name otel_decision \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc=host --security-opt seccomp=unconfined -p 8000:8000 \
  -v "$PWD/OTel-2.0-Decision-26B-A4B-4bit:/model:ro" \
  farbodatdocker/otel-decision-vllm-rocm:v0.30.0 \
    /model \
    --served-model-name OTel-2.0-Decision-26B-A4B-4bit \
    --max-model-len 16384 \
    --enable-prefix-caching \
    --max-logprobs 64 \
    --moe-backend flydsl \
    --port 8000
```

Use `--moe-backend flydsl` only with Option 2. The first start takes 4 to 10 minutes.

### AMD Ryzen AI MAX+ PRO 395 (Strix Halo): vLLM on ROCm

Serve on one AMD Ryzen AI MAX+ PRO 395 (Strix Halo: Radeon 8060S GPU, 128 GB unified memory, configured as 96 GB GPU and 32 GB NPU) with vLLM 0.26.0 and ROCm 7.13 for gfx1151.

```bash
# 1. ROCm 7.13 + vLLM for gfx1151 (managed by the ROCm CLI)
rocm engines install vllm
PY=~/.rocm/runtimes/wheel/release-wheel-gfx1151-7-13-0/bin/python

# 2. Config shim: transformers 5.19 drops Gemma 4's global head sizes that vLLM 0.26 reads
mkdir -p shim && cat > shim/sitecustomize.py <<'EOF'
from transformers import PretrainedConfig
from transformers.models.gemma4 import configuration_gemma4 as g4
PretrainedConfig.allow_global_per_layer_attribute_access = True
_init = g4.Gemma4TextConfig.__init__
def init(self, *a, **k):
    _init(self, *a, **k)
    lt, pl = list(getattr(self, "layer_types", []) or []), getattr(self, "per_layer_config", None)
    if pl is not None and "full_attention" in lt:
        full = pl[lt.index("full_attention")]
        for dst, src in (("global_head_dim", "head_dim"), ("num_global_key_value_heads", "num_key_value_heads")):
            if dst not in self.__dict__ and getattr(full, src, None) is not None:
                object.__setattr__(self, dst, getattr(full, src))
g4.Gemma4TextConfig.__init__ = init
EOF

# 3. Serve (the torch wheel links OpenMPI 4's libmpi_cxx.so.40; point at its lib dir)
LD_LIBRARY_PATH=/path/to/openmpi4/lib PYTHONPATH=$PWD/shim \
$PY -m vllm.entrypoints.cli.main serve ./OTel-2.0-Decision-26B-A4B-4bit \
  --served-model-name OTel-2.0-Decision-26B-A4B-4bit \
  --max-model-len 16384 --enable-prefix-caching --max-logprobs 64 \
  --gpu-memory-utilization 0.55 --max-num-seqs 256 --port 8000
```

| Measured (AC power, performance profile) | Value |
|---|---:|
| Peak aggregate output, 256 concurrent requests | 377 tok/s |
| 64 concurrent requests | 257 tok/s |
| Single request | 30 tok/s |
| JevBench public, 8 concurrent | 1.27 tasks/s |

Peak throughput again needs near-maximum utilization: aggregate output keeps rising up to 256 concurrent requests. Keep HIP graphs on (the default) — `--enforce-eager` cuts single-request speed to about 19 tok/s. The shim is not needed with vLLM releases that read Gemma 4's per-layer config directly. torch fails to import without an OpenMPI 4.x runtime. Distributions that ship only OpenMPI 5 (for example Ubuntu 26.04) lack `libmpi_cxx.so.40`, so extract an OpenMPI 4 package into a local directory and use that directory; no `sudo` is needed.

### Prompting and Label Probabilities

Put the text first, then the question and labels. Use letters (`A`, `B`, …) for options or Yes/No, and numbers for ordered levels. Add a system message that asks for exactly one label, and request only the label through the chat endpoint, which applies the repository's template with thinking disabled.

```python
import math
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
question = """State:
Order #1182. Status: shipped on 3 September. Carrier: DHL.

Question: Has the order been shipped? Answer strictly from the facts stated in the text.
Options:
A. Yes
B. No
Answer with the letter only."""
labels = ["A", "B"]

r = client.chat.completions.create(
    model="OTel-2.0-Decision-26B-A4B-4bit",
    temperature=0,
    max_tokens=1,
    logprobs=True,
    top_logprobs=64,
    messages=[
        {"role": "system", "content": "You are a typed decision engine. Reply with exactly one answer label "
                                      "(a single letter or number) and nothing else. Do not explain."},
        {"role": "user", "content": question},
    ],
)
top = {t.token: t.logprob for t in r.choices[0].logprobs.content[0].top_logprobs}
probs = {label: math.exp(top[label]) if label in top else 0.0 for label in labels}
print("probabilities:", probs)
print("decision:", max(probs, key=probs.get))
print("answer_mass:", sum(probs.values()))
# Expected: decision A, answer_mass approximately 1.0
```

Choose the label with the highest probability. `answer_mass` is the summed probability of the answer labels and should be close to 1.0; a low value indicates an invalid decision readout. For multiple questions, keep the shared text byte-identical so prefix caching can reuse it.

### Verify the Deployment

Check that `http://localhost:8000/v1/models` lists the branded model name, then run the example above and confirm decision `A` with `answer_mass` near 1.0. Validate representative prompts against a reference run before deployment.

## Limitations and Responsible Use

Evaluation covers English text only. JevBench is small and does not guarantee production accuracy. Validate on your own human-labeled data, and do not assume label probabilities are calibrated for your workload. Keep human review for consequential decisions.

The recipe starts an unauthenticated server. Add authentication before exposing it beyond your machine.

## Resources

- [Project Page](https://huggingface.co/farbodtavakkoli)
- [OTel code](https://github.com/farbodtavakkoli/OTel)
- [OTel 1.0 media coverage list](https://github.com/farbodtavakkoli/OTel/blob/main/docs/OTel-1.0-media-coverage.md)
- [OTel 2.0 blogs](https://github.com/farbodtavakkoli/OTel/blob/main/docs/OTel-2.0-blogs.md)

## Citation

```bibtex
@misc{otel_2_models_2026,
  title  = {OTel 2.0: Open Telco AI Datasets, Benchmarks, and Models},
  author = {Austin, Mark and Tavakkoli, Farbod},
  year   = {2026},
  note   = {Open Telco (OTel 2.0) model release},
  url    = {https://huggingface.co/farbodtavakkoli}
}
```

## Contact

For technical questions, contact mark.austin@att.com, farbod.tavakkoli@att.com, or farbodtavakoli@gmail.com.