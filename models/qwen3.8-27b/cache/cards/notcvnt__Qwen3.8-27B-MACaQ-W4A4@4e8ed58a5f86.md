---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - macaq
  - mxfp4
  - w4a4
  - fp4
  - vllm
  - blackwell
  - qwen3.8
---

# Qwen3.8-27B-MACaQ-W4A4

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) quantized with
[MACaQ](https://github.com/huawei-csl/MACaQ) ([arXiv 2611.10649](https://arxiv.org/abs/2611.10649))
to MXFP4 weights **and** activations (W4A4). Text only: the vision tower is not included.

This is an independent community quantization made to check the MACaQ claims on
Qwen3.8-27B. It is not an official Huawei or Qwen release.

- Size: 16.8 GiB (BF16 original: 52 GB)
- Quantized: every linear layer of the language model except the small
  `linear_attn.in_proj_a/b` projections. Embeddings, `lm_head` and norms stay in BF16.
- Needs MACaQ's vLLM plugin and **vLLM 0.20.2**. Standard vLLM, llama.cpp, LM Studio and
  Ollama cannot load this checkpoint.
- Speedup needs native FP4 tensor cores, i.e. an NVIDIA Blackwell GPU
  (RTX 50xx, RTX PRO 6000, B200). A single 32k-token request should need roughly 24 GB of
  VRAM: 17.8 GiB of weights plus KV cache and runtime overhead. This is an estimate from the
  vLLM memory log; 32k context was not run.

## Results

One RTX PRO 6000 Blackwell Server Edition (96 GB), 2026-11-18. Thinking disabled, greedy
decoding. Unsloth's NVFP4 checkpoint is included as a ready-made FP4 alternative.

| | BF16 | MACaQ W4A4 | MACaQ W4A4 + FP8 KV | [Unsloth NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) |
|---|---:|---:|---:|---:|
| Weights in VRAM | 50.2 GiB | 17.8 GiB | 17.8 GiB | 20.4 GiB |
| KV cache tokens at 90% of 96 GB | 303k | 606k | 901k | 873k |
| WikiText-2 test PPL, ctx 2048 | 6.726 | 6.943 (+3.2%) | 6.951 (+3.3%) | 6.891 (+2.5%) |
| ≈KLD vs BF16, mean (top-50) | — | 0.0575 | 0.0575 | 0.0298 |
| ≈KLD vs BF16, p99 | — | 0.613 | 0.611 | 0.293 |
| Top-1 token agreement with BF16 | — | 89.6% | 89.7% | 92.1% |
| GSM8K (1319, 0-shot) | 96.1% | 95.8% | 96.2% | 95.5% |
| IFEval prompt strict (541) | 80.6% | 83.0% | 83.7% | 82.1% |
| IFEval instruction strict | 86.6% | 88.1% | 88.5% | 87.3% |
| Prefill, batch 1, 2048 tokens | 6 825 tok/s | 14 007 (2.05×) | 13 998 (2.05×) | 14 394 (2.11×) |
| Decode, batch 1 | 26.8 tok/s | 64.9 (2.42×) | 65.1 (2.43×) | 60.5 (2.26×) |
| Decode, batch 32 | 668 tok/s | 1 134 (1.70×) | 999 (1.50×) | 1 330 (1.99×) |

For comparison, the MACaQ README reports 2.33× prefill, 2.58× decode and +1.9% perplexity
for this model.

Read the table with these limits in mind:

- GSM8K and IFEval differences between all four columns are within noise (standard error
  about 0.5 points for GSM8K and 1.7 points for IFEval prompt-level). The quantized models
  do not follow instructions better than the original.
- KLD is approximate: top-50 logprobs from vLLM plus one bucket for the remaining mass,
  over 16 × 2048 WikiText-2 positions. It is not the full-vocabulary KLD that llama.cpp
  reports, so do not compare it with llama.cpp KLD numbers. The same method was used for
  every column.
- Unsloth NVFP4 is mixed precision: attention, `lm_head` and the MLPs of the last eight
  layers are FP8, and its FP8 KV cache is calibrated. The FP8 KV cache used with MACaQ here
  is uncalibrated (`kv_cache_dtype="fp8"`), and quality was checked at 2048-token context only.
- MACaQ and BF16 ran on vLLM 0.20.2. Unsloth NVFP4 needs a newer vLLM because its FP8
  `lm_head` does not load in 0.20.2, so it ran on vLLM 0.33.0. BF16 speed was the same on
  both versions (prefill 6 817 / 6 825 tok/s, decode 26.6 / 26.8 tok/s).

## Usage

Install MACaQ's vLLM environment (Linux, Python 3.12), following its
[vLLM guide](https://github.com/huawei-csl/MACaQ/blob/main/vllm/README.md):

```bash
git clone https://github.com/huawei-csl/MACaQ && cd MACaQ
git checkout a8c6c4756a47aaf05eadddbfb04249a128bc483b
python3.12 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install -r requirements/vllm-qwen35.txt -r requirements/build.txt
.venv-vllm/bin/python -m pip install --no-build-isolation --no-deps -e .
hf download notcvnt/Qwen3.8-27B-MACaQ-W4A4 --local-dir models/qwen38-27b-macaq
```

Serve through MACaQ's CLI. It handles one sequence at a time and defaults to a 2048-token
context; `--context` and `--kv-cache-mib` raise it (the values below were not tested):

```bash
.venv-vllm/bin/python vllm/cli.py serve --model models/qwen38-27b-macaq --gpu 0 \
  --context 32768 --kv-cache-mib 8192
```

For batched offline inference, load it with vLLM directly. These are the settings used
for the benchmark above:

```python
import os
from macaq_env import setup_cuda_env
from macaq_plugin import register
from vllm import LLM, SamplingParams

setup_cuda_env()
register()
os.environ["MACAQ_VERIFY_WORKER"] = "1"

llm = LLM(
    model="models/qwen38-27b-macaq",
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=64,
    language_model_only=True,
    kv_cache_dtype="auto",  # "fp8" for the FP8 KV column
    async_scheduling=False,
    kernel_config={"ir_op_priority": {"rms_norm": ["vllm_c"]}},
    compilation_config={"custom_ops": ["all"]},
    worker_extension_cls="macaq_validation.DenseValidationWorkerExtension",
)
tok = llm.get_tokenizer()
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "What is 17 plus 26?"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
print(llm.generate([prompt], SamplingParams(max_tokens=256, temperature=0))[0].outputs[0].text)
```

Run the Python snippet from the MACaQ repository root with `vllm/` on `PYTHONPATH`.

## How it was made

- Source: `Qwen/Qwen3.8-27B` at revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.
- MACaQ commit `a8c6c4756a47aaf05eadddbfb04249a128bc483b`, `vllm/cli.py quantize` with
  default settings: 16 384 calibration tokens in 512-token chunks, 512 flip-refinement
  iterations, W4A4.
- Calibration text: WikiText-2 raw train split. Perplexity was measured on the test split.
- Quantization took 60.4 minutes on one RTX PRO 6000. `flash-linear-attention` was not
  installed, so the linear-attention layers ran through the slower PyTorch path during
  calibration.
- `macaq_run.json` and `macaq_source.json` in this repository record the per-layer settings
  and the SHA-256 of every source file.
- The full benchmark script is in `bench/bench.py` (Modal).

## License and attribution

Apache-2.0, the same as the base model and MACaQ. Base model by the Qwen team; quantization
method and tooling by Huawei Computing Systems Lab.

```bibtex
@misc{muller2026macaq,
  title         = {{MACaQ}: Fast, Model Agnostic Calibrated Quantization for Low-Precision Weights and Activations},
  author        = {Muller, Lorenz K. and Bich, Philippe and Boretti, Chiara and Chang, Hyun-Min and Zhuang, Jiawei and Cavigelli, Lukas},
  year          = {2026},
  eprint        = {2611.10649},
  archivePrefix = {arXiv}
}
```
