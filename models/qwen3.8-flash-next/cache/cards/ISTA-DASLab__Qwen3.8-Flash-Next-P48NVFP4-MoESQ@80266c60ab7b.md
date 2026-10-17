---
language:
- en
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- qwen4-exp
- moe
- nvfp4
- w4a4
- paired-4:8
- semi-structured-sparsity
- sparse-storage
- activation-quantization
- compressed-tensors
- blackwell
---

# Qwen3.8-Flash-Next-P48NVFP4-MoESQ

A **W4A4 + paired-4:8 sparse** compressed checkpoint of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), produced with **MoESQ**.
The routed MoE expert weights are NVFP4 with paired-4:8 structured sparsity and are **stored
sparse**: only the kept values plus a small mask are on disk. The target is NVIDIA Blackwell
(SM100 and SM120) sparse tensor cores.

- **Base model:** Qwen/Qwen3.8-Flash-Next (MoE, 512 routed experts with 10 active plus a shared
  expert, 48 layers: 36 linear-attention and 12 sparse-attention, hyper-connections, per-layer
  n-gram embedding)
- **Compression:** NVFP4 W4A4 plus paired-4:8 sparsity on the routed MoE experts
- **Routed experts: 225.0 GiB in BF16 → 38.7 GiB here, 5.8× smaller.** That is 2.75 bits per
  weight including the sparsity mask and scales (2 bits per weight for the kept values).
- **GPU-resident weights: 239.9 GiB in BF16 → 53.5 GiB, 4.5× smaller** (about 27 GiB per GPU at
  TP2). The per-layer n-gram embedding tables (95.4 GiB, BF16, unchanged) are lookup tables
  served from CPU memory, as in the base model.
- **Checkpoint size:** 149.0 GiB on disk, 64% of it the n-gram tables (see the breakdown below).
- **Kernel:** paired-4:8 sparse NVFP4 grouped GEMM (CUTLASS, SM100 and SM120) through vLLM's
  `paired48_nvfp4` MoE backend

**Links**
- Paper: [Hardware-Native Joint Sparse-Quantization for Trillion-Scale Mixture-of-Experts](https://arxiv.org/abs/2610.02241) (arXiv:2610.02241)
- Code: [IST-DASLab/MoESQ](https://github.com/IST-DASLab/MoESQ). It contains the compression code, the vLLM integration and the kernels, and becomes public together with the paper.
- Other MoESQ checkpoints: [ISTA-DASLab/moesq collection](https://huggingface.co/collections/ISTA-DASLab/moesq)

## Usage

This checkpoint does **not** load in upstream vLLM. The MoESQ repository installs a patched
vLLM v0.30.0 (the `paired48_nvfp4` MoE backend and sparse-storage loader) together with the
kernels. It needs an NVIDIA Blackwell SM100 (B200, GB200) or SM120 (RTX 5090, RTX PRO 6000) GPU and a
CUDA toolkit >= 12.8; SM103 (B300) and SM121 (DGX Spark) are not supported.

```bash
git clone --recurse-submodules https://github.com/IST-DASLab/MoESQ.git && cd MoESQ
bash integrations/vllm/install.sh && source .venv-vllm/bin/activate

vllm serve ISTA-DASLab/Qwen3.8-Flash-Next-P48NVFP4-MoESQ \
  --tensor-parallel-size 2 --enable-expert-parallel   # 2x B200, ~27 GiB of weights per GPU
```

vLLM selects the backend automatically. The model is chat-only: use the chat endpoint (plain
few-shot completions end immediately, for the BF16 model too). Pass
`--default-chat-template-kwargs '{"enable_thinking": false}'` to serve without thinking.

### Recommended sampling

Use the base model's thinking-mode setting: `temperature=1.0`, `top_p=0.95`, `top_k=20`.

**For long single-turn reasoning (competitive programming, math), also set
`presence_penalty=1.5`.** On very long reasoning traces this model is more likely than the BF16
model to fall into repetition loops ("maybe maybe maybe …"), which run until the token budget is
exhausted. The base model card allows a presence penalty of 0–2 to reduce endless repetition.
On LiveCodeBench v6 at 65,536 new tokens, the penalty lifts this model from 43.09 to 49.03 pass@1;
on one seed it moved the BF16 model by only +1.1. We saw no language mixing at 1.5.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="ISTA-DASLab/Qwen3.8-Flash-Next-P48NVFP4-MoESQ",
    messages=[{"role": "user", "content": "..."}],
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,  # long single-turn reasoning only; keep 0 for agentic tool calling
    max_tokens=131072,
    extra_body={"top_k": 20},
)
```

**For agentic tool calling, keep `presence_penalty=0`.** vLLM applies the penalty to the tokens
already generated in the current response. After the model reasons about a command, the tool
call's own tokens get penalized: on SWE-bench Verified, tool calls without their required
argument nearly tripled, and resolution dropped (70.6% vs 74.7% on the same 170 instances).

## Evaluation

### OpenLLM Leaderboard v1: 6-task average

ARC-Challenge (25-shot), HellaSwag (10-shot), MMLU (5-shot), TruthfulQA-MC2 (0-shot) and
Winogrande (5-shot) are scored with lm-evaluation-harness on `/v1/completions`. GSM8K (5-shot,
greedy, strict match) runs through the chat template with the few-shot examples as turns and
thinking off, because the model is chat-only. Mean ± sd over few-shot seeds 1234, 0 and 1;
both rows served with the vLLM above (BF16 at TP4, this model at TP2 + EP, KV cache bf16).

| Model | ARC-C | GSM8K | HellaSwag | MMLU | TQA-MC2 | Winogrande | Avg | Recovery |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next (BF16) | 78.90 | 96.92 | 88.81 | 86.71 | 67.66 | 73.56 | 82.09 ± 0.28 | — |
| **This model (MoESQ)** | 74.74 | 94.34 | 83.54 | 84.56 | 61.84 | 75.16 | **79.03 ± 0.15** | **96.3 %** |

### Coding: LiveCodeBench v6 and SWE-bench Verified

Thinking on, `temperature=1.0`, `top_p=0.95`, `top_k=20` for both models. This model adds
`presence_penalty=1.5` on LiveCodeBench (see [Recommended sampling](#recommended-sampling));
the BF16 model runs at its card's default of 0. Both models are served with the vLLM above (BF16
at TP4, this model at TP2 + EP, KV cache bf16, context 131,072, or 147,456 for the 131,072-token
LiveCodeBench run).

| Model | LCB-v6, 65,536 new tokens | LCB-v6, 131,072 new tokens | SWE-bench Verified | Recovery (LCB 131k / SWE) |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (BF16) | 55.26 ± 0.97 | 57.03 ± 1.26 | 79.8 | — |
| **This model (MoESQ)** | 49.03 ± 1.34 | **53.54 ± 1.45** | **67.0** | **93.9 % / 84.0 %** |

- **LiveCodeBench v6:** 175 problems (`lcb:codegeneration_v6` in lighteval), 0-shot, pass@1,
  mean ± sd over 10 seeds. An answer still inside its reasoning when the token budget runs out
  counts as wrong. At 131,072 tokens that is 19.8% of this model's answers and 1.4% of BF16's.
  Without the presence penalty, this model scores 43.09 ± 1.33 at 65,536 tokens.
- **SWE-bench Verified:** 500 instances, one run per model. The agent is mini-swe-agent 2.4.6
  with native tool calls (one bash tool, parsed by vLLM's `qwen3_coder` parser), at most 250
  steps, and no network inside the sandbox. Grading uses the SWE-bench harness logic under
  Apptainer instead of Docker. On the 484 instances whose reference patch passes in that setup,
  the scores are 81.8 (BF16) and 68.6 (this model). No presence penalty for either model.

## Size breakdown

Measured from the safetensors headers of this checkpoint and of the BF16 base model.

| Component | BF16 base | This model | Where it lives when served |
|---|---|---|---|
| **Routed experts** (48 layers × 512, 120.8 B params) | 225.0 GiB | **38.7 GiB** (paired-4:8 NVFP4, sparse storage) | GPU |
| Attention, linear attention, hyper-connections, shared experts, embeddings, `lm_head`, norms, router, vision tower | 10.0 GiB | 10.0 GiB (BF16) | GPU |
| MTP module (speculative decoding) | 4.9 GiB | 4.9 GiB (BF16) | GPU |
| **GPU-resident total** | **239.9 GiB** | **53.5 GiB** | |
| Per-layer n-gram embedding tables (51.2 B entries) | 95.4 GiB | 95.4 GiB (BF16) | CPU |
| **Checkpoint total** | **335.3 GiB** | **149.0 GiB** | |

Only the routed experts are compressed. The same expert weights stored as dense NVFP4 (pruned
values kept as zeros) would take 59.8 GiB; the paired48 sparse storage drops the zeros and keeps a
4-bit mask per 4-byte chunk.

## Compression details

| Field | Value |
|---|---|
| Weights | NVFP4 (E2M1), one FP8-E4M3 scale per 32 dense K elements, FP32 per-tensor global scale |
| Activations | NVFP4, dynamic per-32 group scales, FP32 per-tensor global scale stored per expert linear |
| Sparsity | Paired 4:8 along K: in every 8 consecutive elements (4 pairs), exactly 2 pairs are kept |
| Compressed layers | Routed MoE experts (`gate_proj`, `up_proj`, `down_proj`) in all 48 layers |
| Left in BF16 | `lm_head`, embeddings, attention (linear and sparse), norms, router (`mlp.gate`), shared expert, hyper-connections, per-layer n-gram embedding, MTP head |
| Format | `compressed-tensors`, `nvfp4-pack-quantized`, with a `paired48_sparse` storage marker |

Gate and up projections carry separate global scales; the patched vLLM folds their ratio into
the block scales at load time (upstream vLLM would apply the gate scale to both).

### Sparse storage layout (`paired48_sparse`, `pair-bitmask` v1)

Each routed-expert linear stores `weight_sparse_packed` (uint8 `[out, K/4]`, the 2 kept bytes of
every 4), `weight_sparse_mask` (uint8 `[out, K/16]`, 4 bits per 4-byte chunk with exactly 2 set),
`weight_scale` (float8_e4m3 `[out, K/32]`) and the float32 global scales. The encoding is
lossless, and every tensor in this upload was checked with a round trip.

## Recipe (MoESQ, arm "gw2")

The full MoESQ config for this run is in [`moe_sq_config.yaml`](moe_sq_config.yaml).

- **Init:** GPTQ, 4-bit, 512 calibration samples, percdamp 0.1, block size 128. Masks come from
  paired-4:8 pruning.
- **Refinement:** masks and weight values are learned jointly for 10 epochs on 8,192 mixed
  calibration sequences of up to 4,096 tokens (masks LR 2e-4, weights LR 2.5e-5), with
  activations fake-quantized to NVFP4. The objective is the expert reconstruction error weighted
  by the router gate (`gate_weight_exponent = 2`).

## License

This checkpoint is a derivative of Qwen/Qwen3.8-Flash-Next and is distributed under the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).

## Citation

```bibtex
@misc{lee2026hardwarenativejointsparsequantizationtrillionscale,
      title={Hardware-Native Joint Sparse-Quantization for Trillion-Scale Mixture-of-Experts},
      author={Kwanhee Lee and Namhoon Lee and Dan Alistarh},
      year={2026},
      eprint={2610.02241},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      doi={10.48550/arXiv.2610.02241},
      url={https://arxiv.org/abs/2610.02241},
}
```

## Contact

For questions, open a discussion here or contact
[kwanhee.lee@postech.ac.kr](mailto:kwanhee.lee@postech.ac.kr).
