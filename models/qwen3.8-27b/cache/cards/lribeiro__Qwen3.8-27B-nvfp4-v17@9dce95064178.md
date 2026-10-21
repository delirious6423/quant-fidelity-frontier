---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - compressed-tensors
  - fp8
  - gptq
  - quantized
  - vllm
  - w8a8
  - blackwell
  - sm120
  - qwen3.8
  - vision
  - image-text-to-text
  - vlm
language:
  - multilingual
---
# Qwen3.8-27B-FP8-GPTQ-v17 (Vision)

FP8 (W8A8) uniform-precision, GPTQ-quantized variant of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [llm-compressor](https://github.com/vllm-project/llm-compressor) and
saved in `compressed-tensors` format for direct use with vLLM.

This is experiment **v17** in a 38-run quantization sweep — now with the **vision tower
intact**. Every linear language-model layer — MLP, self-attention, linear-attention, and
lm_head — uses FP8 E4M3 (W8A8) with per-channel weights and per-token dynamic activations,
plus GPTQ Hessian error correction. The **visual encoder and MTP head are kept in BF16**,
so the model retains full image/video understanding capability. v17 is the best all-FP8
config: it beats the official Qwen FP8 release on KLD (0.0123 vs 0.0134) thanks to GPTQ
weight correction, at 2.5× the prefill throughput.

> **What changed vs the previous v17 upload.** The original v17 checkpoint was saved as
> `Qwen3_5ForCausalLM` (text-only) — the visual tower was stripped during the save step
> even though the quantization recipe ignored it. This checkpoint merges the original
> quantized LM shards (byte-identical) with the 333 BF16 visual tensors + 15 BF16 MTP
> tensors extracted from the base model, and ships the full `Qwen3_5ForConditionalGeneration`
> config. The language-model weights are unchanged; KLD/benchmark numbers are identical
> to the text-only v17.

> **2026-08-17 — MTP speculative decoding fixed.** The VLM checkpoint's `config.json`
> omitted `re:^mtp.*` from `quantization_config.ignore`, so vLLM's compressed-tensors
> loader applied dynamic FP8 activation quantization to the BF16 MTP weights at runtime,
> collapsing speculative-acceptance to ~0.3%. The MTP tensors were always BF16 on disk —
> the defect was purely a config omission. Adding `re:^mtp.*` to the ignore list restores
> acceptance to ~70–77% (accept length 1.70–1.77), yielding a **54% decode speedup**
> (106.9 tok/s vs 69.4 tok/s at concurrency 1) with MTP speculative decoding. No weights
> changed; only `config.json` was updated. Thanks to **[@madarkadonk](https://huggingface.co/madarkadonk)**
> (Aaron Putnam) for reporting and root-causing this in
> [discussion #1](https://huggingface.co/lribeiro/Qwen3.8-27B-nvfp4-v17/discussions/1).

**The best uniform-precision config, superseded only by mixed-precision v31.**
v31(FP8 W8A16 weight-only attention) achieves 18% lower KLD by eliminating attention
activation quantization error, but v17 remains the reference all-FP8 baseline and the
fastest config in the Good KLD tier.
v23/v24/v26 confirmed that GPTQ dampening_frac is irrelevant for FP8 — v17's default 0.01 is optimal.


| Topic  | Notes |
|---|---|
| **Checkpoint size** | 29.97 GB (15 safetensors shards) |
| **BF16 base** | 55.6 GB → **1.85× smaller** |
| **Architecture** | `Qwen3_5ForConditionalGeneration` (VLM: vision encoder + language model + MTP) |
| **Vision tower** | 333 BF16 tensors (460.7M params), kept unquantized from base |
| **Hardware requirement** | NVIDIA Blackwell (SM120) or Ampere/Hopper (FP8 E4M3 support) |
| **Engine** | vLLM 0.26.1rc0 (infernal-invocation fork), FlashInfer backend |

---

## Model Details

**Qwen3.8-27B** (`qwen3_5` / `Qwen3_5ForConditionalGeneration`):

- **Type:** Vision-language model (hybrid linear + full attention language model + vision encoder)
- **Parameters:** 27B (language model) + 460.7M (vision tower)
- **Layers:** 64 — 48 linear-attention (GatedDeltaNet) + 16 full-attention (every 4th layer: 0, 4, 8, …, 60)
- **Hidden size:** 5,120
- **Intermediate size:** 17,408
- **Attention heads:** 24 × 256 dim
- **Vocabulary:** 248,320
- **Activation:** SiLU
- **Context:** 8,192 tokens (benchmark configuration)
- **Vision encoder:** 27-layer ViT, hidden size 1,152, patch size 16, 16 heads

Each linear-attention layer has `in_proj_qkv`, `in_proj_z`, `out_proj` plus recurrent state
paths `in_proj_a`/`in_proj_b`. Each full-attention layer has standard `q/k/v/o_proj`. All
layers share `gate_proj`/`up_proj`/`down_proj` MLP blocks. The vision tower
(`model.visual.*`) processes image/video patches and injects embeddings into the language
model via image/video token boundaries.

---

## Quantization Configuration

### Precision Assignment

| Module group | Format | Weights | Activations | Layers |
|---|---|---|---|---|
| MLP (gate/up/down) | **FP8 W8A8** (E4M3) | 8-bit float, sym, `channel`, `memoryless_minmax`, `actorder=static` | 8-bit float, sym, `token`, dynamic | all 64 |
| self_attn (Q/K/V/O) | FP8 W8A8 (E4M3) | same | same | all 16 full-attn layers |
| linear_attn (qkv/z/out) | FP8 W8A8 (E4M3) | same | same | all 48 lin-attn layers |
| lm_head | FP8 W8A8 (E4M3) | same | same | — |
| KV cache | FP8 E4M3, `tensor`, static | — | — | — |

**GPTQ:** Hessian-based weight correction, sequential targets `Qwen3_5DecoderLayer`,
`actorder=static`, `dampening_frac=0.01` (default).

### Modules Kept in BF16

| Pattern | Reason |
|---|---|
| `visual.*` | Vision tower — small, quantization-sensitive |
| `linear_attn.norm` | Numerically fragile normalization path |
| `linear_attn.in_proj_a` | Non-power-of-64 dims — CUTLASS constraint |
| `linear_attn.in_proj_b` | Non-power-of-64 dims — CUTLASS constraint |
| `mtp.*` | Multi-token-prediction head — kept BF16; functional for MTP speculative decoding (accept length ~1.70–1.77 after 2026-08-17 config fix) |

### Calibration

- **Source:** `malaiwah/qwen38-27b-fidelity-suite-v3` tokens (181 contexts × 2048 tokens)
- **Samples:** 181
- **Max sequence length:** 2048
- **Format:** Decoded token IDs from suite, not raw text

---

## Usage

```bash
# vLLM serve (requires Blackwell SM120 + FlashInfer)
vllm serve /models/Qwen3.8-27B-nvfp4-gptq-v17-vision \
  --max-model-len 8192 \
  --max-num-seqs 512 \
  --quantization compressed-tensors \
  --trust-remote-code
```

> **`--max-num-seqs` matters on this architecture.** 48 of 64 layers use linear attention,
> and vLLM allocates one Mamba-style cache block per decode sequence. The default
> `max_num_seqs=1024` can exceed available blocks and fail during CUDA graph capture with
> `max_num_seqs (1024) exceeds available Mamba cache blocks`. Lower to 512 or raise
> `--gpu-memory-utilization`. This is a property of the base model, not of quantization.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/models/Qwen3.8-27B-nvfp4-gptq-v17-vision",
          quantization="compressed-tensors",
          max_model_len=8192,
          trust_remote_code=True)

# Text-only generation (identical to text-only v17)
out = llm.generate(
    ["Explain 4-bit quantization in two sentences."],
    SamplingParams(temperature=0.7, max_tokens=256),
)
print(out[0].outputs[0].text)

# Vision: image + text via chat template
from vllm.inputs import TextPrompt
messages = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."},
    ],
}]
prompt = llm.get_tokenizer().apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
out = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": image_placeholder}},
    SamplingParams(temperature=0.7, max_tokens=512),
)
print(out[0].outputs[0].text)
```

### Docker Compose

A `docker-compose.yml` is provided for one-command deployment. The image bundles the
vLLM infernal-invocation fork with FlashInfer and SM120 kernels; only the quantized
checkpoint must be mounted from the host.

```bash
# Place the checkpoint at ./models/Qwen3.8-27B-nvfp4-gptq-v17-vision, then:
docker compose up -d

# Or override the model path and port:
MODEL_DIR=/data/models/Qwen3.8-27B-nvfp4-gptq-v17-vision PORT=8001 docker compose up -d
```

The endpoint is OpenAI-compatible: `http://localhost:8000/v1`.

<details>
<summary><b>docker-compose.yml</b></summary>

```yaml
# docker-compose.yml — Qwen3.8-27B-FP8-GPTQ-v17
#
# Serves the v17 all-FP8 W8A8 GPTQ quantization via vLLM with FlashInfer.
# Image: lribeiro/qwen38-quant:ii-r15-p1  (vLLM infernal-invocation fork, SM120)
#
# Requirements:
#   - NVIDIA Blackwell (SM120) GPU with FP8 E4M3 tensor cores
#   - NVIDIA Container Toolkit (nvidia-docker) installed on host
#   - Quantized checkpoint at ${MODEL_DIR} (default: ./models/Qwen3.8-27B-nvfp4-gptq-v17)
#
# Customize via .env file or shell environment variables:
#
#   PORT=8000                           API server port
#   SERVED_MODEL_NAME=qwen38-27b        Model name returned by /v1/models
#   MAX_MODEL_LEN=8192                  Maximum context length (tokens)
#   MAX_NUM_SEQS=512                    Maximum concurrent decode sequences
#   MAX_NUM_BATCHED_TOKENS=8192         Maximum tokens per batching iteration
#   GPU_MEMORY_UTILIZATION=0.90         Fraction of GPU memory for vLLM (0.0–1.0)
#   TENSOR_PARALLEL_SIZE=1              Tensor-parallel degree (1, 2, 4)
#   KV_CACHE_DTYPE=fp8                  KV cache dtype: fp8 | auto | bfloat16
#   CUDA_VISIBLE_DEVICES=0              GPU index(s) to use
#   MODEL_DIR=./models/Qwen3.8-27B-nvfp4-gptq-v17   Host path to checkpoint

services:
  qwen38-27b-v17:
    image: lribeiro/qwen38-quant:ii-r15-p1 # ii with mxfp4 patches and other minor corrections
    container_name: qwen38-27b-v17
    ipc: host
    ports:
      - "${PORT:-8000}:${PORT:-8000}"
    volumes:
      # Mount the quantized model checkpoint (read-only)
      - ${MODEL_DIR:-./models/Qwen3.8-27B-nvfp4-gptq-v17}:/models/Qwen3.8-27B-nvfp4-gptq-v17:ro
      # Persistent JIT / compile caches (avoids recompilation on restart)
      - vllm-cache:/cache
      # HuggingFace hub cache (tokenizer / config downloads)
      - hf-cache:/data/hf_cache
    environment:
      # ── GPU / runtime ──
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - SAFETENSORS_FAST_GPU=1
      - OMP_NUM_THREADS=16
      # ── vLLM engine ──
      - VLLM_USE_AOT_COMPILE=1
      - VLLM_USE_FLASHINFER_SAMPLER=1
      - VLLM_USE_V2_MODEL_RUNNER=1
      - VLLM_ALLOW_INSECURE_SERIALIZATION=1
      # SM120: disable broken FP4/FP8 kernels — use FlashInfer / b12x /  CUTLASS paths
      - VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel,MarlinFP8ScaledMMLinearKernel
      # ── Cache directories ──
      - HF_HOME=/data/hf_cache
      - VLLM_CACHE_DIR=/cache/vllm
      - TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor
      - TRITON_CACHE_DIR=/cache/triton
      - FLASHINFER_WORKSPACE_BASE=/cache/flashinfer
    entrypoint: ["python", "-m", "vllm.entrypoints.cli.main"]
    command:
      - serve
      - /models/Qwen3.8-27B-nvfp4-gptq-v17
      - --served-model-name=${SERVED_MODEL_NAME:-qwen38-27b}
      - --host=0.0.0.0
      - --port=${PORT:-8000}
      - --trust-remote-code
      - --quantization=compressed-tensors
      - --tensor-parallel-size=${TENSOR_PARALLEL_SIZE:-1}
      - --kv-cache-dtype=${KV_CACHE_DTYPE:-fp8}
      - --block-size=128
      - --load-format=fastsafetensors
      - --gpu-memory-utilization=${GPU_MEMORY_UTILIZATION:-0.94}
      - --max-model-len=${MAX_MODEL_LEN:-262144}
      - --max-num-seqs=${MAX_NUM_SEQS:-512}
      - --max-num-batched-tokens=${MAX_NUM_BATCHED_TOKENS:-8192}
      - --generation-config=vllm
      - --reasoning-parser=qwen3
      - --attention-backend=flashinfer
      - --linear-backend=auto
      - --max-cudagraph-capture-size=16
      - --async-scheduling
      - --enable-chunked-prefill
      - --enable-prefix-caching
      - --enable-flashinfer-autotune
      - >-
        --compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

volumes:
  vllm-cache:
  hf-cache:
```

</details>

**Configurable variables** (set in `.env` or shell):

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | API server port (OpenAI-compatible) |
| `SERVED_MODEL_NAME` | `qwen38-27b` | Model name returned by `/v1/models` |
| `MAX_MODEL_LEN` | `262144` | Maximum context length in tokens |
| `MAX_NUM_SEQS` | `512` | Maximum concurrent decode sequences (lower if Mamba cache blocks exhausted) |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Maximum tokens per batching iteration |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of GPU memory allocated to vLLM |
| `TENSOR_PARALLEL_SIZE` | `1` | Tensor-parallel degree (1, 2, or 4) |
| `KV_CACHE_DTYPE` | `fp8` | KV cache dtype: `fp8`, `auto`, or `bfloat16` |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU index(s) to use |
| `MODEL_DIR` | `./models/Qwen3.8-27B-nvfp4-gptq-v17` | Host path to quantized checkpoint |

---

## Evaluation

### Distribution Fidelity (KLD)

Hidden-state capture after final RMSNorm → shared BF16 LM-head replay → full-vocabulary
`KL(BF16_ref ‖ candidate)`, two-pass log-sum-exp normalization. Body-only comparison
(candidate LM head unused). Suite: 136 contexts × 2048 tokens, 278,392 scored positions
over 248,320-token vocabulary.

| Metric | Value |
|---|---:|
| **Token mean KLD** | **0.012310** |
| Token median KLD | 0.002351 |
| p95 KLD | 0.047520 |
| p99 KLD | 0.191077 |
| p99.9 KLD | 0.694279 |
| Max KLD | 3.961384 |
| Mean JSD (bits) | 0.004245 |
| **Top-1 agreement** | **96.17%** |
| Validation tier | Good (KLD < 0.013, Top-1 > 96%) — matches/beats official FP8 |

### KLD by Domain

| Stratum | Contexts | Mean KLD |
|---|---:|---:|
| scientific | 39 | 0.00409 |
| multilingual | 7 | 0.00439 |
| encyclopedic | 13 | 0.00866 |
| code | 36 | 0.01276 |
| literary | 41 | 0.02224 |

Literary text is the dominant KLD contributor — 5.4× the scientific stratum. FP8
quantization affects creative/prose generation more than technical text, but the gap is
narrower than in 4-bit configs (v6: 4.9×, v7: 4.9× — v17 has a wider but lower-magnitude
spread due to its lower baseline KLD).

### Throughput

Single NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB), vLLM 0.26.1rc0, FlashInfer backend.
Duration-based sustained decode, 30s cells, temperature 0.0, `ignore_eos=true`.

| Metric | Value |
|---|---:|
| **Prefill (2k context)** | **7,306 tok/s** |
| TTFT (2k) | 0.279 s |
| Decode C1 (0 ctx) | 49.8 tok/s |
| Decode C1 (2k ctx) | 49.4 tok/s |
| Decode C4 (0 ctx) | 180.6 tok/s |
| Decode C4 (2k ctx) | 176.1 tok/s |

### Composite Score

$$\text{score} = \frac{\text{prefill}_{2k} \times \text{decode}_{c1}}{\text{KLD} \times 10^6} = \frac{7306 \times 49.8}{0.01231 \times 10^6} = \mathbf{29.55}$$

---

## Context Within Experiment Series

v17 is the reference all-FP8 W8A8 baseline (Phase 2). It established that uniform FP8
with GPTQ correction strictly dominates the entire NVFP4 W4A4 line: lower KLD than the
best 4-bit config (v6: 0.0160) at comparable speed, with much higher precision (E4M3
8-bit vs E2M1 4-bit). Three follow-up experiments confirmed v17's hyperparameters are
optimal:

| Experiment | Change vs v17 | KLD | vs v17 | Score |
|---|---|---:|---|---:|
| **v17 (baseline)** | — | **0.01231** | — | **29.55** |
| v23 | damp=0.001 | 0.01241 | +1% | 29.03 |
| v24 | damp=0.005, block256 | 0.01253 | +2% | 28.52 |
| v26 | damp=0.1 | 0.01245 | +1% | 29.31 |
| v16 | 512 calib samples | 0.01245 | +1% | 28.37 |

**GPTQ dampening is irrelevant for FP8.** Tested 0.001–0.1 — all produce 0.0124 ± noise.
The GPTQ Hessian is well-conditioned for FP8 regardless of dampening_frac.

**Calibration sample count doesn't matter.** v16 (512 samples) vs v17 (181 samples) —
KLD 0.01245 vs 0.01231, within noise. The Hessian is well-estimated at 181 samples.

Phase 3 (v18–v20, v27–v30) used v17 as the baseline for KLD attribution by layer type,
finding that MLP FP8 contributes ~0.009 KLD (dominant), linear_attn activation
quantization ~0.0025, and self_attn ~0.0006. Phase 4 (v30–v34) built on this to produce
v31 (W8A16 weight-only attention), the overall best model at KLD 0.01014, score 31.45.

### Pareto Position

| Objective | Best Model | KLD | Prefill | Decode | Score |
|---|---|---:|---:|---:|---:|
| Best composite score | v31 | 0.01014 | 6,385 | 49.9 | **31.45** |
| **Best all-FP8 (uniform)** | **v17** | **0.01231** | **7,306** | **49.8** | **29.55** |
| Lowest KLD (our quants) | v18 | 0.00903 | 6,023 | 40.8 | 27.19 |
| Best NVFP4 W4A4 | v6 | 0.01597 | 7,756 | 52.3 | 25.42 |

---

## Provenance & Integrity

| Artifact | SHA-256 |
|---|---|
| Model index | `aa62fb91085564f542299f80c1447e044fca41fbb89350d2979adc0e2804e097` |
| Config | `dc8f1def22f7c0f5ed182444c8431d16dbdc9aab398c894918c864e96c78ad74` |
| Suite tokens | `3f9d17f1b55f64872ad3ac19c8711654e09ba70b7ca14b0851525088fe735691` |
| Capture manifest | `fb2317dd353c01b4544427ded1c2966b4276e370ae2d1b492ac052858ce9d997` |
| Shared LM head | `25a30fd5f826da0abc4efc4cc71def9f02bcb8085f7175eee284d221dee4cfff` |

14 model shards individually hashed in `reports/report-nvfp4-gptq-v17.json` →
`candidate_identity.shard_sha256`. Full fidelity report, benchmark JSON, and hidden-state
captures archived in the experiment workspace.

---

## Hardware & Runtime

| Field | Value |
|---|---|
| GPU | 4× NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 96 GiB GDDR7 each) |
| Driver | 595.58.03 |
| Benchmark GPU count | 1 |
| VRAM used | 86,606 / 97,887 MB (88.5%) |
| Temp (avg / max) | 53.0 °C / 59.0 °C |

---

## Reproduction

The language-model weights are byte-identical to the original text-only v17. This
checkpoint was produced by merging the existing v17 quantized shards with the BF16
visual tower + MTP head extracted from the base model — no re-quantization needed.

### Step 1: Original v17 quantization (already done)

```bash
# Inside Docker (vLLM infernal-invocation image, FlashInfer, SM120)
CUDA_VISIBLE_DEVICES=0 python /data/scripts/quantize_nvfp4_gptq_v17.py \
  --model /data/models/Qwen3.8-27B-bf16 \
  --output /data/models/Qwen3.8-27B-nvfp4-gptq-v17 \
  --calib-samples 181 \
  --calib-max-len 2048
```

Calibration tokens from `$SUITE_DIR/tokens` (default `/data/suite-v3/tokens`).
Script: `scripts/quantize_nvfp4_gptq_v17.py`. Log: `quant_v17.log`.

### Step 2: Merge visual tower + MTP from base (this checkpoint)

```bash
# Merge v17 quantized LM shards with BF16 visual+mtp from base model
python /workspace/scripts/merge_v17_vision.py \
  --v17 /models/Qwen3.8-27B-nvfp4-gptq-v17 \
  --base /models/Qwen3.8-27B-bf16 \
  --output /data/models/Qwen3.8-27B-nvfp4-gptq-v17-vision
```

Script: `scripts/merge_v17_vision.py`. Extracts 333 visual + 15 MTP tensors from the
BF16 base, writes them as a 15th shard, copies the 14 v17 shards unchanged, and patches
`config.json` to `Qwen3_5ForConditionalGeneration` (VLM) with `vision_config`.

> **Note:** The v17 script's default `--calib-samples` is 1024, but the suite contains
> only 181 contexts. llm-compressor warns `Requested 1024 samples but the provided
> dataset only has 181 samples` and uses all 181. The effective calibration set is
> 181 samples, matching v6/v16/v31. Pass `--calib-samples 181` to suppress the warning.

---

## Limitations

- **KLD far from malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated** [**malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated**](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6)
- **Superseded by v31.** v31 (FP8 W8A8 MLP + W8A16 weight-only attention) achieves 18%
  lower KLD (0.01014 vs 0.01231) at similar decode speed. Use v17 when uniform FP8 is
  required or when the faster prefill (7,306 vs 6,385 tok/s) matters more than the KLD
  gap.
- **Attention activation quantization is the second-largest KLD contributor.** FP8 W8A8
  quantizes attention activations; v31 eliminates this by keeping BF16 activations for
  attention (W8A16 weight-only). KLD attribution: MLP ~0.009, linear_attn ~0.0025,
  self_attn ~0.0006.
- **Blackwell-optimized.** FP8 E4M3 MMA is supported on SM120 (Blackwell), SM89/SM90
  (Ada/Hopper), but the benchmark and KLD capture were run on SM120 only. Validate on
  your hardware.
- **Body-only KLD.** Fidelity measured against a shared BF16 LM head; end-to-end
  generation quality may differ slightly.
- **Literary domain degradation.** 5.4× higher KLD on literary text vs scientific —
  creative/prose generation is disproportionately affected, though less severely than in
  4-bit configs.
- **Quantization is lossy.** Validate on your own workload before production use.


## Acknowledgements

Standing on the shoulders of giants. This work would not exist without the
[**rtx6kpro**](https://github.com/local-inference-lab/rtx6kpro) community and the
broader local-inference-lab Discord — the open protocol, tooling, and baseline
measurements that made systematic quantization comparison possible.

The distribution-fidelity (KLD) methodology follows the published protocol from
[Kimi-K3 distribution fidelity 1024×2048](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/kimi-k3/distribution-fidelity-1024x2048.md).
The reference harness (`glm52_exl3_shared_h_kld.py`) and the benchmark tool
(`llm-inference-bench`) are adapted from the rtx6kpro repository. The Gilded
Gnosis EXL3 model cards established the per-bit error-ladder and validation-tier
framework that this sweep builds on.

---