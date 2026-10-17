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
  - w8a16
  - weight-only
  - blackwell
  - sm120
  - qwen3.8
  - vision
  - mtp
  - image-text-to-text
  - vlm
language:
  - multilingual
---
# Qwen3.8-27B-FP8-Pessoa

FP8 (W8A16) weight-only, GPTQ-quantized variant of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [llm-compressor](https://github.com/vllm-project/llm-compressor) and
saved in `compressed-tensors` format for direct use with vLLM.

**Pessoa** is the best FP8 quantization of Qwen3.8-27B of a set of experiments with autoresearch
it reached the the lowest KLD (0.0056)
achieved by any practical-size FP8 model in a 97-model quantization sweep. 
Every linear language-model layer MLP, self-attention, and linear-attention uses
FP8 E4M3 **weight-only** (W8A16) with per-channel weights, BF16 activations, and
GPTQ Hessian error correction. The **vision tower and MTP head are kept in BF16**,
so the model retains full image/video understanding and multi-token-prediction
speculative decoding capability.

> **Named after Fernando Pessoa** (1888–1935), the Portuguese poet who wrote under
> dozens of heteronyms — each a distinct literary persona with its own voice. This
> model too has multiple selves: the quantized language model, the unquantized vision
> tower, and the MTP speculative head three complete personas in one checkpoint.

| Topic | Notes |
|---|---|
| **Checkpoint size** | 31.24 GB (3 safetensors shards) |
| **BF16 base** | 55.6 GB → **1.78× smaller** |
| **Architecture** | `Qwen3_5ForConditionalGeneration` (VLM: vision encoder + language model + MTP) |
| **Vision tower** | 333 BF16 tensors (460.7M params), kept unquantized from base |
| **MTP head** | 15 BF16 tensors, kept unquantized from base |
| **Hardware requirement** | NVIDIA Blackwell (SM120) or Ampere/Hopper (FP8 E4M3 support) |
| **Engine** | vLLM 0.26.1rc0 (infernal-invocation fork), FlashInfer backend |

---

## Model Details

**Qwen3.8-27B** (`qwen3_5` / `Qwen3_5ForConditionalGeneration`):

- **Type:** Vision-language model (hybrid linear + full attention language model + vision encoder + MTP head)
- **Parameters:** 27B (language model) + 460.7M (vision tower) + 82.3M (MTP head)
- **Layers:** 64 — 48 linear-attention (GatedDeltaNet) + 16 full-attention (every 4th layer: 0, 4, 8, …, 60)
- **Hidden size:** 5,120
- **Intermediate size:** 17,408 (MLP is 3.4× hidden — dominates parameter count)
- **Attention heads:** 24 × 256 dim
- **KV heads:** 4 (GQA 6:1) — only on 16 full-attention layers
- **Vocabulary:** 248,320
- **Activation:** SiLU
- **Context:** 262,144 tokens native
- **Vision encoder:** 27-layer ViT, hidden size 1,152, patch size 16, 16 heads, temporal patch size 2
- **MTP:** 1 hidden layer, multi-token-prediction speculative decoding head

Each linear-attention layer has `in_proj_qkv`, `in_proj_z`, `out_proj` plus recurrent
state paths `in_proj_a`/`in_proj_b`. Each full-attention layer has standard
`q/k/v/o_proj`. All layers share `gate_proj`/`up_proj`/`down_proj` MLP blocks. The
vision tower (`model.visual.*`) processes image/video patches and injects embeddings
into the language model via image/video token boundaries. The MTP head (`mtp.*`)
provides speculative decoding tokens for inference acceleration.

---

## Quantization Configuration

### Precision Assignment

| Module group | Format | Weights | Activations | Layers |
|---|---|---|---|---|
| MLP (gate/up/down) | **FP8 W8A16** (E4M3) | 8-bit float, sym, `channel`, `memoryless_minmax`, `actorder=static` | **BF16 (none)** | all 64 |
| self_attn (Q/K/V/O) | **FP8 W8A16** (E4M3) | same | **BF16 (none)** | all 16 full-attn layers |
| linear_attn (qkv/z/out) | **FP8 W8A16** (E4M3) | same | **BF16 (none)** | all 48 lin-attn layers |
| KV cache | FP8 E4M3, `tensor`, static | — | — | — |

**Key difference from W8A8:** This model uses **weight-only** FP8 (W8A16) — weights
are FP8 E4M3 per-channel, but activations remain BF16. This eliminates activation
quantization error entirely, halving KLD versus W8A8 (0.0056 vs 0.0102). The tradeoff
is ~18% slower prefill (4,875 vs 6,213 tok/s) because BF16 activations require
dequantization before GEMM.

**GPTQ:** Hessian-based weight correction, sequential targets `Qwen3_5DecoderLayer`,
`actorder=static`, `dampening_frac=0.01`, `block_size=128`. GPTQ corrects the weight
quantization residual by adjusting remaining weights using the calibration data's
second-order statistics — this is what gives Pessoa 2.3× lower KLD than ModelOpt's
max-calibration FP8 (0.0056 vs 0.0127).

### Tensor Inventory

| Category | Count | Format | Shard |
|---|---:|---|---|
| MLP (gate/up/down) | 384 | FP8 W8A16 | shards 1–2 |
| self_attn (Q/K/V/O + norms) | 192 | FP8 W8A16 | shards 1–2 |
| linear_attn (qkv/z/out) | 576 | FP8 W8A16 | shards 1–2 |
| embed/norm/lm_head | 211 | BF16 (unquantized norms/embed) | shards 1–2 |
| **Vision tower** | **333** | **BF16** | shard 3 |
| **MTP head** | **15** | **BF16** | shard 3 |
| **Total** | **1,631** | | 3 shards |

### Modules Kept in BF16

| Pattern | Count | Reason |
|---|---:|---|
| `re:.*visual.*` | 333 | Vision tower — small (460M), quantization-sensitive, not worth the risk |
| `re:^mtp.*` | 15 | MTP speculative-decoding head — kept BF16 for functional speculative decoding |
| `lm_head` | 1 | Output projection directly produces logits — quantizing increases KLD by ~30% |
| `linear_attn.norm` (×48) | 48 | Numerically fragile normalization path in GatedDeltaNet |
| `linear_attn.in_proj_a` (×48) | 48 | Recurrent state accumulator — quantizing compounds error across sequence, produces NaN |
| `linear_attn.in_proj_b` (×48) | 48 | Recurrent state accumulator — same as in_proj_a |
| `linear_attn` (whole, ×48) | 48 | Parent module ignore — covers any submodules not explicitly listed |
| **Total ignore entries** | **195** | |

### Calibration

- **Source:** `malaiwah/qwen38-27b-fidelity-suite-v3` tokens (181 contexts × 2048 tokens)
- **Samples:** 181
- **Max sequence length:** 2048
- **Format:** Decoded token IDs from suite, not raw text
- **Dampening:** `dampening_frac=0.01` (default — confirmed optimal for FP8; tested 0.001–0.1, all within noise)

### Compressed-Tensors Config

```json
{
  "quant_method": "compressed-tensors",
  "format": "float-quantized",
  "quantization_status": "compressed",
  "version": "0.18.0",
  "config_groups": {
    "group_0": {
      "targets": ["Linear"],
      "weights": {
        "num_bits": 8,
        "type": "float",
        "symmetric": true,
        "group_size": null,
        "strategy": "channel",
        "dynamic": false,
        "actorder": "static",
        "observer": "memoryless_minmax"
      },
      "input_activations": null
    }
  },
  "kv_cache_scheme": {
    "num_bits": 8,
    "type": "float",
    "symmetric": true,
    "strategy": "tensor",
    "dynamic": false,
    "observer": "static_minmax"
  }
}
```

---

## Evaluation

### Distribution Fidelity (KLD)

Hidden-state capture after final RMSNorm → shared BF16 LM-head replay → full-vocabulary
`KL(BF16_ref ‖ candidate)`, two-pass log-sum-exp normalization. Body-only comparison
(candidate LM head unused). Suite: 136 contexts × 2048 tokens, 278,392 scored positions
over 248,320-token vocabulary.

| Metric | Value |
|---|---:|
| **Token mean KLD** | **0.005559** |
| Token median KLD | 0.001166 |
| p95 KLD | 0.020902 |
| p99 KLD | 0.085659 |
| p99.9 KLD | 0.316350 |
| Max KLD | 4.421645 |
| Mean JSD (bits) | 0.001958 |
| **Top-1 agreement** | **97.47%** |
| Validation tier | Excellent (KLD < 0.006, Top-1 > 97%) |

### KLD by Domain

| Stratum | Contexts | Mean KLD |
|---|---:|---:|
| multilingual | 7 | 0.00192 |
| scientific | 39 | 0.00193 |
| encyclopedic | 13 | 0.00377 |
| code | 36 | 0.00596 |
| literary | 41 | 0.00985 |

Literary text remains the dominant KLD contributor — 5.1× the scientific stratum.
Creative/prose generation is disproportionately affected by quantization, but the
gap is narrower than in W8A8 configs (v17: 5.4×) and much narrower than 4-bit
configs. Pessoa's weight-only approach preserves activation distributions, keeping
literary KLD under 0.01.

### Throughput

Single NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB), vLLM 0.26.1rc0, FlashInfer backend.
Duration-based sustained decode, 30s cells, temperature 0.0, `ignore_eos=true`.

| Metric | Value |
|---|---:|
| **Prefill (2k context)** | **4,875 tok/s** |
| TTFT (2k) | 0.418 s |
| Decode C1 (0 ctx) | 45.6 tok/s |
| Decode C1 (2k ctx) | 45.0 tok/s |
| Decode C4 (0 ctx) | 166.6 tok/s |
| Decode C4 (2k ctx) | 158.9 tok/s |

NOTE: This was not optimized for speed, but for equal grounds between all experiments and models tested.
An optimized setup with a 6kpro reaches >100tok/s with MTP.

### Composite Score

$$\text{score} = \frac{\text{prefill}_{2k} \times \text{decode}_{c1}}{\text{KLD} \times 10^6} = \frac{4875 \times 45.6}{0.00556 \times 10^6} = \mathbf{40.02}$$

**Highest composite score among all practical-size FP8 models** in the 97-model sweep.

---

## Context Within Experiment Series

Pessoa is the FP8 weight-only winner from a systematic 97-model quantization sweep
of Qwen3.8-27B, exploring NVFP4, MXFP8, and FP8 formats via both ModelOpt (NVIDIA)
and GPTQ (llm-compressor). The sweep evaluated GPTQ vs ModelOpt, weight-only vs
weight+activation, single-format vs mixed-precision, and per-layer sensitivity.

### Key Finding: Weight-Only Beats Weight+Activation for FP8

| Config | KLD | Prefill | Decode | Score |
|---|---:|---:|---:|---:|
| **Pessoa (W8A16 weight-only)** ★ | **0.0056** | 4,875 | 45.6 | **40.02** |
| FP8 W8A8 (GPTQ) | 0.0102 | 6,213 | 46.6 | 28.46 |
| FP8 W8A16 (ModelOpt) | 0.0127 | 6,026 | 45.4 | 21.65 |

Weight-only FP8 eliminates activation quantization error entirely. GPTQ corrects
the weight quantization residual. The result is 2× better KLD than W8A8 and 2.3×
better than ModelOpt at the cost of ~18% slower prefill (BF16 dequantization
before GEMM).

### Key Finding: GPTQ Beats ModelOpt by 2.3×

| Format | ModelOpt best KLD | GPTQ best KLD | Ratio |
|---|---:|---:|---:|
| FP8 (weight-only) | 0.01265 | 0.00556 | 2.3× |
| FP8 (W8A8 mixed) | 0.01725 | 0.01016 | 1.7× |
| MXFP8 (weight-only) | 0.01265 | 0.01072 | 1.2× |
| NVFP4 (mixed) | 0.35026 | 0.01087 | 32× |

GPTQ's Hessian-based weight correction compensates for quantization error by
adjusting remaining weights using calibration data's second-order statistics.
ModelOpt's max calibration only computes scale factors without weight correction.

### Pareto Position

| Objective | Best Model | KLD | Prefill | Decode | Score |
|---|---|---:|---:|---:|---:|
| Best quality (any size) [way too big] | gptq-fp8-attn-only | 0.0026 | 4,458 | 30.9 | 52.53 |
| **Best FP8 (~28 GB)** | **Pessoa** ★ | **0.0056** | **4,875** | **45.6** | **40.02** |
| Best MXFP8 (~28 GB) | gptq-mxfp8-mixed | 0.0107 | 5,917 | 46.1 | 25.44 |
| Best NVFP4 (~27 GB) | gptq-nvfp4-mixed-8 | 0.0109 | 5,956 | 48.1 | 26.38 |
| Best all-FP8 (W8A8) | v17 | 0.0123 | 7,306 | 49.8 | 29.55 |

Pessoa dominates the practical-size Pareto frontier: 2× lower KLD than any other
~28 GB model, with the highest composite score in its class.

---

## Usage

```bash
# vLLM serve (requires Blackwell SM120 + FlashInfer)
vllm serve /models/Qwen3.8-27B-FP8-Pessoa \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

> **`--max-num-seqs` matters on this architecture.** 48 of 64 layers use linear
> attention, and vLLM allocates one Mamba-style cache block per decode sequence.
> The default `max_num_seqs=1024` can exceed available blocks and fail during CUDA
> graph capture with `max_num_seqs (1024) exceeds available Mamba cache blocks`.
> Lower to 16–512 or raise `--gpu-memory-utilization`. This is a property of the
> base model, not of quantization.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/models/Qwen3.8-27B-FP8-Pessoa",
          quantization="compressed-tensors",
          kv_cache_dtype="fp8",
          max_model_len=262144,
          trust_remote_code=True)

# Text-only generation
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

A `docker-compose.yml` is provided in the experiment workspace for one-command
deployment. The image bundles the vLLM infernal-invocation fork with FlashInfer
and SM120 kernels; only the quantized checkpoint must be mounted from the host.

```bash
# Place the checkpoint at ./models/Qwen3.8-27B-FP8-Pessoa, then:
docker compose up -d

# Or override the model path and port:
MODEL_DIR=/data/models/Qwen3.8-27B-FP8-Pessoa PORT=8001 docker compose up -d
```

The endpoint is OpenAI-compatible: `http://localhost:8000/v1`.

<details>
<summary><b>docker-compose.yml</b></summary>

```yaml
# docker-compose.yml — Qwen3.8-27B-FP8-Pessoa
#
# Serves the FP8 W8A16 weight-only GPTQ quantization via mainline vLLM.
# Image: vllm/vllm-openai:latest
#
# Requirements:
#   - NVIDIA Blackwell (SM120) GPU with FP8 E4M3 tensor cores
#   - NVIDIA Container Toolkit (nvidia-docker) installed on host
#   - Quantized checkpoint at ${MODEL_DIR}
#
# Customize via .env file or shell environment variables:
#
#   PORT=8000                           API server port
#   SERVED_MODEL_NAME=qwen38-27b-pessoa Model name returned by /v1/models
#   MAX_MODEL_LEN=262144                Maximum context length (tokens)
#   MAX_NUM_SEQS=16                     Maximum concurrent decode sequences
#   MAX_NUM_BATCHED_TOKENS=16384        Maximum tokens per batching iteration
#   GPU_MEMORY_UTILIZATION=0.97         Fraction of GPU memory for vLLM (0.0–1.0)
#   TENSOR_PARALLEL_SIZE=2              Tensor-parallel degree (1, 2, or 4)
#   KV_CACHE_DTYPE=fp8                  KV cache dtype: fp8 | auto | bfloat16
#   CUDA_VISIBLE_DEVICES=0,1            GPU indices to use
#   MODEL_DIR=./models/Qwen3.8-27B-FP8-Pessoa  Host path to checkpoint

services:
  qwen38-27b-pessoa:
    image: vllm/vllm-openai:latest
    container_name: qwen38-27b-pessoa
    ipc: host
    ports:
      - "${PORT:-8000}:${PORT:-8000}"
    volumes:
      - ${MODEL_DIR:-./models/Qwen3.8-27B-FP8-Pessoa}:/models/Qwen3.8-27B-FP8-Pessoa:ro
      - vllm-cache:/cache
      - hf-cache:/data/hf_cache
    environment:
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - OMP_NUM_THREADS=16
      - HF_HOME=/data/hf_cache
      - VLLM_CACHE_ROOT=/cache/vllm
      - TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor
      - TRITON_CACHE_DIR=/cache/triton
    entrypoint: ["python3", "-m", "vllm.entrypoints.cli.main"]
    command:
      - serve
      - /models/Qwen3.8-27B-FP8-Pessoa
      - --served-model-name=${SERVED_MODEL_NAME:-qwen38-27b-pessoa}
      - --host=0.0.0.0
      - --port=${PORT:-8000}
      - --trust-remote-code
      - --quantization=compressed-tensors
      - --tensor-parallel-size=${TENSOR_PARALLEL_SIZE:-1}
      - --disable-custom-all-reduce
      - --kv-cache-dtype=${KV_CACHE_DTYPE:-fp8}
      - --block-size=128
      - --gpu-memory-utilization=${GPU_MEMORY_UTILIZATION:-0.96}
      - --max-model-len=${MAX_MODEL_LEN:-262144}
      - --max-num-seqs=${MAX_NUM_SEQS:-16}
      - --max-num-batched-tokens=${MAX_NUM_BATCHED_TOKENS:-16384}
      - --max-cudagraph-capture-size=64
      - --generation-config=vllm
      - --reasoning-parser=qwen3
      - --attention-backend=flashinfer
      - --enable-chunked-prefill
      - --enable-prefix-caching
      - --enable-auto-tool-choice
      - --tool-call-parser=qwen3_coder
      - --mm-encoder-tp-mode=data
      - --speculative-config={"method":"mtp","num_speculative_tokens":3}
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
| `SERVED_MODEL_NAME` | `qwen38-27b-pessoa` | Model name returned by `/v1/models` |
| `MAX_MODEL_LEN` | `262144` | Maximum context length in tokens |
| `MAX_NUM_SEQS` | `16` | Maximum concurrent decode sequences (lower if Mamba cache blocks exhausted) |
| `MAX_NUM_BATCHED_TOKENS` | `16384` | Maximum tokens per batching iteration |
| `GPU_MEMORY_UTILIZATION` | `0.96` | Fraction of GPU memory allocated to vLLM |
| `TENSOR_PARALLEL_SIZE` | `1` | Tensor-parallel degree (1, 2, or 4) |
| `KV_CACHE_DTYPE` | `fp8` | KV cache dtype: `fp8`, `auto`, or `bfloat16` |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU index(s) to use |
| `MODEL_DIR` | `./models/Qwen3.8-27B-FP8-Pessoa` | Host path to quantized checkpoint |

---

## Provenance & Integrity

| Artifact | SHA-256 |
|---|---|
| Model index (merged) | `6cfd4c257d1d22456d48e8ae43a37e1bec4210124d6187f507fc9affeec982c2` |
| Config (VLM) | `ee992684dffb2db4e7ade4d8e8fee9e5402bb3d075487eea16ebc3aa82c07d27` |
| Shard 1 (quantized LM) | `25f10ccdeee734cc6fe0f3ed0f34830f61cc58d8cd4d608c5dcede8a19825b52` |
| Shard 2 (quantized LM) | `7350bf425f9cca42f4376ce88e3069c1a7727bc59096ed81e0024d74ede64fd7` |
| Shard 3 (BF16 vision + MTP) | `bf23280788e616231c29db327f7df25254673c0e8c33c717bf321e310ec5ef64` |
| Suite tokens | `3f9d17f1b55f64872ad3ac19c8711654e09ba70b7ca14b0851525088fe735691` |
| Shared LM head | `25a30fd5f826da0abc4efc4cc71def9f02bcb8085f7175eee284d221dee4cfff` |

The quantized LM weights (shards 1–2) are byte-identical to the original text-only
`gptq-fp8-w8a16-mlp` checkpoint. Shard 3 contains 333 BF16 visual + 15 BF16 MTP
tensors extracted from the base model — no re-quantization was performed on these.

---

## Hardware & Runtime

| Field | Value |
|---|---|
| GPU | 4× NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 96 GiB GDDR7 each) |
| Driver | 595.58.03 |
| Benchmark GPU count | 1 |
| VRAM used | 86,616 / 97,887 MB (88.5%) |
| Temp (avg / max) | 66.3 °C / 74.0 °C |
| GPU util (avg) | 91.0% |
| Power (avg / max) | 280 W / 302 W |

---

## Reproduction

### Step 1: FP8 W8A16 weight-only GPTQ quantization

```bash
# Inside Docker (vLLM infernal-invocation image, llm-compressor, SM120)
CUDA_VISIBLE_DEVICES=0 SUITE_DIR=/data/suite-v3 \
  python scripts/quant_gptq_unified.py \
  --model /data/models/Qwen3.8-27B-bf16 \
  --output /data/models/Qwen3.8-27B-gptq-fp8-w8a16-mlp \
  --format fp8-w8a16-mlp --calib-samples 181
```

Calibration tokens from `$SUITE_DIR/tokens` (default `/data/suite-v3/tokens`).
Script: `scripts/quant_gptq_unified.py`. Format: `fp8-w8a16-mlp` (FP8 E4M3
per-channel weight-only for all Linear, BF16 activations, GPTQ Hessian correction).

### Step 2: Merge visual tower + MTP from base

```bash
# Merge quantized LM shards with BF16 visual+mtp from base model
python /workspace/scripts/merge_vision_mixed.py \
  --src /models/Qwen3.8-27B-gptq-fp8-w8a16-mlp \
  --base /models/Qwen3.8-27B-bf16 \
  --output /data/models/Qwen3.8-27B-FP8-Pessoa
```

Script: `scripts/merge_vision_mixed.py`. Extracts 333 visual + 15 MTP tensors
from the BF16 base, writes them as a 3rd shard, copies the 2 quantized shards
unchanged, and patches `config.json` to `Qwen3_5ForConditionalGeneration` (VLM)
with `vision_config`, `text_config`, and `re:^mtp.*` / `re:.*visual.*` added to
the quantization ignore list.

---

## Limitations

- **Weight-only means slower prefill.** BF16 activations require dequantization
  before GEMM, making prefill ~18% slower than W8A8 (4,875 vs 6,213 tok/s). Decode
  speed is comparable (45.6 vs 46.6 tok/s). If prefill throughput matters more than
  KLD, use the W8A8 variant (v17).
- **Blackwell-optimized.** FP8 E4M3 MMA is supported on SM120 (Blackwell),
  SM89/SM90 (Ada/Hopper), but the benchmark and KLD capture were run on SM120 only.
  Validate on your hardware.
- **Body-only KLD.** Fidelity measured against a shared BF16 LM head; end-to-end
  generation quality may differ slightly.
- **Literary domain degradation.** 5.1× higher KLD on literary text vs scientific —
  creative/prose generation is disproportionately affected, though less severely
  than in W8A8 or 4-bit configs.
- **MTP speculative decoding acceptance** depends on vLLM version and configuration.
  Ensure `re:^mtp.*` is in the quantization ignore list (it is, in this checkpoint)
  so vLLM's compressed-tensors loader leaves the BF16 MTP weights untouched.
- **Quantization is lossy.** Validate on your own workload before production use.

---

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

> *"I am the empty stage where the actors perform their various roles."*
> — Bernardo Soares (heteronym of Fernando Pessoa), *The Book of Disquiet*
