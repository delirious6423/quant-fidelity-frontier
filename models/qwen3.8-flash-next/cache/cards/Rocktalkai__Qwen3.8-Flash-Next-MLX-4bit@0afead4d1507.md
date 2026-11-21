---
license: other
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx
tags:
  - mlx
  - qwen4_exp
  - text-generation
  - apple-silicon
  - speculative-decoding
  - quantized
  - gmlx
language:
  - en
pipeline_tag: text-generation
---

# Qwen3.8-Flash-Next MLX 4-bit

First open-code MLX port of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(`qwen4_exp` architecture). Runs at **25–35 tok/s** on Apple Silicon with MTP speculative decoding
(batch-verify mode, lossless).

**Code repository:** [Rocktalk-Holdings/mlx-qwen4exp](https://github.com/Rocktalk-Holdings/mlx-qwen4exp)

> **Want vision, video, tool calling, and speculative decoding on Apple Silicon?**
> See [Fast multimodal serving with gmlx](#fast-multimodal-serving-with-gmlx-vision--video--mtp-drafter)
> below — measured 55–86 tok/s decode (content-dependent) on an M3 Ultra with
> the standalone MTP drafter in front of the 4-bit model.

---

## What is this model?

Qwen3.8-Flash-Next is Qwen's experimental preview of the Qwen4 architecture.
~180B total parameters: 125B MoE transformer (6B active, 512 experts, top-10+shared)
+ 51.2B n-gram hash embedding table + 4B MTP draft head.

Architecture highlights:

- **Hyper-connections**: 4-stream widened residual replaces all layer norms with gated low-rank mixers
- **QSA sparse attention** on 12/48 layers: lightning indexer selects top-2048-token budget
- **Gated DeltaNet** on 36/48 layers: sigmoid-gated linear attention with `1/sqrt(head_v_dim)` readout scale
- **PLE n-gram embedding**: 320M-row hash table, gathered per-token off memmap (~95 GB, optional)
- **MTP draft head**: 1-layer decoder for speculative decoding (66–86% acceptance rate)

See [mlx-lm issue #1798](https://github.com/ml-explore/mlx-lm/issues/1798) for the QSA sparse mask.

---

## Quantization

4-bit affine quantization (group size 64) via [mlx_qwen4exp](https://github.com/Rocktalk-Holdings/mlx-qwen4exp).

Per-path overrides that match the reference GGUF (Q8_0 for these):
- `embed_tokens`, `lm_head` → 8-bit g64
- `mlp.gate` (router), `shared_expert_gate` → 8-bit g64
- All 512 switch expert weights → 4-bit g64
- All other linear layers → 4-bit g64

MTP weights (`mtp-weights.safetensors`) are quantized with the same policy.

---

## Performance (M3 Ultra, 512 GB, Metal)

| Mode | tok/s |
|------|-------|
| Plain greedy | 26.4–26.6 |
| MTP-v2 speculative | 27.6–35.3 (avg 30.3) |

MTP-v2 acceptance: 66–86%, content-dependent. Token sequences identical to plain greedy (verified).

---

## Fast multimodal serving with gmlx (vision + video + MTP drafter)

The highest-throughput way we've found to run Flash-Next on Apple Silicon is
[gmlx](https://github.com/asher/gmlx) (GGUF-on-MLX inference platform, ≥0.4.4),
serving the 4-bit GGUF with the **standalone MTP drafter in front** and the
vision projector attached. This is our production deployment; everything below
is measured against the live server.

Three artifacts:

| Piece | Repo |
|-------|------|
| Target (4-bit GGUF, UD-Q4_K_XL, ~104 GB) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| MTP drafter (Q4_K_L, ~1.9 GB) | [ashbash/Qwen3.8-Flash-Next-MTP-Drafter-GGUF](https://huggingface.co/ashbash/Qwen3.8-Flash-Next-MTP-Drafter-GGUF) |
| Vision projector (`mmproj-F16.gguf`) | same Unsloth GGUF repo |

The drafter is the model's own 4B MTP head packaged as a separate GGUF: it has
no embedding or output tensors of its own — at load time it binds the target's
`embed_tokens`/`lm_head` and consumes the target's 4-stream pre-mixer hidden
state, so acceptance is far higher than a generic small-model drafter.
llama.cpp cannot run this drafter; gmlx is its intended runtime.

```bash
pip install gmlx
gmlx serve Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    --mmproj mmproj-F16.gguf \
    --speculative --draft-gguf Qwen3.8-Flash-Next-mtp-drafter-Q4_K_L.gguf \
    --host 0.0.0.0 --port 8912
```

### Measured throughput (M3 Ultra, 512 GB, 2026-08-31)

OpenAI-compatible `/v1/chat/completions`, temperature 0, thinking disabled
(`chat_template_kwargs: {"enable_thinking": false}`), ~121 GB resident:

| Workload | Tokens generated | Decode tok/s | Draft acceptance |
|----------|-----------------|--------------|------------------|
| Code (single-file JSON parser) | 3000 | **80.0** | ~0.85 |
| Repetitive (markdown table) | 3000 | **86.1** | 0.99 |
| Long-form prose (essay) | 2472 | 55.6 | 0.53 |
| Short answer | 83 | 69.6 | 0.69 |

Baselines on the same machine and quant with no drafter: llama.cpp Metal
26.4 tok/s, this repo's MLX port 26.4–26.6 tok/s plain greedy — the drafter is
a **2.1–3.3× speedup**, largest exactly where agents spend their tokens (code,
structured output).

### Vision, video, and tool calling (verified)

- **Images:** standard OpenAI `image_url` parts (https or base64 `data:` URI)
  against the `--mmproj` model. Chart reading test: all values and colors exact.
- **Video:** sample frames evenly across the clip (e.g. ffmpeg) and send them as
  ordered `image_url` parts. A 3-scene test clip came back scene-by-scene with
  all burned-in text read exactly; real-world TikTok clips parse cleanly in our
  production chat app through this path.
- **Tool calling:** OpenAI `tools` arrays return proper `finish_reason:
  "tool_calls"` with well-formed arguments — Flash-Next's agent-first training
  shows here.

---

## Usage

Install the code package:

```bash
git clone https://github.com/Rocktalk-Holdings/mlx-qwen4exp
cd mlx_qwen4exp
pip install mlx mlx-lm numpy transformers
```

Download this model:

```bash
huggingface-cli download RockTalk/Qwen3.8-Flash-Next-MLX-4bit \
    --local-dir /path/to/model-dir
```

Run:

```bash
python3 tools/run_mlx.py --model-dir /path/to/model-dir \
    --prompt "The capital of France is" --max-tokens 64

# With MTP-v2 speculative decoding (recommended):
python3 tools/run_mlx.py --model-dir /path/to/model-dir \
    --prompt "The capital of France is" --mtp-v2 --max-tokens 64
```

---

## Files in this repository

| File | Size | Description |
|------|------|-------------|
| `model-00001-of-00049.safetensors` … `model-00049-of-00049.safetensors` | ~71 GB total | 4-bit quantized transformer weights |
| `mtp-weights.safetensors` | ~1.4 GB | Quantized MTP draft head weights |
| `config.json` | — | Model config with quantization spec |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | — | Tokenizer |
| `ple_constants.json` | — | PLE n-gram hash constants (required for full quality) |
| `ngram_table.bin` | ~95 GB | Float16 n-gram embedding table (optional; model runs without it in degraded mode) |

The n-gram table is a large optional download. Without it, the PLE block is skipped and
a single warning is printed; output quality is slightly reduced but the model remains coherent.

---

## Limitations

- The MLX weights in this repo are text-only; for vision/video use the
  [gmlx serving lane](#fast-multimodal-serving-with-gmlx-vision--video--mtp-drafter) above
- QSA sparse mask is O(n²) above 2051 tokens (correctness preserved, memory scales with full KV)
- 4-bit expert quantization is the only serving-viable MLX option (6/8-bit is correct but ~50× slower)

---

## License

Model weights: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
Code: MIT ([Rocktalk-Holdings/mlx-qwen4exp](https://github.com/Rocktalk-Holdings/mlx-qwen4exp))
