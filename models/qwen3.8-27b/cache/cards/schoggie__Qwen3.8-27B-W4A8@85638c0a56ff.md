---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - vllm
  - compressed-tensors
  - quantized
  - w4a8
  - int4
  - int8
  - multimodal
  - vision
---

# Qwen3.8-27B — W4A8 (INT4 weights / INT8 dynamic activations)

INT4 group-128 weights with **INT8 dynamic per-token activations**, produced with
[llmcompressor](https://github.com/vllm-project/llm-compressor) (`compressed-tensors` 0.18.0,
`pack-quantized` format). Data-free symmetric RTN — no calibration set.

Built and validated on **2× NVIDIA CMP 170HX** (GA100 silicon, sm_80, 64 GB after unlock) —
i.e. a card with no INT4/INT8 marketing support and no profiler access. It runs fine.

## Why W4A8 instead of plain INT4 (W4A16)

On a bandwidth-rich but compute-limited card, batch-1 decode is bound by streaming the
weights, so activation precision buys nothing there. The moment you batch — or prefill —
the dequantising GEMM becomes the bottleneck, and INT8 activations relieve it.

Measured on one CMP 170HX, vs the same model as INT4 AWQ (W4A16), vLLM, FlashInfer, fp8 KV:

| workload | INT4 (W4A16) | **W4A8** | delta |
|---|---|---|---|
| Prefill @8K | 1884 tok/s | **2539 tok/s** | **+35%** |
| Prefill @32K | 1657 tok/s | **2171 tok/s** | **+31%** |
| Decode np=1 | 21.53 ms/step | 21.82 ms/step | −1% (wash) |
| Decode np=4 | 34.78 ms/step | 32.16 ms/step | +8% |
| Decode np=8 | 51.59 ms/step | 45.09 ms/step | +14% |
| Decode np=16 | 88.33 ms/step | **70.88 ms/step** | **+25% aggregate** |

The gain scales with batch size and is ~zero at np=1 — exactly what you expect when the
fix relieves a compute-bound GEMM rather than a memory-bound one. Long-context end-to-end
(20 × 232K-token prompts) came in ~10% faster than INT4.

## Vision tower included (unquantized)

The full `model.visual.*` tower is shipped in **bf16**, exactly as in the upstream INT4
release: the vision encoder is in `quantization_config.ignore`, so only the language model
is W4A8. Architecture is `Qwen3_5ForConditionalGeneration`, and `preprocessor_config.json` /
`video_preprocessor_config.json` are included, so image and video inputs work out of the box.

Verified on a single CMP 170HX under vLLM: the ViT is instantiated
(`MMEncoderAttention`), the LM loads through `MarlinLinearKernel for
CompressedTensorsWNA8Int`, resident footprint 19.0 GB, and an image prompt is answered
correctly.

## Quality — gated at parity

Head-to-head against the INT4 AWQ model, same server config, fp8 KV cache:

| gate | INT4 AWQ | **W4A8** |
|---|---|---|
| Needle-in-a-haystack @256K, 20 depths | 20/20 | **20/20** |
| HumanEval pass@1 (greedy) | 93.3% | **92.1%** |
| Agentic tool-use, 7 build/verify/repair tickets | 144/144 checks | **144/144** |

The HumanEval delta is 2 problems out of 164 — not statistically significant (McNemar on the
discordant pairs is nowhere near it), though 10 problems do flip outcome, so the numerics
genuinely differ.

**What is and isn't quantized in the linear-attention layers.** Qwen3.8-27B is a hybrid
model — only 16 of its 64 layers are full attention; the other 48 are GatedDeltaNet linear
attention carrying a recurrent state. In those 48 layers:

- **W4A8:** `in_proj_qkv`, `in_proj_z`, `out_proj` (the large projections, ~23% of GEMM FLOPs)
- **bf16:** `in_proj_a`, `in_proj_b`, `conv1d`, `norm`

(An earlier version of this card said the `linear_attn` layers were entirely unquantized.
That was wrong — the `ignore` list names the parent `linear_attn` module, which does not
exclude its child projections. Thanks to BlairQ for spotting it in the weight index.)

Quantizing activations in the recurrent path risks error accumulating along the sequence; a
sibling linear-attention model lost ~30 points of NIAH under W8A8, and a W8A8 build of this
model lost ~5 points of HumanEval. Here NIAH@256K and the agentic gate hold, and HumanEval is
within noise of INT4, so the gates pass — but these three projections are the most likely
place for the remaining small gap. A variant with them left unquantized would trade roughly
a third of the prefill gain for it; that is untested.

## ⚠️ The one-line fix you need in vLLM, or this checkpoint silently does nothing

vLLM currently hardcodes the activation dtype when building the kernel config, so the
INT8 activation scheme is **silently dropped** and every `mp_linear` kernel runs plain
W4A16. There is **no error and no warning** — you just get INT4 speed and wonder why.

In `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8.py`
(~line 144, inside `CompressedTensorsWNA8Int.create_weights`):

```python
# before — activations silently ignored
            act_type=params_dtype,

# after
            act_type=(torch.int8 if self.input_quant is not None
                      and self.input_quant.num_bits == 8 else params_dtype),
```

With `act_type=torch.int8`, `is_a_8bit` becomes true and Marlin's int8 branch arms
(`marlin_act_int8_process_scales`). Keep Marlin **enabled** — do not set
`VLLM_DISABLED_KERNELS`.

Verified present in 0.27.1 and still present in 0.28.1rc1 (both `dev87` and `dev337`).

Easiest application, no rebuild — mount the patched file over the original:

```bash
podman run -d --name vllm \
  -v /path/to/patched/compressed_tensors_wNa8.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8.py:ro \
  -v /path/to/models:/models \
  vllm/vllm-openai:latest \
  --model /models/qwen38-w4a8 \
  --max-model-len 256000 --max-num-seqs 16 \
  --attention-backend FLASHINFER --kv-cache-dtype fp8 \
  --enable-prefix-caching
```

### How to check it actually took

**Do not trust the log line.** `Using MarlinLinearKernel for CompressedTensorsWNA8Int`
is printed **whether or not** the int8 branch armed. Check the prefill throughput instead:

- ~2500 tok/s @8K on one CMP 170HX → int8 armed ✅
- ~1900 tok/s @8K → silently degraded to W4A16 ❌

Or just read the running file: `sed -n 144p` on the path above.

## Notes

- Speculative decoding works: an external draft model (`num_speculative_tokens=7`) reached a
  mean acceptance length of **5.00** against this W4A8 target — higher than the 4.14 the same
  draft achieved against the INT4 target.
- The `int-quantized` format variant of this recipe is a dead end: it routes to a kernel that
  expects packed weights while `int-quantized` stores them unpacked. Use `pack-quantized`.
- Built from the official Qwen3.8-27B bf16 release. All credit for the model to the Qwen team;
  this is only a quantization.
