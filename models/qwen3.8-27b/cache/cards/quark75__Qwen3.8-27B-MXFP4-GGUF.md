---
base_model:
- Qwen/Qwen3.8-27B
tags:
- qwen3
- gguf
- mxfp4
- quantized
---


# Qwen3.8-27B — MXFP4

An MXFP4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), converted directly from the original BF16 safetensors. Same architecture as the base model (dense, hybrid Mamba/attention — `qwen35` arch with native SSM layers alongside standard attention blocks), same license (Apache-2.0).

## Why MXFP4 on a dense model

MXFP4 support in `llama.cpp` is usually associated with MoE models (it was added for OpenAI's `gpt-oss` family), and the built-in `MXFP4_MOE` quantization preset *only* applies MXFP4 to mixture-of-experts tensors — on a dense model like this one, using that preset directly would silently fall back to plain Q8_0 with none of MXFP4's benefit.

This build instead uses `llama-quantize --tensor-type <pattern>=mxfp4` overrides to force MXFP4 onto the dense linear weight tensors directly (`attn_qkv`, `attn_gate`, `ffn_gate`, `ffn_up`, `ffn_down`, `ssm_out`), while leaving norms, biases, and small SSM state parameters at their default (mostly Q8_0) precision.

The native MTP (multi-token prediction / speculative decoding) draft head bundled in the base model's weights (`blk.N.nextn.*` tensors) survives this conversion intact — confirmed via `--spec-type draft-mtp` in `llama.cpp`, ~82% draft acceptance observed in testing.

## Benchmarks

All benchmarks run on an AMD Radeon PRO V620 (RDNA2, ROCm/HIP backend), `llama.cpp` build `ece963f` (~August 2026 mainline), against the original Q4_K_M quantization of the same base model, same serving flags (`-kvu`, `--ctx-size 65536`, `--spec-type draft-mtp`, `-np 4`) unless noted.

### Speed

Measured with [llama-benchy](https://github.com/eugr/llama-benchy), 3 runs per point, `--pp 1024 --tg 128`:

| Context depth | Q4_K_M (pp / tg, tok/s) | MXFP4 (pp / tg, tok/s) | Gain |
|---|---|---|---|
| 0 | 249.5 / 24.1 | 325.3 / 34.1 | +30% / +42% |
| 4096 | 283.7 / 23.0 | 385.8 / 29.2 | +36% / +27% |
| 16384 | 276.4 / 22.8 | 373.3 / 29.7 | +35% / +30% |

File size: 16.9GB (MXFP4) vs. 17.1GB (Q4_K_M) — the speed gain comes at essentially no size cost.

**Serving tip**: if you don't need real request concurrency, `-np 1` measured 8–25% faster token generation than `-np 4` in this build (prompt processing was unaffected) — likely per-step scheduler overhead that scales with configured slot count, not something specific to MXFP4.

### Quality

Evaluated against a 39-prompt suite spanning logic/reasoning, coding, hallucination resistance, instruction-following, agentic tool-calling, multi-turn memory, and language-specific correctness (Rust, Yew/WASM, Python), graded twice by independent LLM judges:

| Judge | Scored | Average | Notable failures (< 7.0) |
|---|---|---|---|
| granite-8b | 38/39 | 9.09 | 2 |
| gemma4-31 (stricter) | 38/39 | 8.84 | 5 |

The stricter judge caught real, rubric-explicit failures the more lenient judge missed — worth knowing if you're comparing quantizations using a single judge model, results can look better than they are.

## Known limitations

Reported honestly rather than omitted:

- **Confident hallucination on obscure factual recall** (e.g., historical sports trivia): the model fabricated specific, wrong details with high stated confidence rather than expressing uncertainty, on a prompt explicitly asking it to flag anything it wasn't sure of. Both judges flagged this as a serious failure (scored 0.0–3.0). Not confirmed as MXFP4-specific — plausibly inherited from base model training data gaps.
- **wasm-bindgen API errors**: on a Rust/wasm-bindgen interop task, the model used the wrong crate (`js_sys` instead of `web_sys`) and wrong string type. Confirmed by both judges (4.0–4.5).
- **Occasional missed self-verification**: on a "write exactly N words" instruction-following task, the model undercounted its own output but claimed the correct count. Caught only by the stricter judge.
- **State-mutation bug on a Yew hooks task**: model mutated a state handle directly instead of calling `.set()` — a real compile-breaking error in the generated code. Caught only by the stricter judge; the lenient judge missed it and scored the response highly.
- **Non-convergent reasoning on certain precision-heavy factual questions**: on one prompt asking for an exact physical constant at a specific condition with "give sources," the model entered a very long non-terminating reasoning loop, consuming the entire available context window (tested up to 65,536 tokens) without producing a final answer, taking upwards of 30 minutes. **Confirmed to reproduce identically on the unquantized Q4_K_M base model** (same token count, similar wall-clock time) — this is a base-model behavior, not something this quantization introduced, but worth knowing if you're building anything latency-sensitive on top of this model: a query that triggers this failure mode will occupy a generation slot for a long time producing nothing.

## Recommended serving flags

```
llama-server \
  --model Qwen3.8-27B-MXFP4.gguf \
  --spec-type draft-mtp \
  --reasoning auto \
  -kvu \
  --flash-attn on \
  --ctx-size 65536 \
  --cache-type-k f16 --cache-type-v f16 \
  --batch-size 2048 --ubatch-size 1024 \
  -np 4   # or -np 1 for a small serving-speed bump if you don't need concurrency
```

## Provenance

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0)
- Conversion: `convert_hf_to_gguf.py` (BF16) → `llama-quantize` with `--tensor-type` overrides
- `llama.cpp` commit: `ece963f41b0b02d7a0d61436ae365762c073a4c8`