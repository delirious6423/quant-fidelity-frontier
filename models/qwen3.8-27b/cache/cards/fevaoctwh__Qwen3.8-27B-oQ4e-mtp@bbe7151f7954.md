---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
tags:
- mlx
- oq
- quantized
- qwen3.5
- multimodal
- image-text-to-text
- video-text-to-text
---

# Qwen3.8-27B-oQ4e-mtp

Mixed-precision 4-bit quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), converted with [oQ](https://github.com/jundot/omlx) (oMLX v0.5.7) **imatrix-calibrated, mixed-precision quantization** for Apple Silicon (MLX format).

Runs fully local with MLX in 24 GB+ unified memory.

## Model description

- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0)
- **Architecture:** `qwen3_5` (Qwen3_5ForConditionalGeneration), multimodal — text + image + video inputs
- **Format:** MLX safetensors (sharded, 4 files)
- **Size:** ~17 GB total
- **Context:** inherits base model context length (see base model card)
- **Quantization tool:** [oQ / oMLX](https://github.com/jundot/omlx) v0.5.7

## Quantization details

| Parameter | Value |
|---|---|
| Primary precision | **4-bit** (group_size 64, affine mode) |
| Sensitive layers | **5-bit** (group_size 64, affine mode) |
| Calibration | `oqe_code_multilingual`, 128 samples × 512 tokens (adaptive up to 1024), 504 recorded activations |
| imatrix cache | Freshly computed (not reused), stored in `oq_imatrix_report.json` |
| MoE | None (dense 27B) |

Layers kept at 5-bit to protect quality: all `linear_attn` projections, MLP `down_proj` in most blocks, and selected `self_attn` output/key projections. Everything else at 4-bit.

> Full per-layer bit map is in `config.json` → `quantization_config`.

## Quantization report

The repo ships `oq_imatrix_report.json` with the full imatrix collection stats:

- Calibration dataset: `oqe_code_multilingual`
- Samples: 128 requested, 128 processed (adaptive mode, max 1024)
- Sequence length: 512
- Activation entries recorded: 504 (615 Linear modules captured)
- Expert coverage: N/A (dense model)

## Requirements

```bash
pip install -U mlx omlx   # oMLX for oQ-family models; or plain mlx-lm
```

## Usage (oMLX / MLX LM)

```python
# OpenAI-compatible server (oMLX)
omlx serve fevaoctwh/Qwen3.8-27B-oQ4e-mtp

# Or via mlx_lm for a quick test
python -m mlx_lm generate\
  --model fevaoctwh/Qwen3.8-27B-oQ4e-mtp\
  --prompt "Please introduce yourself"
```

Or drop the folder into LM Studio/any MLX-compatible app by cloning the repo:

```bash
git clone https://huggingface.co/fevaoctwh/Qwen3.8-27B-oQ4e-mtp
```

## Files

| File | Purpose |
|---|---|
| `model-00001..00004-of-00004.safetensors` | Quantized weights (4 shards) |
| `config.json` | Architecture + per-layer quantization map |
| `oq_imatrix_report.json` | Imatrix calibration report |
| `chat_template.jinja`, `tokenizer.json`, `merges.txt`, `vocab.json` | Tokenizer & chat template |
| `preprocessor_config.json`, `generation_config.json` | Multimodal preprocessor & generation defaults |

## Intelligence benchmarks

Measured with the oMLX built-in Smart Benchmark (`/admin` dashboard), thinking mode on, fixed-seed sampling. Hardware: Apple Silicon M5 Max (128 GB unified memory).

| Benchmark | Sample | Score | Notes |
|---|---|---|---|
| MMLU-Pro | 300 / 12,032 | **80.7%** | Knowledge • 14 subjects (10-way) |
| GSM8K | 100 / 1,319 | **96.0%** | Grade-school math reasoning |
| HumanEval | 164 (full) | **98.2%** ⚠️ | See note below — re-run with standard unit tests |

> **Reference points (official, full-set):** Qwen3.8-27B's predecessor Qwen3.6-27B scores 86.2% MMLU-Pro; the same-generation Qwen3.6-35B-A3B 85.2% and Google's Gemma 4 31B 85.2% (all official model cards, 2026). This quantized 27B's sampled 80.7% sits a few points below the official full-set numbers of its non-quantized peers — expected for a sampled run with a different eval harness, and well within the "quantization costs a few points" range.

### Comparison vs. non-quantized models (same generation, 2026)

To calibrate where this quantized model actually stands, here are three **non-quantized** models from the same 2026 generation with **official** benchmark numbers (full-set, vendor-reported):

| Model | Type | Released | MMLU-Pro | LiveCodeBench v6 | SWE-bench Verified |
|---|---|---|---|---|---|
| **Qwen3.8-27B-oQ4e-mtp (this model)** | **Quantized 4/5-bit MLX, 27B dense** | **Aug 2026** | **80.7%** * | — (not run) | — |
| Qwen3.6-27B | Non-quantized, 27B dense | Apr 2026 | 86.2% | ~83.9 | 77.2% |
| Qwen3.6-35B-A3B | Non-quantized, 35B MoE | Apr 2026 | 85.2% | 80.4 | 73.4% |
| Gemma 4 31B | Non-quantized, 31B dense | Mar 2026 | 85.2% | 80.0 | 52.0% |

\*This card's MMLU-Pro is a **300/12,032 sampled run** via the oMLX harness (thinking mode), not the vendor's full-set protocol — expect the full-set number to be several points higher. GSM8K (96.0%, sampled) and HumanEval (98.2%, full, standard unit tests) are independent measurements: the 2026 official evals **no longer publish GSM8K/HumanEval** (saturated above ~90%), replacing them with AIME 2026 / LiveCodeBench v6 / SWE-bench, so no same-generation official comparison exists for those two. Against the 2025-generation Qwen3 family's official numbers (GSM8K ~94.4%, HumanEval ~93.3%), this quantized model is at or above.

*Sources: Qwen official model cards / blogs (Qwen3.6-27B, Qwen3.6-35B-A3B), Google DeepMind Gemma 4 card. Treat cross-harness differences of a few points as noise; the takeaway is that a 4/5-bit quantized 27B sits **within a few points of its non-quantized same-generation peers** on the shared benchmark (MMLU-Pro), while trailing them on agentic-coding benchmarks it was not tested against here.*

### ⚠️ HumanEval scoring note — original vs. actual

The oMLX built-in benchmark initially reported **10.4% (17/164)** for HumanEval. That number is **invalid** and should be ignored — it was caused by a bug in oMLX's code-extraction pipeline, not by the model:

| Step | Score | How it was produced |
|---|---|---|
| oMLX built-in (reported) | 10.4% | oMLX truncated the model's raw output at `...` when extracting code — **151/164** predicted snippets ended mid-statement (e.g. `if abs(numbers...`), making them syntactically invalid and failing every unit test. Only 24/164 of the extracted snippets even compiled. |
| Actual (this card) | **98.2% (161/164)** | Re-extracted the code from the saved `raw_response` of every question (after `</think>`, stripping code fences and trailing thinking residue), then executed the **standard HumanEval unit tests** (OpenAI `human-eval` test harness, `check(candidate)` asserts) locally with Python 3.11. |

The 3 remaining failures are generation truncation, not model capability: the thinking-mode output ran out of generation budget mid-function (`return string +`, `for i in range(n):\n for`), i.e. a benchmark configuration artifact. 154/164 of the model's raw outputs contained syntactically compilable code.

Full evidence: the exported oMLX benchmark result JSON contains both `raw_response` (the model's full raw output, incl. thinking) and `predicted` (what oMLX actually scored) per question — the truncation is visible there per-question. Re-run script and per-question results available on request.

## Performance benchmarks

Measured with the built-in oMLX benchmark (`/admin` dashboard), single-request mode:

- **Engine:** Auto
- **Context:** Code (Python)
- **Test pattern:** `pp<T> / tg128` = prefill T tokens, then generate 128 tokens
- **Hardware:** Apple Silicon (M5 Max, 128 GB unified memory)

| Test | TTFT (ms) | TPOT (ms) | Prefill (tok/s) | Generation (tok/s) | E2E (s) | Throughput (tok/s) | Peak Mem |
|---|---|---:|---:|---:|---:|---:|---:|
| pp1024 / tg128 | 1161.5 | 15.84 | 881.6 | 63.6 | 3.184 | 361.9 | 16.72 GB |
| pp4096 / tg128 | 5155.9 | 19.23 | 794.4 | 52.4 | 7.614 | 554.8 | 18.19 GB |
| pp8192 / tg128 | 11856.7 | 18.13 | 690.9 | 55.6 | 14.176 | 586.9 | 19.24 GB |
| pp16384 / tg128 | 25982.8 | 17.80 | 630.6 | 56.6 | 28.260 | 584.3 | 20.78 GB |

**Key takeaways:**

- **Generation speed** stays ~52–64 tok/s across context lengths — plenty for interactive chat and agent use
- **Prefill throughput** degrades gracefully with context length (882 → 631 tok/s from 1k → 16k)
- **Memory footprint** grows only ~4 GB from 1k to 16k context (16.7 → 20.8 GB) — fits comfortably in 24 GB-class machines
- TTFT at 16k context is ~26 s; expect long-prompt latency to scale with context size

## License

Apache-2.0 (inherited from the base model). The quantization itself is an independent transformation; check the base model card for the original terms.

## Disclaimer

This is a quantized model — minor quality degradation vs. the bf16 original is expected, particularly on complex reasoning and fine-grained multimodal understanding. Use at your own discretion; not intended for medical, legal, or safety-critical decisions.

## Acknowledgements

- [Qwen team](https://qwenlm.github.io/) for the base model
- [oMLX / oQ](https://github.com/jundot/omlx) for the mixed-precision quantization toolchain
