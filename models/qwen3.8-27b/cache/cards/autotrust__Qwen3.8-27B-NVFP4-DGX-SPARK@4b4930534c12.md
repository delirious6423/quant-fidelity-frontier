---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: vllm
tags:
- qwen
- qwen3.8
- nvfp4
- fp4
- mtp
- speculative-decoding
- modelopt
- dgx-spark
- gb10
- blackwell
---

# autotrust/Qwen3.8-27B-NVFP4-DGX-SPARK (lossless MTP speculative-decoding boost)



A **lossless speed distribution** of `nvidia/Qwen3.8-27B-NVFP4`: the exact same
weights, shipped with a fully benchmark-tuned MTP speculative decoding setup.
Single-stream decode on a DGX Spark (GB10) goes from **12.5 → 48.2 tok/s (3.9×)**.

- Weights are bit-exact identical to the upstream repo (21.9 GB, ModelOpt mixed
  precision: MLP → NVFP4, linear attention → FP8)
- The speedup comes from the speculative-decoding configuration plus a cheaper
  drafter. The target model's outputs are **completely untouched** — the lm_head
  used for verification and all 64 transformer layers are never modified, so at
  temperature 0 the tokens are identical with and without MTP.
- New in this repo: a 64,064-row draft-vocabulary mask (frequency-truncated over
  mixed ZH/EN/code/math corpora), drafter patch scripts, and the full benchmark
  report.

## Speedup (DGX Spark / GB10, single-stream, 24 × 1024 tokens, temperature 0)

| Config | tok/s | τ (mean acceptance length) | Speedup |
|---|---:|---:|---:|
| No MTP (baseline) | 12.5 | — | 1.0× |
| Native MTP k=6 (unoptimized drafter) | 37.5 | 4.84 | 3.0× |
| Native MTP k=7 (plateau) | 37.4 | 5.20 | 3.0× |
| **This recipe, k=8 (64k vocab + Marlin draft head)** | **48.2** | 5.29 | **3.9×** |


## Demo (DGX Spark)

<video controls muted playsinline preload="metadata" width="100%">
  <source src="https://huggingface.co/autotrust/Qwen3.8-27B-NVFP4-DGX-SPARK/resolve/main/dgx_spark_demo.mov" type="video/mp4">
  Your browser can't play this video inline —
  <a href="https://huggingface.co/autotrust/Qwen3.8-27B-NVFP4-DGX-SPARK/blob/main/dgx_spark_demo.mov">open dgx_spark_demo.mov</a>.
</video>

Screen recording on a DGX Spark: a coding benchmark script streams the model's reasoning and code, then executes
the generated program and checks its output
([dgx_spark_demo.mov](https://huggingface.co/autotrust/Qwen3.8-27B-NVFP4-DGX-SPARK/blob/main/dgx_spark_demo.mov), 4 min 49 s).

## Demo (NVIDIA B200)

<video controls muted playsinline preload="metadata" width="100%">
  <source src="https://huggingface.co/autotrust/Qwen3.8-27B-NVFP4-DGX-SPARK/resolve/main/b200_demo.mp4" type="video/mp4">
  Your browser can't play this video inline —
  <a href="https://huggingface.co/autotrust/Qwen3.8-27B-NVFP4-DGX-SPARK/blob/main/b200_demo.mp4">open b200_demo.mp4</a>.
</video>

The same weights and drafter recipe on one NVIDIA B200 (vLLM 0.31.0 OpenAI server, single stream, temperature 0,
MTP k=7 — the best setting on B200; k=8 is best on Spark). Four coding tasks: the model streams its thinking and code,
then the generated program is executed against tests (all pass). The video is a real-time replay of the measured token
arrival times (no speed-up or cuts); token counts include thinking + answer.

| Task | Tokens | Time | TTFT | Decode |
|---|---:|---:|---:|---:|
| Sharded LRU cache with TTL | 2594 | 5.80 s | 0.07 s | 453 tok/s |
| Arithmetic expression evaluator | 1896 | 3.92 s | 0.10 s | 496 tok/s |
| Dijkstra shortest paths | 396 | 0.88 s | 0.09 s | 497 tok/s |
| Fenwick tree (BIT) | 547 | 1.16 s | 0.08 s | 508 tok/s |

On the same B200 (24 prompts × 1024 tokens, single stream): no MTP 128.6 tok/s · native MTP k=7 444.0 tok/s ·
this recipe k=7 463.5 tok/s. B200 drafting is cheap, so the recipe adds ~4% there versus ~28% on DGX Spark.

## Speedup (DGX Spark / GB10, single-stream, 24 × 1024 tokens, temperature 0)

| Config | tok/s | τ (mean acceptance length) | Speedup |
|---|---:|---:|---:|
| No MTP (baseline) | 12.5 | — | 1.0× |
| Native MTP k=6 (unoptimized drafter) | 37.5 | 4.84 | 3.0× |
| Native MTP k=7 (plateau) | 37.4 | 5.20 | 3.0× |
| **This recipe, k=8 (64k vocab + Marlin draft head)** | **48.2** | 5.29 | **3.9×** |

By category (k=8): code 52.9 / math 56.9 / chat 44.4 / tool 45.2 tok/s.

**Why native MTP plateaus at 37:** on a bandwidth-bound machine, the drafter
reads the full 248k-vocab lm_head every draft step. Past k=6, that draft cost
eats the acceptance gains. This recipe cuts bytes-read per draft step from
~3.4 GB to ~0.42 GB (8×) via: ① 64k high-frequency draft vocabulary
(τ drops only 1.8%), ② draft lm_head requantized to NVFP4 Marlin W4A16
(GEMV 13791 → 757 µs), ③ NVFP4-quantized MTP layers. The verification lm_head
is untouched — hence lossless. Byte ledger and per-position acceptance curves
are in `BENCH_REPORT.md`.

### Temperature sensitivity (important)

The numbers above are at temperature 0 (greedy verification — the τ ceiling).
Higher sampling temperatures lower the acceptance rate: at temperature 0.7,
set `num_speculative_tokens` to **4–6** and re-benchmark for your workload.
Rule of thumb by workload: code/math 4–5×, open-ended chat 3–4×.

## Quick start (vLLM ≥ 0.31.0; verified on sm_120/121 Blackwell)

```bash
git clone <this repo> && cd Qwen3.8-27B-NVFP4-MTP-Fast
export MODEL_PATH=$PWD
```

### Option A — offline batch (in-process patch)

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # in-process engine so the patch applies
import sys; sys.path.insert(0, "scripts")

import real_draft
real_draft.install("artifacts/draft_vocab_64k.pt", "marlin", "marlin")

from vllm import LLM, SamplingParams
llm = LLM(
    model=os.environ["MODEL_PATH"],
    speculative_config={"method": "mtp", "num_speculative_tokens": 8,
                        "use_local_argmax_reduction": True},
    gpu_memory_utilization=0.60,   # prevents system OOM on unified-memory machines; zero speed cost
    max_model_len=32768,
)
print(llm.chat([{"role": "user", "content": "hello"}],
               SamplingParams(temperature=0, max_tokens=128))[0].outputs[0].text)
```

### Option B — vllm serve (OpenAI API)

`scripts/spark_mtp_draft.py` is a vLLM general plugin that injects the patch
into the EngineCore subprocess:

```bash
pip install -e plugin/          # registers the vllm.general_plugins entry point
export SPARK_DRAFT_HOME=$PWD \
       SPARK_DRAFT_VOCAB=$PWD/artifacts/draft_vocab_64k.pt \
       SPARK_DRAFT_HEAD_Q=marlin SPARK_DRAFT_MTP_Q=marlin
vllm serve $MODEL_PATH \
  --served-model-name qwen3.8-27b-nvfp4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":8,"use_local_argmax_reduction":true}' \
  --gpu-memory-utilization 0.60 \
  --max-model-len 131072 --kv-cache-dtype fp8 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

Measured on the served endpoint: 48–50 tok/s (temperature 0), 128k context,
FP8 KV cache (916k-token capacity), tool calling (the Qwen `<function=...>`
XML format) and reasoning separation (`reasoning_content`) all verified
end-to-end.

> The plugin activates only when `SPARK_DRAFT_VOCAB` is set, so the same venv
> keeps running unpatched servers.

## Repository layout

```
├── model-0000{1,2,3}-of-00003.safetensors   # bit-exact identical to upstream nvidia repo
├── config.json / hf_quant_config.json / tokenizer*
├── artifacts/
│   └── draft_vocab_64k.pt          # bool mask over 248,320 vocab; 64,026 kept (special tokens forced-kept)
├── scripts/
│   ├── real_draft.py               # drafter patch: truncated-vocab head + Marlin NVFP4 (offline path)
│   ├── nvfp4_ops.py                # Marlin/CUTLASS NVFP4 linear layer (dependency of real_draft)
│   └── spark_mtp_draft.py          # vLLM general plugin (serve path, entry-point wrapper)
├── plugin/                          # for pip install -e (pyproject registers the entry point)
├── sha256sums.txt                   # weight checksums (match upstream)
└── BENCH_REPORT.md                  # full benchmark report (microbenchmarks, per-position acceptance, marginal analysis)
```

## How it works (one paragraph)

GB10/Spark decode is locked to LPDDR5X bandwidth (~273 GB/s; without MTP every
token reads the full 17.6 GB of weights). MTP speculative decoding lets one
forward pass verify k drafted tokens — but the native drafter reads the full
248k-vocab lm_head every draft step, so raising k backfires. This recipe cuts
the draft-step cost 8× with the three pieces above, unlocking k=8–10.
Because the target model and its verification head are untouched, decoding is
lossless. See `BENCH_REPORT.md` for the full derivation.

## Compatibility & limits

- **Hardware:** Blackwell (verified on sm_120/121 GB10 / DGX Spark).
  Data-center Blackwell works too; the gains matter most on bandwidth-bound
  systems.
- **vLLM:** verified on 0.31.0 (`Qwen3_5ForConditionalGeneration` + ModelOpt
  NVFP4 path).
- τ is workload-dependent: deterministic outputs (code/math) accept more;
  open-ended chat less (see the temperature section).
- On unified-memory machines use `gpu_memory_utilization` 0.60 (0.85 trips the
  system OOM killer due to flashinfer autotune peaks; 0.60 measured at zero
  speed cost).

## Acknowledgements

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B);
  quantization [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
- Full report and reproduction scripts: `BENCH_REPORT.md` and `scripts/` in this repo.
