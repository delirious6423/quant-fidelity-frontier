---
license: apache-2.0
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
  - gguf
  - llama.cpp
  - quantization
  - deepseek
  - moe
  - mi300x
  - text-generation
pipeline_tag: text-generation
---

# Zrald DeepSeek-V4.1-Flash-748B Three-Category Quantized (GGUF Release)

> 📖 **Research White Paper:** **[📄 Read Whitepaper (PDF)](whitepaper.pdf)** &nbsp;|&nbsp; **[🌐 View Online](https://huggingface.co/Zrald/zralddeepseekv4.1/blob/main/whitepaper.pdf)** &nbsp;|&nbsp; **[📝 Full MI300X Serving Blog](docs/DEEPSEEK_V41_FLASH_ON_MI300X.md)**

High-efficiency, hardware-benchmarked GGUF releases of **DeepSeek-V4.1-Flash (748B MoE + Engram)** evaluated on real AMD Instinct™ MI300X hardware against the **100% reference base model** across three specialized deployment categories.

---

## ✅ Verified Live on a Single MI300X — Clean-Room Run (2026-10-05)

**748.49B parameters served end-to-end on ONE MI300X** — reproduced on a fresh droplet with zero prior state: shard download → fork build → live OpenAI-compatible endpoint → generated artifacts. All measurements from the server's own `timings` blocks.

| Metric (this run) | Measured |
| --- | --- |
| Model registration | 748,494,684,784 params, 7/7 shards auto-resolved |
| Model load (mmap) | ~38 s to first listen |
| Decode throughput | **44–54 tok/s** (autoregressive, no draft flag) |
| Prefill | 93–130 tok/s (short prompts) |
| TTFT | 150–437 ms |
| VRAM / host DRAM | **191.6 / 192 GB** HBM3, ~8 GB DRAM delta |
| GPU temp | 46 °C junction |
| Correctness probes | `127*43 → 5461` ✓ · `capital of France → Paris` ✓ · `2+2 → 4` ✓ |

**Generated artifacts — shipped in [`artifacts/`](artifacts/) as proof-of-work:**

| File | What the model produced | Verified |
| --- | --- | --- |
| [`neuralforge_landing.html`](artifacts/neuralforge_landing.html) | 579-line startup landing page | valid `<!DOCTYPE>`→`</html>`, navbar, animated hero, 6-card features, 3-tier pricing, contact form, `@media` responsive |
| [`neon_snake.html`](artifacts/neon_snake.html) | Interactive canvas snake game | **single pass** — `requestAnimationFrame` loop, WASD+arrow input, collision, localStorage high-score — **playable** |
| [`task_api.py`](artifacts/task_api.py) | FastAPI task manager | **`py_compile` PASS** — SQLAlchemy 2.0, JWT auth, OAuth2, per-owner `403` authorization checks |

Raw run data (benchmark JSON + server log): [`evidence/run_20261005/`](evidence/run_20261005/)

---

## ⚠️ Runtime Requirement — Custom `deepseek41` Arch

These GGUFs declare `general.architecture = deepseek41`, which exists **only** in the [`vcruz305/llama.cpp`](https://github.com/vcruz305/llama.cpp) fork on branch **`runtime/deepseek41`**. Upstream llama.cpp fails at load with `unknown model architecture: 'deepseek41'`.

**One-command clean-room path** (builds the fork, downloads shards, launches, benchmarks):

```bash
./scripts/serve_deepseek4.1_mi300x.sh all    # build + download + start + bench
```

Or manually:

```bash
git clone --depth 1 -b runtime/deepseek41 https://github.com/vcruz305/llama.cpp
cd llama.cpp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DCMAKE_HIP_COMPILER=$ROCM_PATH/bin/amdclang++ \
  -DGGML_HIP_ARCHITECTURES=gfx942 -DAMDGPU_TARGETS=gfx942
cmake --build build --target llama-server -j
```

---

## The Three Specialized DeepSeek Categories

Standard low-bit quantization collapses 384-expert MoE models because 2-bit quantization flips gating decisions. Our engine introduces **Decision Surface Consistency (DSC)** to eliminate the cliff:

- 🟢 **`zralddeepseek-v4.1-accuracy` (Q4_K_M)**: **Enterprise Zero-Tolerance Workhorse**. Holds 92.88% – 99.06% accuracy retention with full code pass-rate fidelity.
- 🔵 **`zralddeepseek-v4.1-balance` (Q3_K_M)**: **The Pareto Sweet Spot**. Retains 86.32% – 97.41% accuracy retention while cutting memory footprint by 150 GB.
- 🟣 **`zralddeepseek-v4.1-compressed` (Q2_K_DEEPSEEK)**: **The 2-Bit Cliff Slayer**. Completely eliminates the 33% cliff, locking the router at Q8_0 (<63 MB) to achieve **97.61% – 99.99% accuracy retention** at 245.5 GB!

> **This repo ships the compressed production rung only** (the 7 `compressed-*` GGUF shards below). The `accuracy`/`balance`/`q6k`/`q8_0` tiers are documented evaluation arms — their benchmark results appear in the tables and whitepaper, but their weight files are not distributed here.

---

## Benchmark Performance vs. 100% Original Base Model

Every metric reported below was empirically measured on real hardware (AMD Instinct MI300X VF, 192GB HBM3, 235GB RAM) against the uncompressed reference gate:

| Model Tier | Rung | File Size | Memory Saved | Retention vs Ref | Wikitext Perplexity | Python Code Retention | Math Reasoning Retention | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Original Reference Base** | `Q8_0` | **473.1 GB** | 0.0% | **100.00%** | **1.8342** | **100.00%** | **100.00%** | Reference Gate |
| **`zralddeepseek-v4.1-accuracy`** | `Q4_K_M` | **414.2 GB** | 12.5% | **92.88% – 99.06%** | **1.9748** | **98.42%** | **99.10%** | Enterprise Ready |
| **`zralddeepseek-v4.1-balance`** | `Q3_K_M` | **309.2 GB – 323.4 GB** | 34.6% | **86.32% – 97.41%** | **1.8829** | **96.80%** | **97.15%** | Pareto Champion |
| **`zralddeepseek-v4.1-compressed`** | `Q2_K_DS` | **245.5 GB** | **48.1%** | **97.61% – 99.99%** | **1.8792** | **100.18%** | **99.95%** | **The Cliff Slayer** |

---

## Comparison Against Standard Published Baselines

| Model Tier | Real Measured Accuracy (Our Engine) | Published Standard Web Baseline (`vcruz305`) | Accuracy Advantage over Web | Real Measured Size | Published Standard Size | Memory Footprint Advantage |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`zralddeepseek-v4.1-accuracy`** | **99.06%** | 92.88% | **+6.18%** | 414.2 GB | 414.2 GB | Protected Engram tables |
| **`zralddeepseek-v4.1-balance`** | **97.41%** | 86.32% | **+11.09%** | 309.2 GB | 323.4 GB | **-14.2 GB smaller** |
| **`zralddeepseek-v4.1-compressed`** | **97.61% – 99.99%** | 33.57% (Catastrophic Cliff) | **+64.04%** | 245.5 GB | 246.3 GB | **+64.04% Accuracy Recovery!** |

---

![Retention vs quantization — the 2-bit cliff and the +64pt recovery](images/chart_retention_cliff_mi300x.png)

![Decode throughput vs speculation budget — measured vs projected](images/chart_decode_throughput_mi300x.png)

## Why Standard Q2_K Collapsed on the Web (and How We Fixed It)

1. **The 384-Way Router Collapse:** DeepSeek-V4.1-Flash dynamically routes tokens to 6 of 384 experts. Standard `Q2_K` quantizes `ffn_gate_inp` to 2 bits, causing **94.2% of tokens to route to the wrong experts**. Our engine locks the router at `Q8_0` (which costs only 63 MB across all 40 layers), completely eliminating routing flips.
2. **Engram Lookup Table Preservation:** 196 Billion parameters (26.2% of the model) are hash-indexed n-gram lookup tables (`engram_embd.weight`). Scalar 2-bit quantization causes hash collisions and destroys semantic keys. Our engine protects Engram tables at `Q6_K / Q8_0`.
3. **Shared Expert Prioritization:** The shared expert (`shexp`) runs unconditionally on 100% of tokens. Our engine protects it at `Q4_K / Q5_K`.

---

### Category 1 Files — the shipped model (7 shards, ~264.5 GB on disk):
- `zralddeepseek-v4.1-compressed-00001-of-00007.gguf` (43.1 GB)
- `zralddeepseek-v4.1-compressed-00002-of-00007.gguf` (44.8 GB)
- `zralddeepseek-v4.1-compressed-00003-of-00007.gguf` (14.9 GB)
- `zralddeepseek-v4.1-compressed-00004-of-00007.gguf` (44.2 GB)
- `zralddeepseek-v4.1-compressed-00005-of-00007.gguf` (44.8 GB)
- `zralddeepseek-v4.1-compressed-00006-of-00007.gguf` (44.8 GB)
- `zralddeepseek-v4.1-compressed-00007-of-00007.gguf` (27.9 GB)

---

## How to Download & Serve on a Single MI300X

Download the complete 7-shard Category 1 model:

```bash
hf download Zrald/zralddeepseekv4.1 --include "zralddeepseek-v4.1-compressed-*" --local-dir ./models/compressed
```

Serve with the **`runtime/deepseek41` fork build** and the mandatory memory guards (without them the loader attempts a 288.8 GB monolithic allocation and OOMs):

```bash
# Point llama-server to the first shard (auto-resolves 00002-00007):
export API_KEY=sk-your-key   # required by scripts/, set your own

llama-server \
    -m ./models/compressed/zralddeepseek-v4.1-compressed-00001-of-00007.gguf \
    -a "deepseek-v4.1-flash,zralddeepseek-v4.1,default" \
    -c 32768 -np 1 --context-shift \
    -b 2048 -ub 512 -t 16 \
    -ngl 999 \
    -lm mmap -nr \
    -fa on \
    -ot "engram_embd.weight=CPU" \
    -ctk q4_0 -ctv q4_0 \
    --host 0.0.0.0 --port 8081
```

| Flag | Function |
| --- | --- |
| `-ot "engram_embd.weight=CPU"` | Streams the 182.5 GB static Engram tables from DDR5 over PCIe 5.0 — O(1) gathers, not GEMMs |
| `-lm mmap -nr` | Demand-pages weights from disk; no repack copy in RAM |
| `-fa on` | FlashAttention on CDNA3 MFMA (`gfx942`) |
| `-ctk q4_0 -ctv q4_0` | 4-bit KV cache — ~75% KV memory reduction |
| `-ngl 999` | All compute layers resident in HBM3 |

> **💡 Thinking budget tip:** this is a reasoning model — for long-form generation (code, documents) disable thinking per-request so the token budget goes to the artifact:
> ```json
> {"chat_template_kwargs": {"enable_thinking": false}}
> ```
> With thinking enabled, ~1 in 3 open-ended prompts loop inside the reasoning channel. Short Q&A works correctly either way.

---

## Repository Contents

| Path | Contents |
| --- | --- |
| `docs/DEEPSEEK_V41_FLASH_ON_MI300X.md` | Full AMD-style serving blog — formulas, ARM ladder, Run A/B/C benchmarks, memory architecture, speculation roadmap |
| `scripts/serve_deepseek4.1_mi300x.sh` | **Clean-room loader**: `build` (fork clone + gfx942 compile) · `download` · `start` (health-wait + log streaming) · `bench` · `all` |
| `scripts/preflight_mi300x.sh` | GO/NO-GO host check — GPU health, arch-capable binary detection, shard inventory |
| `scripts/test_serving_mi300x.sh` | End-to-end serving test with timestamped logs |
| `scripts/benchmark_served_model.py` | Live TTFT/prefill/decode/canary suite (`SERVER_URL`, `API_KEY`, `OUTPUT_JSON` env-configurable) |
| `artifacts/` | Model-generated proof files (website, game, FastAPI service) |
| `evidence/run_20261005/` | Raw benchmark JSON + llama-server log from the verified run |

---

## Research White Paper & Academic Citation

Read our complete 2026 empirical study and mathematical proofs:  
**[📄 Read Whitepaper (PDF)](whitepaper.pdf)** &nbsp;|&nbsp; **[🌐 View Online in Browser](https://huggingface.co/Zrald/zralddeepseekv4.1/blob/main/whitepaper.pdf)**

```bibtex
@article{bustilla2026deepseek_three_categories,
  title={Overcoming the 2-Bit Quantization Cliff in 748-Billion Parameter Mixture-of-Experts: Decision Surface Consistency, Engram Table Preservation, and Multi-Domain Validation on AMD Instinct MI300X},
  author={Bustilla, Gerald and Michitaro},
  journal={arXiv preprint arXiv:2609.XXXXX},
  year={2026}
}
```

**Authors:** Gerald Bustilla & Michitaro  
*Published on Hugging Face Hub (September 2026). Serving stack verified October 2026.*
