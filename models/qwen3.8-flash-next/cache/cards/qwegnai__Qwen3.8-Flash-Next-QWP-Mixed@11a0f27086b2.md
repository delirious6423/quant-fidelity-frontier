---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: qwegn
pipeline_tag: text-generation
tags:
  - qwegn
  - apple-silicon
  - metal
  - quantized
  - hq8
  - hq4
  - vq2
  - moe
  - latent-sparse-attention
  - long-context
---

# Qwen3.8-Flash-Next · `HQ8_G32 + HQ4_G64` + `VQ2_G32` PLE · Qwegn native pack

A native weight pack for the **[Qwegn inference engine](https://github.com/qwegnai/qwegn)**, a Swift + Metal 4 runtime for Apple Silicon. Combines **mixed Hadamard quantization (`HQ8_G32` dense trunk + `HQ4_G64` routed MoE experts, `71.29 GiB`)** with **`QNGRAM v3 VQ2_G32` 2-bit `e-means` Per-Layer Embedding (PLE) tables (`27.13 GiB`, `-46.7%` vs `50.86 GiB` `q8g32`)** — shrinking total pack storage from `122.15 GiB` to **`98.42 GiB` (`-23.73 GiB`)** so the entire PLE table fits **100% resident in Metal GPU RAM on a 128 GB Mac** (**`95.1–113.7 tok/s` novel decode, `145.4 tok/s` speculative predictable, `3,035 tok/s` peak prefill, `10/10` core reasoning · `9/10` Code-10**) or streams from NVMe via `--stream-ngrams --no-load-vision` (**saving `-28.48 GiB` RAM**).

<p align="center"><img src="media/qwegn_model_flash_next.png" alt="Qwen3.8-Flash-Next HQ8+HQ4 + VQ2 PLE on Qwegn" width="960"></p>

<p align="center"><img src="media/qwegn_lsa_scaling.png" alt="Qwen3.8-Flash-Next 64K to 1M scaling on Qwegn" width="960"></p>

> This pack uses Qwegn's own `QWP1` + `QNGRAM v3` containers and loads only in Qwegn (`v0.14.0+`). It is not a safetensors, GGUF or MLX checkpoint.

## At a glance

| Property | Value |
| :--- | :--- |
| Upstream model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) @ `7ecef55a2e549814c62143fcf8b5be457695e3b9` |
| Parameters | 128.80 B `.qwp` backbone + `320,001,536 × 160` N-gram PLE rows |
| Architecture | 48-layer flagship hybrid MoE: 12 Query-Aware Sparse Attention (QSA) + 36 Gated-DeltaNet (GDN) layers, 512 experts (10 routed + 1 shared active), 4-branch Hyper-Connections (HC), 4-gram Per-Layer Embeddings (PLE), MTP-3 head, vision tower |
| Total size on disk | **98.42 GiB** (`71.29 GiB` `model.qwp` + `27.13 GiB` `assets/ngram.bin`; `-23.73 GiB` vs v2 PLE) |
| Backbone weights | `HQ8_G32` (`3.36%`) + `HQ4_G64` (`95.74%`) + `BF16` (`0.90%`), **4.75 bits/param** (`2.62×` lower KLD vs BF16 than uniform `HQ4_G32`) |
| PLE N-gram table | `QNGRAM v3 VQ2_G32` (`90 bytes/row`, 2-bit spherical `e-means` codebook, `320M` rows) |
| Native context | 262,144 tokens; verified to **1,048,576 (1M) tokens** with native QSA / LSA (`model.qlsa` included) |

## Performance on Qwegn (`v0.20.5 – v0.42.0`)

Apple M5 Max (`Mac17,6`, 128 GB unified memory, 40-core GPU, 585 GB/s), `paired_perf.py --strict`, `llmprobe`, & quiet-GPU `512–1M` scaling, `--kv-mode q4 --temperature 0`.

| Metric | Result |
| :--- | :---: |
| Prefill (`4K` / `8K` peak · `32K` / `64K` / `256K` / `1M`) | **2,947.3 / 3,034.8 tok/s** (`2,724.7` @32K · `2,385.1` @65K · `1,839.7` @256K · `1,887.6` @1M) |
| Cold TTFT / Warm prefix-cache TTFT | 113–155 ms / **7–11 ms (`63×–99×`)** |
| Repo `.qwcache` TTFT (`32.7K` Two-Tier Cortex) | **0.24 s** (`74.3×` vs `17.84 s` cold) |
| Standard decode (serial) | 53.4–81.5 tok/s |
| Speculative decode — novel (MTP-3 + GPU VQ2 PLE) | **95.1–113.7 tok/s** (`79.9 tok/s` CLI `4K`, `+13.9%` vs SSD v2 PLE) |
| Speculative decode — predictable (8-slot copy ring) | **145.4 tok/s** (`4.92 tok/step`) |
| 4-stream concurrent aggregate | **134.5 tok/s** |
| 1M-token prefill / decode (`e-means-4096` vs Native QSA) | **1,887.6 / 63.74 tok/s** (`+39.8%` pre / `+55.3%` dec vs `1,350.3 / 41.05`) |
| Core reasoning eval (`GPQA` / `SuperGPQA` / `AIME25` / `COMPSEC`) | **10 / 10 (`100%`)** (`9/10` on `v0.25.0`) |
| `v0.42.0 llmprobe --full` (`--stream-ngrams`, `mtp=3`, `40K q4`) | **`291 / 291 (100%)`** conformance · **`100%`** cap & fidelity (`36/36`) · **`76 / 88 (86.4%)`** agentic · **`65×`** cache (`845 → 13 ms`, `1541/1541` cached) · **`82.1 / 2,203 tok/s`** (`105.1 / 75.8`, `4.68/step`, `118.3` 4-stream) |
| `v0.42.0` Core-92, Hard-50 & External Evals (`--no-thinking`) | **`75 / 92 (82%)`** `Core-92` (`19/25` AIME25, `22/25` SuperGPQA, `17/24+1/1` GPQA, `16/17` COMPSEC) · **`44 / 50 (88%)`** `Hard-50` (`28/30` MMLU-Pro, `6/10` OlympiadBench, `5/5` LiveBench, `5/5` NIST Juliet) · **EvalPlus HE `93.9%` base / `92.1%+` (`154/164`)** · **EvalPlus MBPP `91.5%` base / `77.8%+` (`346/378`)** · **`87.8%`** IFEval (`n=30`) · **`93.3%`** GSM8K 5-shot (`n=30`) · **`8/8 (100%)`** Promptfoo |
| `v0.28.0 llmprobe --full` (`--stream-ngrams + DFlash2 w=8`) | **`291 / 291 (100%)`** conformance · **`100%`** cap & fid · **`74 / 88 (84.1%)`** agentic · **`83 / 100`** `Code-100` (**`48/50` `96%` HE**, `35/50` MBPP) · `60.3×` cache (`663 → 11 ms`) · `79.1 / 2,434.5 tok/s` (spec `135.8 / 71.4`, `5.05/step`) · **`9/9`** Law 28 QA |
| `v0.25.0` zero-workaround API, tool & reasoning eval (`inspect_ai` / `promptfoo`) | **`292 / 292 (100%)`** `--full` conformance · **`8 / 8 (100%)`** `promptfoo` · **`0.800 ± 0.1333`** `GPQA Diamond` (`40/40`, `responses_api=true`) |
| Code eval (`Code-10` · `Code-100`) & JSON fidelity (`9/9`) | **9 / 10 (`83 / 100` `v0.28.0`) · 9 / 9 (`100%`)** |
| Needle recall `16K → 1M` | **3 / 3 (`100%`) at every length** |

## Long context with Native QSA, LSA (`Ward-512`) & Capacity-Balanced `e-means` (`64K → 1M`)

| Context rung | Native QSA (`--lsa-mode auto`) Prefill / Decode | LSA `Ward-512` Prefill / Decode | **LSA `e-means-4096` (`qsa-cluster`) Prefill / Decode** | Needle recall |
| :--- | :---: | :---: | :---: | :---: |
| 64K *(GPU PLE)* | **2,279.9 / 72.67 tok/s** | 2,155.4 / 56.66 tok/s | 2,183.8 / 72.45 tok/s | **3 / 3** |
| 160K *(GPU PLE)* | **2,157.6** / 66.78 tok/s | 1,741.2 / 40.41 tok/s | 2,041.0 / **70.93 tok/s** (`+6.2%`) | **3 / 3** |
| 262K *(GPU / SSD PLE)* | 2,051.8 / **68.57 tok/s** | 1,522.1 / 32.38 tok/s | **2,057.6** / 63.54 tok/s | **3 / 3** |
| 524K *(`--stream-ngrams`)* | 1,721.9 / 54.49 tok/s | 976.2 / 18.70 tok/s | **1,969.6 / 68.10 tok/s** (`+14.4%` / `+25.0%`) | **3 / 3** |
| 1M *(`--stream-ngrams`)* | 1,350.3 / 41.05 tok/s (`770.1 s`) | — | **1,887.6 / 63.74 tok/s** (`550.9 s`, **`-3.65m` TTFT**) | **3 / 3** |

## Residency toggles & unified memory footprint

Qwegn `v0.14.0+` exposes orthogonal startup flags for multimodal vision planes and PLE N-gram residency so operators can trade GPU RAM headroom against NVMe streaming with zero pack modification:

| Residency flags (`16K` `q4` KV, `c = 4096`) | Vision | PLE (`ngram.bin`) | Combined RAM | Savings vs Full Residency |
| :--- | :---: | :---: | :---: | :---: |
| `--load-vision --no-stream-ngrams` | Loaded (FP32) | Metal GPU RAM | `~112.4 GiB` | Baseline |
| `--no-load-vision --no-stream-ngrams` *(default)* | Unmapped (`0 B`) | Metal GPU RAM | `~110.8 GiB` | **`-1.60 GiB` CPU RSS** |
| `--load-vision --stream-ngrams` | Loaded (FP32) | NVMe + `64 MiB` L1 | `~85.1 GiB` | **`-27.30 GiB` Metal GPU RAM** |
| `--no-load-vision --stream-ngrams` | Unmapped (`0 B`) | NVMe + `64 MiB` L1 | **`~83.9 GiB`** | **`-28.48 GiB` total RAM saved** |

Steady-state decode executes with **0 heap and 0 Metal buffer allocations**.

## Quantization breakdown

| Component | Format | Parameters / Rows | Size on disk |
| :--- | :--- | :---: | :---: |
| Routed MoE `gate_up` + `down` (`98` 3D stacks) | `HQ4_G64` (`4.50 bpw`) | `123.31 B` (`95.74%`) | `64.60 GiB` |
| Dense trunk, QSA, GDN, 4-branch HC, shared experts, `lm_head` (`597` planes) | `HQ8_G32` (`8.50 bpw`) | `4.33 B` (`3.36%`) | `4.54 GiB` |
| Embeddings, vision tower, routers & norms (`832` tensors) | `BF16` (`16.00 bpw`) | `1.16 B` (`0.90%`) | `2.15 GiB` |
| External 4-gram PLE table (`assets/ngram.bin`) | `QNGRAM v3 VQ2_G32` (`90 B/row`) | `320,001,536 × 160` | **`27.13 GiB`** |

## Quick start

Install Qwegn (`v0.25.0+` recommended, `v0.14.0+` minimum) from [https://github.com/qwegnai/qwegn](https://github.com/qwegnai/qwegn), then:

```bash
# Download the pack
qwg pull qwegnai/Qwen3.8-Flash-Next-QWP-Mixed

# Maximum throughput (GPU-resident 27.1 GiB VQ2 PLE on 128 GB Mac, vision off by default)
qwg serve ~/.qwg/models/Qwen3.8-Flash-Next-QWP-Mixed \
  --kv-mode q4 --context-tokens 262144 --lsa-mode auto

# Low-memory mode (-28.5 GiB RAM saved: streams VQ2 PLE from NVMe with 64 MiB L1 cache)
qwg serve ~/.qwg/models/Qwen3.8-Flash-Next-QWP-Mixed \
  --kv-mode q4 --context-tokens 262144 --no-load-vision --stream-ngrams

# Lossless DFlash 2 block-diffusion speculative decoding (v0.24.0+, auto-discovers Qwen3.8-Flash-Next-DFlash2-QWP_4)
qwg pull qwegnai/Qwen3.8-Flash-Next-DFlash2-QWP_4
qwg serve ~/.qwg/models/Qwen3.8-Flash-Next-QWP-Mixed \
  --kv-mode q4 --lsa-mode auto --stream-ngrams --mtp-backend dflash2
```

## Requirements

- macOS 26+ on Apple Silicon; M5-class GPUs (Apple10) enable NAX cooperative-tensor prefill
- 96+ GB unified memory recommended (`~84 GiB` with `--stream-ngrams --no-load-vision`, `~111 GiB` with GPU-resident VQ2 PLE)
- **Qwegn 0.14.0 or newer** (required for `QNGRAM v3 VQ2_G32` PLE gather and `--stream-ngrams`)

## Files

| File | Contents |
| :--- | :--- |
| `model.qwp` | Backbone weights in Qwegn's `QWP1` container (`71.29 GiB`, `1,527` tensors) |
| `model.qwp.sha256` | SHA-256 of `model.qwp` |
| `model.qlsa` | `QLSA0001` LSA projection sidecar |
| `assets/ngram.bin` | `QNGRAM v3 VQ2_G32` 4-gram PLE table (`27.13 GiB`, `320,001,536` rows) |
| `assets/ple_ngram_locations.json` | PLE shard map and prime router constants (`16` heads, order `4`) |
| `assets/` | Upstream config, tokenizer, chat template, `README.md`, `LICENSE`, and `manifest.json` |
| `README.md` / `media/` / `LICENSE` | This model card, infographics (SVG + PNG), and Apache-2.0 license |

## License and attribution

Apache-2.0, inherited from [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Quantized by the Qwegn team; all credit for the base model belongs to its original authors.
