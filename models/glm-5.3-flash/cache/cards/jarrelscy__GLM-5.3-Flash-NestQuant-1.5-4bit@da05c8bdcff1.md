---
license: mit
base_model: zai-org/GLM-5.3-Flash
pipeline_tag: image-text-to-text
tags:
- vision
- multimodal
- image-text-to-text
- quantization
- moe
- glm
---

# GLM-5.3-Flash NestQuant — run on one DGX Spark

## One command: model, image, serving, and complete benchmark

Run one of these commands on a separate scorer workstation with SSH access to an idle native ARM64 DGX Spark. The launcher securely prompts for the serving API key, downloads and prepares the pinned model and ARM64 Docker image on the Spark, launches the exact accepted 2,630-slot configuration, opens an authenticated SSH tunnel, installs the pinned official scorer, and runs all 80 tasks twice.

**Reasoning on — reproduces the canonical 89.2151 run configuration:**

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/jarrelscy/spark-bench/228ff83a27608189efb82349e4adf13bfb253e8f/scripts/reproduce_nestquant_single_spark.sh) --spark USER@SPARK_HOST --reasoning on
```

**Reasoning off — reproduces the 86.6870 run configuration:**

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/jarrelscy/spark-bench/228ff83a27608189efb82349e4adf13bfb253e8f/scripts/reproduce_nestquant_single_spark.sh) --spark USER@SPARK_HOST --reasoning off
```

The reasoning-on command was cold-tested end to end on a rebooted Spark: the filtered 434-file serving snapshot, all 42 expert-layer packs, backbone conversion, guarded native startup, authenticated tunnel, pinned scorer checkout, **73/73 golden-gate checks**, structured-tool preflight, and official IF-01 score 1.00 with zero errors all passed. Reasoning off uses the same tested launcher and serving path with the pinned thinking-off template. Add `--smoke-test` for that short validation path, `--reuse-serving` for an already healthy service on Spark port 8001, or `--help` for SSH identity, known-hosts, output, port, and keep-serving options. Full-run scores are stochastic observed outcomes; the command pins the methodology and configuration, not the generated trajectories.

## Validated Spark Bench results: thinking on and off

Both modes were tested on one native ARM64 DGX Spark with the same 2,630-slot, 20%-fixed, 75% jT / 25% EMA configuration.

| Mode | TrueScore | Quality | Completion | Evidence |
|---|---:|---:|---:|---|
| **Thinking on** | **89.2151** | **94.1884** | 160/160, zero transport errors | [GitHub traces](https://github.com/jarrelscy/spark-bench/tree/39b1edbcf41058e3313491a062e95b0fa87d6a11/results/nestquant-spark-thinking-on-20261015) · [HF evidence](evaluations/spark-bench-thinking-on-20261015) |
| **Thinking off** | **86.6870** | **84.3660** | 160/160, zero transport errors | [GitHub traces](https://github.com/jarrelscy/spark-bench/tree/d6efbe8/results/nestquant-spark-sampled-ahead-20261013) |

Use `"chat_template_kwargs":{"enable_thinking":true}` with `--thinking on` for the thinking-on result. Use `"chat_template_kwargs":{"enable_thinking":false}` with `--thinking off` for the thinking-off result. All other serving, allocation, sampling, context, and benchmark settings were matched. The thinking-on run's 15 late transport-error placeholders were rerun with exact scenario/repeat selectors under the unchanged configuration before official full-run aggregation.

## Recommended default: thinking on

| Metric | Validated result |
|---|---|
| **Spark Bench TrueScore** | **89.2151 /100** — 80 tasks ×2 repeats, **160/160**, validity PASS |
| **Quality** | **94.1884 /100**, thinking on, zero transport errors |
| **Decode KLD** | **0.09006744 KL(teacher ∥ student)** — all **8,188 positions**, four pinned BF16 teacher windows |
| **Hardware** | **One native ARM64 NVIDIA DGX Spark GB10** |

**Canonical configuration:** sampled-ahead asynchronous delivery; **1.5-bit base /4-bit hot experts**; **2,630 active slots =526 fixed (20%) +2,104 floating**, plus8spares; **75%jT /25%EMA**, updated every committed token (G=1), U050 flat50, half-life64/hysteresis4. **MTP2**, TP1, one sequence, **262,144-token context /2GiB KV**, FP4-g16 MLA +FP8 KV, eager, prefix caching off; **temperature0.7, top_p0.95, thinking on, native uncapped**.

**[Start on one Spark with Docker ↓](#run-on-one-spark-with-docker)** · [Official benchmark result and 160 transcripts](https://github.com/jarrelscy/spark-bench/tree/39b1edbcf41058e3313491a062e95b0fa87d6a11/results/nestquant-spark-thinking-on-20261015) · [KLD evidence and raw captures](https://huggingface.co/jarrelscy/GLM-5.3-Flash-NestQuant-1.5-4bit/tree/98e1106f01798a942fb824c55d1e223a4dd8a0b7/evaluations/spark-decode-sampled-ahead-20261013) · [Pinned source recipe](https://github.com/jarrelscy/nestquant/tree/spark-bench-reproduction-20261015-v9/spark/flash/sampled_ahead_sentinel)

The tested setup uses this **linux/arm64 Docker image plus the pinned source overlays**:

```text
docker.io/jarrelscy/nestquant@sha256:3859a4e0018ec915ca0d4a3f9488158026fc2525534051433e2b37a729102364
```

Requirements: native Spark, Docker GPU support, at least400GB disk, and the tested host-memory telemetry at `/proc/spl/kstat/zfs/arcstats`. The setup instructions below include download, preparation, staging, and guarded launch. The published recipe passed a cold native Spark E2E. The launcher automatically reboots an idle Spark after first-time packing to clear unified-memory file cache, then resumes startup.

This **Flash model uses1.5-bit base /4-bit hot routed experts**, not1.75/4. It requires the NestQuant runtime and prepared expert files; stock Transformers, vLLM or MLX cannot load it directly. The Docker image plus the pinned source overlays below are both required.

## Tested configuration

| Setting | Validated Spark default |
|---|---|
| Predictor | 75% jT / 25% EMA, updated every committed token (G=1) |
| Active expert slots | 2,630: 526 fixed (20%) + 2,104 floating; 8 additional spares |
| Layer allocation | U050 / `spark_256K_U_2630_flat50` |
| EMA / hysteresis | Half-life 64 / 4 |
| Speculative decoding | MTP2, probabilistic |
| Context / KV budget | 262,144 tokens / 2 GiB |
| Cold E2E launcher context / KV | 110,592 tokens / 1 GiB (Spark memory-safe default) |
| Attention cache | FP4-g16 MLA; FP8 KV |
| Serving | TP1, one sequence, eager, prefix cache disabled |
| Sampling | Temperature 0.7, top_p 0.95, thinking on |
| Pacing | Native, uncapped |

The published 89.2151 and 86.6870 full-run artifacts used the original 262,144-token / 2 GiB KV setting shown above. The tested one-command launcher defaults to 110,592 tokens / 1 GiB because the 2 GiB pool crossed the LXC memory guard during cold startup. This does not change expert slots, predictor, MTP, sampling, or scorer methodology; full-run scores remain observed artifacts rather than a promised rerun value.

### Effective routed-expert bits per weight

The 2,630 active hot slots cover **21.74% of the 12,096 routed experts** (42 layers × 288 experts). With a nominal 1.5-bit base and 4-bit hot experts, this allocation is **2.044 nominal bits per routed-expert weight**:

`1.5 + (4 − 1.5) × 2630 / (42 × 288) = 2.0436 bpw`

The 526 fixed slots are included in the 2,630 total, not additional slots. The eight transition spares are excluded from active hot capacity. This is the capacity-weighted nominal expert precision, not measured activation hit rate, salience coverage, whole-model storage bpw, or total RAM usage; scales, metadata, backbone, KV cache and runtime buffers add overhead.

## Run on one Spark with Docker

Use native ARM64 DGX Spark with Docker GPU support and the tested ZFS host-memory telemetry exposed at `/proc/spl/kstat/zfs/arcstats`. Start on an idle machine. Keep at least400GB disk for checkpoint, packed weights and caches; diagnostic teacher/student logits need additional space.

```bash
git clone --branch spark-flash https://github.com/jarrelscy/nestquant.git
cd nestquant
git checkout spark-bench-reproduction-20261015-v9
export NQ_FLASH_DATA="$HOME/nq-flash"
export NQ_FLASH_IMAGE='docker.io/jarrelscy/nestquant@sha256:3859a4e0018ec915ca0d4a3f9488158026fc2525534051433e2b37a729102364'
export NQ_FLASH_CONTAINER=nq-spark-kld-sampled-ahead-sentinel
export NQ_THINKING="${NQ_THINKING:-on}"  # on or off
# Set VLLM_API_KEY privately in your environment; do not put it in shell history.
: "${VLLM_API_KEY:?Set your serving API key privately}"
docker pull "$NQ_FLASH_IMAGE"
# Skip fetch/prepare when these exact checkpoint and packed files already exist.
bash spark/flash/sampled_ahead_sentinel/run_spark_kld.sh fetch
bash spark/flash/sampled_ahead_sentinel/run_spark_kld.sh prepare
python3 spark/flash/sampled_ahead_sentinel/recipe.py stage
python3 spark/flash/sampled_ahead_sentinel/recipe.py up
docker logs -f "$NQ_FLASH_CONTAINER"
```

`fetch` pins model revision `4023870a6f2c6097e37fa5d2e54040db64b011e9`. `prepare` is CPU-only and packs one layer at a time; retain the checkpoint because the backbone references unchanged files. Do not prepare weights while serving. The staged exact overlays live at `$NQ_FLASH_DATA/kld-sampled-ahead-sentinel`. The launcher verifies hashes, architecture, idle-container state and guards before starting. API: `http://127.0.0.1:8001/v1`, model `glm-5.3-flash-nq`. Keep the listener local or use an authenticated tunnel.

The capture hook does not force tokens unless its `force` file exists. Normal serving after `up` does not create that file. For graceful shutdown with evidence preserved:

```bash
python3 spark/flash/sampled_ahead_sentinel/recipe.py down
```

Archive previous diagnostic directories and remove/rename the stopped named container explicitly before a fresh diagnostic. The wrapper refuses to overwrite a prior run. The pinned recipe was tested on native Spark through fresh download and packing, staging, guarded startup, module-provenance checks, sentinel release, tool-call parsing and official scoring. Clean source-file cache is released after each byte-verified resident layer copy to keep startup within Spark memory.

## Reproduce the benchmark

Run scoring on a separate machine, using Spark Bench commit `f2ac7876c1782047fe4ac37290a1466daa3a1c05`. Install that checkout's dependencies. Forward local port 18975 to the Spark's localhost port 8001 and set `SPARK_BENCH_API_KEY` privately to the serving key.

```bash
export SPARK_BENCH_FORCE_TEMPERATURE=0.7
export SPARK_BENCH_EXTRA_BODY='{"top_p":0.95,"chat_template_kwargs":{"enable_thinking":true}}'
export SPARK_BENCH_CONTEXT=110592
export SPARK_BENCH_SEED_PER_REPEAT=42
unset SPARK_BENCH_REASONING_EFFORT
python -u spark_bench.py eval --endpoint http://127.0.0.1:18975/v1 \
  --model glm-5.3-flash-nq --label spark-native-full80-sampled-ahead \
  --topology single-dgx-spark-gb10 --parallelism TP1 \
  --spec-decode mtp2-probabilistic --repeats 2 --temperature 0.7 \
  --thinking on --uncapped --tier all --skip-throughput \
  --timeout 7200 --out-dir results
```

This runs all 80 scenarios with repeat seeds 43 and 44. Generated trajectories can differ despite matched settings. The published evidence includes the exact scorer outputs and per-repeat transcripts, so 86.69 is an observed result, not a promised score for every rerun.

## Decode KLD and throughput

Full-vocabulary decode KL(teacher || student): **0.09006744**, across all8188 positions in four pinned Brandonmusic BF16 confirmation windows. [Raw captures, scores, traces and hashes](https://huggingface.co/jarrelscy/GLM-5.3-Flash-NestQuant-1.5-4bit/tree/98e1106f01798a942fb824c55d1e223a4dd8a0b7/evaluations/spark-decode-sampled-ahead-20261013).

The separate512-token synthetic generation probe measured21.25decode tokens/s. It used a single-token prompt and repetitive output; this is not a representative Spark Bench throughput measurement. The full benchmark skipped the dedicated throughput sweep.

The final score includes CODE-13 repeat2 terminating at the context limit after257644 output tokens and receiving0 from the official scorer. The reported86.69 is the completed score, not a partial estimate. A separate thinking-on benchmark is in progress; no final thinking-on score is claimed.
