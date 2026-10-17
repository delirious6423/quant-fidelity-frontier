---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
language:
  - en
  - zh
tags:
  - qwen3.8
  - flash-next
  - nvfp4
  - compressed-tensors
  - dgx-spark
  - blackwell
  - vllm
  - mtp
  - uncensored
  - abliterated
  - vision-language
  - function-calling
  - reasoning
---

# Qwen3.8 Flash-Next Uncensored NVFP4 — DGX Spark

The Flash-Next checkpoint and runtime overlays used by the AI Companion cluster, packaged for reuse on an NVIDIA DGX Spark. This is a deployment package of OrcaRouter's existing quantized model; the weights have not been fine-tuned, merged, or requantized for this repository.

Upstream: [orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4), pinned to revision `38efbffeb2152219d979336d46371578c1177b98`. The original model card and configuration are preserved under [`upstream/`](upstream). [`provenance.json`](provenance.json) records checkpoint sizes, SHA256 hashes, the active runtime patches, and the image digest.

## Download

Download the complete public package with the Hugging Face CLI:

```bash
hf download DevelopingDad/Qwen3.8-Flash-Next-Uncensored-NVFP4-DGX-Spark \
  --local-dir ./flash-next-dgx-spark
```

The checkpoint occupies approximately **183.5 GB (170.9 GiB)** on disk, including the BF16 PLE embedding shard and the MTP weights. Allow additional space for the Docker image, download metadata, and compilation caches. The included launcher requires ARM64 Linux, an NVIDIA DGX Spark/GB10 with 128 GB unified memory, Docker, and the NVIDIA container runtime. This package has not been qualified on other hardware.

## Run on DGX Spark

```bash
cd flash-next-dgx-spark
python3 runtime/serve.py --dry-run
python3 runtime/serve.py --port 8000
```

The launcher fetches the exact ARM64 vLLM 0.30.0 image by registry digest if needed, verifies its image ID and the packaged overlay checksums, and starts an OpenAI-compatible API at `http://127.0.0.1:8000/v1`. It runs in the foreground; stop it with Ctrl-C. The download and image pull require internet access; the serving engine uses local files with Hugging Face/Transformers offline mode enabled.

The launcher mounts the bundled runtime patches automatically. An ordinary `vllm serve` invocation does not reproduce these patches. A generic Transformers, llama.cpp, or Ollama installation is not a verified loader for this package.

The runtime profile was read from a live production replica on October 9, 2026:

| Setting | Value |
| --- | --- |
| Tensor parallelism | 1 |
| Maximum model context | 131,072 tokens |
| Active sequence limit | 8 |
| Batched token budget | 8,192 |
| GPU memory utilization | 0.770 |
| KV cache | BF16 |
| Speculative decoding | Native MTP, 2 tokens, online FP8 draft weights |
| Draft vocabulary | Reduced English/code vocabulary, 47,172 entries |
| PLE | Memory mapped from local storage, parallel reads |
| CUDA graphs | Breakable piecewise graphs with the deployed capture sizes |
| Container memory limit | 100 GiB, with no swap allowance |
| Defaults | Thinking off; temperature 0.7; top_p 0.8; top_k 20 |
| Reasoning / tools | qwen3 / qwen3_xml |
| Runtime | vLLM 0.30.0, Transformers 5.17.0 |

The sequence limit is a scheduler limit; eight simultaneous full-context requests have not been qualified. The production pool also has infrastructure health checks and host memory guards; those services are separate from this standalone launcher. Run with sufficient available host memory and monitor the machine during initial loading and warmup.

## Our measured results

These are DevelopingDad's internal synthetic measurements on a **single DGX Spark/GB10**, collected on October 8, 2026, with the pinned OrcaRouter weights and vLLM 0.30.0. Both baseline and K1b ran the same frozen 52-request workload, with temperature 0, seed 91027, thinking disabled, and streaming enabled. Prose, code, and copy speed tests generated 512 tokens per request in three waves; several deliberately stopped at the output limit. The baseline was our earlier OrcaRouter serving profile, rather than the unquantized Qwen model or a competing model.

The K1b benchmark used **4 active slots, GPU memory utilization 0.790, 128K configured context**, BF16 KV, and the same patches included here. The baseline used 4 slots, 0.786, and 32K context. The bundled October 9 snapshot uses **8 slots and 0.770**; the 52-request speed suite has not been rerun at those settings. Results describe the measured profiles and are not a throughput guarantee for the bundled settings or other hardware.

### Throughput and first content

Values below are the median of three waves, measured as total completion tokens divided by full wave wall time, including prefill and scheduling. Multi-client rows report **aggregate** throughput, not per-user throughput. Single-client decode-only throughput is listed separately where useful. See [`results/performance.json`](results/performance.json) for the wave values and calculation details.

| Workload | Earlier profile | Optimized K1b | Change |
| --- | ---: | ---: | ---: |
| Prose, 1 client | 25.05 tok/s | **37.29 tok/s** | +48.9% |
| Prose, 2 clients, aggregate | 41.00 tok/s | **58.51 tok/s** | +42.7% |
| Prose, 4 clients, aggregate | 62.60 tok/s | **89.55 tok/s** | +43.1% |
| Code, 1 client | 37.33 tok/s | **51.60 tok/s** | +38.2% |
| Verbatim copy, 1 client | 45.39 tok/s | **55.43 tok/s** | +22.1% |

K1b's median single-client decode-only rates were **38.13 tok/s prose**, **52.80 tok/s code**, and **56.97 tok/s copy**. Cold document first-content measurements were **9.08 → 3.99 seconds** at 8,045 input tokens and **14.71 → 14.46 seconds** at 32,058 input tokens; both reported zero cached input tokens. A separate repeated-prefix 32K request returned first content in **0.93 seconds**, with 30,400 cached tokens. The cold-document results are single observations, not three-wave medians.

### Capacity, long context, and reliability

- **52/52 requests completed** in the K1b suite, with no HTTP failures, missing visible answers, or OOM; minimum sampled host MemAvailable was **15.80 GiB**. Completion here means a successful captured response, including intentionally length-limited speed probes, rather than a perfect quality score.
- Drafter-only online FP8 increased the measured KV pool from **232,825 to 448,822 tokens (+92.8%)**, with allocated KV memory growing from **11.08 to 13.89 GiB**. The target weights were unchanged; the target still verifies the draft tokens.
- A separate **96,080-token input** recalled a marker near the beginning correctly in **44.49 seconds** total. This was one synthetic fact-recall test, not broad long-context reasoning or document-comparison qualification.
- Four independent production replicas each passed a separate **8-request concurrent test**, with **30,075 input tokens per request**, a 256-token output cap, correct per-session reference codes, and a peak of eight executing sessions. **32/32 bounded requests** completed across these four tests, without OOM, automatic restarts, guard trips, or added preemptions. The slowest request per replica was **117.84–120.06 seconds**. These tests used 0.790 GPU memory utilization and were separate per-replica runs; they do not establish 32 simultaneous application requests or eight full-128K contexts on one Spark.
- Bounded deployment checks passed for grammar editing, conversation history, explicit reasoning, automatic/forced tools, structured JSON, SSE completion/usage, and synthetic image/video input. Application readiness and a two-turn writing conversation also passed during the deployment.

Sanitized supporting data: [`results/capacity-and-acceptance.json`](results/capacity-and-acceptance.json).

### Quality and limits

An assistant review of **20 synthetic diagnostic cases** scored K1b at **14/20 strict passes**, versus **13/20** for the preceding C1 runtime profile. A strict pass required full marks for factual fidelity, useful completion, and format. K1b received full marks on **14/20 fidelity**, **17/20 useful completion**, and **18/20 format** cases. Seventeen outputs reused scores from identical earlier outputs; three changed outputs were scored from their source material. The reviewer was not blind to prior reviews, and several cases remained flagged for human adjudication.

This is a small diagnostic regression check, **not a human gold evaluation or evidence of improved model quality**. Remaining misses included factual fidelity, requested length limits, and handling source disagreement or missing information. These runtime optimizations do not fix those model behaviors. The benchmark material was synthetic; there is no claim here about MMLU, GPQA, SWE-bench, safety alignment, production-scale user capacity, sustained soak behavior, or reboot recovery. See [`results/quality-summary.json`](results/quality-summary.json).

## API example

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DevelopingDad/Qwen3.8-Flash-Next-Uncensored-NVFP4-DGX-Spark",
    "messages": [{"role": "user", "content": "Rewrite clearly: The meeting have moved to Wednesday."}],
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

The endpoint also supports explicit reasoning, image/video input, structured JSON, and tool calling through the runtime. These capabilities passed bounded synthetic checks during the October 8 cluster deployment; the packaged launcher has been checked for syntax, command/profile parity, and artifact integrity, without starting a second GPU workload for this upload.

## What is included

- All 17 main weight shards, the separate MTP checkpoint, tokenizer files, model index, and media processor configuration from the pinned upstream revision.
- `config.json`: the normalized configuration mounted by the deployed engine, including its attention-layer naming fix. The original configuration is retained at `upstream/config.json`.
- `runtime/ngram_embedding-r3.py`: the deployed PLE memory-mapping and parallel-read implementation.
- `runtime/mtp-kv.py`: reduced draft vocabulary and online FP8 MTP draft weights.
- `runtime/low-latency-gemm-sm121.py`: the deployed GB10/SM121 GEMM selection patch.
- `runtime/draft_vocab_en_code_47k.txt`, a portable runtime profile, and the launcher.

Checkpoint LFS hashes match the original deployment's verified file manifest and the pinned upstream objects. Regular checkpoint files and the active overlays were freshly hashed on the serving host when this package was prepared; large weight shards were not re-read on the production host. Root configuration intentionally differs from upstream; all weight tensors are unchanged.

## Attribution and license

| Contribution | Attribution |
| --- | --- |
| Base model, architecture, tokenizer, vision stack, and native MTP | [Qwen — Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Abliteration and the checkpoint's NVFP4/FP8 quantization | [OrcaRouter](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4), revision `38efbffeb2152219d979336d46371578c1177b98` |
| Inference engine and original runtime modules | [vLLM contributors](https://github.com/vllm-project/vllm), v0.30.0, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`; original SPDX/copyright headers are retained |
| Reduced 47,172-entry English/code draft-token list | [Mia-AiLab](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4); local experiment records identify source revision `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`. The repository was unavailable when this card was prepared. The packaged list is pinned by SHA256 in `provenance.json`. |
| Community recipes that informed Spark PLE and speculative-decoding work | [tonyd2wild](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark) and [blazux](https://github.com/blazux/qwen3.8-Flash-DGX) |
| GB10/SM121 deployment modifications, online FP8 drafter integration, benchmarking, and packaging | [DevelopingDad](https://huggingface.co/DevelopingDad), building on the projects above |

**Weights and inherited model artifacts:** the original Qwen release carries the **Qwen Community License 1.0**, reproduced at [`LICENSE`](LICENSE). Its license is identical in the initial Qwen release and the revision checked for this package. OrcaRouter's preserved model card labels its checkpoint Apache-2.0; that label does not replace the original base-model license, so this package retains the Qwen license for the weights and documents the discrepancy explicitly.

**vLLM source and local runtime modifications:** Apache-2.0, reproduced at [`runtime/LICENSE`](runtime/LICENSE), with original source notices retained. The included token list is credited separately above; no new ownership of it or the upstream model is claimed. See [`NOTICE`](NOTICE) and [`upstream/README.md`](upstream/README.md) for the preserved upstream context.
