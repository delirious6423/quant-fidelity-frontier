---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
quantized_by: wiesion
library_name: vllm
pipeline_tag: image-text-to-text
tags:
  - ornith
  - qwen3_5_moe
  - moe
  - autoround
  - gptq
  - int4
  - w4a16
  - intel
  - intel-xpu
  - xpu
  - battlemage
  - arc-pro
  - llm-scaler
  - vllm
---

# Ornith-1.5-35B-A3B — int4 AutoRound (GPTQ format) for Intel Arc Pro / llm-scaler

4-bit weight-only (W4A16, symmetric, group 128) quantization of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), made with Intel [AutoRound](https://github.com/intel/auto-round) and exported in **GPTQ format**, so the MoE experts load natively in Intel's vLLM fork **llm-scaler** on Arc Pro Battlemage GPUs. Runs the full 35B MoE with **4 × 64k context on one 24 GB Arc Pro B60** (or 6.8 × 128k on one 32 GB B70). With 64 GB of VRAM or more, use the official FP8 checkpoint instead.

> **Tested stack — everything on this card was measured with exactly this:**
> - `intel/llm-scaler-vllm:0.26.0-b2` — `intel/llm-scaler-vllm@sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b` (vLLM 0.26.1.dev0+g568afb3a1.xpu, torch 2.12.0+xpu, vllm-xpu-kernels 0.1.11.2; no IPEX)
> - Intel Arc Pro B60 (24 GB, chipset-attached PCIe 4.0 x4) and Arc Pro B70 (32 GB, CPU-attached PCIe 5.0 x8), Ryzen 9 9900X, Ubuntu 26.04, compute-runtime from the image. For single-GPU serving the PCIe link only affects model load time; prefill/decode differences between the cards come from Xe-core count (20 vs 32), not the link.
>
> **Measured October 2026.** llm-scaler is a fast-moving project: new images regularly fix some issues and regress others, for correctness *and* performance. Every number and known issue on this card is tied to the exact image above and may not hold for the next release — re-run the checks after upgrading (harness, eval and benchmark scripts are in `eval/`).
>
> **Not tested:** upstream vLLM (neither Intel XPU nor CUDA), transformers, llama.cpp, OpenVINO. It is a standard GPTQ checkpoint (sym, g128), so upstream CUDA vLLM *should* load it via GPTQ/Marlin — unverified.

## Why this repo exists

- **AutoRound quality:** the weights are quantized with AutoRound's signed-gradient rounding (SignRound, 200 iterations per block on 128 × 2048-token calibration samples) instead of plain round-to-nearest.
- **Loads natively on llm-scaler:** exported in GPTQ format (`quant_method: gptq`), so vLLM uses AutoGPTQ → `XPUwNa16LinearKernel` for linear layers and the XPU WNA16 MoE backend (`XPUExpertsWNA16`) for the experts.
- **Validated end-to-end on Arc Pro:** correctness harness, recovery vs the official FP8 checkpoint, B60/B70 performance, vision check — with scripts and raw results in `eval/`.
- **Sized for 24 GB:** everything except the MoE routers is int4, so 4 × 64k context fits on one Arc Pro B60.

Two common 4-bit routes do not work on `intel/llm-scaler-vllm:0.26.0-b2`:

- **On-the-fly `--quantization sym_int4`** from the bf16 checkpoint loads, but the MoE produced garbage in our tests (cf. [llm-scaler #671](https://github.com/intel/llm-scaler/issues/671)).
- **AutoRound's default export (`quant_method: auto-round`)** is routed to vLLM's INC backend, which on XPU in this image does not handle quantized MoE layers and falls back to the unquantized MoE method (per the loader code in the image; not run unmodified here).

Other GPTQ-format int4 builds of this model may load through the same path; they were not tested here.

## What is quantized

| component | format | size |
|---|---|---|
| experts (256 × 3 per layer), shared expert, attention + linear-attention projections | int4 GPTQ, fp16 scales | 17.7 GB |
| vision tower | bf16 (unchanged) | 0.9 GB |
| MoE routers, `shared_expert_gate`, norms | bf16 (unchanged) | <0.1 GB |
| embeddings, `lm_head` | bf16 (unchanged) | 2.0 GB |
| MTP head (`model_extra_tensors.safetensors`) | bf16 (unchanged) | 1.7 GB |

GPU footprint (vLLM "weights"): **18.3 GiB** text-only, **19.1 GiB** with vision. The MTP head is only loaded if speculative decoding is enabled (not recommended on llm-scaler, see Limitations).

## Quick start (llm-scaler)

```bash
docker run -d --name ornith35b --privileged --ipc host \
  --device /dev/dri:/dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  -v ~/.cache/huggingface:/root/.cache/huggingface -p 8001:8001 \
  -e ZE_AFFINITY_MASK=0 \
  -e NEOReadDebugKeys=1 -e EnableDeferBacking=0 \
  --entrypoint bash intel/llm-scaler-vllm:0.26.0-b2 -lc '
source /opt/intel/oneapi/setvars.sh --force
python3 -m vllm.entrypoints.openai.api_server --port 8001 \
  --model wiesion/Ornith-1.5-35B-A3B-int4-AutoRound-GPTQ-XPU --served-model-name Ornith-1.5-35B \
  --quantization gptq --enforce-eager --attention-backend FLASH_ATTN \
  --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.93 \
  --max-model-len 65536 --max-num-seqs 4 --max-num-batched-tokens 4096 --enable-chunked-prefill \
  --enable-prefix-caching --mamba-cache-mode align --mamba-ssm-cache-dtype float16 \
  --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --default-chat-template-kwargs "{\"enable_thinking\": false}" --trust-remote-code'
```

Already inside an llm-scaler container, the minimal form is:

```bash
# Environment setup for Intel XPU / llm-scaler
source /opt/intel/oneapi/setvars.sh --force
# Run with vLLM from llm-scaler (add the remaining flags from above as needed)
vllm serve wiesion/Ornith-1.5-35B-A3B-int4-AutoRound-GPTQ-XPU --quantization gptq --enforce-eager \
  --kv-cache-dtype fp8_e4m3 --enable-prefix-caching --mamba-cache-mode align --language-model-only
```

Notes on the flags:
- `NEOReadDebugKeys=1 EnableDeferBacking=0`: works around a compute-runtime host-RAM mirroring regression in this image (without it vLLM can consume tens of GB of host RAM).
- `--enable-prefix-caching --mamba-cache-mode align`: hybrid (Gated DeltaNet) prefix caching on XPU; warm agent turns go from seconds to sub-second (see performance tables). Cache granularity is 2048-token blocks.
- `--language-model-only`: skips the vision tower. On a 24 GB B60 vision costs a lot of KV room (see Vision).
- `--default-chat-template-kwargs {"enable_thinking": false}` only sets the server default; clients can still request thinking per call.

| config | KV dtype | KV pool | fits |
|---|---|---|---|
| 1× B60 24 GB, text-only | fp8 | 265,593 tokens | 4 × 64k |
| 1× B60 24 GB, text-only | fp16 | 144,179 tokens | 2 × 64k or 4 × 32k |
| 1× B60 24 GB, with vision | fp8 | 53,833 tokens | 1 × 48k |
| 1× B70 32 GB, text-only | fp8 | 891,289 tokens | 6.8 × 128k |
| 1× B70 32 GB, text-only | fp16 | 453,861 tokens | 3.5 × 128k or 7 × 64k |

fp8 KV doubles capacity; fp16 KV prefills 14–32 % faster on this image (see [fp8 vs fp16 KV cache](#fp8-vs-fp16-kv-cache)).

## Quality vs official FP8 (same llm-scaler stack)

| corpus | tokens | KL (ref top-20) | top-1 agreement | ppl FP8 | ppl int4 | ppl ratio |
|---|---|---|---|---|---|---|
| all | 108,911 | 0.0892 | 87.5 % | 6.628 | 6.499 | 0.981 |
| prose | 48,297 | 0.1003 | 85.4 % | 11.407 | 11.612 | 1.018 |
| code | 38,987 | 0.0753 | 92.7 % | 2.183 | 2.247 | 1.029 |
| chat | 21,627 | 0.0893 | 82.6 % | 14.591 | 12.064 | 0.827 |

Run-to-run noise floor (same server, two passes): KL 0.0008, top-1 99.6 %.

| task | FP8 (TP=2) | int4 (this repo) | recovery |
|---|---|---|---|
| GSM8K 5-shot (500), flexible | 0.914 | 0.918 | 100.4 % |
| GSM8K 5-shot (500), strict | 0.870 | 0.900 | 103.4 % |
| IFEval prompt-level strict | 0.762 | 0.760 | 99.8 % |
| IFEval instruction-level strict | 0.836 | 0.835 | 99.9 % |
| MMLU-Pro CoT 5-shot (420) | 0.738 | 0.731 | 99.0 % |

### RULER, int4 build (20 samples per task and length, 1x B70; see notes on VT/QA output limits)

| task | 16k | 32k | 60k |
|---|---|---|---|
| niah_single_2 | 1.000 | 1.000 | 1.000 |
| niah_multikey_2 | 0.950 | 0.950 | 1.000 |
| niah_multivalue | 1.000 | 1.000 | 1.000 |
| ruler_vt | 1.000 | 1.000 | 0.800 |
| ruler_qa_squad | 0.787 | 0.838 | 0.742 |

### BFCL v4 (single-turn subset, OpenAI tools API through vLLM's qwen3_coder parser)

| category | FP8 | int4 | recovery |
|---|---|---|---|
| irrelevance | 0.713 | 0.650 | 91.2 % |
| live_multiple | 0.761 | 0.760 | 99.9 % |
| live_parallel | 0.750 | 0.688 | 91.7 % |
| live_simple | 0.756 | 0.779 | 103.1 % |
| multiple | 0.925 | 0.915 | 98.9 % |
| parallel | 0.930 | 0.900 | 96.8 % |
| parallel_multiple | 0.870 | 0.845 | 97.1 % |
| simple_python | 0.920 | 0.917 | 99.7 % |

Notes on quality:
- **Recovery is relative to the official FP8 checkpoint served on the same stack** (2× B70, TP=2). FP8 is near-lossless vs bf16 but not identical, and TP=2 vs single-GPU numerics differ slightly; KL/top-1 therefore include a small reference offset that applies equally to any 4-bit build.
- KL is computed over the reference's top-20 next-token distribution on a fixed 108,911-token corpus (wikitext-2, Python source, UltraChat conversations); missing int4 logprobs are imputed with its 20th logprob.
- GSM8K, IFEval and MMLU-Pro are older public benchmarks that are likely in modern training data; read them as *relative* (int4 vs FP8 of the same model), not as absolute capability. For headline capability see the base model card.
- RULER is reported for the int4 build only, as a long-context recall check, not as a recovery figure. Its variable-tracking and SQuAD-QA tasks allow only 30–32 output tokens; through the chat endpoint the expected answer prefix cannot be pre-filled, so a model that restates it runs out of tokens before answering. These two tasks therefore partly measure answer brevity. The needle tasks (128 output tokens) are the more reliable recall signal.
- RULER uses 20 samples per task and length; single cells move by ±0.05–0.1 between runs, so look at trends, not individual cells.
- BFCL: the clearest loss is `irrelevance` (correctly *not* calling a tool when none fits): −6 points vs FP8; the other categories stay within ~3 points (`live_parallel` has only 16 cases).
- All evals: chat endpoint, thinking off, greedy, 4 concurrent requests.

## Performance (`vllm bench serve`, random prompts, unique per run)

### 1x Arc Pro B60 24 GB - 64k context, 4 sequences

| scenario (in/out) | c | TTFT p50 | TTFT p99 | TPOT p50 | tok/s per stream | output tok/s total |
|---|---|---|---|---|---|---|
| chat (1024/512) | 1 | 0.18 s | 0.22 s | 24.1 ms | 41.5 | 41 |
| chat (1024/512) | 2 | 0.26 s | 0.35 s | 21.6 ms | 46.3 | 90 |
| chat (1024/512) | 4 | 0.58 s | 0.58 s | 21.1 ms | 47.3 | 180 |
| mid (16384/512) | 1 | 2.98 s | 4.37 s | 23.8 ms | 42.0 | 33 |
| mid (16384/512) | 2 | 4.06 s | 6.31 s | 26.7 ms | 37.4 | 57 |
| mid (16384/512) | 4 | 5.45 s | 13.37 s | 39.8 ms | 25.1 | 80 |
| long (32768/512) | 1 | 7.80 s | 7.82 s | 24.1 ms | 41.6 | 25 |
| long (32768/512) | 2 | 10.38 s | 16.67 s | 36.4 ms | 27.5 | 35 |
| long (32768/512) | 4 | 14.16 s | 36.57 s | 77.8 ms | 12.8 | 39 |
| xlong (60000/256) | 1 | 20.16 s | 20.17 s | 24.4 ms | 40.9 | 10 |
| xlong (60000/256) | 2 | 27.56 s | 43.21 s | 95.9 ms | 10.4 | 10 |
| xlong (60000/256) | 4 | 38.94 s | 97.98 s | 325.5 ms | 3.1 | 9 |
| cached (32k cached + 2k / 256) | 1 | 0.88 s | 8.08 s | 24.1 ms | 41.5 | 32 |
| cached (32k cached + 2k / 256) | 2 | 0.98 s | 17.97 s | 24.3 ms | 41.1 | 45 |
| cached (32k cached + 2k / 256) | 4 | 11.61 s | 32.14 s | 71.7 ms | 13.9 | 34 |

### 1x Arc Pro B70 32 GB - 128k context, 8 sequences

| scenario (in/out) | c | TTFT p50 | TTFT p99 | TPOT p50 | tok/s per stream | output tok/s total |
|---|---|---|---|---|---|---|
| chat (1024/512) | 1 | 0.18 s | 0.32 s | 23.4 ms | 42.7 | 42 |
| chat (1024/512) | 2 | 0.20 s | 0.27 s | 20.6 ms | 48.6 | 95 |
| chat (1024/512) | 4 | 0.42 s | 0.43 s | 20.3 ms | 49.3 | 191 |
| chat (1024/512) | 8 | 0.60 s | 0.81 s | 20.6 ms | 48.5 | 368 |
| mid (16384/512) | 1 | 2.12 s | 2.13 s | 23.8 ms | 42.0 | 36 |
| mid (16384/512) | 2 | 2.85 s | 4.42 s | 23.9 ms | 41.8 | 68 |
| mid (16384/512) | 4 | 3.73 s | 9.34 s | 32.4 ms | 30.8 | 101 |
| mid (16384/512) | 8 | 4.64 s | 21.33 s | 60.9 ms | 16.4 | 117 |
| long (32768/512) | 1 | 5.39 s | 5.41 s | 23.7 ms | 42.1 | 29 |
| long (32768/512) | 2 | 7.11 s | 11.43 s | 30.1 ms | 33.3 | 46 |
| long (32768/512) | 4 | 9.83 s | 24.77 s | 56.6 ms | 17.7 | 54 |
| long (32768/512) | 8 | 12.34 s | 58.35 s | 142.2 ms | 7.0 | 50 |
| xlong (60000/256) | 1 | 13.53 s | 13.55 s | 23.2 ms | 43.1 | 13 |
| xlong (60000/256) | 2 | 18.15 s | 28.81 s | 70.3 ms | 14.2 | 14 |
| xlong (60000/256) | 4 | 25.81 s | 63.58 s | 214.4 ms | 4.7 | 13 |
| xlong (60000/256) | 8 | 33.77 s | 155.00 s | 706.2 ms | 1.4 | 10 |
| cached (32k cached + 2k / 256) | 1 | 0.58 s | 5.54 s | 23.0 ms | 43.4 | 36 |
| cached (32k cached + 2k / 256) | 2 | 0.67 s | 12.27 s | 22.9 ms | 43.7 | 55 |
| cached (32k cached + 2k / 256) | 4 | 1.83 s | 20.26 s | 26.7 ms | 37.5 | 73 |
| cached (32k cached + 2k / 256) | 8 | 2.22 s | 39.13 s | 46.5 ms | 21.5 | 75 |

### fp8 vs fp16 KV cache

Same build, single stream unless noted.

fp16 KV prefills 14–32 % faster (the fp8 KV path costs prefill time on this image) and halves the KV pool; decode is ~7 % slower on the B60 but ~5 % faster on the B70.

| GPU | scenario | TTFT fp8 KV | TTFT fp16 KV | Δ TTFT | decode fp8 | decode fp16 |
|---|---|---|---|---|---|---|
| B60 | 16k/512 | 2.98 s | 2.52 s | -15 % | 42.0 tok/s | 38.8 tok/s |
| B60 | 32k/512 | 7.80 s | 5.94 s | -24 % | 41.6 tok/s | 38.5 tok/s |
| B60 | 60k/256 | 20.16 s | 13.81 s | -32 % | 40.9 tok/s | 38.1 tok/s |
| B60 | 32k cached + 2k | 0.88 s | 0.64 s | -28 % | 41.5 tok/s | 37.5 tok/s |
| B60 | 2 × 32k/512 | 10.38 s | 7.65 s | -26 % | 27.5 tok/s | 30.5 tok/s |
| B70 | 16k/512 | 2.12 s | 1.83 s | -14 % | 42.0 tok/s | 44.9 tok/s |
| B70 | 32k/512 | 5.39 s | 4.33 s | -20 % | 42.1 tok/s | 45.0 tok/s |
| B70 | 60k/256 | 13.53 s | 9.94 s | -27 % | 43.1 tok/s | 45.1 tok/s |
| B70 | 32k cached + 2k | 0.58 s | 0.43 s | -25 % | 43.4 tok/s | 45.0 tok/s |
| B70 | 2 × 32k/512 | 7.11 s | 5.51 s | -23 % | 33.3 tok/s | 38.0 tok/s |

Performance notes:
- Random prompts are unique per run (no accidental prefix-cache hits); the `cached` scenario deliberately reuses a 32k prefix per session, the first request per prefix is cold (p99).
- Long-context bursts are prefill-bound: several simultaneous 32–60k prompts share the GPU's prefill throughput and stall each other's decode. Typical agent use (warm prefix + a few k new tokens) stays fast.


## Correctness on llm-scaler

Harness run against this build on 1× B70 (64k × 4, otherwise the quick-start flags):

| check | what it verifies | result |
|---|---|---|
| cold vs warm prefix cache | identical request answered from scratch and from cache | token-identical at 8k / 32k / 62k |
| 12-turn agent session (24k → 54k) | cache hits on every follow-up turn; final warm turn vs cold (cache-busted) reference | turns 2–12 served 22.5k–49k tokens from cache (83–92 % of the prompt), final turn identical to reference |
| needle recall | 3 facts at 10 / 50 / 90 % depth | 9/9 at 16k / 32k / 65k |
| tool calls (20 prompts) | valid JSON arguments; right tool or correctly no tool | 100 % valid, decisions 100 % cold / 100 % warm |
| 4 concurrent sessions | no cross-contamination, no garbage | pass |
| garbage detector | `!!!!` runs, repeated tokens, phrase loops, NaN logprobs | none in any harness run |

Other configurations tried on `0.26.0-b2`:

| configuration | result |
|---|---|
| this build (AutoRound, GPTQ export) | all checks pass |
| on-the-fly `--quantization sym_int4` from the bf16 base | garbage: "capital of France" → ".", prompt-echo loops, needle recall 0/9 |
| MTP speculative decoding (K=1 or K=3) **with** prefix caching | `!!!!` corruption in 2 of 6 long generations |
| MTP K=3 **without** prefix caching | clean, up to 1.9× decode at 1–4 short streams, but slower at 8 streams and at long context, −35 % KV pool, every agent turn re-prefills |

## Vision

The vision tower is unquantized bf16 and works (checked with text reading, structured invoice extraction, shapes/colors and chart values — all correct). Serving with vision on a 24 GB B60 leaves ~54k tokens of KV (the image encoder reserves ~2 GiB for profiling), so on a B60 prefer `--language-model-only` unless you need images; on a B70 / TP=2 there is plenty of room.

## Known issues (observed on `0.26.0-b2`, may change with newer images)

- **Do not enable MTP speculative decoding together with prefix caching on llm-scaler 0.26.0-b2**: output degenerates into `!!!!` after a few hundred tokens. Without prefix caching MTP is clean and faster at 1–4 short streams, but it costs ~35–39 % KV pool, loses at 8 streams and at long context, and every agent turn re-prefills.
- Do not use `--quantization sym_int4` on the bf16 base model (garbage MoE output on this image).
- Thinking mode was not evaluated; all numbers are with `enable_thinking: false`.

## Reproduction

```bash
# AutoRound 0.16.0 inside intel/llm-scaler-vllm:0.26.0-b2, one Arc Pro B70, ~11 h (~16 min per decoder block)
python3 -m auto_round --model ornith-ai/Ornith-1.5-35B-A3B --device_map xpu \
  --scheme W4A16 --bits 4 --group_size 128 \
  --iters 200 --nsamples 128 --seqlen 2048 --dataset NeelNanda/pile-10k \
  --format auto_gptq --output_dir ./out
```

AutoRound keeps the MoE routers (not `nn.Linear` in transformers 5.8) and `shared_expert_gate` in bf16 automatically. A calibration ablation with a code/chat-weighted mix (40 pile-10k + 56 OpenCodeInstruct + 32 UltraChat, otherwise identical) was tested and was not better (KL vs FP8 0.0926 vs 0.0892 for this build); this repo ships the pile-10k build.

## License

MIT, inherited from the base model [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B). All credit for the model goes to the Ornith authors; this repo only changes the weight format.
