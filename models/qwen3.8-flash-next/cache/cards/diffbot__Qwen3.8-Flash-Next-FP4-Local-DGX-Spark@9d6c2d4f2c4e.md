---
license: other
license_name: nvidia-open-model-agreement
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
tags:
- qwen3.8
- qwen3.8-flash-next
- nvfp4
- modelopt
- mixture-of-experts
- vllm
- b12x
- speculative-decoding
- mtp
- dgx-spark
- gb10
- blackwell
---

# Qwen3.8-Flash-Next FP4-Local for one DGX Spark

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (125B total / 6B active parameters, vision, 51B
n-gram embedding tables) served on **a single DGX Spark**. The weights are NVIDIA's
[Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) quantization, with an MTP
draft head fine-tuned for agentic workloads. The serving recipe is a pinned public vLLM + b12x image plus patched and
new vLLM / b12x files and custom sm_121a CUDA kernels, all in `recipe/`.

Against stock vLLM on the same Spark:

- **1.34× faster decode** on agentic coding (65.3 vs 48.8 tok/s);
- **1.45× faster prefill** at 33K tokens (3,236 vs 2,226 tok/s);
- **4.1× faster agent follow-up turns** (0.90 vs 3.70 s to first token);
- **1.68× shorter wall clock** over a 92-turn agentic coding benchmark;
- the same quality on GSM8K and HumanEval.

## Results

| One request at a time, same DGX Spark | Stock vLLM | This repo | Speedup |
|---|---:|---:|---:|
| Decode, agentic coding | 48.8 tok/s | **65.3 tok/s** | **1.34×** |
| Decode, fixed prompts (`recipe/bench/c1bench.py`) | 50.0 tok/s | 70.0 tok/s | 1.40× |
| Prefill, 8.6K-token prompt | 2,141 tok/s | 2,931 tok/s | 1.37× |
| Prefill, 33K-token prompt | 2,226 tok/s | **3,236 tok/s** | **1.45×** |
| Prefill, 112K-token prompt | 2,194 tok/s | 3,168 tok/s | 1.44× |
| Time to first token, 112K-token prompt | 51.3 s | 35.5 s | 1.44× |
| Time to first token, agent follow-up turn (median) | 3.70 s | **0.90 s** | **4.1×** |
| Agentic coding benchmark, wall clock (92 turns) | 1,161 s | 693 s | **1.68×** |
| GSM8K (first 200) ¹ | 97.0 % | 97.5 % | |
| HumanEval ¹ | 97.6 % | 97.0 % | |
| Needle at 29K / 96K / 125K tokens, tool calls | pass | pass | |
| Context window (KV cache) | 262,144 tokens (464K) | 131,072 tokens (145–160K) | |

¹ Greedy, thinking off.

**Stock vLLM** is NVIDIA's documented DGX Spark command for this checkpoint: the `eugr/spark-vllm-b12x` image at the build
this recipe pins, with NVIDIA's environment and `vllm serve` flags. Upstream vLLM cannot load the checkpoint's compressed
FP4 scales yet. Stock keeps a larger KV cache at the same memory fraction; this recipe spends that memory on larger
prefill chunks and a bf16 draft-layer cache, and serves a 131K context. Details are under
[How we measured](#how-we-measured).

## What's different from stock vLLM

- **An MTP draft head fine-tuned for agentic workloads.** Only the 29 dense tensors of the MTP layer (91M parameters)
  were trained; its experts and every other weight are NVIDIA's. Speculative decoding checks every drafted token against
  the full model, so the head changes speed, not outputs: 3.48 instead of 3.26 tokens per step on agentic coding,
  +6.8% decode.
- **Decode kernels.** The drafter scores a 64K-token vocabulary subset through the target's MXFP8 LM head; FP8 and
  split-K GEMVs for the small projections; fused HyperConnection, shared-expert and RMSNorm kernels; the linear-attention
  state is committed once per step after acceptance; the decode MoE kernel reads the NVFP4 scales in their compressed form.
- **Prefill kernels.** 8192-token chunks; a persistent prefill MoE kernel for sm_121a (weight ring, scales decoded inline,
  fixed-order expert sum); the HyperConnection combine, norm and projections in CUDA; strided linear-attention
  convolution, pipelined sparse-attention gathers and tuned MXFP8 GEMM tiles; the n-gram rows of the next chunk read
  ahead from NVMe.
- **Fast agent follow-ups.** The prefix-cache checkpoint of each prompt is placed just before its end, so a follow-up turn
  recomputes only the new message.
- **bf16 activations in every MoE call.** Stock uses FP4 activations for prefill-size MoE calls; this recipe keeps
  bf16 activations there too, which stays closer to the BF16 model.

Every change has an off switch in `recipe/serve/serve.sh`; `recipe/README.md` describes each one with its measured effect.

## Quick start

```bash
hf download diffbot/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark --local-dir ~/models/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark
cd ~/models/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark/recipe
API_KEY=change-me bash serve/serve.sh     # OpenAI-compatible API on :8000, model qwen3.8-flash-next
```

The first start compiles and autotunes the kernels into `~/.cache/qwen38-flash-next-spark`, which takes about 25
minutes; later starts take about 5 minutes. Text, image and video input are on; tool calls use vLLM's
`qwen3_xml` parser and reasoning the `qwen3` parser.

Check the running server:

```bash
cd ~/models/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark/recipe/bench && bash fetch-evaldata.sh
export API_KEY=change-me SERVED_MODEL=qwen3.8-flash-next EVALDATA=$PWD/evaldata
python3 test-toolcall.py && python3 needle-test.py 1800 && python3 quality-probe.py
```

## Hardware requirements

- One DGX Spark, or another GB10 system with 128 GB of unified memory. Tested with NVIDIA driver 580.126.09 (CUDA 13.0).
- Docker with the NVIDIA container toolkit.
- About 110 GB of free disk on fast NVMe: the 102 GB download plus about 5 GB of kernel caches. The n-gram tables are
  read from disk at every step.
- vLLM gets 73% of the unified memory and needs 88.8 GiB free to start; the server peaks at about 93 GiB during a
  112K-token prefill.

To leave room for another memory-heavy service, this setting frees about 7 GiB, keeps the 131K context and turns off
image and video input; prefill is about 8% slower and decode is unchanged:

```bash
MAX_NUM_BATCHED_TOKENS=4096 GPU_MEMORY_UTILIZATION=0.68 \
EXTRA_ARGS="--limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0" bash serve/serve.sh
```

## Configuration

`recipe/serve/serve.sh` reads these environment variables:

| Variable | Default | Effect |
|---|---|---|
| `API_KEY` | none | bearer token for the API; unset serves without one |
| `PORT`, `NAME` | `8000`, `qwen38-fp4` | port and container name |
| `MTP_HEAD` | `tuned` | `stock` serves NVIDIA's original MTP head (same outputs, about 6% slower on agentic coding); see the note in `serve.sh` |
| `MTP_TOKENS` | `4` | draft tokens per step; `0` turns speculative decoding off (about 30 tok/s) |
| `DRAFT_VOCAB` | `en` | `cjk` adds every CJK, kana and Hangul token to the draft vocabulary for Chinese, Japanese and Korean answers |
| `MAX_MODEL_LEN` | `131072` | context window |
| `GPU_MEMORY_UTILIZATION` | `0.73` | memory fraction for vLLM |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | prefill chunk size |
| `MAX_NUM_SEQS` | `4` | concurrent requests |
| `COMPILE_JOBS` | `2` | parallel kernel compile jobs on the first start; caps the memory the first start needs |
| `COMPILE_CACHE_ENTRIES` | `256` | compiled kernel programs kept in memory during the first start; caps its memory use |
| `EXTRA_ARGS`, `EXTRA_ENV` | | extra `vllm serve` flags, extra `docker run -e` flags |

## Fidelity

KL divergence to the original BF16 model, full vocabulary, in llama.cpp `--kl-divergence` semantics, on 24 chunks of
4,096 tokens of AtomicChat's held-out neutral set
([reference](https://huggingface.co/datasets/AtomicChat/Qwen3.8-Flash-Next-Uncensored-GGUF-metrics)): mean KLD
**0.0465**, top-1 agreement 92.1 %. The values come from the server's own forward pass
(`recipe/patches/qwen4_exp/nvidia/logit_dump.py`) scored with `recipe/kl/atomic_kld.py`.

## Limitations

- Tested on one DGX Spark with the pinned image digest. The patches replace files of that build; newer
  `eugr/spark-vllm-b12x` tags are not drop-in replacements.
- The custom kernels are built for sm_121a (GB10) only.
- The context window is 131K tokens; stock vLLM fits 262K in the same memory.
- Tuned for one agentic workload at a time: up to 4 concurrent requests.
- The fine-tuned MTP head helps most on agentic coding. The default draft vocabulary favors English and code; use
  `DRAFT_VOCAB=cjk` for Chinese, Japanese or Korean answers.
- b12x's MoE sums expert outputs with atomics, so greedy outputs can differ between runs, and GSM8K and HumanEval move by
  about 1.5 points between boots.
- The quality evidence is GSM8K, HumanEval, KL divergence, needle and tool-call tests, not a full benchmark suite.

## How we measured

- One DGX Spark (GB10, 128 GB, driver 580.126.09), a ~19 GB side service resident, no other clients, each server on a
  fresh boot. Stock vLLM and this repo ran back to back for prefill and fixed-prompt decode; the other rows come from
  boots of the same two configurations on the same Spark.
- **Stock vLLM:** NVIDIA's DGX Spark command from `NVIDIA_MODEL_CARD.md` (262K context, 8 sequences, 4096-token chunks,
  MTP with 4 draft tokens, its environment block) on `eugr/spark-vllm-b12x@sha256:03a95654…`, with three changes:
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (without it, boots run out of unified memory), a 0.73 memory
  fraction instead of 0.80 (the same budget as this recipe; speed at one request does not depend on it), and local model
  files. `recipe/bench/stock-vllm.sh` starts it exactly as measured.
- **This repo:** `recipe/serve/serve.sh` with its defaults.
- **Agentic coding:** 92 multi-turn coding-agent turns with tool definitions and tool calls in the history and prompts up
  to 131K tokens, replayed in order at temperature 0 with up to 1,024 output tokens per turn. Decode speed is completion
  tokens over decode time, weighted by tokens. The turn set is not part of this repo.
- **Prefill and fixed-prompt decode:** `recipe/bench/c1bench.py --prefill 8000,32000,120000 --decode 1 --max-tokens 512
  --reps 2`. Prompts carry a fresh nonce, so nothing comes from the prefix cache; decode uses four fixed prompts. The
  script sends no API key.
- **Quality:** `recipe/bench/quality-probe.py`: GSM8K first 200 and HumanEval 164, greedy, thinking off.
- Raw results: `recipe/results/vs-stock-vllm.txt` and the files it lists.

## What's in this repo

| Files | Contents |
|---|---|
| `tensors/`, `config.json`, tokenizer, processor and generation files, `chat_template.jinja` | NVIDIA's `nvidia/Qwen3.8-Flash-Next-FP4-Local`; `tensors/model-00015-of-00021.safetensors` carries the fine-tuned MTP head (only its 29 dense MTP tensors differ from NVIDIA's file), every other file is unmodified |
| `NVIDIA_MODEL_CARD.md` | NVIDIA's model card for the checkpoint |
| `recipe/` | launcher, patches, CUDA kernels (source, build lines and built libraries), draft vocabularies, benchmarks, KL client, raw results |
| `LICENSE`, `LICENSE.Qwen`, `NOTICE` | NVIDIA Open Model Agreement, Qwen Community License 1.0, attribution and the list of modified tensors |

## Credits and license

Built on the work of the Qwen team (Qwen3.8-Flash-Next), NVIDIA (the FP4-Local checkpoint and Model Optimizer),
local-inference-lab and the FlashInfer team (b12x), eugr (the DGX Spark vLLM + b12x image), the vLLM team, and AtomicChat
(the KL reference). This repo is not affiliated with or endorsed by NVIDIA or Qwen.

- **Weights.** Licensed by NVIDIA Corporation under the NVIDIA Open Model Agreement (`LICENSE`). The fine-tuned MTP head
  is a derivative of NVIDIA's checkpoint under the same agreement; `NOTICE` lists the tensors that differ.
- **Base model.** Qwen3.8-Flash-Next is under the Qwen Community License 1.0 (`LICENSE.Qwen`). It requires a separate
  license from Qwen before commercial "Model as a Service" or "AI Work Assistant" use, and a visible model name in
  products above 100M monthly active users or US$20M monthly revenue.
- **Code.** The recipe's own code is MIT; the modified and new vLLM / b12x files in `recipe/patches/` are Apache-2.0
  (`recipe/THIRD_PARTY_NOTICES.md`).
