---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- exl3
- exllamav3
- vllm
- deepseek
- deepseek-v4.1-flash
- deepseek-v4.1-flash-1017
- mixture-of-experts
- 3-bit
- rtx-pro-6000
- blackwell
- sm120
---

# DeepSeek-V4.1-Flash-1017 EXL3 3-bit for 2× RTX PRO 6000 Blackwell

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on two RTX PRO 6000 Blackwell
cards (sm_120, 96 GB each), with its full 1M-token context. The routed experts are our own calibrated EXL3 3-bit
quantization (coolbho3k's method); the other tensors keep DeepSeek's FP8. The pack serves with vLLM at tensor-parallel 2:
a third of the routed experts stay in pinned host RAM, run either by custom sm_120 kernels on the GPUs (the default) or
by an optional CPU runtime on a large AVX-512 host, which is the fastest way to serve it.

- **277-289 output tokens/s** on long multi-turn agentic coding, one session, with the CPU runtime.
- **214.8-223.2 tokens/s** decoding against a 46K-token code prompt, and **14,333-15,244 tokens/s** fresh prefill at 40K tokens.
- **1M tokens of context**: a 1.04M-token prompt prefills in 93 s, with a 1.07M-token fp8 KV cache.
- **Close to the original**: mean KL 0.011 nats on agentic coding text, GSM8K-200 98.5%, HumanEval 97.0%.

## Speed

Output tokens per second on 2× RTX PRO 6000, built and served from `recipe/` as in the quick start. Ranges span fresh
server boots.

| Setup | Agentic coding, 1 session | Code, 1 stream (46K prompt) | Fresh prefill (40K tokens) | Aggregate under load | Context |
|---|---:|---:|---:|---|---|
| **CPU runtime** (`CPU_EXPERTS=1`) | **277-289** | **214.8-223.2** | **14,333-15,244** | 4 agentic sessions 154-159; four 46K streams 470.0-505.2 | 1M window; 1.07M-token fp8 KV cache in 2.81 GiB |
| GPU-only (default) | 224.1¹ | 172.8 | 11,157-11,467 | 4 agentic sessions 110.4¹; four 46K streams 362.9 | 1M window; 1.07M-token fp8 KV cache |

"Agentic coding" replays a held-out multi-turn agentic coding benchmark (104 turns with tool calls, turn by turn, at 1
and 4 concurrent sessions), with the tool definitions sent in every request. ¹ The GPU-only agentic figures were
measured with an earlier version of the replay client that did not send the tool definitions.

**At 1M tokens** (CPU runtime): a 1,040,034-token prompt prefills in 93 s (11,179 tokens/s), the needle is found, a tool
call afterwards works, and each card keeps about 2.5 GB free at the peak.

Test setup and every benchmark: [docs/SPEED.md](docs/SPEED.md).

## Quality

| | KL, neutral text | KL, code | KL, agentic | KL, short100 | GSM8K-200 | HumanEval |
|---|---:|---:|---:|---:|---:|---:|
| Original 1017, second run (noise floor) | 0.0082 | 0.0050 | 0.0047 | — | — | — |
| **This pack, CPU runtime** | **0.0520** | **0.0309** | **0.0110** | **0.0349** | **98.5%** | **97.0-97.6%** |
| This pack, GPU-only | 0.0522 | 0.0306 | 0.0111 | 0.0338 | 98.5% | 97.0% |

Mean KL in nats (a lower bound) to the original 1017 on AtomicChat's three corpora (top-512 logprobs, with our own run of
the original checkpoint as the reference) and on kyuz0's full-vocabulary short100 panel; GSM8K-200 and HumanEval with
thinking off and greedy decoding. The KL readings come from the prefill path, whose results this release leaves
bit-identical. One decode-side change is not bit-exact (the CPU runtime skips expert routing for draft rows it is about to
reject); it passed teacher-forced agreement and agentic behaviour checks instead. Details, top-1 agreement and the
per-change gates: [docs/FIDELITY.md](docs/FIDELITY.md).

## Quick start

You need two sm_120 GPUs with 96 GB each and a CUDA 13 driver, Docker with the NVIDIA container toolkit, about 260 GiB
of free host RAM and 340 GB of disk.

```bash
hf download diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 --local-dir ~/models/DeepSeek-V4.1-Flash-1017-EXL3-3bpw
cd ~/models/DeepSeek-V4.1-Flash-1017-EXL3-3bpw/recipe
bash docker/build.sh                      # serving image dsv41-flash-1017-exl3-3bpw-sm120
bash kernels/build.sh                     # xmoe, xmoe_bulk, xmoe_do, xmoe_do_i8, fp8ar, rowtopk, sparse_mla_epilogue, mxfp8_pingpong; ~6 min
API_KEY=change-me bash serve/serve.sh     # OpenAI-compatible API on :8000, model deepseek-v4.1-flash
```

On an x86_64 host with AVX-512 (VNNI, VBMI, VBMI2) and many cores and memory channels (see `recipe/cpuexp/README.md`),
build and check the CPU runtime on the serving machine, then turn it on:

```bash
bash cpuexp/build.sh && bash cpuexp/check_host.sh
API_KEY=change-me CPU_EXPERTS=1 bash serve/serve.sh
```

Optional, for one stream at a time on mostly code: the code-tuned DSpark head at k=7 ([docs/DRAFTER.md](docs/DRAFTER.md)).

```bash
API_KEY=change-me DS41_DRAFTER=code SPEC_K=7 bash serve/serve.sh
```

Check the running server with:

```bash
export API_KEY=change-me
python3 bench/needle-test.py 200 && python3 bench/test-toolcall.py && python3 bench/test-image.py
```

`recipe/serve/serve.sh` documents every setting and `recipe/README.md` explains the stack. The same download also serves
an 8x RTX PRO 6000 node as two 4-GPU replicas with every expert in VRAM: [docs/PROFILE-8x.md](docs/PROFILE-8x.md).

## What's in this repo

| Files | Contents |
|---|---|
| `model-00001` to `model-00048`, `config.json` | The pack (332 GB): our calibrated EXL3 3-bit routed experts, DeepSeek's FP8 for the other tensors, int4 Engram tables, and the DSpark drafter with EXL3 3-bit experts. Layout: [docs/HOW-IT-FITS.md](docs/HOW-IT-FITS.md#pack-layout). |
| `drafter-code-k7/` | Optional (5.8 GB): a DSpark head tuned for code and agentic tool use (`DS41_DRAFTER=code`; [docs/DRAFTER.md](docs/DRAFTER.md)). `--exclude 'drafter-code-k7/*'` skips it. |
| `drafter-code-fp8/` | Optional (7.9 GB): the same code-tuned head in DeepSeek's original FP8/FP4 format, as drop-in replacements for `model-00044` to `model-00046` of `deepseek-ai/DeepSeek-V4.1-Flash-1017`, for stacks serving the original weights (its `README.md`). `--exclude 'drafter-code-fp8/*'` skips it. |
| `recipe/` | Image build, patches, kernels, CPU runtime, host-expert plans, the 8x profile, and serving, benchmark, KL and quantization scripts. |
| `docs/` | The detail pages listed below. |

The tokenizer, `encoding/`, `inference/` and `LICENSE` come from DeepSeek's release.

## Limitations

- Two prefill changes are lossy (8-bit all-reduce, int8 decode-once kernel), each within KL noise; `DS41_FP8AR=0` / `DO_INT8=0` turn them off.
- The quality evidence is the KL table, GSM8K-200, HumanEval and needle tests, not a full benchmark suite; image input is only probe-checked.
- The host-expert plan assumes two 96 GB cards with ~2.5 GB per card used by another process; at 1M the peak leaves about 1 GB per card on the GPU-only path (about 2.5 GB with the CPU runtime's compact KV cache).
- The default plan is tuned for long agentic coding; workloads that route differently use host experts more (`recipe/plans/README.md`).
- The CPU runtime needs a large AVX-512 CPU, must be built on the serving host, and can be slower than GPU-only on smaller CPUs.
  Its draft-row routing skip is not bit-exact; `DS41_N216=` turns it off. Its pool layout is tuned for the tested 96-core
  CPU; other CPUs get pools derived from their topology.
- Prompt encoding follows the official DeepSeek API; a few API behaviors are not ported, and the stack is pinned to one vLLM nightly.
- For evaluations and other single-answer prompts, use DeepSeek's recommended sampling (temperature 1.0, top_p 1.0) and leave a large output budget. With thinking on at temperature 0, DeepSeek-V4.1-Flash (the original weights as well as this pack) can deliberate past 32K tokens on ambiguous or ill-posed questions. If you need short, bounded answers, turn thinking off.

## More detail

- [docs/SPEED.md](docs/SPEED.md): test setup and every speed table, including the 8x profile.
- [docs/FIDELITY.md](docs/FIDELITY.md): KL and top-1 agreement in full, the per-change KL gate, quality probes, lossy
  changes and API compatibility.
- [docs/PROFILE-8x.md](docs/PROFILE-8x.md): the 8x RTX PRO 6000 profile, its settings and limits.
- [docs/DRAFTER.md](docs/DRAFTER.md): the optional code-tuned DSpark head.
- [docs/HOW-IT-FITS.md](docs/HOW-IT-FITS.md): how the model fits in 2× 96 GB, the pack layout, how the experts were made.
- [docs/CHANGELOG.md](docs/CHANGELOG.md): revision history and upgrade notes.
- `recipe/README.md` (the stack) and `recipe/results/RESULTS.md` (every measurement).

`recipe/MODEL_CARD.md` is a copy of this card; its links are relative to the repo root.

## Credits and license

Built on the work of:

- DeepSeek;
- coolbho3k (the calibration method);
- sfxnz (the V4.1 EXL3 recipe);
- turboderp (ExLlamaV3);
- vcruz305 and Mia's AI Lab (vllm-exl3);
- AtomicChat (the KL protocol and calibration corpus) and kyuz0 (the short100 reference panel);
- the vLLM, FlashInfer and CUTLASS teams.

The weights keep DeepSeek's MIT license (`LICENSE`). The code in `recipe/` is MIT, except `recipe/patches/vllm-exl3/`, which
is AGPL-3.0-only like the plugin it patches. `recipe/THIRD_PARTY_NOTICES.md` lists every upstream component.
