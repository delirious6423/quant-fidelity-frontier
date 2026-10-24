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

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) with our own calibrated EXL3
3-bit routed experts (coolbho3k's method), ready to serve on two RTX PRO 6000 Blackwell cards (sm_120, 96 GB each) with
vLLM at tensor-parallel 2. A third of the routed experts stay in pinned host RAM, served by custom sm_120 kernels or an
optional CPU runtime. The same pack also runs on an 8x RTX PRO 6000 node as two 4-GPU replicas with every expert in VRAM
(`PROFILE=8x`).

## At a glance

Output tokens per second, built and served from `recipe/` as in the quick start below. Conditions and every other
benchmark: [docs/SPEED.md](docs/SPEED.md).

| Setup | Agentic replay, 1 stream | Code, 1 stream | Fresh 40K-token prefill | Context / KV cache | Under load (aggregate) |
|---|---:|---:|---:|---|---|
| **2x, default** (GPU-only) | **224.1** | 172.8 (46K prompt) | 11,157-11,467 | 1M window, 1.07M tokens fp8 | agentic replay, 4 concurrent 110.4; four 46K streams 362.9 |
| **2x, CPU runtime** (`CPU_EXPERTS=1`) | **261.8** | 210.1 (46K prompt); 308 code generation | 11,624-12,182 | same | 127.8; 406.2 |
| **8x, per replica** (TP4 + EP4, 4 GPUs) | 205-222 (one 46K agentic prompt) | **435-457** code generation | 19,131-19,302 | 1M window, 4.54M tokens, 128 sequences | 128 code streams 4,404-4,469 |
| **8x, whole node** (2 replicas) | — | 809 (one stream per replica) | — | 2 × 4.54M tokens | 128 code streams per replica 8,430; long generations 5,250 |

"Agentic replay" is the held-out multi-turn agentic coding benchmark of [docs/SPEED.md](docs/SPEED.md) (104 turns with tool calls,
replayed turn by turn); on it the default configuration decodes 28% faster than the previous V4.1-Flash release (224 vs
176 tok/s).

## Quality

| | Neutral text | Code | Agentic | short100 | GSM8K-200 | HumanEval |
|---|---:|---:|---:|---:|---:|---:|
| Original 1017, second run (noise floor) | 0.0082 | 0.0050 | 0.0047 | — | — | — |
| **This pack, 2x default** | **0.0522** | **0.0306** | **0.0111** | **0.0338** | **98.5%** | **97.0%** |
| This pack, 2x CPU runtime | 0.0522 | 0.0306 | 0.0109 | 0.0348 | — | — |
| This pack, 8x profile | — | — | — | — | 98.0-99.0% | 97.0-97.6% |

Mean KL (nats, lower bound) to the original 1017 on AtomicChat's three corpora (top-512 logprobs, with our own run of the
original checkpoint as the reference) and on kyuz0's full-vocabulary short100 panel; GSM8K-200 and HumanEval with
thinking off and greedy decoding. Top-1 agreement, per-change KL gates and earlier releases:
[docs/FIDELITY.md](docs/FIDELITY.md).

## Quick start

You need two sm_120 GPUs with 96 GB each and a CUDA 13 driver, Docker with the NVIDIA container toolkit, about 260 GiB
of free host RAM and 340 GB of disk.

```bash
hf download diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 --local-dir ~/models/DeepSeek-V4.1-Flash-1017-EXL3-3bpw
cd ~/models/DeepSeek-V4.1-Flash-1017-EXL3-3bpw/recipe
bash docker/build.sh                      # serving image dsv41-flash-1017-exl3-3bpw-sm120
bash kernels/build.sh                     # the xmoe, xmoe_do, xmoe_do_i8, fp8ar and rowtopk kernels, about 4 minutes
API_KEY=change-me bash serve/serve.sh     # OpenAI-compatible API on :8000, model deepseek-v4.1-flash
```

With an x86_64 host CPU with AVX-512 (VNNI, VBMI, VBMI2) and many cores and memory channels (see
`recipe/cpuexp/README.md`), build and check the CPU runtime on the serving machine and turn it on:

```bash
bash cpuexp/build.sh && bash cpuexp/check_host.sh
API_KEY=change-me CPU_EXPERTS=1 bash serve/serve.sh
```

On an 8x RTX PRO 6000 node (8 sm_120 GPUs with 96 GB, two NUMA nodes; up to 400 GiB of host RAM per replica), with the
whole repo downloaded as above, including `drafter-code-k7/`:

```bash
bash docker/build.sh                      # the same image, now with the chunked-indexer patch
bash kernels/build.sh 8x                  # xmoe_p8k, xmoe_do_i8, fp8ar, fp8arg and rowtopk
API_KEY=change-me bash serve/serve-8x.sh  # replicas on :8000 (GPUs 0-3) and :8001 (GPUs 4-7); route between them
```

Optional on two cards, for one stream at a time on mostly code: the code-tuned DSpark head at k=7
([docs/DRAFTER.md](docs/DRAFTER.md)).

```bash
API_KEY=change-me DS41_DRAFTER=code SPEC_K=7 bash serve/serve.sh
```

Check the running server with:

```bash
export API_KEY=change-me
python3 bench/needle-test.py 200 && python3 bench/test-toolcall.py && python3 bench/test-image.py
```

`recipe/serve/serve.sh` documents every setting, `recipe/serve/profile-8x.env` the 8x profile's, and `recipe/README.md`
explains the stack. Upgrading from an earlier revision: [docs/CHANGELOG.md](docs/CHANGELOG.md#upgrading-from-an-earlier-revision).

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
- The host-expert plan assumes two 96 GB cards with ~2.5 GB per card used by another process; at 1M the peak leaves about 1 GB per card.
- The default plan is tuned for long agentic coding; workloads that route differently use host experts more (`recipe/plans/README.md`).
- The CPU runtime needs a large AVX-512 CPU, must be built on the serving host, and can be slower than GPU-only on smaller CPUs.
- The 8x profile was tested on one cloud VM, and its k=7 is tuned for code (`SPEC_K=3` is 8-10% faster on prose-heavy bulk generation).
- Prompt encoding follows the official DeepSeek API; a few API behaviors are not ported, and the stack is pinned to one vLLM nightly.

## More detail

- [docs/SPEED.md](docs/SPEED.md): test setup and all speed tables (2x default, CPU runtime and development builds, 8x
  single replica and both replicas under load, power).
- [docs/FIDELITY.md](docs/FIDELITY.md): KL and top-1 agreement in full, the per-change KL gate, quality probes, lossy
  changes and API compatibility.
- [docs/PROFILE-8x.md](docs/PROFILE-8x.md): the 8x RTX PRO 6000 profile, its settings and limits.
- [docs/DRAFTER.md](docs/DRAFTER.md): the optional code-tuned DSpark head.
- [docs/HOW-IT-FITS.md](docs/HOW-IT-FITS.md): how the model fits in 2× 96 GB, the pack layout, how the experts were made.
- [docs/CHANGELOG.md](docs/CHANGELOG.md): the 2026-10-24 and 2026-10-21 updates, revision history, upgrade notes.
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
- the vLLM and FlashInfer teams.

The weights keep DeepSeek's MIT license (`LICENSE`). The code in `recipe/` is MIT, except `recipe/patches/vllm-exl3/`, which
is AGPL-3.0-only like the plugin it patches. `recipe/THIRD_PARTY_NOTICES.md` lists every upstream component.
