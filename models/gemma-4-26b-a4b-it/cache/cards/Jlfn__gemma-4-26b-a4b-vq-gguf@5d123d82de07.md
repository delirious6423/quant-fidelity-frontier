---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
- multilingual
tags:
- gemma
- gemma-4
- google
- gguf
- vector-quantization
- vq
- mixture-of-experts
- moe
- 262k-context
- ds4
- ds4-gemma4
- vq-tools
---

# Gemma 4 26B A4B VQ GGUF

> [!WARNING]
> **This is the first published build of the R13+8 recipe.** The conversion
> line is still being researched, so any file here may be replaced by a new
> revision. To keep a fixed build, download a pinned commit:
> `hf download Jlfn/gemma-4-26b-a4b-vq-gguf --revision <commit>`.

**Google DeepMind's Gemma 4 26B A4B, with its 262,144-token context, sized for
one 12 GiB consumer card.** The BF16 checkpoint becomes **9.95 GiB** of GGUF
parts: an 8.00 GiB trunk that stays resident, a 1.38 GiB BF16 token table that
is read in rows from SSD, and a 0.58 GiB optional drafter. The routed experts
use 8-dimensional vector quantization: a 13-bit main index (K=8192, a 12-bit
stream plus a 1-bit plane) and an 8-bit GVR8 residual on every expert stack,
with E4M3 codebooks. Everything the model reads on every token besides the
experts stays at Q4_K or Q8_0, and the router, norms and layer scalars stay
BF16. This is a text-only release: the vision tower is not included.

> **Status, 2026-11-03:** the `ds4-gemma4` runtime (a fork of
> `JLFN/ds4-dfm-rs`) serves this release on one RTX 4070 SUPER (sm_89). The
> serving checklist passes at ctx 2,048: prefix, rewind, continue, disk
> session, snapshot and the banked lane, 10 works, 0 FAIL, 0 WRONG,
> 0 undeclared. The last measured speed record is 13.42 tok/s decode and
> 374 tok/s prefill (ctx 256, eight greedy tokens, device-resident, commit
> `d4f6d4f9`); the current HEAD `659a152b` adds fused expert gate/up and
> optional FP8 full-layer KV and has not been re-measured. The reference gate
> on a GB10 compared post-softcap full-vocabulary distributions with a fp32
> transformers reference at commit `cc1451df`: cosine 0.9906 and 0.9901 on
> the two probes, KL 0.000548 and 0.419508. No held-out KL, perplexity or
> benchmark was run.

## At a glance

| | |
|---|---|
| Source | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) at `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (BF16, 2 shards); drafter from [`google/gemma-4-26B-A4B-it-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant) at `6e5aaaf4c42b98394530b8fda2e95cadd65c151c` |
| Recipe | `gemma-4-26b-a4b-family-r8` ([RECIPE.md](RECIPE.md), [`recipe.json`](recipe.json)) |
| Routed experts | 30 MoE layers × 128 experts × gate/up/down. 8-dim VQ, one E4M3 main book and one 256-entry E4M3 residual book per expert stack, one F16 gain per row (1.0 in this uncalibrated build). **Main: 13-bit (K=8192, 12-bit stream + 1-bit plane); residual: 8-bit GVR8** — 1.64 + 0.59 index bits per weight. |
| Attention, dense FFN gate/up, output head | Q4_K; the dense down matrices are Q8_0 because 2,112 columns are not a multiple of the Q4_K block of 256 |
| Kept at source precision | Router weights, norms, layer scalars, token embedding (BF16) |
| Token embedding | BF16, in its own part, read in rows from SSD (not resident) |
| Context | **262,144 tokens**, the model's own: 25 sliding-window-1024 layers and 5 full-attention layers; no RoPE scaling |
| Chat template | The checkpoint has no official `chat_template.jinja`; the family records the token grammar (turn 105/106, channel 100/101, `<|think|>` 98, tool 46–51, image 258880, video 258884) |
| Fidelity (reference forward, release bytes vs fp32) | Probe A, 28 tokens: cosine **0.9906**, KL **0.000548**, argmax 818; probe B, 1,243 tokens: cosine **0.9901**, KL **0.419508**, argmax 1509. No held-out KL, perplexity or benchmark. |
| Drafter | `google/gemma-4-26B-A4B-it-assistant`, 48 tensors, Q4_K/BF16, 0.58 GiB; the draft path (`--mtp-draft`) is not declared yet |
| Runtime | `ds4-gemma4` (fork of `JLFN/ds4-dfm-rs`), branch `feat/gemma4-family`, HEAD `659a152b` at writing; CUDA on sm_89 |
| RTX 4070 SUPER speed | 13.42 tok/s decode, 374 tok/s prefill at `d4f6d4f9`, ctx 256, eight greedy tokens, device-resident; not re-measured at the current HEAD |

## Why this size: the 12 GiB card decides it

Gemma 4 26B A4B is 25.2B parameters total and 3.8B active. The routed-expert
stacks are 90.5% of the encoded elements, so that is where the VQ budget goes.
The release is 9.95 GiB: an 8.00 GiB trunk, a 1.38 GiB embedding part and a
0.58 GiB optional drafter. The engine maps the 9.37 GiB release for device
access and keeps the embedding part on the host: each token's row is gathered
from the page cache. At ctx 2,048 on the 12 GiB card the fit chose two banks
with 4.23 GiB spare.

The full 262,144-token context is the model's own, but only ctx 2,048 is
qualified. `ds4-server` needs `--mem-floor-gb 2` on a 12 GiB card. SSD
streaming is optional and off by default; the 4070 SUPER refuses it (coherent
access), so the resident path is the one measured. This is why the dense
skeleton stays at Q4_K/Q8_0 and the token table is a separate part: the VQ
budget goes to the experts, and the bytes read on every token stay cheap.

## Recipe

- **Routed experts.** Every 8 consecutive weights of an expert row become one
  index into an E4M3 codebook, times one F16 row gain. One main book per
  expert stack serves all 128 experts and the gate/up/down matrices; one
  256-entry E4M3 residual book per stack serves the GVR8 plane. The main index
  is 13 bits (K=8192): a 12-bit stream plus a 1-bit plane. The residual is one
  8-bit index per 8-weight vector. Reconstruction is
  `W[r, 8j+d] = gain[r] * (C1[i1][d] + C2[i2][d])`, rounded to the runtime's
  BF16 storage.
- **Codebook training.** The encoder strided-subsamples at most 200,000
  segments, starts from evenly spaced samples, runs eight Lloyd iterations,
  and round-trips every centroid through f16 and E4M3 before the final
  assignment. The builders pass `X = NULL`: the codebooks are fitted to the
  weights alone, with no calibration corpus and no activation importance. The
  F16 row-gain field is therefore `1.0` throughout this artifact; the codeword
  carries the whole approximation. Calibration is the remaining quality lever,
  not a property of the format.
- **What is VQ.** 60 expert stacks (30 layers × gate_up/down). The emitter
  splits a fused gate_up stack into gate and up, and each stack into one matrix
  per expert: 11,520 GVQ1 matrices, 11,520 GVR8 residuals and 120 books in the
  trunk.
- **What is not VQ.** Attention q/k/v/o and output-gate, the dense FFN's
  gate/up and the head are Q4_K; the dense down matrices are Q8_0; norms,
  layer scalars, router weights and the token embedding are BF16. The embedding is a separate
  part. The source's FP32 router logits, tanh-GELU expert activation and
  tanh-30 vocabulary cap stay in the runtime.
- **Weight-space reconstruction.** 23.38% mean relL2 over the 60 expert stacks
  at about 2.22 bits per weight, against 39.29% for the same stacks at 1.6377
  bits per weight without the residual; the residual recovers 15.8 points for
  0.59 more bits per weight. The payload containers are 7.01 GiB (4.35 stage
  plus 2.66 residual). On layer 0's `down_proj`, the measured error is 23.4212%
  with the residual against 39.2647% without; the residual is 31,719,552 bytes
  in a 52,330,552-byte container.

## Measured on the converted bytes

The reference gate ran on a GB10 and compared post-softcap full-vocabulary
distributions at the last position against transformers at commit `cc1451df`
(fp32 reference, same ids, thinking off). The release side reads the weights
from this repository's files; the reference side reads the BF16 checkpoint.

| Probe | ids | Artifact | cosine | KL ref‖port | top-10 | argmax |
|---|---:|---|---:|---:|---:|---:|
| A short | 28 | **R13+8** | **0.9906** | **0.000548** | 6/10 | 818 |
| A short | 28 | 13-bit | 0.9879 | 0.002371 | 6/10 | 818 |
| B long | 1,243 | **R13+8** | **0.9901** | **0.419508** | 8/10 | 1509 |
| B long | 1,243 | 13-bit | 0.9873 | 0.003730 | 8/10 | 1509 |

Read the B KL carefully: it is one token. 1509 contributes 0.425 of the
0.4195, because this artifact puts the top logit 0.6 nats below the reference
(24.5 against 25.125), where the 13-bit artifact put it 0.75 above (25.875).
Every shape metric improved: cosine, rms delta (2.44 to 2.38 on A, 1.72 to
1.36 on B) and the runner-up order. Weight error 39 to 23 percent is a better
approximation of the same model, not a different model; no claim beyond these
numbers, and no perplexity or benchmark was run.

**Serving checklist.** `ds4 --serving-check` on this artifact at ctx 2,048
(sm_89, 2026-11-03): prefix, rewind, continue, disk session, snapshot, banks
warm/fork/partial/disk/width — 10 works, 0 FAIL, 0 WRONG, 0 undeclared. The
check is a structural and numerical comparison against the live state it
saved; it is not a quality claim.

**Speed.** The last published record is `d4f6d4f9`: 13.42 tok/s decode and
374 tok/s prefill on the RTX 4070 SUPER, ctx 256, eight greedy tokens,
device-resident (`DS4_CUDA_COPY_MODEL=1`, R13+8). The current HEAD adds fused
expert gate/up and optional FP8 full-layer KV; those numbers have not been
re-measured.

## Serving with ds4-gemma4

Build the runtime with `make cuda CUDA_ARCH=sm_89`, then pass the release
directory as the model:

```sh
# the serving checklist
DS4_LOCK_FILE=/tmp/ds4-gemma4-gate.lock \
  ./ds4 -m <release-dir> --cuda -c 2048 --serving-check -p "Hello"

# a server on a 12 GiB card
./ds4-server --cuda -m <release-dir> --mem-floor-gb 2 --host 0.0.0.0 --port 8000
```

- Pass the directory, not a single file. The three parts are independent GGUF
  files: `gemma4-vq8-trunk.gguf`, `gemma4-vq8-embedding-BF16.gguf` and
  `gemma4-vq8-mtp-Q4_K.gguf`. Do not concatenate them.
- Standard llama.cpp, Ollama and Transformers loaders do not read this format.
- The release is text-only. The vision tower is not included. Audio is declared
  in the checkpoint but has zero tensors and `audio_config: null`; do not wire
  it from the processor configuration.
- The trunk part carries the tokenizer. There is no official chat template, so
  the runtime renders the prompt from the token grammar.
- The drafter part is present and validated by the inventory; `--mtp-draft` is
  not declared yet.

## What stays native

- **Context:** 262,144 tokens. Sliding layers: 25 layers, window 1,024,
  head_dim 256, 8 KV heads, RoPE theta 1e4. Full layers: 5 layers, head_dim
  512, 2 KV heads, proportional RoPE (0.25 of the dims, 128 rotated dims),
  theta 1e6, and `attention_k_eq_v` so the value projection is the key
  projection. Nothing is rescaled or extended.
- **Sampling:** the source `generation_config.json` (do_sample true,
  temperature 1.0, top_p 0.95, top_k 64). The runtime's own defaults may
  differ; check the runtime record.
- **Numerics kept in FP32 by the source:** router logits, tanh-GELU experts,
  logit soft-capping at 30.
- **Chat:** the token grammar, not a Jinja template. Special tokens: turn
  105/106, channel 100/101, think 98, tool 46–51, image 258880, video 258884,
  audio 258881 (declared, unbacked), eos 1, bos 2.

## Files

| Path | Contents | Size |
|---|---|---:|
| `gemma4-vq8-trunk.gguf` | The text model: 11,520 GVQ1 + 11,520 GVR8 + 120 GVBC, 175 Q4_K, 30 Q8_0, 391 BF16; 23,756 tensors | 8,588,461,824 B (8.00 GiB) |
| `gemma4-vq8-embedding-BF16.gguf` | BF16 token table, 262,144 × 2,816, one tensor | 1,476,395,648 B (1.38 GiB) |
| `gemma4-vq8-mtp-Q4_K.gguf` | The assistant drafter, 48 tensors, Q4_K/BF16 | 621,995,136 B (0.58 GiB) |
| **All 3 parts** | | **10,686,852,608 B (9.95 GiB)** |
| `release.json` | Entry point: every part with role, size and SHA-256 | |
| `files.json` | Index of every file with its size and SHA-256 | |
| `config.json` | The checkpoint config, byte-identical to the copy the runtime validates | |
| `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, `processor_config.json` | Original files, unchanged | |
| `RECIPE.md`, `FORMAT.md`, `SOURCE.md` | Recipe, byte format, provenance | |
| `SHA256SUMS` | Hashes of every file in this repository | |

## Upstream scores

Google's scores for the original checkpoint, from the source model card. They
describe the source model, not this quantization.

| Benchmark | Gemma 4 26B A4B |
|---|---:|
| MMLU Pro | 82.6% |
| AIME 2026 no tools | 88.3% |
| LiveCodeBench v6 | 77.1% |
| GPQA Diamond | 82.3% |
| Tau2 (average over 3) | 68.2% |
| HLE no tools / with search | 8.7% / 17.2% |
| MMMLU | 86.3% |
| MMMU Pro | 73.8% |
| MRCR v2 8 needle 128k (average) | 44.1% |

## Limits

- The Hub's GGUF panel parses one `.gguf` file at a time. Its "Model size"
  comes from the BF16 embedding part (0.7B), not the 25.2B text model, and its
  variant selector reads the filenames. The `BF16` and `Q4_K` suffixes are for
  that panel; the release is the directory and `release.json`, and the panel's
  selector is not the runtime. The hardware line is the viewer's suggestion for
  the selected file, not a requirement; the release runs on a 12 GiB RTX 4070
  SUPER.
- This is lossy: 23.38% mean relL2 over the routed experts at about 2.22 bits
  per weight. The release has no held-out KL, perplexity or benchmark; the
  reference gate is two probes, not a corpus.
- The B probe's KL is 0.4195, one token's 0.6-nat deficit at the sampled
  position; the distributions are close, not equal, and the residual is the
  encoding's error.
- Only ctx 2,048 is qualified. `ctx_max` is 262,144 but `qualified_ctx` is
  unset.
- Text-only. The vision tower is skipped; audio is declared but unbacked.
- There is no official chat template; the token grammar is the protocol.
- The drafter part is present, but `--mtp-draft` is not declared.
- The serial session has no ring-resume guard: a divergent continuation that
  reaches back past the rows the ring still holds can read overwritten slots.
  The banked lane is exact and the checklist does not exercise the serial
  case.
- `ds4-server` needs `--mem-floor-gb 2` on a 12 GiB card.
- SSD streaming is optional and refuses on the 4070 SUPER (coherent access);
  the resident path is the one measured.
- The codebooks are uncalibrated (`X = NULL`). Calibration is the next quality
  lever, not a property of this release.
- The GPU codebook update sums in FP64 with atomics, so a rebuild is not
  byte-deterministic run to run. The published bytes are the record; the CPU
  path is the byte-deterministic reference.
- The artifact-vs-checkpoint checker in VQ_tools (`vq-quality`) currently
  decodes the GVQ1 stage only: it reports 39.29% and flags the manifest's
  23.39% as drift. The 23.38% figure is the build's combined reconstruction
  including GVR8; a checker that decodes the residual is the next tool unit.
- The current runtime HEAD `659a152b` is under active development and is not
  the commit that produced the 13.42 tok/s record (`d4f6d4f9`).

## Credits and sources

- **[Google DeepMind](https://deepmind.google/models/gemma/)** — Gemma 4
  26B A4B, Apache-2.0.
- **[YoungAi](https://github.com/fodelf/YoungAi)** by Wenzhou Wu (吴文周,
  [fodelf](https://github.com/fodelf) / [wenzhouwu](https://huggingface.co/wenzhouwu)),
  MIT — the method: 8-dim VQ, per-layer E4M3 codebooks, F16 row gains, the
  12 + 1 bit-plane layout for 13-bit layers, shallow-layers-first bit
  allocation.
- **[ds4-dfm-rs](https://github.com/JLFN/ds4-dfm-rs)** — the runtime: the
  GVQ1/GVR8 reader, the Gemma 4 family and its CUDA kernels. The Gemma 4 work
  is the `feat/gemma4-family` branch of the `ds4-gemma4` worktree.
- **[llama.cpp / GGML](https://github.com/ggml-org/llama.cpp)** — the GGUF
  container and the Q4_K and Q8_0 layouts.
- **VQ_tools** — the conversion tools (MIT): the codebook, the builder, the
  `vq-gguf` emitter and the release index.

## License

Apache-2.0, as the source model. The conversion tools are MIT. The release
includes a copy of the Apache-2.0 license. Gemma is a trademark of Google;
this is an unofficial quantization, not a Google release.
