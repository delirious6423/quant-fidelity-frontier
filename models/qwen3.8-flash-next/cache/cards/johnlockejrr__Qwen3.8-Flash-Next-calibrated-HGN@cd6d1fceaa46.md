---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: halogen
inference: false
tags:
  - halogen
  - halogen-flash
  - qwen
  - qwen3.8
  - flash-next
  - moe
  - mixture-of-experts
  - quantized
  - hgn
  - bf16-to-hgn
  - calibrated
  - gptq
---

# Qwen3.8-Flash-Next — Calibrated Halogen HGN (v2)

Native **Halogen `.hgn`** weights for
[halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server),
converted from the official Hugging Face **BF16** snapshot
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
with the opt-in **`--quality calibrated`** tier: block-diagonal per-expert
Hessians drive GPTQ code selection for **I4R routed experts** (Variant A —
Optimal Lloyd-Max group scales held bit-identical; decode contract unchanged).

**These weights are Halogen’s `.hgn` format and will not load in Transformers,
vLLM, llama.cpp, or similar.** They are meant to be mounted into a
halogen-flash-server container (or an equivalent Halogen runtime).

Upload target:
[`johnlockejrr/Qwen3.8-Flash-Next-calibrated-HGN`](https://huggingface.co/johnlockejrr/Qwen3.8-Flash-Next-calibrated-HGN).

Sibling calibration-free pack:
[`johnlockejrr/Qwen3.8-Flash-Next-HGN`](https://huggingface.co/johnlockejrr/Qwen3.8-Flash-Next-HGN)
(`standard` / historical v2 policy).

## What this is

A **v2** Halogen pack built so MTP, vision, and n-gram come from Qwen’s own
BF16 safetensors — not from older author-extracted HGN sidecars used for
GGUF imports. Relative to the standard v2 pack, **only I4R routed-expert
codes** change (warm experts with enough calibration tokens); HT / Q4C-P /
vision / n-gram policy matches v2. Cold or low-count experts (&lt;256 routed
tokens) fall back to calibration-free Optimal codes.

| File | Size | Role |
|---|---|---|
| `model.hgn` | ~63 GiB | Trunk + **embedded MTP** (identity `qwen3.8-flash-next-v2`) |
| `qwen38-flash-next-ngram.hgn` | ~48 GiB | N-gram lookup table (**required**; keep beside the checkpoint) |
| `qwen38-flash-next-vision.hgn` | ~0.84 GiB | Vision tower (optional; enable with `HALOGEN_VISION_TOWER=1`) |
| `tokenizer/` | — | Tokenizer + chat template assets |
| `config.json` | — | Model config (from source) |
| `conversion-manifest.json` | — | Exact convert provenance / plan (includes `calibration` node) |
| `sha256.txt` | — | SHA-256 of the three `.hgn` payloads |

Rough total download: **~111 GiB**. Plan on a machine that can hold the
resident checkpoint (~63 GiB) plus room for the memory-mapped n-gram table
and runtime KV — typically a dedicated host (e.g. Strix Halo–class unified
memory), same class of deployment as the upstream Halogen packs.

There is **no separate `mtp.hgn`**: the draft head is inside `model.hgn`.

### tool-eval-bench (greedy)

On an internal `tool-eval-bench` leaderboard (request `temperature: 0` /
greedy; Halogen `serve_api` request fields win over server defaults), this
pack scored:

| Run | Score | Pass^3 | Deploy |
|---|---|---|---|
| `calibrated-temp0` (this pack) | **95.3 ± 1.5** | **91.3%** | 86 |

For comparison on the same board, `thinkingcap-temp0` led at 96.0 ± 0.0 /
94.2% Pass^3. Prefer `--parallel 1` for temp-0 Halogen benches (higher
concurrency can fault in `k_argmax_part`).

## How it was made

```text
source:     Qwen/Qwen3.8-Flash-Next   (HF safetensors, BF16, 131 shards)
tool:       bf16-to-hgn 0.1.0         (CUDA build, HGN_CUDA_ARCH=86)
profile:    v2
quality:    calibrated
policy:     bf16-hgn-v2-calibrated-gptq-i4r-v1
gptq-percdamp: 1.0
calibration: block-diagonal per-expert Hessian sidecar
             format_version 1, block_size 128, ~920k tokens (449×2048)
             multi-domain blend (PerfectBlend / RazorCal / WikiText / dialogue)
hgn-spec:   1.1.0 @ 79e75fd4ae7642478823525b0a3212951f584af5
```

Collect Hessian sidecar (streaming, one layer at a time; weights on **NVMe**):

```bash
cd /path/to/bf16-to-hgn-optimal/calibration
uv run hgn-calibrate collect \
  --model /path/to/Qwen3.8-Flash-Next \
  --tokens work/tokens_calibrated.npy \
  -o work/sidecar \
  --block-diagonal --per-expert --block-size 128 \
  --device cuda:0
```

Convert + verify:

```bash
BIN=/path/to/bf16-to-hgn-optimal/target-cuda/release/bf16-to-hgn
SRC=/path/to/Qwen3.8-Flash-Next
OUT=/path/to/qwen38-flash-next-calibrated-v2

$BIN convert --source "$SRC" --output "$OUT" \
  --profile v2 --quality calibrated \
  --calibration /path/to/calibration/work/sidecar \
  --gptq-percdamp 1.0 \
  --seed 0 --threads 8 --ht-beam 128 \
  --ht-backend cuda --i4r-backend cuda --q4cp-backend cuda

$BIN verify --source "$SRC" --output "$OUT" --decode-backend cuda \
  > verify-qwen38-flash-next-calibrated-v2.json
```

On this build: convert published successfully; verify completed with
`provenance_attested: true` (sidecar index + block files re-hashed against
the manifest `calibration` record). N-gram / vision payload hashes match the
sibling standard v2 pack; `model.hgn` differs where calibrated I4R codes
were selected.

HF inputs do **not** take `--mtp` / `--vision` / `--ngram` / `--assets`;
those flags are for GGUF/HGN imports. Everything auxiliary was taken from
the BF16 snapshot itself.

This is **not** a BF16-equivalence claim. The calibrated objective is
Hessian-weighted output error on I4R experts, not perplexity. Damping
`--gptq-percdamp 1.0` was used (probe-validated; textbook `0.01` overfits
at depth).

## How to run

Same idea as the upstream Halogen weight repos: download the pack, mount it
read-only as `/models`, run halogen-flash-server.

```bash
hf download johnlockejrr/Qwen3.8-Flash-Next-calibratedHGN \
  --local-dir ~/halogen-models/Qwen3.8-Flash-Next-calibratedHGN

# This pack ships the checkpoint as model.hgn (not qwen38-flash-next-v2.hgn).
podman run --rm -p 8731:8731 \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ipc=host --ulimit memlock=-1:-1 \
  -e HALOGEN_CHECKPOINT=/models/model.hgn \
  -e HALOGEN_VISION_TOWER=1 \
  -v ~/halogen-models/Qwen3.8-Flash-Next-calibratedHGN:/models:ro \
  ghcr.io/peonist-ai/halogen-flash-server:0.19.1
```

On Docker, replace `--group-add keep-groups` with
`--group-add video --group-add render`.

OpenAI-compatible API on `:8731` (`/v1/chat/completions`, `/v1/completions`,
`/v1/models`, `/v1/responses`). Full server docs:
[peonist-ai/halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server).

Optional: rename `model.hgn` → `qwen38-flash-next-v2.hgn` if you want the
stock filename some Halogen setups auto-pick; keep the n-gram and vision
files beside it with their current names.

## Precision (v2 + calibrated I4R)

| | |
|---|---|
| Checkpoint | ~63 GiB, Halogen v2 + calibrated I4R GPTQ |
| Routed experts | I4R (**GPTQ codes** when Hessian warm; else Optimal fallback) |
| Dense HT matrices / LM head | HT4 |
| Hyper-connection mixers | Q6G64 |
| Embeddings / small projections | Q4C-P |
| MTP dense / mixer matrices | Q8G64 (embedded in `model.hgn`) |
| Vision | BF16 sidecar |
| N-gram table | FP8G, own file, typically paged |

This is a **requantization of official BF16**, not a re-import of an
already-quantized GGUF. Format compatibility is still not the same as
publisher stock-checkpoint equivalence: Halogen’s public converter policy
is explicit and calibrated independently of any private publisher cookbook.

## Credits

- **Qwen / Alibaba** — [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) BF16 weights, architecture, and license terms in `LICENSE`.
- **[Quat3rnion](https://github.com/Quat3rnion)** — [`bf16-to-hgn`](https://github.com/Quat3rnion/bf16-to-hgn), the converter family used to produce these `.hgn` files (CUDA backends for HT / I4R / Q4C-P; calibrated GPTQ tier for I4R).
- **[peonist-ai](https://github.com/peonist-ai)** — [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) and the Halogen `.hgn` runtime this pack is intended for. See also their reference weight card style at [`peonist-ai/halogen-qwen3.8-flash-next`](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next).

## SHA-256

```
2ff30c5edd1527d2fcce7274a347ae7cdd58dc3729321e9601bc1e2c9f9b9ba1  model.hgn
339f5c508ca592ee2613ac5c3082587a88991d5592c4160d49a32a6dbdb3f30b  qwen38-flash-next-ngram.hgn
b6ca6716eaab601773f7155701bb09f39c72922caa21620bab577f0281f2ea2a  qwen38-flash-next-vision.hgn
```

N-gram and vision hashes match the sibling
[`Qwen3.8-Flash-Next-HGN`](https://huggingface.co/johnlockejrr/Qwen3.8-Flash-Next-HGN)
pack; only `model.hgn` is unique to calibrated I4R.

## Disclaimer

Community conversion for Halogen. Not an official Qwen or peonist release.
Verify hashes after download. Inference quality depends on the Halogen
runtime, sampling settings, and your workload — use `tool-eval-bench` or
your own evals before trusting it in production.
