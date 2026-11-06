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
---

# Qwen3.8-Flash-Next — native Halogen HGN (v2)

Native **Halogen `.hgn`** weights for
[halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server),
converted from the official Hugging Face **BF16** snapshot
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

**These weights are Halogen’s `.hgn` format and will not load in Transformers,
vLLM, llama.cpp, or similar.** They are meant to be mounted into a
halogen-flash-server container (or an equivalent Halogen runtime).

Upload target: [`johnlockejrr/Qwen3.8-Flash-Next-HGN`](https://huggingface.co/johnlockejrr/Qwen3.8-Flash-Next-HGN).

## What this is

A **v2** Halogen pack built so MTP, vision, and n-gram come from Qwen’s own
BF16 safetensors — not from older author-extracted HGN sidecars used for
GGUF imports.

| File | Size | Role |
|---|---|---|
| `model.hgn` | 62.11 GiB | Trunk + **embedded MTP** (identity `qwen3.8-flash-next-v2`) |
| `qwen38-flash-next-ngram.hgn` | 47.68 GiB | N-gram lookup table (**required**; keep beside the checkpoint) |
| `qwen38-flash-next-vision.hgn` | 0.84 GiB | Vision tower (optional; enable with `HALOGEN_VISION_TOWER=1`) |
| `tokenizer/` | — | Tokenizer + chat template assets |
| `config.json` | — | Model config (from source) |
| `conversion-manifest.json` | — | Exact convert provenance / plan |
| `sha256.txt` | — | SHA-256 of the three `.hgn` payloads |

Rough total download: **~111 GiB**. Plan on a machine that can hold the
resident checkpoint (~62 GiB) plus room for the memory-mapped n-gram table
and runtime KV — typically a dedicated host (e.g. Strix Halo–class unified
memory), same class of deployment as the upstream Halogen packs.

There is **no separate `mtp.hgn`**: the draft head is inside `model.hgn`.

## How it was made

```text
source:  Qwen/Qwen3.8-Flash-Next   (HF safetensors, BF16, 131 shards)
tool:    bf16-to-hgn 0.1.0         (CUDA build)
profile: v2
policy:  bf16-hgn-v2-rtn-nf4-signed-rms-trellis-v1
hgn-spec: 1.1.0 @ 79e75fd4ae7642478823525b0a3212951f584af5
```

Convert (CUDA encode on RTX 3090–class GPUs):

```bash
BIN=/opt/halogen/bf16-to-hgn-cuda/target/release/bf16-to-hgn
SRC=/path/to/Qwen3.8-Flash-Next          # full HF BF16 snapshot
OUT=/path/to/qwen38-flash-next-bf16-v2-cuda

$BIN convert --source "$SRC" --output "$OUT" \
  --profile v2 --seed 0 --threads 8 --ht-beam 128 \
  --ht-backend cuda --i4r-backend cuda --q4cp-backend cuda
```

Verify (CUDA decode):

```bash
$BIN verify --source "$SRC" --output "$OUT" --decode-backend cuda \
  > verify-qwen38-bf16-v2-cuda.json
```

On this build: convert published successfully; verify completed with
`provenance_attested: true`. Vision tensors matched BF16 exactly
(raw-copy). Quantized tensors show the usual trellis / Q4C-P / I4R
reconstruction error vs BF16 — that is expected and is **not** an inference
quality score.

HF inputs do **not** take `--mtp` / `--vision` / `--ngram` / `--assets`;
those flags are for GGUF/HGN imports. Everything auxiliary was taken from
the BF16 snapshot itself.

## How to run

Same idea as the upstream Halogen weight repos: download the pack, mount it
read-only as `/models`, run halogen-flash-server.

```bash
hf download johnlockejrr/Qwen3.8-Flash-Next-HGN \
  --local-dir ~/halogen-models/Qwen3.8-Flash-Next-HGN

# This pack ships the checkpoint as model.hgn (not qwen38-flash-next-v2.hgn).
podman run --rm -p 8731:8731 \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ipc=host --ulimit memlock=-1:-1 \
  -e HALOGEN_CHECKPOINT=/models/model.hgn \
  -e HALOGEN_VISION_TOWER=1 \
  -v ~/halogen-models/Qwen3.8-Flash-Next-HGN:/models:ro \
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

## Precision (v2)

| | |
|---|---|
| Checkpoint | ~62.1 GiB, Halogen v2 policy |
| Routed experts | I4R |
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
- **[Quat3rnion](https://github.com/Quat3rnion)** — [`bf16-to-hgn`](https://github.com/Quat3rnion/bf16-to-hgn), the converter used to produce these `.hgn` files (CUDA backends for HT / I4R / Q4C-P).
- **[peonist-ai](https://github.com/peonist-ai)** — [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) and the Halogen `.hgn` runtime this pack is intended for. See also their reference weight card style at [`peonist-ai/halogen-qwen3.8-flash-next`](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next).

## SHA-256

```
3dd0598087b3529251c56cf7884c08819e0064f81b5e959f566e97af0c1f4033  model.hgn
339f5c508ca592ee2613ac5c3082587a88991d5592c4160d49a32a6dbdb3f30b  qwen38-flash-next-ngram.hgn
b6ca6716eaab601773f7155701bb09f39c72922caa21620bab577f0281f2ea2a  qwen38-flash-next-vision.hgn
```

## Disclaimer

Community conversion for Halogen. Not an official Qwen or peonist release.
Verify hashes after download. Inference quality depends on the Halogen
runtime, sampling settings, and your workload — use `tool-eval-bench` or
your own evals before trusting it in production.
