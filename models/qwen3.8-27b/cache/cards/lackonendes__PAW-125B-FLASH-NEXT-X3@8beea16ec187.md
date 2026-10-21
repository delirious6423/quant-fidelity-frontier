---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
tags:
- quantization
- trellis-codec
- moe
- paw
- gguf
---

# PAW-125B-FLASH-NEXT-X3

_If this model is useful to you, support the work by following [@nawariokra](https://x.com/nawariokra) on X._


**Flash-Next MoE build of `Qwen/Qwen3.8-27B`**, expert bank trellis-coded at
2.631 bpw (PAW X3, no pruning), decoder at Q8_0, with a **256k context**
target. Two serving shapes are measured: **2x RTX 3090 full offload**
(`llama-paw`, 45.5 GiB VRAM at 81,920 ctx) and **1x RTX 3090 + CPU hybrid**
(`TRUSS`, our own engine — measured decode 55-85 tok/s and prefill
2,146 tok/s at full 256k context on one card; see [Serving](#serving)).

| | |
|---|---|
| files | 2 GGUF shards, 90.36 GiB total (self-contained — no separate sidecar) |
| shard 1 | decoder (Q8_0, 5.07 GiB) + expert bank (2.631 bpw PAW X3 trellis, 37.00 GiB, mixed K1-K4 per-layer allocation) |
| shard 2 | PLE per-layer token embedding table (q8), 48.28 GiB |
| context | 256k design target; 45.5 GiB VRAM at 81,920 ctx (2x3090, full offload); TRUSS runs the full 256k on 1x3090 + CPU |
| code (fastgate, no-think) | **97.0% HumanEval / 93.9% HumanEval+ / 96.3% MBPP / 80.7% MBPP+** |
| general (fastgate, no-think) | 65.4% MMLU-Pro (327/500), 97.0% GSM8K (97/100), 59.1% GPQA (117/198) |

## Demo

A voxel-art scene (Eiffel Tower garden, Paris at night), generated in one shot from a text prompt
(three.js/WebGL code, written by this model, rendered in a browser):

<video controls width="100%" src="https://huggingface.co/lackonendes/PAW-125B-FLASH-NEXT-X3/resolve/main/assets/eiffel_tower_flashnext.mp4"></video>

An interactive soft-body physics demo (Watermelon Jelly - Specimen 001), generated in one shot from a text prompt
(WebGPU/WGSL code, XPBD tetrahedral soft body, written by this model, rendered in a browser):

<video controls width="100%" src="https://huggingface.co/lackonendes/PAW-125B-FLASH-NEXT-X3/resolve/main/assets/watermelon-jelly.mp4"></video>

## Requirements — read this first

This is **not** a standard GGUF. It uses the PAW x3 trellis format with a
Flash-Next MoE expert bank. **Stock llama.cpp cannot read these models.**
Two engines can load it: [`llama-paw`](https://github.com/irawanw/llama-paw)
(2x3090, full offload) or [`TRUSS`](https://github.com/irawanw/truss) (1x3090
+ CPU, our own from-scratch engine — see [Serving](#serving) for measured
numbers). Both shards below are required either way; nothing else is needed
except the MTP head noted under [Serving](#serving).

## Files

| file | contents | size |
|---|---|---:|
| `qwen38-flash-next-paw-x3-q8_0-00001-of-00002.gguf` | decoder + PAW X3 expert bank | 42.08 GiB |
| `qwen38-flash-next-paw-x3-q8_0-00002-of-00002.gguf` | PLE per-layer token embedding table | 48.28 GiB |

Both files are one GGUF split into two parts (declared in the GGUF header
itself, `split.count=2`) — this is not a bundle of a model plus an optional
extra. `llama-paw` requires both shards present with these exact filenames
(`-0000N-of-0000M.gguf`) to load the model at all.

## Fastgate evaluation

The point of this table is coherence, not a leaderboard chase: at 2.631 bpw
for the expert bank, the question worth answering first is whether the
model is still a functioning model at all. High pass rates across code,
math and knowledge benchmarks — not near-zero scores or empty completions —
show the quantization has not collapsed the model at this bit rate.

Every score below is from the same no-think fastgate harness, greedy
decoding, `--reasoning off` on the server (thinking suppressed at the
template level, not just hidden). Two passes were run for this arm: an
initial pass and a `_p0` rescue pass with a larger token cap on the items
that came back unparsed the first time (MMLU-Pro unparsed dropped 11 -> 2,
GPQA got a full second pass at a larger cap). Numbers below use the more
complete pass for MMLU-Pro/GPQA/IFBench and the single available pass for
the rest.

| benchmark | items | max_tokens | score |
|---|---:|---:|---:|
| MMLU-Pro | 500 | 128 | 327/500 (65.4%) |
| GPQA | 198 | 4096 (2048 in the initial pass) | 117/198 (59.1%) |
| IFBench-64 | 64 | 2048 | strict 15/64 (23.4%), loose 21/64 (32.8%) |
| HumanEval (EvalPlus) | 164 | 2048 | pass@1 97.0% |
| HumanEval+ | 164 | 2048 | pass@1 93.9% |
| MBPP | 378 | 2048 | pass@1 96.3% |
| MBPP+ | 378 | 2048 | pass@1 80.7% |
| GSM8K | 100 | 4096 (ceiling 8192) | 97/100 (97.0%) |

All are single-artifact, single-pass fastgate numbers on this exact GGUF —
not copied from any model card. No thinking-mode rescue has been run for
this arm (unlike the sibling B3.5 dense-X3 model, where a separate 32k
thinking-mode pass substantially raises MMLU-Pro and IFBench); treat the
table above as the no-think floor, not a ceiling.

## Serving

Full instructions and measured numbers for both engines are in
**[SERVING.md](SERVING.md)**. Summary:

| | [`llama-paw`](https://github.com/irawanw/llama-paw) | [`TRUSS`](https://github.com/irawanw/truss) |
|---|---|---|
| GPUs | 2x RTX 3090, full offload | **1x RTX 3090 + CPU hybrid** |
| tuned command line | not yet verified for this artifact | **below, measured on 1x3090** |
| decode (256k, 9-prompt bench) | unmeasured | **84.8 tok/s greedy** / 54-68 tok/s served (sampled) |
| prefill (256k, 21.7K-token prompt) | unmeasured | **2,146 tok/s bench** / 1,964-2,163 tok/s served |

The two shards in this repo are the complete model either way — no separate
sidecar file exists or is needed. Keep both on fast local storage (NVMe),
not spinning disk — the PLE table in shard 2 is read on the hot path during
decode. One piece used in our own TRUSS runs is **not** included in this
repo:

- **MTP head** (~1.1 GiB) — built from the TRUSS repo's own tooling, not
  bundled here; see [SERVING.md](SERVING.md) and Limitations.

For `llama-paw`, a fully tuned, verified `llama-server` command line
(context/batch/KV-quant settings analogous to the B3.5 256k recipe in
`PAW-27B-X3-GGUF`) has not been re-derived for this Flash-Next artifact yet;
reuse of the X3 dense recipe's `GGML_PAW_X3_GEMV` / `-fa on` /
`--reasoning off` family of flags is a reasonable starting point, not a
confirmed one (full flags in [SERVING.md](SERVING.md)).

## Limitations

- No `llama-paw` recipe has been verified for this exact artifact (TRUSS has
  a measured one — see Serving).
- The MTP head is not included in this HF repo (buildable from the TRUSS
  repo).
- IFBench-64 here is a 64-item sentinel, not the full benchmark, and no
  thinking-mode rescue pass has been run for this arm.
- Long-context (256k) *generation quality* is unmeasured on either engine;
  45.5 GiB VRAM at full offload (llama-paw) is measured at 81,920 ctx, not
  the full 256k target. TRUSS's PP/TG numbers above *are* measured at the
  full 262,144-token context.
- TRUSS's decode is currently 0.65-0.85x of Strata (a different, lower-bit
  quantization of this base model) on the same single-GPU hardware; prefill
  already exceeds it. This is open work in the TRUSS repo, not a ceiling.
- Neither engine's output quality has been measured against the BF16 parent
  at model level (KL) for this specific artifact — see SERVING.md.

## Integrity

```
e0d01657af0bd88bed2c6d8127f9425bb9e36adfb5df03b80bcdf5e34157c8d9  qwen38-flash-next-paw-x3-q8_0-00001-of-00002.gguf
1296edd1881163cf0b26962dc59c416efdfc29987dbc2cd8a7fbc79f69a2149d  qwen38-flash-next-paw-x3-q8_0-00002-of-00002.gguf
```

## License

Apache 2.0, inherited from the base model `Qwen/Qwen3.8-27B`.

## Credits and attribution

- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp — its
  EXL3 format (a QTIP-style trellis/Viterbi quantizer) is the basis for the
  PAW X3 / X3.1 codec that compresses this model's expert bank to ~2.6
  bits/weight; the TRUSS decode kernels and encoder's refit/drift step port
  that design.
- [Strata](https://github.com/Niko1221/Strata) by Niko1221 — the serving
  architecture TRUSS's 1-GPU path is built on: keep the hot experts on the
  GPU, compute the rest on the CPU from RAM, hand off between them through a
  flag in pinned memory instead of a blocking copy, and recover the speed
  with speculative decoding. TRUSS reimplements this shape independently
  (no shared code), measured against Strata on the same hardware in
  [SERVING.md](SERVING.md).
- [PrismML Bonsai](https://github.com/PrismML-Eng/Bonsai-demo) and its
  [llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) — the fastgate
  evaluation harness used for every score in this card.
- [TRUSS](https://github.com/irawanw/truss) — our own from-scratch inference
  engine (no ggml runtime), built for serving this model on a single 24 GiB
  GPU with a CPU expert tier. See [SERVING.md](SERVING.md) for measured
  numbers.
- Base model [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by Qwen.
