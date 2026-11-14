---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
  - glq
  - quantized
  - trellis
  - tcq
  - moe
---

# gemma-4-26B-A4B-it — GLQ trellis (3INST) 4 bpw

[gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) quantized with
[GLQ](https://github.com/cnygaard/glq) using the QTIP-style trellis (TCQ) codebook at
**4 bits/weight**, lookup-free 3INST variant. 128 routed experts per layer, **13.95 GB on
disk** against ~52 GB for the bf16 original.

```bash
python -m venv .venv
source .venv/bin/activate
pip install 'glq[vllm]>=0.8.1' "transformers>=5.13.1,<5.15"
vllm serve xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw --quantization glq
```
**Pin transformers below 5.15**, or vLLM dies before loading a single weight — 5.15.0 moved
gemma-4 to a per-layer config and `config.head_dim` now raises. Verified on 5.14.1; details
under [Runtime support](#runtime-support).

**glq 0.8.1 is the minimum** — it added the fused grouped trellis MoE decode. Earlier
releases refuse this checkpoint at load rather than mis-serve it.

## Quality

| | AIME 2026 |
|---|---|
| precision | weights | AIME 2026 (avg@8) | source |
| :-- | :-- | :-- | :-- |
| bf16 | ~52 GB | 88.3% | Google's published figure |
| NVFP4 (`nvidia/Gemma-4-26B-A4B-NVFP4`) | 18.2 GiB | 86.25% | measured here |
| **this checkpoint, GLQ 4 bpw** | **14.4 GiB** | **86.25%** | measured here |

GLQ and NVFP4 were run **in the same session on the same GPU with identical settings**:
avg@8 over the 30 problems, 32k thinking budget, the model's own sampling defaults
(temp 1.0 / top_p 0.95 / top_k 64), vLLM 0.27.1, RTX PRO 6000 Blackwell. Mean generation was
15,026 tokens (GLQ) and 15,117 (NVFP4) — long enough to confirm reasoning actually engaged,
which is the only reliable check that a thinking eval was not silently run in no-think mode.

|  | GLQ 4 bpw | NVFP4 |
| :-- | :-- | :-- |
| avg@8 | 207/240 | 207/240 |
| pass@8 | 29/30 | 28/30 |
| solved on all 8 samples | 23/30 | 21/30 |
| truncated at 32k | 29 | 24 |
| no answer | 5 | 10 |

**Read GLQ vs NVFP4 as a tie.** The avg@8 figures are identical and the tail differences are
one or two problems at n=30, where the standard error of the mean is 5.4 pt. The bf16 row is
a *published* number from a different harness, not a paired arm run here — treat 88.3% as a
reference point, not a measured delta.

Nothing else is measured — no perplexity, no MMLU-Pro. A 5-task Terminal-Bench 2 smoke run
scored 0/5 for both this checkpoint and NVFP4, which is too small to rank anything.

## Speed and footprint

All three measured in one session on one RTX PRO 6000 Blackwell, vLLM 0.27.1, 128-token
prompts / 256-token outputs, `ignore_eos`, `max_model_len 2048`:

|  | GLQ 4 bpw | bf16 | NVFP4 |
| :-- | :-- | :-- | :-- |
| loaded (vLLM's own figure) | **14.4 GiB** | 48.5 GiB | 18.2 GiB |
| load time | **6.8 s** | 67.0 s | 37.7 s |
| **B=1** decode, per stream | 88.4 tok/s | 131.4 tok/s | 153.9 tok/s |
| **B=1** TTFT | **57 ms** | 88 ms | 81 ms |
| **B=1** TPOT | 11.31 ms | 7.29 ms | 6.21 ms |
| **B=32** aggregate | 944 tok/s | 1600 tok/s | 2966 tok/s |
| **B=32** TTFT | 2233 ms | 138 ms | 88 ms |
| **B=32** TPOT | 25.25 ms | 19.52 ms | 10.47 ms |

**On this GPU, GLQ is the smallest and the slowest of the three.** NVFP4 uses Blackwell's
native FP4 tensor cores, which the trellis decode does not; bf16 has no dequantization step
at all. GLQ's advantages here are footprint — 21% under NVFP4 and 3.4x under bf16 — and a
much faster load.

Two things not to gloss over. GLQ's **prefill degrades badly under concurrency**: 2233 ms to
first token at B=32 against 138 ms (bf16) and 88 ms (NVFP4), a far larger gap than the 1.3x
seen in per-token decode. And these rankings are **GPU-specific**: GLQ's advantage comes from
reading 4x less weight memory, so it is largest on bandwidth-starved cards and smallest on a
96 GB Blackwell. A 24 GB card will not reproduce any of these numbers — including the ones
that flatter the alternatives, which cannot fit bf16 at all.

## Runtime support

| runtime | status |
|---|---|
| **vLLM** ≥ 0.25 with glq ≥ 0.8.1 | **works** — the numbers above |
| vLLM with glq < 0.8.1 | refuses at load, by design |
| **HF transformers** | **broken — not a fault of this checkpoint** |

The HF path produces incoherent output for gemma-4 MoE under *every* GLQ codebook: the
published, vLLM-validated E8-shell checkpoint of this same base model fails identically.
Nothing outside GLQ's quantizer reassembles gemma-4's stacked `Gemma4TextExperts` tensors, so
the per-expert artifacts are never wired up. Use vLLM.

### transformers version — pin below 5.15

vLLM reads this model's config through transformers, and **5.15.0 moved gemma-4 to a
per-layer config**: `config.head_dim` raises `AmbiguousGlobalPerLayerAttributeError`, which
vLLM hits while building its `ModelConfig`. The server then dies *before a single weight is
loaded*.

Verified on **transformers 5.14.1** with vLLM 0.27.1 (the configuration the numbers above
were measured on). This is not specific to GLQ or to this checkpoint — stock bf16
`google/gemma-4-26B-A4B-it` fails the same way on 5.15.0.

## How it was produced

```bash
GLQ_TRELLIS_VARIANT=3inst glq-quantize \
    --model google/gemma-4-26B-A4B-it \
    --codebook trellis --bpw 4 --nsamples 128 --seqlen 2048 --streaming
```

7049 s, 128 calibration samples × 2048 tokens (WikiText-2); average weight SQNR 22.11 dB over
242 layers (min 21.2, max 22.8).

The 128 routed experts per layer are quantized as one fused `[gate; up]` matrix and split back
per projection. That split is 32-row aligned by necessity: the packed trellis is stored in
MMA-fragment order, where the byte permutation pairs 16-row blocks, so a cut inside a pair
interleaves gate and up bytes into weights that still load and still decode — incorrectly.
See [docs/trellis-3inst.md](https://github.com/cnygaard/glq/blob/main/docs/trellis-3inst.md).

gemma-4 is a thinking model — use its chat template and give it a real reasoning budget.

## License

Inherits [Apache 2.0](https://huggingface.co/google/gemma-4-26B-A4B-it) from the base model.

---
⭐ [GLQ on GitHub](https://github.com/cnygaard/glq)
