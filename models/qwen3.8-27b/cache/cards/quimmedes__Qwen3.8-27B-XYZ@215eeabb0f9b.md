---
license: apache-2.0
tags:
- qwen
- qwen3.5
- multimodal
- vision
- gguf
- ssm
- quantization
- coding
- code
base_model: Qwen/Qwen3.8-27B
---

# Qwen3.8-27B — GGUF quantizations (XYZ)

GGUF quantizations of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(multimodal, hybrid SSM + attention, 64 layers + 1 MTP layer), converted with
upstream llama.cpp. All files include the built-in **MTP block**
(`nextn_predict_layers=1`), usable for speculative decoding.

## What the XYZ series is (and isn't)

**XYZ is 100% focused on coding.** These quantizations are built and tuned
for programming work: the per-tensor recipe keeps the parts that matter for
code generation (attention K/V, SSM state, and the MTP head) in higher
precision, and the build is evaluated against a code test suite so the
models stay reliable on code tasks — even at low bits.

That focus is deliberate. **General-knowledge domains are not the goal.**
There are already plenty of generalist models out there — this one is the
coding one. If you need the best answer on biology, history, politics, or
other non-code topics, a general-purpose build of the base model will usually
serve you better. XYZ optimizes for coding behavior first, and that's the
trade you're opting into here.

> TL;DR — reach for XYZ when the job is **writing, reading, or debugging
> code**. For everything else it's still a capable general model, but don't
> expect it to out-perform an untuned build on pure knowledge tasks.

![Celestial Dragon](CelestialDragon.png)

## v2 — main series

All v2 files carry the `-v2` suffix.

| File | Size (GiB) | BPW |
|---|---|---|
| Qwen3.8-27B-Q2.5-v3-XYZ.gguf | 9.06 | 3.05 |
| Qwen3.8-27B-Q2.5-XYZ-v2.gguf | 8.60 | 2.70 |
| Qwen3.8-27B-Q3-XYZ-v2.gguf | 10.39 | 3.26 |
| Qwen3.8-27B-Q3.5-XYZ-v2.gguf | 11.95 | 3.75 |
| Qwen3.8-27B-Q4-XYZ-v2.gguf | 14.03 | 4.41 |
| Qwen3.8-27B-Q4.5-XYZ-v2.gguf | 14.62 | 4.59 |
| Qwen3.8-27B-Q5-XYZ-v2.gguf | 16.61 | 5.22 |
| Qwen3.8-27B-Q5.5-XYZ-v2.gguf | 19.51 | 6.13 |
| Qwen3.8-27B-Q6-XYZ-v2.gguf | 20.94 | 6.58 |
| Qwen3.8-27B-Q7-XYZ-v2.gguf | 24.18 | 7.60 |
| Qwen3.8-27B-Q8-XYZ-v2.gguf | 26.71 | 8.39 |
| Qwen3.8-27B-Q9-XYZ.gguf | 29.27 | 9.20 |
| Qwen3.8-27B-ULTRA-XYZ-v2.gguf | 15.76 | 4.93 |

## v2 — low-bit / emergency files

| File | Size (GiB) | BPW |
|---|---|---|
| Qwen3.8-27B-Q1Z-XYZ-v2.gguf | 7.12 | 2.24 |
| Qwen3.8-27B-Q2-XYZ-v2.gguf | 8.15 | 2.56 |
| Qwen3.8-27B-Q1Q-XYZ-v2.gguf | 6.15 | 1.98 |

## Usage

Serve with llama-server (OpenAI-compatible API):

```bash
llama-server -m Qwen3.8-27B-Q5-XYZ-v2.gguf --port 1234
```

With the vision projector (multimodal):

```bash
llama-server -m Qwen3.8-27B-Q5-XYZ-v2.gguf --mmproj mmproj-Qwen3.8-27B-bf16.gguf --port 1234
```

With MTP speculative decoding:

```bash
llama-server -m Qwen3.8-27B-Q5-XYZ-v2.gguf --spec-type draft-mtp --port 1234
```

**Recommended sampling**: `--temp 0.6` for balanced, coherent output.

## What I like to use, in a 24gb gpu

```bash
llama-server -m Qwen3.8-27B-Q5-XYZ.gguf \
--host 127.0.0.1 --port 8080 --ctx-size 128000 -b 2048 --parallel 1 \
-ngl 999 --threads 8 -ub 512 -ctk q8_0 -ctv q8_0 -fa on -kvu \
--no-mmap --temp 0.6 --spec-type draft-mtp,ngram-mod \
--spec-draft-type-k q4_0 --spec-draft-type-v q4_0 --spec-draft-n-max 6 \
--spec-draft-p-min 0.75 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 \
--spec-ngram-mod-n-max 64 --reasoning-budget 2048 --reasoning-preserve \
-mm G:\LLM\Models\mmproj-Qwen3.8-27B-bf16.gguf


using --reasoning-budget 2048 is like set to medium the thinking, use 512 for low thinking and 8196 for high, and use nothing to xHigh.
```


## What I use for 16GB, using mtp is an option but you will be limited to 64k. You can also use q4_0 and double context but there will be degradation, 128k is a sweet spot for most cases.

```bash

llama-server -m Qwen3.8-27B-Q3-XYZ-v2.gguf --host 0.0.0.0 --port 8080
--ctx-size 128000 -b 2048 -ngl 999 --threads 4 -ub 512
-ctk q8_0 -ctv q8_0 -fa on -kvu --temp 0.6
--spec-type ngram-mod --spec-ngram-mod-n-match 24
--spec-ngram-mod-n-min 48
--spec-ngram-mod-n-max 64
--reasoning-preserve


using --reasoning-budget 2048 is like set to medium the thinking, use 512 for low thinking and 8196 for high, and use nothing to xHigh.
```







### Extreme quantizations (Q1/Q2) — sampling to reduce hallucinations

The low-bit files (Q1Q/Q2) are more prone to hallucination — use this tuned
sampling with llama-server for much more stable output:

```bash
llama-server -m Qwen3.8-27B-Q2-XYZ-v2.gguf --host 0.0.0.0 --port 1234 \
  --temp 0.3 \
  --top-p 0.9 \
  --top-k 40 \
  --repeat-penalty 1.10 \
  --repeat-last-n 512 \
  --dry-multiplier 0.8 \
  --dry-base 1.75 \
  --dry-allowed-length 2
```

Try more flags and combinations with 
https://github.com/TRI-Tech-Revolution-Intelligence/LocalLLM/releases

How to configure Pi harness to work with low, medium and high thinking limits,
https://github.com/quimmedes/pi-thinking-budgets-extension


The DRY sampler (repetition-aware) is the key flag for low-bit models —
`--dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2` keeps generation
on-track without over-penalizing. Works for Q1Q-XYZ-v2 too.

## Notes

- Model is Apache-2.0, architecture `Qwen3_5ForConditionalGeneration` (hybrid
  SSM + attention, full attention every 4th layer), vocab 248,320,
  `tie_word_embeddings=false`.
- Quantized with upstream llama.cpp, 4 threads, `--tensor-type` per-tensor
  recipe with imatrix. Key tensors (attention K/V, SSM params, MTP head) are
  kept in **BF16** — full precision where it matters.
- The Q1/Q2 files are experimental: below ~2.5 BPW a reasoning model can get
  stuck in the thinking loop on some prompts.
- Sizes are exact file sizes on disk (GiB, base 1024 — matches the sizes shown
  on the HF file browser).

## Files

Quant ladder (v2 builds) + the new Q2.5-v3. Sizes are exact on-disk GiB (base 1024).

| File | Size |
|---|---|
| `CelestialDragon.png` | 2.96 MB |
| `Qwen3.8-27B-Q1Q-XYZ-v2.gguf` | 6.15 GiB |
| `Qwen3.8-27B-Q1Z-XYZ-v2.gguf` | 7.12 GiB |
| `Qwen3.8-27B-Q2-XYZ-v2.gguf` | 8.15 GiB |
| `Qwen3.8-27B-Q2.5-XYZ-v2.gguf` | 8.60 GiB |
| `Qwen3.8-27B-Q2.5-v3-XYZ.gguf` | 9.06 GiB |
| `Qwen3.8-27B-Q3-XYZ-v2.gguf` | 10.39 GiB |
| `Qwen3.8-27B-Q3.5-XYZ-v2.gguf` | 11.95 GiB |
| `Qwen3.8-27B-Q4-XYZ-v2.gguf` | 14.03 GiB |
| `Qwen3.8-27B-Q4.5-XYZ-v2.gguf` | 14.62 GiB |
| `Qwen3.8-27B-Q5-XYZ-v2.gguf` | 16.61 GiB |
| `Qwen3.8-27B-Q5.5-XYZ-v2.gguf` | 19.50 GiB |
| `Qwen3.8-27B-Q6-XYZ-v2.gguf` | 20.94 GiB |
| `Qwen3.8-27B-Q7-XYZ-v2.gguf` | 24.18 GiB |
| `Qwen3.8-27B-Q8-XYZ-v2.gguf` | 26.71 GiB |
| `Qwen3.8-27B-Q9-XYZ.gguf` | 29.27 GiB |
| `Qwen3.8-27B-ULTRA-XYZ-v2.gguf` | 15.76 GiB |
| `README.md` | 6.9 kB |
| `mmproj-Qwen3.8-27B-bf16.gguf` | 0.87 GiB |

