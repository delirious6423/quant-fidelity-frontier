---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - speculative-decoding
  - dspark
  - draft-model
  - llama.cpp
  - gguf
language:
  - en
  - zh
library_name: gguf
---

# Qwen3.8-27B-DSpark

> ### ⚡ Day-1 support for Qwen3.8-27B
>
> Trained, benchmarked and released within a day of the base model going public --
> speculative decoding you can turn on immediately, not months later.

A DSpark speculative-decoding drafter head for **Qwen3.8-27B**, trained against the
**Q4_K_M GGUF that actually gets served** rather than the bf16 checkpoint.
On a single RTX 4090D it delivers **1.69x-2.51x decode speedup** depending on the
workload.

## Measured decode speed (llama.cpp, 1x RTX 4090D 48G, Q4_K_M target, temperature 0)

All throughput numbers below are **decode tokens/s** (llama.cpp `predicted_per_second`),
measured on generation only -- prompt processing is excluded. Batch size 1, 256 new tokens.

Baseline is the same target with no draft: **48.86 decode tok/s** (stable within +/-0.06
across all nine sets, so the differences below are the workload, not measurement noise).

| dataset | decode tok/s | accepted length | acceptance | speedup |
|---|---|---|---|---|
| math500 | 122.77 | 4.055 | 0.755 | **2.51x** |
| gsm8k | 121.40 | 4.087 | 0.765 | **2.49x** |
| humaneval | 103.55 | 3.459 | 0.617 | 2.12x |
| mbpp | 99.24 | 3.216 | 0.555 | 2.03x |
| aime25 | 98.58 | 3.199 | 0.561 | 2.02x |
| alpaca | 90.06 | 3.102 | 0.442 | 1.84x |
| arena-hard-v2 | 85.70 | 2.767 | 0.440 | 1.75x |
| mt-bench | 83.55 | 2.773 | 0.436 | 1.71x |
| livecodebench | 82.35 | 2.727 | 0.432 | **1.69x** |

Structured reasoning (math) speculates best; open-ended writing worst. Competitive
programming (livecodebench) is notably harder than textbook code (humaneval),
which is worth knowing if your workload looks more like the former.

## Works across quantizations

The head was trained on hidden states captured from **Q4_K_M**, but it transfers to
other quantizations of the same target without retraining. Same draft, same prompts,
only the target file changed:

| target | size | baseline decode tok/s | with draft decode tok/s | speedup | mean accepted length |
|---|---|---|---|---|---|
| UD-Q4_K_XL | 17.9 GB | 46.18 | **98.84** | 2.14x | 3.230 |
| Q8_0 | 29.0 GB | 31.36 | 74.44 | **2.37x** | 3.266 |

Two things worth knowing:

**Accepted length barely moves** (3.230 vs 3.266, well under 1%) even across a 2x
precision change. The drafter conditions on the target residual stream at five layers,
and those intermediate representations stay consistent across quantizations, so a head
trained on one quant works on the others.

**Higher precision gets a larger relative speedup** (2.14x -> 2.37x) because the
non-speculative baseline slows down with weight size (46.18 -> 31.36 t/s) while
speculation amortizes weight reads across several verified tokens, trading a
memory-bandwidth bottleneck for a compute one. Absolute throughput is still higher on
the 4-bit file (98.84 vs 74.44 t/s), so pick UD-Q4_K_XL for speed and Q8_0 for quality.

Per-dataset speedup range: UD-Q4_K_XL 1.80x-2.64x, Q8_0 2.00x-2.99x.

## Usage

### Requirements

Upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) -- pull the latest
master and build. No fork needed.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server
```

### Running

```bash
llama-server \
  -m Qwen3.8-27B-Q4_K_M.gguf \
  -md Qwen3.8-27B-DSpark-Q8_0.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 4 \
  -ngl 99 -ngld 99 -c 8192
```

`-ngl 99` offloads the target, `-ngld 99` the draft -- **both are needed**; leaving the
draft on CPU erases the speedup.

**`--spec-draft-n-max` is the one flag you must set.** The default of 3 is close,
but the optimum depends on the head, not on the target:

| n_max | decode tok/s (median) | accepted length |
|---|---|---|
| 2 | 91.36 | 2.488 |
| 3 | 99.35 | 2.937 |
| **4** | **101.25** | **3.272** |
| 5 | 96.88 | 3.531 |
| 6 | 91.67 | 3.807 |
| 8 | 80.25 | 3.740 |

Accepted length keeps rising past n_max=4, but the per-step drafting cost rises
faster, so throughput peaks at 4. **If you swap in a different head, re-sweep** --
a head with longer accepted length will peak at a higher n_max.

Use the **Q8_0** file: same accepted length as bf16, about 7% faster, half the size.

## Files

| file | size | use |
|---|---|---|
| `Qwen3.8-27B-DSpark-Q8_0.gguf` | 2.0 GB | llama.cpp (recommended) |
| `Qwen3.8-27B-DSpark-bf16.gguf` | 3.7 GB | llama.cpp, full precision |
| `model.safetensors` | 3.7 GB | training / vLLM; 62 tensors, 1.86B params |
| `config.json` | | geometry + `target_layer_ids` + `mask_token_id` |

`model.safetensors` deliberately **excludes** `embed_tokens` and `lm_head`:
those are frozen copies of the target weights (2.54B params, 58% of the raw
checkpoint). `convert_hf_to_gguf.py --target-model-dir <target>` pulls them from
the target, so nothing is lost and the target weights are not redistributed.

## Training

| | |
|---|---|
| data | 40,000 prompts from `mlabonne/open-perfectblend`, answers regenerated by the served **Q4_K_M** target itself (thinking traces kept inline) |
| hidden states | captured from the **Q4_K_M GGUF**, not from bf16 -- the head is matched to the deployed artifact |
| warm start | our own [Qwen3.6-27B-Dspark-v1](https://huggingface.co/DimInfer/Qwen3.6-27B-Dspark-v1) head (same architecture and tokenizer), which already carries trained markov/confidence heads |
| geometry | 5 layers, hidden 5120, head_dim 128, 32 heads / 8 kv, block_size 15, markov_rank 256 |
| target layers | `[1, 16, 31, 46, 61]` |
| schedule | 936 steps / 3 epochs, lr 5e-4, anchors 128, bf16, DDP on 4x RTX 4090D |

Warm-starting from the 3.6 head rather than a DFlash head works because Qwen3.6 and
Qwen3.8 share the same config (except `transformers_version`) and the same
`vocab.json`/`merges.txt`, so the weights transfer directly.

## Future work

This head was trained on 40,000 samples, which is where the current results come from
rather than any ceiling of the method. Data volume is the lever we have seen move the
number most: on Qwen3.6-27B, going from ~11K to 35K target-regenerated samples raised
mean accepted length from 2.27 to 3.46 on the same evaluation set. We plan to:

- **Train on substantially more data.** Accepted length was still improving with scale
  at 40K, so the current numbers are a floor, not a limit.
- **Release the training pipeline.** The end-to-end path -- answer regeneration from the
  served quantized target, hidden-state capture from the GGUF, sharded multi-GPU capture,
  and the consumer-GPU patches needed to train these heads on 4090-class cards -- will be
  published so others can build heads for their own targets and quantizations.
- **Cover more targets and quantizations.** Same recipe, different base models.

Follow this repo for updates.
