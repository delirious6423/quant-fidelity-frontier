---
language:
- en
tags:
- mlx
- turboquant
- quantization
- mixture-of-experts
- apple-silicon
pipeline_tag: text-generation
library_name: mlx
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
---

# Qwen3.8-Flash-Next — TurboQuant `tq4a-tq2e-g64`

> Runs with a plain `pip install "turboquant-mlx-full>=0.27.0"`. Add **`--ngram-offload`**
> and the model needs 34 GiB of GPU memory instead of 52, with identical output. See
> [Install and run](#install-and-run).

A **52.00 GiB** compression of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(180B parameters, `qwen4_exp`: 512 routed experts at top-10, 640-wide, hybrid
Gated DeltaNet + Qwen Sparse Attention, 262K context), **plus the 0.84 GiB vision
tower in bf16** — it reads images as well as text.

**Why it exists:** a data-free build (no calibration set) that runs **fully in
memory on a 64 GB Mac at ~16-18 tokens/sec** with a plain `pip install`, reads
images, and, with `--ngram-offload`, keeps its 17.9 GiB n-gram table on disk so only
**34 GiB** has to sit in GPU memory. The 2-bit oQ2 (63.0 GiB) cannot load resident on
64 GB.

**If speed is what you want, use the other build that fits.** ddalcu's
imatrix-calibrated [iQ-MLX 3.3 bpw](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw)
(50.6 GiB, for `mlx-serve`) decodes **2.3-2.4x faster** on the same M4 Max (39.6-41.5
vs 17.2-17.6 tok/s warm, both without MTP, thinking off) and prefills a cold 16K prompt
1.8x faster. This build is the calibration-free one, with a vision tower, and it is
the one that runs in 34 GiB.

**Documentation: [manjunathshiva.github.io/turboquant-mlx](https://manjunathshiva.github.io/turboquant-mlx/)**: the [Qwen3.8-Flash-Next page](https://manjunathshiva.github.io/turboquant-mlx/model-pages/qwen38-flash-next/), [getting started](https://manjunathshiva.github.io/turboquant-mlx/getting-started/), [choosing a model for your Mac's memory](https://manjunathshiva.github.io/turboquant-mlx/models/), guides for [serving an OpenAI-compatible API](https://manjunathshiva.github.io/turboquant-mlx/guides/serve/), [local coding agents](https://manjunathshiva.github.io/turboquant-mlx/guides/coding-agents/), [vision models](https://manjunathshiva.github.io/turboquant-mlx/guides/vision/) and [memory and macOS settings](https://manjunathshiva.github.io/turboquant-mlx/guides/memory/).

## Measured

M4 Max, 64 GB, `sudo sysctl -w iogpu.wired_limit_mb=60416`, `turboquant-mlx-full`
0.26.0, current weights:

| | |
|---|---|
| Size on disk | **52.00 GiB** text model + 0.84 GiB vision tower |
| Resident peak, short prompt, text | **56.30 GB** |
| Resident peak, with an image | **58.1 GB** |
| Load time, text + vision tower | ~11 s (internal SSD) |
| Decode, thinking disabled | **16.2-17.9 tok/s** |
| With `--ngram-offload`: GPU memory after load | **34.13 GiB** instead of 52.01 (the n-gram table is read from disk), bit-identical output |
| With `--ngram-offload`: serving, 16K prompt | 0 swapouts inside requests (621K-706K without it), wired peak 47.1 GB instead of 63.4, first token 70.1 s cold instead of 62.8 |
| Decode, thinking enabled | **14.7-15.4 tok/s** |

## Will it fit my Mac?

`turboquant-plan --ngram-offload` from `turboquant-mlx-full` 0.27.0, for a 64 GB Mac
with the Metal cap raised to 60416 MiB (63.35 GB):

```
TurboQuant plan — manjunathshiva/Qwen3.8-Flash-Next-tq4a-tq2e-g64

Model
  type                 qwen4_exp  (turboquant 4-bit g64)
  MoE                  512 experts, top-10, 48 layers
  weights (exact)      55.84 GB
    experts            33.97 GB  (streamable)
    resident           21.86 GB  (attention, embeddings, routers)
    n-gram table       19.20 GB  (movable off the GPU with --ngram-offload)

Machine  (assumed, not this one)
  Metal working set    63.35 GB   ← the real ceiling
  system RAM           68.72 GB

Projection at 32,768 tokens of context
  weights              36.64 GB  (+ 19.20 GB n-gram table in the page cache, not wired)
  KV cache             0.81 GB  (24.0 KB/token, hybrid: 12/48 full-attention)
  prefill workspace    3.75 GB  (estimate, at --prefill-step-size 2048)
  runtime reserve      1.00 GB  (buffer cache, activations, fragmentation)
                       ----------------------------------
  peak                 42.19 GB of 63.35 GB usable   21.16 GB headroom

Verdict: ✅ RESIDENT — fits fully in memory
```

**With offload, a 64 GB Mac runs it without the `sysctl`, but long prompts are faster
with it.** Measured on the M4 Max at its default cap (Metal working set 55.7 GB),
`turboquant-serve --ngram-offload --prefill-step-size 512`, thinking off, 0 swapouts
inside requests throughout:

| | default cap | cap raised to 60416 MiB |
|---|---|---|
| decode, short / code prompt | 17.3 / 16.7 tok/s | 17.2 / 16.8 tok/s |
| 16K prompt, first token (cold) | 150.3 s | 70.1 s |
| 16K prompt, repeat | 0.35 s | 0.27 s |
| decode after a 16K prompt | 9.4-11.0 tok/s | 17.2 tok/s |

So for chat, skip the `sysctl`; for long documents, run it. Without `--ngram-offload`
the model needs the raised cap regardless: it projects 58.67 GB at 16K (86% of RAM),
and serving it that way swapped in every session measured.

**48 GB Macs (projected; not run on a 48 GB Mac):**

```
TurboQuant plan — manjunathshiva/Qwen3.8-Flash-Next-tq4a-tq2e-g64

Model
  type                 qwen4_exp  (turboquant 4-bit g64)
  MoE                  512 experts, top-10, 48 layers
  weights (exact)      55.84 GB
    experts            33.97 GB  (streamable)
    resident           21.86 GB  (attention, embeddings, routers)
    n-gram table       19.20 GB  (movable off the GPU with --ngram-offload)

Machine  (assumed, not this one)
  Metal working set    38.65 GB   ← the real ceiling
  system RAM           51.54 GB

Projection at 16,384 tokens of context
  weights              36.64 GB  (+ 19.20 GB n-gram table in the page cache, not wired)
  KV cache             0.40 GB  (24.0 KB/token, hybrid: 12/48 full-attention)
  prefill workspace    2.13 GB  (estimate, at --prefill-step-size 2048)
  runtime reserve      1.00 GB  (buffer cache, activations, fragmentation)
                       ----------------------------------
  peak                 40.18 GB of 46.39 GB usable   6.21 GB headroom

Verdict: ⚠️  RESIDENT — fits, but only after raising the Metal wired cap

Recommended:
  sudo sysctl -w iogpu.wired_limit_mb=40747   (raises the 38.65 GB Metal cap — the binding limit here, not your 52 GB of RAM; resets on reboot)
```

To test that case, the same serve benchmark was run on the 64 GB Mac with the GPU cap
lowered to the 48 GB class. At that cap, short prompts decode at 17.3 tok/s and code
at 16.8, with 0 swapouts inside requests. A 16K prompt answers in 62.3 s cold and
0.27 s on a repeat at a cap of 41,603 MiB. At a tighter cap (39,648 MiB) the server
dropped its prompt cache and every repeat re-read the prompt (61-235 s). The 40,747
MiB above is what `plan` recommends for the text model; the tested 41,603 also counted
the 0.84 GiB vision tower. **What this doesn't cover:** that Mac still had 64 GB of
RAM to keep the 17.9 GiB table in the page cache; a real 48 GB Mac has 16 GB less, so
expect more disk reads on long prompts.

### Measured, not projected

Without `--ngram-offload`, 0.26.0. A fact buried at 70% depth in filler text,
`--prefill-step-size 512`, greedy, thinking disabled:

| prompt | prefill | decode | peak | fact retrieved |
|---|---|---|---|---|
| 4,110 tok | 149.6 tok/s | 18.1 tok/s | 58.28 GB | yes |
| 15,700 tok | 255.2 tok/s | 17.8 tok/s | 59.43 GB | yes |
| **29,782 tok** | 185.4 tok/s | **14.2 tok/s** | **60.93 GB** | yes |

**The planner is optimistic, more so at long context.** At 16K it projects
58.67 GB; 15,700 tokens measured 59.43 GB, 0.8 GB more. At 32K it projects
59.48 GB, but 29,782 tokens already peak at 60.93 GB, 1.5 GB more, leaving
**2.42 GB of real headroom rather than 3.87**. It still fits. Plan for ~2 GB of
headroom at 30K. (Measured text-only; the projection includes the 0.90 GB vision
tower, so with images loaded expect less.)

Decode is flat from 4K to 16K (18.1 -> 17.8 tok/s) and then falls **21% by 30K**.
Long-range retrieval survived at every length tested.

## What was quantized, and how

Data-free: randomized Hadamard rotation into Lloyd–Max codebooks, no calibration set.

| tier | precision |
|---|---|
| Attention (GDN + QSA), `lm_head` | 4-bit codebook, group 64 |
| Routed experts (512 × 3 × 48) | **2-bit codebook, group 64** |
| n-gram/PLE table + token embedding | **2-bit affine, group 32** |
| MoE routers, QSA block indexer, hyper-connection gating | **full precision** |
| **Vision tower (0.45B)** | **bf16, unquantized** — 0.84 GiB against ~4 GB of headroom; quantizing it saves ~0.6 GiB and buys nothing |
| Multi-token-prediction head | dropped |

Two choices worth calling out:

- **The n-gram/PLE table is 51.2B parameters — 28% of this model.** It is an
  embedding, which codebook quantizers skip by default; left alone it would put
  this build near **124 GiB** instead of 52.
- **Routers and the QSA block indexer stay exact.** Both make a *discrete*
  choice — which expert, which KV block — so quantization error there changes
  which computation runs rather than degrading it smoothly. Together they cost
  well under a gigabyte.

## Quality

An 8-task battery (two coding tasks, arithmetic, a syllogism, delayed literal
recall, a short explanation, strict JSON, list formatting), resident, temp 0.7,
every run seeded. This battery was measured on the weights before the
2026-09-14 rebuild; the current weights are re-validated in the next section.

| mode | result |
|---|---|
| **Thinking disabled** | **8/8 tasks pass, 8/8 terminate cleanly, 28-544 tokens** |
| Thinking enabled | content correct on 22/24 runs; **16/24 close `</think>` within 5,000 tokens** |
| Delayed literal recall | **4/4 exact** (invoice no., IP, weight, code, after intervening text) |

**Thinking does terminate — it is the cost that makes it impractical.** Re-running
the tasks that failed at a 5,000-token budget with 12,000 tokens instead:

| task | at 5,000 | at 12,000 |
|---|---|---|
| harder coding task | 0/3 closed | **2/2 closed**, at 9,437 and 11,690 tokens |
| "explain in exactly three sentences" | 1/3 | **2/2 closed**, ~4,400 tokens each |
| one-word syllogism | 1/3 | 1/2 — the other exceeded 12,000 |

So most non-terminations were the token budget, not a runaway. What remains is
spend: **a syllogism whose answer is one word consumed over 12,000 thinking
tokens** — about 14 minutes at this model's decode rate — while thinking
disabled answered all eight tasks correctly in 28-544 tokens.

### Re-validated on the current weights (2026-09-14 rebuild)

Same seeds, same settings, before and after the gating matrices were restored
to bf16 (see Changelog):

| check | before rebuild | **current weights** |
|---|---|---|
| WikiText-2 perplexity (32 x 512-token chunks) | 9.92 | **9.31** (6.2% lower; lower on 31 of 32 chunks) |
| Thinking disabled, 5 seeded runs | 5/5 clean | **5/5 clean** |
| Thinking enabled, closes `</think>` within 5,000 tokens, 15 seeds | 13/15 | **7/15** |

The rebuild is the better language model by a clear margin, but with thinking
enabled it more often keeps deliberating up to the token budget. Thinking
disabled is unaffected.

**Recommendation: run with thinking disabled.** If you do enable it, budget at
least 12,000 tokens, expect minutes per answer, and resample runs that hit the
budget.

```python
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False)
```

## Vision

The vision tower ships in `vision.safetensors` (bf16, 333 tensors) with its
preprocessor configs. It is a **Qwen3-VL tower**: `mlx_vlm.models.qwen3_vl` loads
these weights `strict=True`, 333/333. `vision_config`, `image_token_id` (248056)
and the vision start/end ids are in `config.json`.

Measured on this quantized build through the package's vision API (the code under
[Install and run](#install-and-run)), greedy, thinking disabled:

| task | answer | mlx-vlm 0.6.14 | mlx-vlm 0.7.0 |
|---|---|---|---|
| read the text in an image | `VOLTAGE 47` | pass | pass |
| which side is the red square on | `left` | pass | pass |
| count the circles | `5` | pass | pass |

**1.3-4.1 s per answer once warm** with mlx-vlm 0.7.0, 58.1 GB peak. The first
image in a session pays a one-time Metal kernel compile, 25-100 s. These are
three small synthetic images, one sample each: enough to show the path works,
not a vision benchmark.

Interleaved MRoPE (`mrope_section [11, 11, 10]`) is applied to image token
positions. With plain 1-D positions instead, the same three tasks also passed,
so on small images the difference is not measurable.

## Install and run

Needs **`turboquant-mlx-full` >= 0.26.0** for `qwen4_exp` support, and **>= 0.27.0 for
`--ngram-offload`**. mlx-lm PR #1788 is still open, so stock mlx-lm cannot load this
model. The generate and server commands below were re-run from a clean `pip install`
of 0.27.0 on 2026-09-17, and the vision example from a clean install of 0.28.0 on
2026-09-19; the streaming command was last run on 0.26.0.

```bash
pip install "turboquant-mlx-full>=0.27.0"
sudo sysctl -w iogpu.wired_limit_mb=60416   # needed without --ngram-offload; with it, only speeds up long prompts. Resets on reboot
```

**Resident, thinking disabled (the recommended mode).** `--ngram-offload` keeps the
n-gram table on disk: 34 GiB of GPU memory instead of 52, the same output, and on a
64 GB Mac no swapping. Keep the model folder in place while it's loaded, since the
table is read from those files.

```bash
turboquant-generate --model manjunathshiva/Qwen3.8-Flash-Next-tq4a-tq2e-g64 \
    --ngram-offload --no-think \
    --prompt "Explain sparse attention in two sentences." --max-tokens 300
```

**OpenAI-compatible server:**

```bash
turboquant-serve --model manjunathshiva/Qwen3.8-Flash-Next-tq4a-tq2e-g64 \
    --ngram-offload --prefill-step-size 512 --chat-template-args '{"enable_thinking": false}'

curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "manjunathshiva/Qwen3.8-Flash-Next-tq4a-tq2e-g64",
       "messages": [{"role": "user", "content": "What is the capital of France?"}],
       "max_tokens": 200}'
```

**Less memory: stream the experts from disk.** No `sysctl` needed; 34.4 GB peak
with a 12 GB expert cache, ~7.6 tok/s once warm (internal SSD), about half the
resident speed:

```bash
python -m turboquant_mlx.stream.stream_generate \
    --model manjunathshiva/Qwen3.8-Flash-Next-tq4a-tq2e-g64 \
    --cache-budget-gb 12 --max-active-experts 0 --no-think \
    --prompt "What is 17 * 23?" --max-tokens 100
```

**Vision** has no command-line flag yet; use the Python API. It needs the `[vlm]`
extra, and works with mlx-vlm 0.6.14, 0.7.0 and 0.7.1. With `ngram_offload=True` the three
test images (text, left/right, counting) get token-for-token the same answers as
without it, at a 36.2 GiB peak instead of 54.1:

```bash
pip install "turboquant-mlx-full[vlm]>=0.27.0"
```

```python
import json, os
import mlx.core as mx
from huggingface_hub import snapshot_download
from PIL import Image
from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor
from turboquant_mlx.generate import load_turboquant
from turboquant_mlx.models.qwen4_exp_vision import encode_image, load_vision_tower, prepare_inputs

path = snapshot_download("manjunathshiva/Qwen3.8-Flash-Next-tq4a-tq2e-g64")
model, tok = load_turboquant(path, ngram_offload=True)  # 34 GiB instead of 52; same answers
config = json.load(open(os.path.join(path, "config.json")))
tower = load_vision_tower(config["vision_config"], os.path.join(path, "vision.safetensors"))
processor = Qwen3VLImageProcessor.from_pretrained(path)

features, grid_thw = encode_image(tower, processor, Image.open("photo.png").convert("RGB"))
text = tok.apply_chat_template(
    [{"role": "user", "content": "<|vision_start|>" + "<|image_pad|>" * features.shape[0]
      + "<|vision_end|>" + "What is in this image?"}],
    add_generation_prompt=True, tokenize=False, enable_thinking=False)
ids = mx.array(tok.encode(text))[None]
emb, rope_cs = prepare_inputs(model, ids, features, grid_thw.tolist(), config)

cache = model.make_cache()
logits = model(ids, cache=cache, input_embeddings=emb, rope_cs=rope_cs)
eos, out = set(tok.eos_token_ids), []
for _ in range(200):
    tid = int(mx.argmax(logits[0, -1]).item())
    if tid in eos:
        break
    out.append(tid)
    logits = model(mx.array([[tid]]), cache=cache)
print(tok.decode(out))
```

**Don't use `--kv-bits` with this model.** All 12 attention layers carry the
sparse-attention indexer, so KV quantization leaves them at full precision (it
prints a warning) and saves nothing.

## Limitations

- **Thinking mode is unreliable** (7/15 close within 5,000 tokens on the current weights). Use it disabled, or set a large token budget and resample.
- **2-bit routed experts.** Delayed literal recall passed 4/4 here, but this is the most aggressive expert tier in TurboQuant.
- **Without `--ngram-offload`, headroom is tight:** ~4 GB at 16K context and ~2.4 GB at 30K, measured, and serving swapped on a 64 GB Mac. Use the flag.
- **`--ngram-offload` reads the table from the model folder:** keep it in place while the model is loaded, and expect the first long prompt after loading to be ~12% slower while those rows are read from disk.
- **Vision is verified on three small synthetic images.** Large images, multiple images and video are untested.
- Quality evidence is batteries on one machine, not a benchmark suite.

## Changelog

- **2026-09-19 — vision with `ngram_offload=True`.** The vision example now loads
  with the n-gram table on disk: the three test images get token-for-token the same
  answers as without it, at a 36.2 GiB peak instead of 54.1. Re-run from a clean
  `pip install "turboquant-mlx-full[vlm]==0.28.0"` (mlx-vlm 0.7.1): 3/3 correct. The
  weights are unchanged.

- **2026-09-17 — `turboquant-mlx-full` 0.27.0: `--ngram-offload`.** The 17.88 GiB
  n-gram table can now stay on disk, memory-mapped, so the model needs 34.13 GiB of GPU
  memory instead of 52.01. The output is unchanged: logits are bit-identical over 1,024
  positions. The weights are the same files, so there's nothing to re-download. This
  card's generate and server commands use the flag, and "Will it fit my Mac?" is
  regenerated with it. Also added: the same-Mac speed comparison with the iQ-MLX 3.3
  bpw build, replacing "not compared yet".

- **2026-09-14 — corrected "no other MLX build fits a 64 GB Mac".** That stopped
  being true on 2026-09-13, when a 50.6 GiB imatrix-calibrated build for `mlx-serve`
  was published.
- **2026-09-14 — `turboquant-mlx-full` 0.26.0 released.** This build now loads with
  a plain `pip install`. Every command on this card was re-run from that install
  (resident generate, server, expert streaming, vision on mlx-vlm 0.6.14 and 0.7.0),
  and the work-in-progress label is removed.
- **2026-09-14, 05:07 UTC — weights rebuilt.** The quantizer's extras pass was
  also quantizing 96 hyper-connection gating matrices (`block_inject_weight`,
  shape 4 x 640) to 2-bit; they are now kept in bf16, like the MoE routers and
  the QSA indexer. Everything else is unchanged. **If you downloaded before that
  time, re-pull.** Found in code review; the fix is in the same TurboQuant-MLX
  release that adds `qwen4_exp`. Re-validated the same day: perplexity improved
  (9.92 -> 9.31), thinking-disabled unchanged (5/5), thinking-enabled closes
  less often within 5,000 tokens (13/15 -> 7/15). Details under Quality.

## Credits and licence

- Base model: **Qwen** — [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), **Qwen Community License 1.0** (`LICENSE`, included).
- MLX architecture implementation vendored from **mlx-lm PR #1788** by `eauchs` (MIT).
- Runs on **[Apple MLX](https://github.com/ml-explore/mlx)**.
- Compressed with **[TurboQuant-MLX](https://github.com/manjunathshiva/turboquant-mlx)** — Copyright 2026 Manjunath Janardhan.
