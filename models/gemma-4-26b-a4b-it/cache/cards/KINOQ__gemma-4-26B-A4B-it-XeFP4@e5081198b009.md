---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model:
- google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- xefp4
- 4-bit
- intel-arc
- opencl
- moe
---

# gemma-4-26B-A4B-it-XeFP4

[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) quantized to **XeFP4**, an open 4-bit
weight format for Intel Arc GPUs. It fits in 16 GB of VRAM and runs on **XeQ Runtime**, the OpenCL engine for XeFP4:
**https://github.com/kinoqdev/XeFP4**.

> These files do **not** load in transformers, llama.cpp or vLLM. They need XeQ Runtime.

## Configurations

| | **Quality** (recommended) | **Speed** |
|---|---|---|
| experts (128 × MoE) | XeFP4-8 | XeFP4-8 |
| attention and dense MLP | INT8 | XeFP4-8 |
| embedding / lm_head (tied) | INT8 | INT8 |
| size | 14.2 GB | 13.4 GB |
| decode tok/s (A770, short context, MTP off) | 72.1 | 92.1 |
| perplexity, base model (BF16 = 6.670) | 6.731 (+0.9%) | 6.981 (+4.7%) |

- Pick **Quality** unless decode speed matters most.
- Perplexity: base model (google/gemma-4-26B-A4B), 16 FineWeb documents × 512 tokens, weights quantized as in each
  configuration, computed on CPU. For comparison, GGUF IQ4_XS (13.9 GB): 6.827 (+2.4%). The engine itself
  (INT8 activations in prefill) gives 6.741 (Quality) and 6.994 (Speed) on the same text.
- Vision encoder (optional, `vision/`): INT8, 0.60 GB. Text-only use does not need it.
- MTP assistant (optional, `assistant/`): the source model's assistant model, unmodified (BF16, 0.84 GB). With it the
  engine uses MTP speculative decoding; the output is the same as without it.

## Files

| File | Size | SHA-256 (`.bin`) | Content |
|---|---|---|---|
| `quality/g4it_xe8_ne8.bin` + `.json` | 14,208,776,704 B | `147c36f84158858da7ccf2b99c391e9b0ea9de9f847c18a8f905faa345022285` | Quality configuration |
| `speed/g4it_xe8.bin` + `.json` | 13,405,650,944 B | `2fd99fd3e0bc47b2ed60561b5c373bd966f3932873899845ae4564c358509dcd` | Speed configuration |
| `vision/g4v_i8.bin` + `.json` | 598,346,816 B | `2423e7a863d6bbff4be639bd2b3967bffa92eb35be5cd9d40c79acddbee00dc6` | vision encoder, INT8 (optional) |
| `assistant/model.safetensors`, `config.json`, `generation_config.json`, `README.md` | 839,427,840 B | `006c8713b29b046e296c0f0b7c524aa9264ae07ae8b0c3888b9bc062242297ff` | MTP assistant model, unmodified copy of the source model's assistant (optional) |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | | | from the source model |
| `LICENSE`, `NOTICE` | | | |

`.json` header SHA-256: `speed/g4it_xe8.json` `f74a0b2451780aae76cfbf1ac5e769b8a96b65ae1781df99c0974d8242b5944b`,
`quality/g4it_xe8_ne8.json` `d23086a267bb2f121441f5f69b87ab16de9547c56a07ebb201c5e5f01deeb3b5`,
`vision/g4v_i8.json` `f87d8c8342fc550b338b68829d51607dfd9e18691ec270c2bc18f0c0f923b7fc`.

## Usage

```bash
git clone https://github.com/kinoqdev/XeFP4 && cd XeFP4
pip install -r requirements.txt
python -m xeq selftest <path to this repository>/quality
python -m xeq chat <path to this repository>/quality --ctx 32768
python -m xeq run <path to this repository>/speed --image photo.png -p "Describe the image." --ctx 8448
python -m xeq serve <path to this repository>/quality --port 8080 --ctx 16384
```

Pass `quality/` or `speed/` as the model folder. The tokenizer and chat template are read from the repository root,
the vision encoder from `vision/` and the MTP assistant from `assistant/`. `--no-mtp` turns MTP off.

### Recommended setup on a 16 GB GPU

| Use | Setup |
|---|---|
| default (chat, code, thinking, long documents) | Quality + MTP, `--ctx 32768` (FP16 KV cache, the default) |
| the GPU also drives a display, or other programs use VRAM | same, `--ctx 16384` (32K uses 15.53 of 15.56 GiB on an A770) |
| image input | Speed with `--image` |

## Quantization

- XeFP4-8: E2M1 values, block 32, 3-bit block factor, FP16 row scale. INT8 uses a per-row scale.
- Router, norms and layer scalars are FP32.
- The encoder is deterministic and uses no calibration data.
- The format is specified in [XEFP4_SPEC.md](https://github.com/kinoqdev/XeFP4/blob/main/spec/XEFP4_SPEC.md).

## Performance

Intel Arc A770 16 GB (driver 32.0.101.9150), batch 1, greedy decoding, FP16 KV cache, scripted benchmark (2026-11-17).

Decode, tok/s, by context position (MTP off):

| position | Quality | Speed | llama.cpp b11053 SYCL, IQ4_XS (13.9 GB) |
|---|---|---|---|
| 0 | 72.1 | 92.1 | 32.4 |
| 1K | 69.9 | 88.6 | |
| 4K | 66.9 | 83.8 | 25.3 |
| 10K | 65.3 | 81.3 | |
| 16K | 62.2 | 76.6 | 19.3 |
| 32K | | 68.8 | |

Prefill, tok/s (time to first token):

| prompt | Quality | Speed | llama.cpp IQ4_XS pp512 at that depth |
|---|---|---|---|
| 512 | 1,648 (0.31 s) | 1,662 (0.31 s) | 694 |
| 4K | 1,730 (2.4 s) | 1,737 (2.4 s) | 684 |
| 16K | 1,526 (10.7 s) | 1,532 (10.7 s) | 630 |
| 32K | | 1,326 (24.1 s) | |

MTP (assistant model), decode tok/s, MTP off → on, answers end at the stop token. The gain depends on the task:

| task | Quality | Speed |
|---|---|---|
| thinking | 73 → about 150 | 94 → about 140 |
| code | 73 → about 118 | 94 → about 107 |
| prose explanation | 73 → about 100 | 94 → about 92 |
| long-document summary, 30K context | 57 → 60–64 | 64 → 65 |

- The output tokens with MTP are the same as without it. When the assistant guesses poorly for a request, the engine
  switches to plain decoding by itself, so prose with Speed costs at most a few percent.
- Quality + MTP at 32K context uses 15.53 GiB; Speed at 32K uses 14.29 GiB (FP16 KV cache).
- `--kv fp32` selects an FP32 KV cache: about 2% faster at position 0, 2–9% slower from 1K on, 0.6–0.9 GiB more memory.

**Arc 140T** (Xe-LPG+ Core Ultra iGPU with shared memory, driver 32.0.101.9033), same benchmark:

| position | Quality | Speed |
|---|---|---|
| 0 | 18.6 | 23.6 |
| 1K | 16.9 | 20.8 |
| 4K | 16.3 | 19.9 |
| 16K | 14.6 | 17.4 |
| 32K | | 15.2 |

MTP (assistant model) on the 140T, decode tok/s, MTP off → on (`bench --mtp`, 4K context, answers end at the stop token):

| task | Quality | Speed |
|---|---|---|
| thinking | 18.5 → **39.4** | 23.5 → **38.1** |
| code | 18.6 → **30.0** | 23.5 → **29.1** |
| prose explanation | 18.3 → **23.9** | 23.0 → 24.6 |
| long-document summary, Quality, real 16K / 30K document | 14.8 → 15.4–17.7 / 13.3 → 14.4–15.4 | |

- On the 140T, Quality + MTP is faster than Speed + MTP for thinking and code and slightly slower for prose (23.9 vs 24.6), and it is closer to the original model. The output tokens with MTP are the same as without it.
- Prefill: about 380 tok/s with the 11-17 engine (Speed 32K: 98 s to the first token); see the 2026-11-20 update below.
- Quality + MTP with the 30K document uses 15.51 GiB (16K: 15.05 GiB). Shared-memory GPUs also need enough free system RAM.
- Output tokens match the A770.

**XeQ Runtime update (2026-11-20).** Decode speed and the selftest tokens are unchanged.
- Prefill, A770, Speed: 1,744 / 1,828 / 1,604 / 1,386 tok/s at 512 / 4K / 16K / 32K tokens (the 11-17 engine in the same run: 1,662 / 1,730 / 1,532 / 1,329).
- Prefill, Arc 140T, Quality, 4K tokens: about 460 tok/s with the default 512-token chunk (the 11-17 engine: about 375), about 640 tok/s with a 2048-token chunk.
  The 2048-token chunk is chosen automatically on GPUs that report 20 GiB or more; on a 32 GB machine set the shared GPU memory to 75%. Quality + MTP at 32K then uses 16.3 GiB (15.3 GiB with `--prefill-chunk small`).
- Chat cache: a follow-up turn reuses the previous prompt. In a 5K-token conversation the next turn is prefilled in 0.6 s on the A770 (2.3 s for the full prompt) and in 1.0–1.6 s on the 140T (9–12 s); the output tokens are the same as with the full prompt.

## Chat

- The chat template is the source model's `chat_template.jinja`, rendered as is. Its token ids match the reference implementation.
- Thinking mode can be turned on or off.
- The engine includes an OpenAI-compatible server (`/v1/chat/completions`, streaming, `enable_thinking`). It handles one request at a time. Decoding is greedy by default; sampling parameters in the request are honored.

## Sampling

Decoding is greedy by default (same output every time; best for code, facts, translation and tool calls).
For open-ended chat, writing and thinking mode, turn on sampling with `--temperature`, `--top-p`, `--top-k`, `--min-p`,
`--presence-penalty`, `--repetition-penalty` and `--seed` (the server reads the same fields from the request).
The source model recommends temperature 1.0, top_p 0.95, top_k 64. `top_k` is capped at 64.
Sampling costs about 4% of decode speed. MTP also works when sampling (same output distribution), with a smaller gain.

## Limitations

- Tested on Intel Arc A770 (Xe-HPG) and Arc 140T (Xe-LPG+). Other Intel GPUs are untested.
- Batch size 1.
- Tested context length: up to 32K.

## License and attribution

- Derived from Gemma 4 26B-A4B-it by Google DeepMind, licensed under Apache-2.0
  (https://ai.google.dev/gemma/docs/gemma_4_license).
- The weights were modified by quantization to the XeFP4 format.
- "Gemma" is used only to identify the source model.
- See `LICENSE` and `NOTICE`.
- The `assistant/` folder is an unmodified copy of the source model's MTP assistant (Apache-2.0).
- XeFP4 and XeQ Runtime are not affiliated with or endorsed by Google or Intel.
