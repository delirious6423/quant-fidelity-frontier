---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
tags:
  - gguf
  - qwen3_5
  - text-generation
  - conversational
  - text-only
  - endpoints_compatible
pipeline_tag: text-generation
---

# Qwen3.8-27B-OnlyText-GGUF

GGUF quants of [OnlyTextLLMs/Qwen3.8-27B-OnlyText](https://huggingface.co/OnlyTextLLMs/Qwen3.8-27B-OnlyText)
— the text-only derivative of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with the
**MTP draft head embedded in-file** (no separate draft file). The MTP tensors
are included by the converter by default (`nextn_predict_layers = 1`), so
speculative decoding is available on a recent llama.cpp via `--spec-type draft-mtp`.

## Quant files

| File | Size |
|---|---|
| `Qwen3.8-27B-OnlyText-MTP-Q4_K_M.gguf` | 16.8 GB |
| `Qwen3.8-27B-OnlyText-MTP-Q6_K.gguf` | 22.4 GB |
| `Qwen3.8-27B-OnlyText-MTP-Q8_0.gguf` | 29.0 GB |


## Benchmarks

Measured 2026-08-24 on 2× AMD Radeon AI PRO R9700 (gfx1201, 34 GB each) with
llama.cpp `f280b26983ad` (HIP/ROCm build, ROCm 7.14, `-c 32768 -fa on`). Baseline =
llama-bench PP512/1024/2048 + TG128/256; chat decode = llama-cli single-turn,
256 generated tokens, median conditions, `-np 1`.

| Quant | Device | Prefill t/s (PP512) | Gen t/s chat, no MTP | Gen t/s chat, best MTP | MTP Δ |
|---|---|---|---|---|---|
| Q4_K_M | single R9700 | 1035 | 26.7 | 42.7 (MTP n=3) | +60% |
| Q6_K | single R9700 | 660 | 23.8 | 44.0 (MTP n=2) | +85% |
| Q8_0 | single R9700 | 1222 | 19.7 | 43.6 (MTP n=4) | +121% |
| Q4_K_M | dual R9700 (layer-split) | 869 | 23.8 | 39.8 (MTP n=3) | +67% |
| Q6_K | dual R9700 (layer-split) | 545 | 22.1 | 39.6 (MTP n=2) | +79% |
| Q8_0 | dual R9700 (layer-split) | 1110 | 19.0 | 41.5 (MTP n=4) | +118% |



### Baselines cited (to beat — published results)

- 51.8 tok/s, single Radeon AI PRO R9700 (MTP=2, Vulkan, Windows) — [AMD blog — Run Qwen 3.8 27B Day 0](https://www.amd.com/en/blogs/2026/run-qwen-3-8-27b-on-amd-ryzen-ai-max-and-radeon-graphics-cards-day-0.html)
- 24.5 tok/s, Ryzen AI Max+ 395 / Strix Halo (MTP=4) — [AMD blog (same)](https://www.amd.com/en/blogs/2026/run-qwen-3-8-27b-on-amd-ryzen-ai-max-and-radeon-graphics-cards-day-0.html)
- 17.4 tok/s decode, Qwen3.6-27B Q4_K_M, R9700, no MTP — [willitrunai](https://willitrunai.com/can-run/qwen-3.6-27b-on-radeon-ai-pro-r9700-32gb)
- 12.1 → 21.2 tok/s (MTP n=3, 1.81×), Qwen3.6-27B Q4_K_M, Strix Halo ROCm — [calebcoffie — llama.cpp MTP on Strix Halo](https://calebcoffie.com/blog/benchmarking-llama-cpp-mtp-on-strix-halo)
- ~75 → >90 tok/s, Q4_K_S, RTX 5090 (LM Studio) — [DavidAU MTP GGUF model card](https://huggingface.co/DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF)

MTP reference speedup: ~1.5–2×, no accuracy loss at temperature 0
([unsloth Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF);
merge PR
[ggml-org/llama.cpp#22673](https://github.com/ggml-org/llama.cpp/pull/22673)).

## Usage

Requires a llama.cpp build from 2026-05-16 or later (MTP support, commit
`4f13cb7`+). Enable MTP with `--spec-type draft-mtp`; the best draft count is
`--spec-draft-n-max 2` on Radeon PRO R9700 and `4` on Ryzen AI Max / Strix Halo.

```bash
# llama-server (OpenAI-compatible)
llama-server -m Qwen3.8-27B-OnlyText-MTP-Q4_K_M.gguf -ngl 99 -c 32768 -fa on --spec-type draft-mtp --spec-draft-n-max 2

# llama-cli single-turn chat
llama-cli -m Qwen3.8-27B-OnlyText-MTP-Q4_K_M.gguf -ngl 99 -c 32768 -fa on \
    -st --reasoning off --spec-type draft-mtp --spec-draft-n-max 2 -p "Your prompt"
```

Notes:

- The chat template keeps Qwen's **thinking mode on by default** (`reasoning_effort: xhigh`) — the model reasons extensively before answering.
Disable it with `--reasoning off` (llama-cli) or
`"chat_template_kwargs": {"enable_thinking": false}` (server API) for fast
direct answers.
- Dual-GPU: unset `*_VISIBLE_DEVICES` env pins; llama.cpp auto layer-splits
  (`--split-mode layer` is the default).

## Details

- Base: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (apache-2.0): 27.32B params, 64 layers (48 Gated-DeltaNet linear-attention + 16 full-attention), hidden 5120, vocab 248302, `Qwen3_5ForCausalLM`, `qwen35` GGUF architecture.
- Tokenizer: 15 text-only special tokens at ids 248044–248058 (EOS 248046 = `<|im_end|>`); vision/audio tokens removed.
- Conversion: `convert_hf_to_gguf.py` (llama.cpp `f280b26983ad`) then
  `llama-quantize` with the file's named type; MTP tensors embedded by default.
- Weights are untouched derivatives — no additional training.

## Attribution

This model is a derivative of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the
[Qwen team](https://huggingface.co/Qwen), released under the
[apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
All credit for the underlying weights and capabilities belongs to the original
authors; this repository only removes modalities and quantizes, it does not add
new training.
