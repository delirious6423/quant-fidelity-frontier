---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language: en
library_name: transformers
tags:
  - audio
  - speech
  - ultravox
  - adapter
  - gguf
datasets:
  - openslr/librispeech_asr
  - mozilla-foundation/common_voice_17_0
---

# qwen3.8-27b-ultravox-0.6

An [Ultravox](https://github.com/fixie-ai/ultravox) audio projector trained against `Qwen/Qwen3.8-27B` (bf16) with a frozen `openai/whisper-large-v3-turbo` encoder. Only the projector (52,441,088 parameters) is trained and released; the LLM and Whisper are not included. Code, patches and evaluation scripts: https://github.com/ianblenke/qwen3.8-27b-ultravox-0.6.

## Files
- `projector.safetensors` (105 MB, bf16): the four projector tensors (`multi_modal_projector.*`) from training step 37,500. Load them into an `UltravoxModel` built from `config.json` (see the repo's `scripts/bf16_infer.py`; `from_pretrained` does not work under transformers 5, build the model directly and load these tensors).
- `config.json`: Ultravox config (`text_model_id` is `Qwen/Qwen3.8-27B`).
- `mmproj-ultravox-whisper-large-v3-turbo-f16.gguf` (1.4 GB): the audio projector plus the Whisper encoder converted for llama.cpp (projector type `ultravox`; requires a llama.cpp build with Ultravox audio support). It contains no vision tower and no LLM weights. To serve audio and vision together, merge it with a vision mmproj using the repo's `scripts/merge_mmproj.py`.
- `SHA256SUMS`, `loss.csv`, `wer.md`.

## Use with llama.cpp: apply the trim patch
llama.cpp feeds the Whisper encoder's full 30-second window, about 188 audio positions even for a short clip, while this projector was trained on embeddings trimmed to the real clip length (17 positions for 2.6 s). Without trimming, the model repeats the sentence instead of stopping. The repo's `patches/llama.cpp/0001-ultravox-trim-audio-padding.patch` trims them (it assumes one audio item per request).

## Results
Word error rate of a "repeat the following text" request, greedy decoding, first 100 validation clips per LibriSpeech subset (clean 2,038 words; other 1,820 words), raw replies scored:

| path | clean | other |
|---|---|---|
| bf16 Qwen3.8-27B, transformers | 3.68% | 8.24% |
| ternary Qwen3.8-27B build, llama.cpp with the trim patch | 3.97% | 8.35% |

One sample per cell; differences of a few tenths of a point are noise.

## Latency

Measured 2026-11-06 on one NVIDIA GB10 (DGX Spark, unified memory) with nothing else running: the patched llama.cpp server built with CUDA, the ternary Qwen3.8-27B build and the step-37,500 audio projector, all layers on the GPU, 4,096-token context, the "repeat the following text" prompt, greedy decoding. Every request used a different LibriSpeech clip, so the server's prompt cache never held the audio. Medians of 6 clips per length, in milliseconds after the request arrives:

| Clip length | First text token | Server encode + prompt | Full reply (tokens) |
|---|---|---|---|
| ~2 s | 379 | 343 | 523 (2.5) |
| 3 s | 488 | 347 | 826 (9.5) |
| 6 s | 525 | 357 | 1,124 (17) |
| 10 s | 553 | 374 | 1,592 (30.5) |

The Whisper encoder always runs on a fixed 30-second window, so encode + prompt time barely grows with clip length. Generation runs at about 35 ms per token. Not measured: end-of-speech detection (the pause before a non-streaming model can start), text-to-speech, other GPUs, and any server load. Part of the first-token time is HTTP overhead and CPU audio preprocessing. Reproduce with `scripts/latency.py` in the GitHub repo. A first version of the benchmark that reused each clip's audio hit the prompt cache and understated latency; the numbers above are from the corrected run.

## Training
2 x DGX Spark, data parallel, global batch 8, 37,500 steps (~300k samples) of LibriSpeech clean + other and Common Voice 17 English, lr 2e-3 cosine, 500 warmup steps; paused once at step 24,000 and resumed with optimizer state. Mean loss per window fell from 0.43 to 0.10.

## Limits
English read speech only. Not evaluated on noisy, accented, conversational or overlapping speech, on clips over 16 seconds (the longest used in training), or with multiple audio items. One early bf16 test clip got a refusal reply that has not been investigated. No safety evaluation. Tested against `Qwen/Qwen3.8-27B` and a ternary build of it; fine-tunes of the base are untested.

## License and attribution
Apache-2.0 (base model Apache-2.0; Whisper MIT; Ultravox MIT). Training data: LibriSpeech (CC BY 4.0, Panayotov et al., 2015) and Common Voice English (CC0), via the fixie-ai mirrors.
