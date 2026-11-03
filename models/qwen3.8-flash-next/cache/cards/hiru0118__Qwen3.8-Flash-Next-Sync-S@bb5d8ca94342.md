---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- qwen3.8-flash-next
- moe
- trellis
- strata
- experimental
---

# Qwen3.8-Flash-Next Sync-S (experimental)

An expert pack for **Qwen3.8-Flash-Next** made for one consumer GPU plus 32 GB of RAM (developed on an RTX 3080 12 GB
with 32 GB DDR5). It runs on **sync-strata**, a fork of [Strata](https://github.com/Niko1221/Strata).
**This is not a GGUF: it does not load in llama.cpp or in upstream Strata.**

**Status: test release (2026-11-03) for a few testers.** The full release (engine source, setup, measurements and
method) follows. Expect rough edges.

## What it is

- The routed experts of Qwen3.8-Flash-Next re-quantized to a trellis format (R1Q2, in the Gyro / QTIP family) with a
  per-expert bit allocation: about 1.7 bits per weight on average, 24.07 GiB.
- The other weights (dense layers, embeddings, PLE) are read from agentionai's
  [Gyro-S GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) (revision
  `1ada08422e404aca51ab9df665f089f34c419c6b`), which you download separately (Strata's setup does it).
- Calibration used public text (code, documentation, Wikipedia, several languages, agent traces) and some of the
  author's own chat logs. The calibration text is not distributed.

Preliminary quality: on the author's 8-domain evaluation, the KL divergence to the original model is about 34% lower
than Gyro-S. Method and full numbers come with the full release.

## Test kit (Windows, NVIDIA)

`kit/sync_s_test.py` downloads this pack and the sync-strata engine (`kit/engine`), writes a config from your Strata
Gyro-S config, runs a short test of Sync-S and of Gyro-S, and zips the results. It needs Strata **v0.2.1.1** installed
with Gyro-S and an NVIDIA driver for CUDA 13 (R580 or newer). See `kit/README-ja.md` (Japanese).

## Credits

- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team.
- [agentionai](https://huggingface.co/agentionai): the Gyro-S GGUF this pack builds on, and the Gyro trellis format
  ([specification and reference code](https://gitlab.com/agentionai-group/agention-gyro), Apache 2.0). AgentionAI is a
  one-person team: [Buy Me A Coffee](https://buymeacoffee.com/agentionai).
- [Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors (MIT), which sync-strata forks.
- QTIP (Tseng et al., 2024); [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT).

## License

The weights are a derivative of Qwen3.8-Flash-Next and are under the **Qwen Community License 1.0** (see `LICENSE`):
Copyright (c) 2026 Qwen. The engine binary in `kit/engine` is under the MIT license (see the `LICENSE-*` files there).
