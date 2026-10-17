---
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- mtp
- speculative-decoding
- rocmfp4
- vulkan
- strix-halo
- qwen4exp
---

# Qwen3.8-Flash-Next MTP draft head (ROCmFP4-FAST)

The multi-token-prediction head from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), 2.28 GiB.
Qwen trains it jointly with the target model, so it drafts better than a separate small
model would.

**This is a draft head, not a model.** On its own it does nothing. It is used with
[agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-GGUF),
which is an experimental build; see that card first.

## Setup

`qwen4exp` and the ROCmFPx quant types are not in upstream llama.cpp yet, so build this branch:

```bash
git clone https://github.com/LaurentZuijdwijk/llama.cpp
cd llama.cpp && git checkout vulkan/qwen4exp-rocmfpx
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```bash
./build/bin/llama-server \
  -m Qwen3.8-Flash-Next-ROCmFP4-FAST-00001-of-00005.gguf \
  -md Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST.gguf \
  -ngl 99 --n-gpu-layers-draft 99 \
  --spec-type draft-mtp --spec-draft-n-max 3 -c 32768
```

Adds ~2.3 GiB to the ~85 GiB the target uses.

## Measurements

Radeon 8060S (Ryzen AI MAX+ 395), 250 tokens at temp 0, each config warmed up first.

| draft | t/s | acceptance |
|---|---|---|
| none | 28.1 | -- |
| n-max 2 | 31.8 | 0.695 |
| n-max 3 | 32.4 | 0.612 |

Quantized to match the target, not above it. A Q8_0 draft measured *worse* on both
throughput and acceptance and cost 1.5 GiB more (30.3 t/s, 0.587): acceptance is the
draft agreeing with the target, and two models quantized the same way are wrong in the
same places.

Adaptive drafting also measured worse here (28.4 t/s, 0.468). It drafts longer when
acceptance looks good, and this head carries its own 512-expert MoE, so each extra
drafted token is a real forward pass.

## Credits

`qwen4exp` support is the work of **Daniel Han**
([@danielhanchen](https://github.com/danielhanchen)), from
[ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742), and the MTP
graph is from [#27739](https://github.com/ggml-org/llama.cpp/pull/27739) (JJJYmmm). Both
are unmerged drafts; if they land upstream, prefer upstream.

Quant formats hand-ported from [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX).
The ROCmFP4 format was created by [charlie12345](https://github.com/charlie12345) in
[charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX), which ciru-ai's tree forks.
Both upstream projects are MIT-licensed.
Base model by the Qwen team.

Quantized and published by [Agention](https://agention.ai).

## License

Qwen Community License 1.0, included as `LICENSE`.
