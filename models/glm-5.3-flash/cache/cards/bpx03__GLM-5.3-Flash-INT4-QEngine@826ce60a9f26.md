---
license: mit
base_model:
- zai-org/GLM-5.3-Flash
tags:
- qengine
- glm-5.3
- int4
- gptq
---

# GLM-5.3-Flash INT4 for QEngine

GLM-5.3-Flash quantized for [QEngine](https://github.com/bpx03/QEngine) 1.5 or later, an open-source (MIT) inference
engine for 3x NVIDIA RTX 3090. These files are laid out for QEngine's GLM engine; other runtimes won't read them as a
normal checkpoint.

```bash
hf download bpx03/GLM-5.3-Flash-INT4-QEngine --local-dir models/GLM-5.3-Flash-INT4-QEngine
server/run.sh 8080 --model models/GLM-5.3-Flash-INT4-QEngine
```

## What you need

- 3x 24 GB GPUs is what I use (it also starts on 2 cards, with fewer expert slots in VRAM). P2P between the cards helps
  but isn't required.
- About 192 GB of host RAM: the routed experts (about 157 GB) stay pinned in host RAM and the cards cache the hot ones.
  With less RAM the engine starts with a smaller conversation archive; below the expert size it can't load.
- An x86-64 CPU with AVX2. AVX-512 is used when the CPU has it (the CPU expert lane picks its kernels at start).
- About 186 GB of disk.

## What is in it

- Routed experts: GPTQ INT4 (`q_g4_experts-*.safetensors`), calibrated on my own records, with QEngine's expert layout.
- KDA (linear attention) projections: GPTQ INT4 g128 with act-order, from the BF16 weights (`q_kda-*.safetensors`).
  QEngine keeps them in INT4 in VRAM; the decode reads the packed values directly.
- Everything else (attention, MLA, the shared expert, router, embeddings, head, the MTP draft layer) from the INT4
  AutoRound release of GLM-5.3-Flash by Intel (W4A16; its non-quantized tensors), repacked into
  `model-dense-*.safetensors`.
- `SHA256SUMS` for every file.

Both quantization steps are lossy. Numbers (KL against the FP8 reference, task accuracy, speed) are on the QEngine
README for 1.5.

## License

MIT, as the original GLM-5.3-Flash model. Please also follow the license of the original model.
