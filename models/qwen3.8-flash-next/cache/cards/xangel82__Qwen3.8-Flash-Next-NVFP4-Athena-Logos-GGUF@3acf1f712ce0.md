---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/xangel82/Qwen3.8-Flash-Next-NVFP4-Athena-Logos-GGUF/blob/main/LICENSE-Qwen
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- gguf
- nvfp4
- mxfp8
- athena-engine
- dgx-spark
- gb10
---

# Qwen3.8-Flash-Next NVFP4 Athena Logos, GGUF for Athena's Engine

<div align="center">
<table align="center">
  <tr>
    <td align="center"><a href="https://github.com/xangel82/Athena-Engine"><img src="https://raw.githubusercontent.com/xangel82/Athena-Engine/main/docs/assets/athena-engine-logo.png" alt="Athena's Engine — AI for everyone" width="150"></a></td>
    <td align="center"><a href="https://github.com/xangel82/Athena-Logos"><img src="https://raw.githubusercontent.com/xangel82/Athena-Logos/main/docs/assets/athena-logos-logo.png" alt="Athena Logos" width="150"></a></td>
  </tr>
</table>
</div>

**Frontier AI, on your desk.** This model is part of **[Athena](https://palaferri.com/athena)**:
[Athena's Engine](https://github.com/xangel82/Athena-Engine) runs frontier-class models on a single
NVIDIA GB10 — a DGX Spark or an ASUS Ascent GX10, a computer that sits on a desk — and
[Athena Logos](https://github.com/xangel82/Athena-Logos) is the assistant built on it. Your prompts,
your documents and your code never leave your machine.

- **Athena's Engine** — an inference engine written for the GB10: DeepSeek V4 Flash, its Vision-Exp
  checkpoint that reads images, Qwen3.8 Flash Next (this model) and DeepSeek V4.1 Flash, a model
  larger than the machine streamed from its SSD; the full 256k context; OpenAI- and
  Anthropic-compatible APIs with streaming and tool calls; speech, transcription, multimodal search
  and video with its sound on the same machine, at the same time.
  `docker pull ghcr.io/xangel82/athena-engine:1.0.5`
- **Athena Logos** — the assistant: open it in a browser on the Spark, your computer or your phone,
  talk to her hands-free, give her your documents by the thousand and she answers with the page
  the answer came from, build documents, presentations and drawings next to the conversation, and
  let agents work for you and report back. For Windows, macOS and Linux:
  [latest release](https://github.com/xangel82/Athena-Logos/releases/latest).

Free for personal use, research, education, and developers and small companies at work.
More at **[palaferri.com/athena](https://palaferri.com/athena)**.

<p align="center">
  <img src="https://raw.githubusercontent.com/xangel82/Athena-Logos/main/docs/screenshots/home.jpg" alt="Athena Logos: Athena's presence, your widgets, one box to write or talk" width="760">
</p>

## This model

Our own four-bit quantization of the official
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) weights, made for the tensor
cores of the GB10 and named after the assistant it serves. It is the Qwen that Athena's Engine runs
by default from release 1.0.5.

- **Routed experts and the per-layer n-gram table in NVFP4** — blocks of 16 values with an E4M3
  scale, each scale chosen by searching every E4M3 value for the smallest error, weighted by an
  importance matrix the engine collected while serving about 800,000 tokens of calibration prompts:
  prose in several languages, code, long documents, reasoning, and agent sessions with their tool
  calls.
- **Dense projections in MXFP8** — blocks of 32.
- **Everything else as in the original**, BF16 and F32.

Made from `Qwen/Qwen3.8-Flash-Next` at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.

**These files run on Athena's Engine 1.0.5 or later only.** NVFP4 and MXFP8 are stored in tensor
types the engine defines for its GB10 kernels (contiguous tiles for the experts and the projections,
16-value blocks for the n-gram table): llama.cpp and other GGUF readers do not know these types and
cannot load them.

## Files

| File | Role | SHA-256 |
|---|---|---|
| `Qwen3.8-Flash-Next-NVFP4-Athena-Logos-00001-of-00003.gguf` | part 1 of 3 (all the metadata) | `e0f51892c78e0bbbf9fbf542e30f1d67539cbdff1c1303bbd65391f6d4e39a48` |
| `Qwen3.8-Flash-Next-NVFP4-Athena-Logos-00002-of-00003.gguf` | part 2 of 3 | `c4112a1080749e7b278ed79dc5b92e18e13433e5ecc7502337d4878bf1b25a31` |
| `Qwen3.8-Flash-Next-NVFP4-Athena-Logos-00003-of-00003.gguf` | part 3 of 3 | `28ced215eeece8df940c48f3bdf9166b0afd2050a2bc2a578ae2b123194a8df7` |
| `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | the model's MTP head, for speculative decoding | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |
| `mmproj-F16.gguf` | the vision encoder, to read images | `1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980` |

The three parts are one 104 GB file (SHA-256
`5ac66e733fccf51119602d2106195e2111f18558ff3f026c219cc771ce0fa4bf`) split in llama.cpp's shard
layout, because Hugging Face takes files of up to 50 GB; every tensor is byte for byte the one in
that file, and the engine gives the same answers from the parts as from the single file.
`SHA256SUMS` lists the checksums.

## Use

```bash
hf download xangel82/Qwen3.8-Flash-Next-NVFP4-Athena-Logos-GGUF --local-dir models/qwen-athena
```

Then point the engine at the first part:

```bash
export ATHENA_MODEL=models/qwen-athena/Qwen3.8-Flash-Next-NVFP4-Athena-Logos-00001-of-00003.gguf
export ATHENA_DRAFT=models/qwen-athena/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
export ATHENA_MMPROJ=models/qwen-athena/mmproj-F16.gguf
./athena-engine/athena-engine.sh
```

The engine keeps the 28.8 GB n-gram table on disk and reads only the rows each token needs, so the
model leaves more memory to the rest of the machine; `ATHENA_ROW_TABLE=resident` locks it in memory
instead; the speed is the same within a few percent.

## Speed and quality on one GB10

One DGX Spark, Athena's Engine, the full 262,144-token context, MTP on, against the same model
as a 4-bit IQ4_XS GGUF on the same engine:

| | This model | IQ4_XS |
|---|---|---|
| Prefill, 8k-token prompt | 1,762 tokens/s | 1,277 tokens/s |
| Decode after 8k | 39.6 tokens/s | 37.9 tokens/s |
| Decode, a chat answer | 46.6 tokens/s | 41.6 tokens/s |

tool-eval-bench 2.7 (temperature 1.0, seed 42, `--parallel 4`) scores it **88/100**; the IQ4_XS
GGUF scored 86 to 88 with the same settings.

## License

The base model, Qwen3.8-Flash-Next, is © Qwen and released under the **Qwen Community License 1.0**,
included here as `LICENSE-Qwen`; these files are a derivative work under the same license, including
its conditions for very large commercial products and for Model-as-a-Service and AI Work Assistant
businesses. They are published by the author of Athena's Engine and are not published or endorsed
by the Qwen Team.

The chat template, the MTP head and the vision encoder come in GGUF form from
[Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), with thanks.
