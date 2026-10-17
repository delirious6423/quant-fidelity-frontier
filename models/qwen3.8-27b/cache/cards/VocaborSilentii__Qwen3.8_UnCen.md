---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: llama.cpp
language:
- en
- zh
tags:
- gguf
- qwen3.8
- quantized
- q6_k
- abliterated
- uncensored
- mtp
- conversational
---

# Qwen3.8 UnCen — 27B · Q6_K

A local GGUF release of a reported reduced-refusal Qwen3.8-27B derivative, distributed by [VocaborSilentii](https://huggingface.co/VocaborSilentii). The repository contains a **Q6_K language model** and a **separate vision projector** for compatible runtimes.

The reported lineage is [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) → [JonathanColetti's Uncensored checkpoint](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored) → GGUF conversion and quantization. Exact build details for this upload are not documented; see [Provenance](#provenance).

| At a glance | This release |
| --- | --- |
| Model class | 27B parameters |
| Format | GGUF v3 · Q6_K |
| Runtime architecture | `qwen35` |
| Language model | 22.431 GB / 20.890 GiB |
| Optional projector | 0.931 GB / 0.867 GiB |
| Context declaration | 262,144 tokens; actual allocation depends on runtime and memory |
| License designation | Apache 2.0 |

[Files](#files) · [Quick start](#quick-start) · [Thinking settings](#thinking-and-sampling) · [Memory](#memory-and-context) · [Provenance](#provenance)

## Files

| Download | Purpose | Size |
| --- | --- | --- |
| [Language GGUF](https://huggingface.co/VocaborSilentii/Qwen3.8_UnCen/resolve/main/Qwen3.8-27B-Uncensored-Q6_K.gguf?download=true) | Text inference | 22.431 GB |
| [Vision projector](https://huggingface.co/VocaborSilentii/Qwen3.8_UnCen/resolve/main/mmproj-Qwen3.8-27B-Uncensored-f16.gguf?download=true) | Image input | 0.931 GB |

For text-only use, download the language GGUF. Image input also requires the projector and a runtime that supports this model/projector combination.

**Projector precision:** the filename ends in `f16`, but the recorded tensor inspection identifies **BF16/F32** tensors. Treat the filename as an identifier; check runtime compatibility before relying on image input.

## Quick start

Use a current [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases) supporting `qwen35` and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli). On Windows, use `llama-cli.exe` / `llama-server.exe`, or their full paths, if needed. Commands below are single-line examples.

These commands follow the documented interfaces. Successful inference with the exact uploaded files has not been established by this card.

### 1. Download the language model

```shell
hf download VocaborSilentii/Qwen3.8_UnCen Qwen3.8-27B-Uncensored-Q6_K.gguf --local-dir ./models
```

### 2. Start a text chat

```shell
llama-cli -m ./models/Qwen3.8-27B-Uncensored-Q6_K.gguf -c 8192 -ngl 99 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0
```

The example requests extensive GPU offload. Reduce `-ngl` and/or `-c` if memory is insufficient. An 8,192-token allocation is a starting configuration, not a measured hardware fit.

### 3. Run a local web UI and API

```shell
llama-server -m ./models/Qwen3.8-27B-Uncensored-Q6_K.gguf --alias qwen3.8-uncen --jinja -c 8192 -ngl 99 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 --host 127.0.0.1 --port 8080
```

After the model loads, open **http://127.0.0.1:8080**. The OpenAI-compatible API base URL is `http://127.0.0.1:8080/v1`, with model alias `qwen3.8-uncen`. See the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

### Optional: add image input

```shell
hf download VocaborSilentii/Qwen3.8_UnCen mmproj-Qwen3.8-27B-Uncensored-f16.gguf --local-dir ./models
```

Stop the text server first, then launch it with the additional projector:

```shell
llama-server -m ./models/Qwen3.8-27B-Uncensored-Q6_K.gguf --mmproj ./models/mmproj-Qwen3.8-27B-Uncensored-f16.gguf --alias qwen3.8-uncen --jinja -c 8192 -ngl 99 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 --host 127.0.0.1 --port 8080
```

Projector presence does not establish successful image inference. Validate loading and an image request with the runtime you intend to use.

### Ollama

```shell
ollama run hf.co/VocaborSilentii/Qwen3.8_UnCen:Q6_K
```

This is the command exposed by the [Hugging Face Ollama integration](https://huggingface.co/docs/hub/ollama). It requires an Ollama version supporting the architecture. Automatic projector loading, vision, tool calling, and MTP have not been validated for this upload.

### LM Studio

Search for `VocaborSilentii/Qwen3.8_UnCen`, select the Q6_K file, and choose a runtime supporting `qwen35`. Start with a modest context allocation and adjust GPU offload to available memory. Check projector recognition separately for image input.

## Thinking and sampling

The embedded template enables thinking by default. It accepts `enable_thinking`, `reasoning_effort` (`xhigh`, `medium`, or `low`), and `preserve_thinking`; runtime support determines which options are exposed to users.

The following are [Qwen's published starting settings](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices), not benchmarked settings for this quantization:

| Parameter | Thinking | Non-thinking |
| --- | --- | --- |
| Temperature | 1.0 | 0.7 |
| Top-p | 0.95 | 0.80 |
| Top-k | 20 | 20 |
| Min-p | 0.0 | 0.0 |
| Presence penalty | 0.0 | 1.5 |
| Repetition penalty | 1.0 | 1.0 |

For a non-thinking request to the local server, set the template option and sampling values together. Example JSON body for `POST /v1/chat/completions`:

```json
{
  "model": "qwen3.8-uncen",
  "messages": [
    {"role": "user", "content": "Explain how a hash table works."}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 1.5,
  "repeat_penalty": 1.0,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Use the embedded chat template rather than manually inserting role tokens. Template support for tools or thinking does not, by itself, establish reliable tool execution or reasoning performance.

## Memory and context

The language model occupies **22.431 GB (20.890 GiB)** on disk; the projector adds **0.931 GB (0.867 GiB)**. GB is decimal; GiB is binary.

Runtime memory also includes cache, compute buffers, runtime state, image processing, and other applications. Full offload to a nominal 24 GB GPU is **not guaranteed**. Partial offload may be possible with sufficient system RAM and a supporting backend.

The **262,144-token context** is a metadata declaration. It is not a tested context allocation for this upload. Increase `-c` only after confirming memory use and acceptable behavior at a smaller allocation. No hardware-specific throughput or maximum tested context is published here.

## Capabilities and limitations

- **Text generation:** this is the primary workflow documented above; exact-upload inference still requires validation.
- **Reduced refusal:** “UnCen,” “uncensored,” and “abliterated” describe the reported derivative. They do not guarantee that every request will be answered or that output will be correct.
- **Vision:** a separate projector is included. Its BF16/F32 tensor types and compatibility need to be supported by the chosen runtime.
- **Thinking and tools:** the embedded template contains support; end-to-end behavior depends on the backend and client.
- **MTP:** the recorded header inspection identifies an additional multi-token prediction block. Tensor presence does not mean speculative decoding is enabled or faster.
- **Evaluation:** no benchmarks or refusal-rate measurements for these exact uploaded files are supplied. Upstream results apply to their tested artifacts and settings.

## Technical details

The Hub currently indexes **27,320,697,856 parameters**, architecture **`qwen35`**, and a **262,144-token** context declaration for the language GGUF. The expanded specifications below preserve the release's recorded header inspection and upstream configuration details. Header inspection establishes structure and declared metadata, not full tensor integrity or model quality.

<details>
<summary>Architecture and tensor inventory</summary>

| Property | Value |
|---|---|
| Repository | `VocaborSilentii/Qwen3.8_UnCen` |
| Model | Qwen3.8-27B-Uncensored-Q6_K |
| Reported base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Reported reduced-refusal checkpoint | [JonathanColetti/Qwen3.8-27B-Uncensored](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored) |
| Parameter class | 27B; Hub reports 27,320,697,856 parameters for the language GGUF |
| Upstream architecture | `Qwen3_5ForConditionalGeneration`, a hybrid linear/full-attention architecture |
| GGUF architecture / version | `qwen35` / GGUF v3 |
| Blocks | 64 main blocks + 1 MTP block; GGUF declares 65 |
| Tensor count | 866; includes 15 tensor entries in `blk.64.*` |
| Vocabulary | 248,320 tokens, as specified by the base configuration |
| Embedding dimension | 5,120 |
| Full-attention heads / KV heads | 24 / 4 |
| Feed-forward dimension | 17,408 |
| Native context metadata | 262,144 tokens |
| Quantization | Q6_K (`general.file_type = 18`); 506 Q6_K and 360 F32 tensors |
| Language GGUF size | 22,430,999,936 bytes; 22.431 GB / 20.890 GiB |
| Vision | Separate projector included; runtime compatibility requires validation |
| Thinking / tool use | Supported by the embedded template; behavior depends on runtime support |
| MTP | Tensor presence verified; speculative decoding requires runtime support |
| License designation | Apache 2.0 in uploaded metadata and upstream model cards |

</details>

<details>
<summary>Q6_K quantization and calibration provenance</summary>

The uploaded language model uses llama.cpp's Q6_K format: its header records `general.file_type = 18`, and its tensor directory contains 506 Q6_K tensors and 360 F32 tensors. Q6_K is a quantization format designation, not a claim that every tensor is stored at exactly six bits.

The credited upstream release documents importance-matrix calibration with WikiText-2 raw text, using 200 chunks of 512 tokens. **The uploaded file here contains no `quantize.imatrix.*` metadata**, while the currently published upstream Q6_K file does. That absence does not prove no calibration was used, but importance-matrix use cannot be verified for this upload from its header.

The language GGUF file occupies approximately **22.431 GB (20.890 GiB)**. See [Provenance](#provenance) for file hashes and the unresolved build details.

</details>

## Provenance

The original release description reports this lineage:

```text
Qwen/Qwen3.8-27B
    -> JonathanColetti/Qwen3.8-27B-Uncensored (BF16 abliteration)
    -> GGUF conversion and Q6_K quantization
    -> VocaborSilentii/Qwen3.8_UnCen
```

The metadata's `base_model` field identifies the original Qwen model; the checkpoint lineage above is reported by the release description. The exact source checkpoint revision and build process remain undocumented for this upload.

### Uploaded artifact identifiers

The following SHA-256 values come from Hugging Face's LFS file records at repository revision [`c7c685c9d60c501d02b9df1be91e395e0d5b75c9`](https://huggingface.co/VocaborSilentii/Qwen3.8_UnCen/tree/c7c685c9d60c501d02b9df1be91e395e0d5b75c9). They have not been independently recomputed by downloading the full files.

```text
Qwen3.8-27B-Uncensored-Q6_K.gguf
1a0d6fe2e48417f1571a57a7104b4d5da0c45c3d00de18fa7aa48affca5a24d7

mmproj-Qwen3.8-27B-Uncensored-f16.gguf
add205b7bfdb3f71f6da36b0a82aa20928dd829a920878c602628cdfbebc5288
```

### Comparison with the credited GGUF release

At upstream revision [`45d0fc0ad6cfcf9eeee6008ddaea925536409b18`](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF/tree/45d0fc0ad6cfcf9eeee6008ddaea925536409b18), its Q6_K file is 22,430,999,968 bytes with SHA-256 `a50aa1478295b58ee3d93eabe02c17f6d5fcf6cb787fd8a0ab07ac629a46cae6`. Its F16 projector is 927,606,912 bytes with SHA-256 `5ac423f8a29059dc24e51bc6a43e9380dcd57a9347f28b62591e0b3f60b7081c`.

Both differ from the files here. Different hashes establish different file contents; they do not, by themselves, establish how the language-model weight values differ. Header inspection also found missing importance-matrix metadata in this upload and different projector tensor precision.

**Still unverified:** the exact conversion/quantization commands and tool revisions, calibration procedure for this upload, tensor-value equivalence to upstream, local inference, image inference, MTP speed, and benchmark performance. Header inspection verifies structure and declared metadata, not full tensor integrity or model quality.

## Troubleshooting

- **Unknown `qwen35` architecture or unsupported tensors:** check the runtime version and its model support. Projector support must be checked separately from text-model support.
- **Out of memory:** lower `-c`, reduce `-ngl`, and try text-only loading before adding the projector. File size alone is insufficient to predict VRAM use.
- **Unexpected thinking output:** confirm the embedded template is active and the client passes `chat_template_kwargs`; use the matching sampling profile above.
- **Vision or MTP does not activate:** confirm backend support and launch options. Downloading the files does not enable those features automatically.

## License and attribution

**Apache License 2.0** is the designation in the uploaded GGUF metadata and both cited upstream model cards. The base model's license text is available in [Qwen/Qwen3.8-27B/LICENSE](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE).

This card records the published license designation; consult the license text for its terms.

- Base model and architecture: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
- Reported reduced-refusal checkpoint: [JonathanColetti/Qwen3.8-27B-Uncensored](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored).
- Credited GGUF reference release and upstream measurements: [JonathanColetti/Qwen3.8-27B-Uncensored-GGUF](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF).
- Abliteration tooling: [Heretic](https://github.com/p-e-w/heretic).
- GGUF format and inference tooling: [llama.cpp](https://github.com/ggml-org/llama.cpp).
- This repository: [VocaborSilentii](https://huggingface.co/VocaborSilentii).

No new upstream training, quantization work, or benchmark results are attributed to the repository publisher by this card.
