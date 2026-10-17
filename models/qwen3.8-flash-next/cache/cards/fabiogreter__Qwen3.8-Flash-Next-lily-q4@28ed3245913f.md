---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
inference: false
tags:
  - lily-qwen3.8-flash-next
  - metal
  - apple-silicon
  - quantized
  - 4-bit
  - speculative-decoding
  - vision
---

# Qwen3.8-Flash-Next, 4-bit, for lily-qwen3.8-flash-next

A 4-bit conversion of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
in the checkpoint format used by [lily-qwen3.8-flash-next](https://github.com/fabiogreter/lily-qwen3.8-flash-next),
a Metal inference server for Apple Silicon. It runs the model, text and
images, on a single Mac with 128 GB of unified memory, with speculative
decoding through the checkpoint's own multi-token-prediction head.

**There is an XL variant,
[fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl).**
It keeps attention, the shared expert, the LM head and the embedding at 8
bits, which made the model noticeably more reliable on long agent turns, at
a cost of 2 to 5 % in decode speed with speculative decoding and about 15 %
without it. This 4-bit checkpoint is the faster one.

**These weights load only in lily-qwen3.8-flash-next.** They are not readable by Transformers,
vLLM, SGLang, MLX or llama.cpp. The tensor layout, the quantization metadata
and the packed n-gram tables are specific to its loader. If you want the
model in a portable format, start from the base repository instead.

## What is in here

| | |
|---|---|
| size on disk | 98 GiB across 56 safetensors files: 54 for the model, 1 for the draft head, 1 for the vision tower |
| tensors | 3 842, of which 78 belong to the draft head and 333 to the vision tower |
| format | `qwen4_exp-affine-v1` |
| source revision | `f5d08274bafd880402bd16f5e3e6c514136ec06c` |

The architecture is unchanged from the source: 48 layers with full attention
every fourth layer and a gated delta net in between, 512 experts with 10
active per token, 24 query heads over 2 key-value heads, a hidden size of
2 560, a 248 320-token vocabulary, and 262 144 positions. The learned n-gram
embeddings occupy 16 hash tables totalling 320 001 536 rows.

Two things about what is kept:

- **The vision tower is kept in bf16**, byte for byte from the source
  (`model.visual.*`, 0.9 GB), so the server takes PNG and JPEG images in
  chat requests. The server verifies its tower against the reference
  implementation; the checks are described in the engine's architecture
  document.
- **The multi-token-prediction head is kept and quantized**, so the server
  can draft with it. Runtimes that ignore the head are unaffected by its
  presence.

## Quantization

Affine quantization throughout, applied per module rather than uniformly.

| weights | bits | group size |
|---|---|---|
| experts, attention, embeddings, draft head | 4 | 64 |
| vision tower | bf16, not quantized | |
| n-gram embedding tables | 4 | 32 |
| mixture-of-experts gates, shared expert gate, hyper-connection input mixes, block inject, n-gram key and value projections, attention indexer projection | 8 | 64 |

The 8-bit set is small in parameter count and sits where 4-bit rounding
measurably moves the output: routing decisions, the hyper-connection mixes
and the sparse-attention indexer. The n-gram tables use a tighter group size
because their rows are short and read individually.

## Requirements

- Apple Silicon, macOS 26 or newer. The server targets the Metal 4 command model
  and does not build on earlier systems.
- 128 GB of unified memory for the default configuration. At runtime the
  server holds about 73 GB of weights and scratch in GPU memory and reads the
  32 GB of n-gram tables through the page cache, leaving the rest for the
  session cache.
- 106 GB of free disk, plus room for the session cache if you enable its
  disk tier.

## Use

```bash
hf download fabiogreter/Qwen3.8-Flash-Next-lily-q4 \
  --local-dir ~/models/Qwen3.8-Flash-Next-lily-q4

git clone https://github.com/fabiogreter/lily-qwen3.8-flash-next
cargo build --release --locked --manifest-path lily-qwen3.8-flash-next/Cargo.toml

./lily-qwen3.8-flash-next/target/release/lily \
  --model ~/models/Qwen3.8-Flash-Next-lily-q4 \
  --bind 127.0.0.1:8000 --max-seq 131072
```

The server speaks the OpenAI chat and completions API:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Say hello."}]}'
```

Flags, session caching, idle unloading and the launchd service are documented
in the engine's README.

## Speed

Measured on an M5 Max with a 40-core GPU and 128 GB of memory over the HTTP
API, on prompts cut from real documentation and code, 256 greedy tokens of
new text per run, medians of three repeats, 2026-10-07. Prefill is tokens
per second over the prompt; decode is generated tokens per second.

| context | prefill | decode | decode with 2 drafts |
|---:|---:|---:|---:|
| 4K | 2 158 | 87 | 103 |
| 16K | 2 208 | 82 | 100 |
| 32K | 2 240 | 85 | 99 |
| 64K | 2 205 | 79 | 99 |

Draft acceptance was 63 % on this text. Requests that decode together in a
batch decode without drafts. On the same machine, prompts and model,
Unsloth's llama.cpp fork (UD-IQ4_XS, with its MTP drafter) prefilled at 887
to 550 tokens per second over the same range and decoded at 52 to 27
(measured 2026-09-17).
The method, the noise band and the comparison are in the engine's
[performance document](https://github.com/fabiogreter/lily-qwen3.8-flash-next/blob/main/docs/performance.md).

## Limitations

- Images arrive as base64 data URIs (PNG or JPEG), scaled to at most 2
  megapixels; no video and no URL fetching.
- Up to four requests decode together; a request decoding alone uses
  speculative decoding, batched ones do not.
- Quantized output is not bit-identical to the bf16 reference. Near-ties in
  the argmax can resolve differently.

## Provenance

Converted from `Qwen/Qwen3.8-Flash-Next` at revision `f5d0827` with the
converter in the engine repository. The conversion is deterministic: the same
source revision and the same converter produce the same shards. The recorded
policy, including the per-module group sizes and the n-gram hash seed, lives
in the `lily` block of `config.json`, and the loader refuses a checkpoint
whose format string it does not recognise.

## License

The base model is released under the Qwen Community License 1.0, and this
derivative is distributed under the same terms. The full text is in
[LICENSE](LICENSE). Two conditions deserve attention before you build on it:

1. The copyright and permission notice must travel with any copy or
   substantial portion. Products above 100 million monthly active users or 20
   million US dollars of monthly revenue must display the model name in their
   user interface.
2. A separate license from Qwen is required for commercial use by a Model as
   a Service business or an AI work assistant product, as those terms are
   defined in the license. Internal use that does not expose the model or its
   outputs to third parties is exempt.

Nothing here is legal advice. Read the license.

## Credits

The model is Qwen's. The engine is a fork of Perplexity's
[pplx-garden/lily](https://github.com/perplexityai/pplx-garden/tree/main/lily),
which contributed the Metal kernel foundations and the original server.
