---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: image-text-to-text
tags:
  - mlx
  - quantized
  - qwen4-exp
---

# Qwen3.8-Flash-Next MLX Mixed 2-bit/4-bit

This is a reproducible MLX conversion of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at immutable
revision `f5d08274bafd880402bd16f5e3e6c514136ec06c`.

The 48 base routed-expert banks use MLX affine Q2/group-128, the 160-wide n-gram table and remaining eligible core/MTP matrices use Q4, while token/output embeddings, QSA indexers, routers, vision, and MTP fusion heads remain BF16. This is the 128 GB Mac profile.

- Artifact payload: 68.08 GiB
- Quantized Q2 modules: 144
- Quantized Q4/group-32 modules: 128
- Quantized Q4/group-64 modules: 682
- Source: 180B parameters including 125B main, 51B n-gram embedding, and 4B MTP
- Native model context: 262,144 tokens; see the v0.45.0 runtime limit below.

## Run locally with mere.run

Native support is available as of [**mere.run v0.45.0**](https://mere.run/releases).
The instructions below use its Apple Silicon Swift/MLX runtime. No Python
environment, local quantization, or separate MTP-model download is needed.

### Requirements and current limits

- Apple Silicon Mac, macOS 15 or later, with **128 GB unified memory recommended**.
- Enough free disk space for the approximately 68.08 GiB checkpoint; a fresh
  v0.45.0 pull budgets about 82 GiB including safety headroom. Leave memory
  available for macOS and runtime buffers.
- **In v0.45.0, text generation is qualified; image input is not yet qualified**,
  despite the model's upstream multimodal architecture and `vision-chat` ID.
- **Prompt plus requested output must fit within 2,048 tokens**, including chat
  template/system tokens. The upstream 262,144-token context is not yet available
  in this runtime: long-context QSA selection is still being integrated. Requests
  above the current budget are rejected. Set `--max-tokens` explicitly; the
  generic 2,048-token output default leaves no room for a prompt.

### 1. Install or update mere.run

Download mere.run v0.45.0 or newer from the [mere.run releases page](https://mere.run/releases),
install MereRun in Applications, then open the app's
Settings and choose **Install CLI**. Existing installations can use
**MereRun > Check for Updates…** and reinstall the CLI from Settings if needed.

```bash
mere.run --version
```

This should report `0.45.0` or later. See the
[getting-started guide](https://mere.run/getting-started) for installation details.

### 2. Pull this pre-quantized checkpoint

After reviewing the [Qwen Community License 1.0](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit/blob/main/LICENSE):

```bash
mere.run model pull vision-chat-q38-flash-next-mixed \
  --accept-license-terms
```

Use this **managed model ID**, not the Hugging Face repository name, with the
commands below. It resolves to this Sawfwair mixed Q2/Q4 distribution and includes
the bundled MTP head. The repository is public and ungated; the acceptance flag
records acknowledgement of the model's terms, not a request for gated access.

For an external SSD, add `--cache-dir /Volumes/Models/huggingface-cache` to the
pull command, substituting your mounted drive's path. Keep that drive mounted
while using the model. To inspect support and download requirements without
downloading, add `--preflight --json` to the pull command.

### 3. Generate text with verified MTP

```bash
mere.run text chat \
  --model vision-chat-q38-flash-next-mixed \
  --context-size 2048 \
  --max-tokens 256 \
  --temperature 0 \
  --no-thinking \
  --stream \
  --stats \
  --prompt "Explain sparse attention in three short sentences."
```

`--temperature 0` selects greedy decoding, which uses the bundled MTP draft head
by default for eligible requests. Proposed tokens are verified by the full target
model. No opt-in flag or extra checkpoint is required. `--stats` reports local
generation timing; `--no-thinking` keeps this short example focused on the answer.

For a target-only comparison or to avoid loading the draft head, prefix the same
command with `MERERUN_Q35_MTP_SPECULATION=0`.

### 4. Serve a local OpenAI-compatible API

After the pull completes, start the server:

```bash
mere.run api serve \
  --engine text-chat-q36 \
  --model vision-chat-q38-flash-next-mixed \
  --context-size 2048 \
  --host 127.0.0.1 \
  --port 8080
```

`text-chat-q36` is the shared Qwen-family serving engine; it also dispatches this
Qwen3.8-Flash-Next model. Startup loads and warms the target and MTP graphs before
the server begins listening, so allow the initial warmup to finish.

From another terminal:

```bash
curl http://127.0.0.1:8080/health

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vision-chat-q38-flash-next-mixed",
    "messages": [{"role": "user", "content": "Explain sparse attention in three short sentences."}],
    "max_tokens": 256,
    "temperature": 0
  }'
```

The example binds to localhost only. If you have configured `MERERUN_API_KEY`,
include its bearer token in requests. See the
[API guide](https://mere.run/runtime/api-server) for authentication and client setup.

## Runtime status

The tensor inventory, source hashes, MLX packing, fused-expert split, convolution
layout, and zero-centered RMSNorm conversion are validated by the bundled
`MERERUN_CONVERSION.json`. A Qwen4Exp-aware MLX runtime is required; do not
expect an older `mlx-lm` or `mlx-vlm` release to dispatch this new architecture.

## License

This redistribution retains the upstream **Qwen Community License 1.0** in
`LICENSE`. Review it before use. In particular, it contains attribution/display
requirements for very large commercial products and separate-license conditions
for certain commercial Model-as-a-Service and AI Work Assistant uses. The model
is not gated; downloading or using it does not remove those terms.

The upstream model card is preserved as `README.upstream.md`.
