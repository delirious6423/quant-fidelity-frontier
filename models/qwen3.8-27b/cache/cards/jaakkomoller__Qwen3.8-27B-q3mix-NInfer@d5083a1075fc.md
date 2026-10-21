---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - q3
  - multimodal
  - conversational
  - cuda
  - blackwell
  - rtx-5060ti
model-index:
  - name: Qwen3.8-27B-q3mix-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: RTX 5060 Ti 5,113-token code prompt
          type: custom
        metrics:
          - type: throughput
            name: MTP3 prefill tok/s
            value: 615.9
            verified: false
          - type: throughput
            name: MTP3 decode tok/s
            value: 46.13
            verified: false
          - type: accuracy
            name: MTP3 acceptance %
            value: 86.67
            verified: false
          - type: throughput
            name: MTP0 decode tok/s
            value: 23.03
            verified: false
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: LiveCodeBench v6 (175-problem subset)
          type: livecodebench
        metrics:
          - type: accuracy
            name: "Prompt-level strict (greedy, 1 sample)"
            value: 24.57
            verified: false
---

# Qwen3.8-27B (q3mix) for NInfer

This model card is the version-controlled source for
[jaakkomoller/Qwen3.8-27B-q3mix-NInfer](https://huggingface.co/jaakkomoller/Qwen3.8-27B-q3mix-NInfer).

The repository contains
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) converted to the native
[NInfer](https://github.com/jaakkomoller/ninfer-5060ti) `.ninfer` artifact format with the
selective 3-bit `q3mix` weight profile, the production profile for NVIDIA RTX 5060 Ti cards with
16 GB of VRAM. The artifact is intended only for NInfer; it is not a Transformers checkpoint,
Safetensors distribution, or GGUF file.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_q3mix.ninfer` |
| Size | 14,823,303,168 bytes (13.73 GiB) |
| SHA-256 | `704282c8aa3fd7763cd3a93bc8fa41580c204db978ecf895150d3e08ede630ea` |
| Container version | 2 |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `groupwise-int` |
| NInfer target key | `qwen3_8_27b` |
| Stored objects | 1,124 (1,118 tensors and 6 resources) |

The Text body uses the selective `q3mix` allocation: the 64 text `mlp/down` projections
(layers 0–63, `[5120,17408]`) are stored as `Q3G64_F16S`, while the remaining four text linear
roles (`attention/gate_value`, `attention/output`, `gdn/value_z`, `gdn/output`) stay
`Q4G64_F16S`. The MTP transformer matrices use the Q4 mix, the token embedding and full output head
use `W8G32_F16S`, and the Vision matrices use the W8/Q6 allocation. The GDN recurrent state is
stored INT8 with per-(head, dv-row) FP16 scales, and the paged KV cache runs as INT4 with one
FP16 scale per 64-element group in the production configuration. This removes ~680 MiB of resident
weights versus the pure-Q4 16 GB profile, which is what lets a ~100,000-token MTP3 context fit in
16 GB. The file also contains the registered Vision, MTP, proposal-head, tokenizer, chat-template,
generation, and media-processor objects required by NInfer.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '704282c8aa3fd7763cd3a93bc8fa41580c204db978ecf895150d3e08ede630ea' \
  'qwen3_8_27b_q3mix.ninfer' | sha256sum --check
```

## Requirements

- [NInfer `rtx-5060ti` branch](https://github.com/jaakkomoller/ninfer-5060ti/tree/rtx-5060ti)
  of `jaakkomoller/ninfer-5060ti` at
  [`b2c9f8c6`](https://github.com/jaakkomoller/ninfer-5060ti/commit/b2c9f8c662209ff07f31d2727f59ce07054102cf)
  or later, built from source;
- 64-bit Linux;
- an NVIDIA Blackwell `sm_120a` GPU with at least 16 GB of VRAM (validated on a
  GeForce RTX 5060 Ti 16 GB);
- CUDA Toolkit 13.1 or newer.

NInfer does not provide an install target or packaged binary. See the
[repository README](https://github.com/jaakkomoller/ninfer-5060ti/blob/rtx-5060ti/README.md)
for source-build dependencies.

## Download and run

```bash
hf download jaakkomoller/Qwen3.8-27B-q3mix-NInfer \
  qwen3_8_27b_q3mix.ninfer \
  --local-dir models
```

```bash
./build/apps/ninfer models/qwen3_8_27b_q3mix.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --kv-dtype int4 \
  --prefill-chunk 128 \
  --spec mtp --draft-tokens 3
```

HTTP serving, production profile (~100k-token context):

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_q3mix.ninfer \
  --host 127.0.0.1 --port 8086 \
  --model-id qwen3.8-27b-q3mix \
  --max-context 100000 \
  --kv-capacity auto --kv-dtype int4 \
  --max-concurrency 1 --max-pending-requests 2 --pending-timeout-ms 60000 \
  --prefill-chunk 128 \
  --spec mtp --draft-tokens 3 --preserve-thinking \
  --default-max-tokens 32768 \
  --cors
```

Add `--vision` to the server when it must accept image or video input; vision is disabled by
default. For images, videos, structured chat history, tool calls, and the full endpoint list, see
the [NInfer documentation](https://github.com/jaakkomoller/ninfer-5060ti/tree/rtx-5060ti/docs).

## Performance (RTX 5060 Ti 16 GB)

Context ceiling under the production INT4 KV configuration:

| Configuration | Max context | Auto `--kv-capacity` | Note |
|---|---:|---:|---|
| `--no-prefix-reuse --spec mtp --draft-tokens 3 --max-context 100000` | 100,000 | 100,032 (1,563 groups) | 371 MiB planned slack; 90k-token prompts verified end-to-end |
| Production flags, no explicit cap | 119,168 | 119,168 (1,862 groups) | 33 MiB planned slack; 119,232 is rejected before upload |
| Defaults (prefix reuse enabled) | 107,904 | 107,904 (1,686 groups) | 193 MiB planned slack |

Head-to-head against the pure-Q4 profile on one 5,113-token code prompt, 128 generated tokens,
greedy, `--max-context 6144 --prefill-chunk 128 --no-prefix-reuse`:

| Metric | q3mix | q4 |
|---|---:|---:|
| MTP3 prefill | 615.9 tok/s | 650.2 tok/s |
| MTP3 decode | 46.13 tok/s | 46.00 tok/s |
| MTP3 acceptance | 86.67% (3.60 tok/round) | 85.05% (3.53 tok/round) |
| MTP0 prefill | 617.1 tok/s | 658.2 tok/s |
| MTP0 decode | 23.03 tok/s | 24.65 tok/s |

Decode is at parity under MTP3 and 6.6% slower under MTP0; prefill is 5–6% slower because the
3-bit down-projection GEMM is less efficient per byte. The weight savings buy the ~100k-token
context ceiling that the Q4 profile cannot reach on 16 GB.

## Quality

On the 175-problem LiveCodeBench v6 subset (greedy, max 2,000 tokens, single sample): 43/175 for
`q3mix` versus 41/175 for `q4` — the Q3 profile introduces no measurable regression, and the shared
gap to the stored vLLM reference (60/175) is runtime/decode-budget behavior, not quantization. The
small quality gate scores 29/35 for `q3mix` versus 28/35 for `q4`.

## Supported use

The artifact supports:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages (server started with `--vision`);
- MTP speculative decoding; the production configuration uses 3 draft tokens;
- INT4 group-64 paged KV cache (production) and INT8 KV at smaller contexts;
- CUDA Graph decode and compatible-prefix reuse;
- startup-bounded small-scale concurrent serving with true batched decode;
- the NInfer CLI;
- OpenAI- and Anthropic-compatible serving.

## Limits

- The artifact is accepted only by the `rtx-5060ti` branch of NInfer at `b2c9f8c6` or later and
  the matching registered target.
- NInfer executes on one GPU and one CUDA device, with a startup-fixed capacity of 1–8 active
  requests per Engine; on a 16 GB card the practical configuration is single-request serving.
- It does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  multi-GPU execution, CPU/GPU offload, or distributed serving.
- Deep contexts (beyond roughly 3,300 tokens) require the INT4 KV cache; INT8 KV fits only
  smaller contexts on 16 GB.
- NInfer does not execute generated tool calls.

## Provenance

| Field | Value |
|---|---|
| Source repository | `Qwen/Qwen3.8-27B` |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Download source | `modelscope.cn/models/Qwen/Qwen3.8-27B` |
| Conversion recipe | `qwen3_8_27b-q3mix-v1` |
| Converter repository | [`jaakkomoller/ninfer-5060ti`](https://github.com/jaakkomoller/ninfer-5060ti), branch `rtx-5060ti` |
| Converter revision | `3b64a0e612a17f3885d9ba4f375242c1fa49e987` |
| Minimum runtime revision | `b2c9f8c662209ff07f31d2727f59ce07054102cf` (branch `rtx-5060ti`) |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

The local source configuration, tensor index, frontend resources, and published CRC32 inventory
were checked against the canonical source revision above before conversion publication.

The artifact identity, summarized object inventory, and conversion provenance are published in
[`artifact-manifest.json`](https://huggingface.co/jaakkomoller/Qwen3.8-27B-q3mix-NInfer/blob/main/artifact-manifest.json).

## License

This NInfer artifact is distributed under the Apache License 2.0. The source
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) repository is also licensed under
Apache-2.0. Users remain responsible for complying with the license and applicable laws.