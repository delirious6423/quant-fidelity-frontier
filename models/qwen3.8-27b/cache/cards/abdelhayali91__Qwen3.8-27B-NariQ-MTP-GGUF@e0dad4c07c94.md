---
license: apache-2.0
library_name: gguf
pipeline_tag: image-text-to-text
base_model:
  - nari-labs/Qwen3.8-27B-NariQ-GGUF
  - unsloth/Qwen3.8-27B-GGUF
  - Qwen/Qwen3.8-27B
tags:
  - gguf
  - nariq
  - trellis
  - mtp
  - speculative-decoding
  - llama-cpp
  - qwen35
---

# Qwen3.8-27B NariQ + MTP (GGUF)

[nari-labs/Qwen3.8-27B-NariQ-GGUF](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF) ships without an MTP head, and external drafters (DFlash, DSpine) do not start with it because they share the target's embedding and LM head, which are trellis-coded in NariQ. These files add a working **MTP (nextn) speculative-decoding layer**, which makes decoding **~1.8x faster** with output that is unchanged.

**All NariQ tensors are byte-for-byte identical to the originals.** Added tensors (from `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-UD-Q5_K_XL.gguf`):

| Tensor | Why |
|---|---|
| `blk.64.*` (attention, FFN, `nextn.eh_proj`, norms) | The Qwen3.8-27B MTP layer |
| `blk.64.nextn.shared_head_head.weight` | NariQ's LM head is split into 16 trellis bands, and the MTP graph needs a plain head (`missing LM head` assert otherwise) |
| `blk.64.nextn.embed_tokens.weight` | NariQ's token embedding is trellis-coded, and the MTP step can't use it (`cur_backend_id != -1` assert otherwise) |

Metadata changes: `qwen35.block_count` 64 → 65, `qwen35.nextn_predict_layers` = 1, and `attention.recurrent_layers` extended with one full-attention entry.

| File | Base | Size |
|---|---|---:|
| `Qwen3.8-27B-NariQ-3.0bpw-mtp.gguf` | NariQ 3.0 bpw | 12.9 GB |
| `Qwen3.8-27B-NariQ-2.5bpw-mtp.gguf` | NariQ 2.5 bpw | 11.2 GB |

For vision, use `Qwen3.8-27B-NariQ-mmproj-BF16.gguf` from the original repo.

## Engine

The [nari-labs llama.cpp fork](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF) (tested with the Windows CUDA 12.8 build `b12431-nariv0`). NariQ kernels need Ampere or newer.

```
llama-server -m Qwen3.8-27B-NariQ-3.0bpw-mtp.gguf -c 65536 -fa on -ngl all -sm none -mg 0 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ot "nextn\.embed_tokens=CPU" \
  --mmproj Qwen3.8-27B-NariQ-mmproj-BF16.gguf \
  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0
```

`-ot "nextn\.embed_tokens=CPU"` keeps the added embedding table (~1 GB) in system RAM, where its lookups are cheap. Placing the MTP layer on a second GPU with `-ot blk\.64\.=CUDA1` is refused by this engine.

## Measured (RTX A4500 20 GB, model alone on it, vision projector on a second GPU)

| Setup | Decode tok/s | MTP acceptance |
|---|---:|---:|
| 3.0 bpw, no MTP (original file) | 29.3 | - |
| 3.0 bpw + MTP, n-max 2 | 46.7 | 79% |
| **3.0 bpw + MTP, n-max 3** | **53.0** (code 61 / chat 45) | 72% |
| 3.0 bpw + MTP, n-max 4 | 53.0 | 64% |
| 2.5 bpw + MTP, n-max 3 | 53.1 | 72% |
| 2.5 bpw + MTP, n-max 3, 128k ctx | 52.3 | 72% |

The local hard suite (72 tasks: coding, reasoning, instruction following, search, agent, vision, long context; greedy decoding) gave the **same scores with and without the grafted layer**, so the graft does not change the model's answers:

| File | Overall | Coding | Decode during the suite |
|---|---:|---:|---:|
| Original `NariQ-3.0bpw.gguf` | 85.0 | 75.0 (12/16) | 27.4 tok/s |
| `NariQ-3.0bpw-mtp.gguf`, MTP 3 | 85.0 | 75.0 (the same 4 tasks missed) | **56.6 tok/s** |

With reasoning on (thinking enabled, 5x token budget) the 3.0 bpw + MTP file scored **91.3** at 51.5 tok/s: instruction following 50 → 100, vision 88 → 100, search 88 → 100. Coding drops to 44 because most coding answers are still reasoning when the budget ends and never write the code (the same happens to full Qwen3.8-27B): keep thinking off for coding, turn it on for instruction-heavy or agent work.

For comparison, the 2.0 bpw original scored 76.1, and unsloth Qwen3.8-27B scored 90.8 (UD-Q4_K_M) and 80.4–82.6 (UD-Q5_K_XL, two builds) on the same suite.

262k context with MTP does not fit on a 20 GB card.

## Reproduce

[`mtp_graft.py`](mtp_graft.py) (no dependencies):

```
python mtp_graft.py Qwen3.8-27B-NariQ-3.0bpw.gguf Qwen3.8-27B-UD-Q5_K_XL.gguf Qwen3.8-27B-NariQ-3.0bpw-mtp.gguf --head
```

Tested in the [DualGPU-V100-A4500](https://github.com/Abdelhayali/DualGPU-V100-A4500) project.

## License and credits

Apache-2.0, as both sources are.

- **NariQ quantization:** [nari-labs](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF).
- **MTP layer, LM head and embeddings:** [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).
- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
