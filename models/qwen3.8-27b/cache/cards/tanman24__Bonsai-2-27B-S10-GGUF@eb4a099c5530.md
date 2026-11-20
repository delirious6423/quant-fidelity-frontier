---
license: apache-2.0
base_model:
- prism-ml/Ternary-Bonsai-2-27B-gguf
- Qwen/Qwen3.8-27B
tags:
- gguf
- ternary
- mtp
- speculative-decoding
- android
- postmarketos
- edge
pipeline_tag: text-generation
datasets:
- HuggingFaceH4/ultrachat_200k
- ise-uiuc/Magicoder-Evol-Instruct-110K
- grammarly/coedit
---

# Ternary Bonsai 2 27B for phones (condensed vocabulary + MTP head)

GGUF files of [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (PTQ1_0, Prism ML),
prepared for a Samsung Galaxy S10 (Exynos 9820, 8 GB RAM) running postmarketOS:

| File | Size | What it is |
|---|---|---|
| `Bonsai2-27B-PTQ1_0-Condensed-P2TG.gguf` | 5.70 GB | Bonsai 2 27B with a condensed vocabulary (242 MB smaller) |
| **`Bonsai2-27B-PTQ1_0-Condensed-P2TG-mtp-v2.gguf`** | 5.94 GB | **Recommended.** The same plus a multi-token-prediction (MTP) head retrained on Bonsai's own answers, for speculative decoding (2026-11-20) |
| `Bonsai2-27B-PTQ1_0-Condensed-P2TG-mtp.gguf` | 5.94 GB | First MTP version: Qwen3.8-27B's head as is. Kept because published numbers refer to it |

Both need **[llama.cpp-s10](https://github.com/tannerbruhn/llama.cpp-s10)** (Prism ML's llama.cpp with PTQ1_0, plus the
phone kernels and the MTP support used here), release
[s10-b12179-d480e84](https://github.com/tannerbruhn/llama.cpp-s10/releases/tag/s10-b12179-d480e84) or newer for the
numbers below (pre-built for postmarketOS aarch64). Upstream llama.cpp cannot load PTQ1_0.
Step-by-step phone setup: [s10-llm-recipes](https://github.com/tannerbruhn/s10-llm-recipes).

## Speed on a Galaxy S10

4 CPU cores (2 Samsung M4 + 2 Cortex-A75) plus the Mali-G76 GPU, greedy, 128-token answers, with airflow:

| | prose | code | edits |
|---|---|---|---|
| P2TG, no MTP | ~1.3 tok/s | ~1.3 | ~1.2 |
| P2TG-mtp, MTP-1 (first head) | 1.46 | 1.51 | 1.64 |

MTP-1: the head guesses one token, and the model checks it with the next one in a single 2-token pass.

**v2 head (P2TG-mtp-v2):** in an A/B on the phone (6 mixed prompts, same settings, order old/new/new/old) it needs
3.1% fewer passes and runs **+3.4% faster** with thermal pauses factored out (+2.3% raw). A pass costs the same
with either head. The answers are identical: the model checks every guess, the head only changes how many are right.

| Guesses accepted (greedy) | first head | v2 head |
|---|---|---|
| prose | 57% | 64% |
| code | 89% | 96% |
| short edits | 100% | 100% |
| larger set (chat, code, edits) | 74.9% | 83.0% |

The ceiling for MTP-1 is 2 tokens per pass; v2 averages 1.83. The phone settings that make this fit in 8 GB: 4096 context,
8-bit KV cache (`-ctk q8_0 -ctv q8_0 -fa on`), and the fork's replay rollback (on by default). The recipes have the
full command.

## What was changed

**Condensed vocabulary (P2TG).** The tokenizer keeps 142,744 of the original tokens: Latin-script text, code,
common typography (curly quotes, dashes, arrows, math signs) and Greek. Only `token_embd.weight` and
`output.weight` change shape; the other 849 tensors are byte-identical to Prism ML's file. Wikitext-2 perplexity:
6.948 ± 0.276 against 6.815 ± 0.268 for the original (+1.9%, inside the error bars; the difference comes from a
few CJK/Burmese characters that fall back to bytes). **Text in scripts outside the kept set (CJK, Cyrillic,
Arabic, ...) is still possible but needs more tokens and works worse.** Use the original file for those languages.

**MTP head (P2TG-mtp, first version).** Prism ML ships Bonsai 2 without the MTP head; its base model Qwen3.8-27B has one. The 15
`mtp.*` tensors from Qwen3.8-27B were converted with llama.cpp's own rules (checked bit-identical against an
independent conversion of Qwen3.8-27B) and appended as block 64; the matrices are stored in Q4_0. The head was
trained for Qwen3.8, not for Bonsai's ternary weights, so acceptance is lower than on Qwen3.8 itself, but it still
pays everywhere: most for edits, least for prose. Without `--spec-type draft-mtp` the file behaves exactly like
the P2TG file.

**MTP head v2 (P2TG-mtp-v2).** The same 15 `mtp.*` tensors, fine-tuned for Bonsai: Bonsai 27B answered ~4,300
prompts (chat from ultrachat_200k, code from Magicoder-Evol-Instruct-110K, edits from grammarly/coedit), and the head
was trained to predict Bonsai's own next tokens. Only the MTP tensors differ from P2TG-mtp (still Q4_0); the other
tensors are byte-identical. Same llama.cpp release and settings as the first version.

## Checksums (SHA-256)

```
66c692ff5e7934d4fdbff582c7ffa80d16d35e24bbf2a6f3e3d23d9edcf75bf0  Bonsai2-27B-PTQ1_0-Condensed-P2TG.gguf
0b32e1de70aab4d92963f5974ec74ad1dca88a33f58f2824870574423ea17229  Bonsai2-27B-PTQ1_0-Condensed-P2TG-mtp.gguf
25fb17ecb2bac52f3d3d2f223c32ad337faf9ea0c2553e07dac6ee883e4cc297  Bonsai2-27B-PTQ1_0-Condensed-P2TG-mtp-v2.gguf
```

## License and attribution

Apache 2.0, as the originals. Created using Bonsai by Prism ML. Built from Qwen3.8-27B, Copyright 2026 Alibaba
Cloud (Apache 2.0). Changes: vocabulary condensed; Qwen3.8-27B's MTP head added; in v2 that head fine-tuned on Bonsai's
own answers to prompts from ultrachat_200k (MIT), Magicoder-Evol-Instruct-110K and grammarly/coedit (Apache 2.0). See `LICENSE` and `NOTICE.txt`.
