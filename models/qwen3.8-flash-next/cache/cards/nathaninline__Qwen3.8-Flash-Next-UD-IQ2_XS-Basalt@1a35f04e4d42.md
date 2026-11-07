---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
base_model:
- Qwen/Qwen3.8-Flash-Next
- unsloth/Qwen3.8-Flash-Next-GGUF
tags:
- gguf
- basalt
- moe
- quantized
- unsloth
---

# Qwen3.8-Flash-Next UD-IQ2_XS (L4 fix) for Basalt

Unsloth's **UD-IQ2_XS** (Dynamic v3.1) quant of Qwen3.8-Flash-Next, with **one tensor changed** so that the
[Basalt](https://github.com/jesdga95/basalt) engine keeps its fast decode path, plus a ready-to-run `.basalt` pack.

On a single RTX 5060 Ti 16 GB it decodes at ~84 tok/s on average and reads prompts at 1,570 to 2,280 tok/s, with a
perplexity within 0.6% of the BF16 model.

## Files

| File | Size | For |
|---|---:|---|
| `Qwen3.8-FN-UD-IQ2_XS-L4fix.basalt` | 74.3 GiB | Basalt (`.basalt` format version 2, engine 0.3+) |
| `gguf/Qwen3.8-Flash-Next-UD-IQ2_XS-L4fix-0000{1,2,3}-of-00003.gguf` | 75.7 GB | llama.cpp builds / engines that read the `qwen4exp` GGUF |

## What was changed

Basalt's device-planned MoE tail (and route-tail-skip, which depends on it) only runs when **every** layer's routed
experts use a format its kernels cover. In Unsloth's UD-IQ2_XS, layer 4's `ffn_down_exps` is **Q5_0**, which those kernels
do not cover (down formats covered: IQ4_NL, Q2_0, NVFP4, Q5_1, Q8_0). That single layer turns the fast path off for the
whole model: missed experts are then read by the GPU over PCIe, and decode drops by about 25%.

This repository requantizes **only `blk.4.ffn_down_exps.weight`: Q5_0 -> IQ4_NL** (5.5 -> 4.5 bits per weight). The other
1,223 tensors are copied unchanged (same type and byte size, checked tensor by tensor). The `.basalt` pack was then built
with Basalt's `tools/convert/convert.py --compat-bf16 --sections-from` the official `GSQ-RCO-IQ3_XXS` pack, so the MTP
drafter and the vision projector are the official Basalt sections, byte for byte.

The engine log confirms the fast path: every layer reads `decode device-planned tail + grouped kernels` and the PCIe share
of routed experts is 0%.

## Quality

Perplexity on wikitext-2 (test), 100 chunks of 512 tokens, scored through Basalt's decode path (`tools/ppl_gate.py`,
T 3, int8 KV). This is the same method as the official Basalt quality table. Lower is better.

| Model | Size | PPL | vs BF16 |
|---|---:|---:|---:|
| BF16 (official Basalt figure) | 354 GB | 3.273 | - |
| **UD-IQ2_XS L4fix (this repo)** | 74.3 GiB | **3.293 ± 0.041** | **+0.6%** |
| UD-Q4_K_XL (official Basalt figure) | 106.7 GiB | 3.321 | +1.5% |
| GSQ-RCO IQ3_S (official Basalt figure) | 81.0 GiB | 3.383 | +3.4% |
| GSQ-RCO IQ3_XXS (measured here) | 74.4 GiB | 3.472 ± 0.046 | +6.1% |

The GSQ-RCO IQ3_XXS measured here (3.472) matches its official figure (3.469), which checks the setup. The GGUF itself
reads PPL 3.2803 ± 0.041 in `llama-perplexity` (same text and chunking).

For task-level results, see Unsloth's charts on the
[Qwen3.8-Flash-Next guide](https://unsloth.ai/docs/models/qwen3.8-next): average recovery over 12 benchmarks of 99.4% for
UD-IQ2_XS against 96.8% for GSQ-RCO IQ3_XXS. Those were measured on Unsloth's original file, not on this one.

The perplexity of the one requantized layer was not measured separately; the whole-model figures above include it.

## Speed

One RTX 5060 Ti 16 GB (PCIe 4.0 x8, 14 GB/s host to device), Ryzen 9 3900X, 46 GB DDR4, Linux, Basalt 0.3
(`d10ccd4bc`), context 131,072 with int8 KV, vision on, nothing else on the card.

Decode, 600 tokens per request, server counter:

| Prompt | tok/s | Drafts accepted | VRAM expert hits |
|---|---:|---:|---:|
| Story | 75.3 / 75.4 | 58% | 82-84% |
| Python code | 97.6 / 96.0 | 74-78% | 71% |
| Explanation | 77.5 / 81.4 | 58-62% | 78-79% |
| **Mean** | **~84** | | |

Prompt reading (fresh prompt, two runs):

| Prompt | tok/s | Time to first token |
|---|---:|---:|
| ~2K tokens | 1,571 | 1.4 s |
| ~11K tokens | 2,109 / 2,119 | 5.3 s |
| ~25K tokens | 2,089 / 2,279 | 11-12 s |
| ~85K tokens | 2,200 / 2,206 | 38.5 s |

Before the fix, the same Unsloth quant on the same machine decoded at 61-75 tok/s (mean ~65), with 4-8% of routed
experts going over PCIe.

## Running it with Basalt

Basalt 0.3 or later (`.basalt` version 2). Example config:

```json
{
  "version": 2,
  "engine": { "expert-cache": "auto", "prefill": "auto", "ram-reserve-mib": 2048 },
  "models": [
    { "model": "Qwen3.8-FN-UD-IQ2_XS-L4fix.basalt", "gpu": "0", "kv": "int8",
      "max-context": 131072, "vision": "dynamic" }
  ],
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

```sh
basalt serve --config config.json
```

On a 46 GB machine, about 37.8 GiB of experts stay in page-locked RAM. On less RAM, set `resident-budget-gib` lower and
expect some experts to be read from the SSD. Leave room for Basalt's adaptive swaps: a `resident-budget-gib` that fills all
the RAM it is given leaves no free places for them and makes decode slower, not faster.

## Credits and license

- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team.
- Quantization: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), UD-IQ2_XS (Dynamic v3.1).
- Engine, pack format, MTP drafter and vision sections: [Basalt](https://github.com/jesdga95/basalt) and
  [jesdga/Qwen3.8-Flash-Next-Basalt](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt).

All files here are derived from Qwen3.8-Flash-Next and are distributed under the **Qwen Community License 1.0**, the
license of the base model. Use of these files is subject to its terms.
