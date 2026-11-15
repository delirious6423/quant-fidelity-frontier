---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: tensorfold
tags:
- tensorfold
- nvfp4
- modelopt
- gptq
- rtx-pro-6000
- blackwell
---

# Qwen3.8-Flash-Next NVFP4 (GPTQ experts) for TensorFold

**NVFP4 quant of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by BertholomusAI.**
Built from revision `812d96e10a0dd82b69d749e68693698cf6c36daa`. Model and weights © Qwen under the
[Qwen Community License 1.0](LICENSE). Not affiliated with or endorsed by Qwen or TensorFold.

Made to run on **one RTX PRO 6000 Blackwell (96 GB)** with the full **262,144-token** context, served by the stock
[TensorFold](https://github.com/ashhart/TensorFold) 1.2.9 release binary (Apache-2.0). No engine changes.

Recipe (one-command start, agent install file, quality tools): to follow on GitHub.

## What is in it

| part | format |
|---|---|
| routed experts | NVFP4 (E2M1, FP8 scale per 16, FP32 tensor scale), GPTQ-calibrated, ModelOpt `MIXED_PRECISION` layout |
| attention, DeltaNet, shared experts, router, embeddings, LM head, MTP head, vision | BF16, unchanged |
| n-gram embedding table | NVFP4 rows (`ngram-*.safetensors`) |
| size | 109.1 GB |

Activations stay BF16 (weight-only W4A16).

## Measured on one RTX PRO 6000 Blackwell (96 GB, sm_120)

TensorFold 1.2.9 release (`tensorfold-1.2.9-linux-x86_64.tar.gz`, sha256 `d17ba88f…27187`), thinking off, greedy.

| row | result |
|---|---|
| decode, one stream (768 new tokens) | **152 tok/s** |
| prefill, 32,797-token prompt, cold | **6,702 tok/s** |
| context | **262,144** tokens |
| needle at 130K and at 250K tokens | **found / found** |
| greedy output repeats exactly | yes |
| 4 short smoke prompts (math, fact, code, exact JSON) | 4 / 4 |

### Several agents at once

`--parallel 8 --context 32768`, prompt cache on. Each agent shares an ~8K-token system prompt and sends 3 turns with
256-token replies. Total = all new tokens / wall time. 0 errors in every row.

| agents | total tok/s | per agent | first token p50 |
|---|---|---|---|
| 1 | 114 | 161 | 0.11 s |
| 4 | 315 | 92 | 0.12 s |
| 8 | **416** | 61 | 0.14 s |
| 16 (8 wait in line) | 370 | 50 | 5.6 s |

### Four agents at the full window (experimental)

`TF_FLASHNEXT_KV_PAGES=256 --parallel 4 --context 262144`: 4 agents each send a different ~241K-token document at the
same time. **4 / 4 needles found**, in 51-185 s. The streams share one page pool, so some requests wait for pages.
The prompt cache is off in this mode.

## Quality

KL divergence against the BF16 model, teacher-forced, per token, KL(BF16 || quant), held-out prose and code.
The n-gram table's error is included.

| check | this quant | plain round-to-nearest NVFP4 (same layout) |
|---|---|---|
| mean KL | **0.083** | 0.107 |
| top-1 token agreement | **89.3%** | 88.8% |
| perplexity (BF16 = 6.19) | **6.22** | n/a |

Calibration: WikiText-2 train articles and Python source, with zero documents shared with the evaluation set.

## Serve

```bash
tensorfold-native serve bertholomus/Qwen3.8-Flash-Next-NVFP4-GPTQ --context 32768 --parallel 8     # 8 agents
tensorfold-native serve bertholomus/Qwen3.8-Flash-Next-NVFP4-GPTQ --context 262144 --parallel 1    # one full window
```

- One full 262,144 window needs the default memory reserve; two full windows fit with `TENSORFOLD_MEMORY_RESERVE_GIB=1`.
- `TF_FLASHNEXT_KV_PAGES=256 --parallel 4 --context 262144` lets 4 streams share one page pool, so each can reach the
  full window when the others are shorter. TensorFold marks this experimental, and the prompt cache is off in that mode.
- Tested only with TensorFold on sm_120. Other engines and GPUs: not tested.

## Licence

The Qwen Community License 1.0 applies to these weights. Its clause 2: a licensee that runs a Model as a Service or an
AI Work Assistant business needs a separate licence from Qwen before commercial use; internal use that gives no third
party access is exempt. Read the [LICENSE](LICENSE) before you host this model for others.

## Credits

- Qwen team: base model and weights.
- TensorFold (Ash Hart and contributors, Apache-2.0): the engine and its NVFP4 / ModelOpt loader.
- GPTQ (Frantar et al., 2022): the calibration method.
- WikiText-2 (CC BY-SA 4.0) and exllamav3 Python source (MIT, turboderp): calibration text only, not redistributed.
