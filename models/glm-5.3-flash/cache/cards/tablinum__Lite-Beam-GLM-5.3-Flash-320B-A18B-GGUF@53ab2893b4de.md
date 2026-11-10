---
license: mit
license_link: LICENSE
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
language:
- en
- zh
tags:
- lite-beam
- tablinum
- glm
- glm-5
- mixture-of-experts
- moe
- gguf
- llama.cpp
- quantized
- 2-bit
- imatrix
- mtp
- agentic
- coding
pretty_name: Lite-Beam GLM-5.3-Flash 320B A18B
---

<img src="assets/hero.gif" alt="Lite-Beam GLM-5.3: a 96.1 GB mixed 2-bit GGUF of GLM-5.3-Flash, 320B total, 18B active, all 288 experts retained" width="100%">

# Lite-Beam GLM-5.3-Flash

**GLM-5.3-Flash, Z.ai's 320B mixture-of-experts model, in a 96.1 GB file.** This is a mixed 2-bit GGUF that keeps **all 288
experts in every MoE layer**, plus the multi-token-prediction (MTP) layer for faster generation. It spends its bytes where a
2-bit model breaks first: attention, the router and the expert down-projections. It is **12.6 GB smaller than unsloth's
UD-Q2_K_XL** and, on our rig, **about 8–10% faster** at both prompt processing and generation.

The trade is fidelity. UD-Q2_K_XL stays measurably closer to the reference model (mixed chat/code/math KLD **0.067 vs 0.080**),
and it wins most windows on both test sets. If it fits your memory, it is the more accurate 2-bit file. This one is for machines
where 12.6 GB decides between a model that fits with room for context and tools and one that doesn't, and for anyone who wants
the faster file.

Text only: the base model's vision tower is not included. MIT-licensed, like the base model.

**Quick links:** [Files](#files) · [Run locally](#run-locally) · [How it was made](#how-it-was-made) ·
[Against UD-Q2_K_XL](#against-unsloths-ud-q2_k_xl) · [Base model](https://huggingface.co/zai-org/GLM-5.3-Flash) ·
[Lite-Beam family](https://huggingface.co/tablinum)

## The result

<img src="assets/results.png" alt="Scorecard against unsloth UD-Q2_K_XL: Lite-Beam is smaller (96.1 vs 108.7 GB) and faster (63.5 vs 57.9 t/s prompt, 8.07 vs 7.60 t/s MTP generation); UD is closer to the reference (mixed KLD 0.067 vs 0.080, WikiText 0.347 vs 0.414). Exact values follow in the text tables." width="100%">

| build · file size | mixed chat/code/math: KLD · same top token | WikiText: KLD · same top token |
|---|---:|---:|
| **Lite-Beam IQ2_XXS · 96.1 GB** | 0.080 ± 0.002 · 92.3% | 0.414 ± 0.007 · 78.3% |
| unsloth UD-Q2_K_XL · 108.7 GB | **0.067 ± 0.002 · 92.8%** | **0.347 ± 0.006 · 80.2%** |
| Q3_K_M reference · 153.7 GB | reference | reference |

**Speed on the reference rig** (experts in DRAM, attention on two small GPUs, 20 threads pinned to one CPU socket, same
settings for both files):

| | Lite-Beam IQ2_XXS | UD-Q2_K_XL |
|---|---:|---:|
| prompt processing, `llama-bench` pp2048 (`-ub 2048`) | **63.5 t/s** | 57.9 t/s |
| generation, `llama-bench` tg128 | **6.74 t/s** | 6.24 t/s |
| generation with MTP drafting (`--spec-draft-n-max 2`), code prompt | **8.07 t/s** | 7.60 t/s |
| generation with MTP drafting (`--spec-draft-n-max 2`), chat prompt | **7.30 t/s** | 6.70 t/s |

The MTP numbers are `llama-server` runs of 384 generated tokens with thinking on. Lite-Beam's drafter acceptance was 82%
(code) and 69% (chat) at two drafted tokens; with one drafted token it was 92% and 80% (7.72 and 7.26 t/s). Without drafting,
both files generate at about 6.2 t/s in the server.

**How to read KLD.** KL divergence measures how far each build's next-token probabilities drift from the reference model's; lower
is better. "Same top token" is how often both pick the same most likely next token. Measured with `llama-perplexity
--kl-divergence` on 18 windows of 2,048 tokens of mixed chat, code and math, and 15 windows of WikiText, with the experts going
through the GPU the way llama.cpp runs them by default.

> **About the reference.** Both builds are scored against a **Q3_K_M** of GLM-5.3-Flash, not the 642 GB BF16. On our machine the
> BF16 could not be read fast enough for scoring: each window touches nearly every expert, and 642 GB does not fit in 156 GB of
> RAM. Both builds are measured against the same reference, so the ranking is fair, but absolute KLD values will read lower than
> they would against BF16.

## Files

| file | size | use |
|---|---:|---|
| `Lite-Beam-GLM-5.3-Flash-320B-A18B-IQ2_XXS-00001-of-00003.gguf` | 47.3 GB | llama.cpp (load this one; it finds the other shards) |
| `Lite-Beam-GLM-5.3-Flash-320B-A18B-IQ2_XXS-00002-of-00003.gguf` | 47.7 GB | second shard |
| `Lite-Beam-GLM-5.3-Flash-320B-A18B-IQ2_XXS-00003-of-00003.gguf` | 1.1 GB | third shard |
| `quantization_manifest.json` | | exact tensor map, measurements, per-window KLD, SHA-256 |
| `LICENSE` | | MIT license of the base model |

Total: 96,143,131,264 bytes (96.1 GB, 89.5 GiB), 2.40 bits per weight. The file is split because Hugging Face caps single files
at 50 GB.

## Run locally

```bash
hf download tablinum/Lite-Beam-GLM-5.3-Flash-320B-A18B-GGUF --include "*IQ2_XXS*" --local-dir lite-beam-glm-5.3

llama-server -m lite-beam-glm-5.3/Lite-Beam-GLM-5.3-Flash-320B-A18B-IQ2_XXS-00001-of-00003.gguf \
  -ngl 99 --n-cpu-moe 99 -fa on -c 32768 -b 2048 -ub 2048 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --temp 1.0 --top-p 0.95
```

- `-ngl 99 --n-cpu-moe 99` puts attention, norms and the router on the GPU and leaves the routed experts in system RAM. That is
  the layout this build was made for. With two GPUs, add `-ts` to split the attention between them.
- `--spec-type draft-mtp` uses the model's own MTP layer as a drafter: no second model to download. Drop the two `--spec-*` flags
  to turn it off. Use `--spec-draft-n-max 1` if you mostly chat; 2 is best for code.
- `-ub 2048 -b 2048` matters more than anything else for prompt processing: each micro-batch copies the experts it needs to
  the GPU, so bigger batches spread that cost over more tokens (on our rig, `-ub 2048` is about 3× faster than `-ub 512`).
  Larger needs more than a 6 GB card for compute buffers.
- The sampling settings are Z.ai's recommendation. GLM-5.3-Flash always thinks; its chat template takes a
  `reasoning_effort` of `low`, `high` or `max` (the default). Set it per request with
  `"chat_template_kwargs": {"reasoning_effort": "low"}`, or for the whole server with
  `--chat-template-kwargs '{"reasoning_effort":"low"}'`.

## What you need

- At least **104 GB of combined RAM and VRAM** for the weights and a short context. **128 GB or more** is comfortable with a
  32K context.
- A GPU for attention and the KV cache (experts stay in RAM) is the recommended split. The reference rig uses 12 GB + 6 GB cards.
- A **llama.cpp** build with GLM-5.3-Flash (`glm5-next`) support. Tested with build `954de24`.
- Reference rig: 2 × Xeon Gold 6248, 156 GB DDR4, RTX A2000 12 GB + RTX 2060 6 GB.

## How it was made

Same recipe as every Lite-Beam build:

1. **From the official weights.** Z.ai's FP8 release, converted to a 642 GB BF16 GGUF, quantized from BF16.
2. **Our own importance matrix.** 128 windows of 2,048 tokens from Tablinum's calibration corpus (mixed chat, code, math and
   reference text), not the llama.cpp default. Computed on the Q3_K_M reference for the same reason it is the KLD reference.
3. **A per-tensor map** instead of one quant type for everything:

   | tensors | type | why |
   |---|---|---|
   | routed experts, gate and up (`ffn_gate_exps`, `ffn_up_exps`) | IQ2_XXS | most of the bytes; the imatrix protects the columns that matter |
   | routed experts, down (`ffn_down_exps`) | IQ2_S | one step up: errors here land directly in the residual stream |
   | MTP layer experts (`blk.45`) | Q3_K | the imatrix has no data for the MTP layer, so it gets a safer type; keeps drafts accepted |
   | attention q, k, v, output; shared expert; the 3 dense layers | Q6_K | small and always active, so cheap to keep sharp |
   | MLA projections, sparse-attention indexer, linear-attention (KDA) projections, MTP `eh_proj` | Q8_0 | small, and errors here compound over long contexts |
   | router (`ffn_gate_inp`), expert bias, hyper-connection mixers | F32 | a wrong route sends a token to the wrong expert entirely |
   | output head | Q6_K | |
   | token embeddings | IQ4_XS | |

4. **A KLD gate.** Every build is scored on the same held-out sets before anything ships, with the experts going through the GPU
   the way llama.cpp runs them by default. Both scores here are finite on every window. No expert tensor uses Q2_K, the type
   that turned our scores NaN on the GPU path in an earlier build.

### Against unsloth's UD-Q2_K_XL

We scored unsloth's UD-Q2_K_XL of GLM-5.3-Flash against the same reference, on the same windows and with the same GPU
offload.

| | Lite-Beam IQ2_XXS · **96.1 GB** | UD-Q2_K_XL · 108.7 GB |
|---|---|---|
| mixed: mean KLD | 0.0796 ± 0.0019 | **0.0674 ± 0.0015** |
| mixed: same top token | 92.34% | **92.84%** |
| mixed: 99th-percentile KLD | 1.01 | **0.91** |
| mixed: worst single token KLD | **7.34** | 7.66 |
| WikiText: mean KLD | 0.414 ± 0.007 | **0.347 ± 0.006** |
| WikiText: worst single token KLD | **9.98** | 10.93 |

**Where UD is ahead:** almost everywhere on fidelity. Paired by window, UD-Q2_K_XL is closer to the reference in 16 of 18 mixed
windows (sign test p = 0.001) and 13 of 15 WikiText windows (p = 0.007). On chat, code and math the gap is small in top-token
terms (92.3% vs 92.8%); on WikiText it is larger (78.3% vs 80.2%). UD spends its extra bytes on a bigger type for every routed
expert: IQ3_XXS down-projections (ours are IQ2_S, about 7.5 GB less) and IQ2_XS gate/up (ours are IQ2_XXS, about 6.6 GB less),
while spending less than we do on attention, the output head and the embeddings. On GLM-5.3-Flash those expert bytes buy real
accuracy.

**Where Lite-Beam is ahead:** size and speed. It is 12.6 GB smaller, reads fewer expert bytes per token, and is faster on every
speed test we ran (above). Its single worst tokens drift a little less on both sets.

**Which file?** With 128 GB or more and accuracy as the priority, take UD-Q2_K_XL. Take this file if the 12.6 GB matters (for
example, to keep a 64K+ context, an editor, browser and tools resident beside the model on a 128 GB machine) or you want the
faster generation. Full per-window numbers are in `quantization_manifest.json`.

## What's next

- **IQ3_XXS expert down-projections** (as UD uses) with our IQ2_XXS gate/up and attention choices, about 104 GB, to see how much
  of the fidelity gap the down-projections alone close while staying smaller than UD-Q2_K_XL.
- **Task evals** alongside KLD: coding and agentic benchmarks, and Tablinum-3 open-book retrieval.

## License and attribution

**Built with GLM-5.3-Flash.** This is a quantized derivative of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
and carries Z.ai's **MIT license** ([LICENSE](./LICENSE), unchanged). Copyright (c) 2026 Z.AI Co., Ltd. Quantization by
Tablinum. Character art: Beam by Tablinum. No endorsement by Z.ai is implied.

## Citation

```bibtex
@misc{litebeamglm53flash,
  title  = {Lite-Beam GLM-5.3-Flash 320B-A18B (IQ2_XXS)},
  author = {Lightspeed / tablinum},
  year   = {2026},
  note   = {Mixed 2-bit GGUF of GLM-5.3-Flash with all experts and the MTP layer retained},
  url    = {https://huggingface.co/tablinum/Lite-Beam-GLM-5.3-Flash-320B-A18B-GGUF}
}
```
