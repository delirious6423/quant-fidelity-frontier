---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
language:
- en
- ru
library_name: mlx
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
datasets:
- SWE-bench/SWE-smith-trajectories
- open-thoughts/OpenThoughts-114k
- open-r1/OpenR1-Math-220k
- bigcode/self-oss-instruct-sc2-exec-filter-50k
- open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled
- nvidia/OpenCodeReasoning
- allenai/tulu-3-sft-mixture
- HuggingFaceM4/ChartQA
- HuggingFaceM4/DocumentVQA
- agentsea/wave-ui-25k
- lmms-lab/textvqa
- detection-datasets/coco
tags:
- mlx
- omlx
- oq
- reap
- expert-pruning
- dwq
- m4q
- quantization
- 3-bit
- mixed-precision
- mtp
- speculative-decoding
- qwen3.8
- qwen4_exp
- moe
- image-text-to-text
- vision
- multimodal
---

# Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX

> ### Flash-Next on a 48 GB Mac
> A **REAP-320 expert-pruned `oQ3e`** (3-bit experts, enhanced mixed precision) build of Qwen3.8-Flash-Next,
> distilled **end-to-end against the bf16 model's own logits** in three KL-DWQ rounds (v7 → v9).
> The DWQ rounds cut the held-out teacher gap by **14.9% (CE) / 11.2% (KL20)** over the pruned
> initializer and recovered **33.6%** of the pruning gap, while keeping the bf16 **vision tower**
> and a REAP-matched native **MTP** speculative-decoding head.

A mixed-precision **MLX build** of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — **vision + text** —
with **320 of 512 routed experts** kept per MoE layer, 3-bit routed experts, 5/6/8-bit sensitive
trunk tensors and an 8-bit lm_head, sized to run in **oMLX on a 48 GB Apple Silicon Mac**.

> ⚠️ These are pruned and quantized weights. **The model's capability comes from Qwen's base
> model** — please star/cite it first. This repo contributes the expert pruning, the mixed-precision
> layout, the end-to-end DWQ distillation and MTP/vision packaging; DWQ tunes quantizer
> scales/biases and does not add new knowledge. Pruning 192 experts per layer is lossy.

## Model lineage

```
Qwen/Qwen3.8-Flash-Next                   (Qwen Community License 1.0 · MoE 512 experts · vision + MTP)
  ├─ Jundot/Qwen3.8-Flash-Next-oQ4e-mtp   (oMLX oQ4e donor checkpoint)
  ├─ dfp-official/Qwen3.8-Flash-Next-oQ8e-mtp (8-bit expert codes)
  └─ THIS REPO: REAP 512→320 → oQ3e: 3-bit experts (M4Q init) → end-to-end KL-DWQ v7→v8→v9
               + REAP-matched MTP + bf16 vision
       └─ Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MTPLX (same weights, MTPLX pack)
```

- **Architecture:** `qwen4_exp`, 48 layers — 36 Gated-DeltaNet (linear attention) + 12 full-attention
  layers with a sparse QSA indexer; MoE with **320 routed experts per layer (top-10)** plus a shared
  expert; hyper-connections; a 3-gram PLE embedding table; architectural context 262,144 tokens.
- **Vision tower:** kept in **bf16** (333 `vision_tower.*` tensors).
- **MTP head:** one native multi-token-prediction layer for oMLX speculative decoding, its experts
  pruned to the same 320 by weight-energy saliency (the drafter has no routing data; this can only
  affect draft acceptance, never output correctness).

## Quantization: REAP-320 → `oQ3e` (3-bit experts, M4Q init) → end-to-end KL-DWQ

1. **REAP expert pruning (512 → 320).** Per-layer keep sets built on sh0wie's REAP manifests; the
   kept expert ids ship as `reap_kept_experts.json`.
2. **3-bit routed experts.** Routed experts are re-quantized to affine 3-bit (group size 64) from the
   8-bit expert codes of the oQ8e checkpoint, with Hessian-guided **M4Q** initialization.
3. **End-to-end KL-DWQ against the bf16 teacher.** The bf16 model generated completions — including
   reasoning — with **top-20 logprobs for every token** (≈99.6% of the probability mass). The full
   resident student is trained through the whole network on KL over those logprobs; affine
   scales/biases are trainable, with gradient checkpointing and validation early stopping.
   **v9** doubled the data (5,377 text records / 11.75 M teacher tokens + 1,175 vision records) and
   ran 12 layer bands with gradient accumulation 2; every accept was re-confirmed on 24 unseen
   records — **6 of 12 bands passed** (cumulative −2.03% on the confirmation set). v9 also tried
   fp32 master weights for the lm_head/trunk and straight-through updates of the 3-bit expert codes;
   none survived the gates, so only the affine polish shipped.
4. **Component splice.** The bf16 vision tower is unchanged; the REAP-matched MTP head is carried.

| | |
|---|---|
| Experts | **320 / 512** routed experts per layer (REAP), top-10 routing |
| Scheme | 3-bit g64 routed experts (144 modules) · 4-bit g64 base (245 modules) · 217×8-bit, 95×6-bit, 174×5-bit trunk modules · 8-bit lm_head and token embedding · 4-bit g32 n-gram/PLE table |
| Precision | **133.7 B parameters, 4.24 bits/weight overall**: 56.5% of them 3-bit (the routed experts), 40.5% 4-bit (51.2 B of that the n-gram table), 2.7% 5/6/8-bit, 0.4% bf16. Excluding the n-gram table: 82.5 B weights at **3.76 bits/weight**. Every module's bits/group are stated explicitly in `config.json`. |
| Weight files | **70.8 GB** safetensors (incl. ~32 GB n-gram/PLE table, bf16 vision tower and MTP) |
| Memory | ≈36 GiB of weights resident on a 48 GB Mac; the ~32 GB n-gram table is memory-mapped |
| Teacher | bf16 Qwen3.8-Flash-Next, top-20 logprobs incl. reasoning tokens |
| Tooling | oMLX `oq` + M4Q + custom end-to-end KL-DWQ for `qwen4_exp` |

### Calibration mix (v9)

| Domain | Share | Prompts (all completions and logprobs come from the bf16 teacher) |
|---|---:|---|
| General calibration | 18% | SWE-smith, OpenThoughts, OpenR1-Math, self-OSS-Instruct, verifiable coding problems, OpenCodeReasoning, Tulu 3 |
| Reasoning | 16% | curated long-reasoning prompts |
| Knowledge QA | 16% | curated knowledge questions |
| Agent / tool use | 14% | curated agent tool-call trajectories |
| Vision | 14% | synthetic renders + real images from ChartQA, DocumentVQA, WaveUI-25k, TextVQA and COCO train splits |
| Code | 12% | curated code tasks |
| Russian chat | 10% | curated Russian conversations |

No benchmark test sets were used for calibration.

## Evaluation

### Teacher gap measured on this build (lower is better)

Held-out CE-gap | KL20 against the bf16 teacher (41-record table, same records for every round):

| Domain | v7 | v8 | **v9 (this repo)** | ΔKL20 v8→v9 |
|---|---|---|---|---:|
| Reasoning | 0.0629 \| 0.1098 | 0.0620 \| 0.1095 | **0.0596 \| 0.1077** | −1.7% |
| Code | 0.0460 \| 0.0616 | 0.0446 \| 0.0605 | **0.0437 \| 0.0595** | −1.7% |
| Russian chat | 0.1607 \| 0.1183 | 0.1585 \| 0.1153 | **0.1561 \| 0.1136** | −1.5% |
| Knowledge QA | 0.0976 \| 0.1519 | 0.0966 \| 0.1507 | **0.0950 \| 0.1483** | −1.6% |
| Agent | 0.2308 \| 0.1921 | 0.2307 \| 0.1892 | **0.2235 \| 0.1854** | −2.0% |
| **Overall** | 0.0934 \| 0.1065 | 0.0921 \| 0.1052 | **0.0901 \| 0.1035** | **−1.6%** |

- **Independent gate** (17 held-out caches, 456,602 tokens): CE-gap 0.0986 → **0.0968**,
  KL20 0.0845 → **0.0833**.
- **Real-image vision** (120 held-out records, unseen images): CE-gap 0.2637 → **0.2533** (−3.9%),
  KL20 0.2342 → **0.2245** (−4.2%).
- **Against the pruned initializer:** CE-gap −14.9%, KL20 −11.2%; 33.6% of the 0.047-nat gap to the
  unpruned 512-expert 3-bit reference recovered.

These numbers measure fidelity to the bf16 teacher on held-out calibration distributions. They are
**not task accuracy**. No task or vision-benchmark score is claimed here yet.

## Repos in this family

| Repo | Runtime | Safetensors | Use case |
|---|---|---:|---|
| **`Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX`** (this) | oMLX | **70.8 GB** | oMLX layout, native MTP + vision |
| [`Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MTPLX`](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MTPLX) | MTPLX | **71.6 GB** | same weights repacked for MTPLX + a self-distilled MTP sidecar |

## How to run

These are **MLX weights for Apple Silicon**. The tested serving path is
**[oMLX](https://omlx.app)** (Qwen3.8-Flash-Next / `qwen4_exp`, vision input, native MTP).

```bash
# Download directly into the oMLX model directory
hf download Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX \
  --local-dir ~/.omlx/models/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX

# Enable native MTP once
curl -X PUT \
  http://127.0.0.1:8003/admin/api/models/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX/settings \
  -H "Authorization: Bearer sk-local" -H "Content-Type: application/json" \
  -d '{"mtp_enabled": true}'

# OpenAI-compatible multimodal API
curl -X POST http://127.0.0.1:8003/v1/chat/completions \
  -H "Authorization: Bearer sk-local" -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX",
       "messages":[{"role":"user","content":[
         {"type":"image_url","image_url":{"url":"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"}},
         {"type":"text","text":"Describe the image and explain the visual clues."}
       ]}],
       "max_tokens":4096,"temperature":1.0,"top_p":0.95,"top_k":20}'
```

The oMLX model id is **case-sensitive** and matches the downloaded folder name.

### Recommended sampling

The shipped `generation_config.json` defaults: **temperature 1.0, top_p 0.95, top_k 20**, thinking
enabled. Give hard reasoning/code tasks a generous output budget.

## Intended use & limitations

- **Best suited to:** coding, tool/agent workflows, reasoning and multimodal chat on 48 GB Macs.
- **Pruning is lossy:** 192 of 512 experts per layer were removed. Domains the keep sets and
  calibration cover less may degrade more than the teacher-gap tables suggest.
- **DWQ is calibration, not SFT:** it improves fidelity to the teacher on represented activations; it
  does not teach facts or guarantee benchmark gains.
- **Vision:** the tower is bf16 and unchanged; vision data is 14% of the DWQ mix and no vision
  benchmark score is claimed.
- **MTP head:** pruned by weight saliency, so draft acceptance is below the full 512-expert drafter;
  for a sidecar self-distilled on this model's own traces, see the MTPLX repo.
- **Long context costs memory:** the 262k architectural maximum is not a promise that a 48 GB machine
  can hold the corresponding KV state.
- **MLX-only:** not loadable by vanilla `transformers`.

## Acknowledgements

- **Qwen team** — Qwen3.8-Flash-Next (Qwen Community License 1.0).
- **Jundot** — `Qwen3.8-Flash-Next-oQ4e-mtp`, the oQ4e donor checkpoint.
- **dfp-official** — `Qwen3.8-Flash-Next-oQ8e-mtp`, the 8-bit expert codes.
- **sh0wie** — REAP keep-set manifests (`Qwen3.8-Flash-Next-REAP-288-MLX-4bit`).
- **REAP** — router-weighted expert activation pruning (Cerebras Research).
- **Apple MLX** and **oMLX** — `mlx`, `mlx-lm`, `mlx-vlm`, enhanced `oq` quantization, M4Q and the
  MTP + vision serving runtime.
- **nick** ([@nmukhanov](https://huggingface.co/nmukhanov)) — diagnosed the per-layer
  quantization-key mismatch behind [PR #1](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX/discussions/1): the tree-path aliases in
  `config.json` are his find.
- **Calibration-data authors** — SWE-bench/SWE-smith, OpenThoughts, Open-R1, BigCode, NVIDIA
  OpenCodeReasoning, AllenAI Tulu, and the ChartQA, DocumentVQA, WaveUI-25k, TextVQA and COCO teams.

## License

**Qwen Community License 1.0**, inherited from `Qwen/Qwen3.8-Flash-Next`; a copy is included as
[`LICENSE`](LICENSE). Note its conditions — in particular, operating a *Model as a Service* or an
*AI Work Assistant* business commercially requires a separate license from Qwen.

## Citation

Please cite the original base model:

```bibtex
@misc{qwen3.8-flash-next,
  title  = {Qwen3.8-Flash-Next},
  author = {Qwen Team, Alibaba Group},
  year   = {2026}
}
```
