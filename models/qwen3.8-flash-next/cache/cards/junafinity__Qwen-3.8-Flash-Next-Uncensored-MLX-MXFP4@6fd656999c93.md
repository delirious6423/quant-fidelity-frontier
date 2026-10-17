---
license: other
license_name: qwen-community-license
license_link: LICENSE
base_model:
  - orcarouter/Qwen3.8-Flash-Next-Uncensored
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: mlx
language:
  - en
  - zh
tags:
  - mlx
  - mxfp4
  - 4-bit
  - apple-silicon
  - abliterated
  - uncensored
  - qwen
  - qwen4
  - qwen3.8
  - flash-next
  - moe
  - vision-language
  - function-calling
  - reasoning
  - mtp
---

# Qwen-3.8-Flash-Next-Uncensored-MLX-MXFP4

A mixed-precision **MXFP4** MLX conversion of **[orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored)** — OrcaRouter's full-precision BF16 abliterated (refusal-removed) build of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

This is **not** quantized from the official Qwen checkpoint and **not** from OrcaRouter's FP8 or stock MLX 4/6/8-bit releases. Every weight here was streamed from OrcaRouter's BF16 uncensored source.

The bulk format is native MLX **MXFP4 / group-32**, but this is deliberately **not** a uniform 4-bit quant. Routed expert gate/up tensors and the 51B PLE n-gram table take MXFP4; routed expert down-projections, attention, GDN, hyper-connections, shared experts, and vocabulary-boundary tensors stay at higher affine bit widths or BF16. The **full vision + video tower** stays BF16, and the **one-layer MTP head is kept** in the checkpoint.

| | |
|---|---|
| **Source (this quant)** | [`orcarouter/Qwen3.8-Flash-Next-Uncensored`](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored) (BF16, abliterated) |
| **Original architecture** | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`qwen4_exp`) |
| **Payload** | **100.53 GB** safetensors (93.63 GiB) across **20** shards |
| **Tensors** | **3245** (vision **333** BF16; MTP **66** including scales/biases; PLE **128** shards + scales; `skipped_div` **0**) |
| **Context** | 262,144 tokens |
| **Runtime** | [oMLX 0.6.3+](https://github.com/jundot/omlx) (vision + native MTP) or [mlx-vlm ≥ 0.6.17](https://github.com/Blaizzy/mlx-vlm) (vision; MTP keys are dropped on load in 0.6.17) |

---

## ⚠️ Disclaimer — read before use

This conversion inherits the **abliteration** of the OrcaRouter source. Safety alignment has been **substantially removed**. As a direct consequence:

- **It will comply with harmful, unethical, offensive, or illegal requests** that the original `Qwen3.8-Flash-Next` would refuse. It has no meaningful built-in guardrails.
- It is released **strictly for legitimate research** — interpretability, AI-safety and refusal-mechanism study, red-teaming, robustness evaluation, and controlled experiments.
- **You assume full responsibility and liability** for how you use it and for everything it generates. Do not deploy it to end users or in production without adding your own safety, moderation, and abuse-prevention layers.
- Use must comply with the **[Qwen Community License 1.0](LICENSE)** shipped with these files (and any terms of the OrcaRouter source), and all laws that apply to you.
- The authors and uploaders **accept no liability** for any misuse or harm. Outputs do **not** reflect the views of the uploaders, OrcaRouter, or Qwen / Alibaba.

By downloading or using this model you acknowledge and accept the above.

---

## What this is (and is not)

- **Derived from** OrcaRouter's **full-precision uncensored** BF16 checkpoint. Abliteration is a surgical residual-writer edit on that source (see their card); this repo only changes numeric format.
- **Mixed MXFP4**, not uniform 4-bit. Uniform 4-bit on this architecture is a known quality cliff ([PipeNetwork](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit): WikiText **+20.6%** PPL vs BF16). Mixed 4/8 on the official (non-uncensored) line is **+1.3%**. This recipe follows that mixed layout, using MXFP4 where group-32 is required (n-gram width 160; expert gate/up).
- **Not** Qwen's official block-128 FP8. Official FP8 is described as nearly identical to BF16; nobody has published MXFP4 vs that FP8 on this model. Treat quality as **unmeasured on this file**.
- **Not** OrcaRouter's `Qwen3.8-Flash-Next-Uncensored-MLX` 4/6/8-bit repo (those are larger affine MLX quants; fused experts / n-gram stay high precision, ~163 GB at “4-bit”).
- Independent conversion. Inspired by the tensor-priority profile used in community mixed MXFP4 MLX builds (e.g. [txgsync](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-MXFP4-MLX) on the *official* weights). Those omit MTP; **this file keeps MTP**.

---

## Quantization recipe

No sensitivity calibration / imatrix. Architecture-driven, reproducible from tensor names. Conversion streamed the 131 source shards and never materialized the ~360 GB BF16 model in RAM.

| Tensors | Format |
|---|---|
| Vision + video tower (`vision_tower.*`, 333 tensors) | **BF16**, byte-copied |
| `embed_tokens`, `lm_head` | **BF16** |
| Routers (`mlp.gate`, `shared_expert_gate`), norms, `A_log`, `dt_bias`, `layer_multipliers`, GDN `in_proj_a`/`in_proj_b`, QSA indexer, `conv1d`, `mtp.fc_*` / `mtp.pre_fc_norm` | **BF16** |
| Routed expert **gate / up** (`switch_mlp.gate_proj` / `up_proj`) | **MXFP4**, group 32 |
| Routed expert **down** (`switch_mlp.down_proj`) | affine **Q4**, group 64 |
| Shared expert gate / up | affine **Q5**, group 64 |
| Shared expert down | affine **Q8**, group 64 |
| QSA `q/k/v/o_proj` | affine **Q5**, group 64 |
| GDN `in_proj_qkv` / `in_proj_z` | affine **Q5**, group 64 |
| GDN `out_proj` | affine **Q6**, group 64 |
| Hyper-connection mix (`input_mix_weight_*`) | affine **Q8**, group 64 |
| `block_inject_weight` | affine **Q8**, group 128 |
| PLE `key_proj` / `value_proj` | affine **Q8**, group 64 |
| PLE n-gram embedding (128 shards, width 160) | **MXFP4**, group 32 |
| MTP block | **same family rules**, keys kept as `mtp.*` |

Default `quantization` in `config.json`: `{ "mode": "mxfp4", "group_size": 32, "bits": 4 }` plus per-module affine overrides. No RMSNorm +1 fold (mlx-vlm 0.6.17 zero-centered Qwen4 convention).

HF expert `gate_up_proj` / `down_proj` that split across source shards are stitched, then exported as MLX `switch_mlp.{gate,up,down}_proj`. N-gram keys are `ngram_embedding.shards.N`.

---

## Memory (128 GB unified Mac)

Fits a 128 GB machine, close to the ceiling.

| Component | Approx. |
|---|---|
| Weights | 93.63 GiB (100.53 GB) |
| Persistent cache at long context | several GiB (grows with length) |
| Runtime workspace | 4–8 GiB |
| Practical working set | ~103–110 GiB |

Close other apps. If Metal wired-memory is too low:

```bash
sudo sysctl iogpu.wired_limit_mb=124519
```

That relaxes the allocation ceiling; it does not add RAM. Start with a modest context.

---

## Runtime

**Preferred: [oMLX 0.6.3+](https://github.com/jundot/omlx/releases)** — add this folder as a local model, keep vision on, enable native MTP (draft depth 3). Images yes; video is not supported in 0.6.3.

**mlx-vlm ≥ 0.6.17** (architecture `qwen4_exp`):

```bash
pip install "mlx>=0.32" "mlx-vlm>=0.6.17"

python -m mlx_vlm.generate \
  --model junafinity/Qwen-3.8-Flash-Next-Uncensored-MLX-MXFP4 \
  --prompt "Prove that sqrt(2) is irrational." \
  --max-tokens 512
```

`mlx-vlm` 0.6.17 **strips `mtp.*` on load**. The tensors are in the shards for oMLX / future runtimes. LM Studio needs a `qwen4_exp` build and may refuse a ~100 GB folder.

Thinking-mode sampler (from the Qwen / OrcaRouter cards): temperature **1.0**, top-p **0.95**, top-k **20**, min-p **0.0**. Instruct / non-thinking: temperature **0.7**, top-p **0.80**, top-k **20**, presence penalty **1.5**.

---

## Quality

This checkpoint has **not** been scored with WikiText PPL, KLD, or the OrcaRouter eval suite. Structural conversion succeeded (`skipped_div = 0`; vision 333 with no quant scales; MXFP4 tensors have scales and no affine biases; affine tensors have scales + biases).

Related public numbers (different checkpoints — **do not treat as this file's score**):

- Qwen official **FP8** (block 128): “nearly identical” to BF16.
- PipeNetwork mixed affine **4/8** vs BF16 on *official* Flash-Next: WikiText **4.5286 vs 4.4708 (+1.3%)**. Uniform 4-bit: **+20.6%**. Their 8-bit is indistinguishable from BF16.
- Unsloth **uniform MXFP4** vs 8-bit on *Qwen3.6-27B* (different model): mean KLD 0.0479 vs 0.0028; PPL 4.902 vs 4.812. That uniform-MXFP4 KLD does **not** apply to this mixed recipe.

If you measure this file, a PR / discussion with PPL or KLD vs the OrcaRouter BF16 source would be useful.

---

## Abliteration (from the OrcaRouter BF16 source)

Unchanged by quantization. Summary of [orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored):

Refusal-direction removal following Arditi et al. (2024). A single direction `r` is taken from the block-input residual (Hyper-Connections GR-Read, layer 24; `harmful 0.00 / KL 0.085`) and orthogonalized out of residual-writing matrices: `W' = W − r(rᵀW)`.

| Component | matrices edited |
|---|---|
| `self_attn.o_proj` (12 full-attention layers + MTP) | 13 |
| `linear_attn.out_proj` (36 GDN layers) | 36 |
| `mlp.experts.down_proj` (512 experts × 49 layers) | 49 |
| `mlp.shared_expert.down_proj` | 49 |
| `ple.value_proj` + `embed_tokens` (row space) | 2 |
| **Total** | **149** |

Vision tower, routers, expert gate/up, hyper-connection mixers, QSA indexer, n-gram table, `mtp.fc_*`, norms, and `lm_head` were not abliterated. Max residual leakage after the edit: **0.0755**.

### Evaluation on the BF16 source (not this MXFP4 file)

OrcaRouter measured the **BF16 uncensored** weights vs official `Qwen/Qwen3.8-Flash-Next` with vLLM. Refusal = rule-based opening-phrase classifier. Thinking off unless stated.

**Harmful-prompt refusal, thinking OFF (lower = more uncensored)**

| Benchmark | n | Base | OrcaRouter BF16 | Caveat |
|---|---:|---:|---:|---:|
| AdvBench | 100 | 100.0% | 2.0% | 53.0% |
| JailbreakBench (harmful) | 100 | 94.0% | 0.0% | 52.0% |
| StrongREJECT | 150 | 99.3% | 1.3% | 62.0% |
| HarmBench (standard) | 150 | 100.0% | 1.3% | 50.7% |
| MaliciousInstruct | 100 | 98.0% | 0.0% | 55.0% |
| SimpleSafetyTests | 50 | 64.0% | 2.0% | 68.0% |
| ForbiddenQuestions | 150 | 75.3% | 3.3% | 57.3% |
| Custom probes (zh/en) | 11 | 63.6% | 0.0% | 36.4% |

**Capability vs official base (BF16 source)**

| Benchmark | n | Base | OrcaRouter BF16 | Δ |
|---|---:|---:|---:|---:|
| MMLU (0-shot letter) | 300 | 90.0% | 87.7% | −2.3 |
| MMLU-Pro (CoT) | 400 | 77.8% | 76.2% | −1.5 |
| GSM8K (CoT) | 150 | 92.0% | 93.3% | +1.3 |
| CMMLU (0-shot, Chinese) | 500 | 81.8% | 81.6% | −0.2 |

Vision/OCR, tool-calling, and reasoning were verified on that BF16 build. This MXFP4 file keeps the same vision tensors in BF16; text-path quality vs that BF16 source is not re-measured here.

---

## Architecture (unchanged)

`Qwen4ExpForConditionalGeneration` / `qwen4_exp`. 48 layers, hidden 2560, hybrid attention (36 Gated DeltaNet + 12 QSA full-attention, interval 4), 512 routed experts top-10 + shared expert, 51B PLE n-gram embedding, hyper-connections, native vision + video, one MTP layer. ~125B + 51B n-gram + ~4B MTP; ~6B activated MoE.

`language_model_only: false`, `mtp_num_hidden_layers: 1`.

---

## Files

20 shards `model-00001-of-00020.safetensors` … `model-00020-of-00020.safetensors`, `model.safetensors.index.json`, `config.json` (per-module `quantization` / `quantization_config`), tokenizer + chat template, vision/video preprocessor configs, `LICENSE`.

Trust Remote Code is not required for mlx-vlm 0.6.17 / oMLX 0.6.3.

---

## License

Weights are distributed under the **Qwen Community License 1.0** in [`LICENSE`](LICENSE), as shipped with the Qwen3.8-Flash-Next line. Quantization and this card do not change those obligations. The abliterated BF16 source is [`orcarouter/Qwen3.8-Flash-Next-Uncensored`](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored) (OrcaRouter documents Apache 2.0 on their card; the license file copied with these weights is Qwen Community License 1.0).

---

## Acknowledgements

- [Qwen / Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — architecture and original training.
- [OrcaRouter](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored) — full-precision uncensored BF16 source this quant is derived from.
- mlx / mlx-vlm `qwen4_exp` support; oMLX for vision + MTP serving.
- Community mixed 4/8 and mixed MXFP4 profiles (PipeNetwork, txgsync) for tensor-priority guidance.

## Bias, risks, and limitations

- **Safety guardrails removed** — see the disclaimer.
- Inherits biases and limitations of Qwen3.8-Flash-Next and of the OrcaRouter abliteration.
- MXFP4 / mixed affine error vs the BF16 source is **not quantified** on this file.
- `mlx-vlm` 0.6.17 will not use the packed MTP head even though the tensors are present.
