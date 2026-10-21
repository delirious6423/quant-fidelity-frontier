---
license: apache-2.0
license_link: LICENSE
base_model: google/gemma-4-26B-A4B-it
tags:
- abliterated
- uncensored
- gemma4
- moe
- direct-weight-editing
- expert-granular-abliteration
- projected-abliteration
---

# Gemma 4 26B-A4B IT — Abliterated (V6)

This is an abliterated (uncensored) version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), a 128-expert / 4-active MoE thinking model, created using [Abliterix](https://github.com/wuwangzhang1216/abliterix).

## Method

Gemma 4's double-norm architecture (4× RMSNorm per layer) and Per-Layer Embeddings (PLE) make LoRA and hook-based steering completely ineffective. On top of that, the A4B variant is a **Mixture-of-Experts** model: refusal signal is distributed across all 128 experts per layer, not concentrated in a handful of safety experts. Targeting only the top-N experts identified by router profiling is insufficient.

This model uses **direct weight editing with Expert-Granular Abliteration (EGA) + Projected Abliteration** — norm-preserving orthogonal projection applied to every expert slice in every MoE layer, plus the attention output projection, with grimjim's projected refinement to preserve helpfulness-aligned signal.

Key techniques:
- **Projected Abliteration (V6 addition):** grimjim's method builds a low-rank basis from the refusal *and* the helpfulness directions, then projects out only the refusal-orthogonal component. Compared to raw EGA orthogonal projection, this preserves the helpfulness-aligned dimensions of each weight matrix. V5 had this disabled; enabling it dropped refusals from 25/100 to **2/100** at matched KL.
- **Expert-Granular Abliteration (EGA):** orthogonal projection applied to all 128 expert `mlp.down_proj` slices across 30 layers (≈3,840 expert blocks total). Reference result on 26B-A4B: 3/100 refusals with EGA vs 29/100 without, reported by TrevorS.
- **Sharp-peak EGA profile:** `min_weight_frac_max = 0.10` on `mlp.down_proj` forces TPE to explore "high peak at one layer, flat decay elsewhere" trials rather than flat-high steering that blows KL without gaining ASR. Mirrors the gpt-oss-20b V6 winner fingerprint.
- **Thinking-model-aware generation:** `max_gen_tokens = 200` to ensure the judge sees the actual answer after Gemma-4's `<|channel|>thought` prefix. V5 used 100 tokens, which was sometimes consumed entirely by the thought channel, causing false "refused" classifications.
- **Reduced component set:** only `mlp.down_proj` (through EGA) and `attn.o_proj` are steered. Q/K/V projections were dropped after V4 showed they contribute effectively zero signal on a 128-expert MoE — the refusal pathway lives in the expert path, not attention.
- **Norm-preserving** row magnitude restoration (critical for Gemma 4's double-norm architecture).
- **Linear decay kernel + high min_weight_frac** — flat-decay profile concentrates the search on the strong-peak region that TPE reliably converges to.
- **Tighter KL budget:** target 0.004, prune threshold 0.02. V5's 0.008 target let the optimizer settle in a safe-but-useless basin; 0.004 forces exploitation past it.

## Evaluation

| Metric | Value |
|---|---|
| **Refusals (private eval dataset, 100 prompts)** | **2/100** |
| **KL divergence from base** | **0.0005** |
| Baseline refusals (original model) | 97/100 |
| Optimization trials completed | 80 (25 warmup + 55 TPE) |
| Best trial | #47 (TPE exploit phase) |
| Hardware | 1× H100 SXM 80 GB, bf16 |
| Total optimization wall-clock | ~11h |

## A note on honest evaluation

Many abliterated models on HuggingFace claim near-perfect scores ("3/100 refusals", "0.7% refusal rate", etc.). **We urge the community to treat these numbers with skepticism** unless the evaluation methodology is fully documented.

Through our research, we have identified a systemic problem: **most abliteration benchmarks dramatically undercount refusals** due to short generation lengths. Gemma 4 models exhibit a distinctive "delayed refusal" pattern — they first produce 50-100 tokens of seemingly helpful context (educational framing, disclaimers, reframing the question), then pivot to an actual refusal. When evaluation only generates 30-50 tokens, the refusal hasn't appeared yet, and both keyword detectors and LLM judges classify the response as compliant.

### Our evaluation standards

We believe accurate benchmarking requires:

- **Sufficient generation length (≥100 tokens):** short generations systematically miss delayed/soft refusals. Our optimizer-loop evaluation uses 200 tokens (increased from V5's 100) to fully capture Gemma 4's refusal pivot point after the thought channel.
- **Hybrid detection:** keyword matching for obvious refusals + LLM judge (Google Gemini 3 Flash) for ambiguous cases. Neither method alone is sufficient.
- **Challenging, diverse prompts:** our private evaluation dataset contains 100 prompts spanning English and Chinese, multiple sophistication levels (from direct requests to socially-engineered framings), and diverse harm categories.
- **Reproducible methodology:** all parameters (generation length, detection method, dataset characteristics) should be documented on the model card. If they aren't, the numbers are meaningless.

**We report 2/100 refusals honestly.** This number was obtained with the same evaluation pipeline as V5's 25/100 — a stricter standard than the 3/100 result commonly cited for this model.

## Usage

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer
import torch

model = AutoModelForImageTextToText.from_pretrained(
    "wangzhang/gemma-4-26B-A4B-it-abliterix",
    dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("wangzhang/gemma-4-26B-A4B-it-abliterix")

messages = [{"role": "user", "content": "Your prompt here"}]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Minimum VRAM for bf16 inference: ≈ 50 GB (one H100 80 GB or one RTX Pro 6000 Blackwell 96 GB).

## Disclaimer

This model is released for research purposes only. The abliteration process removes safety guardrails — use responsibly.


<!-- wangzhang-provenance-license-v1 -->
## Provenance and Modification Notice

- **Immediate source checkpoint:** [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **Exact base revision used:** Not recorded in the existing release artifacts; the current upstream HEAD is not substituted.
- **Modification method:** Abliterix weight-space / representation intervention intended to reduce refusal behavior.
- **Modified and published by:** Wangzhang Wu
- **Repository first published:** 2026-04-14 (Hugging Face repository metadata)

The original model weights and/or derived checkpoint were modified. This repository is an independent derivative and is **not** an official release of the upstream model developer.

## License and Attribution

The governing upstream license is **Apache License 2.0**. A copy is included in [`LICENSE`](LICENSE). License source audited on 2026-08-29: https://ai.google.dev/gemma/apache_2

All applicable upstream copyright, attribution, acceptable-use, and other license terms remain in effect. This repository grants no rights beyond those provided by the upstream license. Downstream users must preserve applicable license and attribution notices.

---

<!-- wangzhang-safety-disclaimer-v1 -->
## Disclaimer and Responsible Use / 免责声明与安全使用声明

**English**

This is an experimental, modified model provided for research, evaluation, and other lawful purposes. Its safety alignment, refusal behavior, or other safeguards may have been weakened or removed. It may produce inaccurate, biased, offensive, explicit, dangerous, or illegal content. Outputs are not professional advice and must not be relied on for medical, legal, financial, safety-critical, or other high-stakes decisions without qualified human review.

You are solely responsible for how you access, use, deploy, fine-tune, or redistribute this model and its outputs, including compliance with applicable laws, regulations, licenses, third-party rights, platform policies, and the original model's terms. Do not use it to facilitate harm, illegal activity, malware, fraud, privacy violations, targeted harassment, weapons development, or decisions that materially affect a person's rights or access to essential services without appropriate authorization, safeguards, and qualified oversight.

Before deployment, perform a context-specific risk assessment and testing; use human oversight, access controls, content filtering, rate limits, monitoring, logging, and incident-response procedures as appropriate. Preserve this notice in downstream redistributions.

The model is provided **"AS IS"**, without warranties of any kind. To the fullest extent permitted by applicable law, the maintainer disclaims liability for claims, damages, or losses arising from use, misuse, inability to use, or redistribution of the model or its outputs. Nothing in this notice overrides applicable law or the governing license, and this notice is not legal advice.

**中文**

本模型属于实验性改造模型，仅供研究、评测及其他合法用途。其安全对齐、拒答机制或其他防护可能已被削弱或移除，因此可能生成不准确、偏见、冒犯、露骨、危险或违法内容。输出不构成医疗、法律、金融等专业意见；涉及高风险或重大权益的决定，必须由具备资质的人员复核。

使用者须对模型及其输出的访问、使用、部署、微调和再分发承担全部责任，并遵守适用法律法规、许可证、第三方权利、平台政策及原模型条款。不得将本模型用于促成伤害、违法活动、恶意软件、欺诈、侵犯隐私、定向骚扰、武器开发，或在缺乏适当授权、防护和专业监督时，用于实质影响个人权利或基本服务获取的决策。

部署前应进行与具体场景相匹配的风险评估和测试，并酌情采用人工监督、访问控制、内容过滤、限流、监控、日志和事件响应措施；下游再分发时应保留本声明。

本模型按“现状”提供，不附带任何形式的保证。在适用法律允许的最大范围内，维护者不对因使用、误用、无法使用或再分发本模型及其输出而产生的索赔、损害或损失承担责任。本声明不取代适用法律或管辖本模型的许可证，也不构成法律意见。
