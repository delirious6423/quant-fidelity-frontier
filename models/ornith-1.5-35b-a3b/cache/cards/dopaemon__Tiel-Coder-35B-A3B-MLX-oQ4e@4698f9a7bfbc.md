---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: mlx
language:
  - en
  - zh
tags:
  - mlx
  - omlx
  - apple-silicon
  - oq
  - qwen35moe
  - moe
  - imatrix
  - unsloth-dynamic
  - agentic-coding
  - token-efficient
  - vision
---

<div align="center">
  <img src="assets/tiel_banner_eyebrow.png" alt="TielCoder — Ornith-1.5 35B-A3B, sharpened, dynamically quantized" width="100%">
</div>

<div align="center">
  <img src="assets/card_tiel_swe.png" alt="SWE-bench-Live: Tiel solves 12 of 25, level with Opus 4.6 and ahead of KAT-Coder, Nail, stock Qwen3.6-35B-A3B and Ornith" width="100%">
</div>

> **These numbers were measured on the GGUF build, not this one.** The plates on this page are the best
> evidence we have about TielCoder, and we would rather show them than show nothing — but this file
> uses a different quantizer (oMLX's oQ against llama.cpp's k-quants), and changing quantizer moves
> results. On the same Nail weights we measured a 0.7-point MMLU-Pro gap and a 24% difference in
> token counts between MLX and GGUF. Read them as evidence about the *model*, not as measurements of
> *this* file. If you need numbers you can hold us to, use the
> [GGUF build](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF).

# Straight to the point

*Tiel* is the fast coder of the arsenal. At 4-bit quantization and 21 GB it fixes real codebase issues at the rate (and speed, with
the right GPU) of Opus 4.6 medium, while holding the best multi-turn conversation of any local model we
have measured. It is also cheerfully bad at trivia.

Pick it for work. Pick something else for exams.

> **This is [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) re-quantized
> with oMLX's oQ4e quantizer and carrying the [Sharp chat template](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates)**
> inside the checkpoint. For speculative decoding, the [identical build with Ornith's trained MTP head](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-MLX-oQ4e-MTP) is a separate repo.
> **The uncensored and improved CyberTiel is out now!** [Get it here](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ4e).

## The numbers

<div align="center">
  Where it sits against the other local builds
  <img src="assets/card_tiel_local.png" alt="SWE-bench-Live: Qwen3.8-27B solves 16 of 25 at 50.2 minutes per attempt, Dirk 15 at 20.1, TielCoder 12 at 8.6, stock Qwen3.6-35B-A3B 8 at 5.5" width="100%">
  Multi-turn conversation
  <img src="assets/card_tiel_claw.png" alt="Claw-Eval multi-turn: Tiel 67.2 overall against Ornith 65.3 and Nail 60.5, over 114 scored conversations each" width="100%">
  Reasoning and knowledge
  <img src="assets/card_tiel_mmlu.png" alt="MMLU-Pro: Tiel 73.7 against Nail 84.0, both at 4-bit, with the control arm isolating the template" width="100%">
</div>

**Where it stands.** On 25 SWE-bench-Live problems Tiel fixes 12 — the same as Opus 4.6 (medium),
four more than Ornith-1.5 itself, three more than Nail, and four more than Sonnet 5 (medium). Among
models of its own class it is first; the ones ahead are dense 27Bs and Opus 5. Its time per attempt
is also steadier than Nail's: an 8.6 minute median against 7.2, but a 12.3 minute mean against 15.7,
because it lacks Nail's tail of expensive attempts.

**How it talks.** On Claw-Eval's multi-turn tasks Tiel scores 67.2 against Nail's 60.5 and its own
base's 65.3, over 114 scored conversations each. It earns that by answering better rather than by
asking more: against the base it is 3.8 points up on answer quality and 5.1 down on clarifying
questions. The score weights answers four to one, so the trade pays — but if you want a model that
interrogates a vague request before acting, the base does that better.

**What it costs.** 73.7 on MMLU-Pro against Nail's 84.0, both at 4-bit. Most of that is inherited
rather than built: Ornith-1.5 scores 78.0 where stock Qwen3.6-35B-A3B scores 85.3. Our quantization
is not the cause — the same quant carrying Ornith's own template scores exactly what Ornith scores.
The remaining 4.3 points are the Sharp template buying shorter answers, which is the trade this
build exists to make.

**Which one.** Agentic coding, or long conversations that have to stay useful → Tiel. Exam-style
knowledge and hard reasoning → [Nail](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-GGUF),
which is 10.3 points better on MMLU-Pro and 6.7 worse in conversation.
The most fixes per problem regardless of weight → [Dirk](https://huggingface.co/peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF),
the dense 27B that solves 15 of the same 25 — one behind stock Qwen3.8-27B, at 2.5x its speed.

## Run it

One tier: **oQ4e**, 4-bit dynamic mixed precision with an imatrix pass. Vision is included in the
same folder — no separate projector file.

**oMLX** — put the folder under `~/.omlx/models/peculiar-ragdoll/Tiel-Coder-35B-A3B-MLX-oQ4e`, or pull it from the oMLX
admin dashboard.

Sampling: `temperature 1.0`, `top_p 0.95`, `top_k 20`. For agentic coding we ran `temperature 0.6`.

**Prefer to keep the files yourself?**

```bash
hf download peculiar-ragdoll/Tiel-Coder-35B-A3B-MLX-oQ4e --local-dir TielCoder-MLX
python -m mlx_vlm.generate --model TielCoder-MLX --max-tokens 512 \
  --prompt "Explain what this function does."                       # text
python -m mlx_vlm.generate --model TielCoder-MLX --max-tokens 512 \
  --prompt "What is in this screenshot?" --image photo.jpg          # vision
```

**Load it with `mlx-vlm`, not `mlx-lm`.** This is a vision-language checkpoint. `mlx_lm.load()`
accepts it and then emits garbage tokens — a loader mismatch, not a bad quant, but it fails quietly.

Both runtimes apply the embedded template automatically — nothing to pass.

## No multi-token-prediction head

Ornith-1.5's GGUF conversion carries an MTP (`nextn`) block whose weights are untrained — a fresh
random initialization, which we measured and removed from the GGUF ladder. It is not in the
safetensors this build quantizes from, so there was nothing to strip here.

## How the quantization was done

oQ4e is oMLX's dynamic quantizer: 4-bit base with mixed precision by layer position and selective
non-quantization, plus an imatrix pass — the "e" — that measures which weights carry the most
signal before deciding what to keep at higher precision. It is the same idea as the GGUF ladder's
Unsloth-Dynamic-plus-imatrix recipe, implemented for MLX, but it is *not* the same computation:
oQ derives its own importance data rather than consuming the GGUF imatrix we baked.

That is the reason for the caveat at the top. Two quantizers pursuing the same goal by different
routes do not land in the same place, and only the GGUF route has been benchmarked.

## Limitations

- **Exam scores are its weak axis.** If you are picking on MMLU-Pro, Nail is 10.3 points better.
- **It asks fewer clarifying questions than its base**, by 5.1 points. Terser is not always better;
  a vague request gets answered rather than questioned.
- Benchmarks are one run per problem on SWE-bench-Live and three seeds on MMLU-Pro. Treat small
  differences as noise.
- Chinese and English only, inherited from the base.

## Credits

- [**ornith-ai**](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — the Ornith-1.5-35B-A3B weights (MIT).
- [**oMLX**](https://github.com/jundot/omlx) — the oQ quantizer this build uses.
- [**froggeric**](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) — the template lineage Sharp builds on.
- [**eaddario**](https://huggingface.co/datasets/eaddario/imatrix-calibration) — the calibration corpora the imatrix was measured on (MIT).
- [**MLX**](https://github.com/ml-explore/mlx) and [**mlx-vlm**](https://github.com/Blaizzy/mlx-vlm) — the runtime.

MIT, inheriting Ornith-1.5's license.
