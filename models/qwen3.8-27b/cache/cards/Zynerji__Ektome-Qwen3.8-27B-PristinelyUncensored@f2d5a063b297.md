---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - uncensored
  - abliterated
  - abliteration-repair
  - capability-preserving
  - ektome
  - surrogate-null
  - qwen3.8
  - mtp
  - multi-token-prediction
  - speculative-decoding
language:
  - en
pipeline_tag: text-generation
---

> ### GGUF status — 2026-08-15
>
> **`Q4_K_M` has been rebuilt and re-uploaded. It is correct.** If you downloaded any GGUF from
> this repo before 2026-08-15 01:00 UTC, delete it and pull again.
>
> The earlier files failed with `error loading model: check_tensor` in llama.cpp / LM Studio,
> with MTP on *or* off. The header declared `block_count=65` / `nextn_predict_layers=1` while the
> file shipped only `blk.0..63` and no MTP tensors — the header disagreed with its own contents,
> so the block-count check failed before MTP was ever consulted. Root cause was upstream: the
> multi-token-prediction head was lost when the abliterated model was saved, so metadata (read
> from `config.json`) and tensors disagreed.
>
> The rebuilt `Q4_K_M` is verified: header `block_count=65` matches 65 actual blocks, `blk.64`
> (the MTP block) is present, and it loads and generates.
>
> **The bf16 safetensors were never affected** — 1199 tensors, exact parity with `Qwen/Qwen3.8-27B`.
>
> Reported and correctly diagnosed by **MrGuzzy** and **Scottbyrd**. Thank you both.

## Quantized siblings

| repo | format | size | max context on 24 GB | runtime |
|---|---|---|---|---|
| **this repo** | bf16 safetensors | ~54 GB | — | transformers, vLLM, anything |
| [**-HOMEUSER-16-24**](https://huggingface.co/Zynerji/Qwen3.8-27B-PristinelyUncensored-HOMEUSER-16-24) | mixed-precision int4 | **14.7 GB** | **262,144** | **patched vLLM** ([fork](https://github.com/Zynerji/vllm)) |
| [**-GPTQ-MTP**](https://huggingface.co/Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored-GPTQ-MTP) | 4-bit GPTQ / Marlin | 17.7 GB | weight-limited | **stock vLLM / SGLang** |
| [**-HYBRID**](https://huggingface.co/Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored-HYBRID) | mixed-precision | 18.7 GB | weight-limited | stock vLLM |
| **GGUF Q4_K_M** | llama.cpp | 16.8 GB | — | llama.cpp, LM Studio, Ollama |

**Which one do you want?**

- **Long context, and you'll install a fork → HOMEUSER-16-24.** It is the only build that quantizes
  `embed_tokens` *and* `lm_head` — two 248320 x 5120 tensors every other release leaves in bf16 —
  which frees ~6 GB for KV cache and is why **262,144 tokens** fit on a 24 GB card. Measured on
  eight GPUs (5090 / 4090 / 3090 Ti / 3090 / A4000 / 5060 Ti / 5070 Ti / 4080 Super); MMLU 0.7995,
  HumanEval 0.8902, MMStar 0.6633. Stock vLLM cannot bind those two tensors, hence the fork.
- **You want unmodified vLLM or SGLang → GPTQ-MTP.** 72.6 tok/s at batch 1 / 512.1 at batch 8 on an
  RTX PRO 6000 (sm_120), vLLM 0.27.1. It cannot be loaded by `transformers` — the `[48, 5120]`
  DeltaNet projections are excluded from Marlin via GPTQModel's `dynamic` config, which only vLLM
  and SGLang understand. See that repo's card for the two load requirements.
- **llama.cpp / LM Studio / Ollama → the GGUF.**

## MTP (multi-token prediction) — present, and off unless you enable it

These weights include the complete 15-tensor MTP head, byte-identical to `Qwen/Qwen3.8-27B`
(Ektomē edits the language decoder only). **Speculative decoding is not automatic** — vLLM
ignores the head unless you ask for it:

```python
from vllm import LLM
llm = LLM(
    model="Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored",
    speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 1},
)
```

```bash
vllm serve Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored   --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
```

**Measured on the 4-bit GPTQ sibling** (RTX PRO 6000, vLLM 0.27.1, greedy): **+35%** at batch 1,
**+30%** at batch 8, and **+63% on long-form code generation** — the gain is largest on
structured output, because the draft head is accepted more often there.

⚠️ Those figures are from the [4-bit build](https://huggingface.co/Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored-GPTQ-MTP),
**not from these bf16 weights**. bf16 inference is more compute-bound, so the speedup here may
differ — we have not measured it. The mechanism and the invocation are identical; only the
magnitude is unverified for bf16.

One caveat that applies to both: at greedy decoding, speculative and non-speculative output are
each individually deterministic but **differ from one another**. Pick one configuration and stay
on it if you cache completions or run evaluation harnesses.

## What this bf16 contains

- **1199 tensors — exact parity with `Qwen/Qwen3.8-27B`**, including the 15-tensor **MTP
  (multi-token-prediction) head** and the full vision tower. Verified tensor-by-tensor against base.
- **128 residual-write matrices excised** (`o_proj` / `out_proj` / `down_proj`), language decoder
  **only**. The MTP head, the vision tower, and the 2 MTP-block matrices that pass a naive dim
  check were all left untouched.
- Ektomē receipts: refusal compliance **0.000 → 1.000** (n=100), MMLU-val **0.812 → 0.818**
  (Δ +0.005 against SE 0.020 — capability **held**, not improved), generative gate clean
  (foreign 0.000, degeneration 0.000, instruction-following 0.400 → 0.400). Winner `A:frac=0.6`.
- This is the **parent artifact**: every quantized sibling is derived from these weights, so any
  fine-tune, re-quantization or re-abliteration should start here rather than from a GGUF.


## The receipt — pristine vs Ektomē (before / after)

| model | capability (MMLU-val, 0-shot) ↑ | refusal on harmful ↓ |
|---|---|---|
| pristine Qwen3.8-27B (before) | 0.812 | 100% |
| **Ektomē (this model)** | **0.818** | **0%** |

Fully uncensored, capability **+0.005** vs pristine. Capability = MMLU-validation loglikelihood
accuracy; refusal on harmful = AdvBench, judge-free keyword classifier, thinking disabled so
chain-of-thought can't mask the answer.

## How we know it's real — the catcher's verdict

Every claim here is gated by a **surrogate-null catcher**, not asserted. The excision is applied
to *pristine* and measured; a capability change counts only if it beats the MMLU sampling floor.

<details>
<summary>📋 Show the raw before/after</summary>

```text
$ ektome_abliterate  Qwen/Qwen3.8-27B
  PRISTINE (before): compliance=0.000   mmlu-val=0.812
  EKTOME   (after):  compliance=1.000   mmlu-val=0.818   (dcap=+0.005)
  -> uncensored AND capability preserved, zero training.
```

</details>

## Why it's different

- **Uncensored** — complies on harmful prompts the base refuses.
- **Capability-preserved** — capability stays close to the *pristine* base, not the
  capability-lower crude-abliterated models most tools ship.
- **Zero training** — a single surgical weight edit on pristine, seconds of compute.
- **Honest** — every claim is gated by a surrogate-null test, not marketing.

## Run it locally

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored")
model = AutoModelForCausalLM.from_pretrained("Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored", torch_dtype="bfloat16", device_map="auto")
```

**GGUF (any gaming PC):** `Q4_K_M` (16.8 GB) — verified to load in `llama.cpp` / LM Studio / Ollama. **4-bit GPTQ/Marlin** (17.7 GB, vLLM or SGLang only) is available as a [separate repo](https://huggingface.co/Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored-GPTQ-MTP) and is the fastest option — see its card for measured throughput.
LM Studio / Ollama.

## The Ektomē line & feedback

If this is useful, a ❤️ **like** helps others find it. It's part of a **line** — the same
catcher-gated surgery across frontier small models (Qwen, Mistral, Llama, Nemotron) and larger
bases. Want a specific base next or have head-to-head numbers? **Open a
[Discussion](https://huggingface.co/Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored/discussions)** — the line is steered by what people actually run.

## Honest notes

- **Uncensored:** this model follows instructions the base refuses. Use it lawfully and
  responsibly; you are accountable for what you generate.
- Compliance uses a judge-free keyword classifier (thinking disabled); the capability delta is
  gated against a surrogate-null noise floor.
- **Method:** produced by *Ektomē*, a capability-preserving surgical-abliteration method. Results
  are reported; the full recipe is not (yet).

## License

Inherits the base model's license (**Qwen3.8-27B → apache-2.0**). No warranty; your use, your
responsibility.
