---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
library_name: transformers
tags:
- ektome
- abliterated
- uncensored
- pristinely-uncensored
- no-finetuning
- no-training
- gptq
- marlin
- 4-bit
- vllm
- mtp
- multi-token-prediction
- speculative-decoding
---

# Ektome-Qwen3.8-27B-PristinelyUncensored-GPTQ-MTP

**Qwen3.8-27B, made PristinelyUncensored by Ektomē — with ZERO training and ZERO fine-tuning —
then packed to 4-bit GPTQ (Marlin) with the MTP head preserved.**

> Ektomē (ἐκτομή, *"excision"*) is **weight surgery**, not training. It reads the model's own
> refusal direction from its activations and surgically removes it, then refuses to ship unless
> capability measurably survives.

## Also available: a 14.72 GB build with 256k context

If you can install a patched vLLM, **[HOMEUSER-16-24](https://huggingface.co/Zynerji/Qwen3.8-27B-PristinelyUncensored-HOMEUSER-16-24)**
is the same model, smaller and far longer-context:

| | this build (GPTQ-MTP) | HOMEUSER-16-24 |
|---|---|---|
| size | 19.56 GB | **14.72 GB** |
| max context on 24 GB | limited by weights | **262,144** |
| runs on | **stock vLLM** | requires the [Zynerji/vllm fork](https://github.com/Zynerji/vllm) |
| 16 GB card | no | yes (10-14k context, card-dependent) |

The difference is that HOMEUSER also compresses `embed_tokens` and `lm_head` — two
248320 x 5120 tensors this build leaves in bf16 — which frees ~6 GB for KV cache. Stock vLLM
cannot bind those, hence the fork.

**Stay on this build if you want to run on unmodified vLLM or SGLang.** That is a real reason to
prefer it, and it is why this repo is not deprecated.

## ⚠️ Read this before downloading — two hard load requirements

**1. Serve this with vLLM or SGLang. `transformers` cannot load it.**
The DeltaNet projections `linear_attn.in_proj_a` / `in_proj_b` are `[48, 5120]`, and Marlin
requires `out_features % 64 == 0`. They are therefore excluded via GPTQModel's per-module
`dynamic` config and left at bf16. GPTQModel states plainly that `dynamic` is *"fully supported
in latest vLLM and SGLang but not yet available in hf transformers."* This is not a defect — it
is what makes the checkpoint valid. **407 of 503 decoder Linears are 4-bit; the 96 excluded ones
are 0.095% of decoder parameters.**

**2. On Python 3.11, uninstall `flashinfer-python` or vLLM will not start.**
`flashinfer` annotates with `array.array[int]`, which requires Python ≥ 3.12, and fails as an
unrelated-looking `TypeError: type 'array.array' is not subscriptable` during engine init:

```bash
pip uninstall -y flashinfer-python        # or use Python 3.12+
export VLLM_USE_FLASHINFER_SAMPLER=0
```

## Receipts (measured, not claimed)

Ektomē ships nothing that fails its gate. From the produce run on the bf16 parent:

| metric | pristine | edited |
|---|---|---|
| refusal compliance (n=100, anchored) | 0.000 | **1.000** |
| MMLU-val (n=400) | 0.812 | **0.818** (Δ +0.005, SE 0.020 — **held**) |
| generative: foreign / degen / instr-follow | 0.000 / 0.000 / 0.400 | 0.000 / 0.000 / 0.400 |

Winner `A:frac=0.6`, **128 residual-write matrices** excised — the language decoder only. The MTP
head and the vision tower were **not** edited. `Δcap +0.005` against `SE 0.020` means capability
**held**; it is not an improvement, and should not be read as one.

## Measured performance

RTX PRO 6000 Blackwell (sm_120), vLLM 0.27.1, 128 new tokens, median of 3 runs after warm-up:

| batch | tok/s |
|---|---|
| 1 | **72.6** |
| 8 | **512.1** |

**7.05× scaling from batch 1 to 8** — near-linear, which is what Marlin exists for. Weights occupy
**17.7 GiB** (from 54 GB bf16), so it fits a 24 GB card with room for KV cache. These numbers are
from sm_120 and will not transfer unchanged to other architectures.

## MTP (multi-token prediction) — measured, and worth enabling

The 15-tensor MTP head is present and intact at bf16. **It is off unless you turn it on**, and
turning it on is a large win:

```python
from vllm import LLM
llm = LLM(
    model="Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored-GPTQ-MTP",
    speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 1},
)
```

```bash
vllm serve Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored-GPTQ-MTP   --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
```

### Measured on RTX PRO 6000 Blackwell (sm_120), vLLM 0.27.1, greedy

| workload | MTP off | MTP on | gain |
|---|---|---|---|
| short prompts, batch 1 (128 tok) | 75.7 tok/s | **102.5 tok/s** | **+35%** |
| short prompts, batch 8 (128 tok) | 531.8 tok/s | **691.8 tok/s** | **+30%** |
| **deep coding, 4 tasks × 1024 tok** | 265.3 tok/s | **433.6 tok/s** | **+63%** |

**The gain is largest on code.** Structured output is more predictable, so the draft head is
accepted more often — the opposite of the usual intuition that long generations erode
speculative gains. If you are using this for programming, enable MTP.

Weights occupy **17.7 GiB**, so it fits a 24 GB card with room for KV cache.

### Caveats, stated honestly

- The head is **carried, not quantized**. GPTQModel cannot reach it —
  `Qwen3_5ForConditionalGeneration` does not expose MTP in its module tree — so unlike Qwen's
  official FP8 release (which quantizes the MTP projections to `F8_E4M3`), it stays bf16 and
  costs ~849 MB. Quantizing it is *not* obviously desirable now that it is known to be in the
  compute path: a lower-precision draft may reduce acceptance and cost more than the space saves.
- These figures are from sm_120 and will not transfer unchanged to other GPUs.
- At greedy decoding the speculative and non-speculative outputs were **not byte-identical** in
  our coding run (one task's truncated output parsed in one configuration and not the other).
  Speculative decoding is designed to be output-preserving, so treat this as unexplained rather
  than as a quality claim in either direction.

## Receipts measured ON THE 4-BIT WEIGHTS

Almost every quantized abliteration on the Hub reports the receipts of its full-precision parent
and ships the quant unmeasured. These numbers are from **this artifact**, not its parent.

| | bf16 parent | **this 4-bit build** | delta |
|---|---|---|---|
| refusal compliance (n=100) | 1.000 | **0.990** | −0.010 |
| MMLU-val (n=400) | 0.8175 | **0.8025** | −0.015 — **within SE (0.0195)** |
| vision (synthetic image) | — | **PASSED 4/4** | — |

**Capability held.** The MMLU delta is inside the standard error, i.e. statistically
indistinguishable from no loss. For comparison, the `nf4` build of Qwen3.5-9B in this line cost
~2.5 MMLU points; GPTQ at 4 bits did materially better here.

**Still uncensored after quantization.** 99 of 100 harmful prompts complied. 4-bit quantization
can partially restore refusal behaviour — it did not here, and that is measured rather than assumed.

**Vision survived abliteration *and* quantization.** Given a synthetic image (red square outline,
blue circle centred inside), the model returned:

> *"A large square is defined by a thick, solid red border... The interior of this square is
> white. Perfectly centered within the red square is a solid blue circle."*

All four checks passed (red, blue, square/border, circle). The vision tower was excluded from
both the excision and the quantization, and it works.

**Method.** Compliance uses opening-anchored refusal detection over `mlabonne/harmful_behaviors`
(n=100, greedy) — a refusal marker buried inside complied content does not count as a refusal.
MMLU is `cais/mmlu` validation, 400 questions, seed 0, single-letter generative answer. Vision is
a synthetic-image description with keyword checks. Run on vLLM 0.27.1, RTX PRO 6000 (sm_120).

## Concurrency — serving an agent fleet

Measured on a single **RTX PRO 6000 Blackwell (sm_120, 96 GB)**, vLLM 0.27.1, MTP enabled,
greedy, 256 new tokens per request, 4096 context.

| concurrent requests | aggregate tok/s | per-agent tok/s |
|---:|---:|---:|
| 1 | 116 | 116 |
| **8** | **860** | **107** |
| 32 | 2,134 | 67 |
| 64 | 2,433 | 38 |
| **128** | **2,607** | 20 |
| 256 | 2,532 | 10 |

**Two numbers matter here.**

**8 concurrent requests cost only ~7% per-agent throughput** (116 → 107 tok/s) while delivering
8× the aggregate. If you are running a small fleet of coding agents, this is close to free
parallelism.

**Aggregate peaks at 128 concurrent (~2,600 tok/s) and *declines* at 256.** Past 128 you are
buying queueing latency, not work. Size your fleet at or below that.

Throughput is Marlin doing what it exists for — holding the 4-bit advantage as batch grows,
where naive INT4 kernels decay back toward FP16 speed.

⚠️ **On a 24 GB card the limit is KV cache, not compute.** Weights take 17.7 GB, leaving ~6 GB
of cache — roughly **8–12 concurrent agents at 4–8k context**. The numbers above come from a
96 GB card with ~62 GB of cache available.

## Reproducibility caveat (measured, not theoretical)

Speculative decoding here is **not bit-identical to non-speculative decoding**, even at greedy.
Tested directly by running each configuration twice:

| | result |
|---|---|
| baseline vs itself | **identical** |
| MTP vs itself | **identical** |
| MTP vs baseline | **differs** |

Each configuration is individually deterministic; they simply produce different text from each
other. The divergence is a paraphrase, not corruption — e.g. `"(additive) — lowest precedence"`
versus `": lowest precedence (additive)"` — same content, same quality.

This matters if you **cache completions, run evaluation harnesses, or diff agent outputs across
runs**: pick one configuration and stay on it. It is not evidence of quality loss in either
direction, and we make no claim about which is "correct".

## Provenance

Single-pass quantization from the repaired bf16 parent
([Ektome-Qwen3.8-27B-PristinelyUncensored](https://huggingface.co/Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored)),
512 wikitext-2 calibration samples, 4-bit, group size 128, `desc_act=false`. Verified before
upload by reloading the saved checkpoint and generating, then by serving it in vLLM.

## Limitations

- Uncensored by construction. **You are responsible for what you generate with it.**
- 4-bit quantization carries a real quality cost that is not measured in this card; the receipts
  above are from the bf16 parent, not from this artifact.
- The vision tower is preserved but vision was not re-benchmarked after quantization.
