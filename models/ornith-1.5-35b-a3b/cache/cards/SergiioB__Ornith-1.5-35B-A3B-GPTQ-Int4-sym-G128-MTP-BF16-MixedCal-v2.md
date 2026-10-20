---
license: apache-2.0
base_model: ornith-ai/Ornith-1.5-35B-A3B
library_name: transformers
pipeline_tag: text-generation
tags:
  - qwen3_5_moe
  - gptq
  - int4
  - moe
  - intel
  - arc-pro-b70
  - vllm
  - mtp
---

# Ornith-1.5-35B-A3B GPTQ-Int4 G128 MixedCal-v2 (MTP BF16)

Self-reported Intel Arc Pro B70 measurements (E2) with raw evidence.
Isolated C1, cache off, greedy diagnostic. Not independently reproduced.
LocalMaxxing payloads are validate-local only and **not submitted**.

Local GPTQ INT4 conversion of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
for Intel Arc Pro B70 / vLLM XPU. This is **not** an official Ornith GPTQ.

## What changed vs the first conversion

The first B70 GPTQ (`Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16`) used
WikiText-style calibration. GPTQModel fell back to round-to-nearest on
**7,605 / 30,720 = 24.76%** of expert projection matrices (layer 39: 297/768).

MixedCal-v2 keeps the **same format and tensor scope** and only changes
calibration coverage:

| | Original | MixedCal-v2 |
|---|---:|---:|
| Format | GPTQ INT4 sym G128, `desc_act=false` | same |
| Quantized | routed expert `gate_proj` / `up_proj` / `down_proj` only | same |
| Expert `qweight` | 30,720 | 30,720 |
| MTP tensors | 785 BF16, 0 quantized | 785 BF16, 0 quantized |
| Forbidden leaks | 0 | 0 |
| Calibration | WikiText-style | 128 × 1,536 mixed-domain tokens (196,608), seed `15035260819` |
| RTN fallback | 24.76% | **10.37%** (3,186 / 30,720) |
| Layer 39 RTN | 297/768 | 75/768 |
| Size | ~24.4 GB | 24,454,916,052 bytes |

RTN fallback is a **quantizer** weight-only fallback, not a runtime CPU/XPU
kernel fallback. The tensors remain GPTQ INT4.

Held-out no-spec quality A/B vs the original: both artifacts produced coherent
answers on the six diagnostic domains. No BF16 logit/KL or task-suite score is
claimed.

Speed on vLLM XPU at **150 W**, C1, cache off, n=5 confirmation: **parity**
(instance-median of three loads):

| Cell | Original | MixedCal-v2 |
|---|---:|---:|
| p512/g128 client post-first | 70.80 tok/s | 70.74 tok/s |
| p8192/g128 client post-first | 64.86 tok/s | 64.95 tok/s |
| p2048/g1 cold input | 6935 tok/s | 6968 tok/s |

Do not call this artifact “faster.” The conversion win is calibration coverage
and fewer RTN fallbacks.

## Quantization contract (fail-closed)

Quantized:

- `model.language_model.layers.*.mlp.experts.{gate,up,down}_proj`

**Not** quantized:

- `lm_head`, embeddings
- attention / GDN
- router (`mlp.gate`)
- shared experts
- entire `mtp.*` tree (draft stays BF16 on disk)
- vision tower

```python
dynamic = {
    "-:.*attn.*": {},
    "-:.*mlp\\.gate$": {},
    "-:.*mtp.*": {},
    "-:.*shared_expert.*": {},
    "-:.*visual.*": {},
    "lm_head": {},
    "model.language_model.embed_tokens": {},
}
```

Quantizer: GPTQModel 7.3.2. Source revision
`fbb995a79eedd569a5edc5f2af9644c0fa1124fc`. Conversion wall ~7.7 h on B70
inside `vllm/vllm-openai-xpu@sha256:f01e24f6…`.

## Serve on Intel Arc Pro B70

Pinned image (same digest as the Qwen3.8-27B nightly path, **not** the
historical v0.21 native-int4moe image):

```text
vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
vLLM 0.27.2rc1.dev77+gac7509e2b
vllm-xpu-kernels 0.1.12.3
MoE backend observed: XPU WNA16 (int_wna16)
```

Research default is **MTP1**, not MTP4. This head is a single MTP layer;
acceptance past pos0 collapses (day-0 per-pos 81 / 15 / 2.5 / 0.5%).

```bash
vllm serve /model \
  --quantization gptq --dtype float16 \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype auto --block-size 64 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching --language-model-only --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

### Vision serving (full VLM)

The 333-tensor vision tower ships unquantized **BF16 (~0.89 GB)** in this
artifact — same exclusion scope as the MTP heads (see the quantization
contract above). Serve image inputs by dropping `--language-model-only`;
everything else stays identical (MTP1, same image digest, same patches).

VRAM note (fp8 KV, `--gpu-memory-utilization 0.90`, MTP1): the tower plus
multimodal profiling take ~0.9 GiB from the KV budget, so
`--max-model-len 160000` fails the startup KV check (1.82 GiB needed,
1.77 GiB available; vLLM estimates a 154,176 maximum). Use
`--max-model-len 131072` (~1.49 GiB needed) or raise
`--gpu-memory-utilization`.

Verified 2026-08-23 on the identically-scoped AutoRound twin artifact
(same packing, same 333 BF16 visual + 785 BF16 MTP tensors): a 64×64
red-PNG request returns `red` with ~78 image prompt tokens; text-only
requests unchanged. Step-by-step:
[ORNITH-VLLM-XPU.md §3b](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/main/docs/ornith15-35a3/ORNITH-VLLM-XPU.md).

### Measured C1, cache off (client SSE)

| Cell | MixedCal-v2 | Notes |
|---|---:|---|
| no-spec 32K p512/g128 @150 W n=5 | 70.74 tok/s | instance-median of 3 loads |
| MTP1 16K p512/g128 @150 W n=5 | 96.43 tok/s | accept 77.0% (595/773) |
| MTP2 16K p512/g128 @150 W n=5 | 84.16 tok/s | accept 41.7% |
| MTP4 16K p512/g128 @150 W n=5 | 66.27 tok/s | accept 22.1%; **slower than no-spec** |
| exact p65408/g128 no-spec n=3 | 54.49 tok/s | 65,536 serve |
| exact p130944/g128 MTP1 n=3 | 70.25 tok/s | 131,072 + boundary patch |
| exact p262016/g128 no-spec n=3 | 35.35 tok/s | 262,144 + `--kv-cache-memory 6623879680` |

Cold input = actual endpoint prompt tokens / client TTFT, **not** llama-bench pp.

| Cap | p2048/g1 round medians | p8192/g1 round medians |
|---|---|---|
| 150 W | 7271 / 7212 / 7055 | 7036 / 7050 / 7062 |
| 230 W | 9748 / 9713 / 9771 | 9647 / 9683 / 9670 |

Paired A/B on one warm MixedCal-v2 no-spec 32K server, three alternating rounds,
matched except configured cap. 230 W recovers the day-0 ~9.5–9.7k class.
After a 230→150 drop the first retained p2048 sample is ~8.7k before later
samples settle ~6.9k; do not treat that first sample as 150 W sustained.

LocalMaxxing long-prompt no-spec 32K @230 W (unique entropy, actual
**2,899** prompt tokens, `--max-tokens 1`, n=5): `tokSPrefill` **9556.4**.
Same-load HTTP harness medians: p2048/g1 **9428** (n=4 valid TTFT), p8192/g1
**9608** (n=5). Prefix-cache hit delta 0. `validate-local` valid. **Not
submitted.** Do not publish LMX `tokSOut` from a 1-token completion.

Optional local **DraftINT4** overlay (`B70_DRAFT_LMHEAD_INT4` +
`B70_DRAFT_MTP_INT4`) does **not** change these weight files. On this image,
MTP1 DraftINT4 n=5 screened **106.27 / 97.16** tok/s vs BF16-draft
**96.43 / 89.85**, accept 81.9% vs 77.0%. Overlay is optional and local-only.

Exact 131,072-token MTP completions need `patch_mtp_boundary.py`. A 262,144
no-spec serve is a capacity completion, not a quality claim.

Recipe: [intel-arc-pro-b70-inference-cookbook `docs/ornith15-35a3/`](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook).

## What this is not

- Not a drop-in for Qwen3.6-35B MTP4 **204.6** tok/s (that cell is a different
  image generation: vLLM 0.21 native int4moe + int8 store, short g32).
- Not DFlash2. No Ornith DFlash2 draft is published; the Arc Pro DFlash2 write-up
  is SGLang / Qwen3.8-27B.
- Not a LocalMaxxing submission unless a later record says otherwise.

## License

Follow the Ornith-1.5 source license (Apache-2.0 on the recorded card) plus
any additional terms of `ornith-ai/Ornith-1.5-35B-A3B`.
