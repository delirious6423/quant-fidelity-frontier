---
license: gemma
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
tags:
- nvfp4
- quantized
- gemma4
- moe
- vllm
- blackwell
- quantized-lm-head
- llm-compressor
- compressed-tensors
---

# gemma-4-26B-A4B-it — NVFP4, quantized lm_head

NVFP4 build with the output head quantized instead of the usual BF16
(`ignore=["lm_head"]`). Loads and serves in stock vLLM — no source patches —
where ModelOpt-format quantized heads currently fail
(vllm-project/vllm #44081, #45102). Per-token weight reads: 3.23 GB → 2.17 GB.

Full recipe in [`recipe/`](recipe/), raw benchmark tables in
[`data/`](data/).

All measurements: DGX Spark (GB10, ~273 GB/s), vLLM 0.26-series dev, CUDA
graphs on, marlin MoE, fp8 KV. Full flags below.

## Live agent session (unbounded, real traffic)

Multi-hour session through an agent framework (~18k-token system prompt,
persona loaded, tools on, prefix caching). ~28 requests, 15.7k tokens
generated, context to 59k. Decode from engine ITL counters:

| Resident ctx | Generation | TTFT | tok/s |
|---|---|---|---|
| 22k cold | 899 | 5.6 s | 67.0 |
| 22k warm | 517 | 0.13 s | 67.2 |
| 50k (15k fresh) | 709 | 11.3 s | 61.4 |
| 51k warm | 2,052 | 0.14 s | 61.4 |
| 59k | 4,553 (one answer) | 0.85 s | 59.7 |

Scripted 11-turn document session (two ~20k-token docs, code gen,
cross-doc analysis): ~64 tok/s prefill-corrected at 45k, single 5,529-token
answer.

## Context ladder (unbounded, natural stops)

Single-stream, accumulating conversation, no output caps:

| Resident ctx | 26 tok | 2.8k | 6.3k | 13k | 26k | 40k | 53k |
|---|---|---|---|---|---|---|---|
| tok/s | 80.5 | 77.4 | 75.8 | 72.5 | 67.7 | 64.3 | 62.7 |

## Head A/B (isolation, fixed 256-token decode)

Same rig, same flags, only the head differs. Comparison arm: a publicly
available NVFP4 build of the same base with the standard BF16 head.
TPOT-derived decode; fixed output length so the arms are strictly
comparable:

| Input | 2k | 4k | 8k | 16k | 32k | 65k |
|---|---|---|---|---|---|---|
| quantized head | 74.2 | 73.4 | 71.2 | 68.9 | 63.8 | 59.5 |
| BF16 head | 52.1 | 51.7 | 50.6 | 49.3 | 46.7 | 44.1 |

+35–42% at every depth; the ladder and the live session agree at matching
depths. A third arm (uniform-NVFP4 control with the same quantized head) is
in `data/decode_ladder_head_ab.csv`.

## Thinking A/B (unbounded)

`enable_thinking` off/on, same prompts, natural stops: 79.7 → 77.9 tok/s and
78.0 → 77.1 tok/s. Thinking costs volume (1.8–2.5× tokens), not rate.

## Quality

- Tool calling: 8/8 correct decisions, byte-identical arguments vs a
  BF16-head control.
- Side-by-side probes (factual, refusal, code): indistinguishable from base.
- 262,144 context window preserved (config here caps 252k).
- Official MTP drafter compatible (patch included): 88% acceptance. On this
  hardware the drafter's full-context attention makes speculation profitable
  only under ~2k resident context — full off/on ladder in
  `data/mtp_context_ladder.csv`.

## Recipe

Published in full: [`recipe.yaml`](recipe.yaml) as recorded by the build, and
[`recipe/quant_fp8global.py`](recipe/quant_fp8global.py) — the complete
script. Scheme: NVFP4 on every Linear **including `lm_head`**; FP8_DYNAMIC on
the attention projections of the five global-attention layers (5, 11, 17, 23,
29), where massive-activation outliers would blow out NVFP4 block scales; BF16
router/embeddings/norms. Three post-quant config/tokenizer fixups make the
quantized head loadable in stock vLLM — [`recipe/README.md`](recipe/README.md)
documents each.

## Data

Machine-readable copies of every table above, plus recorded results without a
table, in [`data/`](data/):

- [`decode_ladder_head_ab.csv`](data/decode_ladder_head_ab.csv) — the head A/B
  ladder, with a third arm the table above omits: a uniform-NVFP4 control
  carrying the same quantized head. It isolates the head from the
  FP8-global-layers choice — the uniform control runs ~4–5% faster than this
  build (fewer FP8 weight bytes), both arms hold +35–42% over the BF16 head.
- [`context_ladder_unbounded.csv`](data/context_ladder_unbounded.csv) — the
  unbounded single-stream context ladder.
- [`live_agent_session.csv`](data/live_agent_session.csv) — the live agent
  session, per-request: resident context, generation length, TTFT, decode rate.
- [`mtp_context_ladder.csv`](data/mtp_context_ladder.csv) — MTP drafter off/on
  at every context rung. Shows the ~2k crossover: +11% at tiny context,
  progressively unprofitable beyond it (drafter re-attends the full KV per
  draft step; acceptance stays healthy throughout — it is a compute cost, not
  acceptance decay).
- [`data/README.md`](data/README.md) — methodology, dates, and the recorded
  results that have no CSV: thinking on/off rates, MTP acceptance band,
  tool-calling 8/8 with byte-identical arguments, quality probes vs the
  BF16-head control.

## Setup

Requirements: a Blackwell-class GPU and vLLM 0.26+ — its compressed-tensors
0.17 reader is what loads this container (older engines cannot). Measured on
a 0.26-series dev build, GB10/SM121.

```bash
hf download Zek-Takai/gemma-4-26B-A4B-it-NVFP4-fp8global --local-dir gemma4-nvfp4-qhead
```

Serve — stock vLLM, no source modifications required
([`examples/serve_vllm.sh`](examples/serve_vllm.sh) is this command as a
script):

```bash
vllm serve ./gemma4-nvfp4-qhead \
  --served-model-name gemma4-nvfp4-quantized-head \
  --moe-backend marlin \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --max-model-len 252000 \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes $((26 * 1024**3))
```

Flag notes: `--moe-backend marlin` is what was measured on SM121;
`--enable-auto-tool-choice --tool-call-parser gemma4` enables tool calling;
the absolute KV budget is unified-memory hygiene (GPU-fraction settings
overcommit shared pools) — discrete-VRAM GPUs can use standard settings.

Query — OpenAI-compatible endpoint; thinking is controlled per request via
the chat template:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma4-nvfp4-quantized-head",
    "messages": [{"role": "user", "content": "Hello"}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

### Optional vLLM source patches (`patches/`)

Base serving needs no vLLM changes. Two optional lanes do — apply into your
vLLM install, and re-apply after any vLLM upgrade:

```bash
VLLM_ROOT=$(python -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
```

1. **Completions-style pre-templated input** —
   [`patches/vllm_bos_dedupe/`](patches/vllm_bos_dedupe/) collapses the
   doubled leading BOS such input produces, which measurably degrades gemma-4
   (chat requests are unaffected; skip this if you only use
   `/v1/chat/completions`):

   ```bash
   patch -p1 -d "$VLLM_ROOT" < patches/vllm_bos_dedupe/bos_dedupe.patch
   ```

2. **Speculative decoding with Google's official MTP drafter** —
   [`patches/vllm_gemma4_mtp_embed_share/`](patches/vllm_gemma4_mtp_embed_share/)
   lets vLLM share the target's embedding with the
   `gemma-4-26B-A4B-it-assistant` drafter (upstream-reported; see patch
   README). Then serve with:

   ```bash
   patch -p1 -d "$VLLM_ROOT" < patches/vllm_gemma4_mtp_embed_share/embed_share.patch
   ```

   ```
   --speculative-config '{"model": "google/gemma-4-26B-A4B-it-assistant", "method": "gemma4_mtp", "num_speculative_tokens": 1}'
   ```

   Worth it only under ~2k resident context on bandwidth-bound hardware —
   see [`data/mtp_context_ladder.csv`](data/mtp_context_ladder.csv).

## Credits

Quantization recipe, benchmarks, and documentation co-authored with Claude
(Anthropic). Recipe code and benchmark data: Apache-2.0.

## License

Derived from `google/gemma-4-26B-A4B-it`; subject to the Gemma Terms of Use.
