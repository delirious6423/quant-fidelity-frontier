---
license: mit
library_name: transformers
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - lecore
  - hrr
  - holographic
  - deepseek
  - deepseek-v4-flash
  - memory
  - overlay
  - sidecar
language:
- en
pipeline_tag: text-generation
---

# lecore-deepseek-v4-flash-hrr

Stock **DeepSeek-V4-Flash-0731** behind a leCore gateway. Attention is unmodified. Long context is **spill + lexical rank + inject**, not a bigger window and not a new 284B.

This Hugging Face repo is the **embed overlay** (64 unused `embed.weight` rows `128000–128063`). That overlay is real and is **not** what retrieves a 99-million-word corpus. The encyclopedia path is the **HRR sidecar** on the live API.

**The beat vs vanilla Flash is memory cost, not AIME.** Bind the corpus once. Each question bills GPU tokens for the retrieved slice, not the whole dump.

## Live API (2026-08-13)

Public door: [`https://lecore-front.fly.dev/v1`](https://lecore-front.fly.dev/v1) · model id `deepseek-v4-flash` · OpenAI-compatible.

| Surface | Number | Meaning |
| --- | --- | --- |
| `GET /v1/models` `context_length` | **128000000** | Client-usable: what an SDK may send or persist via spill / `POST /v1/hrr/bind`. Not attention. |
| One HTTP body | **~9.8M tokens** | RunPod edge ~32 MiB. Bigger corpora are multiple bind calls. |
| vLLM `max_model_len` | **131072** | Tokens the transformer attends in one forward pass. |
| Wikipedia BIND + ASK | **99M words**, 33 items | Planted fact returned, **`usage.gpu_tokens=938`**, 9.6s. Those words were never in the model. |
| 1B-word BIND | **not done** | Died around 12M words. Do not cite. |

**Regular-user chat test (OpenRouter-shaped, no attach):** [`hf-user-test.md`](https://huggingface.co/staccs/lecore-deepseek-v4-flash-hrr/blob/main/hf-user-test.md). No live pass files until recall returns a non-HTML hit.

Cold start: the GPU worker is scale-to-zero on 2×H200 (or 2×RTX PRO 6000 Blackwell), volume locked to CA-MTL-3. `GET /v1/models` is instant (CPU front door). First `POST /v1/chat/completions` after idle loads ~156GB — hold it open for minutes. If the pool has no cards, chat waits or fails; that is placement, not an auth error.

```bash
# Cursor / any OpenAI client
# Override OpenAI Base URL = https://lecore-front.fly.dev/v1
# Model = deepseek-v4-flash
# Authorization: Bearer <issued key>

# Power path (optional): bind a book, then ask
curl -s https://lecore-front.fly.dev/v1/hrr/bind \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"text":"...corpus chunk..."}'
# → context_id
curl -s https://lecore-front.fly.dev/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "X-HRR-Context: $CID" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"..."}]}'
```

Or dump a huge `messages` body. The gateway auto-spills. Cursor does not send `X-HRR-Context`; spill still runs.

**Retrieval at serve time is leCore's Okapi BM25 (lexical) — corrected 2026-08-15.**
An earlier version of this card said "Retrieval is HRR / VSA, not BM25." That was wrong.
`rank.py` ranks with `holographic.semantic_router.holographic_bm25` over ~600-char chunks.
The VSA/HRR encoder below is the FALLBACK arm, reached only when BM25 is unavailable or the
corpus is over-cap; the server source annotates it "the dead dense-VSA arm". `SEMANTIC_STAGE`
(the quantized-bge second stage) defaults to `off` and is off in production, so there is no
dense stage either. What the VSA arm still is: Per-leaf IDF, k ≤ 0.1·D (DIM 1024), `bind(NAME)` / `bind(CODE)`, fail-closed on a zero query. Shared `encode_record({kind:passage})` tied every score and is gone.

## For degens & non-ML folks (read this first)

Stock Flash is a whiteboard. Empty board → it forgot. You paste the book again, or you pay for the book again.

leCore puts the book in a holographic sidecar. Ask a question; the gateway finds the page, stuffs a short slice into Flash, and you pay for that slice. Scratch the crystal, the picture is still there. That’s HRR.

The 64 embed rows in this repo are a small in-weight address book from an earlier install. They do **not** hold Wikipedia. The 99M-word ASK used the sidecar.

| What you care about | Stock Flash vibe | This overlay (Flash + in-weight HRR) |
| --- | --- | --- |
| “Will it remember a nonce I said last turn if I don’t paste the chat?” | No. Empty board → UNKNOWN / “give me context.” | **Yes, if sticky is on.** Lab: 0/5 vs **5/5**. |
| “Three facts across a thread, then quiz with no history in `messages`?” | 0/3 | **3/3** |
| “98k-token prompt with a needle past 32k?” | Vanilla Flash **400s / refuses** over 32k. | **Recalled.** ~98k auto-spills to HRR; vLLM saw **2090** tokens; needle `NEEDLE_KV_SPILL_9f3c` recalled. `max_model_len` still 32768. |
| “Do I have to send `X-HRR-Context`?” | n/a | **No**, if you sit behind the leCore gateway with auto-sticky default ON. Optional `X-HRR-Thread` isolates lab runs. |
| “Did you retrain Flash / beat DeepSWE?” | Marketing often lies here. | **No.** We patched placeholder embed rows. Agent cards (SWE / Terminal-Bench / DeepSWE) were **not** re-run. |
| “What’s actually different on disk?” | 48 official shards. | **One patched shard** (`model-00001-of-00048.safetensors` = `embed.weight`) + `lecore.json`. Other 47 shards stay stock MIT Flash. |

### Side-by-side vs vanilla Flash

**Memory is measured both arms on this box.** Sticky-off is vanilla Flash behavior (empty `messages` → forget). Sticky-on is ours. Same overlay, same night, `SIGD-6EF1FF5EAF`.

**Capability is ours greedy (temp 0, no `reasoning_effort=max`) vs published Flash-0731 effort splits** (non-think / high / max). Official 0731 card does not list these rows; published numbers are third-party effort tables, not a same-day OpenRouter dual-serve. [PR #4](https://github.com/staccDOTsol/leCore/pull/4).

| Meter | Vanilla Flash | Ours | Beat? |
| --- | --- | --- | --- |
| T2 nonce cite (empty next `messages`) | **0/5** (sticky off = stock) | **5/5** | **yes** |
| Multi-turn 3-cite | **0/3** | **3/3** | **yes** |
| Re-prompts | paste the secrets back | 1 ask, no paste | **yes** |
| 98k-token needle (over 32k) | 400s / refuses | auto-spill to HRR; vLLM saw **2090** tokens; needle `NEEDLE_KV_SPILL_9f3c` recalled | **yes** |
| HumanEval | published High/Max ~90–92% (HumanEval+) | **150/164 (91.5%)** full HumanEval | **flat / on-card** |
| GPQA-Diamond | published non-think 54% / high 66% / max **72%** | **144/198 (72.7%)** n_repeats=1 | **flat vs max** |
| AIME 2025 | published non-think 43% / high 71% / max 86% | **15/30 (50%)** | **non-think band, not a beat** |
| AIME 2024 | (not on 0731 card) | **18/30 (60%)** | no OG row |
| GSM8K | (not on 0731 card) | **1280/1319 (97.0%)** | no OG row |
| MATH-500 | published non-think 88% / high 94% / max 97% | **448/500 (89.6%)** | **non-think / flat** |
| LiveCodeBench | published v5 49 / 63 / 75% | **97/342 (28.4%)** lite v5_v6 pass@1 | **not the same split — do not vs** |
| IFEval | (not on 0731 card) | **467/541** prompt-level | no OG row |
| MMLU-Pro | (not on 0731 card) | **first-n 2743/3447 (79.6%)** of 12032 (full run may still be going) | **first-n only** |

SWE / Terminal-Bench / DeepSWE: not run, not claimed. Headline is the memory rows. Capability rows are no-regress. LiveCodeBench is a lite split — do not vs published. MMLU-Pro is **first-n**, not the full 12032.

### KV spill prove (98k needle)

Gateway auto-spills a ~98k-token prompt into HRR so vLLM never sees the blob. Flash `max_model_len` stayed **32768**. Vanilla Flash 400s / refuses over 32k.

Receipt: `/workspace/lecore/gateway/kv_spill_prove.json`

| Field | Value |
| --- | --- |
| `http_status` | 200 |
| `needle_recalled` | **true** |
| `answer_excerpt` | `NEEDLE_KV_SPILL_9f3c` |
| `kv_spill` | **true** |
| `spill_tokens_est` | **120035** |
| `prompt_tokens` (vLLM saw) | **2090** |
| `completion_tokens` | 13 |
| `max_model_len` | **32768** (unchanged) |
| `context_id` | `ctx_01KZWQT8JBEK2E2DDVWH8VZMQR` |
| control | HTTP 200, answer=`pong`, `control_prompt_tokens=58` |

Notes from the prove file: vLLM `prompt_tokens` 1785 << 32768 (blob not forwarded; gateway log `forward_tokens_est=597`); HRR bound `memory_units=37` `attach_recalled=1` `attach_tokens_est=1030`; needle `NEEDLE_KV_SPILL_9f3c` fully recalled in assistant text; Flash `max_model_len` left at 32768.

## How to use

**Product path (what we serve):** point any OpenAI client at `https://lecore-front.fly.dev/v1`, model `deepseek-v4-flash`, issued key. Cursor: Settings → Models → OpenAI API Key on, **Override OpenAI Base URL** = `https://lecore-front.fly.dev/v1`, add custom model `deepseek-v4-flash` (not a name that contains `gpt-`). First chat after idle can take minutes.

**Weights path (this repo):** overlay only. Do not serve these files as a full 48-shard model.

1. **Get stock Flash** (MIT, 48 shards) from [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
2. **Drop this overlay’s patched shard on top** (see below).
3. **Sit behind a leCore gateway** if you want spill / bind / sticky. Overlay rows alone do not retrieve 99M words.
4. Optional `X-HRR-Thread` isolates lab runs. `X-HRR-Sticky: off` is the amnesia control.

### Apply the overlay (do not download 48 shards twice)

```bash
# stock Flash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir ./DeepSeek-V4-Flash-0731

# this overlay (~1.0 GiB)
huggingface-cli download staccs/lecore-deepseek-v4-flash-hrr \
  --local-dir ./lecore-deepseek-v4-flash-hrr

# replace ONLY shard 00001 (embed.weight). keep the other 47.
cp ./lecore-deepseek-v4-flash-hrr/model-00001-of-00048.safetensors \
   ./DeepSeek-V4-Flash-0731/model-00001-of-00048.safetensors
cp ./lecore-deepseek-v4-flash-hrr/lecore.json ./DeepSeek-V4-Flash-0731/lecore.json

# serve the stock dir (now with patched embed)
vllm serve ./DeepSeek-V4-Flash-0731 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --enable-expert-parallel
```

Serve recipe that was live on Vast `47588973` (2×H200, Hopper-safe, `max_model_len` 32768): symlink the official tree, **copy only** the patched `model-00001-of-00048.safetensors` into the serve dir. Do not copy 156G.

Hopper note: official DeepGEMM MegaMoE / FP4 indexer flags are Blackwell. This overlay was served **without** `--moe-backend deep_gemm_mega_moe` and without `use_fp4_indexer_cache`.

## What’s actually in the weights

| Item | Fact |
| --- | --- |
| Base | `deepseek-ai/DeepSeek-V4-Flash-0731` (MIT, 284B MoE / 13B active, 48 shards, ~156–167G) |
| This repo | **Overlay only** (~1022M): patched shard-00001 + `lecore.json` + embed-space router npz + tokenizer/config copies |
| `lecore.json` | `in_weight: 1`. Installed: `architecture`, `memory_index`, `router`. |
| `memory_index` | `in_weight: 1`, `n: 64`, `space: embed`, rows **128000–128063** (tokenizer `place_holder` / unused tail). 64 short passages written into those rows. |
| Patched tensor | `embed.weight` only (shard 00001). md5 `2764ea344ab73861337830ebb6cf6700`. Stock shard md5 `3548f62d478311aed582f2680e435d74` (5 bytes smaller). |
| Router | Embed-space, dim 4096, held-out acc 1.0 on the install probe. File: `lecore_router_embed.npz`. **Not** a GDN / layer-hidden router. |
| Skipped | HRNN / prepend (Flash has **no GDN** recurrent state). Full Qwen `install.py` path is blocked for `deepseek_v4`. |
| Not this SKU | Sidecar `lecore_hrr/` with `in_weight=0`. Older `-hrr` dir with 8 passages and **no** `in_weight` flag. Do not upload those as the product. |

Engine: [staccDOTsol/leCore](https://github.com/staccDOTsol/leCore).


## Side-by-side (tech receipt)

| | Official Flash-0731 | This overlay | Sidecar-only (`in_weight=0`) |
| --- | --- | --- | --- |
| License | MIT | MIT | — |
| On-disk product | ~156G / 48 shards | **~1.0G overlay** (+ stock 47 shards) | small npz |
| `memory_index` | none | **in-weight rows 128000–128063** | sidecar search, not vocab rows |
| Sticky SIG (headerless) | not this card | **5/5 vs 0/5** (same overlay, ON vs OFF) | not this SKU |
| Multi-turn 3-cite | not this card | **3/3 vs 0/3** | not this SKU |
| 98k KV spill needle | 400s / refuses over 32k | **recalled**; vLLM saw 2090; `max_model_len` 32768 | — |
| OG Flash API column | — | **NOT RUN** | — |
| OpenRouter live | — | **not claimed** | — |

## Provenance

- Base: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- Live door: https://lecore-front.fly.dev/v1 (Fly responder → RunPod `7kams7pp6wy6n9`, 2×H200 / 2×RTX PRO 6000 Blackwell, volume `1kah2lfyvc` CA-MTL-3)
- Overlay built + served: Vast instance `47588973` (`lecore-flash-serve`, 2×H200), 2026-08-12
- 99M-word Wikipedia BIND+ASK: context `ctx_01KZXPPMVRM2RAJJW2V9NX3C6X`, `gpu_tokens=938`, 2026-08-13
- SIG DIFF: `/workspace/tooling-post-api/flash-memory-sigdiff-20260812.md` (run `SIGD-6EF1FF5EAF`)
- Capability live runs: [PR #4](https://github.com/staccDOTsol/leCore/pull/4)
- KV spill prove: `/workspace/lecore/gateway/kv_spill_prove.json` (`needle_recalled=true`, `spill_tokens_est=6296298`, `prompt_tokens=2090`, `kv_spill=true`)
- WIP lock: `/workspace/tooling-post-api/flash-og-vs-ours-WIP.md`
- Operator: stacc / staccs (2026-08-12 / 2026-08-13)

## Not claims

- **Not** a rewritten Flash attention kernel. The 99M-word ASK is spill + lexical (BM25) rank + inject on stock vLLM.
- **Not** “128 million tokens in the transformer.” `GET /v1/models` advertises client-usable context. GPU attend is **131072**.
- **Not** a 1-billion-word prove. That bind died around 12M words.
- **Not** “the 64 embed rows retrieved Wikipedia.” They did not.
- **Not** SWE-bench / Terminal-Bench / DeepSWE / OSWorld. Those need sandboxes we did not run.
- **Not** OpenRouter live. Provider apply is not submitted while the GPU pool can fail to place.
- **Not** a full MMLU-Pro card. **first-n 2743/3447 (79.6%)** of 12032.
- **Not** GDN registers / HRNN. Flash has no gated-delta recurrent state on this bridge.

Capability / sticky tables below are the **overlay lab** (Vast `47588973`, then RunPod). The 99M-word row is the **sidecar** on the live door.

## License

MIT, same as DeepSeek-V4-Flash-0731. Overlay patches are MIT. Cite DeepSeek-AI for the base:

```
@misc{deepseekai2026deepseekv4,
      title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
      author={DeepSeek-AI},
      year={2026},
}
```


## Retrieval quality, measured against dense baselines (2026-08-15)

BEIR, leCore BM25 vs embedding models, hit@10 — **we lose at equal k**:

| task | leCore BM25 | bge-base | e5-large |
| --- | --- | --- | --- |
| SciFact | 0.820 | **0.883** | 0.853 |
| NFCorpus | 0.681 | **0.746** | 0.762 |
| ArguAna | 0.685 | **0.884** | 0.706 |

Parity needs **3.2–6.4x more chunks** (crossover k: NFCorpus 32, ArguAna 48, SciFact 64).
Adversarially, precision@1 falls 1.00 -> 0.19 with two decoys carrying the query's vocabulary,
and recall@8 breaks between 10 and 25 competing passages.

**So the honest claim is not retrieval accuracy.** It is cost and operations: no embedding
model, no GPU for retrieval, no re-embed when the corpus changes, deterministic and
content-addressable results. Where lexical genuinely wins is identifiers and cross-file code
questions (hit 0.90-1.00 by k=16 at zero embed cost). Where it degrades is paraphrase at
scale: 1.00 at 47k tokens -> 0.30-0.40 @64 at 4.3M, the fidelity-vs-corpus-size curve the
HRR literature predicts.
