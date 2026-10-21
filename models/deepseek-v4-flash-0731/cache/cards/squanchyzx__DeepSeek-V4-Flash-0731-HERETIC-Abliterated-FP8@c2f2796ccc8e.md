---
license: mit
library_name: transformers
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - deepseek-v4
  - abliterated
  - heretic
  - fp8
  - reasoning
  - tool-use
---

# DeepSeek-V4-Flash-0731 HERETIC Abliterated FP8 — v2

A native-FP8 behavioral derivative of [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

**v2 replaces the original λ1.5 release on `main`.** The original commit remains available as an immutable tagged revision because it developed a long-context degeneration failure in real agent histories. Do not deploy the old revision for long-lived agent sessions.

## What was wrong with v1

The issue was not a single repeated word. In long tool/agent histories, v1 could fall into several degeneration patterns:

- repeated words such as `kanka`, `my`, or similar tokens;
- repeated CJK characters and unexpected script switching;
- thousands of empty `()` fragments or repeated punctuation;
- repeated lines and low-entropy n-gram loops;
- 100K–200K-character malformed generations that poisoned subsequent history.

The failure came from an overly aggressive rank-3 edit at `lambda=1.5`, combined with speculative/MTP amplification. Merely restoring stock MTP was not enough: the λ1.5 stock-MTP control still produced a 324-character consecutive `我` run.

## What changed in v2

- Kept the same three-mode rank-3 refusal subspace for `chat`, `think-high`, and `think-max`.
- Edited attention output projections in backbone layers `10–42`.
- Reduced the projection strength from `lambda=1.5` to `lambda=1.35`.
- Left all MTP/DSpark tensors stock and untouched.
- Left routed experts, shared experts, embeddings, norms, heads, tokenizer, encoder, and model configuration untouched.
- Re-baked once from the protected official base; v2 was not baked on top of v1.
- Preserved the official native FP8 representation; no additional lower-bit quantization pass was applied.

## Tool attribution

This checkpoint was produced with [Heretic](https://github.com/p-e-w/heretic) v1.4.0 by Philipp Emanuel Weidmann and contributors, using base commit `7675b90d648154cdfefa597372cc477df9848eab`. Heretic is their project and is licensed `AGPL-3.0-or-later`; it is not owned or authored by this model publisher.

The release-specific work here is the DeepSeek V4/mHC two-node adaptation, separate refusal-direction capture, rank-3 subspace construction, native-FP8 attention-only bake, long-context incident reproduction, generic degeneration detector, and release evaluation. See `HERETIC_ATTRIBUTION.md`.

## Mechanical receipts

- Edited tensors: `66` (`33` weight + scale pairs)
- Untouched indexed tensors: `72,251`
- MTP/DSpark edited tensors: `0`
- Overlay bytes: `1,107,370,672`
- Overlay SHA-256: `ca4a043ae3a306a50b680a08c3f175d7a7df20b955f28a9d4e45161692c45638`
- Rewritten index SHA-256: `393a1b09ffacdf5e4e226a84f0b8d5467947ad0f35c91a71127e609003375a56`
- Cross-node overlay/index equality: verified on two independently materialized DGX Spark trees
- Dangling official-base shard links before packaging: `0/48` on both nodes

## Long-context acceptance

The detector does not key on one literal word. It evaluates answer and reasoning content for same-word/character/symbol runs, repeated 2/3/4/8-grams, repeated lines, dominant-token fraction, vocabulary uniqueness, entropy, compression, empty-parenthesis floods, and unexpected script switching.

Results with speculative decoding enabled:

- Real cron incident fixture, `92,501` prompt tokens: `5/5` seeds clean
- Corrected real WhatsApp fixture, `64,245` prompt tokens: `5/5` seeds clean
- Synthetic `124,983` prompt-token replay: clean
- Synthetic `249,971` prompt-token replay: clean
- Total long-history acceptance replays: `12/12` clean
- Non-stop finishes: `0/12`
- Tool calls: `6/6` valid and correct
- Deterministic short quality smoke with a 512-token generation budget: `11/12`
- Refusal smoke: `6/16` (`37.5%`; lower is less refusal)

These are bounded release tests, not a universal capability or safety benchmark. Full compact receipts are in `EVAL_RESULTS.md` and the `eval/v2/` directory.

## Runtime

Use a DeepSeek-V4-compatible runtime and the official encoder. This checkpoint retains the official V4/DSpark architecture and native FP8 layout. Generic runtimes without DeepSeek V4 support may not load it correctly.

The release candidate was exercised on two NVIDIA DGX Spark nodes with tensor parallelism `2`, a 1,048,576-token configured context window, the official V4 encoder, and a DeepSeek-V4-capable vLLM build.


## 2026-08-12 runtime qualification and stock-vs-v2 A/B

**No checkpoint revision was created for this update.** The v2 weight shards, index, configuration, tokenizer, and immutable `v2-lam1p35-mtpstock` tag are unchanged. This section documents a verified runtime profile and a bounded matched comparison against the unmodified official `DeepSeek-V4-Flash-0731` checkpoint.

### Runtime profile

The qualified two-DGX-Spark deployment uses the runtime lineage from [`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) at commit [`0817b11f5bfd53aeb85c90d1e8ae3755898b5314`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/commit/0817b11f5bfd53aeb85c90d1e8ae3755898b5314). That repository supplies runtime/deployment fixes; it is not a separate Mia-tuned checkpoint.

The validated profile kept this repository's v2 weights and used TP=2, MTP=3, a 1,048,576-token configured window, the official encoder, regular CUDA graphs, and the NVFP4 sparse-MLA fix. Direct API gates passed `5/5`, including strict JSON and tool-history re-encoding. Long-context gates completed at `66,637`, `249,971`, and `609,847` prompt tokens without OOM, engine death, traceback, HTTP 5xx, or container restart. The 609,847-token run decoded at `38.768 tok/s`, but its prefill/TTFT was `541.203 s`; the configured 1M window should not be presented as low-latency interactive UX.

The qualified deployment currently uses host-mounted patch files. An immutable runtime image and mount-independence receipt remain future work.

### Matched stock-vs-v2 result

Both arms were cold-loaded into the same TP=2 runtime envelope with identical serving flags, tokenizer, prompts, seeds, sampling, MTP=3, and default `thinking=false` chat-template behavior.

- Strict objective/instruction suite: official stock `14/20`; Heretic v2 `11/20`.
- Stock-only wins: `3`; Heretic-only wins: `0`; exact paired McNemar two-sided `p=0.25`.
- Tool selection: both `6/6`.
- Deterministic 512-token quality smoke: both `11/12`.
- Legacy bounded short-generation quality smoke: stock `10/12`; Heretic v2 `9/12`.
- Refusal smoke: stock `15/16`; Heretic v2 `6/16` (lower means less refusal).
- Blind qualitative majority (3 judges, 12 randomized pairs): stock `6`, Heretic v2 `2`, ties `4`; mean scores stock `7.500`, Heretic v2 `7.417`.

Interpretation: stock showed a directional edge on this small strict objective suite, while v2 preserved much lower refusal behavior and equal tool-call compliance. The sample is too small to establish a universal capability gap. Use the official stock checkpoint when upstream-aligned behavior and maximum baseline correctness are the priority; evaluate this derivative when reduced refusal behavior is an explicit requirement. Do not treat abliteration as a free quality improvement.

Machine-readable receipts: [`matched-ab-summary.json`](eval/runtime-0817/matched-ab-summary.json), [`blind-judge-summary.json`](eval/runtime-0817/blind-judge-summary.json), [`suite-manifest.json`](eval/runtime-0817/suite-manifest.json), and [`runtime-qualification.json`](eval/runtime-0817/runtime-qualification.json).

## Revision policy

- `main`: current v2 release
- `v2-lam1p35-mtpstock`: immutable v2 tag
- `v1-lam1p5-known-long-context-issue`: preserved v1 tag; not recommended for long-lived agent sessions

## Limitations and safety

- This is an unofficial community derivative and is not affiliated with or endorsed by DeepSeek.
- Abliteration reduces some refusal behavior and can increase compliance with unsafe requests. This checkpoint is not safety-certified.
- Downstream deployers remain responsible for evaluation, access controls, monitoring, output limits, loop guards, and compliance with applicable law and platform policy.
- Refusal and long-context behavior remain prompt-, template-, runtime-, and decoder-sensitive.
- The quality suite is intentionally small. Broader benchmark coverage is welcome.
- A model-side repair does not replace runtime output caps and generic degeneration guards.

## License

The upstream repository and model weights are MIT licensed. This derivative retains the MIT license and original DeepSeek copyright notice.

Base model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
