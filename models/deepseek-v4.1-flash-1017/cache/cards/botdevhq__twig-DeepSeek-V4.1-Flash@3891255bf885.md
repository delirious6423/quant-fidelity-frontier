---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags: [mlx, apple-silicon, quantized]
---

# twig-DeepSeek-V4.1-Flash

A mixed-precision pack of DeepSeek-V4.1-Flash-1017 from [bot.dev](https://bot.dev) (`@botdevhq`). It keeps 5,927 frequently used experts in native MXFP4, stores the other 9,433 at two or three bits, and keeps the DSpark draft stages in native MXFP4. All 15,360 routed experts fit in memory. The two Engram tables stay on SSD. Tested on an Apple M5 Ultra (256 GB) in our lab.

## How we measure

We optimize for real-world usage (cached follow-up turns in coding sessions, cold prompt reading at long context, steady per-user speed under concurrency, verified output fidelity, and safe memory headroom) rather than cherry-picked benchmark showmanship. Every number below states its method: drafting on or off, context length, cold or cached prompt state, and greedy (`T=0`) or sampled (`T=1.0`, `top_p=0.95`) decoding. Every speculative number is labelled **drafted** next to its **drafting-off (plain)** baseline. All serving runs use exact verification (`--verify-exact served,fused`) so greedy tokens and log probabilities with drafting on match drafting off bit for bit. Anything not yet tested on a given release build is marked **unmeasured**.

## Real-use results (Apple M5 Ultra, 256 GB)

### Unified mlx-serve 26.11.4 headline results (`botdevhq/mlx-serve`, commits `34368bc9` / `a38ec95b`)

Measured on this exact unchanged weight pack using our unified [botdevhq/mlx-serve](https://github.com/botdevhq/mlx-serve) `26.11.4` engine build (rebased onto `ddalcu/mlx-serve` `main` with David Tai's PR #1256 lossless prefill improvements and exact mixed-precision decode):

| Metric | Method (drafting, context, cache, sampling) | Measured result |
|---|---|---:|
| Decode by kind: code / prose / edit / reasoning | Drafted (DSpark), fixed cold prompts (256–700 tok out), 0% cached, greedy `T=0`, 1 user | **149.8 / 90.1 / 186.7 / 122.7 tok/s** *(149–187 tok/s code/edit, drafted)* |
| Decode baseline: drafting off (plain) | Drafting off (plain), same fixed cold prompts, 0% cached, greedy `T=0`, 1 user | **70.2 tok/s** *(plain)* |
| Prompt reading & cold TTFT: 8K / 32K / 128K | Drafting on, 8,198 / 32,759 / 131,050 tok, 0% cached (cold nonce), greedy `T=0`, 1 user | **3,193** (2.57 s) / **3,923** (8.35 s) / **3,965** (33.05 s) tok/s |
| Cached coding-turn TTFT (prefix reuse) | Drafted (DSpark), 14K–15.6K ctx, 94.3% cached (+835–889 new tok), greedy `T=0`, 1 user | **0.50 s** *(0.49–0.50 s, 109.5–111.0 tok/s drafted, 0.019 s near-full hit)* |
| Bit-identical output vs. drafting off | `--verify-exact served,fused`, all 4 fixed kinds (`on == off == on2`), greedy `T=0` | **0/4 greedy diffs** *(code & edit match v0.1 verbatim)* |
| Memory headroom (MLX peak / wired peak / GPU limit) | Up to 128K ctx, chunk 4096, 5 GiB buffer cap, 512 MiB RAM prefix cache, Nominal thermal | **210.5–211.3 GiB MLX** / **218.9–219.1 GiB wired** / 224 GiB limit |

### Full v0.1 validation matrix (same pack weights, initial v0.1 build)

The full quality, concurrency, sampled-decoding, and 250K context validation matrix was measured on this exact pack using the initial v0.1 engine build:

| Metric | Method (drafting, context, cache, sampling) | Measured result |
|---|---|---:|
| Cached coding-turn TTFT (prefix reuse) | Drafted (DSpark), 4.8K–37.9K ctx, 89.9–98.6% cached (+532–640 new tok/turn), greedy `T=0`, 1 user | 0.67–0.75 s *(86.4–105.9 tok/s, drafted)* |
| Cold coding-turn TTFT (first turn) | Drafted (DSpark), 4.8K / 9.6K / 18.4K / 36.8K ctx, 0% cached (cold), greedy `T=0`, 1 user | 2.91 / 4.24 / 6.75 / 11.17 s *(1,675–3,305 tok/s)* |
| Prompt reading & cold TTFT: 8K / 32K / 128K | Drafting on, 8,198 / 32,759 / 131,050 tok, 0% cached (cold nonce), greedy `T=0`, 1 user | 2,461 (3.3 s) / 3,313 (9.9 s) / 3,400 (38.5 s) tok/s |
| Prompt reading & memory: 250K | Drafting on, 249,998 tok (`262,144` ctx), 0% cached (cold), greedy `T=0`, 1 user | 2,379.7 tok/s (105.1 s, incl. ~32.5 s cold Metal JIT) · 94.4 tok/s decode *(drafted)* · 209.8 GiB MLX / 218.41 GiB wired *(Nominal)* |
| Per-user speed: 1 / 2 / 3 / 4 users | 1u drafted, 2–4u yield to exact plain (`--max-concurrent 4`), cached coding (~60K active KV), greedy `T=0` | 1u 96.8 (drafted) · 2u 34.0–35.3 · 3u 22.9–24.0 · 4u 17.7–18.2 tok/s/user *(agg: 96.5 / 68.2 / 68.2 / 67.9)* |
| Decode by kind: code / prose / edit / reasoning | Drafted (DSpark), fixed cold prompts (256–700 tok out), 0% cached, greedy `T=0`, 1 user | 140.1 / 85.6 / 170.5 / 119.6 tok/s *(drafted)* |
| Decode by kind, 3-run median: code / prose / edit / reasoning | Drafted (DSpark), nonced cold prompts (6.1K code/prose, 1.1K edit, ~120 tok reasoning), 0% cached, greedy `T=0`, 1 user | 142.5 / 91.8 / 174.0 / 124.0 tok/s *(drafted)* |
| Decode baseline: drafting off (plain) | Drafting off (plain), same fixed cold prompts (and 3-run 6.1K cold median), 0% cached, greedy `T=0`, 1 user | 68.4 tok/s *(3-run median: 67.6)* |
| Sampled decode: coding / prose (drafted vs. plain) | Drafted (stochastic DSpark) vs. drafting off, cached 4.8K–19.6K coding & prose, `T=1.0`, `top_p=0.95`, 1 user | 94.4 vs. 65.5 (1.44×) / 71.3 vs. 67.6 (1.05×) tok/s |
| Bit-identical output vs. original path | `--verify-exact served,fused`, probe `k=2..8` / fixed 4K–128K / draft `on=off` (4 kinds) / 1–4u, greedy `T=0` | 0/702 (`max \|Δ\|=0`) · 8/8 · 4/4 · 0/16 *(standalone = overlay 61/61 logits)* |
| Quantization quality (KL / assistant top-1) | Drafting off (teacher-forced), 150 held-out 8K seqs (42,250 assistant pos, top-K + tail) / engine KL vs. original checkpoint | 0.00718 KL / 98.18% top-1 *(engine KL: 0.007246, native 512 GB: 0.00448 / 98.64%)* |
| Tool calling (BFCL V4 AST & smoke) | Drafting on, 210 AST items + 10 smoke cases, high reasoning effort, greedy `T=0` | 166/210 (79.05%), smoke 10/10 *(native 512 GB: 164/210, 78.10%)* |
| Memory headroom (MLX peak / wired peak / GPU limit) | Up to 250K ctx, chunk 4096, 5 GiB buffer cap, 512 MiB RAM prefix cache | 210.1 GiB MLX (209.8 GiB at 250K) / 217.4–218.4 GiB wired (220.8 live) / 224 GiB limit |

Measurement notes:

- **Prefix-cache hits:** Near-complete prefix hits (99.98%+ reused, 1 new token) measured 0.019 s TTFT on `26.11.4` and 0.02–0.17 s TTFT at 6K–19.6K context on `v0.1` (2.67 s on a 37.9K turn queued behind 3 concurrent sessions at N=4).
- **Concurrency:** In the `v0.1` validation matrix, multi-user runs used `--max-concurrent 4` on four cached coding sessions (~60K active KV across 6K/10.8K/19.6K/37.9K contexts, 0/16 greedy mismatches). With `--verify-exact served,fused` in `v0.1`, `N >= 2` yields drafting to exact single-row decode, keeping aggregate throughput near the plain ceiling (~68 tok/s). A 4-user mixed burst (3 cached + 1 cold 18.6K prompt) reached 46.7 tok/s aggregate (13.3–27.6 tok/s/user, 3/3 cached exact). Sampled 4-user runs (`T=1.0`, `top_p=0.95`) reached 17.3–19.0 tok/s/user on coding (68.0 agg decode vs. 66.7 plain) and 62.5 agg decode vs. 69.2 plain on prose. Unified `26.11.4` also adds exact batched decode (`--exact-batch --exact-batch-max 8`). Four-user 128K active KV is **unmeasured** on 256 GB.
- **Standalone pack & BFCL:** The standalone pack matched the overlay layout bit for bit across 600 prompt rows × 129,280 logits, 16 decode rows, 24 DSpark tensors (`61/61` tests), `8/8` fixed prompts (answers + top-20 logprobs), `12/12` `dident` runs (4,911 tokens), and 215,160 MiB MLX peak. The BFCL V4 AST score (166/210, 0 errors) was measured on that bit-identical overlay layout, and the standalone pack passed 10/10 tool-call smoke checks directly.
- **250K screen & remaining unmeasured cases:** On `v0.1`, a cold 249,998-token run (`262,144` context window, 0% cached, greedy `T=0`, `--verify-exact served,fused`) passed within the 256 GB memory envelope at 218.41 GiB wired (< 221.0 GiB), 214,810 MiB (209.8 GiB) MLX active (<= 217,907 MiB), Nominal thermal, 94.4 tok/s drafted decode, and 2,379.7 tok/s (105.1 s) cold prefill (including ~32.5 s of first-4,096-row Metal shader JIT, with its 832.6 MB snapshot spilling cleanly to disk). On `26.11.4`, 250K cold prefill is currently **unmeasured** (the 128K screen sat at 218.92 GiB wired). Full 262K-token prompts and 4-user 128K active KV remain **unmeasured**.

## Calibration mix and fit

The 5,927 hot native-MXFP4 experts (covering **80.6%** of calibration picks across **35% 8K / 35% 32K / 30% 128K+** contexts) were selected from English real-use workloads:

| Category | Share | Primary focus |
|---|---:|---|
| Coding & agentic software engineering | 38% | Multi-turn repo edits, diffs, refactoring, tool calling, build/test repair |
| Systems, DevOps & CLI operations | 25% | Shell debugging, logs, infrastructure configs, troubleshooting |
| General chat & creative / interactive | 15% | Conversational Q&A, brainstorming, interactive writing |
| Technical research & synthesis | 12% | Long-context docs, architecture trade-offs, technical analysis |
| Security review & auditing | 7% | Code and configuration audits, hardening, threat modeling |
| Email & business writing | 3% | Drafting, summarizing, concise English prose |

- **Best fit:** English coding agents, refactoring, DevOps, tool calling, and long-context technical work (~70% of calibration weight, while dense, attention, shared-expert, and DSpark layers remain native precision).
- **Less ideal:** Heavy non-English (multilingual/CJK), specialized math/legal/medical domains, or creative roleplay. Those workloads route more frequently into the 9,433 2–3 bit cold experts. On 512 GB, use the native all-hot layout instead (zero cold experts, workload-neutral).

## Memory and headroom

- **Resident footprint:** The pack keeps ~206.6 GiB (211,574 MiB) MLX active at idle (~211–213 GiB wired, engine RSS ~200 GiB). Up to 250K context (249,998 tokens in a 262,144-token window on `v0.1`) or 128K context on `26.11.4` with 4,096-token prefill chunks, a 5 GiB MLX buffer cap, and a 512 MiB RAM prefix cache (spilling larger snapshots such as the 832.6 MB 250K state to disk), MLX peaked at 210.1–211.3 GiB and system wired memory peaked at 217.4–219.1 GiB in controlled screens (220.8 GiB peak in continuous serving, 216.9 GiB with four concurrent cached sessions, Nominal thermal). The two Engram shards (189.1 GiB) stay on internal SSD via `pread`, guarded by `--engram-mmap`'s 90% residency threshold.
- **GPU wired limit:** Set `sudo sysctl iogpu.wired_limit_mb=229376` (224.0 GiB, resets on reboot) before starting the server.
- **Headroom and dedicated use:** Roughly 35–44 GiB remains outside the wired model footprint (~32 GiB above the 224 GiB GPU ceiling) for macOS, the kernel, and filesystem cache (~14–16% free memory). Treat a 256 GB Mac running this pack as a dedicated server, and avoid running browsers, IDEs, or other memory-heavy applications alongside it.
- **Headless FileVault note:** On a FileVault Mac unlocked over SSH with no desktop login, macOS's `audiomxd` and `configd` can spin. Log in at the desktop once to clear them.

## Run

Build the unified [botdevhq/mlx-serve](https://github.com/botdevhq/mlx-serve) engine (see its `docs/building.md`). Then use the Hugging Face CLI to download this pack (209.38 GiB of payload across 90 files including `hybrid.json`) and the two original Engram files (189.1 GiB). Keep both on the internal SSD and allow additional space for the prompt cache.

```sh
hf download botdevhq/twig-DeepSeek-V4.1-Flash --local-dir pack
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 08c8657f02291e11b6f6b63e0e16014f8e748518 --local-dir pack/engram
(cd pack && shasum -a 256 -c SHA256SUMS)
sudo sysctl iogpu.wired_limit_mb=229376
export MLXS_API_KEY="$(openssl rand -hex 32)"
export MLX_SERVE_CACHE_LIMIT=$((5 << 30))
./zig-out/bin/mlx-serve \
  --model ./pack --host 127.0.0.1 --port 8080 \
  --api-key-env MLXS_API_KEY --api-key-strict \
  --no-plugins --no-vision --no-pld \
  --ctx-size 262144 --max-resident-models 1 \
  --prefill-chunk 4096 --prefix-cache-entries 2 \
  --prefix-cache-mem 512MB --prefix-cache-disk 50GB \
  --dspark --no-drafter --dec-bw --dspark-copy-rows 8 --dec-rows 8 \
  --fusion commitlate,routers6,engramcast,onehotc,stagefp4,stagebw \
  --verify-exact served,fused --woa-fast --router-fast=2 \
  --attn-proj-fast-mode 5 --engram-mmap --prefill-pipe \
  --dec-attn-out-fast --dec-proj-fast --dspark-warm \
  --max-concurrent 8 --exact-batch --exact-batch-max 8
```

The server warms DSpark shapes at startup (~1–2 min) and listens at `http://127.0.0.1:8080/v1`. Use the same request key in your client.

## License and credits

MIT. The original DeepSeek-V4.1-Flash-1017 model and MIT license are from DeepSeek. David Dalcu (`@ddalcu`) published the lossless MLX layout and [mlx-serve](https://github.com/ddalcu/mlx-serve). David Tai (`@TheDavidTai`, PR #1256) contributed the lossless DeepSeek-V4.1-Flash prefill optimizations used in our unified `26.11.4` engine build. Inco (`@inco_ai`) contributed open-source DFlash2 block-diffusion speculative decoding supported in the unified engine. [bot.dev](https://bot.dev) (`@botdevhq`) built this mixed-precision pack and the exact hybrid decode stack. Sponsored by bot.dev and Chatlio.
