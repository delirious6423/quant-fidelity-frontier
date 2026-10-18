---
license: mit
base_model:
  - deepseek-ai/DeepSeek-V4-Flash-0731
  - unsloth/DeepSeek-V4-Flash-0731-GGUF
pipeline_tag: text-generation
base_model_relation: quantized
tags:
  - gguf
  - llama.cpp
  - unofficial
  - moe
  - expert-pruning
  - speculative-decoding
  - apple-silicon
---

# DeepSeek-V4-Flash-0731-Pruned-155B-A13B-GGUF

**Not a DeepSeek release.** This is an unofficial, community-made expert-pruned derivative of [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), not affiliated with or endorsed by DeepSeek. This is a derivation chain (base → quantization → pruning), not a model merge.

The builds are sized to run fully GPU-resident on a 64 GB Apple Silicon machine with llama.cpp — concretely, a Mac mini M4 Pro 64 GB, the machine this build targets and was developed and measured on. Routed experts are reduced from 256 to 128 on the 40 backbone MoE layers; the first 3 MoE layers ("hash layers") keep all 256 experts and their original fixed routing tables. The result is ~155.5B total parameters with ~13B active per token, at IQ3_XXS quantization.

Capability check: on a 100-instance SWE-bench Verified subset, this build resolves **77.5%** running agentically on that single Mac mini, averaged over two passes (76/100 and 79/100). Under the identical harness, DeepSeek's official API serving the unpruned FP8 reference weights resolves 84.0% (86/100 and 82/100) — a 6.5-point gap (details, comparison table, and caveats under Quality and limitations).

Quantized bytes are unmodified slices of [unsloth's UD-IQ3_XXS shards](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF): pruning slices the expert axis byte-for-byte, with no dequantize/requantize step, so no new quantization loss is introduced. Note that the imatrix calibration behind those bytes was computed for the unpruned model; whether it remains optimal after pruning has not been evaluated separately.

## Performance

All figures measured by the author on the Mac mini M4 Pro 64 GB. This project started from running the unpruned model on the same machine: no unpruned quantization fits in 64 GB of RAM, so the weights stream from SSD via mmap and the best you can do without modification is partial GPU offload. Pruning is what makes the model fit fully on GPU — roughly 4–5× the decode and 3–4× the prefill of that best unmodified setup.

| Configuration | Decode | Prefill |
|---|---|---|
| This build + DSpark sidecar (patched llama.cpp, all-GPU) | up to 21.2 t/s short context; 14.3–16.6 t/s at 8k–32k+ agent context (band medians, 1,110 turns) | 99–126 t/s (12k context) |
| Unpruned base, best unmodified setup on the same machine (vanilla llama.cpp, partial GPU offload, mmap-streamed) | ~4 t/s | ~33 t/s |

Decode of the unpruned model barely moves with offload or quantization choice (3.3–4.1 t/s across configurations) because the weights that matter stream from disk; only making the model small enough to live entirely on GPU changes that picture.

## Files

| File | Size | What it is |
|---|---|---|
| `DeepSeek-V4-Flash-0731-Pruned-155B-A13B-UD-IQ3_XXS-*.gguf` (2 shards) | 55 GiB | Main pruned build. Needs the llama.cpp patch below |
| `DeepSeek-V4-Flash-0731-Pruned-155B-A13B-ZP256-UD-IQ3_XXS-*.gguf` (3 shards) | 97 GiB logical / ~63 GiB after re-sparsifying | Vanilla-compatible build. Runs on vanilla llama.cpp |
| `DeepSeek-V4-Flash-0731-DSpark-Draft-2.9B-IQ2_XXS.gguf` | 1.1 GiB | Pruned DSpark draft model for speculative decoding (patched llama.cpp only) |
| `llama-cpp-DeepSeek-V4-Flash-0731-Pruned-155B-A13B.patch` | 13 KB | llama.cpp patch: per-layer expert count loading |
| `punch_zero_holes.py` | — | Re-sparsify tool for the ZP256 build (macOS/APFS) |

## Which build should I use?

- **Main build (patched llama.cpp)** — best speed. The hash layers keep 256 experts while backbone layers have 128, so the loader needs a small patch (5 files, +54/−15 lines) that reads a `deepseek4.hash_layer_expert_count` metadata key. Vanilla llama.cpp rejects these files with:

  ```
  llama_model_load: error loading model: check_tensor_dims: tensor 'blk.0.ffn_gate_inp.weight'
  has wrong shape; expected 4096, 128, got 4096, 256
  ```

  If you see that error, you are on an unpatched build.

- **ZP256 build (vanilla llama.cpp, or other GGUF runtimes that run the base model)** — structurally identical to the base 256-expert model. Instead of physically removing experts, the router bias (`exp_probs_b`) of non-selected backbone experts is set to a large negative value so they are never routed to, and their weight slabs are zeroed. It should load anywhere the base GGUF loads (validated on vanilla llama.cpp; verify on other runtimes before relying on it). The catch: the file is 97 GiB logically, so fitting a 64 GB machine relies on mmap lazy loading plus keeping backbone experts on CPU (llama.cpp `--override-tensor`), which is considerably slower (~7.8 t/s decode vs ~17 t/s for the main build).

## Quickstart: main build (patched llama.cpp)

Build llama.cpp from the branch/tag [`DeepSeek-V4-Flash-0731-Pruned-155B-A13B-v1`](https://github.com/ishikawa/llama.cpp/tree/DeepSeek-V4-Flash-0731-Pruned-155B-A13B) — upstream master with the single loader commit applied — or apply the bundled `.patch` to a recent upstream checkout (validated against upstream commit `fc35562ba`; on much newer checkouts the patch may need trivial rebasing):

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git apply /path/to/llama-cpp-DeepSeek-V4-Flash-0731-Pruned-155B-A13B.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

On a 64 GB machine, **raise the GPU wired-memory limit before serving** — the macOS default (~75% of RAM) is not enough for the ~55 GiB model plus KV and compute buffers. The setting resets on reboot:

```sh
sudo sysctl iogpu.wired_limit_mb=61440
```

Even with the raised limit, the practical ceiling is ~60 GiB of total Metal buffers; configurations that exceed it (for example `-ub 2048` with the sidecar loaded) fail with a runtime Metal OOM.

Serve fully GPU-resident with the speculative-decoding sidecar (the author's long-running local server configuration on the Mac mini M4 Pro 64 GB):

```sh
./build/bin/llama-server \
  -m DeepSeek-V4-Flash-0731-Pruned-155B-A13B-UD-IQ3_XXS-00001-of-00002.gguf \
  -c 196608 --parallel 2 -ngl 99 -ub 512 \
  --fit off --no-repack \
  -md DeepSeek-V4-Flash-0731-DSpark-Draft-2.9B-IQ2_XXS.gguf \
  --spec-type draft-dspark -ngld 99 --spec-draft-n-max 1 \
  -ctkd q8_0 -ctvd q8_0 \
  --jinja
```

Pass the first shard as `-m`; the loader picks up the rest automatically.

Configuration notes, all from measurement on the machine above:

- **Do not quantize the main KV cache** (`-ctk/-ctv q8_0`): the DeepSeek-V4 attention takes a slow path and decode drops roughly in half (16.5 → 9.3 t/s). Quantizing the draft KV (`-ctkd/-ctvd q8_0`) is fine and has no measurable effect on acceptance.
- `-ub 512` is required when the sidecar is loaded on 64 GB: with `-ub 2048` the combined compute buffers exceed the practical wired-memory ceiling (~60 GiB of Metal buffers) and the server hits a runtime Metal OOM. The prefill cost of the smaller batch is 10–20%. If you drop the sidecar, `-ub 2048` is the better prefill setting.
- Avoid greedy sampling (temp 0); it measurably degrades agentic task success on this model family. Server default sampling works well.

### Serving agent harnesses (OpenAI Codex CLI)

Long unattended agentic runs (hours of SWE-bench-style rollouts) are markedly more stable with three adjustments, all found the hard way on this build:

- Disable Codex tools the server cannot represent. The patched server converts Responses-API tools to Chat-Completions form and silently drops any non-`function` tool type, logging `unsupported Responses tool type '...' skipped`. Codex declares its multi-agent tool group as a `namespace`-type tool and web search as a `web_search`-type tool; both get dropped, so the model sees a smaller tool set than Codex assumes. Set top-level `web_search = "disabled"` and `[features] multi_agent = false` in the Codex config. Top-level keys must appear before any `[table]` header in the TOML, or they are silently ignored.
- Serve one deep slot instead of two for a single-harness workload: `--parallel 1 -c 262144` (256K). Long rollouts stop hitting per-slot context truncation, and Codex compacts context far less often; set Codex's `model_context_window` to the slot size.
- After a Codex upgrade, check the server log for new `unsupported Responses tool type` warnings — new tool types show up there first.

A rare failure mode to know about: on a small fraction of first turns (roughly 0.1–0.3% in our runs before these adjustments) the model narrates a whole session instead of calling tools, emitting orphan `</think>` closers until it fills the slot. There is no output cap by default, so one runaway generation can burn the full slot (~90 minutes at 256K). It shows up in the server log as `truncated = 1` with `n_tokens` near the slot size.

## Quickstart: ZP256 build (vanilla llama.cpp)

```sh
llama-server \
  -m DeepSeek-V4-Flash-0731-Pruned-155B-A13B-ZP256-UD-IQ3_XXS-00001-of-00003.gguf \
  -c 32768 -ngl 99 --jinja \
  --override-tensor 'blk\.([3-9]|[1-3][0-9]|4[0-2])\.ffn_(gate|up|down)_exps\.weight=CPU'
```

The `--override-tensor` pattern keeps the backbone expert tensors on CPU, which is what makes a 64 GB machine workable (~7.8 t/s decode). With enough unified/system memory you can drop it and load everything on GPU. Do not combine the ZP256 build with the speculative sidecar: with experts on the CPU path, drafting is bandwidth-bound and counterproductive.

### Reclaiming disk space (ZP256 sparse slabs)

The zeroed expert slabs make ~42 GiB of the ZP256 shards pure zeros. On macOS/APFS you can turn those regions back into filesystem holes after download, shrinking physical usage from 97 GiB to ~63 GiB without changing file contents:

```sh
python3 punch_zero_holes.py DeepSeek-V4-Flash-0731-Pruned-155B-A13B-ZP256-UD-IQ3_XXS-*.gguf
```

Note that ordinary copies (`rsync`, `scp`, cross-volume `cp`) re-materialize the holes; within one APFS volume use `cp -c`.

## Speculative decoding sidecar

`DeepSeek-V4-Flash-0731-DSpark-Draft-2.9B-IQ2_XXS.gguf` is a pruned build of the official [DSpark draft model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark): 32 experts, IQ2_XXS expert quantization, 1.1 GiB. Because the DSpark draft conditions on the target model's hidden state, its top-1 agreement tolerated expert pruning well in the author's measurements: this K32 build's acceptance rate was within run-to-run noise of the unpruned draft on the workloads tested (not a benchmark-grade claim).

Measured effect on the machine and server configuration above (`--spec-draft-n-max 1`):

- +14–17% decode over no speculation; short-context single-slot up to 21.2 t/s
- On the author's coding-agent workload (SWE-bench-like turn structure, 1,110 turns): band medians 16.6 t/s at 8–16k, 15.8 t/s at 16–32k, and 14.3 t/s at 32k+ context depth, with acceptance 0.71–0.76
- `--spec-draft-n-max 1` is the fastest setting on this machine, not a conservative fallback: per-position acceptance decays (~0.75 → ~0.55 → ~0.39) while Metal's small-batch verify cost grows almost linearly with batch size, so only the first draft position pays for itself. Measured across context lengths, n_max 2 gains less than n_max 1 and n_max 3 is a net loss. On backends where small-batch verify is closer to flat cost (e.g. CUDA), larger n_max may come out ahead — re-measure before porting this setting. On M5-generation Apple Silicon, reports so far show the Metal 4 tensor units engaged only for large-batch matmul, with measured gains limited to prefill, so the small-batch verify path — and with it this ordering — is not expected to change there for now

The sidecar requires the patched build and the main (non-ZP256) model, fully GPU-resident.

## How it was made

1. **Expert selection.** Per-layer expert utilization statistics were collected from the router (both prefill and generation) over a curated corpus mixture, and the top 128 experts per backbone layer were kept. The mixture combines the author's own real coding-agent sessions with self-built domain corpora: synthetic dev-QA, Japanese text (0.27 of the mix), math (0.10), and long-form reasoning (0.20). On top of the utilization ranking, a per-token-class coverage pass (byte / CJK / code / digit / latin / whitespace classes) swaps in experts that dominate rarely-routed token classes so they are not lost to global frequency ranking. Selection is per-layer; the layers do not share a common expert subset. The real-traffic component is the author's own coding sessions; nothing beyond aggregated router statistics (per-layer expert utilization) feeds the selection, and no text content flows into the released artifacts.
2. **Hash-layer preservation.** The first 3 MoE layers of DeepSeek-V4-Flash route with fixed hash tables rather than learned gating. Pruning these layers causes low-temperature reasoning to spin in repetitive loops; keeping all 256 experts and the original tables on just these 3 layers eliminates the failure completely in a 112-cell probe grid (0 occurrences, vs 7/35 at temp 0.3 for a variant with pruned hash layers). This per-layer expert count split (`n_expert` 128 + `hash_layer_expert_count` 256) is exactly what the loader patch implements.
3. **Byte-slice pruning.** Kept expert slabs are copied byte-for-byte from unsloth's UD-IQ3_XXS shards; router tensors are row-sliced to match. No requantization anywhere.
4. **ZP256 variant.** Base-shaped build with non-selected backbone experts suppressed via large negative router bias and their slabs zeroed (stored sparse).

The GGUF `general.*` metadata (model name, source URLs) intentionally retains the upstream quantization's values, so inspection tools will report the original model identity; the file names and this card are the authoritative description of what the files contain.

## Quality and limitations

### SWE-bench Verified, 100-instance subset: 77.5%

The subset is 100 instances stratified by repository: an original 50, plus 50 drawn from the complement so that the original 50 stay fully contained. Both passes cover all 100: 76/100 = 76.0%, Wilson 95% CI [66.8, 83.3] on the first, and 79/100 = 79.0%, CI [70.0, 85.8] on the second. The pass@1 mean is 77.5%.

For scale, the same 100 instances recomputed from public per-instance leaderboard data, which is single-pass and so compares against this build's first pass (`per_instance_details.json` under `evaluation/bash-only` in [SWE-bench/experiments](https://github.com/SWE-bench/experiments), mini-SWE-agent submissions):

| Model | Same 100 instances | Full 500 |
|---|---|---|
| GLM 5 (high) | 78/100 = 78% | 72.8% |
| MiniMax M2.5 (high) | 75/100 = 75% | 75.8% |
| **This build, on one Mac mini M4 Pro 64 GB** | **76/100 = 76%** | — |
| DeepSeek V3.2 (high) | 72/100 = 72% | 70.0% |
| Kimi K2.5 (high) | 70/100 = 70% | 70.8% |
| GLM 4.6 (T=1) | 58/100 = 58% | 55.4% |
| Qwen3-Coder 480B/A35B | 55/100 = 55% | 55.4% |

Measurement conditions:

| Item | Value |
|---|---|
| Sample | 100 of SWE-bench Verified, repository-stratified random, fixed seed |
| Agent | Codex CLI, k=1, 120-minute cap per instance |
| Passes | 2 over all 100 |
| Model | this IQ3_XXS build with the DSpark speculation sidecar, 256k-token context |
| Network | external access blocked (asserted at run start) |

Caveats: the leaderboard rows use a different agent harness (mini-SWE-agent vs Codex CLI here), so read that table as a sanity anchor, not a strict head-to-head ranking. For the degradation relative to the reference weights under this exact harness, see below.

#### Delta vs. the FP8 reference weights (same harness)

The unpruned reference was measured under the identical harness: same 100 instances, same Codex CLI configuration and per-instance time cap, same network isolation (the agent container reaches only the model API through a relay; external access asserted blocked at run start and observed blocked in-run). The reference is DeepSeek's official API serving the release weights, which are FP8 e4m3 — no BF16 release exists, so FP8 is the cleanest available baseline. Input volume and cache-hit rates matched closely between the pass-1 and API runs (67–70M input tokens, ~98% cache hit), supporting that both sides processed equivalent context.

| Model | Pass 1 | Pass 2 | pass@1 mean |
|---|---|---|---|
| DeepSeek official API (FP8 reference, unpruned) | 86/100 = 86.0% | 82/100 = 82.0% | 84.0% |
| This build (pruned + IQ3_XXS, one Mac mini) | 76/100 = 76.0% | 79/100 = 79.0% | 77.5% |

The mean is 6.5 points below the reference; the per-pass gaps are 10 points on the first and 3 on the second.

Every cell is a single k=1 draw; compare cells only k=1 against k=1. In particular, do not read the union of two passes (pass@2) against a single run — that would be best-of-2 versus best-of-1.

Per pass, paired:

| Pass | Gap | This build only | Reference only | Exact two-sided p |
|---|---:|---:|---:|---:|
| 1 | 10 points | 0 | 10 | 0.002 |
| 2 | 3 points | 4 | 7 | 0.549 |

Read the delta as the combined cost of expert pruning, IQ3_XXS quantization, and serving-stack differences (llama.cpp vs DeepSeek's API implementation), not the pruning delta alone. Known residual asymmetries, all minor: tool-call parsing is llama.cpp's on the local side and DeepSeek's own on the API side; the API executes parallel tool calls unconditionally while the local build declares them unsupported; sampling runs at each side's upstream defaults.

### Other measurements

Claims are limited to what was measured, on coding-agent workloads:

- Collapse-type failures (degenerate repetition) occur on the same input set as the unpruned base at IQ3_XXS — pruning does not add new collapse cases in the probes used.
- Coding-agent task success is at the same level as the unpruned base in paired trials (small n; not a benchmark-grade claim). The higher-powered SWE-bench comparison above is the better estimate for agentic coding: 6.5 points below the FP8 reference averaged over two 100-instance passes, with per-pass gaps of 10 and 3 points.
- Low-temperature reasoning spin: 0 occurrences across the full 112-cell probe grid (see above).
- Knowledge probes show the expected mild long-tail degradation of expert pruning; rarely-routed domains lose the most. In agent setups this may be partly mitigated by wiring in retrieval — e.g. adding a web-search MCP server — so the model can look up what it no longer memorizes. If your workload is far from coding/agentic use, evaluate before adopting.
- Expect the pruned long tail to surface as confident hallucination in niche knowledge domains, plausibly more often than the base model. Do not use this build where wrong answers are costly (safety-critical, medical, legal, financial decisions) without your own evaluation.

Broader evaluations are limited by the single consumer machine this project runs on — every suite that needs an unpruned-base rerun doubles the GPU time, so a full benchmark-suite comparison is not realistic. Apart from the SWE-bench subset above, direct Japanese-language comparison against base and further standardized benchmarks have not been completed. Treat this as a workload-specific build, not a general-purpose replacement for the base model.

## Attribution and licensing

- Model weights: MIT, Copyright (c) 2023 DeepSeek ([deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731))
- Quantized bytes: sliced from [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) (MIT)
- Loader patch: contains code from [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT, Copyright (c) The ggml authors
- Included scripts and the pruning tooling: MIT, Copyright (c) 2026 Takanori Ishikawa

See the bundled `LICENSE` file for the full notices.

## References and prior work

Work that directly shaped this build:

- Roller et al., [Hash Layers For Large Sparse Models](https://arxiv.org/abs/2106.04426) — the fixed-hash-table routing used by the first MoE layers of this architecture. Its rigidity (no learned router, no renormalizing compensation when experts are removed) is why the hash layers are kept at the full 256 experts here.
- [REAP the Experts](https://arxiv.org/abs/2510.13999) — reports that for generative tasks, pruning experts outright beats merging them, and defines saliency as gate value × activation norm. This informed choosing deletion over merging; the final expert selection here uses per-layer router utilization measured on real coding-agent traffic rather than REAP's saliency score.
- [Seeing but Not Thinking](https://arxiv.org/abs/2604.08541) — observes that when routing fails to activate reasoning experts, reasoning degrades in isolation while other capabilities survive. Consistent with the reasoning-spin failure this build eliminates by preserving the hash layers.

Surveyed MoE-pruning alternatives (not adopted; names searchable on arXiv, unlinked here to keep this repo off those papers' citation listings):

- "Super Experts" — protecting activation-outlier experts
- "SHAPE" — Shapley cooperation scores for expert pruning ([implementation](https://github.com/Alizen-1009/Shapley-Moe))
- "C-Prune" — cluster-driven, per-layer redundancy
- "Sub-MoE" — subspace expert merging
- "Preserving Long-Tailed Expert Information" — the long-tail degradation trade-off discussed under Quality

## Reproducing

The pruning/selection pipeline consumes the base GGUF shards plus recorded expert-selection IDs, and is deterministic byte-slicing from there. The loader patch is one commit on top of upstream llama.cpp; the branch and tag above pin the exact revision it was validated against.
