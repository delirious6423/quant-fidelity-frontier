---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: text-generation
library_name: anemll-forge
tags:
  - coreai
  - ane
  - quantized
  - speculative-decoding
  - dflash2
  - kv-cache
  - long-context
---

# ANEMLL Forge · Qwen3.8-27B for ANE · release 0.2.1, 192K context

**Research project for inference of large dense models on the M6 Apple Neural Engine.** Release 0.2.1 runs the **release 0.2 target weights** of [anemll/anemll-forge-qwen3.8-27B](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B) with a context of up to **196,608 tokens (192K)**, three times the 64K of the release 0.2 packages. Four memory optimizations make it fit on a 32 GB Mac (below), and three runtime fixes free memory the previous runtime kept. The target weights, their quantization and their measured quality are those of release 0.2; the drafters (re-quantized, without their own output head) and the INT8 key cache are new. Speculative decoding verifies every token against the target, so the drafter affects speed, not the output.

The normal inference path uses the included Core AI DFlash2 speculative drafter: each T=8 verifier cycle checks one anchor and seven draft proposals. Do not substitute a Core ML or unpaired drafter.

ANEMLL independently converts and quantizes [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Credit for the target belongs to the **Qwen Team and Alibaba Cloud**, with upstream **Copyright 2026 Alibaba Cloud** and Apache-2.0 licensing. The paired drafters derive from [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2), whose upstream notice preserves the z-lab DFlash2 and Prism Bonsai provenance. No endorsement by these projects or Apple is claimed.

**M6 only.** These packages use FP8 attention, which the M5 ANE compiler rejects. Up to 64K, the [release 0.2 packages](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B) are faster (about 15% more prefill at short context) and also run on the M5 family.

## How to use

You need an M6 Mac with 32 GB of memory, macOS 27, and the [ANEMLL Forge](https://github.com/Anemll/anemll-forge) inference environment from its README (source from 19 October 2026 or later). Run the commands from the Forge checkout with its venv active, and start the server with the same Python that compiled: Core AI keeps a compile cache per Python executable, so another interpreter compiles everything again (with a venv outside the checkout, set `PY=` to its `python` for the server wrapper).

**1. Download** (about 15 GB with one drafter; leave out `--exclude` to get both):

```sh
export FORGE_192K="$HOME/Models/anemll-forge-qwen3.8-27B-192K"
hf download anemll/anemll-forge-qwen3.8-27B-192K \
  --local-dir "$FORGE_192K" --exclude "drafter/gu2a3/*"
```

**2. Compile for the context you need.** `--ctx` compiles only the context entries up to that cap (see [why](#why-compile-per-context)). Start with a quick 16K test:

```sh
python forge.py compile --ctx 16K --build "$FORGE_192K/coreai/m6-192k" \
  --draft "$FORGE_192K/drafter/all3/dflash2_lut4_gptq_nohead.aimodel"
```

**3. Start the server with the same `CTX`** (stop a running server first with `bash scripts/qwen38_server.sh stop`):

```sh
MODEL="$FORGE_192K/model" BUILD="$FORGE_192K/coreai/m6-192k" \
DRAFT="$FORGE_192K/drafter/all3/dflash2_lut4_gptq_nohead.aimodel" \
DRAFTER="$FORGE_192K/drafter/all3" \
CTX=16K bash scripts/qwen38_server.sh start
```

The startup log says `[ctx] using the build capped at 16K`. The server listens on `http://127.0.0.1:8765/v1` (OpenAI-compatible), updates Pi's model entry for the cap, and answers `curl -s http://127.0.0.1:8765/health`. For the smaller drafter, point `DRAFT` and `DRAFTER` at `drafter/gu2a3`.

### Quick test: 16K

Steps 2 and 3 with `--ctx 16K` and `CTX=16K` are the quick test: about 10 minutes from download to a running server, instead of 2 h 11 min for every entry up to 192K. Then check it:

1. **Startup:** the wrapper prints `[ctx] using the build capped at 16K` and ends with `serving OpenAI API ... (speculative ON, ctx 16384)`. On the compile, `[ANE compile]` lines show each chunk at about 30 seconds.
2. **Health:** `curl -s http://127.0.0.1:8765/health` reports the model, the context and the loaded build.
3. **A reply:**

   ```sh
   curl -s http://127.0.0.1:8765/v1/chat/completions -H "Content-Type: application/json" \
     -d '{"model": "qwen38-27b-ane", "max_tokens": 200,
          "messages": [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome."}]}'
   ```

4. **Speed:** `bash scripts/qwen38_server.sh log` shows a `prefill:` line with the prompt's rate and `decode:` lines with tokens per second and tokens per verify call. Longer prompts give steadier prefill numbers, for example from a Pi session.

Reference, measured on a 32 GB M6 (macOS 27.0.1): compile **8 min 37 s** (about 32 s per chunk); prefill **267 tok/s** for a 5,926-token prompt and 280 tok/s for 1,619 tokens; server process 2.0 GB. Decode depends on the content: at 16K the full build measured 32 tok/s for code, 17 for prose and 58 for JSON (see Longer context below).

#### Runtime memory fixes

- **Old caches are released when the context grows.** A loaded Core AI function keeps the buffers of its last call, so the cache of every entry the runtime had left stayed in memory: about 4.9 GB of stale caches at 64K with the release 0.2 packages. The runtime now releases them after each move (about 13 ms).
- **The drafter keeps its selector tables as stored**, converting only the rows a draft reads: 0.51 GB less RAM, identical drafts.
- **Models on external drives no longer crash the server.** The embedding and selector are read safely instead of memory-mapped from other volumes.

These are runtime changes: they apply to the release 0.2 packages too, with current source.

## Longer contexts

Compile and start with the cap you want, in the same two commands: `--ctx 64K` with `CTX=64K`, `--ctx 128K` with `CTX=128K`, and so on. Each cap is its own build with its own one-time compile and one more copy of the chunks on disk (about 10 GB, in `~/.anemll-forge/builds/`). A server uses the smallest capped build that holds its `CTX`; without one, it uses the full download, which holds every entry up to 192K and compiles all of them on its first start (**2 h 11 min** on M6). `--ctx 192K`, or no `--ctx`, compiles the full download.

| CTX | Compile on M6 | Pi compacts above | Top entry: prefill / decode, mean of five tasks | Process memory at the top |
| --- | --- | ---: | --- | ---: |
| 16K | 8 min 37 s (measured) | about 8K | 237 / 34.0 tok/s | 2.0 GB |
| 64K | not yet measured | about 45K | 160 / 31.6 tok/s | 3.8 GB |
| 96K | 45 min 43 s (measured) | about 78K | 123 / 27.4 tok/s | 4.9 GB |
| 128K | not yet measured | about 111K | 95 / 25.1 tok/s | 6.0 GB |
| 192K | 2 h 11 min (full build) | about 176K | 62 / 20.6 tok/s | 8.2 GB |

Decode, and every rate above 16K, was measured on the full build; a capped build runs the same entries at the same speed. On a 32 GB Mac, 128K is the comfortable maximum: a prompt just past 128K runs in the slower 192K entry, and the move into it added 1.3 GB of swap.

Every macOS update, minor ones included, compiles again. Ctrl-C stops a compile safely: finished packages stay cached, and running it again (or starting the server) compiles the rest. To free a capped build's disk, delete its folder under `~/.anemll-forge/builds/`.

### Why compile per context

- **Core AI compiles a package as a whole.** Each target chunk is one package holding all 14 of its functions: a verify entry and a prefill entry for each of the seven contexts. Loading it compiles every function, whatever context the server will use.
- **Most of that time is the long entries.** A chunk of the full build takes about 8 minutes, most of it in the prefill entries' history tiles above 64K; a chunk with only the 8K and 16K entries takes 32 seconds.
- **The time is spent on your Mac.** About 94% of it is the system's Neural Engine compiler, whose output stays in a system cache, so neither a download nor precompiled `.aimodelc` files can skip it.
- **How `--ctx` works.** It writes a copy of each chunk that keeps only the functions of the contexts up to the cap, with the same weights (`coreai/strip_functions.py`, about a second per chunk), and a manifest to match (`scripts/ctx_variant.py`). The download is not modified. Deriving is deterministic: deriving again from the same download gave byte-identical packages (the same `main.hash`, which keys the compile cache).
- **It also lowers memory, not speed.** The Neural Engine reserves each chunk's scratch for its largest function, so a capped build reserves less. Its entries run at the full build's speed: target-only prefill calls measured 255 to 257 tok/s at 8K and 237 at 16K for the full, 96K and 16K builds alike.

## Memory optimizations

### 1. INT8 keys and values (`kv8`)

The 16 full-attention layers store every history position's keys and values as INT8, with FP16 scales per token and KV head. Release 0.2 kept the keys in FP16 (V8). The ANE multiplies each history tile's attention scores by the key scales, so it computes the same scores as from FP16 keys rounded to INT8.

| Cache format | Per history position | At 64K (65,472 rows) | At 192K (196,608 rows) |
| --- | ---: | ---: | ---: |
| FP16 keys and values | 64 KiB | 4.0 GiB | 12.0 GiB |
| V8: FP16 keys, INT8 values (release 0.2) | 48.1 KiB | 3.0 GiB | 9.0 GiB |
| **kv8: INT8 keys and values (release 0.2.1)** | **32.25 KiB** | **2.0 GiB** | **6.0 GiB** |

These are logical cache sizes; measured at 192K, the cache with its I/O buffers takes 6.8 GB of process memory. On the previous weights, `kv8` against V8 measured a direct KL of 6.6e-5 nats with the same top token at 99.75% of positions (KL-512, 64 sequences), at the same speed within 2%.

### 2. Drafters that use the target's output head

The drafter no longer carries its own copy of the 248,320-row output head. It scales its hidden state by the target head's norm gain (`head_normw_fp16.npy`) and runs it through the target's head, which is already loaded.

| Drafter | Weights | Neural Engine memory | Decode speed | Tokens per verify call |
| --- | --- | ---: | ---: | ---: |
| release 0.2 drafter (not in this repository) | four-bit LUT, own head copy | 1.60 GB | 1.000 | 1.000 |
| **`all3` (default)** | every projection three-bit vector LUT | **0.74 GB** | 1.010 | 1.002 |
| `gu2a3` | gate and up two-bit, the rest three-bit | 0.63 GB | 1.008 | 0.989 (prose 6% lower) |

Speed and tokens per verify call relative to the release 0.2 drafter, from paired runs on these target packages at contexts from 1K to 195K tokens and five content types. `all3` saves **0.86 GB** at the same speed. Verification keeps every output a greedy decode of the target with any drafter; greedy outputs can still differ slightly between drafters, because the target's FP16 numerics depend on how accepted runs split the verify blocks.

### 3. Prefill rows per context entry

A chunk's scratch memory on the Neural Engine is about rows times its largest context (the attention scores), and Core AI reserves it for the chunk's largest function. 64-row prefill in every entry would have doubled the scratch at 192K (**+2.8 GB**). These packages prefill 64 rows per call up to 96K and 32 rows at 128K and 192K, so the 64-row entries fit in the scratch the 192K entry needs anyway: **+0.75 GB** for 11 to 17% faster prefill below 128K.

### 4. One cache at a time, grown on demand

The runtime allocates only the current context entry's cache. It starts a conversation in the 8K entry, moves to the next entry when the context outgrows it (moving the kept rows), and releases the old cache; a new conversation goes back to the smallest entry that holds it. Memory follows the conversation, not the cap:

### Memory by context

On a 32 GB M6, for the same contexts as the speed table:

| Context (prompt tokens) | Entry | KV cache (kv8) | KV cache and I/O | Process memory | Neural Engine | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1K (1,000) | 8K | 0.27 GB | 0.58 GB | 1.5 to 1.9 GB | 16.5 GB | 18.0 to 18.4 GB |
| 8K (7,000) | 8K | 0.27 GB | 0.58 GB | 1.5 to 1.9 GB | 16.5 GB | 18.0 to 18.4 GB |
| 16K (15,000) | 16K | 0.54 GB | 0.86 GB | 2.2 GB | 16.5 GB | 18.7 GB |
| 32K (31,500) | 32K | 1.08 GB | 1.40 GB | 2.8 GB | 16.5 GB | 19.3 GB |
| 64K (64,000) | 64K | 2.16 GB | 2.48 GB | 3.8 GB | 16.5 GB | 20.3 GB |
| 96K (97,000) | 96K | 3.25 GB | 3.56 GB | 4.9 GB | 16.5 GB | 21.4 GB |
| 128K (129,500) | 128K | 4.33 GB | 4.64 GB | 6.0 GB | 16.5 GB | 22.5 GB |
| 160K (160,000) | 192K | 6.49 GB | 6.81 GB | 8.2 GB | 16.5 GB | 24.7 GB |
| 192K (195,000) | 192K | 6.49 GB | 6.81 GB | 8.2 GB | 16.5 GB | 24.7 GB |

- **Entry:** the context entry that holds a prompt of that size. Memory follows the entry, not the prompt: every prompt that fits the 8K entry costs the same.
- **KV cache (kv8):** the entry's logical cache, 32.25 KiB per position (computed). **KV cache and I/O** adds the entry's input and output buffers (measured).
- **Process memory:** the server process's footprint, measured with the full build, without the drafter (it adds a few hundred MB of host memory).
- **Neural Engine:** the full build's target packages (about 15.8 GB) plus the `all3` drafter (0.74 GB), the same for every context. A build compiled with `--ctx` reserves less scratch; its Neural Engine memory has not been measured yet (the 16K build's server process measured 2.0 GB).
- **Total** is process plus Neural Engine. The rest of a 32 GB Mac is left for macOS and other apps; the move from the 128K to the 192K entry briefly peaks 4.3 GB higher (both caches exist).

While it moves from one entry to the next, both caches exist briefly: the move from 128K to 192K peaks at 12.5 GB of process memory.

**Neural Engine total:** about 15.8 GB for the full target packages (weights and scratch), against about 13.0 GB for release 0.2's 64K packages, because the long entries need more scratch. With the `all3` drafter the total is about 16.5 GB, about 2 GB more than release 0.2 for three times the context. A build compiled with `--ctx` reserves only what its largest entry needs.

## Longer context

- **Context entries:** 8K, 16K, 32K, 64K, 96K, 128K and 192K (release 0.2: 8K to 64K, with 48K).
- **Tiled history attention above 64K.** The ANE limits one tensor axis to 65,536, so the 64K entry holds 65,472 history rows. Entries above 64K attend over the history in tiles without forming that tensor, and hold their full length.
- **Recall:** six codes planted at depths 0 to 185K were recalled correctly at every context size up to 195K tokens.

Speed on M6 (macOS 27.0.1, 32 GB, `all3` drafter). Prefill in tokens per second for the context entry; decode in generated tokens per second by content type:

| Context (prompt tokens) | Prefill rows | Prefill | Code | Prose | Explain | JSON | Thinking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K (1,000) | 64 | 260 | 28.0 | 17.5 | 29.2 | 62.1 | 40.4 |
| 8K (7,000) | 64 | 260 | 26.7 | 16.6 | 28.1 | 56.8 | 38.8 |
| 16K (15,000) | 64 | 235 | 32.2 | 17.2 | 23.1 | 58.4 | 39.0 |
| 32K (31,500) | 64 | 205 | 24.6 | 14.2 | 22.5 | 58.0 | 40.8 |
| 64K (64,000) | 64 | 160 | 28.8 | 15.6 | 23.7 | 53.1 | 36.7 |
| 96K (97,000) | 64 | 123 | 26.0 | 13.7 | 19.4 | 45.8 | 32.0 |
| 128K (129,500) | 32 | 95 | 23.9 | 11.9 | 18.7 | 41.6 | 29.3 |
| 160K (160,000) | 32 | 62 | 19.5 | 10.1 | 13.8 | 34.5 | 22.4 |
| 192K (195,000) | 32 | 62 | 18.4 | 10.0 | 14.7 | 35.5 | 24.8 |

- **Prefill** is the target's rate in the context entry that holds a prompt of that size (160K prompts run in the 192K entry). A whole cold prompt also feeds the drafter and saves prompt snapshots: a 6,116-token prompt prefilled at 241 tok/s. Time to first token for a cold prompt: about 5 min 42 s at 64K and 33 min at 195K.
- **Decode:** 256 generated tokens, greedy for code, prose, explanation and JSON, thinking sampled at temperature 1.0. How many drafted tokens the target accepts depends on the content (tokens per verify call: JSON 7.8, thinking 5.3, code 3.9, explanation 3.4, prose 2.2), not on the context; the context sets the cost of a verify call: 104 to 114 ms up to 16K, 129 ms at 64K, 163 ms at 128K and about 202 ms in the 192K entry.

## Files

- [coreai/m6-192k/](coreai/m6-192k): 16 target chunks (M6 functions), the output head and the manifest.
- [drafter/all3/](drafter/all3) and [drafter/gu2a3/](drafter/gu2a3): drafter packages with configuration, selector tables, the target head's norm gain, licenses and provenance.
- [model/](model): tokenizer, configuration and FP16 host embedding table, byte-identical to release 0.2.
- [config.json](config.json): exact copy of `model/config.json` for Hub discovery.

Disk: about 15 GB to download with one drafter, plus about 10.5 GB for Core AI's compile cache and about 12 GB for the system's compiled programs.

## Limitations

- Prefill at short context is about 15% below the release 0.2 packages; the cause is being investigated.
- The 192K entry prefills at 62 tok/s: a 195K-token prompt takes about 33 minutes to its first token.
- Long-context quality was checked by recall only. KL and perplexity were measured on the release 0.2 packages, up to 64K.
- One synthetic set of prompts per content type is not a general benchmark.

## Attribution and licenses

- Target: [`Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0). Original Apache-2.0 [LICENSE](LICENSE) and ANEMLL-added [NOTICE](NOTICE) are included; [QWEN_SOURCE.json](QWEN_SOURCE.json) records source checks and remaining provenance limitations.
- Drafter: [`ProCreations/Ternary-Bonsai-2-27B-DFlash2` at `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b). Its original `LICENSE` and `NOTICE` are included unchanged in each drafter folder.
- The drafter notice records donor [`z-lab/Qwen3.8-27B-DFlash2` at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) and training target [`prism-ml/Ternary-Bonsai-2-27B-gguf` at `6ed5e12bf84b7a63069882c91dd9e9218647d17b`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b).

[MODIFICATIONS.md](MODIFICATIONS.md) describes the converted derivatives and the graph changes.

**Model-derived assets follow their upstream Apache-2.0 licenses and notices. ANEMLL Forge source code and documentation are MIT-licensed.** These packages are converted from Apache-2.0 models (Qwen3.8-27B and the ProCreations drafter), so they keep that license; LICENSE, NOTICE and MODIFICATIONS.md carry the license, the upstream notices and what ANEMLL changed. Preserve the applicable model licenses, copyright and modification notices when redistributing derivatives.
