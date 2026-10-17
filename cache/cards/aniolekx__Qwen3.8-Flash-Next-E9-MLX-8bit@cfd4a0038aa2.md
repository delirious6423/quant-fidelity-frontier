---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - mlx
  - qwen
  - qwen3.8
  - quantized
  - 8-bit
  - apple-silicon
  - mtp
---

# Qwen3.8-Flash-Next E9: MLX 8-bit with the MTP head

> [!IMPORTANT]
> **E9 was built for ENGINE, and served and tested end to end with ENGINE only**, on a Mac Studio (M3 Ultra,
> 512 GB unified memory). As a parity reference it was also loaded non-strictly (without the MTP head) by the
> `qwen4_exp` code of an `mlx-lm` pull-request branch (#1788), which scored logits and short greedy continuations
> on four fixtures [5]. That is not support in a released runtime: we have not tested whether a released `mlx-lm`,
> `mlx-vlm` or upstream `mlx-swift-lm` can load `model_type: qwen4_exp` from this checkpoint, so no such support
> is claimed. The Hub's automatic "Use this model" snippet for MLX is generic and has not been tried with this
> checkpoint. See [Other runtimes](#other-runtimes).

E9 is an 8-bit MLX quantisation of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
It was built for [ENGINE](https://github.com/KrakowiakK/ENGINE), a Swift/MLX inference server for Apple
silicon. It keeps two parts of the original that some conversions drop:

- the official single-layer multi-token-prediction (MTP) head, which ENGINE uses as a draft model for
  speculative decoding;
- the vision tower.

"E9" is the recipe's name inside the ENGINE project.

This is a community quantisation, not an official Qwen release. Qwen designed and trained the model, and all
credit for it goes to them. For the architecture, evaluations, intended use, limitations and safety guidance,
read the [original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

## At a glance

| | |
|---|---|
| Base model | `Qwen/Qwen3.8-Flash-Next`, official bf16 weights |
| Architecture | `qwen4_exp` (`Qwen4ExpForConditionalGeneration`). 48 decoder layers: 36 Gated DeltaNet (linear attention) and 12 full-attention layers with a block indexer. The attention is sparse above 2,048 tokens. The MoE has 512 routed experts, 10 active per token, plus a shared expert. The model also has 4-stream hyper-connections, a hashed n-gram (PLE) embedding, a 27-block vision tower and a 1-layer MTP head. |
| Parameters (counted from the tensor shapes) | 180.0 B in total: 125.7 B text trunk (without the n-gram table), 51.2 B n-gram table, 2.6 B MTP head and 0.45 B vision tower |
| Format | MLX safetensors, affine quantisation, 22 shards plus `model.safetensors.index.json` (3,309 tensors) |
| Size on disk | 194,969,549,313 bytes (195.0 GB, 181.6 GiB) for the weights and support files, not counting this README. The tensor data is 194,945,693,688 bytes. |
| Context | 262,144 tokens (`max_position_embeddings`) |
| Tokenizer, chat template, generation config | Qwen's official files, unchanged |
| Audio | none. The vocabulary contains audio tokens, but neither this checkpoint nor the upstream one has an audio tower [10]. |

## Quantisation

We checked this table against `config.json` (the `quantization` block) and against every safetensors header:

| Family | Precision | Source |
|---|---|---|
| Routed experts, 48 trunk layers (`switch_mlp` gate/up/down) | 8-bit, group 64 | quantised from official bf16 |
| Shared experts; attention `q/k/v/o` and indexer `index_qk_proj`; Gated DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj`; hyper-connection `input_mix_weight_{up,down}`; PLE `key_proj`/`value_proj` | 8-bit, group 64 | quantised from official bf16 |
| `lm_head` | 8-bit, group 64 | quantised from official bf16 |
| `embed_tokens` | 8-bit, group 64 | quantised from official bf16 |
| PLE n-gram embedding table (128 shards, 51.2 B parameters) | 8-bit, group 32 | quantised from official bf16. Group 64 is impossible here because each row is 160 wide. |
| MTP head (`mtp.*`: one full-attention layer with its own indexer and 512-expert MoE, `fc_embedding`/`fc_hidden`, hyper-connections) | 8-bit, group 64 | quantised from the official bf16 `mtp.*` tensors |
| MoE routers (`mlp.gate`) | bf16 | not quantised: the unquantised bf16 copy kept in the Vontra conversion credited below; not checked byte for byte against the official files |
| Shared-expert gates, Gated DeltaNet `in_proj_a`/`in_proj_b`, `A_log`, `dt_bias`, conv1d, hyper-connection inject weights, all norms | bf16 | official, not quantised |
| Vision tower (27 blocks plus merger, 333 tensors) | bf16 | not quantised |

The MTP head shares `embed_tokens` and `lm_head` with the trunk, so it has no separate embedding or output head.

**Why nothing is below 8-bit and the gates stay in bf16.** During bring-up we ran an older third-party checkpoint
that was quantised uniformly at 4-bit group 32, including the small gating projections. It produced non-language
output in three independent runtimes. Fixing the n-gram seed alone, which we tried first in ENGINE, did not help:
the output was still non-language. Applying two changes together then restored coherent text in the `mlx-lm`
PR #1788 runtime: replacing 424 small modules (gates, hyper-connection mixes, PLE and indexer projections, norms)
with the official bf16 tensors, and fixing the n-gram seed. The module replacement was not tried without the seed
fix, so its effect on its own was not measured [11].

**Why 8-bit and not 4-bit experts.** We built a variant of E9 whose 48 trunk expert banks were replaced by the 4-bit
group-32 expert banks of
[Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP) (commit
`327c8a6`; everything else byte-identical) and scored both with teacher-forced NLL only (`engine nll`, the same
ENGINE build B44 for both; no generation test was run). The 4-bit variant would have saved 52.9 GB of peak memory.
It was not adopted, because it scored measurably worse on prose (+0.0047 nats/token, 95% CI +0.0035 to +0.0059).
On one code context (about 40 short Python scripts concatenated) both variants drifted into predicting a spurious
end-of-turn token as the scripts accumulated (E9 at up to 4.8% of a file's positions, the 4-bit variant at up to
21.7%, about 5× as often), and that context carried most of the 4-bit variant's extra loss on private code. The
attribution is DERIVED from per-file sums rather than measured directly, and whether the effect would cut generated
code short is unknown [3].

## Quality evidence

Every number below comes from a measurement recorded as MEASURED in the project's internal research log, which is
not published. The IDs in [References](#references) identify those entries, and each result is scoped as stated there.
None of them has been reproduced outside ENGINE.

| What was compared | Result | Ref |
|---|---|---|
| E9's predecessor (the same 8-bit group-64 recipe, but `embed_tokens` and the n-gram table at 4-bit group 32) vs a near-bf16 reference build | NLL 1.79914 vs 1.80030 nats/token over 131,071 positions. The paired difference is 0.00116 ± 0.00147, which is not significant. | [1] |
| E9 vs that predecessor (the two upgrades are `embed_tokens` and the n-gram table raised to 8-bit) | Technical text: 0.110 nats/token better (t = 16.5), perplexity 6.777 → 6.072, top-1 63.18% → 64.57%. Prose: no measurable change (−0.00096, t = −0.66). | [2] |
| E9 vs the same checkpoint with 4-bit group-32 routed experts | The 4-bit variant is worse by 0.00470 nats/token on prose (95% CI 0.00345 to 0.00591), by 0.0003 to 0.0047 per clean text window, and by 0.00119 (95% CI 0.00068 to 0.00171) on public code. | [3] |
| ENGINE on E9 vs Qwen's official `transformers` modules on the same weights, dequantised exactly, in float32 | In the 39 layers with a measured bf16 floor, every stage is within 0.12 percentage points of the official module's own bf16 noise of relative error (largest gap: attention output, up to 2.76% vs a bf16 floor of up to 2.64%; the floor was measured on 39 of the 48 layers). Argmax agreement on the final logits is 99.57% of 16,125 positions, and NLL is 0.17589 (official) vs 0.17602 (ENGINE). | [4] |
| ENGINE on E9 vs an independent `mlx-lm` implementation of `qwen4_exp` | Top-1 agreement is 5/5, 6/6, 316/345 and 4,040/4,096 on four fixtures, and greedy 8-token continuations are identical on 3 of 4. | [5] |

The two parity rows ([4], [5]) check that ENGINE runs this checkpoint as the model's definition says. They do not
measure how much the quantisation itself costs; the first three rows do. Both parity rows ran ENGINE's serial
(one request, no batching) `nll` / `logits` path on earlier ENGINE builds; they were not re-run on the current build
or on the batched serving path that the speed figures below use.

### Long reasoning at `xhigh`

At `reasoning_effort: xhigh` the model can reason for a very long time. We ran a 24-prompt suite (agentic
tool calls, verifiable puzzles and open questions) with greedy serial decoding and a 16,000-token cap, on ENGINE
builds `165ec33` and `d1c8ca9` of 2026-09-14; it has not been re-run on the current build:

- **Without any decoding policy:** 20/24 runs closed their reasoning block, and 16/21 verifiable answers were
  correct. Two runs fell into literal loops (up to 97.7% repeated 12-grams).
- **With ENGINE's "soft stop"** (a logit bias on `</think>` that ramps up with the length of the reasoning):
  24/24 closed, 21/21 were correct, and 22% fewer tokens were used. Every case the no-policy run got right was
  still right [6].

With the soft stop and MTP speculative decoding together (build `d1c8ca9`), the suite also closed 24/24 and scored
21/21. We then ran a six-turn agentic tool loop 8 times (4 at `xhigh`, 4 at `medium`) through ENGINE's HTTP server
with MTP and the soft stop on, a loop guard (3) and a 16,000-token cap, on an ENGINE build of the same day, sampled
with the recommended settings. All 48 turns returned one valid tool call [7]. (In current ENGINE builds a request
with a loop guard or thinking budget decodes without MTP, so that exact configuration no longer runs MTP.)

On another runtime, set a generous `max_tokens` or use `medium` / `low` effort.

## Speed and memory (ENGINE only)

Unless noted, these were measured on a Mac Studio (M3 Ultra, 80-core GPU, 512 GB) under macOS 27.2 with
`iogpu.disable_wired_collector=1`, on ENGINE build B50 (the production build before the current one), with MTP
speculative decoding using the included head. B50's default `reasoning_effort` was `medium`, and no benchmark client
below set it; generated tokens include the reasoning. GB means 10^9 bytes. Other runtimes and other machines will
differ.

| Context | 0.5k | 8k | 32k | 64k | 256k |
|---|---|---|---|---|---|
| Generation, 1 request (tok/s) | 102.7 | 91.3 | 88.6 | 80.4 | 77.0 |
| Prefill (tok/s) | 765 | 1,185 | 1,255 | 1,251 | 1,196 |
| Time to first token (s) | 0.67 | 6.95 | 26.1 | 52.1 | 218.0 |

Single requests to ENGINE's production launcher, measured with the `llm_context_benchmarks` client: a cold prompt
of the given length, 256 generated tokens, `temperature` 1.0, client-side timing. Two runs per point; each figure is
the better of the two runs for that metric, so one column can combine both runs. Generation differed by up to 11%
between the two runs; at 0.5k, prefill and time to first token differed by about 25% [8].

- **Concurrent sessions, summed decode** (production launcher, N cold prompts released together, 256 generated
  tokens each, greedy, one run; the sum of the per-session decode rates): 8 sessions reach 226.0 / 221.7 /
  206.4 tok/s at 2k / 8k / 32k context, and 4 sessions reach 147.8 / 152.4 / 146.4 / 151.5 tok/s at 2k / 8k / 32k /
  64k [8].
- **Full-context cold prefill:** the server-reported time to first token for a 258,089-token prompt was 210.49 and
  210.80 s in two runs of the project's release-gate harness, which starts a fresh `engine serve` per run and sends
  one greedy request; it did not go through the production launcher [8].
- **MTP draft acceptance:** in one real (uncontrolled) agentic coding session through the production launcher,
  its batched MTP rounds accepted 253,428 of 338,510 drafted tokens (solo MTP rounds not counted) [8].
- **Memory:** the weights take 194.9 GB. Peak memory during a single 262,144-token prefill (`engine nll`, chunk
  4096, build B44) was 252.6 GB [3].
- **Reproducibility** (measured on ENGINE build `d1c8ca9`, 2026-09-14): MTP output is not bit-identical to serial
  greedy decoding. The trunk's first-row argmax differs in about 1 to 1.5% of rounds, always at an exact tie or a
  one-bf16-ulp margin. On the suite above the two paths scored the same [7].

## How to run (ENGINE)

E9 has only been run on an M3 Ultra with 512 GB. ENGINE's `engine serve` refuses to start unless
with ENGINE's launcher (`tools/serve.sh`), 0.9 × physical memory − 80 GB − the resident model (about 195 GB) is positive, in decimal GB (10^9 bytes), so it
needs roughly 306 GB (about 285 GiB) or more of physical memory before any request is admitted (DERIVED from the
code, not tested). In practice that means the 512 GB configuration: a 256 GB Mac (256 GiB, about 274.9 GB) is
refused at startup. We have not tested anything in between.

Building ENGINE requires Xcode with the Swift 6.3 toolchain, Xcode's Metal Toolchain
(`xcodebuild -downloadComponent MetalToolchain`) and `cmake`; see the
[ENGINE README](https://github.com/KrakowiakK/ENGINE#readme). The commands, in the README's order:

```sh
git clone https://github.com/KrakowiakK/ENGINE && cd ENGINE
swift build -c release --product engine    # .build/release/engine
tools/build-mlx-metallib.sh                 # .build/release/mlx.metallib, next to the binary
hf download aniolekx/Qwen3.8-Flash-Next-E9-MLX-8bit --local-dir weights/e9
tools/serve.sh                              # OpenAI-compatible API on http://127.0.0.1:8099/v1
```

`tools/serve.sh` loads `weights/e9` by default. Set `ENGINE_MODEL=/path/to/dir` to load it from somewhere else. The
launcher enables MTP speculative decoding with the included head, the soft stop and `xhigh` as the default effort.

```sh
curl http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "e9",
  "messages": [{"role": "user", "content": "Explain Gated DeltaNet in three sentences."}],
  "reasoning_effort": "medium",
  "temperature": 1.0, "top_p": 0.95, "top_k": 20
}'
```

ENGINE decodes greedily (`temperature` 0) unless the request sets `temperature`, so pass the recommended sampling
settings explicitly, as above.

ENGINE's HTTP server is text-only: it keeps the text parts of each message and silently drops image and other
non-text content parts. In ENGINE, image and video input exists only in the `engine generate` development command
(`--image a.png[,b.png]`, `--video clip.mp4`). That path was tested with E9 on ENGINE builds of 2026-08-31 (single
and multiple images, video) [9] and has not been re-tested on the current build.

## Chat template and sampling

The chat template is Qwen's own, unchanged. Its reasoning levels are:

| `reasoning_effort` | Effect |
|---|---|
| `xhigh` (default) | adds a system instruction to think carefully, validate assumptions and consider alternatives |
| `medium` | no extra instruction |
| `low` | adds a system instruction to keep thinking brief |

In the template, `enable_thinking: false` pre-fills an empty `<think></think>` block. ENGINE's HTTP server does not
forward `enable_thinking` or `chat_template_kwargs`, so over ENGINE use `reasoning_effort: low` for brief thinking.
The template also supports tool calls (`<tool_call>` / `<function=...>` format) and images and video in user
messages (ENGINE's HTTP server does not pass images or video on; see above).

The recommended sampling settings come from `generation_config.json`: `temperature 1.0`, `top_p 0.95`, `top_k 20`,
sampling on. The stop tokens are `<|im_end|>` (248046) and `<|endoftext|>` (248044).

## Other runtimes

- **mlx-lm (Python):** an `mlx-lm` pull request that implements `qwen4_exp` (#1788, commit `6cae162`) loaded E9
  non-strictly, skipping the MTP head, and served as an independent reference in ENGINE's parity tests [5]. We have
  not tested a released `mlx-lm`, `mlx-vlm` or upstream `mlx-swift-lm`.
- **Pitfalls for any port** found during ENGINE's bring-up:
  - The n-gram hash must be built with seed 1234. The upstream config omits the seed, and ports that default it to
    0 gather the wrong n-gram rows. E9 sets `text_config.seed: 1234` explicitly [12].
  - The pull-request branch above drops the query's own partial block during sparse-attention prefill; the official
    code keeps it [4].
  - The n-gram table is sharded as `ngram_embedding.shard_N`. The `quantization` block lists both the `shard_N` and
    `shards.N` spellings so that either loader finds it [5].
- **Read the `quantization` block, not `quantization_config`.** `quantization` lists every quantised module with
  its bits and group size, and marks several bf16 modules `false`. It does not list every unquantised module: it has
  no entry for the 48 bf16 trunk MoE routers (`model.layers.N.mlp.gate`; only the MTP head's router is marked
  `false`) or for the bf16 vision tower (its Linear layers `attn.qkv`, `attn.proj`, `mlp.linear_fc1` /
  `linear_fc2` and the merger's `linear_fc1` / `linear_fc2`, and its position-embedding table). Those modules have
  no `scales` tensors, so a loader that checks the tensors finds them unquantised; a loader that applies the block's
  top-level default (4-bit, group 32) to every module without an entry would not. `quantization_config` is an
  older copy that also lacks the entries for `embed_tokens` and the n-gram table, so a loader that reads it would
  treat those as the default 4-bit group 32.
- **Differences from the upstream `config.json`:**
  - added `quantization`, `quantization_config` and `engine_recipe`;
  - added `text_config.seed: 1234`;
  - `rope_parameters` uses the key `type` where upstream uses `rope_type`;
  - `text_config.mtp_num_hidden_layers` is `0` (upstream: `1`) although the MTP head is present (the `mtp` block
    and the `mtp.*` tensors describe one layer). ENGINE loads the head from the `mtp.*` tensors. A loader that keys
    on `mtp_num_hidden_layers` may not see it.
  - `engine_recipe` says the PLE n-gram table is "at the official 8-bit g32". That is imprecise: the upstream
    checkpoint is entirely unquantised, and the table was quantised to 8-bit group 32 by ENGINE's converter from the
    official bf16 table (see [Quantisation](#quantisation)). The `quantization` block is correct.

## License

The base model is released by Qwen under the **Qwen Community License 1.0**, and this derivative is distributed
under the same license. The full text is included as [`LICENSE`](LICENSE), as condition 1 requires. In short (the
`LICENSE` file is what governs):

1. **Notice and attribution.** The copyright and permission notice must be included in all copies or substantial
   portions. If the model or a derivative is used in commercial products or services with more than 100 million
   monthly active users or more than US$20 million monthly revenue, the model name must be displayed prominently in
   the user interface.
2. **Separate license for Model-as-a-Service and AI Work Assistant businesses.** If you or your affiliates run a
   Model as a Service business or an AI Work Assistant business (an independent AI product mainly for coding or
   office productivity), you must get a separate license from Qwen before using the model or its derivatives for any
   commercial purpose. This does not apply to internal use that does not make the model, its outputs or its
   capabilities available to any third party. The license defines "Model as a Service" as "giving a third party
   access to language model inference or fine-tuning (e.g., via API or a hosted endpoint) in a manner that allows
   such third parties to exercise meaningful control over the inputs, parameters, or training data. This does not
   include the mere relaying of requests to models hosted by other third parties."

The software and its outputs are provided "as is", without warranty. See `LICENSE` for the exact terms, the
definitions and the contact address for licensing questions.

## Credits

- **Qwen:** Qwen3.8-Flash-Next, its weights, tokenizer, chat template, MTP head and vision tower.
- The vision-tower tensors and the bf16 MoE router weights (`mlp.gate`) come from the public MLX conversion
  [Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP), which
  keeps them in bf16; the vision-tower tensors are byte-identical to those in that conversion.
- **Quantisation, validation and ENGINE:** Krzysztof Krakowiak
  ([Hugging Face: aniolekx](https://huggingface.co/aniolekx), [GitHub: KrakowiakK](https://github.com/KrakowiakK)).

## References

The IDs below (OBS-ENG-…, REFUT-ENG-…) are internal identifiers of entries in the project's research log, which is
not published. They are listed so that the scope of each figure is explicit. All claims are MEASURED on a Mac Studio
M3 Ultra (512 GB) unless noted.

1. **OBS-ENG-069.** Teacher-forced NLL (`engine nll`, chunk 4096) on 131,071 positions of a Polish novel.
   The predecessor artifact ("E8h") had routed experts and `lm_head` at 8-bit group 64 from bf16 and every other
   projection at 8-bit group 64, but `embed_tokens` and the n-gram table at 4-bit group 32. The reference build was
   bf16 everywhere except `lm_head` and the hyper-connection mixes (8-bit group 64) and the n-gram table (8-bit
   group 32).
2. **OBS-ENG-102.** Paired per-position NLL, chunk 4096. Technical corpus: 16,383 positions of ENGINE
   repository source code and Markdown. Prose corpus: 32,767 positions of a Polish novel. The measured variant's
   outputs are bit-identical to E9's (OBS-ENG-105).
3. **OBS-ENG-204.** E9 vs E9 with 4-bit group-32 trunk experts. 15 teacher-forced series (prose, private
   and public code, agent transcripts), moving-block bootstrap, the same ENGINE binary (build B44) for both arms,
   macOS 27.2. Teacher-forced scoring only; no generation test was run. The attribution of the code-context loss to
   spurious end-of-turn predictions is DERIVED from per-file sums. The peak memory figure is from the same runs.
4. **OBS-ENG-153, OBS-ENG-151, OBS-ENG-152.** Official `transformers` 5.16.1 Qwen4Exp modules in float32
   on E9 weights dequantised exactly, fed ENGINE's own layer inputs, over a 16,126-token reasoning chain (4 prefill
   chunks plus 64 decode steps). OBS-ENG-152 records the partial-block behaviour of the `mlx-lm` pull-request branch.
5. **OBS-ENG-106, OBS-ENG-150.** ENGINE vs the `mlx-lm` pull request #1788 (`qwen4_exp`) on E9, on four
   frozen fixtures of 5, 6, 345 and 4,096 tokens. OBS-ENG-150 records that the reference loaded E9 non-strictly,
   skipping the MTP head.
6. **OBS-ENG-154.** ENGINE builds `165ec33` / `d1c8ca9` (2026-09-14). E9, `xhigh`, greedy, serial decoding.
   24 cases with computed ground truth where one exists, cap 16,000 tokens. The bias ramps from 2,000 to 8,000
   thinking tokens up to +12 on `</think>`.
7. **OBS-ENG-156.** ENGINE build `d1c8ca9` (2026-09-14; the agent loop on a build of the same day). E9, `xhigh`,
   MTP K=3, with the soft stop. The agent loop ran on `engine serve` with
   MTP K=3, the soft stop with its bias deadline at 14,000 thinking tokens (+52 logits on `</think>`), a loop guard
   and a 16,000-token cap, sampled at temperature 1.0 / top-p 0.95 / top-k 20. The first-row divergence figure comes from snapshot/restore
   comparisons on three prompts.
8. **OBS-ENG-207.** ENGINE production build "B50", macOS 27.2, default `reasoning_effort` `medium`.
   - Single-request figures: `llm_context_benchmarks` (commit 5574d0b) against the production launcher, cold
     prompts, 256 generated tokens, temperature 1.0, 2 runs per point; each metric is the better of the two runs,
     computed from the client's timing columns.
   - Concurrent figures: the project's concurrent client against the production launcher, 256 generated tokens per
     session, greedy, cold prompts, one run.
   - Full-context prefill: the release-gate harness (a fresh `engine serve` per run, one greedy request of 64
     output tokens), an ABBA pair against build B44 on a 258,089-token prompt; B50's two runs are quoted.
   - MTP acceptance: one real agentic session with batched MTP through the production launcher.
9. **OBS-ENG-108, OBS-ENG-110, OBS-ENG-118, OBS-ENG-119.** ENGINE builds of 2026-08-31. Through
   `engine generate`, the vision tower loads and answers about images (a blind description test), multiple images
   and video. Not re-tested on the current build.
10. **OBS-ENG-120, OBS-ENG-121.** Audio token ids are present, but there are zero audio tensors in this
    checkpoint and in the upstream release.
11. **OBS-ENG-011, OBS-ENG-012, and the bring-up log of 2026-08-27.** Bring-up on an older uniform 4-bit
    group-32 third-party conversion: the non-language output was seen in three runtimes. The log records that the
    seed-1234 fix alone, applied in ENGINE, still gave non-language output. Applying both changes together (424
    small modules replaced with the official bf16 tensors, plus n-gram seed 1234) restored coherent output,
    measured once, in the `mlx-lm` PR #1788 runtime.
12. **REFUT-ENG-001.** The checkpoint's own `layer_multipliers` buffer equals the seed-1234 rebuild.
