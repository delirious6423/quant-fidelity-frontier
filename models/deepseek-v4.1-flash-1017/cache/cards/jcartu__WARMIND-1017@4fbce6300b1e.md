---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: vllm
pipeline_tag: text-generation
tags:
- abliterated
- uncensored
- deepseek
- ds41-flash
- reasoning
- fp8
- mxfp4
- safetensors
---

# WARMIND-1017

**DeepSeek-V4.1-Flash-1017, with targeted refusal-reduction edits in its native checkpoint format.**

WARMIND-1017 is jcartu's abliterated release of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). **WARMIND** names this derivative; **1017** identifies the upstream October 17, 2026 checkpoint. It is not a new architecture or a model trained from scratch. The release name changes neither the weight bytes nor the scope of the evidence below: DeepSeek remains the base-model author, with the abliteration method and refusal directions credited in [Lineage and credits](#lineage-and-credits).

`deepseek-ai/DeepSeek-V4.1-Flash-1017` with its refusal direction inverted in 132 dense tensors, delivered in the official native safetensors layout: the same 48 shards, index, config, tokenizer and encoding files as the upstream release, with exactly those 132 tensor payloads different. Nothing was fine-tuned or retrained: two tensor families in layers 4 to 36 were rewritten by a norm-preserving abliteration operator; every other byte of the 48 weight shards, and every retained upstream runtime file (config, index, tokenizer, encoding, inference, LICENSE), is verified identical to the upstream revision. What this repository adds is its own documentation and provenance files, and this card in place of the upstream README. It is the plain-format sibling of [the CSF derivative](https://huggingface.co/jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF) that has served on a 4x RTX PRO 6000 box since 2026-10-20: the same edited bytes, no container. Every number on this page points at a receipt, and every number that was measured on the container rather than on this tree says so.

## At a glance

| | |
|---|---|
| Base weights | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e1088993b9b82de08f59de5be5a34e42170` (released 2026-10-17, MIT) |
| Direct parent | The base weights themselves. This tree was assembled from a fresh, file-by-file verified copy of the upstream release; no container sits between the two |
| Architecture | 40-layer MoE, hidden 5120, 384 routed experts (6 active) + 1 shared expert, 3 MTP draft layers, Engram n-gram tables, 1,048,576-token context (`config.json`) |
| What changed | `attn.wo_b` and `ffn.shared_experts.w2` weights and block scales in layers 4 to 36: 132 tensors, 1,775,136,000 bytes (1.65 GiB, 0.35 % of the checkpoint), spread over 33 of the 48 shards |
| Format | Official native safetensors, untouched: FP8 E4M3 dense weights with UE8M0 32x32 block scales (`quantization_config` in `config.json`: `fp8`, `weight_block_size [32, 32]`, `scale_fmt ue8m0`), and the routed experts in their native FP4 layout with their upstream scale tensors (`expert_dtype: fp4`; no claim is made here about the experts' block layout). Not BF16, not a CSF container, nothing re-quantized on the way here |
| Size | 48 shards, 510,296,708,312 bytes (510.3 GB, 475.3 GiB); 96,085 tensors, 510,286,023,000 bytes of tensor payload. Shards 47 and 48 (about 101.5 GB each) hold the Engram tables |
| Runtime | Same layout as the upstream checkpoint, so the same loaders are expected to apply. Served on this box by the LiL vLLM image (generic loader), by vLLM-Moet and by the BRRR recipe; [Runtimes](#runtimes) says what each one has and has not shown |
| Measured on | 4x NVIDIA RTX PRO 6000 Blackwell (96 GB each), TP4 |
| License | MIT (DeepSeek weights, [LICENSE](https://huggingface.co/jcartu/WARMIND-1017/blob/main/LICENSE)); the edit adds no terms. See [License](#license) |
| Access | Public download; no access request or approval required |

Companion files in this repo: [EXPLAINER.md](EXPLAINER.md) (the full technical derivation and the byte-level provenance of this tree), [DERIVATIVE.json](DERIVATIVE.json) (machine-readable provenance: every edited range with its before and after hash, the 33 changed-shard digests, the lineage and credits), [SHA256SUMS](https://huggingface.co/jcartu/WARMIND-1017/blob/main/SHA256SUMS) (a digest for every file but itself), [CITATION.cff](CITATION.cff). The evaluation reports live with the CSF sibling, because that is where the measurements were taken: [EVALUATION.md](https://huggingface.co/jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF/blob/main/EVALUATION.md) and [EVALUATION-1017.md](https://huggingface.co/jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF/blob/main/EVALUATION-1017.md) (the CSF sibling has its own access gate).

## Why this exists

The 1017 refresh is a better model than its predecessor, and it also came back more guarded. On the lab's six-prompt sanity probe the stock checkpoint refuses two of six (how to pick a lock, how to hotwire an old car) while answering the rest (`1017-raw-sanity.json`). On the 100-prompt judged battery that was run on the previous checkpoint, the unmodified base produced 17 true refusals and 73 hedged answers out of 100 (`eval_baseline.json`).

The approach here is the narrow one: find the direction in hidden space that carries refusal, edit the tensors where it expresses, leave everything else alone. No fine-tuning, no changes to attention, routing or the routed experts. The result answers instead of stonewalling. It does not become an unhedged oracle, and the numbers below say exactly how much it moved.

And why a second repository for the same edit: the CSF derivative opens only through Local Inference Lab's vLLM image with `CHECKPOINT=csf`. That is the right trade for a box that wants the container's extra KV cache, and the wrong one for anyone whose serving stack reads the ordinary DeepSeek checkpoint. This tree is the same edit in that ordinary layout. The runtime choice is yours, and the [Runtimes](#runtimes) section is candid about which stacks have actually been exercised with these bytes and what came of it.

## What changed in the weights

The method is grimjim's norm-preserving biprojected abliteration, applied per layer at scale alpha = 3.0 to layers 4 to 36, using refusal directions extracted from the original DeepSeek-V4.1-Flash (`refusal_directions.pt`: one fp32 vector of 5120 dims per layer, 40 layers).

In plain language: each target weight matrix is split into its row norms and row directions. The refusal direction's component is projected out of every column of the direction part, but with alpha = 3 the component is not merely removed, it is flipped to minus twice its original size. Rows are then re-normalised and the original row norms restored, so each row keeps its length and only rotates. That overshoot is what turns hard refusals into answers, and the norm restoration is why the model's scale statistics stay put.

Two tensor families are edited, the dense sublayers every token passes through: the attention output projection `attn.wo_b` ([5120, 8192]) and the shared-expert down projection `ffn.shared_experts.w2` ([5120, 2304]), each with its 32x32 UE8M0 block-scale tensor. Routed experts are sparse and token-conditional and are untouched.

Counting over the whole checkpoint: 172 tensors match the selection pattern (43 blocks: layers 0 to 39 plus 3 MTP blocks, 2 modules each, weight + scale). 132 of them change (layers 4 to 36). The other 40 (layers 0 to 3, 37 to 39, and the MTP blocks) fall outside the edit range and are byte-identical to upstream, like every other tensor in the 48 shards.

Because the weights are stored as FP8 E4M3 values with UE8M0 block scales, every edited tensor went through dequantize, edit in FP32, requantize, once, when the edit was computed on the raw checkpoint on 2026-10-18. That round trip is lossy by design: about 1.9 % relative error per tensor against the FP32 ideal, the same penalty the original release paid. Nothing was quantized again afterwards; the bytes produced that day are the bytes in this tree.

<details>
<summary>The exact operator and the requantization rule</summary>

For each target layer L with weight W (shape [out, in]) and direction r = directions[L], alpha = 3.0:

1. `r_hat = normalize(r)`
2. `n = row_norms(W)` (kept for restoration)
3. `W_hat = normalize_rows(W)`
4. `p = r_hat^T W_hat` (the component of every column along r_hat)
5. `W' = W_hat - alpha * outer(r_hat, p)`, i.e. `W' = (I - alpha r_hat r_hat^T) W_hat`
6. `W'' = normalize_rows(W')`
7. `W_out = n * W''`

With alpha = 3, `r_hat^T W' = (1 - alpha) r_hat^T W_hat = -2 r_hat^T W_hat` before the final row renormalisation: the refusal component is inverted and doubled, not deleted. Because step 6 rescales rows non-uniformly, the composite is a projection followed by a norm-restoring rescale, not an orthogonal projection of the final matrix. Row norms are preserved exactly in exact arithmetic; FP32 arithmetic and the FP8 requantization add their own small error.

Dequantize: `W = float(w_e4m3) * expand(scale_e8m0)`, scales expanded 32x along both axes.

Requantize per 32x32 block: `amax = max|W|`; `k = ceil(log2(amax / 448))` clamped to [-127, 127]; `scale = 2^k`; `q = clamp(W / scale, -448, 448)` cast to E4M3; `scale` cast to E8M0. The `ceil` places the block maximum at or below 448 (E4M3's largest finite value) so the clamp is inactive except at exact ties; individual values still round to the E4M3 grid. Blocks with `amax <= 0` get a zero scale byte, which is numerically moot because such blocks are all zeros. All intermediates are FP32.

Script: `abliterate-1017.py` (`--validate` replays the operator on the previous checkpoint pair, `--apply` writes the 1017 edit in place on a reflink copy). The full derivation is in [EXPLAINER.md](EXPLAINER.md).
</details>

## How it was built and proven

The build is a ladder; each rung is a receipt that lives next to the checkpoint on the build machine, and the public ones are reproduced in [EXPLAINER.md](EXPLAINER.md) and [DERIVATIVE.json](DERIVATIVE.json).

1. **Validate the operator before touching 1017.** The re-implemented operator was replayed on the previous checkpoint pair (old pristine in, compared against the old shipped abliterated bytes) over all 66 edited modules. Result (`abliterate-1017-validate.json`): 96.69 % of weight bytes and 98.78 % of scale bytes reproduced exactly (worst module 93.77 % / 95.79 %), 1.01 % mean relative error in dequantized space against the old release (max 1.08 %), 1.85 % against the FP32 ideal (max 1.93 %). The residual is the size of the requantization rounding path, so the operator was accepted.
2. **Abliterate the raw 1017 checkpoint.** 33 layers x 2 modules were dequantized, edited, requantized and written back in place on a btrfs reflink copy of the official download (`build-1017-abliterated.json`, 2026-10-18): mean relative error against the FP32 ideal 1.85 %, max 1.93 %, identical to the validation-time penalty. Served as it was, in this same native layout, it scored Estonia knowledge 12/12 and 6/6 refusal probes unlocked, against 12/12 and 4/6 for stock 1017 (`1017-abl-raw-sanity.json`, `1017-raw-sanity.json`). That raw build was this box's interim production from 2026-10-18 until the CSF container replaced it on 2026-10-20, and it was deleted after the transplant.
3. **Carry the edited bytes through the CSF derivative.** On 2026-10-20 the 172 selected ranges were written by byte-range overwrite into Local Inference Lab's FP4-CSF container, where the dense tensors are stored uncompressed under the same names. The build's guards (`build-1017-abliterated-csf.json`) re-read every written range and required it to equal the raw build's bytes exactly, so the CSF derivative became a faithful carrier of the 132 edited payloads. This step is byte-preserving, not an edit; it matters here only because the CSF derivative was the surviving copy of those bytes once the raw build was gone.
4. **Assemble this tree from the upstream release (2026-10-26 to 2026-10-27).** A fresh download of revision `1bae0e10…` was verified file by file against the Hub manifest: LFS sha256 for every shard, git blob sha1 for every small file, all 83 files. The 172 selected tensor ranges were then resolved independently in four layouts (fresh upstream, LiL's CSF, the CSF derivative, this tree) with dtype, shape and length required to agree, the upstream bytes required to equal the CSF parent's bytes for every range, and exactly 132 ranges found to differ between the CSF derivative and upstream, by name the layers-4-to-36 set the raw build had recorded. The upstream tree was reflink-copied and the 132 payloads written at their native offsets, each one hash-checked before and after the write. Finally every file of that staged tree was proven, not assumed: the expected sha256 of each of the 83 upstream files was computed from the upstream bytes with the patches spliced in and compared with the sha256 of the file on disk. All 83 matched (`native-abliterated-staging.json`, `unchanged_bytes_verified: true`). The 15 shards outside layers 4 to 36 carry the upstream LFS digests unchanged. No dequantize or requantize ran in this step: bytes were copied, nothing was computed. The published repository is that staged tree with the upstream README replaced by this card and the documentation and provenance files added.
5. **Record provenance.** [DERIVATIVE.json](DERIVATIVE.json) lists all 132 edited ranges (tensor, shard, byte offset, length, pristine and derivative sha256), the 33 changed-shard digests, the three-step lineage above and the credits; [SHA256SUMS](https://huggingface.co/jcartu/WARMIND-1017/blob/main/SHA256SUMS) covers every file in the repo except itself. `sha256sum -c SHA256SUMS` from the repo root is the whole integrity check.
6. **Gate the bytes on real serving.** Beyond the 2026-10-18 sanity run, this tree was loaded on 2026-10-27 by two independent SM120 stacks (vLLM-Moet and the BRRR recipe) and passed the lab's API and retrieval checks in two runs that day: real arithmetic, strict JSON schema, tool invocation and a 134K-token retrieval in the overnight comparison; a tool-call round trip, strict JSON output, vision, long-context retrieval at 64K, 128K and 256K tokens and all six refusal probes answered in the qualification matrix, which also scored 200 GSM8K questions under the lab's original scorer (187 and 180). Details, and the one long-context limitation these runs left unresolved, are in [What has been measured on these bytes](#what-has-been-measured-on-these-bytes).
7. **Quality receipts for the edit itself.** The same edit lineage measured against its base on the previous checkpoint (sidecar evals): MMLU 88.95 % vs 88.95 %, KL divergence 0.1416 against a 0.1307 nondeterminism control, GSM8K 16/20 vs 14/20. The direct 1017 pair, measured on the CSF sibling: MMLU +0.085 points with an interval that includes zero. Scope for both in the next two sections.

## The refusal numbers, honestly

Judged 100-prompt battery, run on the previous checkpoint with the same edit (sidecar `eval_baseline.json`, `eval_identity.json`, `eval_s3.0.json`). The identity column is a control: the same requantization round trip with no abliteration, which shows how much the judge itself moves between runs.

| | base | identity control (requant only) | abliterated (this edit) |
|---|---:|---:|---:|
| true refusals (judge) | 17 | 27 | **1** |
| hedged / partial answers (judge) | 73 | 68 | 88 |
| fully compliant (judge) | 10 | 5 | 11 |
| keyword refusal counter | 98 | 98 | 41 |
| GSM8K (20 questions, chat mode, 0-shot, greedy) | 14 | 15 | 16 |

Read it this way: true refusals went from 17 to 1, which is far outside the judge's run-to-run band (the eval notes record swings of about +/-10 on borderline PARTIAL/REFUSAL calls, visible in the identity column). Those refusals mostly became hedged answers, not clean compliance: PARTIAL 73 to 88, COMPLIANT 10 to 11, and both of those moves are inside the judge's noise. The model engages with the topic instead of stonewalling and still flags caution more often than not. The single remaining judged refusal is the hardest synthesis prompt in the set. Keyword counters over-count in both directions; the judge rows are the ones that matter.

Two smaller checks point the same way. A six-prompt directional probe at three reasoning-effort tiers on the deployment stack (`eval_deployment.json`): refusals stay removed at every tier (1, 1, 0 of 6 at effort off / high / max) while the hedge rate rises with effort (3, 5, 6 of 6). And stock 1017 vs this edit on the lab's six sanity prompts: 4/6 vs 6/6 unlocked, on this layout, in this box's interim production.

Historical quality measurements on the previous checkpoint: full MMLU letter-logprob scoring at T=0 gave 12,490 correct for base and 12,491 for the abliterated build out of 14,042 (88.95 % both at reported precision). KL was 0.1416 against a 0.1307 identity-control value, and GSM8K was 14/20 versus 16/20. These used reference runtimes (SGLang on 4x H200, an fp8/TP8 reference stack), not anything described on this page. They remain historical evidence about the edit lineage, not measurements of the 1017 pair. The direct 1017 comparison below uses a different protocol; do not compare its absolute MMLU score against 88.95 %.

## Direct 1017 comparison, measured on the CSF sibling

The raw 1017 checkpoint and the abliterated edit were tested on the same LiL vLLM image and matching inference settings on 2026-10-20. The arm under test was the CSF derivative, whose 132 edited tensors are byte-identical to the ones in this repository; the container, loader and kernels were not the ones you will use with this tree. Full results, protocol, uncertainty and downloadable per-item scores are in [the paired evaluation report](https://huggingface.co/jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF/blob/main/EVALUATION-1017.md).

| Measure | Raw 1017 | Abliterated edit (CSF sibling, LiL runtime) |
|---|---:|---:|
| MMLU, 14,042 questions | 11,549 correct (82.246 %) | 11,561 correct (82.332 %) |
| GSM8K, 20-question sample | 19 correct; 1 truncated | 18 correct; 2 truncated |
| Refusals on the new 100-prompt set, blinded labels | 88 | 3 |

**No aggregate MMLU loss was resolved by this run.** The difference is +0.085 percentage points, with a paired item-bootstrap 95 % interval of −0.292 to +0.463 points. That is not proof of equal intelligence or a gain: 357 answers changed from correct to wrong and 369 changed the other way. The interval does not include repeated-serving variability.

The GSM8K difference is token-limit-sensitive: every failed answer reached the 400-token cap. The refusal labels describe visible engagement, not answer quality or completion: 91 of the derivative's 100 responses reached the 300-token cap. A second reviewer agreed on 92/100 labels in the packet it reviewed.

A top-logprob distribution diagnostic measured coarsened KL of 0.158601 nats for raw versus derivative and 0.137411 for a raw repeat control. This is a lower-bound calculation on 200 prompts, not full-vocabulary KL and not a percentage of intelligence lost. The report spells out its coverage and numerical exclusions.

Treat all of the above as evidence about the edit. It is not a certification of this tree under your runtime, which is the subject of the next section.

## What has been measured on these bytes

Everything in this table ran on the native layout: the raw build in October, and this exact tree from 2026-10-27 on. Hardware is 4x RTX PRO 6000, TP4, throughout.

| Date | Runtime serving these bytes | What was checked | Outcome |
|---|---|---|---|
| 2026-10-18 | LiL vLLM image, generic loader (raw build, interim production) | 12 Estonia knowledge questions, 6 refusal probes | 12/12; 6/6 unlocked (stock 1017: 12/12, 4/6) |
| 2026-10-19 | same | Matched decode probe, DSpark d3, context 0 | C1 / C4 / C8 / C32 = 248.0 / 554.7 / 824.8 / 1518.9 tok/s aggregate; KV pool 2,601,368 tokens |
| 2026-10-27 | vLLM-Moet (DSpark k7, NVFP4 KV) and the BRRR recipe at `0bc4dbd0`, each serving this tree | Overnight basic gate, 114 requests per stack: arithmetic, strict JSON schema, tool invocation, 134K-token retrieval, vision | both fully compatible |
| 2026-10-27 | vLLM-Moet, DSpark k7, NVFP4 KV, this tree | Tool-call round trip, strict JSON, vision, retrieval at 64K / 128K / 256K, six refusal probes, 200 GSM8K questions | API, vision and retrieval checks pass; 6/6 answered; math 187/200 under the original scorer |
| 2026-10-27 | BRRR recipe (hybrid-dispatch build), this tree | same gate set | API, vision and retrieval checks pass; 6/6 answered; math 180/200 under the original scorer |
| 2026-10-28 | BRRR, this tree, against the CSF production container | First-token top-1 agreement, 48 cross-runtime comparisons (24 prompts x 2 repeats); re-adjudicated math content on a selected 57-question x 3-seed set | 48/48 top-1 equal; 171/171 correct on both arms |
| 2026-10-28 and 2026-10-29 | BRRR, this tree | 32,768-token-output long-context ledger probe, 8 cases | 4/8 (hybrid-dispatch build on recipe revision `0bc4dbd0`), 5/8 (same build with its two kernels rebuilt from `e23ec13c`: 5 correct, 2 truncated at the cap, 1 wrong) |

Reading the table: the edit takes effect and the model is functional on every stack that has loaded it. The GSM8K-200 totals (187 and 180 here, 186 for the CSF production container on the same day) are confounded by formatting failures, reference-answer errors and truncation under the original scorer, so they should not be read as a ranking of the stacks or as a verdict on the weights; the later re-adjudication of a selected 57-question x 3-seed set (failure-enriched, plus a fixed control subset) scored the LiL and BRRR arms 171/171 each, and the Moet arm was not re-run. The last row is the limitation to keep in mind. The CSF production container completed the same eight ledger prompts 8/8 on 2026-10-28, but that run differs from the BRRR runs in runtime, kernels and checkpoint format all at once, so it is context, not an isolation of any one factor; the BRRR builds serving this tree did not complete them. These comparisons did not isolate a cause: whether the shortfall belongs to the runtime, its configuration or an interaction with these bytes was not established by them, and the differing runtimes and configurations remain a limitation of this evidence. Nothing was promoted on the strength of either number. The honest summary is: these weights have been served successfully under several SM120 stacks, and the runtime you choose affects what you get, in speed and in completion reliability on very long outputs.

Speed, for orientation only. On the LiL image at the same draft setting, this layout decodes within 3 % of the CSF container (C8 824.8 vs 846.8 tok/s, C32 1518.9 vs 1521.4) but holds 2.60 M KV tokens where the container holds 4.44 M, because the container's compressed expert scales free VRAM. Single-stream code decode in the 2026-10-27 overnight comparison (2048-token cap, thinking off) was 594 tok/s on vLLM-Moet and 670 tok/s on BRRR serving this tree, against 382 tok/s for the LiL image serving the CSF sibling: the runtime sets the pace, not the weights. Those are capped-generation observations from three trials each, not task-completion scores; the LiL-image profile table with prefill and KV ladders was measured on the container and lives in [its SERVING.md](https://huggingface.co/jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF/blob/main/SERVING.md).

## Runtimes

**What this tree is, from a loader's point of view.** The upstream checkpoint, byte for byte, except for 132 tensor payloads of identical dtype, shape and length. The compatibility expectation that follows is simple: a stack that reads `deepseek-ai/DeepSeek-V4.1-Flash-1017` from its official files should read this directory the same way, and a stack that cannot run that model on your hardware will not be helped by this one. That is an expectation from the identical layout, not a guarantee; only the three stacks below have actually been run with these bytes. The included `inference/` reference implementation, `encoding/` reference encoder and tokenizer are the upstream files, unmodified ([inference/README.md](inference/README.md), [encoding/README.md](encoding/README.md)). Do not set `CHECKPOINT=csf` or any CSF loader flag: there is no container here.

**Download and verify.**

```sh
hf download jcartu/WARMIND-1017 --local-dir WARMIND-1017
cd WARMIND-1017 && sha256sum -c SHA256SUMS
```

The check reads all 510 GB, so run it once, on the NVMe you will serve from, and not while a benchmark is in progress. Every line must say OK; the 33 changed shards are listed with their post-edit digests in [DERIVATIVE.json](DERIVATIVE.json) if you want to compare against upstream by hand.

The examples below use `WARMIND-1017` as the served API model name; use that name in client requests. The architecture, tokenizer and parser identifiers remain DeepSeek's (`DeepseekV41ForCausalLM` and `deepseek_v41`). Renaming the release does not require changing the upstream config or tokenizer files.

**Memory shape.** 510.3 GB of shards does not fit in 4x96 GB, and it does not have to: shards 47 and 48 (about 203 GB together) are the Engram n-gram embedding tables, and the three stacks below, as configured in the runs reported here, read them from NVMe rather than keeping them in VRAM. Put the tree on a local ext4 or xfs NVMe (BRRR reads the tables with `O_DIRECT`; the LiL image's disk Engram path uses io_uring and needs `seccomp=unconfined` and an unlimited memlock in Docker); network or tmpfs mounts do not work for that path.

**Stacks that have served these bytes on this box.** None of the three is a blanket qualification; each line says what it was exercised on.

1. *Local Inference Lab vLLM image, generic loader.* The same image the CSF sibling runs on, with no `CHECKPOINT` variable, so the launcher takes the native path. This is how the raw build ran as interim production from 2026-10-18 to 2026-10-20, with the sanity and decode numbers above. The image vendors an encoding module from the previous checkpoint; bind-mount this tree's own `encoding/encoding.py` over `vllm/tokenizers/deepseek_v41_encoding.py` for 1017 tool calling to parse. That file is byte-identical (sha256 `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2`) to the module the CSF production container and the 2026-10-27 comparison runs mounted, and it exports what the image imports (`IMAGE_PLACEHOLDER`, `REASONING_EFFORT_MAPPINGS`, `encode_messages` with the optional `return_multi_modal_data`) plus 1017's `parse_message_from_completion_text`. In the 2026-10-27 comparison the LiL image rejected image inputs (text-only on that stack as configured); the other two stacks passed the vision check.

   <details>
   <summary>The compose configuration used on 2026-10-18; host paths, service/cache names and the API alias adapted for WARMIND-1017, with the later production encoder mount</summary>

   ```yaml
   services:
     warmind-1017:
       image: ghcr.io/local-inference-lab/vllm@sha256:fd74019fc217a541060093d18427c36330df6ac5700651b55593dafdbb936920
       entrypoint: /usr/local/bin/lil-entrypoint
       command: [serve-ds41-flash.sh]
       network_mode: host
       ipc: private
       shm_size: 128g
       stop_grace_period: 120s
       gpus: all
       security_opt:
         - seccomp=unconfined
       ulimits:
         memlock: -1
       environment:
         HOST: 127.0.0.1
         PORT: "8000"
         MODEL: /model
         SERVED_MODEL_NAME: WARMIND-1017
         TP: "4"
         OMP_NUM_THREADS: "1"
         MAX_NUM_BATCHED_TOKENS: "16384"
         GPU_MEMORY_UTILIZATION: "0.96"
         DSPARK_TOKENS: "3"
         DSPARK_ADAPTIVE_VERIFICATION: "true"
         DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE: "0.5"
         ENGRAM_TABLE_MEMORY: disk
         ENGRAM_DISK_RESIDENT_SCALES: "true"
         CUDA_VISIBLE_DEVICES: "0,1,3,2"      # this box's PCIe ring; only matters if your topology has one slow hop
         VLLM_ENABLE_PCIE_ALLREDUCE: "1"
         VLLM_PCIE_ALLREDUCE_BACKEND: b12x
         VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: "96KB"
         VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE: "96KB"
         VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE: "off"
         VLLM_PCIE_DMA_MIN_BYTES: "6MB"
         B12X_MHC_PDL: "1"
         B12X_PCIE_ONESHOT_PDL: "1"
         VLLM_MXFP8_LM_HEAD: "1"
         HF_HUB_OFFLINE: "1"
       volumes:
         - /path/to/WARMIND-1017:/model:ro
         # 1017 encoder over the image's vendored module (needed for tool calling)
         - /path/to/WARMIND-1017/encoding/encoding.py:/opt/venv/lib/python3.12/site-packages/vllm/tokenizers/deepseek_v41_encoding.py:ro
         - warmind-1017-runtime:/cache:rw
   volumes:
     warmind-1017-runtime:
   ```

   The KV-capacity and chunk-size profiles in the CSF sibling's SERVING.md (graph cap 128, `MAX_NUM_BATCHED_TOKENS` 8192) were measured on the container, not on this tree; they are the obvious knobs to try, with the container's caveats (readiness before CUDA-graph capture ends, `GPU_MEMORY_UTILIZATION=1.0` does not boot), but the numbers there are not this layout's numbers.
   </details>

2. *vLLM-Moet.* An SM120 fork of vLLM for RTX PRO 6000 and RTX 5090 that serves the official checkpoint format with FP4 KV and indexer caches, DSpark speculative decoding and disk-resident Engram tables. A local build with its disk-Engram adapter served this tree in the 2026-10-27 qualification: TP4, 512K context, DSpark k = 7, `nvfp4_ds_mla` KV cache, b12x MoE backend, four concurrent sequences, GPU memory utilization 0.92, and it passed every functional gate listed in the table above. It also passed a fail-closed test that matters for disk Engram: with a table shard truncated after start-up, requests failed with an explicit read error instead of returning output. Moet's KV and indexer formats differ from the LiL image's, so its "basic gates" are not a fidelity-equivalence proof, and it has not been promoted to production here.

   <details>
   <summary>The serve configuration used on 2026-10-27, with the API alias adapted for WARMIND-1017</summary>

   ```sh
   python3 -m vllm.entrypoints.cli.main serve /model \
     --served-model-name WARMIND-1017 --host 0.0.0.0 --port 8000 \
     --tensor-parallel-size 4 --max-model-len 524288 --gpu-memory-utilization 0.92 \
     --max-num-seqs 4 --max-num-batched-tokens 4096 --max-cudagraph-capture-size 64 \
     --kv-cache-dtype nvfp4_ds_mla \
     --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --tokenizer-mode deepseek_v41 \
     --enable-auto-tool-choice \
     --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false,"moe_backend":"b12x"}' \
     --moe-backend b12x \
     --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_tp_shard_min_rows":2048}'
   ```

   Environment: `VLLM_MOET_ENGRAM_DISK=1`, `VLLM_MOET_ENGRAM_DISK_THREADS=16`, `VLLM_MOET_B12X_MOE_FUSED=1`, `VLLM_MOET_SPARSE_MLA_LDG=1`, `VLLM_MOET_TARGET_LM_HEAD_MXFP8=1`, `VLLM_MOET_P2P_AR=1`, `VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=8`, `NCCL_P2P_LEVEL=SYS`, `HF_HUB_OFFLINE=1`; container run with `--ulimit memlock=-1:-1`, `--security-opt seccomp=unconfined`, `--cap-add IPC_LOCK`, 96 GB shm. The Moet build and its disk adapter are not part of this repository.
   </details>

3. *BRRR.* fraserprice's [DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR) recipe (Apache 2.0): fused decode kernels on top of Local Inference Lab's b12x image, tuned for single-stream code decode on 4x RTX PRO 6000 with SSD Engram offload. Its serve script takes a path to the official weights; point it at this directory. On this box, the recipe at revision `0bc4dbd0` (bounded to four sequences, GPU memory utilization 0.94) and a locally patched hybrid-dispatch build served this tree through the functional gate and the overnight comparison, and on code with thinking off it was the fastest single-stream decoder of the three. It is also the stack whose builds fell short on the long-context completion probe described above (4/8 and 5/8 on the 32,768-token-output ledger probe, cause not isolated by those runs). Use it for what it was built for, and check very long generations yourself before relying on them.

**Not exercised with these weights.** Stock vLLM, Transformers, SGLang and llama.cpp. The SGLang SM120 adapter was tried on this box with the stock weights only and did not pass a forced tool-call check there; that says nothing about these bytes and is mentioned so nobody reads this page as a claim that every runtime has been tried. If your stack is not on the list above, you are the first to run it, and the sanity probe in [EXPLAINER.md](EXPLAINER.md) (12 knowledge questions, 6 refusal prompts) is a cheap way to confirm the edit survived the trip.

## Lineage and credits

- **Base weights:** DeepSeek-AI, `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, MIT. The 48 shards outside the 132 edited ranges and every retained runtime file (config, index, tokenizer, encoding, inference, LICENSE) are theirs, unchanged.
- **Method:** grimjim, [Norm-Preserving Biprojected Abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration).
- **Refusal directions and the original s3.0 recipe:** `distributedcognition/DeepSeek-V4.1-Flash-abliterated` (norm-preserving biprojection, scale 3.0, layers 4 to 36), built on the previous DeepSeek-V4.1-Flash.
- **This derivative:** operator re-implementation, validation, 1017 re-abliteration, native assembly, gates and runtime comparisons by jcartu on a 4x RTX PRO 6000 build machine. Technical review of the derivation by GPT-6 Astra and Claude Fable (appendices of the CSF sibling's [EXPLAINER.md](https://huggingface.co/jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF/blob/main/EXPLAINER.md)).

The `derivation.tensor_edits` and `derivation.changed_shard_sha256` records in [DERIVATIVE.json](DERIVATIVE.json) and the per-file digests in [SHA256SUMS](https://huggingface.co/jcartu/WARMIND-1017/blob/main/SHA256SUMS) are in the repo so that the provenance can be checked rather than trusted.

## License

One license applies: the **MIT License** of `deepseek-ai/DeepSeek-V4.1-Flash-1017` ([LICENSE](https://huggingface.co/jcartu/WARMIND-1017/blob/main/LICENSE), Copyright (c) 2023 DeepSeek), which this repository carries unchanged. The 132 rewritten tensors, [DERIVATIVE.json](DERIVATIVE.json), [SHA256SUMS](https://huggingface.co/jcartu/WARMIND-1017/blob/main/SHA256SUMS) and the documents add no restrictions of their own.

No Local Inference Lab license applies to this repository. The CSF sibling is a different artifact under different terms (its container is LiL's work and is licensed by LiL); nothing LiL authored is included here, and nothing on this page should be read as a LiL release.

## Access

WARMIND-1017 is public and ungated: download it directly with the command above, without an access request or manual approval. It is shared with the Local Inference Lab community and anyone else who wants the native-format derivative. The unchanged MIT License governs use and redistribution.
