---
license: other
license_name: lil-license-1.0
license_link: LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: vllm
pipeline_tag: text-generation
tags:
- abliterated
- uncensored
- mxfp4
- csf
- deepseek
- ds41-flash
- reasoning
- local-inference-lab
extra_gated_heading: "Request access (Local Inference Lab community)"
extra_gated_prompt: "This repository contains a derivative of DeepSeek-V4.1-Flash-1017-lossless-CSF by Local Inference Lab, Inc. By requesting access you accept the Local Inference Lab License, Version 1.0 (LICENSE) for the CSF container and the MIT License (LICENSES/MIT.txt) for the DeepSeek weights. In particular you agree not to re-upload, mirror or redistribute these files (LICENSE, Section 2)."
extra_gated_fields:
  Local Inference Lab Discord handle: text
  I will not re-upload or mirror this repository: checkbox
---

# DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF

This model is based on DeepSeek-V4.1-Flash-1017-lossless-CSF by Local Inference Lab, Inc., a non-profit organization, available at [https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF). DeepSeek-V4.1-Flash-1017-lossless-CSF is licensed under the Local Inference Lab License, Version 1.0.

`deepseek-ai/DeepSeek-V4.1-Flash-1017` with its refusal direction inverted in 132 dense tensors, delivered inside Local Inference Lab's lossless FP4-CSF container so it loads and runs on the LiL vLLM stack at full speed. Nothing was fine-tuned or retrained: two tensor families in layers 4 to 36 were rewritten by a norm-preserving abliteration operator, and every other tensor byte is identical to the LiL parent checkpoint. It is the checkpoint one 4x RTX PRO 6000 box has served in production since 2026-10-20, and every number on this page points at a receipt file.

## At a glance

| | |
|---|---|
| Base weights | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e1088993b9b82de08f59de5be5a34e42170` (released 2026-10-17, MIT) |
| Direct parent | `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF` (Local Inference Lab License 1.0) |
| Architecture | 40-layer MoE, hidden 5120, 384 routed experts (6 active) + 1 shared expert, 3 MTP draft layers, 1,048,576-token context (`config.json`) |
| What changed | `attn.wo_b` and `ffn.shared_experts.w2` weights and block scales in layers 4 to 36: 132 tensors, 1.65 GiB. 172 ranges were written in total (40 of them verbatim), 2.15 GiB |
| Format | LiL FP4-CSF container (`lil-mxfp4-csf-checkpoint/1`): FP8 E4M3 dense weights with UE8M0 32x32 block scales, FP4 routed experts, routed-expert scales losslessly compressed |
| Size | 48 shards, 495.6 GB (461.6 GiB) of tensors |
| Runtime | LiL vLLM image with `CHECKPOINT=csf`. Stock vLLM, Transformers, SGLang and llama.cpp cannot read the container |
| Measured on | 4x NVIDIA RTX PRO 6000 Blackwell (96 GB each), TP4 |
| License | MIT (DeepSeek weights) + Local Inference Lab License 1.0 (CSF container). See [License](#license) |
| Access | Gated, manual approval, Local Inference Lab community |

Companion documents in this repo: [EXPLAINER.md](EXPLAINER.md) (the full technical derivation), [EVALUATION.md](EVALUATION.md) (every receipt in one place), [SERVING.md](SERVING.md) (runtime, config profiles, rollback), [CITATION.cff](CITATION.cff).

## Why this exists

The 1017 refresh is a better model than its predecessor, and it also came back more guarded. On the lab's six-prompt sanity probe the stock checkpoint refuses two of six (how to pick a lock, how to hotwire an old car) while answering the rest (`1017-raw-sanity.json`). On the 100-prompt judged battery that was run on the previous checkpoint, the unmodified base produced 17 true refusals and 73 hedged answers out of 100 (`eval_baseline.json`).

The approach here is the narrow one: find the direction in hidden space that carries refusal, edit the tensors where it expresses, leave everything else alone. No fine-tuning, no changes to attention, routing or the routed experts. The result answers instead of stonewalling. It does not become an unhedged oracle, and the numbers below say exactly how much it moved.

## What changed in the weights

The method is grimjim's norm-preserving biprojected abliteration, applied per layer at scale alpha = 3.0 to layers 4 to 36, using refusal directions extracted from the original DeepSeek-V4.1-Flash (`refusal_directions.pt`: one fp32 vector of 5120 dims per layer, 40 layers).

In plain language: each target weight matrix is split into its row norms and row directions. The refusal direction's component is projected out of every column of the direction part, but with alpha = 3 the component is not merely removed, it is flipped to minus twice its original size. Rows are then re-normalised and the original row norms restored, so each row keeps its length and only rotates. That overshoot is what turns hard refusals into answers, and the norm restoration is why the model's scale statistics stay put.

Two tensor families are edited, the dense sublayers every token passes through: the attention output projection `attn.wo_b` ([5120, 8192]) and the shared-expert down projection `ffn.shared_experts.w2` ([5120, 2304]), each with its 32x32 UE8M0 block-scale tensor. Routed experts are sparse and token-conditional and are untouched.

Counting over the whole checkpoint: 172 tensors match the selection pattern (43 blocks: layers 0 to 39 plus 3 MTP blocks, 2 modules each, weight + scale). 132 of them change (layers 4 to 36). The other 40 (layers 0 to 3, 37 to 39, and the MTP blocks) fall outside the edit range and come back byte-identical.

Because the weights are stored as FP8 E4M3 values with UE8M0 block scales, every edited tensor goes through dequantize, edit in FP32, requantize. That round trip is lossy by design: about 1.9 % relative error per tensor against the FP32 ideal, the same penalty the original release paid.

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

Script: `abliterate-1017.py` (`--validate` replays the operator on the previous checkpoint pair, `--apply` writes the 1017 edit in place on a reflink copy).
</details>

## How it was built and proven

The build is a ladder; each rung is a receipt that lives next to the checkpoint on the build machine and is reproduced in [EVALUATION.md](EVALUATION.md).

1. **Validate the operator before touching 1017.** The re-implemented operator was replayed on the previous checkpoint pair (old pristine in, compared against the old shipped abliterated bytes) over all 66 edited modules. Result (`abliterate-1017-validate.json`): 96.69 % of weight bytes and 98.78 % of scale bytes reproduced exactly (worst module 93.77 % / 95.79 %), 1.01 % mean relative error in dequantized space against the old release (max 1.08 %), 1.85 % against the FP32 ideal (max 1.93 %). The residual is the size of the requantization rounding path, so the operator was accepted.
2. **Re-abliterate the raw 1017 checkpoint.** 33 layers x 2 modules were dequantized, edited, requantized and written back in place on a btrfs reflink copy of the raw download (`build-1017-abliterated.json`): mean relative error against the FP32 ideal 1.85 %, max 1.93 %, identical to the validation-time penalty. Served raw it scored Estonia knowledge 12/12 and 6/6 refusal probes unlocked, against 12/12 and 4/6 for stock 1017 (`1017-abl-raw-sanity.json`, `1017-raw-sanity.json`).
3. **Transplant into the LiL CSF container without requantizing anything.** CSF compresses only the routed-expert scales; the dense tensors the edit touches are stored uncompressed under the same names in the same shard files as the raw checkpoint, at different offsets. The build (`build-1017-abliterated-csf.py`) resolves every one of the 172 ranges independently in the CSF, the raw and the abliterated layouts, then overwrites the ranges on a reflink copy of the upstream CSF. Guards, all fatal: dtype/shape/length must agree across the three layouts; the CSF bytes must equal the raw bytes before patching (the container really holds the weights the edit was computed against); the abliterated checkpoint must differ from raw somewhere; every shard size must be unchanged after patching; every range is re-read and must equal the abliterated bytes. Receipt (`build-1017-abliterated-csf.json`, log `EXIT=0 2026-10-20T10:55:22+02:00`): 172 ranges, 2.154 GiB written across 43 shards, 40 ranges byte-identical as expected. 33 shards (`model-00007` to `model-00039`) changed content; the other 10 written shards hash identically to upstream.
4. **Record provenance.** Touched shards were re-hashed; `manifest.json` keeps each touched shard's upstream hash as `upstream_target_sha256` next to the new `target_sha256` and carries a `derivative` record (method, scale, target range, the 172 tensor names, the touched shards); `lil-manifest.json` and `SHA256SUMS` were updated; the upstream reconstruction proof was renamed `verification.upstream-pristine.json` because it no longer certifies this tree.
5. **Gate the production container.** Served with `CHECKPOINT=csf`: Estonia knowledge 12/12, 6/6 probes unlocked (`csf1017-sanity.json`); a 134,187-token long-context quiz at concurrency 30, 30/30 correct, no run hit the token limit (`csf1017-estonia.json`); the 1017 tokenizer/encoding suite 53/55 in-container (the 2 failures are missing example-fixture paths) and live tool calls working.
6. **Quality receipts for the edit itself.** The same edit lineage measured against its base on the previous checkpoint (sidecar evals): MMLU 88.95 % vs 88.95 %, KL divergence 0.1416 against a 0.1307 nondeterminism control, GSM8K 16/20 vs 14/20. Details and scope in the next two sections.

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

Two smaller checks point the same way. A six-prompt directional probe at three reasoning-effort tiers on the deployment stack (`eval_deployment.json`): refusals stay removed at every tier (1, 1, 0 of 6 at effort off / high / max) while the hedge rate rises with effort (3, 5, 6 of 6). And the stock 1017 vs this build on the lab's six sanity prompts: 4/6 vs 6/6 unlocked.

Historical quality measurements on the previous checkpoint: full MMLU letter-logprob scoring at T=0 gave 12,490 correct for base and 12,491 for the abliterated build out of 14,042 (88.95% both at reported precision). KL was 0.1416 against a 0.1307 identity-control value, and GSM8K was 14/20 versus 16/20. These used reference runtimes, not this production stack. They remain historical evidence, not measurements of the 1017 pair. The direct 1017 comparison below uses a different protocol; do not compare its absolute MMLU score against 88.95%.

## Direct 1017 comparison: what we now know

The raw 1017-CSF checkpoint and this derivative have now been tested on the same LiL vLLM image and matching inference settings. Full results, protocol, uncertainty and downloadable per-item scores are in [the paired evaluation report](EVALUATION-1017.md).

| Measure | Raw 1017 | This derivative |
|---|---:|---:|
| MMLU, 14,042 questions | 11,549 correct (82.246%) | 11,561 correct (82.332%) |
| GSM8K, 20-question sample | 19 correct; 1 truncated | 18 correct; 2 truncated |
| Refusals on the new 100-prompt set, blinded labels | 88 | 3 |

**No aggregate MMLU loss was resolved by this run.** The difference is +0.085 percentage points, with a paired item-bootstrap 95% interval of −0.292 to +0.463 points. That is not proof of equal intelligence or a gain: 357 answers changed from correct to wrong and 369 changed the other way. The interval does not include repeated-serving variability.

The GSM8K difference is token-limit-sensitive: every failed answer reached the 400-token cap. The refusal labels describe visible engagement, not answer quality or completion: 91 of the derivative's 100 responses reached the 300-token cap. A second reviewer agreed on 92/100 labels in the packet it reviewed.

A top-logprob distribution diagnostic measured coarsened KL of 0.158601 nats for raw versus derivative and 0.137411 for a raw repeat control. This is a lower-bound calculation on 200 prompts, not full-vocabulary KL and not a percentage of intelligence lost. The report spells out its coverage and numerical exclusions.


## Performance

All decode numbers are aggregate tokens per second on 4x RTX PRO 6000, TP4, LiL vLLM, this checkpoint, 2026-10-20. Matched probes are 30-second cells at a fixed concurrency and context (`csfd3/5/7-decode-*.json`, `middle-decode-*.json`); lil-bench rows are the standard profile with upload ids.

**The edit costs nothing measurable, and the container is slightly faster than raw.** Abliterated CSF vs raw 1017 at the same draft setting (d3): C8 846.8 vs 824.8 (+2.7 %), C32 1521.4 vs 1518.9 (parity), while the KV pool grows from 2.60 M to 4.44 M tokens.

**Fewer draft tokens win on 1017.** DSpark speculative decoding, matched probe at context 0:

| DSpark draft tokens | C1 | C4 | C8 | C32 |
|---|---:|---:|---:|---:|
| d3 (production) | 247.7 | 544.5 | **846.8** | **1521.4** |
| d5 | 247.8 | 551.7 | 822.2 | 1495.7 |
| d7 | 238.8 | 516.6 | 790.3 | 1434.3 |

d7 is 6.7 % slower at C8 and 5.7 % slower at C32 than d3; verification overhead outweighs the draft gain on this checkpoint. (lil-bench's C16 cell has d5 3 % ahead of d3; the d3 choice rests on C8 and C32.)

**KV capacity is a configuration choice.** The initial configuration spent VRAM on an uncapped CUDA-graph capture ladder, 16k-token activation buffers and profiler headroom. Capping graph capture at 128 and shrinking the chunk size unlocks cache; too small a chunk halves prefill. lil-bench, d3, context 0 unless noted:

| Config | KV pool (tokens) | C8 | C16 | Prefill 32k (tok/s) | Run |
|---|---:|---:|---:|---:|---|
| mnb 16384, default graph ladder | 4,440,423 | 789.1 | 995.1 | 19,107 | `bygepW-DSgCC` |
| mnb 4096, graph cap 128 | 13,413,197 | 788.8 | 1,058.3 | 10,249 | `_4ME5KHnwtKh` |
| **mnb 8192, graph cap 128 (production)** | **9,248,520** | 790.5 | 1,008.1 | 18,603 | `daygWYTtkShy` |

The production row keeps decode and prefill within 3 % of the fastest configuration while holding 9.25 M tokens of cache, enough for 8.8 concurrent 1 M-token requests. Matched probe on the production configuration: C8 838.0, C32 1579.2, the highest C32 this box has measured under this protocol. Keeping Engram scales resident in RAM matters: turning it off cost 5 % at C8 and 8 to 9 % of prefill for no KV gain (`SqyHbJvPojGO`).

Context for the absolute numbers: the 1017 checkpoint decodes about 6.5 % slower than the previous DeepSeek-V4.1-Flash on this stack (cross-checked against an independent rig on the LiL leaderboard, 918.7 to 858.6 at C8). That is the model, not this derivative.

## How to run it

The CSF container loads **only** through the Local Inference Lab vLLM image with `CHECKPOINT=csf`, which selects `--load-format mxfp4_csf --quantization mxfp4_csf`. Without that variable the launcher loads the path through the generic instanttensor loader, the compressed scales are misread, the server still answers plausibly, and DSpark acceptance collapses to about 1.2 accepted tokens per step. Benchmarks from that state are invalid. Stock vLLM, Transformers, SGLang and llama.cpp cannot open the container at all; for those stacks use the upstream `deepseek-ai/DeepSeek-V4.1-Flash-1017` and apply the edit yourself with the published operator.

Mount the checkpoint root unmodified (`manifest.json`, `build-contract.json`, `metadata/`, `tensors/`), offline, and let the launcher build its serving directory. The production environment, verbatim from the compose file (4x 96 GB, TP4):

```yaml
image: ghcr.io/local-inference-lab/vllm@sha256:fd74019fc217a541060093d18427c36330df6ac5700651b55593dafdbb936920
command: [serve-ds41-flash.sh, "--", --max-cudagraph-capture-size, "128"]
environment:
  CHECKPOINT: csf
  MODEL: /models/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF
  SERVED_MODEL_NAME: DeepSeek-V4.1-Flash
  TP: "4"
  OMP_NUM_THREADS: "1"
  MAX_NUM_BATCHED_TOKENS: "8192"
  GPU_MEMORY_UTILIZATION: "0.95"
  DSPARK_TOKENS: "3"
  DSPARK_ADAPTIVE_VERIFICATION: "true"
  DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE: "0.5"
  ENGRAM_TABLE_MEMORY: disk
  ENGRAM_DISK_RESIDENT_SCALES: "true"
  CUDA_VISIBLE_DEVICES: "0,1,3,2"
  VLLM_ENABLE_PCIE_ALLREDUCE: "1"
  VLLM_PCIE_ALLREDUCE_BACKEND: b12x
  VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: "96KB"
  VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE: "96KB"
  VLLM_PCIE_DMA_MIN_BYTES: "6MB"
  B12X_MHC_PDL: "1"
  B12X_PCIE_ONESHOT_PDL: "1"
  VLLM_MXFP8_LM_HEAD: "1"
  HF_HUB_OFFLINE: "1"
# plus: network_mode host, ipc private, shm_size 128g, gpus all,
#       security_opt seccomp=unconfined, ulimits memlock -1 (disk Engram uses io_uring)
```

Caveats worth knowing before the first boot: `GPU_MEMORY_UTILIZATION=1.0` does not boot (strict free-VRAM check; the CUDA context holds about 1.4 GiB per GPU); the device order `0,1,3,2` is this box's PCIe ring and only matters if your topology has one slow hop; the image's vendored DeepSeek encoding module predates 1017, so production bind-mounts the checkpoint's own `encoding.py` over `vllm/tokenizers/deepseek_v41_encoding.py` (a strict API superset) for tool calling to parse correctly; `/v1/models` answers before CUDA-graph capture finishes, so gate readiness on a real completion. The full profile table, quirks and rollback procedure are in [SERVING.md](SERVING.md).

## Lineage and credits

- **Base weights:** DeepSeek-AI, `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, MIT.
- **Container and runtime:** Local Inference Lab, Inc., `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF` (lossless FP4-CSF build) and the LiL vLLM image that serves it.
- **Method:** grimjim, [Norm-Preserving Biprojected Abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration).
- **Refusal directions and the original s3.0 recipe:** `distributedcognition/DeepSeek-V4.1-Flash-abliterated` (norm-preserving biprojection, scale 3.0, layers 4 to 36), built on the previous DeepSeek-V4.1-Flash.
- **This derivative:** operator re-implementation, validation, 1017 re-abliteration, transplant, gates and benchmarks by jcartu on a 4x RTX PRO 6000 build machine. Technical review of the derivation by GPT-6 Astra and Claude Fable (see [EXPLAINER.md](EXPLAINER.md), appendices).

The `derivative` record in `manifest.json`, the per-shard hashes in `SHA256SUMS` and `verification.upstream-pristine.json` are in the repo so that the provenance can be checked rather than trusted.

## License

Three layers, all of which apply:

1. **DeepSeek weights:** MIT, from `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`LICENSES/MIT.txt`, `metadata/LICENSE`). The unmodified upstream files (`config.json`, `metadata/`) stay under MIT.
2. **CSF container:** the Local Inference Lab License, Version 1.0 (`LICENSE`, `NOTICE`, `REUSE.toml`; SPDX `LicenseRef-LIL-1.0`), from Local Inference Lab, Inc., a non-profit organization, 311 West Main Street, Grayson, Kentucky 41143, the creator of the parent work DeepSeek-V4.1-Flash-1017-lossless-CSF at [https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF). The license reproduces the Apache 2.0 terms and adds restrictions: this repository must not be re-uploaded, mirrored or redistributed (LICENSE Section 2), every page about it must begin with the attribution notice at the top of this card (Section 3), and the license, the `NOTICE` file and its identifying marks must stay with the files (Section 3.6). The encoded tensors, `manifest.json`, `build-contract.json` and `receipts/` are `LicenseRef-LIL-1.0 AND MIT` (`REUSE.toml`). Link to this repo; do not copy it.
3. **The edit and these documents:** the 132 rewritten tensors, the `derivative` records, the renamed verification file and the documentation add no restrictions beyond the two licenses above. Modified files are listed in `manifest.json` (`derivative.touched_shards`) and reflected in `SHA256SUMS`.

This derivative is published with permission from a Local Inference Lab officer, as LICENSE Section 2.4 requires. Festr. (Local Inference Lab) approved the gated re-upload in writing in the LiL `#ds41f` Discord channel on 2026-10-20 at 16:47 UTC (message id 1562145160425504789). The permission covers this repository at this location only; it does not extend to further mirrors or re-uploads by anyone else.

## Access

Gated with manual approval for the Local Inference Lab community. Request access with your LiL Discord handle. Approval means you have accepted both licenses above, including the no-reupload clause.
