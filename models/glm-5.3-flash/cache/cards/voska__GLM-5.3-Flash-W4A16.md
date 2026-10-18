---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
language: [en, zh]
tags: [glm, moe, compressed-tensors, int4, w4a16, sglang, quantized]
---

# GLM-5.3-Flash-W4A16

INT4 **weight-only** (W4A16, group-128, symmetric) quantization of
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), produced for
self-hosted serving on **SGLang / RTX PRO 6000 Blackwell (SM120)**.

Vision and tool calling are preserved. **Speculative decoding works with this checkpoint's own built-in
MTP head** — see [Speculative decoding](#speculative-decoding-use-the-built-in-mtp-head). Two earlier
revisions of this card got that wrong, the second one confidently; the correction and its cause are in
that section.

An **abliterated (uncensored)** sibling built by the same method, with the same serving config, is at
[`voska/GLM-5.3-Flash-Uncensored-W4A16`](https://huggingface.co/voska/GLM-5.3-Flash-Uncensored-W4A16).

**† on this card means: measured on that sibling, not re-measured on these weights.** The two builds share
architecture, quantizer, `ignore` list and serving config, and are the same 190.8 GB on disk — but a † figure
is not a measurement of this checkpoint, and this card will not pretend otherwise.

## ⚠️ Read this before you download 178 GiB

**Updated 2026-09-17: GLM-5.3-Flash support is merged into SGLang `main`, but no *released tag*
carries it yet.** [sgl-project/sglang#36507](https://github.com/sgl-project/sglang/pull/36507) merged on
2026-09-06. The PR branch (`xinyuan/glm-5.3-flash-support`) was **deleted at merge**, so this card's
earlier instruction — "build from #36507" — no longer names anything you can check out.

Build instead from a **post-merge per-commit nightly**: lmsys publishes these as
`lmsysorg/sglang:nightly-dev-cu13-<date>-<sha>`, which hands you a consistent sgl_kernel / FlashInfer /
python tree for free. You still need the three SM120 patches below on top — they are not upstream.
There is still no stock *released* image that serves this checkpoint.

We would rather open with that than have you find out at the end of the download.

## ⚠️ Three flags are mandatory

```
--disable-shared-experts-fusion      # REQUIRED — without it, output is gibberish
--kv-cache-dtype bfloat16            # never FP4 KV on SM120
--chunked-prefill-size 4096          # 1024 is ~30% slower; see Measurements
```

Plus: **enable the checkpoint's own NEXTN/MTP head** — it is the largest single throughput win available
on these weights, and this card was wrong about it until 2026-09-17. See
[Speculative decoding](#speculative-decoding-use-the-built-in-mtp-head).

**1. `--disable-shared-experts-fusion`.** Without it this model emits fluent gibberish and never
stops. That is not a defect in these weights — it is an SGLang bug that bites any *quantized*
build of this architecture. `determine_num_fused_shared_experts()` sizes the MoE at
`n_routed_experts + 1` = 289 slots, then a separate wrapper gate, evaluated ~2 minutes later
during weight loading, decides the config does *not* support fusion and never populates slot 288.
SGLang's own source comment describes it: *"a divergence drops the shared-expert weights and runs
the fused slot uninitialized."* Every token then routes through an uninitialized expert. The same
divergence happens on the FP8 release and is harmless there — it only turns fatal once the expert
slots are INT4-packed. Verified by flipping only that flag on identical weights.

**2. Re-pull `config.json` if you downloaded this checkpoint before 2026-09-17.** Its
`quantization_config.ignore` list declared the NextN layer only in the `re:` dialect, which SGLang's
NextN gate does not read — so enabling the MTP head silently built a W4A16 draft head against bf16
weights and cost ~33% single-stream instead of roughly doubling it. One literal string fixes it. The
mechanism, the measurements and the two wrong claims this card shipped are in the next section.
*This is the second correction to this line. The first one — "use a standalone DFlash2 draft" — was also
wrong, on correctness grounds; see the end of the next section.*

## Speculative decoding: use the built-in MTP head

> **Correction (2026-09-17). The previous revision of this card was wrong, and the fault was in this
> repo's own metadata, not in the model.**
>
> Earlier revisions told you never to enable this checkpoint's built-in NEXTN/MTP head, and offered a
> generalisable rule to go with it: *an MTP head's acceptance does not survive quantizing the target.*
> **Both were wrong.** The head works. What did not work was how this repo declared it to SGLang.

`quantization_config.ignore` listed layer 45 only in the `compressed-tensors` regex dialect,
`re:.*layers\.45\..*`. SGLang's `glm5_next_nextn.py::_resolve_nextn_quant_config` decides whether to
build the NextN block unquantized with an **exact string membership test**, not a regex match:

```python
nextn_layer_pattern = f"model.layers.{config.num_hidden_layers}.*"   # "model.layers.45.*"
if nextn_layer_pattern in ignored:              # list membership on raw strings
    return None                                 # -> BF16 draft modules
return super()._resolve_nextn_quant_config(config, quant_config)   # else: inherits W4A16
```

The regex entry never satisfied it, so the MTP layer's **routed experts** were built W4A16 against bf16
weights. Drafts were essentially never accepted, and enabling speculation cost about a third of the decode
rate. `self_attn`, `shared_experts`, `mlp.gate`, `eh_proj` and `shared_head` survived by accident — their
regexes still match after the loader rewrites `model.layers.45` to `model.decoder`. The experts did not.

**Nothing errors and nothing warns on that path.** The only tell is the NextN module loading at roughly 45%
of its own weight size — 1.56 GB/rank against the 4.08 GB/rank it should be at TP=4. That number sat in our
journal for twelve days.

**Fixed in this repo as of 2026-09-17**: `config.json` carries the literal string alongside the regex
(11 `ignore` entries → 12). The added entry is **inert for the target model** — `should_ignore_layer`
returns an identical verdict on all 14 probed target-model prefixes with and without it; it only trips
SGLang's NextN gate. **If you pulled this checkpoint before 2026-09-17, re-pull `config.json`.**

Measured on 4x RTX PRO 6000 Blackwell Max-Q (SM120), TP=4, SGLang built from post-merge `main`, with the
speculative flags byte-identical across rows 2 and 3 and **only the `ignore` list changing between them**:

| configuration | accept length | short single-stream |
|---|---|---|
| autoregressive (no `--speculative-*` flags) | n/a | 117.4 tok/s † |
| MTP head, regex-only `ignore` (the defect) | 1.0 of 4 † | 81.0 tok/s † |
| **MTP head, literal `ignore` present** | **2.83 of 4 lifetime †** | **227.2 tok/s †** |

Launch flags:

```
--speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1
```

`--speculative-num-draft-tokens` is **stated, not left to inference**. A standalone-draft report on
identical hardware ([sglang#33800](https://github.com/sgl-project/sglang/issues/33800)) traces a
corruption class specifically to the flag-omitted native-block-size path, so do not omit it.

Confirm the fix is live by looking for this line at startup, **on every rank**:

```
GLM5 NextN layer model.layers.45.* is checkpoint-declared unquantized; using BF16 draft modules
```

**Read acceptance off `GET /server_info` → `internal_states[0].avg_spec_accept_length`** (lifetime since
start), or off `meta_info.spec_accept_length` on `/generate` (per request; the OpenAI
`/v1/chat/completions` route does not return it). **Do not read the `sglang:spec_accept_length` Prometheus
gauge** — it is reset every decode-log interval, so a scrape reports roughly the last second, not an
average. We published two wrong acceptance figures off that gauge before noticing; its observed range on
an *idle* box was 2.025–3.775 within 36 seconds.

### We no longer recommend the DFlash2 standalone draft — on correctness, not speed

A 24 h controlled A/B on production agentic traffic measured DFlash2 causing a **6.9x increase in runaway
generations**: requests that never emit `</think>`, produce no tool call, and burn to `max_tokens`.
**1.12 per 1,000 with speculation off against 8.91 per 1,000 with DFlash2 (P = 8.2e-10)**, and 0.04 against
6.69 per 1,000 on the sharpened `finish_reason='length'` + empty-content predicate. It reproduced on two
independent API keys and on three different checkpoints, so the weights are ruled out; the only constant
was the draft. We parked it permanently on 2026-08-31.

Its throughput numbers were real and we do not dispute them. We do not run it, and we would not run it on
agentic traffic. The MTP head above is faster anyway and ships inside these weights.

**Every figure in the table above is †** — measured on this checkpoint's abliterated sibling
[`voska/GLM-5.3-Flash-Uncensored-W4A16`](https://huggingface.co/voska/GLM-5.3-Flash-Uncensored-W4A16),
which is the same quantization recipe, the same `ignore` list and the same 190.8 GB footprint, but is not
these weights. **The defect and the fix are identical in both repos and were verified in both**; only the
throughput numbers are carried over. We did not restart a live box to re-measure a settled answer.

### The original MTP measurements, kept as the record of a wrong conclusion

> **Scope: this subsection is the dated record of how this card got it wrong, not current advice.**
> "spec ON" below means the built-in MTP head **with the broken regex-only `ignore` list** — i.e. it
> measures the metadata defect, not the MTP head. Do not quote any of it as a fact about this model.

NEXTN acceptance on this checkpoint measured **1.01–1.04** (vs 2.5–2.8 on the FP8 release). At the time we
attributed that to quantization of the target and wrote it up as a general rule. It was the draft head
itself being built W4A16, for the reason above.

Measured warm on an idle box, identical weights, only the speculative flags differ:

| metric | spec ON (defective) | **spec OFF** | delta |
|---|---|---|---|
| single-stream decode | 65.14 tok/s | **102.67** | **+57.6%** |
| aggregate @ concurrency 5 | 226.09 | 314.60 | +39.1% |
| aggregate @ concurrency 10 | 350.00 | **540.38** | **+54.4%** |

Single-stream reps: 102.37 / 102.64 / 102.68 / 102.69 / 102.67 (n=5, spread <0.35%).

**This was always quality-neutral.** Speculative decoding with rejection sampling is distribution-
preserving; the defect cost throughput, not output quality.

**One datapoint from that era survives and now reads as corroboration.** The alternative recorded at the
time was to quantize layer 45 *as well* (`43 × 288 × 3 = 37,152` packed tensors instead of 36,288), which
restored acceptance to **2.3–2.4** "by matching draft precision to the target". That is the same defect
seen from the other side: acceptance recovers whenever draft and target agree. Making the draft bf16 — what
the checkpoint always intended — gets 2.83 rather than 2.4, with no extra quantization work.

## Why weight-only (and not NVFP4/W4A4)

The published KLD panel for this model family shows the divergence cost of 4-bit lives in the
**activations**, not the weights:

| variant | mean KLD vs BF16 | ratio to official FP8 |
|---|---|---|
| official FP8 release | 0.0206 | 1.00× |
| 4-bit **weight-only** | 0.0246 | 1.19× |
| NVFP4 **W4A4** | 0.0605 | **2.9×** |

So this checkpoint quantizes weights to INT4 and leaves activations at 16-bit. It also keeps
**bf16 KV cache** — NVFP4 KV measured 1.7–1.8× *slower* on SM120 with known corruption bugs.

On SM120 specifically, weight-only INT4 via Marlin is also the fastest *working* MoE path:
W4AFP8 is Hopper-only (`cutlass_w4a8_moe_mm` fails init on SM120) and NVFP4 grouped-GEMM MoE
kernels fail all tactics.

## What is and isn't quantized

Quantized: routed-expert `gate_proj` / `up_proj` / `down_proj` for **layers 3–44 only**
(42 × 288 × 3 = **36,288** packed tensors).

Left at BF16 (in the `ignore` list) — each for a reason:

| module | why |
|---|---|
| `self_attn.*` | 34 KDA linear-attention + 11 DSA layers + the sparse indexer; recurrent state is precision-sensitive |
| `visual.*` | vision tower — this model is natively multimodal |
| `layers.45.*`, `eh_proj`, `shared_head` | MTP / NEXTN draft head — declared in **both** the `re:` and the literal `model.layers.45.*` dialect; see [Speculative decoding](#speculative-decoding-use-the-built-in-mtp-head) for why both are needed |
| `*_hc.*` | mHC (Manifold-Constrained Hyper-Connections) |
| `shared_experts.*` | always-active, higher sensitivity than routed experts |
| `mlp.gate` | MoE router — tiny and precision-critical |
| `layers.0/1/2.mlp.*` | dense layers (`first_k_dense_replace = 3`) |
| `embed_tokens`, `lm_head` | standard |

**Note on efficiency:** the 1,050 non-expert tensors that were blockwise-FP8 in the source are
stored here as BF16, which doubles their bytes on the always-active path. A parameter census puts
attention at ~58% of per-token decode bytes and the routed experts at only ~20%, so re-storing
those as FP8 is the obvious next gain (~13% fewer bytes/token). Not done in this checkpoint:
SGLang's FusedMoE path reads only the *top-level* compressed-tensors format and rejects
`mixed-precision`, so a two-group checkpoint fails to load
([sglang#16276](https://github.com/sgl-project/sglang/issues/16276) — fixed for Linear, not MoE).

## Method

Streaming tensor-level transform directly from the **FP8** release (no BF16 checkpoint was
materialised — dequantized BF16 is 642 GB and does not fit in 384 GB VRAM + 125 GB RAM):

1. dequantize blockwise FP8 (128×128 `weight_scale_inv`) → fp32
2. per-group symmetric min/max scales via `compressed_tensors.calculate_qparams`, computed against
   the **BF16-rounded** scale that is actually stored (quantizing against the fp32 value and then
   storing BF16 leaves the checkpoint self-inconsistent at inference)
3. symmetric INT4 group-128, packed 8-per-int32 via `compressed_tensors`' own packer
4. all other FP8 tensors dequantized to BF16; `weight_scale_inv` removed

Mean sampled relative L2 error vs the FP8 source: **11.10%**.

**This is a data-free quantization.** It is *not* GPTQ or AWQ — no calibration set, no
activation-aware scaling. A calibrated build should beat it on quality; this one is reproducible
in ~7 minutes and needs no data.

An MSE-optimal clip search was tried and **discarded**: it reported ~9% lower weight L2 while
being no better in practice, and optimising weight-space error is the wrong objective at 4 bits.

## Serving

### Hardware this was measured on

4× RTX PRO 6000 Blackwell Max-Q, 96 GB each (**SM120**, no NVLink, PCIe Gen5 P2P), power-limited to
260 W; Threadripper PRO 9975WX, 128 GB DDR5; driver `580.159.04-open`. Kernel cmdline
`iommu=off amd_iommu=off` plus `options nvidia_uvm uvm_disable_hmm=1` are both required for NCCL P2P
on this board.

Numbers here will not transfer to datacenter Blackwell (SM100). SM120 has no tcgen05/TMEM and caps
dynamic shared memory near 99 KB, which is what forces patch 3 below.

### Versions

`sglang 0.0.0.dev1+g30705c004` (post-merge `main` 30705c004ca4 + patches) · `torch 2.13.0+cu130` ·
CUDA 13.0 · `transformers 5.12.1` · `compressed-tensors 0.18.0` · `flashinfer 0.6.18` · driver
`580.159.04-open`

A launch line with no version pin rots inside a month — SGLang flag names churn.

### Three SM120 patches, all still required

1. **`SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`** (env). Hyper-connection prenorm calls DeepGEMM
   unconditionally and SM120 has no TMEM/tcgen05 → `NameError: deep_gemm`.
2. **`--dsa-prefill-backend tilelang --dsa-decode-backend tilelang`**. The default decode path routes
   to trtllm-gen FMHA, which reports *"Unsupported architecture"* on SM120.
3. **`num_stages=1` in the TileLang sparse-MLA schedule** — a source patch, not a flag. Upstream uses
   `num_stages=2`, which asks ~148 KB of dynamic shared memory; **SM120 caps it near 99 KB**. Not
   upstream yet.

### KV dtype: keep bf16

`--kv-cache-dtype fp8_e4m3` produces garbled output, or a single token and then a stop. The cause is
a FlashInfer SM120 bug — the backend is missing FP8 dequantization scales, so the ragged+paged split
path reads cached KV back without undoing the scale division. Independently reported on non-GLM
models on the same hardware ([sglang#19603](https://github.com/sgl-project/sglang/issues/19603)).
Expect it on any SM120 model on this backend, not just this one.

### The launch line, exactly as run

```bash
docker run --rm --name glm53 --gpus all --network host --ipc=host \
  -v /path/to/GLM-5.3-Flash-W4A16:/model:ro \
  -v /srv/triton-cache:/triton-cache \
  -e TRITON_CACHE_DIR=/triton-cache \
  -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
  -e SGLANG_ENABLE_SPEC_V2=True \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  YOUR_SGLANG_IMAGE_BUILT_FROM_A_POST_MERGE_NIGHTLY_PLUS_THE_3_PATCHES \
  python3 -m sglang.launch_server \
    --model-path /model --served-model-name glm-5.3-flash \
    --tp 4 --context-length 1048576 \
    --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --kv-cache-dtype bfloat16 \
    --speculative-algorithm NEXTN \
      --speculative-num-steps 3 \
      --speculative-num-draft-tokens 4 \
      --speculative-eagle-topk 1 \
    --mem-fraction-static 0.90 \
    --chunked-prefill-size 4096 \
    --max-running-requests 12 \
    --schedule-policy lpm \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-track-interval 2048 \
    --cuda-graph-max-bs 16 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --enable-hierarchical-cache \
    --hicache-ratio 0.5 \
    --hicache-write-policy write_through \
    --enable-metrics --trust-remote-code \
    --host 0.0.0.0 --port 8100 \
    --disable-shared-experts-fusion \
    --prefill-decode-interval 16
```

Drop the four `--speculative-*` flags for the autoregressive configuration; nothing else changes.

**`--prefill-decode-interval 16` is not cosmetic.** Without it, a live decode stream collapses to
**1.6 tok/s** for the whole duration of anyone else's cold prefill — measured here at 65x, against a 4.6x
figure reported on other stacks, because a 100K+ prompt at `--chunked-prefill-size 4096` is ~29 consecutive
extend batches and the scheduler re-arms on every one of them. At 16 the same victim holds 23.1 tok/s. It
costs ~9% on a synchronized 12-way burst and inflates cold TTFT by ~27%.

**Parsers are not optional.** Without `--reasoning-parser glm45` the chain-of-thought lands in
`content` with a stray `</think>`; without `--tool-call-parser glm47` every agentic client gets a 400
while plain chat looks perfectly healthy. Both failure modes shipped to production here before they
were caught. `/health` says nothing about parsers.

**`--chunked-prefill-size 4096` is a ceiling, not a tuning knob.** 8192 crashed the scheduler on a
305K-token cold prefill: the DeepGEMM MQA-logits buffer is `chunk × max_seqlen_k × ~1.14 bytes`, so
at 8192 against a 1.8M-token pool it wants 4.07 GB — more than this box ever has free. Do not lower
it either; see the chunk-size table under Measurements.

**`--mamba-track-interval 2048` and `--max-running-requests 12` are not arbitrary.** A
*"Can not alloc mamba cache"* failure 502'd a live request here. The mamba/KDA slot pool is small
(**290 slots** in the config above, down from 582 without speculation — spec decode buys its intermediate
SSM and conv state out of this pool) and each running request holds several; leave headroom for the radix
tree's state snapshots. Read all three gauges, not one: `mamba_used_tokens` is live demand,
`mamba_evictable_tokens` is reclaimable radix cache, `mamba_available_tokens` is free. `available` near
zero is normal steady state — only `used` approaching the pool size is pressure.

### The unit, as deployed

systemd, secrets stripped. `MODEL`/`IMAGE`/`SPEC_ARGS`/`EXTRA_ARGS` are passed as `Environment=` to a
wrapper script that execs the `docker run` above.

```ini
[Unit]
Description=GLM-5.3-Flash (W4A16, TP=4) on SGLang with SM120 fixes
After=docker.service network-online.target
Requires=docker.service
# This box has produced a 206,804-restart loop. Rate-limit restarts.
StartLimitIntervalSec=900
StartLimitBurst=4

[Service]
Type=simple
ExecStartPre=-/usr/bin/docker rm -f glm53
ExecStart=/usr/local/bin/glm53-run.sh
ExecStop=/usr/bin/docker stop -t 90 glm53
Restart=on-failure
RestartSec=45
# 178 GiB (191 GB) across 4 GPUs + CUDA graph capture. Model load takes minutes.
TimeoutStartSec=1500
TimeoutStopSec=150
Environment=MODEL=/srv/models-cache/GLM-5.3-Flash-W4A16
Environment="SPEC_ARGS=--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-eagle-topk 1"
Environment="EXTRA_ARGS=--disable-shared-experts-fusion --prefill-decode-interval 16"

[Install]
WantedBy=multi-user.target
```

`TimeoutStartSec` matters: startup to healthy is **~5m30s** with the MTP head enabled, and **135 s of
that is target-verify CUDA-graph capture** — it looks like a hang and is not. A shorter timeout turns a
normal boot into a restart loop.

**Quote multi-word `Environment=` values.** systemd splits on whitespace into separate assignments, so an
unquoted `SPEC_ARGS=--speculative-algorithm NEXTN --speculative-num-steps 3` reaches the wrapper as
`--speculative-algorithm` alone, silently. Confirm with `systemctl show <unit> -p Environment --value`
before you restart.

### Expected health at startup

Measured with the launch line above (`--mem-fraction-static 0.90`, 1M context). If
`max_total_num_tokens` comes up materially below this, your mem-fraction or context length differs —
**the pool, not the `--context-length` flag, is what enforces the input ceiling.**

| | with MTP (NEXTN) † | autoregressive |
|---|---|---|
| weights / GPU | 41.98 GB + 4.08 GB NextN † | 41.98 GB |
| `max_total_num_tokens` (KV pool) | **2,048,896** † (KV 24.27 GB/GPU) | 1,883,328 |
| mamba/KDA slots · `max_running_requests` | 290 · 12 † | 582 · 12 |
| hard input ceiling | 1,048,576 | 1,048,576 |
| free device memory after graph capture | 7.87 GB † | — |
| startup to healthy | ~5m30s † (95 s weights, 10 s NextN, 135 s verify-graph capture) | ~2m |

At this config the sibling measured **227.2 tok/s** single-stream (n=5) and **783–820 tok/s** aggregate at
concurrency 12 (median 799, n=4, clean idle-gated reps), at an accept length of **2.83 of 4** over 7.2 days
of production. All † — see the note in [Speculative decoding](#speculative-decoding-use-the-built-in-mtp-head).

**Note the direction of the pool change: enabling the MTP head makes the KV pool BIGGER** (1,883,328 →
2,048,896, +8.8%) while shrinking the mamba/KDA pool (582 → 290 slots). Spec decode re-partitions those two
pools and the direction is not intuitive — re-read both from the startup lines after any speculative change
rather than inferring them.

**This table supersedes the `spec OFF` block further down in full — throughput *and* capacity.** That block
was measured at an earlier `--mem-fraction-static` and a 469K context, and it reports a KV pool of 2,052,800
against 43.22 GB/GPU. **The table above is the one to size a box from.** The older numbers are kept because
the comparisons around them (FP8 vs W4A16, chunk-size, long-context) are still valid *relative*
measurements, but their absolute pool and weight figures do not describe this card's launch line.

### Measurement traps

These numbers will not reproduce if you hit these:

- **Never count SSE chunks under speculative decoding.** One chunk can carry several accepted tokens.
  Read tok/s from `usage.completion_tokens`.
- **`sglang:num_running_reqs` is not a live concurrency signal.** It refreshes only on the decode-log
  tick (~1 s here) and never at request admission or completion — it read `1.0` while two requests
  concurrently held KV, and `0.0` while one held 15,104 tokens. Gate idleness on
  `sglang:kv_used_tokens`, which moves immediately and counts active requests only.
- **Aggregate throughput is bimodal by warm-up state.** The first run after an idle gap is ~30% faster
  than back-to-back runs, because the HiCache write-through host tier is still draining when
  `kv_used_tokens` already reads zero. State which one you measured.
- **Under speculative decoding, tok/s is comparable across configs on one model but not across
  models.** A tersely-answering model posts a different tok/s at an identical engine step rate. The
  invariant is chunks/s, not tokens/s.

## Measurements

4× RTX PRO 6000 Blackwell Max-Q (SM120, ~260 W cap), TP=4, SGLang, bf16 KV. Warm, idle box,
median of repeated runs.

Best-known config for each checkpoint (FP8 keeps speculation, which works there; this checkpoint
disables it, which is why the single-stream gap is larger than quantization alone would give):

> **Config stamp:** the table below was taken at an earlier `--mem-fraction-static` with a **469,504**
> context and **speculation off**. Its pool (2,052,800) and weights/GPU (43.22 GB) describe *that* config,
> not the launch line above — which gives **2,048,896** and **41.98 GB**. Treat the columns as a valid
> FP8-vs-W4A16 comparison, not as sizing numbers.

| metric | FP8 (official, spec ON) | this W4A16 (spec OFF) |
|---|---|---|
| single-stream decode | 53.73 tok/s | **102.67** |
| aggregate @ concurrency 5 | 145.25 | **314.60** |
| aggregate @ concurrency 10 | 239.07 | **540.38** |
| KV cache pool | 469,760 tok | **2,052,800** |
| weights per GPU | 77.66 GB | **43.22 GB** |
| on-disk | 306 GB | **178 GiB (191 GB)** |

For the controlled single-variable comparison (both with speculation on) this checkpoint measured
65.14 / 226.09 / 350.00 — i.e. **+21% single-stream from quantization alone**, with the rest of the
headline gain coming from disabling the *defective* draft head. **Read that comparison with the correction
above in hand**: the W4A16 "spec ON" column measures a draft head mis-built as W4A16, not what this
checkpoint does today. With the `ignore` list fixed, speculation is a large win here rather than a loss.

Cold prefill is the one axis where this checkpoint **loses**: ~48K-word cold prefill went
27.47 s → 30.32 s (**−10.4%**). INT4 buys bytes, not FLOPs, and prefill is compute-bound.

**Do not try to fix this by lowering `--chunked-prefill-size`.** An earlier revision of this card
suggested that, reasoning that a smaller chunk lowers per-expert batch (`chunk × top_k / 288`)
back under INT4's compute-roofline crossover. **Measured, and it is wrong** — same ~240K-token
prompt, same weights, only the chunk differs:

| `--chunked-prefill-size` | cold prefill |
|---|---|
| 1024 | 43.3–45.0 s |
| **4096** | **33.8 s** |

Per-chunk overhead dominates the roofline effect by a wide margin. Keep chunk at 4096 or higher.

### Long context

Serves the full **1,048,576**-token context. Verified with a real 1,028,250-token prompt.

The binding constraint is not KV — it is the DSA indexer's logits buffer, which scales as
`chunk × Σ(seqlens in batch)` and is allocated *during* prefill, so overflowing it crashes the
scheduler mid-request rather than failing at startup. Measured on 4×96 GB at chunk 4096:

| prompt tokens | indexer buffer | prefill |
|---|---|---|
| 501,102 | 782 MiB | 74.3 s |
| 701,870 | 1,582 MiB | 111.3 s |
| 852,729 | 2,482 MiB | 192.6 s |
| **1,028,250** | **3,144 MiB** | 211.5 s |

So budget **~3.2 GB of free VRAM per GPU** (outside `--mem-fraction-static`) for a full-1M single
request at chunk 4096, and more if long prefills can overlap. Note prefill is superlinear: a full
1M-token cold prompt takes ~3.5 minutes to first token.

Raising context costs essentially **no** KV pool — the pool is sized by `--mem-fraction-static`,
not by context length (2,052,800 → 2,050,752 tokens going from 469K to 1M context, both at the older
mem-fraction; at the launch line above the same relationship holds at 1,814,144).

Capability checks pass: tool calling, vision (image input), `reasoning_effort`.

## Limitations — read this

- **No agentic or coding benchmark was run.** Nobody has published agentic deltas for *any*
  GLM-5.3-Flash quant, and this card does not claim one. The numbers above are throughput,
  capacity, and capability smoke-tests only.
- **No KLD was measured on this specific checkpoint.** The KLD table above is the published panel
  for the *format class*, not a measurement of these weights.
- Data-free; a calibrated GPTQ/AWQ build should beat it on quality.
- Measured on SM120 (RTX PRO 6000 Blackwell Max-Q, power-capped ~260 W). Weight-only INT4 runs
  the Marlin-class path, not native FP4 tensor cores.
- **This card carried a wrong claim about the MTP head for twelve days, through two revisions**, along
  with a generalisable rule ("an MTP head's acceptance does not survive quantizing the target") that was
  false. The cause was a dialect mismatch in this repo's own `ignore` list, fixed 2026-09-17 — see
  [Speculative decoding](#speculative-decoding-use-the-built-in-mtp-head). It is recorded rather than
  quietly edited out because anyone who pulled these weights before that date and followed the card got
  the broken configuration.
- **The MTP throughput figures on this card are † (measured on the abliterated sibling).** The defect and
  the fix were verified in both repos; the tok/s numbers were not re-run here.
- **Serving requires three local SM120 patches on top of a post-merge SGLang nightly.** There is no
  supported stock *released* build as of 2026-09-17.

Base model © Z.ai, MIT. This derivative is released under the same licence.
