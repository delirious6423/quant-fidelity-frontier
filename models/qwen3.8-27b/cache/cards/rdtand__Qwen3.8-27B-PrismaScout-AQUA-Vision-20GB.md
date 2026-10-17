---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: vllm
pipeline_tag: image-text-to-text
tags:
  - prismaquant
  - prismascout
  - prismaaqua
  - aqua
  - aqua-aura
  - compressed-tensors
  - nvfp4
  - fp8
  - mixed-precision
  - quantized
  - multimodal
  - vision
  - video
  - blackwell
  - dgx-spark
  - mtp
  - speculative-decoding
  - qwen3.8
  - vllm
---

# Qwen3.8-27B-PrismaScout-AQUA-Vision-20GB

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), compressed into
a native multimodal checkpoint with a hard **20,000,000,000-byte** artifact
cap. It retains Qwen3.8 image and video input, its language-model head, and its
multi-token-prediction (MTP) head.

The language body uses a fresh PrismaQuant **full-AQUA per-Linear allocation**
for Qwen3.8: AURA weight-side damage plus activation-side pricing.
Each of its 496 Linears has measured per-Linear AURA costs and a final NVFP4,
FP8 E4M3, or BF16 assignment. Activation-changing body candidates are put onto
an activation-inclusive cost scale with measured family-level
estimator-transfer factors; this calibration is described precisely below.
All 110 visual Linears are fixed to **calibrated native NVFP4 W4A4**. The
single `lm_head` and all eight MTP Linears are fixed to measured FP8 E4M3. The
body budget was solved around those fixed components so the complete
multimodal checkpoint stays below the cap.

The result is a stock `compressed-tensors` checkpoint for **vanilla vLLM**:
no vLLM fork, runtime plugin, or custom checkpoint loader is required.

**Final recursive artifact-snapshot size:** **19978899058 bytes**
(18.6068 GiB).  
**Language-body rate:** **5.0737 bits per quantizable parameter** over
24,350,556,160 body parameters. This bpp convention excludes fixed auxiliary
components; the 20 GB cap includes every regular file in the published
artifact snapshot. Hub-managed repository metadata is audited separately.
The publisher retains the Hub-managed root `.gitattributes` (1,570 bytes),
making the projected Hub tree total **19978900628 bytes**; the publication
wrapper verifies this accounting live.

Method, validation artifacts, and the rest of the PrismaQuant family:
**[prismaquant.org](https://prismaquant.org)**

## What is quantized

| component | Linears | NVFP4 W4A4 | FP8 E4M3 | BF16 |
|---|---:|---:|---:|---:|
| Qwen3.8 language body | 496 | 320 | 168 | 8 |
| Qwen3.8 visual tower and merger | 110 | 110 | 0 | 0 |
| `lm_head` | 1 | 0 | 1 | 0 |
| MTP head | 8 | 0 | 8 | 0 |
| **Total** | **615** | **430** | **177** | **8** |

NVFP4 uses 4-bit weights and activations with group size 16 and calibrated
positive global scales. FP8 E4M3 uses 8-bit weights and activations. The eight
BF16 entries are the `in_proj_a` and `in_proj_b` linear-attention projections
in language layers 1, 2, 4, and 13; the allocator chose to leave them at source
precision. Embeddings, norms, biases, and other non-Linear residue remain at
their declared source precision.

The visual format was constrained, not chosen by a zero-information proxy: all
110 visual Linears were rendered from image-conditioned calibration activations
and bound to the exact source revision and calibration manifest. The
language-body allocation remained fully empirical and per-Linear under the
same whole-artifact budget.

## Layer map

![Qwen3.8-27B-PrismaScout-AQUA-Vision-20GB per-Linear layer map](allocation-map.png)

**Key:** violet = NVFP4 W4A4; teal = FP8 E4M3; orange = BF16; blank = that
projection is not applicable at that position. The visual-tower inset covers
its 27 transformer blocks and merger; the language-body panel covers all 64
layers, `lm_head`, and MTP.

**Not rendered to scale.** The PNG carries this disclaimer and its key inside
the image. The project-site layer-map renderer generated it from the frozen
615-entry final allocation
(`layer_config.json` SHA-256
`beed93b6292116634bc66f60ab262121af0ee184e90bb4333ad069cacfe200c2`),
with exact 615/615 assignment coverage and no missing or extra qnames. The PNG
SHA-256 is
`7231dd83802a0a8df311ef5451d135b37c562cd13710c46529a05f40002ab5b4`.

## Measured

The core export and gold gates are bound by `shipcard.json`. The direct
Qwen3.6 comparison and multimodal fixture campaign carry their own immutable
result manifests, commands, input-contract or fixture hashes, and serving-stack
fingerprints.

### Text fidelity

| measurement | result |
|---|---:|
| KL vs BF16, all positions (4,088 positions) | **0.0439023734** |
| KL vs BF16, confident positions (teacher top-1 > 0.5; 2,063 positions) | **0.0254206691** |
| KL p99 / max | 0.409348640 / 2.379668877 |
| held-out PPL, quantized | **9.648316858** |
| held-out PPL, BF16 source | 9.366213045 |
| quantization cost | **+3.0119% PPL, +0.029674624 nats/token** |
| text eager load + generation | **PASS** — 16 generated tokens |
| text CUDA-graph load + generation | **PASS** — 16 generated tokens |

KL and PPL are measured against the pinned BF16 Qwen3.8 source on identical
corpus bytes and sequence boundaries. Mean KL is a screening measurement, not
a complete behavioral guarantee; the tail is published rather than averaged
away.

The KL run uses eight 512-token samples and scores all 4,088 non-initial
positions against a pinned BF16 teacher payload. Each positional KL is
computed over the teacher's top-1,024-token support plus one aggregated
residual-tail bucket; it is not an exact token-by-token sum over the entire
vocabulary. PPL uses 8,176 scored WikiText-2 test tokens in 16 non-overlapping
512-token chunks. The quantized and BF16 PPL arms share
calibration-contract SHA-256
`9f6e722ebc12b88e6b11466668f466c5cb15ae4b9ce001e856aee36a8efe63e6`.

For context, the previously shipped Qwen3.6 PrismaScout now measures
**0.0779318074 mean KL** under the same deep 8 × 512, all-position protocol,
using the same source corpus but its own tokenizer and pinned Qwen3.6 BF16
teacher. The token windows are therefore not byte-identical. This release's
0.0439023734 is **43.7% lower** under that protocol. Qwen3.6 also measured
0.0513707392 on confident positions and 0.813661665 / 4.971227640 at p99 / max,
versus this release's 0.0254206691 and 0.409348640 / 2.379668877. The measured
deep-protocol tail therefore improves along with the mean.

An older Qwen3.6 release note reported roughly **0.0151**, but that number came
from an earlier
2 × 128 terminal-position sanity run that scored only two predictions. Its
teacher payload, position coverage, and calibration depth were not bound to
this contract. It is retained as historical context, not an apples-to-apples
denominator.

The visionless Qwen3.8 PrismaScout-AQUA sibling measures 0.0402201669 under the
same deep contract, so retaining vision under the same 20 GB cap raises this
screening KL by 9.16%. The checkpoint also spends fixed bytes on native
multimodality and FP8 output/MTP heads, leaving the language body at 5.0737
bpp. These are deliberate whole-model tradeoffs; activation-aware allocation
does not guarantee that every aggregate text metric improves.

On the same PPL contract, this release's 9.648316858 is 0.852% lower than the
visionless PrismaScout-AQUA sibling's 9.731222170 and 0.483% higher than the
5.5 bpp PrismaAQUA sibling's 9.601918337. These comparisons isolate neither
vision nor allocation method by themselves; they describe the shipped
whole-artifact tradeoffs.

### Multimodal and runtime gates

| gate | result |
|---|---:|
| served visual W4A4 census: 110/110 native `CompressedTensorsW4A4Fp4`, `use_a16=False` | **PASS** |
| optimized visual kernel | **110/110 `FlashInferCutlassNvFp4LinearKernel`** |
| single-image association at 256 × 256 and 448 × 448 | **PASS** |
| forward/reverse batched-image ordering and stale-buffer replay | **PASS** |
| four-frame 64 × 64 video input | **PASS** — 16 generated tokens in both arms |
| eager multimodal load + generation | **PASS** — 57 total generated tokens |
| explicit encoder CUDA-graph capture/replay + generation | **PASS** — 8 hits, 0 misses; 54 total generated tokens |
| MTP eager + graph generation (`num_speculative_tokens=3`) | **PASS** — 16 tokens per arm |
| vanilla-vLLM no-spec HTTP ship gate | **PASS** — health, four generation checks, and 583-token serving-smoke PPL 4.0725; no speculation detected |
| eight-image quality suite vs BF16, two replicates | **7/8 in both arms and both replicates; identical predicted choices** |

Eager and CUDA-graph modes are validated in separate clean processes. The
strict graph arm enables multimodal encoder graph capture explicitly and
requires observed capture/replay statistics: mode 3, Inductor,
`FULL_AND_PIECEWISE`, encoder budgets 64/256/1,024, maximum batch 2, and maximum
four video frames. The association checks compare standalone, forward-batch,
reverse-batch, and stale-replay token sequences; they are designed to catch
image-cache or batching mistakes, not merely prove that a request returns HTTP
200.

The ship-gate PPL is a separate 583-token HTTP sanity suite, not the gold
WikiText-2 PPL reported under Text fidelity. Because that gate deliberately
disabled speculation, its `mtp_acceptance` check observed zero drafts and was
skipped; MTP evidence comes from the separate eager and CUDA-graph generation
arms above.

The eight-image suite is intentionally reported without inflating it into a
general vision benchmark. The quantized artifact and BF16 source selected the
same multiple-choice letter on every image, including the same one miss, in
both deterministic replicates. Quantized repeat digest
`96504b3ba76f5d6a11dd01bc2457479f5332a6e578b124e5a2762146274f9c1e`
was identical across repeats. Free-form captions were close paraphrases but
**0/8 exact string matches**, and one image received a materially different
action interpretation. The fixture result is therefore choice parity on this
suite, not broad caption equivalence.

On the warm second pass of that same 16-request visual mix, the quantized
artifact decoded at **12.2990 tok/s** versus **4.5573 tok/s** for BF16 (2.699×),
with mean TTFT **0.2627 s** versus **0.5779 s**, and mean end-to-end latency
**1.4976 s** versus **4.3904 s**. Generated-token totals differed (259 versus
294), so these are observed workload results rather than a fixed-token decode
microbenchmark. Warm-pass GPU utilization averaged 95.91% (p50/p95/max 96%)
at 34.01 W mean reported GPU power on DGX Spark.

### Serving performance

This exact-artifact text-path campaign compares the previously shipped Qwen3.6
PrismaScout directly with this release. Both artifacts ran on the same physical
NVIDIA GB10, driver 595.84, pinned runtime image, and matched vLLM flags. Both
used the same Qwen3.6 tokenizer, fixed random prompts and seeds, and
`--language-model-only --skip-mm-profiling` to isolate language serving. The
Qwen3.6 result files were reused immutably from the matched prior run; their
result-set identity was revalidated before analysis. Ratios below are the
Vision artifact's repeat mean divided by the Qwen3.6 repeat mean. Power mode
was not pinned or matched, so this is not a power-efficiency comparison. The
frozen 48-file receipt and analyzer output are identified by hashes in
Provenance.

**No-spec prefill:** eight requests per repeat, each exactly 2,048 input tokens
and one output token. Input throughput is the primary metric; `n=3` per arm.

| concurrency | Qwen3.6 input tok/s, mean (CV) | Vision input tok/s, mean (CV) | Vision / Qwen3.6 | mean TTFT, Qwen3.6 / Vision |
|---:|---:|---:|---:|---:|
| 1 | 3,047.767 (0.016%) | **3,059.426 (0.158%)** | **1.00383× (+0.383%)** | 671.795 / **669.246 ms** |
| 4 | 11,529.786 (0.246%) | **11,357.815 (1.223%)** | **0.98508× (−1.492%)** | **675.438** / 682.336 ms |

**Decode:** four requests per repeat, each exactly 32 input and 256 output
tokens. Output throughput is the primary metric; `n=3` per arm without
speculation and `n=6` per arm with MTP-3.

| mode | concurrency | Qwen3.6 output tok/s, mean (CV) | Vision output tok/s, mean (CV) | Vision / Qwen3.6 | mean TTFT, Qwen3.6 / Vision | mean ITL, Qwen3.6 / Vision |
|---|---:|---:|---:|---:|---:|---:|
| no spec | 1 | 12.581 (0.001%) | **13.086 (0.059%)** | **1.04008× (+4.008%)** | **101.219** / 102.047 ms | 79.397 / **76.318 ms** |
| no spec | 4 | 48.902 (0.036%) | **49.369 (0.029%)** | **1.00953× (+0.953%)** | 193.488 / **188.366 ms** | 81.279 / **80.524 ms** |
| MTP-3 | 1 | 24.639 (7.135%) | **31.206 (6.803%)** | **1.26650× (+26.650%)** | 220.690 / **179.589 ms** | 128.320 / **102.224 ms** |
| MTP-3 | 4 | 83.037 (8.547%) | **102.355 (8.741%)** | **1.23265× (+23.265%)** | 270.469 / **232.762 ms** | 126.491 / **112.066 ms** |

All 48 observed result JSONs passed identity, request, token, failure, metric,
and throughput-arithmetic checks, and all 12 groups were complete. Primary
throughput CVs are shown rather than hidden. All no-spec primary-throughput
rows stayed below their CV thresholds. Both models' MTP rows exceeded the
campaign's 3% (`c=1`) or 5% (`c=4`) CV thresholds; MTP TTFT CV reached 21.51%,
and the Vision no-spec `c=1` TTFT CV was 8.54%. Across primary, derived, and
latency metrics, the analyzer records 27 CV disclosure flags and therefore
sets `strict_pass=false`. The MTP point estimates are informative repeated
means, not a clean CV-gated performance pass or a claim of statistical
significance.

Within this contract, no-spec prefill is within −1.49% to +0.38% of Qwen3.6,
no-spec decode is 0.95% to 4.01% faster, and MTP-3 decode is 23.27% to 26.65%
faster. The previous release appears only as a serving comparator; its
quantization process is not part of this checkpoint's build.

## How it was built

**AURA** supplies per-Linear language-body costs from a fresh Qwen3.8
KL-adjoint probe and the production-rendered weight delta—the same
representation later exported. The allocator had exact probe and cost coverage
for all 496 body Linears, the output head, and eight MTP Linears.

For the body, this run's **AQUA activation correction is calibrated at the
format-family level**. Eight MTP rows carry both activation-inclusive measured
output MSE and the corresponding weight-only estimator for each family. Their
geometric-mean ratios produce estimator-transfer factors of **8,102.699× for
NVFP4** and **7,658.832× for FP8**. Those factors put the 496 per-Linear AURA
body rows for each activation-changing family onto an activation-inclusive
cost scale before the multi-choice solve. This is not a claim that all 992
NVFP4/FP8 body candidates had direct activation-output measurements: their
weight damage is measured per Linear, while the activation correction is the
receipt-bound family transfer described here. The eight BF16 body candidates
selected by the solver remain exact activation-identity rows.

Vision uses a different, exact calibration path. All 110 fixed-NVFP4 visual
Linears were rendered from image-conditioned activation captures with exact
per-Linear activation widths and positive finite global scales. The visual
format was fixed to native NVFP4 W4A4; it was not selected from an
uninformative proxy.

The multi-choice budget solve selects a receipt-bound feasible language-body
assignment across `{NVFP4, FP8_E4M3, BF16}` while the visual tower, `lm_head`,
and MTP formats are fixed auxiliary assignments that still count toward the
hard whole-artifact cap. It does not claim global optimality.

The production render uses GPTQ with fixed damping 1.0, static activation
ordering, and joint scale optimization (JSO). The visual tower uses that native
NVFP4 W4A4 render path with image-conditioned calibration activations and
calibrated input/weight global scales. It is not a separately loaded sidecar.

The two-machine build was striped by complete Linear ownership, rendered into
the existing resident production-weight cache, and combined by an exact
disjoint union. Final export fails closed unless all 607 quantized Linear cache
pairs are resident and receipt-bound. The remaining quantized language-body,
`lm_head`, and MTP weights then passed through the existing production recache:
the quantized upstream weights were installed during replay and downstream
activation maximums were re-fit before export. The 110 image-conditioned
visual scales were explicitly preserved because the text replay cannot observe
them. Quantization itself does not use tensor parallelism; each Linear has
exactly one rendering owner.

## Serving with vLLM

The validated lane is vanilla vLLM with native `compressed-tensors` kernels on
Blackwell. The final runtime fingerprint and exact gate commands are recorded
in the release evidence. This conservative example uses the gated
4,096-token configured serving ceiling:

```bash
vllm serve rdtand/Qwen3.8-27B-PrismaScout-AQUA-Vision-20GB \
  --host 0.0.0.0 --port 8000 \
  --quantization compressed-tensors \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.75 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --safetensors-load-strategy prefetch \
  --limit-mm-per-prompt '{"image":1,"video":1}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

- Add `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
  for the separately validated MTP text-generation path.
- Adjust `--limit-mm-per-prompt` to match your workload and memory budget.
- The source model supports a native 262,144-token context. This release is
  served and gated with a configured 2,048-token ceiling in the strict
  multimodal/MTP checks and a configured 4,096-token ceiling in the
  visual-quality and direct-throughput campaigns. Those ceilings are not
  claims that maximum-length prompts were exercised. Longer contexts require
  their own memory and quality validation.
- Thinking mode is on by default. `--reasoning-parser qwen3` separates the
  thinking block from response content; `qwen3_xml` matches the model's tool
  call format.

Example image request through the OpenAI-compatible endpoint:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")
response = client.chat.completions.create(
    model="rdtand/Qwen3.8-27B-PrismaScout-AQUA-Vision-20GB",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                {"type": "text", "text": "Describe the image."},
            ],
        }
    ],
)
print(response.choices[0].message.content)
```

## Limitations

- Native NVFP4 W4A4 serving is hardware- and kernel-dependent. This artifact is
  validated on NVIDIA GB10 / DGX Spark with the pinned vLLM container below;
  unlisted GPU architectures and fallback kernels are not claimed.
- Quantization can change rare-token, tool-call, OCR, fine-detail, and
  long-context behavior. Mean KL and a small multimodal fixture suite cannot
  cover every failure mode.
- The visual calibration set contains eight pinned COCO-caption images. Final
  validation uses a disjoint eight-image test set plus real association,
  resolution, batching, and video fixtures. Behavior far from those domains is
  not established.
- The 5.0737 bpp figure covers quantizable language-body parameters only. Use
  the exact recursive artifact bytes—not bpp—when comparing storage budgets.
- Body activation distortion is calibrated through one measured
  estimator-transfer factor per format family. Per-Linear body weight damage
  is measured, but the body does not carry a direct activation-output
  measurement for every `(Linear, format)` row. This residual modeling limit
  is one reason the card publishes deep end-to-end KL and PPL.
- MTP is optional. Its eager and graph text-generation arms passed, but the
  strict speculative test did not combine MTP with image/video requests.
- The eight-image suite matched BF16 multiple-choice predictions but not exact
  captions. Treat 7/8 on this fixture as a smoke-scale result, not a general
  visual benchmark.
- This is a quantized derivative, not a retraining or fine-tune. Refer to the
  [Qwen3.8-27B source card](https://huggingface.co/Qwen/Qwen3.8-27B) for the
  base model's capabilities, usage guidance, license, and benchmark scope.

## Provenance

The final `shipcard.json` is the authority. If a value in this README differs
from that machine-verifiable record, treat this README as stale.

| item | pinned value |
|---|---|
| source | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| source `config.json` SHA-256 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| source safetensors index SHA-256 | `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df` |
| body calibration corpus SHA-256 | `e09a138a4903c4af66a3bf2f9367185f3432224391f1dfe8c94ccc29d99315ba` |
| visual calibration manifest SHA-256 | `8c9729996361267f46f232b1349f53423bda7e924bd6a5c8b4e20211f5612b10` |
| final allocation `selection.json` SHA-256 | `0fbb481f2f0edc1b269bfaad1487ac4a864177934030dcc8075b3fdf1e7d7014` |
| final allocation `layer_config.json` SHA-256 | `beed93b6292116634bc66f60ab262121af0ee184e90bb4333ad069cacfe200c2` |
| production-recache PWC SHA-256 | `37f1c26adcf00dbed62575345a0c26435b7b1139b62cf4601670223d0b31dd63` |
| export producer commit | `7267075cecf9306bef3709e74df7acc0633256d9` |
| MTP root-head metadata fix | `9c27d0ba1704db60a0013f55236421bac514a748` (config only; tensor payload unchanged) |
| strict native validator commit | `b4e4ec95bc3528f69b0d026ba2b33084d55af2af` |
| validation runtime | `eugr/spark-vllm@sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869` |
| vLLM / Torch / FlashInfer | `0.26.1rc1.dev693+g7f7a32cfe.d20260812` / `2.13.0+cu130` / `0.6.18` |
| final model identity (`model_sha`) | `2820b67c8a257d6cbecf716bf79e8a3bf1a1f11d72cac1277c1cf10882a32bf8` |
| final `config.json` SHA-256 | `f0abef7da6451903ef951c9ca03ca0bd084f751e5b8df1bce69d432d0bc95216` |
| final tensor index SHA-256 | `5b4b1cfb39da08bea1ebca2fe3ecd8c38672169eac8caf05955159cb1f978be0` |
| BF16 KL teacher payload SHA-256 | `3edc8c77960af0459998778fa15c0426fd19539e0e780ff03caaa7797e8fab01` |
| BF16 KL teacher metadata SHA-256 | `d503841c62866464da1b0e0e586eb658af9df83b1d7d2e687fd377d75a4cf1bd` |
| final KL result SHA-256 | `55cee50441208a0df03297f5113a3622ee68cb35e16dba04692f78546302addf` |
| deep Qwen3.6 comparator KL result SHA-256 | `20fe0320dba247b1ae257600ce16ecd9b36980988a6c40a1af46b9c1ee90168e` |
| final PPL result SHA-256 | `24de3afa14968add3ab7d2a01c1e4dfc127bfd1afc7f3d81f669c90fefe9ca47` |
| visual comparison summary SHA-256 | `c453746bf2e18e3d47932044eef044912b9dfe4b802d45e3288715a45091698d` |
| direct Qwen3.6 throughput result-set identity | `034637c2b117349ccf529122706f70a704fe23b6aae09101aa6012e49736f423` |
| complete 48-file throughput result-set identity | `d3b44e5ed4428944cacf5256f7a20b6397ef76261034fb2f165d49172401d5eb` |
| direct Qwen3.6 throughput analysis SHA-256 | `58ebf824623ceb3050358686c2d5f58b0ea9c649ea49edc193364f0da0950e51` |
| direct throughput post-run receipt SHA-256 | `d0fc304774127600f2b72aa32f8aadfd7311c7359ec51bbbe601a317268750c9` |
| no-spec HTTP ship-gate record SHA-256 | `02c6ae574b0830656637265dac4f38081667bd42f458db8535bb6e3680b0a41f` |
| verified shipcard SHA-256 | `263d5576256e4ca9c5a6f657d189e29867a7a62815cc3d92ae1b00af0b2edcdb` |

The text-only 20 GB sibling is available at
[`rdtand/Qwen3.8-27B-PrismaScout-AQUA-20GB`](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaScout-AQUA-20GB),
and the 5.5 bpp PrismaAQUA sibling is available at
[`rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm`](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm).
The serving comparator is the previously released
[`Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm`](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm);
it remains a supported choice for users who prefer Qwen3.6.

- Project: <https://prismaquant.org>
- Code: <https://github.com/RobTand/prismaquant>
- Base model: <https://huggingface.co/Qwen/Qwen3.8-27B>

## Citation

```bibtex
@software{tand_prismaquant,
  author  = {Tand, Robert},
  title   = {PrismaQuant: production-faithful KL--Fisher bit allocation},
  url     = {https://prismaquant.org}
}
```

Contact: robert.tand@icloud.com
