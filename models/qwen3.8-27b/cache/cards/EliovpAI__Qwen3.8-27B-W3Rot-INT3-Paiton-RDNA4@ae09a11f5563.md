---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: paiton
pipeline_tag: text-generation
inference: false
tags: [paiton, vllm, amd, rocm, rdna4, gfx1201, int3, 3-bit, gptq, hadamard, w3a4, dflash2, speculative-decoding]
---

# Qwen3.8 27B rotated INT3 weights for Paiton on RDNA4

These are 3-bit replacement weights for the decoder projections of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
calibrated with our own GPTQ run on permissively licensed data. They are an add-on to the
[current Paiton Qwen3.8 release](https://github.com/Eliovp-BV/paiton-vllm-plugin/blob/main/models/Qwen3.8-MXFP4-DFlash2/README.md) on one
AMD Radeon AI PRO R9700, not a standalone checkpoint. The same target checkpoint supplies all other tensors, and the
same DFlash2 drafter is used. Against the previous MXFP4 image, weighted decode rises **19.9 %**, prefill rises
**5–13 %**, and model memory falls from 19.18 to 15.89 GiB. The cost was about 3 points of MMLU-Pro knowledge
recall; the 12 October update recovers an estimated 0.5 to 1 point of it.

**12 October 2026: accuracy update.** The BF16 group scales were distilled toward the bf16 model. The format, the
3-bit codes, the runtime image and the launcher are unchanged; only scale values differ. Against the previous
weights, the KL divergence to bf16 falls by **20.9 %** (decode) and **23.8 %** (prefill), and by **19.5 %** when
served on the R9700, at unchanged speed ([details](#accuracy-update-12-october-2026)). The previous weights stay
available at revision `7a2e3d702144f744c1c5a46eaa42c7a2309d5cf7`.

**GPU:** Radeon AI PRO R9700 (`gfx1201`, 32 GB) · **Size:** 9.55 GB · **Context:** 65,536 by default, up to 262,144 (524,288 opt-in) · **Cache:** up to 628,877 tokens, 555,128 with images · **Text weights**

## Requirements

| Component | Requirement |
|---|---|
| GPU | One AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4, 32 GB) |
| Host | Linux x86-64, Python 3, Docker, AMD GPU device access (`/dev/kfd`, `/dev/dri`) |
| Runtime image | `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20261005-r1@sha256:245c71d54f046f89b74dbfbd4c894003754561e55c9d215a8d3ce65364fb0f80` |
| Launcher | `models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh` in the [public plugin repository](https://github.com/Eliovp-BV/paiton-vllm-plugin) |
| Target checkpoint | [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/tree/f0b7c9e722f5565102fff8481c99e4d86ae099c7) at `f0b7c9e722f5565102fff8481c99e4d86ae099c7` |
| Draft checkpoint | [`tcclaviger/Qwen3.8-27B-DFlash2-FP8`](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8/tree/ee0cb26a8279b7910cc28d82a8a3e15e4728d56f) at `ee0cb26a8279b7910cc28d82a8a3e15e4728d56f` |
| These weights | This repository at `3ef7eb149ec0bd9a0f7927491bf30f53f54ced01` |

Use exactly these revisions. The rotated Gated DeltaNet b/a rows were derived from the target revision's bytes.

## Quick start

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin

export PAITON_TARGET_DIR="$PWD/model-cache/qwen38-nvfp4"
export PAITON_DRAFT_DIR="$PWD/model-cache/qwen38-dflash2"
export PAITON_W3ROT_DIR="$PWD/model-cache/qwen38-w3rot-int3"
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-20261005"
mkdir -p "$PAITON_TARGET_DIR" "$PAITON_DRAFT_DIR" "$PAITON_W3ROT_DIR" "$PAITON_CACHE_DIR"

hf download unsloth/Qwen3.8-27B-NVFP4 \
  --revision f0b7c9e722f5565102fff8481c99e4d86ae099c7 --local-dir "$PAITON_TARGET_DIR"
hf download tcclaviger/Qwen3.8-27B-DFlash2-FP8 \
  --revision ee0cb26a8279b7910cc28d82a8a3e15e4728d56f --local-dir "$PAITON_DRAFT_DIR"
hf download EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4 \
  --revision 3ef7eb149ec0bd9a0f7927491bf30f53f54ced01 --local-dir "$PAITON_W3ROT_DIR"
(cd "$PAITON_W3ROT_DIR" && sha256sum -c SHA256SUMS)

bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh
```

Already cloned it? Run `git pull` first; the launcher pins the image above. `run-3bit.sh` serves these 3-bit
weights from `PAITON_W3ROT_DIR`, and `run-mxfp4.sh` serves the MXFP4 path on the same image. Pick one mode by
adding its flag, e.g. `bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long`:

What you get with these weights (one R9700; all commands start with `bash models/Qwen3.8-MXFP4-DFlash2/`):

| You run | Per request (prompt + output) | Shared cache (text) | Cache with `--vision` | Full-length requests that fit together | Prefix caching |
| --- | ---: | ---: | ---: | --- | --- |
| `run-3bit.sh --mode long-kv4`: **coding agents, long documents, screenshots** | 262,144 | **628,877** | **555,128** | 2 (plus one of 100K) | yes |
| `run-3bit.sh --mode long-kv4 --extend-cache` | 262,144 | 628,877 on the GPU, plus about 0.7M to 2.8M in system RAM or on an NVMe/SSD | – | 2 | yes, also after a restart |
| `run-3bit.sh --mode long` | 262,144 (245,000 with images) | 316,325 | 253,298 | 1, plus short ones | yes |
| `run-3bit.sh --mode long-512k` (experimental) | 524,288 | 594,290 | – | 1, plus short ones | yes |
| `run-3bit.sh` (the default, `--mode 65k`): fast everyday chat | 65,536 | 393,216 | 278,050 | 6 | no |

Every mode takes up to eight requests at once and uses the checkpoint's FP8 output head (since the 11 October
launcher update). `--mode long-kv4` keeps the embedding table (2.4 GiB) in pinned
system RAM, uses a 4-bit KV cache, and reads long text-only prompts in
2,048-token steps so short questions from other users are answered meanwhile (about 0.7 s instead of 5.9 s during a
64K read). `--mode long-512k` applies the model's official long-context scaling to every request. Images up to 4K
are read at full resolution. The [plugin README](https://github.com/Eliovp-BV/paiton-vllm-plugin/blob/main/models/Qwen3.8-MXFP4-DFlash2/README.md#what-you-get) has the same table for the MXFP4 path
(`run-mxfp4.sh`), what each row gives up, and the measurements.
Earlier images stay runnable with `--image`; the
[plugin README](https://github.com/Eliovp-BV/paiton-vllm-plugin/blob/main/models/Qwen3.8-MXFP4-DFlash2/README.md#run-an-earlier-release)
lists their exact references.
Once the server is ready, it accepts OpenAI-compatible requests at `http://127.0.0.1:18982/v1` with API model
name `Qwen3.8`.

## Method

- **Weights.** The MLP, attention and Gated DeltaNet projections of all 64 layers (24.3 billion weights) are
  symmetric INT3 with one BF16 scale per 128 input elements (3.125 bits per weight).
- **Rotation.** MLP gate/up, attention q/k/v and Gated DeltaNet `in_proj` are stored in a block-diagonal
  Hadamard-128 basis, and the runtime rotates their inputs to match.
- **Calibration.** We ran sequential GPTQ in the rotated basis with an MSE scale search on 292,864 tokens of
  permissively licensed data. [CALIBRATION.md](CALIBRATION.md) lists the sources.
- **Scale distillation (12 October 2026).** Starting from the GPTQ result, the BF16 group scales of all 64 layers
  were trained end to end so that the 3-bit model's next-token distributions match bf16 Qwen3.8-27B: KL divergence
  on assistant and plain-text tokens, under both serving policies (FP8 activations for decode, 4-bit for prefill).
  Each scale could move by −18 % to +22 %, and no 3-bit code changed. The data were 5.16 million tokens (57 % code)
  from seven of the permissively licensed sources in [CALIBRATION.md](CALIBRATION.md), decontaminated against the
  evaluation sets; the manifest's `finetune` block lists them with revisions and licenses.
- **Prefill.** The rotated projections take 4-bit activations, scaled per token and per 128-element group.
- **Everywhere else.** The other projections, and all projections during decode, take 8-bit (FP8) activations.
- **Execution.** Native RDNA4 kernels in the Paiton image run these weights. All other tensors come from the
  pinned target checkpoint.

## Accuracy

### Accuracy update, 12 October 2026

Each row pairs these weights with the previous weights (`7a2e3d70`) on identical inputs, with a 95 % bootstrap
interval over documents. KL is KL(bf16 ‖ 3-bit) of the next-token distribution, averaged over assistant and
plain-text tokens; top-1 is how often the 3-bit model's most likely next token is also bf16's. The first rows
reproduce the runtime's arithmetic under both activation policies on MI355X GPUs; the R9700 rows come from the
Paiton image itself.

| Metric | Previous weights | These weights | Change [95 % CI] |
|---|---:|---:|---:|
| **KL to bf16, decode** (115 held-out chats, code, math, tool calls, documents) | 0.0851 | 0.0673 | **−20.9 %** [−29.8, −14.1] |
| **KL to bf16, prefill** (4-bit activations, same texts) | 0.1074 | 0.0818 | **−23.8 %** [−39.5, −13.1] |
| Top-1 agreement with bf16, decode / prefill | 88.66 / 87.36 % | 89.70 / 88.38 % | **+1.0 / +1.0 points** |
| Long documents (22, 8K to 32K tokens), KL decode / prefill | 0.0806 / 0.0975 | 0.0746 / 0.0909 | −7.5 / −6.8 % |
| the same, positions beyond 8K tokens | 0.0632 / 0.0770 | 0.0578 / 0.0710 | −8.6 / −7.9 % |
| 131K-token documents (5), KL in each length bucket up to 128K | 0.20 to 0.37 | 0.19 to 0.36 | −2.3 to −4.0 %, every bucket better |
| MMLU-Pro subset (1,400), answer log-likelihood, prefill / decode; bf16: 65.29 % | 61.36 / 62.43 % | 62.93 / 63.00 % | about +0.5 to +1 point (see below) |
| **R9700, served:** KL to bf16 (top-20 log-probabilities, 40 texts) | 0.1732 | 0.1395 | **−19.5 %** [−36.8, −8.5] |
| R9700, served: top-1 agreement with bf16 | 86.90 % | 88.14 % | +1.2 points [+0.6, +2.1] |

- **No category gets worse** under either policy. Chat, code and math gain the most (decode KL −41 %, −35 % and
  −22 %); knowledge questions (−1 % decode, −6 % prefill) and tool calls under prefill (−0.7 %) gain least, within
  noise.
- **Long context.** The gain is smaller at 131K tokens than on shorter texts, but every bucket improves, from the
  first 8K tokens to 128K. On the R9700, the served 131K check found the new weights better or equal in every
  bucket.
- **MMLU-Pro.** This training run scores +1.57 / +0.57 points (prefill / decode policy), a second run with another
  seed +0.57 / +0.79 points; the paired intervals include zero. We read this as about +0.5 to +1 point of the 2.9 to
  3.9 points by which the previous weights trailed bf16 in this scoring.
- **Speed is unchanged**, as the format and kernels are the same. On one R9700 with the current launcher and image
  (`--mode long-kv4`), the decode step takes 28.7–28.9 ms and the first token 66 ms (p50) with either weights, and
  prefill from 2K to 64K tokens is within 0.3 %. BetterBench weighted decode reads 170.1 against 172.8 tok/s, and
  482.2 against 481.9 tok/s at eight requests; the single-stream difference comes from speculative-decoding
  acceptance on replayed conversations, not from kernel speed.
- Outputs differ from the previous weights, greedy ones included. The GSM8K, HumanEval, needle and long-session
  results below were measured with the previous weights and were not re-run for this update.

### Previous weights, 26 September 2026

These results were measured in-model on 2026-09-26 with the Paiton image and the previous weights, using greedy
decoding with thinking off. Each benchmark is paired against the MXFP4 image on identical items. Δ is in points
with a normal-approximation 95 % CI, and p is the exact McNemar test.

| Benchmark | MXFP4 | Previous weights (W3A4) | Δ [95 % CI] | McNemar p |
|---|---:|---:|---:|---:|
| GSM8K 5-shot (1,319) | 95.68 % | 95.30 % | −0.38 [−1.44, +0.68] | 0.58 |
| HumanEval pass@1 (164) | 95.12 % | 93.90 % | −1.22 [−4.99, +2.56] | 0.75 |
| MMLU-Pro subset, 0-shot direct answer (14 × 100) | 62.57 % | 59.71 % | −2.86 [−4.81, −0.90] | 0.005 |
| Needle at 61,440 tokens (80) | 100 % | 100 % | 0 | 1 |

DFlash2 mean acceptance length relative to MXFP4 changes by +0.1 % on 38 chat prompts, −0.4 % on GSM8K,
+2.8 % on HumanEval, +1.8 % on MMLU-Pro and +2.0 % on the needle test.

Math, code and long-context retrieval stay within noise. Knowledge recall drops by about 3 MMLU-Pro points,
which is the trade-off of 3-bit weights. The MXFP4 path (`run-mxfp4.sh`) remains the choice for maximum
knowledge accuracy.

**Long coding sessions.** In 20-turn coding sessions that grow from about 100K to 181K tokens, with a new
HumanEval+ task every turn (8 sessions, 160 tasks), `--mode long-kv4` with the previous weights solved **87.5 %** of the
tasks with no decline from the first turns to the last (change per turn 0.00, 95 % interval −0.71 to +0.64), against
83.75 % for MXFP4 in `--mode long`. Cached turns started in 3.1 to 4.1 s and decoded code at about 210 tok/s
([full table](https://github.com/Eliovp-BV/paiton-vllm-plugin/blob/main/models/Qwen3.8-MXFP4-DFlash2/README.md#long-coding-sessions)).

## Performance

The benchmark is BetterBench 0.6.0 (quick preset) with the 65K release profile on one R9700, using two fresh
server processes per image. The reference is the previous MXFP4 image, `qwen38-rocm10-vllm029-65k-20260924-r3`.
These runs used the earlier v3 calibration of these weights. Speed does not depend on the calibration data or on
the 12 October scale update, because the tensor format and runtime are identical. Changes are computed from
unrounded values.

| Metric | MXFP4 | These weights | Change |
|---|---:|---:|---:|
| **Weighted decode, tok/s** | **153.8** | **184.4** | **+19.9 %** |
| chat / code / file edit | 121.2 / 179.9 / 179.9 | 135.8 / 226.0 / 195.0 | +12.0 / +25.6 / +8.5 % |
| json / math / prose | 217.5 / 183.8 / 78.5 | 269.0 / 228.9 / 94.7 | +23.7 / +24.5 / +20.7 % |
| reasoning / summarization | 117.8 / 138.4 | 133.5 / 158.4 | +13.3 / +14.4 % |
| Aggregate at 1 / 2 / 4 / 8 requests, tok/s | 122.0 / 204.2 / 308.2 / 425.3 | 148.8 / 249.3 / 368.3 / 492.1 | +22.0 / +22.1 / +19.5 / +15.7 % |
| Prefill at 2K / 8K / 16K, input tok/s | 3,689 / 3,834 / 3,871 | 4,156 / 4,165 / 4,103 | +12.7 / +8.6 / +6.0 % |
| Prefill at 32K / 64K, input tok/s | 3,751 / 3,455 | 3,958 / 3,629 | +5.5 / +5.0 % |

Time to first token shortens by the same factors as prefill throughput. Model memory falls from 19.18 GiB to
15.89 GiB, and the launcher gives the freed VRAM to the KV cache: 250,578 cached tokens instead of 174,634, or 3.8
concurrent 65K-token requests instead of 2.7. Since 28 September, the 65K mode stores that cache in 4 bits by
default with these weights: vLLM reports 393,216 tokens, and six 61K-token requests run together. Its accuracy and long-session checks are in the
[plugin reference](https://github.com/Eliovp-BV/paiton-vllm-plugin/blob/main/models/Qwen3.8-MXFP4-DFlash2/REFERENCE.md#more-context-capacity-4-bit-kv-cache).

## Files and verification

The repository holds 64 layer shards (`w3rot-L00.safetensors` … `w3rot-L63.safetensors`, 1,296 tensors),
`w3rot-index.json`, `w3rot-manifest.json`, `CALIBRATION.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md` and `SHA256SUMS`.
The manifest records the format, per-shard SHA-256, b/a source hashes and calibration provenance, including 33
dataset entries with revisions and licenses. Since the 12 October update it also records the new shards' SHA-256,
`weights_version` and a `finetune` block: the base revision and manifest hash, the objective, the data sources with
revisions and licenses, and per-shard counts of changed values (43.6 million of 190.1 million scales, no 3-bit
code). Everything else in it is unchanged. The paths in its `sources` section refer to the calibration environment.

```bash
sha256sum -c SHA256SUMS
sha256sum w3rot-manifest.json   # 2d1f6c312a253397ad5c293d4d7a5dc53c6e50de64e58f05f4fa50d59d211f67
```

The previous weights' manifest (revision `7a2e3d70`) is `335c5c834fb3f455203908d2b5aff8131aca2fe64ad73d59feddf32247d41a70`.

## Limitations

- The weights are lossy. With the previous weights, MMLU-Pro scored about 3 points below MXFP4 (the 12 October
  update recovers an estimated 0.5 to 1 point). Outputs differ from the MXFP4 path, and greedy generations can
  diverge.
- The task benchmarks below were measured with the previous weights. The 12 October update was checked with KL to
  bf16 on held-out texts, long documents up to 131K tokens and MMLU-Pro log-likelihood, and with a served check on
  the R9700; GSM8K, HumanEval, needles and the long coding sessions were not re-run.
- The evaluation covers GSM8K, HumanEval, a 1,400-question MMLU-Pro subset, 38 chat prompts and needle retrieval
  at 61,440 tokens. Multilingual and tool-calling quality are not measured. For the long modes, the plugin reports
  GSM8K, HumanEval and MMLU-Pro with every question placed after a 257K-token document in `--mode long` (95.83 /
  91.46 / 60.29 against 95.30 / 93.90 / 59.71 at short context, within the paired 95 % intervals), and, in
  `--mode long-kv4`, that the 4-bit cache predicts five 131K-token documents as well as the FP8 cache. With
  prefix caching (`--mode long-kv4`), GSM8K, HumanEval, MMLU-Pro and the 61,440-token needle stay within noise of
  the 2 October mode; `--mode long-512k` stays within noise of `--mode long-kv4` on the same tests and found 4 of 4
  planted facts at 300K and at 500K tokens
  ([long-mode results](https://github.com/Eliovp-BV/paiton-vllm-plugin/blob/main/models/Qwen3.8-MXFP4-DFlash2/README.md#current-benchmark-results)).
- The 5 October image serves these weights in all four modes. They cover the language model; with `--vision`
  (not in `--mode long-512k`), image input uses the target checkpoint's vision encoder.
  One R9700 only; Transformers, stock vLLM and llama.cpp cannot load them.
- The default 65K mode uses the 4-bit KV cache. The long modes use the FP8 cache (`--mode long`) or the 4-bit
  cache (`--mode long-kv4` and `--mode long-512k`), all three with prefix caching. The MXFP4 path and other
  settings use the FP8 cache.
- `--mode long-kv4` reads a new long prompt about 3 % slower than without prefix caching (7 to 13 % for short prompts) and
  keeps 2.4 GiB of system RAM pinned; a host that cannot pin it gets a 451,879-token cache instead.
  `--mode long-512k` is opt-in: its long-context scaling applies to every request, it needs that system RAM, and a
  500K-token document takes about 6 minutes to read the first time.
- The weights are tied to the pinned target revision and to Paiton's rotation and activation formats.

## License and attribution

These weights are released under [Apache-2.0](LICENSE). They are a quantized derivative of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0), and the rotated b/a rows derive from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) (Apache-2.0). The method builds on
GPTQ and [GSQ](https://github.com/IST-DASLab/GSQ) (Apache-2.0). The target checkpoint and the DFlash2 drafter
([tcclaviger/Qwen3.8-27B-DFlash2-FP8](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8), Apache-2.0)
are not redistributed here. The calibration and scale-distillation data require these attributions:

- **QASC.** Tushar Khot, Peter Clark, Michal Guerquin, Peter Jansen and Ashish Sabharwal (Allen Institute for
  AI), [allenai/qasc](https://huggingface.co/datasets/allenai/qasc). It is licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and provided as is. We reformatted it into
  calibration prompts, and it is not redistributed.
- **ODC-By datasets.** The calibration contains information from
  [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
  [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and
  [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu) (Hugging Face), and the scale
  distillation contains information from FineWeb-Edu. They are made
  available under the [ODC Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/), and their
  use is also subject to [Common Crawl's Terms of Use](https://commoncrawl.org/terms-of-use).

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) has the full notices ·
[Paiton](https://eliovp.com/products/paiton) · [Public Paiton plugin](https://github.com/Eliovp-BV/paiton-vllm-plugin)
