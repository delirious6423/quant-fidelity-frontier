---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - orcarouter/Qwen3.8-Flash-Next-Uncensored
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: image-text-to-text
inference: false
tags:
  - gguf
  - amd
  - rocm
  - gfx1151
  - strix-halo
  - mixture-of-experts
  - mtp
  - long-context
  - vision
---

![Qwen3.8 Flash CIRU Strix Orca](assets/qwen38-flash-ciru-strix-orca.png)

# Qwen3.8 Flash CIRU Strix Orca — v5.5.0 runtime update

OrcaRouter's Qwen3.8 Flash research checkpoint, converted to a compact expert
format for AMD Strix Halo and the native CIRU v5 runtime. **The original Orca
version remains available through its existing download parts and assembly
helper.** The v5.0 GGUF is an additional, versioned choice. Both versions use Orca's matching Q8 multi-token prediction (MTP) draft,
the required paged token-embedding files, and the matching Orca vision projector.

> [!IMPORTANT]
> The compact expert format needs the matching CIRU runtime. Stock llama.cpp
> and Hugging Face hosted inference do not load it. Keep the required `ple/`
> token-embedding files on fast local storage.

## One slot by default; optional multi-slot serving

Runtime **v5.5.0** retains the v5 model and runtime features, adding the
corrected concurrent MTP/attention paths and post-image memory-pressure fix.
The existing v5.0 weights are unchanged.
The default remains **one slot, 262,144 context tokens, MTP4**, with a
16,384-token batch and an **8,192-token microbatch**.

After rebuilding the matching source as described in [INSTALL.md](INSTALL.md),
opt into two slots by adding `PARALLEL_SLOTS=2` to the launch command:

```bash
PARALLEL_SLOTS=2 MODEL_VARIANT=Orca MODEL_RELEASE=5.0 ENABLE_UI=0 \
  MODEL_DIR="$MODEL_DIR" RUNTIME_DIR="$RUNTIME_DIR" \
  bash "$RUNTIME_DIR/scripts/ciru/run-server.sh"
```

Two slots share **one 262,144-token context pool**, use microbatch 2,048 and
four retained checkpoints per slot. A request can borrow most of the pool
while its peer is idle; both requests cannot each consume the full pool at
the same time. **Multi-slot mode has a performance penalty:** it disables
the single-slot MTP draft-head shortlist. MTP itself remains enabled.
The matched IU4 test measured **47.00 to 42.20 tok/s (-10.21%)** for one
active request when the second slot was configured but idle. Both test arms
used microbatch 2,048; this is not a comparison of the launcher defaults.

The completed IU4 226,712-token, two-image replay finished without swap or
OOM. Its minimum available RAM improved from **7.788 to 13.162 GiB** with
the fenced single-copy input path at microbatch 8,192. This addresses the
demonstrated replay workspace pressure; it is not a claim that all possible
image-memory failures or an idle leak have been eliminated.

The rebuilt runtime also passed **eight image checks across IU4 and Orca,
each in one- and two-slot modes** on Ciru. Orca's separate 226,712-token
two-image replay completed with **13.490 GiB minimum available RAM**,
no swap and no memory-guard event. This confirms the shared correction is
installed and exercised for both variants; there is no matched Orca
before/after memory-gain claim. See the [release check record](benchmarks/release-20261006/validation.json).

See [multi-slot instructions, sweeps and machine/settings details](CONCURRENCY.md)
for the 256K pool comparison, the separate 1/2/4-request diagnostic, optional
GDN prefill results, memory budgets and validation limits. The historical
IU4 measurements must not be read as Orca benchmark results. Existing v5.0
and v5.0.1 benchmark and vision records below remain preserved.

## Earlier vision update in v5.0.1

The v5.0.1 runtime enables image input by default with the matching
projector and keeps MTP depth 4. It restores the QSA attention
position-gap correction needed when image tokens are followed by
speculative decoding. Model, draft and projector weights are unchanged.
Install the new source package and rebuild the runtime; replacing only
the launcher does not apply this fix. Existing v5 users can apply the
[small source update](INSTALL.md#update-an-existing-v50-build) without
rebuilding GPU kernels. Set `ENABLE_VISION=0` for a
text-only launch. See [vision setup and validation](vision/README.md).

On Radeon 8060S, Orca passed six functional checks with its
default MTP depth 4: two separate image descriptions, image
follow-up, two images in one request, a thinking-mode image request,
and a text regression. These are synthetic-image smoke tests, not a
broad vision benchmark. [Validation record](vision/validation-v5.0.1.json).
The speed and memory tables below remain the v5.0 text-only measurements.
Loading the vision projector adds encoder memory; those tables do not
measure the new image-enabled default.

## What changed

The v5.0 GGUF is **72,112,333,504 bytes (67.16 GiB)**, saving **6.785 GiB**
on disk relative to the original Orca GGUF. The measured GPU-visible memory
saving after loading was **6.88 GiB** with the same serving settings.

### How the compact format works

Each routed expert weight is stored as a four-bit integer code. Groups of
16 weights share an index into a table of scales and offsets, with a scale
for each output row. Including those tables and scales, the expert format
averages **4.5175 bits per weight**. This figure describes the routed expert
weights, not the entire model or its auxiliary files.

For short batches, the runtime uses Strix Halo's packed four-bit integer dot
instructions (IU4). For long prompt processing, it decodes compact weights
into temporary 16-bit floating-point (BF16) tiles and uses fused matrix kernels. The weights stay
compact in memory; the complete model is not expanded to BF16. This lets
prefill and generation use different arithmetic while sharing one small
weight file. The complete-model measurements below include both paths and
are not an isolated measure of IU4 acceleration.

### What was kept from Orca

The nonexpert weights, tokenizer and model metadata are retained from the
original Orca conversion. Expert down projections were fitted from Orca's
BF16 weights using Orca calibration activations. The expert gate/up source
shards were byte-identical to the Qwen source used for CIRU Flash v5, so their
already-fitted compact weights were reused after SHA-256 verification. This
preserves the Orca checkpoint's changed expert weights.

The required `ple/` files contain a large per-layer token-embedding lookup
table inherited from the Qwen FP8 release. These n-gram/hash embeddings help
the model represent token sequences. The runtime reads needed portions from
SSD, avoiding a full copy in RAM. They are separate from the routed expert
matrices stored in the GGUF. Keep the matching files on fast local NVMe;
both Orca versions share them.

## Prefill in the native v5 runtime

The measured v5.0 model processed a fresh **64,000-token prompt at
1,251.39 tok/s** and a fresh **256,000-token prompt at 1,002.96 tok/s**.
The second 64K observation was 1,267.40 tok/s. These measure full-prefix
ingest from zero cached tokens; each prefix setup generated one token.
A GPU trace confirmed use of the compact expert matrix kernels, dense Q5
projections, token-embedding convolution and chunked Gated DeltaNet kernels.

## Weight-format comparison within v5

**Both columns below use the v5 runtime.** The retained-weight column is not
the original Orca runtime deployment. These measurements isolate the storage
format tradeoff and do not measure the runtime upgrade from v4.4.1.

Both files ran on the same AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`)
system with the same native v5 runtime, one slot, 262,144-token context,
16-bit (F16) attention cache, 16,384-token batch, Orca MTP depth 4, and matching Orca draft.
Each row is one served observation: append 2,048 prompt tokens after the
listed cached depth, then generate 512 tokens. PP here is append processing and TG
is generation tokens per second, calculated as 512 outputs divided by the
server generation duration. Model loading is excluded. The two 64K passes
used opposite model order. The detailed report explains the small counting
convention difference from the Flash tables.

| Cached depth | Pass | Retained Orca PP | Orca v5.0 PP | Retained Orca TG | Orca v5.0 TG |
| ---: | --- | ---: | ---: | ---: | ---: |
| 64,000 | Initial | 926.0 | 780.0 | 57.7 | 59.2 |
| 64,000 | Reversed | 906.5 | 903.8 | 58.2 | 58.7 |
| 256,000 | Initial | 667.0 | 655.2 | 54.6 | 56.5 |

The initial 64K v5.0 prefill result was lower, but the reversed pass was
nearly tied. The short diagnostic GPU trace also did not reproduce a compact
expert slowdown. At 256K, prefill is close in its one pass. These few runs do
not establish a stable prefill or generation speed difference. The 512
generated token IDs matched exactly in every pair. Complete request wall time
was **11.15 / 11.35 seconds** in the initial 64K pass, **11.11 / 11.05** in
the reversed pass, and **12.67 / 12.41** at 256K for retained Orca / v5.0.
[Full protocol and results](BENCHMARKS-v5.md) explain the model storage
difference between these runs.

The independent objective panel used sixteen fixed 512-token text windows,
scoring 2,048 tail positions. Against the retained Orca output, v5.0 achieved
**95.21% strict top-1 agreement** and **0.047966 mean KL divergence**. Label
negative log-likelihood was **0.710105 / 0.716220** for retained Orca /
v5.0, corresponding to PPL **2.03421 / 2.04668**. This is a short-text
diagnostic. A full-precision Orca logit reference was not available, so
old-to-new agreement is not BF16 agreement. Orca v5.0 task scores have
not been rerun. The separate v5.0.1 image checks do not extend these text metrics.

GPU-visible unified-memory use after loading was **94.19 / 87.31
GiB** for retained Orca / v5.0. Shared-memory GPU and process readings overlap
and should not be added. A fast NVMe drive and 128 GiB unified memory were
used in these measurements.

## More precise projections during speculative decoding

Orca's MTP draft proposes up to four tokens at a time, which the main model
checks together with the next target token. The included runtime keeps
activations in 32-bit floating point for selected dense projections during
these five-token verification batches. On a separate
2,048-position teacher-forced panel, tail perplexity improved from **2.0595
to 2.0516**, while next-token label accuracy changed from **83.50% to 83.20%**
(six fewer correct tokens). This is a small, mixed accuracy tradeoff; there is
no full-precision Orca reference. The same fixed 64K-depth served request
measured **59.04 / 59.79 generation tok/s** with the update off/on and identical
512-token output. Prefill was effectively unchanged. One pair does not
establish a repeatable speed gain. This dense-projection change is independent
of IU4 expert acceleration. [Method and scope](BENCHMARKS-v5.md).

## Choose a version

| File | Purpose | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `Qwen3.8-Flash-CIRU-STRIX-Orca.gguf` | Original Orca, assembled from retained parts | 79,397,818,912 | `c2596055edc45003be20028fa9755d4cdbb46bf18a1035285a88729c8878451e` |
| `Qwen3.8-Flash-CIRU-STRIX-Orca-v5.0.gguf` | Compact v5.0 | 72,112,333,504 | `4512ebcc2b867fc5d34575c418f050fdaff40a850dfed14e71d33dcc586b57fd` |
| `mtp/Qwen3.8-Flash-CIRU-STRIX-Orca-MTP-Q8_0.gguf` | Shared Orca draft | 4,135,893,440 | `0af1741e45b930fb96a55f3ec8c701e59be5a807b8191a33bc2f643700a88f0e` |

The v5.0 target model is provided as a complete `.gguf` file. Download it
with the matching Orca MTP draft and required `ple/` token-embedding files;
the vision projector is included for image input by default. The original [`parts.json`](parts.json),
[`assemble.py`](assemble.py), and original download parts remain available
for the retained Orca version.

Hugging Face counts downloads of actual GGUF files automatically. The complete
v5.0 GGUF therefore uses the Hub's normal download-counting mechanism; the
older byte-split `.part` layout did not provide a GGUF download. This counter
measures file requests rather than unique users or successful installations.
See [Hugging Face's download-counting rules](https://huggingface.co/docs/hub/models-download-stats#how-are-gguf-files-handled).

To use the retained release, follow the
[original v4.4.1 Orca card](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-Orca/blob/8a40c7e9d72f73eb20b802ed26a7ce92c1f351a5/README.md):
download its original parts, run `assemble.py`, and use the corresponding
runtime and launcher. The v5.0 GGUF, launcher, and source archive have separate
filenames, preserving the original installation instructions and files.

## Download and run v5.5.0

Download the model, its matching draft, the embedding-table files, the
matching vision projector, and the complete runtime source. v5.0 is a normal GGUF download; no file assembly is
needed. Existing verified draft and `ple/` files can be reused.

```bash
python3 -m venv .venv-hf
.venv-hf/bin/pip install --upgrade huggingface_hub
.venv-hf/bin/hf download jcbtc/Qwen3.8-Flash-CIRU-STRIX-Orca \
  --revision v5.5.0 \
  Qwen3.8-Flash-CIRU-STRIX-Orca-v5.0.gguf \
  mtp/Qwen3.8-Flash-CIRU-STRIX-Orca-MTP-Q8_0.gguf \
  ple/ple.payload.bin ple/ple.manifest.json ple/ple.scale.bf16 \
  vision/mmproj-Qwen3.8-Flash-Orca-F16.mmproj \
  source-tree-orca-v5.5.0.tar.gz run-server-v5.5.0.sh INSTALL.md CONCURRENCY.md checksums-v5.5.0.sha256 \
  --local-dir ./orca-model
cd orca-model
sha256sum -c checksums-v5.5.0.sha256
```

On Ubuntu/Debian, install `python3-venv` first if `venv` is unavailable.
Follow **[INSTALL.md](INSTALL.md)** to build the included source and matching
AMD runtime, start the server, and send a chat or tool request. The guide
includes driver checks, the pinned SDK/runtime dependencies, context and
memory settings, thinking controls, and troubleshooting. A new installation
does not require the earlier CIRU package or an unpublished GitHub tag.

Use 128 GiB unified memory and fast local NVMe for the measured settings.
The required model files take about 120 GiB; allow additional space for the
SDK and build. The default server is local at `http://127.0.0.1:8080`, with
one slot, 262,144-token context, F16 attention cache, 16,384-token batches and 8,192-token microbatches,
and Orca MTP depth four. The matching build was measured on NixOS; the
Ubuntu/Debian guide is a source-build route, not a claim of measured equal
speed on every distribution.

The download includes the matching vision projector, and the v5.5.0
launcher enables it by default. Text benchmarks do not measure image handling.
See [vision/README.md](vision/README.md) for the separate image test scope.

## Historical Orca results

These results use the original Orca weights and earlier runtimes. They were
**not rerun on v5.0**, and the different task selections and settings prevent
a direct speed comparison with the deep-context v5.0 measurements above.

| Release | Test | Recorded result |
| --- | --- | --- |
| v3 | HumanEval 0–19 | 20/20 base and 20/20 extended |
| v4.3 | HermesAgent 20, two passes | Mean scores 92.5 / 98.5 out of 100; 17/20 / 19/20 full-score tasks |
| v4.4 | HumanEval 0–9, thinking off | 10/10 base and 10/10 extended; 61.57 pooled generation tok/s |
| v4.4 | HermesAgent 20, one pass | Mean 95.25/100; 18/20 full-score tasks |

The historical Hermes scorer missed some instruction-following failures,
including storing a prohibited instruction outside the file it checked.
An earlier trace also retried a deployment after permission was denied.
The scores therefore need to be read alongside the trace review. See the
[benchmark history and limitations](BENCHMARKS-v5.md#historical-task-results)
and the [original v4.4.1 model card](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-Orca/blob/8a40c7e9d72f73eb20b802ed26a7ce92c1f351a5/README.md).

## Source and credits

The source checkpoint is
[OrcaRouter's Qwen3.8 Flash at revision 8336e613](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored/tree/8336e613ea508b13c2159bd0f68965d97a606b95).
[Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) made the base model.
CIRU/Crown (`jcbtc`, `ciru-ai`) converted the Orca checkpoint, fitted the
compact expert weights, built the native runtime, and measured this package.
The runtime builds on [llama.cpp/ggml](https://github.com/ggml-org/llama.cpp),
[pwilkin's Strix Halo work](https://github.com/pwilkin/strix-halo),
[Halogen](https://github.com/peonist-ai/halogen-flash-server), and
[Gufo](https://github.com/gufo-org/gufo). Its long-context implementation
also incorporates ideas from [HaloBox](https://github.com/halo-box/strix-llama.cpp).
AMD provides ROCm/TheRock. See the source archive's notices for
licenses and contributor provenance, and this package's
[NOTICE.md](NOTICE.md). This community release is independent
of Qwen, OrcaRouter, and AMD.

The upstream describes its checkpoint as refusal-removed. This package
retains that provenance and does not add a safety alignment layer. Consult
the [upstream model card](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored)
for intended research use. The source checkpoint's
[Qwen Community License](LICENSE) is preserved.
