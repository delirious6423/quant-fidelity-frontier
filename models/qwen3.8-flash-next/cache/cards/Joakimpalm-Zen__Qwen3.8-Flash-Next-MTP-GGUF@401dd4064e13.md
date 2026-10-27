---
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- speculative-decoding
- mtp
- qwen4exp
- xyntetik
- xyntetik-runner
---

# Qwen3.8-Flash-Next MTP head, GGUF companion

<!-- BEGIN xyntetik-status -->
> **Release: built to be downloaded and served with [Xyntetik Runner](https://github.com/Joakimpalm-Zen/xyntetik-runner).**
> - **Measured status against the parent:** Experimental: the qwen4exp architecture is admitted with its publisher float32 comparison still unmeasured (owed, recorded as such in [coverage.json](https://github.com/Joakimpalm-Zen/xyntetik-runner/blob/main/tests/compatibility/coverage.json)), and the MTP head file runs with ISTA-DASLab's 2-part trunk (`--mtp-file`), not beside the 4-part set in `trunk-with-head/`, whose header already declares its predictor block (`--mtp-file` refuses a second head); what is measured is agreement with another engine, not fidelity: on that 2-part trunk, Runner on the CPU against llama.cpp b11433 agreed on the top-1 token at 472 of 480 positions (the eight misses are near-ties) and on 64 of 64 greedy tokens (2026-10-06). The head's drafts are accepted 69-82% of the time in llama.cpp; the text is the trunk's either way.
> - **Parent model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at revision `de4b8e4d43b9` for the head; the trunk is [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (IQ3_S)
> - **Evidence:** [qwen4exp admission evidence](https://github.com/Joakimpalm-Zen/xyntetik-runner/blob/main/docs/qwen4exp-admission-evidence/README.md) (the engine comparison above), the measured tables on this card
> - **Runner compatibility:** Runner 1.1.1 or later loads the head beside the 2-part trunk with `--mtp-file`, and Runner 1.1.0 or later drafts from the head inside the 4-part set with `--mtp`; llama.cpp b11433 loads both with `--spec-type draft-mtp`. Metal has no path for qwen4exp's Gated DeltaNet blocks or for MTP drafting.
> <!-- BEGIN xyntetik-validated -->
> - **Validation status (Runner support tiers):** each file's row in the Runner roster says which tier it is in, what is re-checked on a release and the date and version of the last run.
>   - `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00001-of-00004.gguf` (4 parts): [roster row](https://github.com/Joakimpalm-Zen/xyntetik-runner/blob/main/docs/model-support.md#qwen3.8-flash-next-gsq-rco-iq3_s-with-mtp-head)
>   - `Qwen3.8-Flash-Next-MTP-Q8_0.gguf`: [roster row](https://github.com/Joakimpalm-Zen/xyntetik-runner/blob/main/docs/model-support.md#qwen3.8-flash-next-mtp-q8_0)
> <!-- END xyntetik-validated -->
<!-- END xyntetik-status -->

**What this is and is not.** This is the multi-token-prediction (MTP, "NextN") draft head of
Qwen3.8-Flash-Next as one GGUF file, built from the publisher's BF16 checkpoint. It is not a
model and it does not run on its own: it is loaded beside the trunk GGUF and proposes draft
tokens that the trunk verifies, so the generated text is the trunk's and only the speed
changes. It was built for the ISTA-DASLab GSQ-RCO **IQ3_S** trunk
([ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
part 1 sha256 recorded in the evidence below), which ships without the head. It is a
draft head of the base model's weights, no fine-tuning, no calibration.

| File | Size | sha256 |
|---|---|---|
| `Qwen3.8-Flash-Next-MTP-Q8_0.gguf` | 4,165,605,056 bytes | `cd2f837a63f0b6ea1d951976e161fc40648932ffaf59194aed3557175638000b` |

## Measured

Same box, same trunk file, same prompts, greedy, 128 new tokens, two runs each
(the one-to-one quant-recovery study, 2026-10-07; engine-reported figures).

| Engine | Setting | 512-token prompt decode | 4K prompt decode | Drafts accepted |
|---|---|---|---|---|
| llama.cpp b11433, CPU only, 24 threads | plain | 13.1, 13.2 tok/s | 12.5, 12.9 tok/s | |
| llama.cpp b11433, CPU only, 24 threads | `--spec-type draft-mtp` with this file | 18.3, 20.4 tok/s | 18.9, 17.0 tok/s | 69-82% |

On a 24 GB MIG slice of a datacenter GPU (CUDA, `--fit on -c 8192 --parallel 2`, the same launch as the plain rows):

| llama.cpp b11433, CUDA | plain | with this head | Drafts accepted |
|---|---|---|---|
| 512-token prompt, decode tok/s | 25.5, 27.2 | 39.0, 34.9 | 67-72% |
| 4K prompt, decode tok/s | 26.7, 25.2 | 37.1, 38.9 | 61-69% |
| two concurrent 512-token requests, decode tok/s each | 20.5 and 20.5 | 26.3 and 25.7 | 69-74% |
| prefill tok/s | 151-222 (512), 285 (4K) | 227-229 (512), 275-279 (4K) | |

The full tables are in the study's report, linked from the Runner repository when it lands.
Known from the same measurements: in llama.cpp the drafted run is not byte-identical to the
plain run (the 4K prompt's text diverged once). In Xyntetik Runner 1.1.1 the head runs on the
CPU and on CUDA (`--mtp-file`), drafts accepted 4.1 to 5.3 of 5 per round, and the greedy text
is byte-identical to plain decoding within one expert split; on a 24 GB device split of this
model it is still slower than plain decoding (IQ3_S: 12.0 against 13.8 tok/s on the same split,
CPU only: 7.7 against 9.3) because the verify tile runs 23 to 30 expert layers on the host one
row at a time and a tile's rows share no expert at 10-of-512 routing. The head pays where the
device holds the trunk; on this slice treat it as a correctness feature. Measured 2026-10-08,
Runner 1.1.1, one MIG 1g.24gb slice; the head's block takes the VRAM of one expert layer under
`--cpu-moe auto`, so compare against a baseline pinned to the same `--cpu-moe N`.

## Use

llama.cpp (b11433 or newer with qwen4exp MTP support):

```
llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  -md Qwen3.8-Flash-Next-MTP-Q8_0.gguf --spec-type draft-mtp --fit on
```

Xyntetik Runner 1.1.1 or newer:

```
runner -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --mtp-file Qwen3.8-Flash-Next-MTP-Q8_0.gguf --gpu off
```

## How it was built

`scripts/make-mtp-companion.py` in the [Xyntetik Runner](https://github.com/Joakimpalm-Zen/xyntetik-runner)
repository, from 33 tensors of `Qwen/Qwen3.8-Flash-Next` at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` fetched by HTTP byte range: the 31 `mtp.*` tensors,
`model.language_model.embed_tokens.weight` and `lm_head.weight` (so the file also loads alone as
a draft model). The layout is llama.cpp's MTP-only export for this architecture: `block_count`
49 with `nextn_predict_layers` 1, the block at `blk.48.*`, the head at `blk.48.nextn.*`,
`eh_proj = [fc_embedding | fc_hidden]`, the fused `gate_up_proj` split gate-first, every RMS-norm
gamma stored +1 as the converter does for this family, the hyperparameters and tokenizer copied
from the trunk's header byte for byte. Storage: norms F32; hyper-connection, router and indexer
tensors BF16 as published; attention, expert, shared-expert, embedding and LM-head matrices Q8_0.
The header carries the provenance under `xyntetik.mtp.*`: source repo, revision, the 33 source
tensor names and their sha256 as fetched, and the builder.

The head's acceptance rate on the real model (69-82% of drafts accepted, in line with the
publisher's own MTP figures) is the check that the construction is right; a wrong head drafts
nothing useful.

## The trunk with the head in one set (`trunk-with-head/`)

For anyone who wants one model directory instead of a trunk from one repository and a head from
another: `trunk-with-head/` holds a 4-part split GGUF that is ISTA-DASLab's IQ3_S trunk with its
header extended for the head (`block_count` 49, `nextn_predict_layers` 1, one more compress ratio, the
split counts) and this head as the last part; the trunk's tensor bytes are untouched (its 54.8 GB first
file is split in two for the hub's 50 GB file limit). Tested: Runner 1.1.0 `--mtp` on part 1 drafts
from the head inside the set, text byte-identical to plain decoding and to the original trunk;
llama.cpp b11433 loads it with `--spec-type draft-mtp` and no `-md` (62 of 99 drafts accepted on
the CPU-only check). Part sha256 in `trunk-with-head/SHA256SUMS`. The trunk is ISTA-DASLab's
work (Apache-2.0); the whole set is a derivative under the Qwen Community License 1.0.

```
llama-server -m trunk-with-head/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00001-of-00004.gguf --spec-type draft-mtp --fit on
runner -m trunk-with-head/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00001-of-00004.gguf --mtp --gpu off
```

## License and attribution

The weights are Qwen's, redistributed as a derivative under the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
with this notice: built from Qwen/Qwen3.8-Flash-Next by Xyntetik (Zenova AB), 2026-10-07.
The trunk this companion was built for is ISTA-DASLab's quantization, published under
Apache-2.0 by its authors; it is not included here.

<!-- xyntetik-program -->
## About Xyntetik

This repository is published by [Xyntetik](https://xyntetik.com), the Zenova AB program for running capable language models on hardware you own and proving what they did. Its released engine is [Xyntetik Runner](https://github.com/Joakimpalm-Zen/xyntetik-runner): a free, single-binary GGUF engine in C for CPU, CUDA and Metal that serves, scores and trains through the quantized file it deploys and records every run as a receipt. The measurements behind the program's claims are collected at [xyntetik.com/evidence](https://xyntetik.com/evidence/). The program's organization on Hugging Face is [Xyntetik](https://huggingface.co/Xyntetik).
