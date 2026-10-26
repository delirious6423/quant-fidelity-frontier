---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags: [deepseek, moe, reap, expert-pruning, gguf, llama.cpp]
---

# DeepSeek-V4.1-Flash-1017-REAP192-GGUF

deepseek-ai/DeepSeek-V4.1-Flash-1017 with 192 of 384 routed experts kept in every decoder layer, as a GGUF for a llama.cpp fork, built to run on one machine with 128 GB of memory and an NVMe drive.

**Text only for now.** The vision tower is not in these files (the converter routes it to an mmproj file, which is not built yet) and the llama.cpp fork has no V4.1 vision path. For images use the MXFP4 master in an engine with V4.1 vision support. An mmproj and vision support for the fork are planned and will be added to this repository.

## Run it in 128 GB: weights in memory, Engram on NVMe

DeepSeek-V4.1-Flash has two Engram tables (n-gram embeddings, about 101 GB each). A token reads only 24 rows of each, and the row ids depend only on the token ids. So the tables stay on NVMe and are read with `O_DIRECT` while the GPU runs; only the pruned experts and the dense path (85.6 for `L3`) are loaded into memory. Upstream llama.cpp cannot do this yet: use the fork below (upstream PR #28696 plus six patches, one of which is the NVMe row store).

You need 128 GB of RAM or unified memory, an NVIDIA GPU with CUDA (measured only on a Jetson AGX Thor, sm_110; `build.sh` targets it by default, set `CUDA_ARCH` for another GPU), and about 210 GB free on a local NVMe drive for the row files (not USB, not a network share). Building the row files also needs the base model's two Engram files once (about 200 GB more, on any disk; delete them afterwards).

```bash
# 1. the fork (pinned upstream PR head + patches) and the row-file builder
git clone https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor && cd deepseek-v4.1-flash-reap-thor
llamacpp/setup.sh
llamacpp/build.sh                     # Jetson AGX Thor (sm_110); any other NVIDIA GPU: CUDA_ARCH=<sm, e.g. 89> llamacpp/build.sh
python3 -m venv .venv && .venv/bin/pip install numpy huggingface_hub

# 2. this GGUF (L3)
.venv/bin/hf download patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF --include "*-L3-*.gguf" --local-dir gguf

# 3. the two Engram files of the base model, plus its config and index (see "Engram" below for sizes and sha256)
.venv/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 config.json model.safetensors.index.json model-00047-of-00048.safetensors \
    model-00048-of-00048.safetensors --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir base

# 4. build and verify the row files on NVMe (a 200 GB sequential copy; the base files can be deleted afterwards)
export V41_SRC=$PWD/base V41_WORK=$PWD/work
.venv/bin/python ops/s09_engram_rowfile.py build  --dst /nvme/v41rows
.venv/bin/python ops/s09_engram_rowfile.py verify --dst /nvme/v41rows

# 5. serve (OpenAI-compatible)
LLAMA_ENGRAM_ROWFILE_DIR=/nvme/v41rows llamacpp/llama.cpp/build-thor/bin/llama-server \
    -m gguf/DeepSeek-V4.1-Flash-1017-REAP192-L3-00001-of-00003.gguf -ngl 999 -c 8192 --jinja --port 8080
```

`-c 8192` is the context the release gate served it at. **Context limit of this build: 16,384 tokens per sequence.** Past that, DeepSeek-V4.1 restricts its later layers to a set of candidate blocks that this fork does not implement yet, so the output would drift from the model without any error; the fork clamps the context there instead (patch 0006). The native CUDA runner implements the candidate blocks and is not limited this way. On a machine whose GPU memory is separate from system RAM, lower `-ngl` until the GPU part fits; the rest runs on the CPU.



Conversion: from the published REAP MXFP4 master (keep-set `1ace9b33134a2ca74174952593682e94edbf3c3348b4ac505ff59597a05d49ec`), then requantized; details below.

## Levels

Each level was scored separately on its own GGUF: a paired, teacher-forced comparison of the served file against the cached unpruned teacher on the pre-registered confirm split of the held-out documents, which no pruning, healing or quantization decision ever looked at (180224 scored tokens). Intervals are ninety-five percent document-cluster bootstrap intervals.

| level | expert bpw | size (GiB) | kl32 (95% CI) | top-1 agreement | dNLL | GPU layers in the eval |
|---|---|---|---|---|---|---|
| L3 | 2.46 | 85.6 | 0.5076 [0.3224, 0.7337] | 0.7725 [0.7116, 0.8287] | 0.3399 [0.1758, 0.5434] | all |

Every level was scored fully offloaded to the GPU.

## Precision

The routed experts carry the per-level mix above (the protected layers sit higher). The dense path **is** requantized by the GGUF writer:

| tensors | precision |
|---|---|
| attention, compressor, indexer, engram_wkv | Q8_0 |
| shared expert, token_embd, output | Q6_K |
| router (ffn_gate_inp) | BF16 (as stored) |
| hyper-connection | F32 |
| norms, sinks, biases | F32 |
| engram tables | not in the GGUF (O_DIRECT row files, llamacpp/ENGRAM_ROWFILE.md) |

The DSpark/MTP layers are not included (`--no-mtp`).

No heal gain is applied: the files carry no `deepseek41.reap.routed_gain` key, so the routed experts' summed output is used exactly as converted from the unhealed MXFP4 master.

## The keep-set

## How it was evaluated

Every number below comes from `s10/final/results.json` of the build pipeline. It is a paired, teacher-forced comparison against the unpruned deepseek-ai/DeepSeek-V4.1-Flash-1017 at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, run in its own MXFP4/FP8 numerics. The comparison uses 185565 scored tokens from 26 held-out documents in the confirmation split. That split was never used to choose the model. The intervals are ninety-five percent document-cluster bootstrap intervals.

- **kl32** is the mean over tokens of the KL divergence on the teacher's top-32 support.
- **top-1 agreement** is the share of tokens where this model's argmax equals the teacher's.
- **dNLL** is this model's next-token NLL minus the teacher's.

The pass/fail thresholds were pre-registered before any result existed (`ops/s10_gates.json`, sha256 `f95b26721124c6a3`). Any threshold amended after a result is listed below with its date, and the result under the original rule is shown beside it.

| arm | kl32 (95% CI) | top-1 agreement | dNLL | confident-flip rate |
|---|---|---|---|---|
| control | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 |
| mxfp4_A | 0.4405 [0.2639, 0.6465] | 0.7980 [0.7431, 0.8481] | 0.2718 [0.1219, 0.4585] | 0.0524 |
| thorA | 0.5194 [0.3261, 0.7423] | 0.7756 [0.7179, 0.8284] | 0.3091 [0.1426, 0.5119] | 0.0619 |
| b128 | 0.8026 [0.5627, 1.0801] | 0.7092 [0.6443, 0.7701] | 0.5469 [0.3271, 0.8010] | 0.0995 |

**Known limitation: multilingual.** Top-1 agreement with the unpruned model on the held-out multilingual documents is lower than in every other domain (confirmation split: mxfp4_A 0.5247, thorA 0.4935, b128 0.4219). The release floor for every other domain is 0.65; for multilingual it was lowered to a collapse guard of 0.4 (2026-10-22, owner decision, before any s10 result). During expert selection, the shipped keep-set scored 0.5233 on the screen split against 0.5592-0.5741 for random keep-sets of the same size. Keep-sets that protected more multilingual experts recovered multilingual only by losing general, finance and science text, so the better all-round keep-set ships. For multilingual use, prefer the unpruned model.

**Gate amended after the result (2026-10-24).** The pre-registered rule required the pruned model to beat a random keep-set of the same size on kl32. Over all domains it did not reach significance (paired difference -0.1014, interval -0.2196 to +0.0400), so this release FAILED its gate as originally registered. The cause is multilingual alone. After seeing that result, the owner amended the rule to exclude multilingual, whose weakness is already documented above. Without it the difference is -0.2144 (interval -0.2633 to -0.1676). Reason: Owner decision AFTER the first s10 final result (confirm split). As registered, mxfp4_A vs random0 paired d_kl32 was -0.1014 [-0.2196, +0.0400]: not significant, entirely because of the multilingual bucket (kl32 1.534 vs random 0.894), the weakness the owner had already accepted on 2026-10-22 with a lowered top-1 floor. In every other bucket the keep-set has roughly half of random's kl32. The as-registered comparison stays in results.json and on the cards beside the amended one; no other threshold changes (thorA top-1 0.78 floor still applies).

**Thor build: top-1 floor waived (2026-10-24, after the result).** The low-bit Thor build scored top-1 0.7756 (interval 0.7179 to 0.8284) against a pre-registered floor of 0.78, so it FAILED that floor; the MXFP4 master scores 0.7980. The owner waived this one clause for the Thor build. Every other clause still applies and passed. Reason: Owner decision after the s10 final result. thorA top-1 0.7756 [0.7179, 0.8283] misses the 0.78 sanity floor (borrowed from GLM REAP50 v2's 0.842) by 0.0044, well inside its interval. It is the only 192-expert build that fits Thor; the bit-exact 128-expert alternative is far worse (top-1 0.7092, paired kl32 +0.283 vs thorA). Requantization costs kl32 +0.079 vs the MXFP4 master (limit 0.25) and top-1 -0.022. Every other clause still applies.

`control` keeps all experts and is re-run through the same evaluation machinery as the others. It measures that machinery, not the model.


## Speculative decoding

Not measured yet. The checkpoint's own DSpark head (the three `mtp` layers) is being refit to this keep-set; this section is filled from that result.


**Release gate:** FAIL on the pre-registered floor (top-1 0.7725 < 0.78); L3 released by owner decision (llama-server smoke passed).

## Engram (required, not included)

The Engram tables are read from NVMe at decode time through row files built locally from the base repository's two Engram files:

- `model-00047-of-00048.safetensors` (101.5 GB, sha256 `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5`)
- `model-00048-of-00048.safetensors` (101.5 GB, sha256 `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd`)

```
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir .
```

The row-file format and build script are documented in the runtime repository (`llamacpp/ENGRAM_ROWFILE.md`).

**License:** MIT. Keep-set `1ace9b33134a2ca74174952593682e94edbf3c3348b4ac505ff59597a05d49ec`; teacher `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e1088993b9b82de08f59de5be5a34e42170`; pre-registered gates `f95b26721124c6a343f41b004b8740bfe6eb22033d77dcc260dceda039082e0a`.
