---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - llama.cpp
  - deepseek
  - deepseek-v4.1-flash
  - moe
  - reap
  - hope
  - expert-pruning
  - imatrix
  - engram
  - mmproj
  - dspark
  - jetson-thor
---

# DeepSeek-V4.1-Flash-1017-REAP192-GGUF

This repository holds [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) pruned to 192 of 384 routed experts per layer, as one GGUF quantization level (**L3**, 85.6 GiB). It also includes a vision projector (mmproj) and the model's own DSpark drafter. The files run on a [llama.cpp branch](https://github.com/patrickbdevaney/llama.cpp/tree/deepseek-v4.1-reap-thor) that reads the 189 GiB Engram tables from NVMe, so they need no memory.

It is aimed at one machine with 128 GB of memory and a local NVMe drive. On a Jetson AGX Thor it decodes at **12.3–12.5 tok/s**, and at 17.3–20.7 tok/s with the DSpark drafter, as measured by the author on 2026-10-30.

> **Release status.** L3 **failed its pre-registered release gate** on top-1 agreement: 0.7725 against a floor of 0.78 (95% CI 0.712–0.829). Its kl32 of 0.508 passes. The owner released it on 2026-10-26 as a near-miss, after its generation smoke test passed. The higher-precision level L2 failed both bars and is not published. The MXFP4 master this was made from was published under a gate amended after its result; see the [master card](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4#release-gates).

## Model overview

| | |
|---|---|
| Base model | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Converted from | [`REAP192-MXFP4`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4) (keep-set `1ace9b33…`, also stored in the GGUF header as `deepseek41.reap.keep_ids`) |
| GGUF architecture | `deepseek41`: 40 blocks, embedding 5120, 64 heads / 1 KV head, key length 512 |
| Routed experts | **192 of 384 per layer**, 6 active, plus 1 shared expert |
| Parameters in the GGUF | 280.05 B (271.79 B routed-expert). Engram tables, MTP layers and vision tower excluded |
| Context | the header says 1,048,576. The branch clamps to **16,384** by default; 32,768 is validated (see [Context length](#context-length)) |
| Routed experts, precision (L3) | 2.46 bpw average: IQ3_XXS on protected layers 0, 1, 2, 8, 14, 20, 37, 38, 39 (27 tensors); IQ2_XS (79), IQ2_XXS (12) and IQ2_S (2) elsewhere |
| Dense path, precision | Q8_0 for attention, compressor, indexer and `engram_wkv` (229 tensors); Q6_K for shared expert, `token_embd` and `output` (122); BF16 router (44); F32 hyper-connections, norms, sinks and biases (529) |
| Overall | 1,044 tensors, 2.62 bpw |
| Size on disk | L3: 91,862,148,960 bytes (**85.55 GiB**) in 3 splits. mmproj: 0.91 GiB. DSpark drafter: 7.43 GiB |
| Heal gain | none (no `deepseek41.reap.routed_gain` key): the routed sum is used as converted from the unhealed master |
| Licence | MIT |

### Files

| file | bytes | what it is |
|---|---:|---|
| `DeepSeek-V4.1-Flash-1017-REAP192-L3-00001-of-00003.gguf` | 44,657,469,792 | L3, split 1 of 3 (pass this one to `-m`) |
| `DeepSeek-V4.1-Flash-1017-REAP192-L3-00002-of-00003.gguf` | 44,568,604,928 | L3, split 2 |
| `DeepSeek-V4.1-Flash-1017-REAP192-L3-00003-of-00003.gguf` | 2,636,074,240 | L3, split 3 |
| `mmproj-DeepSeek-V4.1-Flash-1017-bf16.gguf` | 972,208,160 | vision projector (`clip`, projector type `deepseek41v`): the base model's 32-layer ViT plus aligner, BF16/F32, unpruned |
| `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` | 7,976,903,072 | the base model's own DSpark drafter (`dflash` arch, 3 blocks, 128 MXFP4 experts each, Q8_0 elsewhere), not pruned |
| `gguf_manifest.json` | 3,752 | sha256 of each split, precision table, llama.cpp PR and patch list, evaluation numbers |

The splits exist because Hugging Face limits a file to 50 GB. llama.cpp opens the other two splits by itself when given the first. The header sets `eos_token_id` and `eot_token_id` to 1 (`<｜end▁of▁sentence｜>`) and embeds the V4.1 chat template.

## Which version should I use

| repository | format | size | runtime | measured decode on Thor (2026-10-30) |
|---|---|---:|---|---|
| **REAP192-GGUF** (this repo) | GGUF L3 + mmproj + DSpark | 85.6 GiB | [llama.cpp `deepseek-v4.1-reap-thor`](https://github.com/patrickbdevaney/llama.cpp/tree/deepseek-v4.1-reap-thor) | 12.3–12.5 tok/s; 17.3–20.7 with DSpark |
| [REAP192-Thor](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor) | `thor-v1` (Q2_K/Q3_K experts, FP8 dense, NVFP4 overlay) | 106.9 GiB | [CUDA server](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server) (Jetson AGX Thor only) | 25.3 tok/s plain; 25.1–40.2 with speculation |
| [REAP192-MXFP4](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4) | bit-exact master | 151.6 GiB | none tested | — |

On a Jetson AGX Thor, the Thor build with the CUDA server is about twice as fast and scores slightly higher (top-1 0.7756 against 0.7725, though measured on different token sets). Use this GGUF if you want llama.cpp, or another CUDA GPU with 128 GB of memory (untested).

## Quickstart

These steps follow the branch's [`thor/README.md`](https://github.com/patrickbdevaney/llama.cpp/blob/deepseek-v4.1-reap-thor/thor/README.md). They were run on a Jetson AGX Thor (CUDA 13.0). Other GPUs are untested.

```bash
# 1. llama.cpp branch: upstream PR #28696 plus the V4.1 / Thor commits
git clone -b deepseek-v4.1-reap-thor https://github.com/patrickbdevaney/llama.cpp && cd llama.cpp
thor/build.sh                      # sm_110 by default; CUDA_ARCH=<sm> for another GPU (untested)

# 2. this repository
hf download patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF --local-dir ~/models/DeepSeek-V4.1-Flash-1017-REAP192-GGUF

# 3. Engram row files, built once from base shards 47/48 (see below), at $MODEL_DIR/engram_rows or ENGRAM_DIR

# 4. serve (OpenAI-compatible, http://127.0.0.1:8080)
thor/start.sh                      # text + images, 16K context
DSPARK=1 thor/start.sh             # DSpark speculative decoding (text only)
```

`start.sh` runs `llama-server -ngl 999 -np 1 --jinja`. It refuses row files on removable media, and it refuses to start beside another model-loading process. The settings `MODEL_DIR`, `ENGRAM_DIR`, `CTX`, `VISION`, `DSPARK`, `DRAFT_N`, `P_MIN`, `HOST`, `PORT` and `EXTRA` are environment variables. The direct command is:

```bash
LLAMA_ENGRAM_ROWFILE_DIR=/nvme/engram_rows build-thor/bin/llama-server \
  -m ~/models/DeepSeek-V4.1-Flash-1017-REAP192-GGUF/DeepSeek-V4.1-Flash-1017-REAP192-L3-00001-of-00003.gguf \
  --mmproj ~/models/DeepSeek-V4.1-Flash-1017-REAP192-GGUF/mmproj-DeepSeek-V4.1-Flash-1017-bf16.gguf \
  -c 16384 -np 1 -ngl 999 --jinja --port 8080
# DSpark instead of vision: drop --mmproj and add
#   -md .../dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf -ngld 999 --spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-p-min 0.5
```

### Engram row files (required, not included)

The two Engram tables (layers 1 and 14) are unchanged by pruning and are not in the GGUF. They are read per token with `O_DIRECT` from two `.v41rows` files on a local NVMe drive: 202,758,040,592 bytes (188.8 GiB) together. You build those files once from the base model's shards 47 and 48, using [`ops/s09_engram_rowfile.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_engram_rowfile.py) in the pipeline repository. The format is described in [`llamacpp/ENGRAM_ROWFILE.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/llamacpp/ENGRAM_ROWFILE.md). The same files serve the Thor build.

```bash
git clone https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor && cd deepseek-v4.1-flash-reap-thor
python3 -m venv .venv && .venv/bin/pip install numpy huggingface_hub
.venv/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 config.json model.safetensors.index.json \
    model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
    --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir base     # 2 x 101.5 GB, any disk
export V41_SRC=$PWD/base V41_WORK=$PWD/work
.venv/bin/python ops/s09_engram_rowfile.py build  --dst /nvme/engram_rows
.venv/bin/python ops/s09_engram_rowfile.py verify --dst /nvme/engram_rows --full
```

| base file | bytes | sha256 |
|---|---:|---|
| `model-00047-of-00048.safetensors` | 101,535,150,936 | `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5` |
| `model-00048-of-00048.safetensors` | 101,537,926,640 | `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd` |

Once the row files are built and verified, the base shards can be deleted.

## Evaluation

**Method.** Each published level is scored on its own GGUF, served by llama.cpp (`llamacpp/tools/v41-teacher-kld.cpp`). The comparison is paired and teacher-forced against the cached outputs of the unpruned model. It uses the **22 text rows of the pre-registered confirmation split: 180,224 tokens at 8,192-token context**, all layers on the GPU. No pruning, quantization or release decision looked at this split. Intervals are 95% row (document) bootstrap intervals. The metrics are defined on the [master card](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4#evaluation). Source: `s09/L3/teacher_eval/teacher_kld.json` and `s09/gates.json`, copied into `gguf_manifest.json`.

| level | expert bpw | size | kl32 (95% CI) | top-1 (95% CI) | dNLL (95% CI) | gate (kl32 ≤ 0.8, top-1 ≥ 0.78) |
|---|---:|---:|---|---|---|---|
| **L3** (published) | 2.46 | 85.55 GiB | 0.5076 [0.3224, 0.7337] | 0.7725 [0.7116, 0.8287] | 0.3399 [0.1758, 0.5434] | **FAIL** on top-1; released by owner decision |
| L2 (not published) | 2.85 | — | 1.0097 [0.7601, 1.2672] | 0.7316 [0.6709, 0.7925] | 0.8458 | FAIL on both |

L2 spends more bits on experts than L3 but scored worse on the same tokens. It was not investigated further and is not published. The ladder script also defines L0 (MXFP4 experts copied losslessly, about 152 GiB) and L1 (about 3.4 bpw). Neither was published.

**L3 by domain** (point estimates; each domain has 1 to 5 documents):

| domain | tokens | kl32 | top-1 | dNLL |
|---|---:|---:|---:|---:|
| agentic | 40,960 | 0.2307 | 0.8407 | 0.1061 |
| code | 24,576 | 0.0640 | 0.9692 | 0.0496 |
| finance | 24,576 | 0.4463 | 0.7442 | 0.3045 |
| general | 32,768 | 0.5402 | 0.7489 | 0.1887 |
| math | 8,192 | 0.1429 | 0.9323 | 0.0903 |
| **multilingual** | 24,576 | **1.6375** | **0.5004** | **1.3543** |
| science | 24,576 | 0.4225 | 0.7409 | 0.3259 |

For reference, on the full 26-document confirmation split (185,565 tokens, vision rows included), the MXFP4 master scores kl32 0.4405 and top-1 0.7980, and the Thor build 0.5194 and 0.7756. Those numbers come from a different harness on a different token set, so compare them with L3 only loosely.

**Vision.** The mmproj encoder was golden-tested on CPU and CUDA against the reference torch ViT and aligner in f32. Every token row matched at least as closely as the reference's own bf16 path. In one end-to-end spot check, on llama.cpp's `tools/mtmd/test-1.jpeg` newspaper page, the model read the masthead, the date, the headline and both column headlines correctly. Image-domain quality after pruning was not benchmarked through llama.cpp.

**DSpark drafter.** All 78 tensors are byte-identical to an independent conversion of the reference drafter, and every reference metadata key is present (`s09/dspark2/source_check.json`).

**Not measured.** There are no generative benchmarks, no tool-calling accuracy and no quality past 32K tokens.

## Hardware & memory

The numbers below were measured by the author on one Jetson AGX Thor (128 GB unified memory, CUDA 13.0) on 2026-10-30. They are reported in the branch's [`thor/README.md`](https://github.com/patrickbdevaney/llama.cpp/blob/deepseek-v4.1-reap-thor/thor/README.md). The setup was `thor/start.sh` at its defaults (16K context, all layers on the GPU), with greedy sampling and one stream over `/v1/chat/completions`.

| | `thor/start.sh` | `DSPARK=1 thor/start.sh` |
|---|---:|---:|
| decode, 256-token chat answers (code / explanation / math) | 12.3–12.5 tok/s | 17.3–20.7 tok/s (1.38–1.68×) |
| DSpark acceptance (up to 3 drafted per step, `P_MIN` 0.5) | | 73–81% |
| prefill, 1.8K-token prompt | 159 tok/s | |
| prefill, 5.9K-token prompt | 147 tok/s | |
| image + question (215 prompt tokens) | 84 tok/s prompt, 12.4 tok/s answer | |
| unified memory still free while serving | 25 GiB of 122 | 16 GiB |

The release smoke test on 2026-10-26 (`ops/logs/s09_smoke_L3.log`, 8K context) decoded 52 tokens at 11.86 tok/s. With DSpark, every emitted token is still the target model's greedy pick. The batched verify sums in a different order, though, so output can differ from plain decoding at near-ties.

### Context length

Past 16,384 tokens, V4.1 restricts later layers to a two-level candidate-block selection (2048 blocks of 8). The published GGUF predates the metadata keys that switch this on in the branch. So by default the branch clamps the context to 16,384. `CTX=32768 thor/start.sh` passes the keys with `--override-kv`. On a 32,767-token held-out row at positions 16,384–32,766, the mask was within noise of running without it: KL +0.00055 [−0.00001, +0.00122]. A 32K-token prompt takes about 14.6 minutes to process. 64K and beyond have not been validated.

## How it was made

1. **Pruning.** The experts were pruned with HOPE (`hope_sum_pf0`), 192 of 384 per layer, as bit-exact MXFP4 surgery. See the [master card](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4#how-it-was-made) for the calibration corpus and the selection.
2. **Conversion.** `llamacpp/convert_v41_reap.py` produced a lossless source GGUF from the master: experts as bit-exact GGML MXFP4, FP8 weights as exact BF16, and hyper-connection, norm and sink tensors as F32. Engram tables and MTP layers were left out.
3. **Quantization.** `ops/s09_gguf_ladder.py` quantized L3 once, from that source and not from another level, with `llama-quantize`. It used an importance matrix built from the pipeline's own teacher-pass activations, remapped to the 192-expert keep-set: 240 entries, 552 chunks, held-out rows excluded. The keep-set fingerprint is checked at conversion, at imatrix load and on the written splits.
4. **Branch.** The branch is upstream PR #28696 (head `2076412598e9`) plus commits for: F32 hyper-connections; reader layers reusing their index source's top-k; the Engram `O_DIRECT` row store; the REAP heal-gain hook (unused here); DSpark drafter export; the 16K clamp; the candidate-block mask; a faster indexer transpose; and the `deepseek41v` vision projector. The release was built at `92031e6a`; the branch head is `cf4d51ba2`.

## Limitations

- **The release gate failed as registered** (top-1 0.7725 < 0.78). This level was released as a near-miss by owner decision.
- **Multilingual quality is much lower than in the unpruned model** (top-1 0.50 on L3). For non-English use, prefer the unpruned model.
- **Context** is 16K by default and 32K validated. Long prompts are slow on this branch.
- **Tool calling.** The calibration corpus carried no tool-call turns, and tool-call formatting is not evaluated.
- **Vision** was spot-checked, not benchmarked. A low-resolution 531×229 crop made the model loop. DSpark was measured on text only.
- Only measured on a Jetson AGX Thor. Partial GPU offload and other GPUs are untested.

## Changelog / corrections

- 2026-10-26: L3 published (s12 card generated from s09/s10 artifacts).
- 2026-10-30: vision mmproj and DSpark drafter added. Measured Thor serving numbers and the 32K context validation added.
- 2026-11-06: card rewritten and verified against repo files and build artifacts; a 2026-10-26 automated rewrite contained errors. Corrections: the published ladder is **only L3**. L0, L1 and L2 are not in this repository, and L2 failed its gate. The memory breakdown and the "87.30 GiB resident" figure in the 2026-10-26 rewrite could not be traced to any log and were removed. The Engram row-file commands now match the script (`--dst`, with `V41_SRC`); the rewrite's `--src/--out` flags do not exist. The calibration long-context rows are 32K–64K tokens, not 32K–128K.

## License & attribution

MIT, the same as the base model. The weights are derived from [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) by [DeepSeek-AI](https://huggingface.co/deepseek-ai). The runtime builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) and its PR #28696 (`deepseek41` architecture).

## Citation

```bibtex
@misc{devaney2026dsv41reap192gguf,
  title        = {DeepSeek-V4.1-Flash-1017-REAP192-GGUF: 192-of-384 expert-pruned DeepSeek-V4.1-Flash-1017 for llama.cpp with Engram on NVMe},
  author       = {Patrick Devaney},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF}}
}
```
