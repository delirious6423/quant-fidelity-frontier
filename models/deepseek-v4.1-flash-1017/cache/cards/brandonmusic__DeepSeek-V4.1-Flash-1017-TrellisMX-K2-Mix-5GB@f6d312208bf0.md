---
license: mit
license_link: LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
pipeline_tag: text-generation
tags:
- trellismx
- trellis-quantization
- mixed-precision
- 2-bit
- 3-bit
- 4-bit
- moe
- deepseek
- deepseek-v4.1
- engram
model-index:
- name: DeepSeek-V4.1-Flash-1017-TrellisMX-K2-Mix-5GB
  results:
  - task:
      type: text-generation
      name: Distribution fidelity (KL divergence vs the official checkpoint)
    dataset:
      type: brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits
      name: evaluation set (128 windows x 8,192 tokens, folder windows-8192/; the same tokens as 512 fresh sequences x 2,048 tokens, folder cuts-2048/)
      revision: 68379d81e1d7c4fd7569b8499adc3ff7c7179362
    metrics:
    - type: kl_divergence
      name: Mean tokenwise KLD (reference || candidate), routed experts in serving formats with the official FP8 Engram tables, 64 of the 512 fresh 2,048-token sequences (the subset the widths were chosen on), nats
      value: 0.13782991
      args:
        units: nats
        higher_is_better: false
        direction: reference_to_candidate
        estimator: full_vocabulary_fp64
    - type: top1_agreement
      name: Top-1 agreement with the reference, same path, the same 64 sequences
      value: 0.884519
x_fidelity:
  spec_version: fidelity-provenance/v1
  role: quant
  reference_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
  reference_revision: 1bae0e1088993b9b82de08f59de5be5a34e42170
  fidelity_dataset: brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits@68379d81e1d7c4fd7569b8499adc3ff7c7179362 (cuts-2048/, windows-8192/)
  registry: null
  lane: reference_path
  scope: routed experts as shipped (each expert at its own width, K2/K3/K4) on E4M3 operands with E4M3 activations; Engram rows from the official FP8 tables, not the K4-coded Engram tables this checkpoint ships (see Fidelity); every other tensor as released; emulated in the reference implementation
  head_bits: 16
  measured_by: first-party
---

# DeepSeek-V4.1-Flash-1017 TrellisMX K2 Mix 5GB

A TrellisMX build of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(revision `1bae0e1088993b9b82de08f59de5be5a34e42170`) for two 96 GB RTX PRO 6000 Blackwell cards at tensor parallel 2.
It is the [K2 build](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-TrellisMX-K2) with 1,131 of its 15,360
routed experts (7.4%) stored at a wider trellis rate: 1,111 at 3 bits and 20 at 4 bits, the rest at 2 bits. The wider
experts are the same encodes as the [K3](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-TrellisMX-K3)
and [K4](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-TrellisMX-K4) builds, re-laid out for tensor
parallel 2; nothing was re-encoded.

The encodes, the allocation and every measurement on this card ran on an 8x NVIDIA B200 node generously donated by
[Lambda](https://lambda.ai) ([@LambdaAPI](https://x.com/LambdaAPI)). Thank you, Lambda.

Built with the SM120 vLLM image by Local Inference Lab and Luke Alonso, the b12x kernels by Luke Alonso with
Martin Vit and other contributors, and Local Inference Lab's CSF scale format; TrellisMX was inspired by QTIP and
EXL3, and its FP8 trellis decode by Luke Alonso's KQuant/QSRT. Full credits: [Credits](#credits) and `CREDITS.md`.

Each expert's width was chosen to lower the measured KL divergence most per byte, with Brandon M. Music's ShapleyMCG
allocation, under a byte budget sized to keep about 2 million tokens of KV cache on the two cards. The routed experts
take 5.08 GB more than the K2 build's (143.79 GB against 138.71 GB). A sibling build with a smaller budget, about
3 million tokens of KV and 2.87 GB more than K2, is
[DeepSeek-V4.1-Flash-1017-TrellisMX-K2-Mix](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-TrellisMX-K2-Mix).

Released under the MIT License, the same license as the base model (`LICENSE`).

**Status: research build; only KLD has been measured, on 64 sequences.** The weights are complete; the reference-path
KLD below covers 64 of the 512 evaluation sequences, not the full protocol.
A test serving image that reads this layout is published as
`verdictai/deepseek-v41-flash-trellismx:k2mix-fp8-tp2-test-20261020`
(`sha256:a72e633c8779c6903c4039dc2397e8a55db4a1669f422c9f6516de97e04c0c60`). It passes layer-level GPU tests (each
width's routes bit-identical to the single-rate kernels, FULL CUDA graph replay with changing routes, the K2 path
byte-identical to the published K2 image) but has not been booted as a whole model, so this checkpoint's serving,
speed and KV capacity are not measured. The published K2 image and its `serve.sh` serve the single-rate K2, K3 and K4
layouts only.

## What is stored

| Component | Stored as |
|---|---|
| Routed experts (40 layers x 384) | TrellisMX trellis codes per expert: 14,229 at 2 bits, 1,111 at 3 bits, 20 at 4 bits, decoded to FP8 (E4M3) tensor-core operands, UE8M0 scales per 32 values in Local Inference Lab's lossless CSF format, pre-sliced for tensor parallel 2 |
| Engram tables | The K2 build's tables: TrellisMX 4-bit rows with UE8M0 scales, in host RAM |
| Everything else | As in the K2 build (and the official checkpoint) |

Per-layer widths are in `trellismx/manifest.json` (`expert_bits`, 384 entries per layer). Each
`trellismx/tp2/layer-LLL-rank-R.safetensors` sidecar holds one group of tensors per width present in that layer
(`k2.*`, `k3.*`, `k4.*`), each with its ascending global `expert_ids`, in the same per-expert format as the single-rate
builds.

## Fidelity

KL(official checkpoint || this build) in nats, over the full vocabulary (129,280 ids), in FP64, at every next-token row
(row r predicts token r+1). Teacher: the official checkpoint run in DeepSeek's reference implementation, FP32 logits,
published in `brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits` (revision `68379d81`), folders `cuts-2048/` and
`windows-8192/`.

Measured on 64 of the 512 sequences of 2,048 tokens (the 128 evaluation windows cut into 512 fresh sequences; these 64
are cuts 0, 8, 16, ..., 504, which span every domain), 2,047 rows each, 131,008 rows. The full 512-sequence and
8,192-token-window measurement was not run for this build.

| Path | Mean KLD [BCa 95%, per sequence] | Top-1 |
|---|---|---|
| Routed experts in serving formats (E4M3 operands, E4M3 activations), official FP8 Engram tables | **0.1378** [0.1120, 0.1749] | **88.45%** |

On the same 64 sequences and the same path: the K2 build 0.1687 (top-1 87.05%) and the
[K2 Mix](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-TrellisMX-K2-Mix) build 0.1466 (87.87%). The
intervals overlap because they include the spread between sequences; the builds were scored on the same sequences.

These are the sequences the widths were chosen on, so the figure is not an independent test. For the K2 Mix build the
gain over K2 measured 13.1% on these 64 sequences and 8.6% on all 512.

These figures score the routed experts with the official FP8 Engram tables. This checkpoint ships the K2 build's
4-bit Engram tables; on the K2 build that difference measured 0.1439 against 0.1429 on all 512 sequences, so expect
this checkpoint to sit about 0.001 nats above the figure.

How it was measured: the base model's layers run in DeepSeek's reference implementation, layer by layer on 4x NVIDIA
B200 at model parallel 4, one sequence per forward, with each routed expert reproduced bit for bit at its stored width.
The native SM120 kernel path has not been measured.

## Files

- `config.json`, `model-000NN-of-00048.safetensors`, `model.safetensors.index.json`, tokenizer files and `encoding/`:
  as in the K2 build.
- `engram/`: the K2 build's 4-bit Engram tables.
- `trellismx/manifest.json` (schema `ds41-trellismx-mix-tp2.v1`) and `trellismx/tp2/`: the routed experts.
- `SHA256SUMS`: every file's SHA-256.

## Teacher logits

The reference logits behind the KLD figures are published in
[brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits](https://huggingface.co/datasets/brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits)
(revision `68379d81`).

## Credits

- **Compute: Lambda.** The allocation and every measurement on this card ran on an 8x NVIDIA B200 node that
  [Lambda](https://lambda.ai) donated for this work. Our sincere thanks to Lambda
  ([@LambdaAPI](https://x.com/LambdaAPI), https://x.com/LambdaAPI) for supporting independent open-model research: this
  build would not exist without that compute.
- **Base model.** DeepSeek-V4.1-Flash-1017 by DeepSeek (`deepseek-ai/DeepSeek-V4.1-Flash-1017`), MIT License (`LICENSE`).
- **Runtime: Local Inference Lab and Luke Alonso.** The SM120 runtime this checkpoint targets is built on the vLLM image
  by Local Inference Lab and Luke Alonso (`ghcr.io/local-inference-lab/vllm`, https://github.com/local-inference-lab), a
  fork of vLLM (Apache-2.0, vLLM contributors), and the open-source projects that image bundles, each under its own
  license: FlashInfer, LMCache, InstantTensor, xgrammar and fastokens.
- **Kernels: b12x by Luke Alonso.** The MoE, attention and GEMM kernels of that runtime come from b12x
  (https://github.com/lukealonso/b12x, mirrored at https://github.com/local-inference-lab/b12x) by Luke Alonso, with
  Martin Vit and other contributors (Apache-2.0).
- **CSF encoder: Local Inference Lab.** The lossless compressed scale format (CSF) the expert scales are stored in, and
  its encoder and decoder, are Local Inference Lab's work, shipped in Local Inference Lab's FlashInfer.
- **TrellisMX: QTIP and EXL3.** TrellisMX, Brandon M. Music's trellis-coded format, was inspired by QTIP (Albert Tseng,
  Qingyao Sun, David Hou and Christopher De Sa, "QTIP: Quantization with Trellises and Incoherence Processing",
  NeurIPS 2024) and by ExLlamaV3 and its EXL3 format by turboderp and contributors
  (https://github.com/turboderp-org/exllamav3, MIT License).
- **Codec changes: QSRT and Luke Alonso.** Decoding the trellis directly into FP8 (E4M3) tensor-core operands was
  inspired by Luke Alonso's KQuant/QSRT work and the b12x W4A8 trellis path, which decodes the QSRT trellis to E4M3.
- **Allocation: ShapleyMCG by Brandon M. Music.** The per-expert widths were chosen with Brandon M. Music's
  ShapleyMCG allocation: measured KL divergence attributed to groups of experts with Shapley values, a per-expert
  calibration, and an exact byte knapsack.
- **Methods.** The per-expert allocation attributes measured KL divergence with Shapley values (Lloyd S. Shapley, "A
  Value for n-Person Games", 1953) estimated over sampled permutations, and chooses widths with an exact
  multiple-choice knapsack over byte costs. The encoder's Hessian-based error feedback follows GPTQ (Elias Frantar,
  Saleh Ashkboos, Torsten Hoefler and Dan Alistarh, ICLR 2023). The incoherence processing and rotations follow QuIP
  (Jerry Chee, Yaohui Cai, Volodymyr Kuleshov and Christopher De Sa, NeurIPS 2023), QuIP# (Albert Tseng, Jerry Chee,
  Qingyao Sun, Volodymyr Kuleshov and Christopher De Sa, ICML 2024), QuaRot (Saleh Ashkboos et al., NeurIPS 2024) and
  SpinQuant (Zechun Liu et al., 2024). The UE8M0 block scales follow the OCP Microscaling (MX) Formats Specification
  (Bita Darvish Rouhani et al., 2023).
- **Evaluation data.** The evaluation windows behind the KLD figures were built from public datasets, each under the
  license on its own dataset card: NVIDIA's Nemotron-SFT-Agentic-v2, Nemotron-SFT-Math-v4, Nemotron-SFT-Science-v2
  (CC-BY-SA-4.0), Nemotron-SFT-Competitive-Programming-v2, Nemotron-SFT-Instruction-Following-Chat-v3,
  Nemotron-SFT-Multilingual-v2, Nemotron-SFT-Long-Context-Transduction-v1 (OpenMDW-1.1), Nemotron-SFT-Safety-v2,
  Nemotron-Pretraining-Legal-v1 (CC-BY-4.0) and Nemotron-Pretraining-Dataset-sample (Nemotron-CC text; NVIDIA Open Data
  License Agreement); and Nebius's SWE-agent-trajectories (CC-BY-4.0). The windows' sources are listed in the K2
  build's `evaluation/eval-windows.json`.
