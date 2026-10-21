---
license: mit
license_link: LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
pipeline_tag: text-generation
tags:
- trellismx
- trellis-quantization
- 3-bit
- moe
- deepseek
- deepseek-v4.1
- engram
model-index:
- name: DeepSeek-V4.1-Flash-1017-TrellisMX-K3
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
      name: Mean tokenwise KLD (reference || candidate), serving formats, 512 fresh 2,048-token sequences, nats
      value: 0.066373248800852741
      args:
        units: nats
        higher_is_better: false
        direction: reference_to_candidate
        estimator: full_vocabulary_fp64
    - type: top1_agreement
      name: Top-1 agreement with the reference, serving formats, 2,048-token sequences
      value: 0.91737145823155841
    - type: kl_divergence
      name: Mean tokenwise KLD (reference || candidate), serving formats, 128 windows of 8,192 tokens, nats
      value: 0.12997564902502426
    - type: top1_agreement
      name: Top-1 agreement with the reference, serving formats, 8,192-token windows
      value: 0.9102721355756318
x_fidelity:
  spec_version: fidelity-provenance/v1
  role: quant
  reference_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
  reference_revision: 1bae0e1088993b9b82de08f59de5be5a34e42170
  fidelity_dataset: brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits@68379d81e1d7c4fd7569b8499adc3ff7c7179362 (cuts-2048/, windows-8192/)
  registry: null
  lane: reference_path
  scope: full_checkpoint (K3 routed experts on E4M3 operands with E4M3 activations, the original FP8 Engram tables and every other tensor as released; emulated in the reference implementation)
  head_bits: 16
  measured_by: first-party
---

# DeepSeek-V4.1-Flash-1017 TrellisMX K3

A 3-bit TrellisMX build of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(revision `1bae0e1088993b9b82de08f59de5be5a34e42170`), laid out for three 96 GB RTX PRO 6000 Blackwell cards at
tensor parallel 3, with the two Engram tables kept in their original FP8 form (about 189 GiB) in host RAM.

The encode and every measurement on this card ran on an 8x NVIDIA B200 node generously donated by
[Lambda](https://lambda.ai) ([@LambdaAPI](https://x.com/LambdaAPI)). Thank you, Lambda.

Built with the SM120 vLLM image by Local Inference Lab and Luke Alonso, the b12x kernels by Luke Alonso with
Martin Vit and other contributors, and Local Inference Lab's CSF scale format; TrellisMX was inspired by QTIP and
EXL3, and its FP8 trellis decode by Luke Alonso's KQuant/QSRT. Full credits: [Credits](#credits) and `CREDITS.md`.

Released under the MIT License, the same license as the base model (`LICENSE`).

**Status: research build.** The weights and the reference-path measurements below are complete. The serving image
`verdictai/deepseek-v41-flash-trellismx:k3-fp8-tp3-20261019` runs this checkpoint at tensor parallel 3 on three RTX PRO
6000 Blackwell 96 GB cards with DSpark speculative decoding and full CUDA graphs. It shares its runtime with the K2
image, which was booted and benchmarked on SM120 hardware on 2026-10-19 after fixes to the startup path and to the
expert kernel's token routing. This image carries the same fixes and has passed its CPU checks, but this checkpoint has
not yet been booted at tensor parallel 3, so treat the first run as a test. The tag was rebuilt with these fixes on
2026-10-19: if you pulled it earlier that day, pull it again. `serving/` holds its `docker-compose.yml`, `serve.sh` and
`SERVING.md`.

## Run it

You need three RTX PRO 6000 Blackwell 96 GB cards (SM120) with PCIe peer-to-peer between them, about 190 GiB of free
host RAM for the FP8 Engram tables, an NVIDIA driver with CUDA 13.4 support, the NVIDIA Container Toolkit, docker compose
v2 and about 430 GB of disk.

```bash
docker pull verdictai/deepseek-v41-flash-trellismx:k3-fp8-tp3-20261019
hf download brandonmusic/DeepSeek-V4.1-Flash-1017-TrellisMX-K3 --local-dir /models/DeepSeek-V4.1-Flash-1017-TrellisMX-K3
cd /models/DeepSeek-V4.1-Flash-1017-TrellisMX-K3/serving
export CHECKPOINT_ROOT=/models/DeepSeek-V4.1-Flash-1017-TrellisMX-K3
bash serve.sh
```

`serve.sh` reads the checkpoint's manifest, picks tensor parallel 3 and GPUs 0,1,2 (`DS41_GPUS` changes them), checks
the free host RAM and starts the container. The API is OpenAI-compatible on port 8010, model name
`deepseek-v41-flash-trellismx-k3`:

```bash
curl -s http://127.0.0.1:8010/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41-flash-trellismx-k3","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```

`serving/SERVING.md` has every setting and its default.

## What is quantized

| Component | Stored as |
|---|---|
| Routed experts (40 layers x 384, 543.6B weights) | 3-bit TrellisMX (K3), decoded to E4M3 operands, E4M3 activations (W3A8); UE8M0 scales per 32 inputs stored CSF-compressed (lossless); pre-sliced for tensor parallel 3. 206.5 GB, about 3.04 bits per weight |
| Engram tables (layers 1 and 14, 768,022,850 rows x 256) | as released: FP8 (E4M3) rows with an E8M0 scale per 32, byte-identical to the base checkpoint; about 189 GiB, held in host RAM |
| Everything else (attention, shared experts, router, norms, embeddings, lm_head, the three DSpark/MTP layers) | as released, byte-identical to the base checkpoint |

## Fidelity

KL(official checkpoint || this build) in nats, over the full vocabulary (129,280 ids), in FP64, at every next-token row
(row r predicts token r+1). Teacher: the official checkpoint run in DeepSeek's reference implementation, FP32 logits,
published in `brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits` (revision `68379d81`), folders `cuts-2048/` and `windows-8192/`.
Intervals are window-block bootstrap (BCa 95%) and Wilson for top-1.

Headline, 2,048-token sequences: the 128 evaluation windows of 8,192 tokens cut into 512 sequences of 2,048 tokens, each
run as a fresh sequence from position 0 (2,047 rows each, 1,048,064 rows):

| Path | Mean KLD [BCa 95%] | Median | p90 | p99 | p99.9 | Top-1 [Wilson 95%] |
|---|---|---:|---:|---:|---:|---|
| Serving formats: experts on E4M3 operands with E4M3 activations, the official Engram tables | **0.0664** [0.0607, 0.0732] | 0.0061 | 0.1065 | 1.14 | 4.23 | **91.74%** [91.68, 91.79] |
| Weights only: the same weights, activations unquantized | 0.0645 [0.0589, 0.0712] | 0.0060 | 0.1038 | 1.11 | 4.11 | 91.92% [91.87, 91.97] |

Against the K2 build on the same 512 sequences (serving formats 0.1439 [0.1346, 0.1545], top-1 86.82%), this build
lowers the mean KLD 2.2 times. Its Engram tables are the official FP8 ones, so its whole difference from the official
model comes from the 3-bit experts.

Full 8,192-token windows (8,191 rows each, 1,048,448 rows):

| Path | Mean KLD [BCa 95%] | Median | p90 | p99 | p99.9 | Top-1 [Wilson 95%] |
|---|---|---:|---:|---:|---:|---|
| Serving formats | **0.1300** [0.1005, 0.1675] | 0.0038 | 0.1786 | 2.67 | 7.87 | **91.03%** [90.97, 91.08] |
| Weights only | 0.1256 [0.0969, 0.1622] | 0.0036 | 0.1703 | 2.61 | 7.56 | 91.22% [91.17, 91.28] |

By domain (serving formats, 2,048-token sequences): agentic 0.189 [0.166, 0.213], SWE agent 0.084 [0.070, 0.112], math 0.034 [0.030, 0.040], science 0.044 [0.038, 0.051], competitive programming 0.045 [0.036, 0.063], instruction following and chat 0.034 [0.032, 0.036], multilingual 0.037 [0.034, 0.041], long-context transduction 0.011 [0.009, 0.014], safety 0.054 [0.049, 0.062], web text 0.067 [0.062, 0.074], legal 0.057 [0.050, 0.065]. As with the K2 build, KLD rises at 8,192 tokens: the median falls (0.0038 against 0.0061) but the tail grows (p99 2.67 against 1.14).

Checks: the teacher scored against itself is exactly 0 on every window; the reference model's canary window, run at the
start and end of every session, is bit-identical; on every window the teacher's top-1 agrees with the realized token
inside the [0.20, 0.995] band; the GPU scoring kernel matches a CPU FP64 reference to within 2e-12 on sampled rows of
every window. The evaluation windows share at most 5% of their content 13-grams with the data the build was calibrated
on, and every accepted conversation shared at most 2% with it.

How it was measured: the base model's layers run in DeepSeek's reference implementation, layer by layer on 4x NVIDIA
B200 at model parallel 4, one window per forward, with this checkpoint's experts and Engram rows reproduced bit for bit in
that path (the Engram rows are gathered on the CPU, as the serving path does from host RAM). The native SM120 kernel path
has not been measured. Every figure with its interval: `evaluation/report.json`; per-sequence and per-window statistics:
`evaluation/summary-B1.json` and `summary-B2.json`; per-token records: `evaluation/records/`; window sources and
token-id SHA-256: `evaluation/eval-windows.json`.

## Files

- `model-000NN-of-00048.safetensors` + `model.safetensors.index.json`: every tensor that is not a routed expert of
  layers 0-39, including the two FP8 Engram tables, byte-identical to the base release.
- `trellismx/tp3/layer-LLL-rank-R.safetensors`: the 3-bit routed experts of layer LLL for tensor-parallel rank R (each
  rank holds all 384 experts' 768-wide third of the intermediate dimension), their CSF-compressed UE8M0 scales and the
  per-layer vectors the runtime applies with the experts.
- `trellismx/manifest.json`: the format description and the SHA-256 and size of every expert file.
- `config.json` (the base config with `quantization_config` set for this format), tokenizer and `encoding/` files from
  the base release, `LICENSE`, `SHA256SUMS` (every file).
- `evaluation/`: `report.json` (every figure above with its interval), `summary-B1.json` and `summary-B2.json` (per
  sequence and per window), `records/B1` and `records/B2` (per-token records) and `eval-windows.json` (every window's
  sources, token-id SHA-256 and overlap screen). B1 = serving formats, B2 = weights only.
- `serving/`: `docker-compose.yml`, `serve.sh` and `SERVING.md` for the serving image.

## Teacher logits

The FP32 teacher logits used for the KLD figures are in the dataset
[brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits](https://huggingface.co/datasets/brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits) (revision `68379d81`): folder `cuts-2048/` for the 2,048-token
sequences and `windows-8192/` for the 8,192-token windows, with every window's sources and token-id SHA-256.

## Credits

- **Compute: Lambda.** The encode and every measurement on this card ran on an 8x NVIDIA B200 node that
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
- **Methods.** The encoder's Hessian-based error feedback follows GPTQ (Elias Frantar, Saleh Ashkboos, Torsten Hoefler
  and Dan Alistarh, ICLR 2023). The incoherence processing and rotations follow QuIP (Jerry Chee, Yaohui Cai, Volodymyr
  Kuleshov and Christopher De Sa, NeurIPS 2023), QuIP# (Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov and
  Christopher De Sa, ICML 2024), QuaRot (Saleh Ashkboos et al., NeurIPS 2024) and SpinQuant (Zechun Liu et al., 2024).
  The UE8M0 block scales follow the OCP Microscaling (MX) Formats Specification (Bita Darvish Rouhani et al., 2023).
- **Evaluation data.** The evaluation windows behind the KLD figures were built from public datasets, each under the
  license on its own dataset card: NVIDIA's Nemotron-SFT-Agentic-v2, Nemotron-SFT-Math-v4, Nemotron-SFT-Science-v2
  (CC-BY-SA-4.0), Nemotron-SFT-Competitive-Programming-v2, Nemotron-SFT-Instruction-Following-Chat-v3,
  Nemotron-SFT-Multilingual-v2, Nemotron-SFT-Long-Context-Transduction-v1 (OpenMDW-1.1), Nemotron-SFT-Safety-v2,
  Nemotron-Pretraining-Legal-v1 (CC-BY-4.0) and Nemotron-Pretraining-Dataset-sample (Nemotron-CC text; NVIDIA Open Data
  License Agreement); and Nebius's SWE-agent-trajectories (CC-BY-4.0). Each window's sources are listed in
  `evaluation/eval-windows.json`.
