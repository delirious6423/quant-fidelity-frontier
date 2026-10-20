---
license: mit
license_link: LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
pipeline_tag: text-generation
tags:
- trellismx
- trellis-quantization
- 4-bit
- moe
- deepseek
- deepseek-v4.1
- engram
model-index:
- name: DeepSeek-V4.1-Flash-1017-TrellisMX-K4
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
      value: 0.049785757853216248
      args:
        units: nats
        higher_is_better: false
        direction: reference_to_candidate
        estimator: full_vocabulary_fp64
    - type: top1_agreement
      name: Top-1 agreement with the reference, serving formats, 2,048-token sequences
      value: 0.93353459330727895
    - type: kl_divergence
      name: Mean tokenwise KLD (reference || candidate), serving formats, 128 windows of 8,192 tokens, nats
      value: 0.10972108157831723
    - type: top1_agreement
      name: Top-1 agreement with the reference, serving formats, 8,192-token windows
      value: 0.92305102398974481
x_fidelity:
  spec_version: fidelity-provenance/v1
  role: quant
  reference_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
  reference_revision: 1bae0e1088993b9b82de08f59de5be5a34e42170
  fidelity_dataset: brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits@68379d81e1d7c4fd7569b8499adc3ff7c7179362 (cuts-2048/, windows-8192/)
  registry: null
  lane: reference_path
  scope: full_checkpoint (K4 routed experts on E4M3 operands with E4M3 activations, the original FP8 Engram tables and every other tensor as released; emulated in the reference implementation)
  head_bits: 16
  measured_by: first-party
---

# DeepSeek-V4.1-Flash-1017 TrellisMX K4

A 4-bit TrellisMX build of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(revision `1bae0e1088993b9b82de08f59de5be5a34e42170`), laid out for four 96 GB RTX PRO 6000 Blackwell cards at
tensor parallel 4, with the two Engram tables kept in their original FP8 form (about 189 GiB) in host RAM.

The encode and every measurement on this card ran on an 8x NVIDIA B200 node generously donated by
[Lambda](https://lambda.ai) ([@LambdaAPI](https://x.com/LambdaAPI)). Thank you, Lambda.

Built with the SM120 vLLM image by Local Inference Lab and Luke Alonso, the b12x kernels by Luke Alonso with
Martin Vit and other contributors, and Local Inference Lab's CSF scale format; TrellisMX was inspired by QTIP and
EXL3, and its FP8 trellis decode by Luke Alonso's KQuant/QSRT. Full credits: [Credits](#credits) and `CREDITS.md`.

Released under the MIT License, the same license as the base model (`LICENSE`).

**Status: research build.** The weights and the reference-path measurements below are complete. The serving image
`verdictai/deepseek-v41-flash-trellismx:k4-fp8-tp4-20261019` runs this checkpoint at tensor parallel 4 on four RTX PRO
6000 Blackwell 96 GB cards with DSpark speculative decoding and full CUDA graphs; it was booted and benchmarked on that
hardware on 2026-10-20 (Speed, below). In that testing, two of three startups of this image stopped during the engine's
warmup with a CUDA illegal-memory-access error on one card. Both ran while an `nvidia-smi` loop polled the cards every
five seconds; the clean one did not, nor did two clean startups with CUDA debugging variables set. The cause is not
confirmed; if a startup stops this way, start it again.
`serving/` holds its `docker-compose.yml`, `serve.sh` and `SERVING.md`.

## Run it

You need four RTX PRO 6000 Blackwell 96 GB cards (SM120) with PCIe peer-to-peer between them, about 190 GiB of free
host RAM for the FP8 Engram tables, an NVIDIA driver with CUDA 13.4 support, the NVIDIA Container Toolkit, docker compose
v2 and about 497 GB of disk.

```bash
docker pull verdictai/deepseek-v41-flash-trellismx:k4-fp8-tp4-20261019
hf download brandonmusic/DeepSeek-V4.1-Flash-1017-TrellisMX-K4 --local-dir /models/DeepSeek-V4.1-Flash-1017-TrellisMX-K4
cd /models/DeepSeek-V4.1-Flash-1017-TrellisMX-K4/serving
export CHECKPOINT_ROOT=/models/DeepSeek-V4.1-Flash-1017-TrellisMX-K4
bash serve.sh
```

`serve.sh` reads the checkpoint's manifest, picks tensor parallel 4 and GPUs 0,1,2,3 (`DS41_GPUS` changes them), checks
the free host RAM and starts the container. The API is OpenAI-compatible on port 8010, model name
`deepseek-v41-flash-trellismx-k4`:

```bash
curl -s http://127.0.0.1:8010/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41-flash-trellismx-k4","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```

`serving/SERVING.md` has every setting and its default.

## Speed on four RTX PRO 6000 Blackwell cards

Measured on 2026-10-20 with Local Inference Lab's llm-inference-bench v0.4.29
(https://github.com/local-inference-lab/llm-inference-bench) on four RTX PRO 6000 Blackwell Workstation Edition cards
(power limits 300, 300, 300 and 350 W) at tensor parallel 4, with the serving image
`verdictai/deepseek-v41-flash-trellismx:k4-fp8-tp4-20261019`
(`sha256:d35737d826e0d2a00d7f603e5a76f220103e95312c78dd5580ef862c322a5bfa`) at its defaults: full CUDA graphs, DSpark
speculative decoding with 7 draft tokens, FP8 KV cache. 30 seconds per decode cell.

Aggregate decode throughput, tokens per second, by concurrent requests:

| Context | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| 0 | 168.6 | 241.4 | 296.3 | 437.5 |
| 8K | 166.0 | 244.4 | 294.0 | 431.8 |
| 16K | 171.5 | 244.1 | 298.9 | 430.1 |
| 32K | 167.1 | 236.4 | 295.1 | 427.9 |

DSpark accepted 1.8 to 2.4 tokens per step on the bench's text.

Prefill, one request:

| Prompt | Tokens per second | Time to first token |
|---|---:|---:|
| 8K | 13,010 | 0.63 s |
| 16K | 12,885 | 1.26 s |
| 32K | 12,637 | 2.55 s |
| 64K | 12,244 | 5.26 s |
| 128K | 11,599 | 11.1 s |

The bench's built-in reasoning checks, ten runs each, all ten at once (concurrency 10), with no token cap: lavd-test 10
of 10 exact (about 20,000 tokens per answer, slowest 260 s), estonia 10 of 10 pass (about 1,400 tokens per answer),
hotel-lights 10 of 10 exact (about 16,400 tokens per answer, slowest 227 s). No answer hit a length limit.

KV cache at the default GPU memory utilization of 0.95: 1,427,646 tokens (1.36 requests at the full 1,048,576-token
context), set by the two cards that hold the wider 640-column expert slices, at 1.47 GiB of KV each. The boot also
prints the `KV_CACHE_MEMORY` value that fully uses those cards, about 4.9 GiB; `serving/SERVING.md` explains how to pin
it.

On this serving path, as a check of the native SM120 kernels, the mean negative log-likelihood of the realized next
tokens over 8 of the 512 evaluation sequences (16,376 rows) was 2.2565 nats, against 2.2155 for this checkpoint on the
reference path (Fidelity, below) and 2.2196 for the official model.

## What is quantized

| Component | Stored as |
|---|---|
| Routed experts (40 layers x 384, 543.6B weights) | 4-bit TrellisMX (K4), decoded to E4M3 operands, E4M3 activations (W4A8); UE8M0 scales per 32 inputs stored CSF-compressed (lossless); pre-sliced for tensor parallel 4 in whole 128-column blocks (640, 640, 512 and 512 of the 2,304 intermediate columns). 274.6 GB, about 4.04 bits per weight |
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
| Serving formats: experts on E4M3 operands with E4M3 activations, the official Engram tables | **0.0498** [0.0449, 0.0557] | 0.0038 | 0.0655 | 0.90 | 3.90 | **93.35%** [93.31, 93.40] |
| Weights only: the same weights, activations unquantized | 0.0482 [0.0434, 0.0541] | 0.0036 | 0.0629 | 0.88 | 3.66 | 93.51% [93.46, 93.56] |

Against the K3 build on the same 512 sequences (serving formats 0.0664 [0.0607, 0.0732], top-1 91.74%), this build
lowers the mean KLD 1.33 times, and against the K2 build (0.1439 [0.1346, 0.1545], top-1 86.82%) 2.9 times. Its
Engram tables are the official FP8 ones, so its whole difference from the official model comes from the 4-bit experts.

Full 8,192-token windows (8,191 rows each, 1,048,448 rows):

| Path | Mean KLD [BCa 95%] | Median | p90 | p99 | p99.9 | Top-1 [Wilson 95%] |
|---|---|---:|---:|---:|---:|---|
| Serving formats | **0.1097** [0.0830, 0.1441] | 0.0022 | 0.1205 | 2.41 | 7.55 | **92.31%** [92.25, 92.36] |
| Weights only | 0.1047 [0.0791, 0.1374] | 0.0021 | 0.1140 | 2.30 | 7.10 | 92.51% [92.46, 92.56] |

By domain (serving formats, 2,048-token sequences): agentic 0.154 [0.133, 0.177], SWE agent 0.060 [0.049, 0.086], math 0.026 [0.022, 0.031], science 0.032 [0.027, 0.037], competitive programming 0.034 [0.026, 0.053], instruction following and chat 0.024 [0.022, 0.026], multilingual 0.025 [0.023, 0.029], long-context transduction 0.010 [0.008, 0.012], safety 0.036 [0.032, 0.041], web text 0.040 [0.037, 0.045], legal 0.034 [0.030, 0.039]. As with the K2 and K3 builds, KLD rises at 8,192 tokens: the median falls (0.0022 against 0.0038) but the tail grows (p99 2.41 against 0.90).

Checks: the teacher scored against itself is exactly 0 on every window; the reference model's canary window, run at the
start and end of every session, is bit-identical; on every window the teacher's top-1 agrees with the realized token
inside the [0.20, 0.995] band; the GPU scoring kernel matches a CPU FP64 reference to within 2e-12 on sampled rows of
every window. The evaluation windows share at most 5% of their content 13-grams with the data the build was calibrated
on, and every accepted conversation shared at most 2% with it.

How it was measured: the base model's layers run in DeepSeek's reference implementation, layer by layer on 4x NVIDIA
B200 at model parallel 4, one window per forward, with this checkpoint's experts and Engram rows reproduced bit for bit in
that path (the Engram rows are gathered on the CPU, as the serving path does from host RAM). The KLD of the native SM120
kernel path has not been measured (a served log-likelihood check is under Speed). Every figure with its interval: `evaluation/report.json`; per-sequence and per-window statistics:
`evaluation/summary-D1.json` and `summary-D2.json`; per-token records: `evaluation/records/`; window sources and
token-id SHA-256: `evaluation/eval-windows.json`.

## Files

- `model-000NN-of-00048.safetensors` + `model.safetensors.index.json`: every tensor that is not a routed expert of
  layers 0-39, including the two FP8 Engram tables, byte-identical to the base release.
- `trellismx/tp4/layer-LLL-rank-R.safetensors`: the 4-bit routed experts of layer LLL for tensor-parallel rank R (each
  rank holds all 384 experts' slice of the intermediate dimension: 640 columns on ranks 0 and 1, 512 on ranks 2 and 3),
  their CSF-compressed UE8M0 scales and the per-layer vectors the runtime applies with the experts.
- `trellismx/manifest.json`: the format description and the SHA-256 and size of every expert file.
- `config.json` (the base config with `quantization_config` set for this format), tokenizer and `encoding/` files from
  the base release, `LICENSE`, `SHA256SUMS` (every file).
- `evaluation/`: `report.json` (every figure above with its interval), `summary-D1.json` and `summary-D2.json` (per
  sequence and per window), `records/D1` and `records/D2` (per-token records) and `eval-windows.json` (every window's
  sources, token-id SHA-256 and overlap screen). D1 = serving formats, D2 = weights only.
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
