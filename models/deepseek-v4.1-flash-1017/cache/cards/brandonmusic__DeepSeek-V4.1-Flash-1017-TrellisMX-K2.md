---
license: mit
license_link: LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
pipeline_tag: text-generation
tags:
- trellismx
- trellis-quantization
- 2-bit
- moe
- deepseek
- deepseek-v4.1
- engram
model-index:
- name: DeepSeek-V4.1-Flash-1017-TrellisMX-K2
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
      value: 0.14385943976261922
      args:
        units: nats
        higher_is_better: false
        direction: reference_to_candidate
        estimator: full_vocabulary_fp64
    - type: top1_agreement
      name: Top-1 agreement with the reference, serving formats, 2,048-token sequences
      value: 0.8681549981680508
    - type: kl_divergence
      name: Mean tokenwise KLD (reference || candidate), serving formats, 128 windows of 8,192 tokens, nats
      value: 0.21819613799350493
    - type: top1_agreement
      name: Top-1 agreement with the reference, serving formats, 8,192-token windows
      value: 0.8640609739348065
x_fidelity:
  spec_version: fidelity-provenance/v1
  role: quant
  reference_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
  reference_revision: 1bae0e1088993b9b82de08f59de5be5a34e42170
  fidelity_dataset: brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits@68379d81e1d7c4fd7569b8499adc3ff7c7179362 (cuts-2048/, windows-8192/)
  registry: null
  lane: reference_path
  scope: full_checkpoint (K2 routed experts on E4M3 operands with E4M3 activations, K4 Engram tables, every other tensor as released; emulated in the reference implementation)
  head_bits: 16
  measured_by: first-party
---

# DeepSeek-V4.1-Flash-1017 TrellisMX K2

A 2-bit TrellisMX build of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(revision `1bae0e1088993b9b82de08f59de5be5a34e42170`), laid out for two 96 GB RTX PRO 6000 Blackwell cards at tensor
parallel 2, with the two Engram tables at 4 bits (about 104 GB) meant to stay in host RAM.

The encode and every measurement on this card ran on an 8x NVIDIA B200 node generously donated by
[Lambda](https://lambda.ai) ([@LambdaAPI](https://x.com/LambdaAPI)). Thank you, Lambda.

Built with the SM120 vLLM image by Local Inference Lab and Luke Alonso, the b12x kernels by Luke Alonso with
Martin Vit and other contributors, and Local Inference Lab's CSF scale format; TrellisMX was inspired by QTIP and
EXL3, and its FP8 trellis decode by Luke Alonso's KQuant/QSRT. Full credits: [Credits](#credits) and `CREDITS.md`.

Released under the MIT License, the same license as the base model (`LICENSE`).

**Status: research build.** The weights and the reference-path measurements below are complete. The serving image
`verdictai/deepseek-v41-flash-trellismx:k2-fp8-tp2-20261019` runs this checkpoint at tensor parallel 2 on two RTX PRO
6000 Blackwell 96 GB cards with DSpark speculative decoding and full CUDA graphs; it was booted and benchmarked on that
hardware on 2026-10-19 (Speed, below). The tag was rebuilt that day with fixes to the startup path and to the expert
kernel's token routing: if you pulled it earlier on 2026-10-19, pull it again. `serving/` holds its
`docker-compose.yml`, `serve.sh` and `SERVING.md`.

**2026-10-19:** 43 of the 48 `model-000NN-of-00048.safetensors` shards were rewritten with contiguous tensor offsets.
The first upload padded some tensors to 8-byte boundaries, which the safetensors loader refuses. Every tensor's bytes are
unchanged; `SHA256SUMS` lists the new files. If you downloaded this repository before that date, download it again.

## Run it

You need two RTX PRO 6000 Blackwell 96 GB cards (SM120) with PCIe peer-to-peer between them, about 100 GiB of free host
RAM for the Engram tables, an NVIDIA driver with CUDA 13.4 support, the NVIDIA Container Toolkit, docker compose v2 and
about 262 GB of disk.

```bash
docker pull verdictai/deepseek-v41-flash-trellismx:k2-fp8-tp2-20261019
hf download brandonmusic/DeepSeek-V4.1-Flash-1017-TrellisMX-K2 --local-dir /models/DeepSeek-V4.1-Flash-1017-TrellisMX-K2
cd /models/DeepSeek-V4.1-Flash-1017-TrellisMX-K2/serving
export CHECKPOINT_ROOT=/models/DeepSeek-V4.1-Flash-1017-TrellisMX-K2
bash serve.sh
```

`serve.sh` reads the checkpoint's manifest, picks tensor parallel 2 and GPUs 0,1 (`DS41_GPUS` changes them), checks the
free host RAM and starts the container. The API is OpenAI-compatible on port 8010, model name
`deepseek-v41-flash-trellismx-k2`:

```bash
curl -s http://127.0.0.1:8010/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41-flash-trellismx-k2","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```

`serving/SERVING.md` has every setting and its default.

## Speed on two RTX PRO 6000 Blackwell cards

Measured on 2026-10-19 with Local Inference Lab's llm-inference-bench v0.4.29
(https://github.com/local-inference-lab/llm-inference-bench) on two RTX PRO 6000 Blackwell Workstation Edition cards
(power limits 300 W and 350 W) at tensor parallel 2, with the serving image's defaults: full CUDA graphs, DSpark
speculative decoding with 7 draft tokens, FP8 KV cache. 30 seconds per decode cell.

Aggregate decode throughput, tokens per second, by concurrent requests:

| Context | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| 0 | 120.8 | 161.6 | 197.8 | 299.3 |
| 8K | 119.8 | 164.4 | 205.5 | 297.7 |
| 16K | 123.1 | 165.1 | 200.7 | 291.6 |

DSpark accepted 1.7 to 2.3 tokens per step on the bench's text.

The k2 image was updated later on 2026-10-19 with a faster routed-expert kernel for this checkpoint's shape (the same
arithmetic, with each expert's weights decoded once per call instead of once per routed token). In a same-session
check on the same two cards it raised single-request decode from 121.5 to 147.4 tokens per second and two-request
decode from 164.8 to 221.9 at an empty context; the table above was measured with the earlier kernel, and prefill is
unchanged.

Prefill, one request:

| Prompt | Tokens per second | Time to first token |
|---|---:|---:|
| 8K | 10,154 | 0.81 s |
| 16K | 9,960 | 1.63 s |
| 32K | 9,919 | 3.25 s |
| 64K | 9,703 | 6.63 s |
| 128K | 9,242 | 13.9 s |

The bench's built-in reasoning checks, three runs each at concurrency 3 with no token cap: lavd-test 3 of 3 exact
(about 20,200 tokens per answer), estonia 3 of 3 pass, hotel-lights 3 of 3 exact.

KV cache at the default GPU memory utilization of 0.95: 1.29 GiB per card, 1,248,932 tokens (1.19 requests at the full
1,048,576-token context). The first boot also prints the `KV_CACHE_MEMORY` value that fully uses the cards, about
4.4 GiB per card; `serving/SERVING.md` explains how to pin it.

## What is quantized

| Component | Stored as |
|---|---|
| Routed experts (40 layers x 384, 543.6B weights) | 2-bit TrellisMX (K2), decoded to E4M3 operands, E4M3 activations (W2A8); UE8M0 scales per 32 inputs stored CSF-compressed (lossless); pre-sliced for tensor parallel 2. 138.7 GB, about 2.04 bits per weight |
| Engram tables (layers 1 and 14, 768,022,850 rows x 256) | 4-bit TrellisMX rows (K4), decoded to E4M3 values with a UE8M0 scale per 32; 104.4 GB, held in host RAM |
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
| Serving formats: experts on E4M3 operands with E4M3 activations, K4 Engram | **0.1439** [0.1346, 0.1545] | 0.0201 | 0.3162 | 2.04 | 6.05 | **86.82%** [86.75, 86.88] |
| Weights only: the same weights, activations unquantized | 0.1429 [0.1337, 0.1537] | 0.0198 | 0.3112 | 2.03 | 6.14 | 86.90% [86.83, 86.96] |

Where the loss comes from (serving formats, same 512 sequences): the official experts with the K4 Engram tables give
0.0454 [0.0407, 0.0511] (top-1 94.10%); the K2 experts with the official Engram tables give 0.1429
[0.1339, 0.1536] (top-1 86.82%). Their sum, 0.1883, exceeds the combined build's 0.1439: the two
errors overlap rather than add.

Full 8,192-token windows (8,191 rows each, 1,048,448 rows):

| Path | Mean KLD [BCa 95%] | Median | p90 | p99 | p99.9 | Top-1 [Wilson 95%] |
|---|---|---:|---:|---:|---:|---|
| Serving formats | **0.2182** [0.1779, 0.2690] | 0.0132 | 0.4368 | 3.70 | 9.63 | **86.41%** [86.34, 86.47] |
| Weights only | 0.2158 [0.1755, 0.2665] | 0.0130 | 0.4295 | 3.68 | 9.50 | 86.50% [86.44, 86.57] |
| Official experts, K4 Engram | 0.1020 [0.0766, 0.1347] | | | | | 92.82% |
| K2 experts, official Engram | 0.2158 [0.1757, 0.2662] | | | | | 86.49% |

By domain (serving formats, 2,048-token sequences): agentic 0.341 [0.310, 0.375], SWE agent 0.173 [0.151, 0.210], math 0.075 [0.069, 0.083], science 0.098 [0.088, 0.109], competitive programming 0.107 [0.092, 0.130], instruction following and chat 0.091 [0.085, 0.097], multilingual 0.099 [0.094, 0.106], long-context transduction 0.020 [0.017, 0.025], safety 0.150 [0.139, 0.168], web text 0.189 [0.179, 0.206], legal 0.166 [0.149, 0.191]. Unlike the 3-bit builds, KLD rises at 8,192 tokens: the median falls (0.0132 against 0.0201) but the tail grows (p99 3.70 against 2.04).

Checks: the teacher scored against itself is exactly 0 on every window; the reference model's canary window, run at the
start and end of every session, is bit-identical; on every window the teacher's top-1 agrees with the realized token
inside the [0.20, 0.995] band; the GPU scoring kernel matches a CPU FP64 reference to within 3e-12 on sampled rows of
every window. The evaluation windows share at most 5% of their content 13-grams with the data the build was calibrated
on, and every accepted conversation shared at most 2% with it.

How it was measured: the base model's layers run in DeepSeek's reference implementation, layer by layer on 4x NVIDIA
B200 at model parallel 4, one window per forward, with this checkpoint's experts and Engram rows reproduced bit for bit in
that path (the Engram rows are gathered on the CPU, as the serving path does from host RAM). The native SM120 kernel path
has not been measured. Every figure with its interval: `evaluation/report.json`; per-sequence and per-window statistics:
`evaluation/summary-A1.json` .. `summary-A4.json`; per-token records: `evaluation/records/`; window sources and
token-id SHA-256: `evaluation/eval-windows.json`.

## Files

- `model-000NN-of-00048.safetensors` + `model.safetensors.index.json`: every tensor that is not a routed expert or an
  Engram table, byte-identical to the base release.
- `trellismx/tp2/layer-LLL-rank-R.safetensors`: the 2-bit routed experts of layer LLL for tensor-parallel rank R (each
  rank holds all 384 experts' 1,152-wide half of the intermediate dimension), their CSF-compressed UE8M0 scales and the
  per-layer vectors the runtime applies with the experts.
- `trellismx/manifest.json`: the format description and the SHA-256 and size of every expert file.
- `engram/layer-001.*`, `engram/layer-014.*`: the 4-bit Engram tables (`.trellis.safetensors`, U8 [rows, 128]) and their
  UE8M0 scales (`.scale.safetensors`, F8_E8M0 [rows, 8]).
- `config.json` (the base config with `quantization_config` set for this format), tokenizer and `encoding/` files from
  the base release, `LICENSE`, `SHA256SUMS` (every file).
- `evaluation/`: `report.json` (every figure above with its interval), `summary-A1.json` .. `summary-A4.json`
  (per sequence and per window), `records/A1` .. `records/A4` (per-token records) and `eval-windows.json` (every
  window's sources, token-id SHA-256 and overlap screen). A1 = serving formats, A2 = weights only, A3 = official
  experts with the K4 Engram tables, A4 = K2 experts with the official Engram tables.
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
