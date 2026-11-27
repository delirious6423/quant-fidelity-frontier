---
license: apache-2.0
base_model:
  - google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
tags:
  - gguf
  - gemma4
  - true-performance-network
  - fluid-knowledge
  - model-comparison
---

<div align="center">

# TPN-009 Winner

### `model.gguf`

A compact Gemma 4 26B-A4B mixture-of-experts model selected through the TPN-009 miner competition and measured on True Performance Network infrastructure.

This is the rank-1 TPN-009 submission, not the competition base model. The measurements below cover text generation and fidelity to the base model.

**[View the public competition results](https://stats.trueperformancenetwork.com/competition?id=tpn-009)**

</div>

## At a glance

| Measurement | Result |
| --- | ---: |
| Fluid Knowledge | **88.36%** |
| KL similarity to Gemma 4 26B-A4B-it | **50.51%** |
| Mean KL divergence | **0.6829 nats** |
| Perplexity ratio | **0.3948x** |
| Measured VRAM at 2,048-token comparison context | **13.96 GiB** |
| GGUF file size | **11.46 GiB / 12.30 GB** |

## Fluid Knowledge

![Fluid Knowledge results](charts/fluid-knowledge.png)

The model scored **0.883571** over **1,400 zero-shot items** in Fluid Knowledge epoch `epoch-000003`. The run used an 8,192-token context, up to 2,048 generated tokens, temperature `0`, seed `0`, and the model's chat template with thinking disabled when supported.

| Evaluation | Value |
| --- | --- |
| TPN run | `r2726` |
| Metric | `exact_match,custom-extract` |
| Valid answer format | 96.50% |
| Strict answer format | 96.43% |
| Epoch commitment | `e5b429930c7c9414b54251a621b46b3bf95c35eb628a23c40d9095d46b81dc1d` |
| Manifest digest | `4fa3b27a4f6ecf28a3b9a42579747c30454c14de14838e5470f5ddeabd75d710` |

## Fidelity to the base model

![KL similarity for all comparison pairings](charts/fidelity-summary.png)

Run `r2738` compared the candidate with [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) at revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, using `fidelity-v1` and the pinned platform corpus `corpus-000002`.

| Pairing | Mean KL | KL similarity | Perplexity ratio | Mean hidden NRMSE |
| --- | ---: | ---: | ---: | ---: |
| Base -> unquantized runtime control | 0.343134 | 70.9543% | 1.00125x | 0.145005 |
| Runtime control -> candidate | 0.681941 | 50.5635% | 0.394261x | 0.215146 |
| **Base -> candidate** | **0.682926** | **50.5137%** | **0.394755x** | **0.212145** |

The control is an unquantized GGUF converted from the same base. It separates the Transformers-to-llama.cpp runtime difference from the candidate-to-control difference.

**Runtime agreement was classified as divergent.** The base-to-candidate result includes the runtime difference; the control-to-candidate pairing compares within the GGUF runtime. The harness also detected differing beginning-of-sequence token policies and used the base model's policy for scoring.

KL similarity is `exp(-mean KL)`, not an accuracy score. The lower candidate perplexity is not, by itself, evidence of better general capability. No comparison against other public quantizations is claimed here.

### Hidden-state drift

![Hidden-state drift by transformer layer](charts/hidden-state-drift.png)

Hidden-state NRMSE was measured across all 30 transformer layers using 4,096 deterministically sampled activation rows. Lower values indicate closer internal representations. The candidate's mean NRMSE was **0.212145** and its maximum was **0.490773** relative to the base.

## Artifact footprint

![Weight-file size and measured comparison VRAM](charts/footprint.png)

| Measurement | Base | Winner |
| --- | ---: | ---: |
| Selected source files / GGUF artifact | 48.10 GiB | 11.46 GiB |
| Measured comparison VRAM | 51.20 GiB | 13.96 GiB |

Comparison VRAM was measured after load at a 2,048-token context. It is not a universal minimum requirement or the competition's memory-scoring measurement. Memory changes with context length, cache precision, batch size, runtime, and measurement method.

## Comparison protocol

| Setting | Value |
| --- | --- |
| TPN run | `r2738` |
| Protocol | `fidelity-v1` |
| Reference runtime | Transformers `5.14.0` / PyTorch `2.11.0+cu128`, bfloat16 |
| GGUF runtime | llama.cpp Python `0.3.36` |
| Context length / stride | 2,048 / 1,024 tokens |
| Documents / windows | 60 / 91 |
| Scored / processed tokens | 92,960 / 124,764 |
| Activation samples / seed | 4,096 / 28,445 |
| GPU | NVIDIA B200 |
| Runtime agreement | Divergent |
| Comparison outcome | Complete |
| Measurement date | 2026-11-24 |

## Artifact and provenance

| Field | Value |
| --- | --- |
| Publication repository | `tpnlabs/tpn-009-winner` |
| Published file | `model.gguf` |
| Publication revision | `58abc0715b84a43abe701ecad9cf12411ac9b1bb` |
| Format / architecture | GGUF / `gemma4` |
| File size | 12,301,636,864 bytes (11.46 GiB) |
| Original submission | [`asura625/g26x-a36`](https://huggingface.co/asura625/g26x-a36) |
| Original filename | `g26x-a36.gguf` |
| Evaluated source revision | `2c41fda2f38df98ec51c2d0c5c21a231b0fab5d6` |
| Evaluated artifact SHA-256 | `149d0e90c96d748d138c6e537836650d8d282d70adb28e555d721a6682494559` |

The winner was copied into the TPN Labs repository for publication. Benchmark evidence is tied to the evaluated source artifact and SHA-256 above, rather than to a mutable repository name.

## Running the model

Use a recent Gemma 4-capable llama.cpp runtime:

```bash
llama-cli -m model.gguf -p "Hello" -n 256
```

For a local OpenAI-compatible server:

```bash
llama-server -m model.gguf --ctx-size 8192 --port 8080
```

The evaluations above cover text workloads, not a separate vision projector or multimodal serving configuration.

## Scope and limitations

- Fluid Knowledge measures one pinned epoch, not general capability, safety, or factual reliability in every setting.
- Fidelity measures similarity on one fixed corpus and is not interchangeable with task accuracy or the competition ranking score.
- Results apply to the evaluated artifact, reference revision, runtimes, and settings listed above.
- Runtime agreement was divergent; both the runtime control and the candidate pairings are reported for context.
- These results do not establish superiority over other public GGUF quantizations.

## License

The declared base model license is Apache 2.0. Review the upstream [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) license and this repository's files before redistribution or deployment.
