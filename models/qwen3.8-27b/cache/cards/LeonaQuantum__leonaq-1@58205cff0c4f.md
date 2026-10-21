---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
language:
  - en
  - ja
tags:
  - quantum-inspired
  - tensor-networks
  - matrix-product-operator
  - simulated-quantum-annealing
  - quantization
  - custom-code
---

# LeonaQ-1

LeonaQ-1 is a quantum-inspired 27B language model based on Qwen3.8-27B. It combines low-bit weights with learned tensor-network corrections and a trained quantum-circuit residual. The complete model payload occupies **10.27 GB**.

The release integrates 11 quantum-inspired techniques across training, text generation, model storage, session compression, and auxiliary text analysis. All components run on classical NVIDIA GPUs. The package includes the model weights, trained adapters, tokenizer, and Python runtime.

[Run LeonaQ-1 on NVIDIA GPUs](USAGE.md) provides the CUDA installation commands and inference examples. [Run LeonaQ-1 on a Mac](MAC.md) describes the portable Metal and CPU text runtime.

## October 20, 2026: circuit-preserving GGUF and faster Metal operations

The [Q4_K_M GGUF](gguf/leonaq-1-mpo-circuit-Q4_K_M.gguf) (16.55 GB) retains the trained six-qubit circuit and merges the MPO corrections into the transformer weights. It requires our **dedicated llama.cpp extension**; stock llama.cpp, Ollama, LM Studio and vLLM are not supported for this custom GGUF. Follow the [build and download instructions](ENGINE_COMPATIBILITY.md).

On one A100-SXM4-80GB, the patched engine measured **48.15 tokens/s** for 32-token decoding (three repetitions). Full-model Mac throughput has not been measured. The packed PyTorch runtime also includes fused Metal operations for packed weights and the full six-qubit circuit, a persistent interactive mode, streaming output, and separate loading, first-token and decode timing; see [MAC.md](MAC.md).

The GGUF preserves the circuit computation, but MPO merging and Q8-to-Q4 requantization change numerical results. Broad quality parity with the original packed model has not been established. Original packed weights and adapter artifacts remain available. [Measurements and verification scope](ENGINE_RESULTS.json).

## Release at a glance

| Property | Value |
|---|---|
| Base model | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Model payload | 10,265,779,625 bytes, including the packed base and four auxiliary artifacts |
| Tensor-network corrections | Matrix Product Operators in 192 feed-forward linear modules |
| Generation adapter | Trained six-qubit circuit residual after the final transformer layer's MLP |
| Verified hardware | One NVIDIA A100 80 GB |
| Runtime | PyTorch with CUDA, Metal, or CPU text generation |
| License | Apache 2.0 |

## Eleven quantum-inspired components

### Model preparation and generation

The packed model combines conventional low-bit quantization with tensor-network corrections and a circuit residual. Targeted training fits these additions while the base model remains frozen.

| Component | Implementation |
|---|---|
| Matrix Product Operators, or MPOs | Learned tensor-network corrections in 192 quantized feed-forward linear modules. |
| Six-qubit circuit residual | A trained circuit simulation with Ry, Rz, and controlled-Ry gates adds a residual after the final layer's MLP. |
| Born-Fisher natural-gradient refinement | Circuit training retains parameter updates derived from the geometry of Born probabilities. |
| Ballistic simulated bifurcation, or bSB | An Ising-inspired solver selects 256 of 1,024 activation examples for circuit-adapter training. |
| Simulated quantum annealing, or SQA | Annealing selects rounding decisions for the circuit's int4 readout weights. |

The MPO corrections and circuit residual contribute during generation. The released circuit artifact incorporates the coreset selection, natural-gradient updates, and SQA rounding.

### Model storage and inference state

Three components apply tensor-network and Ising-inspired methods to stored metadata and inference state.

| Component | Implementation |
|---|---|
| Born-MPS metadata coding | A Matrix Product State probability model supports lossless coding of one module's quantization scales and offsets. |
| Query-weighted Schmidt decomposition | Adaptive tensor factorization compresses recurrent states for session checkpoints. |
| bSB attention-cache selection | An Ising-inspired solver selects retained key-value entries across all 16 full-attention layers and preserves their original token positions. |

### Auxiliary text analysis

The runtime includes two trained English sentiment heads and a response-diversity score.

| Component | Implementation |
|---|---|
| Pauli circuit sentiment head | A trained 13-qubit circuit with 52 angles produces 416 Pauli features for sentiment classification. |
| MPS sentiment head | An eight-site tensor-network residual head uses two-site density matrix renormalization group, or DMRG, updates. |
| Quantum density-response scoring | A density-matrix-inspired adjustment summarizes text diversity across eight generated response samples. |

## Measured results

The following measurements describe the released artifacts and their evaluated configurations. Sentiment accuracy uses the project's reused 872-example SST-2 report partition with one training setting.

| Measurement | Result | Scope |
|---|---:|---|
| Pauli circuit sentiment accuracy | **94.38%** | English SST-2 sentiment classification |
| MPS sentiment accuracy | **93.81%** | The same SST-2 partition |
| SQA circuit artifact size | **201,209 bytes** | The preceding circuit artifact occupied 374,705 bytes |
| Retained attention KV tensors | **50% reduction** | 32 MiB to 16 MiB for one 512-token prefix across all 16 full-attention layers, plus 128 KiB of position metadata |
| Checkpoint restoration | **Identical logits over 32 steps** | Full-vocabulary logits from the in-memory compressed state and its saved-and-reloaded counterpart |

A single A100 80 GB ran text generation, session checkpointing, and auxiliary sentiment classification. Release verification also reproduced the generation tokens and sentiment predictions with code and adapters downloaded from the public repository.

The [evaluation records](evidence/) contain the underlying measurements. The [release manifest](release.json) records the software environment and artifact hashes.

## Training and provenance

The base model is Qwen3.8-27B at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. LeonaQ-1 combines low-bit weight packing with targeted training of the MPO corrections, circuit residual, and sentiment heads.

The runtime verifies the packaged files and binds the adapters and sentiment heads to the same model fingerprint:

```text
9fca585d565a04af3a8093c190e87b9c322da6fe018fc86ca50def01863e025f
```

## License and acknowledgments

LeonaQ-1 is distributed under the [Apache License 2.0](LICENSE). The [NOTICE](NOTICE) records attribution and model modifications.

We acknowledge the Qwen team for the base model and the researchers whose work on tensor networks, quantum-inspired optimization, and efficient language models informs this release.
