---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
license: apache-2.0
tags:
  - gguf
  - gsq
  - rco
  - quantization
  - pruning
  - expert-pruning
  - mixed-precision
  - ist-daslab
  - moe
  - code
---

Vocabulary pruned version of the October 17 updated Coder GGUF of Qwen 3.8 Flash-Next by IST-DASLab.

By removing non-english characters irrelevant to coding from token/output weights, we **save 450 MB memory** while retaining quality.
While a small gain, this can make or break it for 8-16 GB VRAM + 32 GB RAM systems.
Not recommended for non-english languages.

The process:
It was merged into a single file using `llama-gguf-split --merge` then [pruned to keep English and coding-related vocabulary using P1G](https://github.com/bsaleh03/ASCII-Condensed-prune-tools) profile.

Logs:
```
policy P1G: keeping 130020 / 248320 (52.36%), dropping 118300
  breakdown: {'special': 276, 'bytefb': 256, 'partial': 816, 'ascii': 127658, 'accented': 0, 'extra': 1014}
  merges 247587 -> 129463 (dropped 118124)
  remap tokenizer.ggml.eos_token_id: 248046 -> 129746
  remap tokenizer.ggml.padding_token_id: 248044 -> 129744
  remap tokenizer.ggml.bos_token_id: 248044 -> 129744
  output.weight: Q6_K 2100 B/row, 248320 -> 130020 rows, 0.521 -> 0.273 GB
  token_embd.weight: IQ4_XS 1360 B/row, 248320 -> 130020 rows, 0.338 -> 0.177 GB

wrote Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-merged_P1G_Pruned.gguf (57.993 GB)
== model ==
  OK   architecture preserved (qwen4exp)
  tokenizer.ggml.model = 'gpt2'
== tokenizer ==
  vocab 248320 -> 130020
  OK   token_type length 130020 == vocab 130020
  OK   every kept token exists in the source vocab (0 missing, e.g. [])
  OK   kept tokens preserve their original relative order
  OK   token_type values follow their tokens
== specials ==
  OK   all 276 non-normal tokens retained (got 276)
  OK   special token strings identical
== byte fallback ==
  OK   all 256 byte-fallback tokens present (0 missing)
== special ids remapped ==
  OK   tokenizer.ggml.bos_token_id: 248044->129744 both = '<|endoftext|>'
  OK   tokenizer.ggml.eos_token_id: 248046->129746 both = '<|im_end|>'
  OK   tokenizer.ggml.padding_token_id: 248044->129744 both = '<|endoftext|>'
== merges ==
  merges 247587 -> 129463
  OK   every surviving merge references only kept tokens (0 bad)
  OK   merges preserve original priority order
== metadata ==
  OK   tokenizer.chat_template preserved
  OK   general.architecture preserved
  OK   tokenizer.ggml.pre preserved
  OK   tokenizer.ggml.model preserved
  OK   general.name preserved
  OK   general.file_type preserved
  OK   qwen4exp.block_count preserved
  OK   qwen4exp.context_length preserved
  OK   qwen4exp.embedding_length preserved
  OK   qwen4exp.attention.head_count preserved
  OK   qwen4exp.attention.head_count_kv preserved
  OK   qwen4exp.rope.dimension_sections preserved
  OK   qwen4exp.rope.freq_base preserved
  OK   qwen4exp.attention.layer_norm_rms_epsilon preserved
  OK   qwen4exp.expert_count preserved
  OK   qwen4exp.expert_used_count preserved
  OK   qwen4exp.attention.key_length preserved
  OK   qwen4exp.attention.value_length preserved
  OK   qwen4exp.expert_feed_forward_length preserved
  OK   qwen4exp.expert_shared_feed_forward_length preserved
  OK   qwen4exp.ssm.conv_kernel preserved
  OK   qwen4exp.ssm.state_size preserved
  OK   qwen4exp.ssm.group_count preserved
  OK   qwen4exp.ssm.time_step_rank preserved
  OK   qwen4exp.ssm.inner_size preserved
  OK   qwen4exp.full_attention_interval preserved
  OK   qwen4exp.rope.dimension_count preserved
  OK   qwen4exp.hyper_connection.count preserved
  OK   qwen4exp.hyper_connection.low_rank preserved
  OK   qwen4exp.attention.indexer.head_count preserved
  OK   qwen4exp.attention.indexer.key_length preserved
  OK   qwen4exp.attention.indexer.top_k preserved
  OK   qwen4exp.attention.compress_ratios preserved
  OK   qwen4exp.ple.layers preserved
  OK   qwen4exp.ple.ngram_size preserved
  OK   qwen4exp.ple.heads_per_ngram preserved
  OK   qwen4exp.ple.conv_kernel preserved
  OK   qwen4exp.ple.eos_token_id preserved
  OK   qwen4exp.embedding_length_per_layer_input preserved
  OK   qwen4exp.ple.layer_multipliers preserved
  OK   qwen4exp.ple.head_offsets preserved
  OK   qwen4exp.ple.image_token_id preserved
== tensors ==
  OK   same tensor set (1224 vs 1224)
  OK   same tensor order
  OK   all ggml types unchanged
  vocab tensors present: ['token_embd.weight', 'output.weight']
  OK   all shapes correct ([])
  hashing non-vocab tensor payloads...
  OK   all 1222 non-vocab tensors byte-identical (0 differ: [])
  verifying gathered vocab rows...
  OK   token_embd.weight: sampled 4065 rows match source row keep[i] (0 mismatches)
  OK   output.weight: sampled 4065 rows match source row keep[i] (0 mismatches)
== policy conformance ==
  replayed P1G: expects 130020 tokens
  breakdown: {'special': 276, 'bytefb': 256, 'partial': 816, 'ascii': 127658, 'extra': 1014}
  OK   kept set is exactly what policy P1G specifies

ALL CHECKS PASSED
```

<div align="center">

<a href="https://github.com/IST-DASLab"><img src="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/main/assets/banner.png" alt="GGUF, GSQ-RCO expert-pruned coder" width="100%"/></a>

<br/>

# Qwen3.8-Flash-Next &middot; GSQ-RCO Coder

**A 512-expert mixture-of-experts model with half of its experts removed**, targeted at code and retaining multimodal capability.
**58.4 GB** against 354 GB at BF16, an effective **1.89 bits per parameter of the original transformer**.

[![arXiv: GSQ](https://img.shields.io/badge/arXiv-GSQ_2604.18556-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.18556)
[![arXiv: RCO](https://img.shields.io/badge/arXiv-RCO_2605.00649-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.00649)
[![GSQ code](https://img.shields.io/badge/code-GSQ-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/GSQ)
[![RCO code](https://img.shields.io/badge/code-RCO-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/RCO)
[![DASLab](https://img.shields.io/badge/DASLab-GitHub-101048?logo=github&logoColor=white)](https://github.com/IST-DASLab)
[![license](https://img.shields.io/badge/license-apache--2.0-19a34a)](#license)

</div>

> **Update, October 17, 2026.** The model has been improved: looping is reduced and coding performance is higher, with SWE-bench Verified rising from 75.60 to 79.20 and LiveCodeBench v6 from 86.28 to 87.43. Only the first shard changed; if you downloaded the earlier build, re-download that shard.

![Coding Benchmarks After 50% Expert Pruning](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/main/assets/plots/Qwen3.8-Flash-Next-Coder-coding_benchmarks.png)

---

## Overview

This release is a capability-targeted compression of Qwen3.8-Flash-Next. Half of the routed experts are removed from the model rather than quantized, and the remaining weights are stored at 3.5 bpw.

The base model has 176.9B parameters and occupies 354 GB at BF16. The compressed model reduces to only 29.6 GB which must reside in memory: the n-gram shard is a lookup table and may be served from disk. The resident working set of a 176.9B-parameter model is therefore within the capacity of a single 32 GB accelerator.

Averaged over the transformer, this corresponds to **1.89 bits per parameter of the original model**. The figure is an effective rate: it amortises the removed experts over the original parameter count, and so expresses the combined effect of pruning and quantization. No individual weight is stored at 1.89 bits. The retained weights are stored at 3.5 bpw, unchanged by pruning, and the halved expert count accounts for the remainder of the reduction. Following the convention of the accompanying releases, the figure is reported over the transformer and excludes the fixed-precision n-gram shard.

Removing half of a model's experts necessarily reduces its capabilities. The contribution of this release is that the reduction is directed towards code, agentic tool use, vision, and spatial reasoning, and the retained experts are therefore those that support these domains. Degradation outside this set is an accepted cost of the method. For general-purpose use, the [unpruned GSQ-RCO releases](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) are the appropriate choice.

| | SWE-bench Verified | LiveCodeBench v6 |
|---|---|---|
| BF16 base (354 GB) | 82.80 | 87.43 |
| **GSQ-RCO Coder (58.4 GB)** | **79.20** | **87.43** |
| retained | 95.7% | 100.0% |

All numbers in this card were measured at **xhigh reasoning effort**.

This is an experimental release and feedback is welcome, particularly on capabilities that were not represented in the calibration mixture.

LiveCodeBench v6 matches the base score. SWE-bench Verified, which evaluates extended agentic trajectories over real repositories, retains 95.7%. The gap between the two is consistent with the expectation that sustained multi-step tasks accumulate error across turns, and therefore tolerate capacity reduction less well than single-problem code generation.

---

## What Expert Pruning Is

A mixture-of-experts layer holds many independent feed-forward blocks and a small router that picks a few per token. Flash-Next has **512 experts per layer across 48 layers, with 10 active per token**, so the experts hold the overwhelming majority of the parameters while only a fraction runs on any given token.

Expert pruning removes entire experts from the model. It is therefore distinct from quantization, which retains every parameter and reduces the precision at which it is stored. Pruning removes parameters from the model outright, reducing both file size and resident memory, and restricts the router to a smaller candidate pool. The two techniques are complementary and are combined here: 50% of experts are removed, and the remaining weights are quantized to 3.5 bpw.

The central question is which experts to remove. Since experts specialise, importance is not an intrinsic property of an expert but is defined relative to a distribution of inputs. That distribution is supplied by the calibration data, which consequently determines which capabilities survive compression.

---

## How the Experts Are Chosen

Selection was done with **RCO** ([paper](https://arxiv.org/abs/2605.00649), [code](https://github.com/IST-DASLab/RCO)), optimizing the KL divergence between the pruned and unpruned model on calibration data, rather than ranking experts by a heuristic importance score.

The GGUF format constrains the problem in a particular way. It stores **a single expert count for the entire model**, so every layer must retain the same number of experts. A single global budget would permit the search to remove more experts from insensitive layers and fewer from sensitive ones, which generally yields a better model, but produces per-layer counts that the format cannot represent without padding that forfeits the memory saved.

RCO enforces budgets exactly by treating the feasible set as a smooth manifold, and admits **multiple simultaneous budgets**. Replacing the global budget with one budget per layer expresses the format's structural requirement directly as constraints. These constraints are mutually independent, as no expert belongs to more than one layer, so the projection and retraction steps decompose into per-layer operations and remain inexpensive.

The constraints decompose, but **the objective does not**, the loss is the KL divergence of the full model, so the gradient for any layer reflects the state of all others; the layers are optimised jointly and only budget enforcement proceeds per layer. This distinguishes the method from the common alternative of retaining a fixed number of top-ranked experts per layer under a heuristic importance score, with no joint search. A penalty-based formulation would additionally require one coefficient per layer, and these coefficients interact; RCO requires none.

---

## Vision Capability

An initial search used code and agentic calibration data only. It removed the experts on which the vision pathway depends, and image capability degraded substantially while coding scores were unaffected. This is the expected behaviour rather than a failure of the method: absent calibration data that exercises vision, the search correctly determines that those experts do not contribute to the measured objective.

The released model was searched with vision represented in the calibration mixture and retains the capability. As an informal qualitative check, the "pelican riding a bicycle" prompt is used, which requires object knowledge, spatial composition and correct SVG syntax simultaneously:

<div align="center">
<img src="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/main/assets/pelican-on-bike.svg" alt="SVG of a pelican riding a bicycle, drawn by the pruned model" width="440"/>
</div>

---

## Available Files

Two shards, both required. `llama.cpp` loads the pair when given the first.

| File | Size | Must be resident | Notes |
|---|---|---|---|
| `...-00001-of-00002.gguf` | 29.6 GB | yes | Transformer weights, 256 of 512 experts per layer, 1.89 bpw over the original parameters |
| `...-00002-of-00002.gguf` | 28.8 GB | no | Per-layer n-gram table (IQ4_NL, 4.5 bpw), neither pruned nor searched |
| **Total** | **58.4 GB** | **29.6 GB** | against 354 GB and 354 GB resident for BF16 |

The n-gram table is a lookup rather than a matmul weight, so it is held at a fixed 4.5 bpw and excluded from both the quantization search and the pruning.

### llama.cpp
```bash
hf download <this-repo> --local-dir .
llama-cli -m IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  -p "Refactor this function to be iterative." -ngl 99
```

### Strata Engine

Many users report strong results with [Strata](https://github.com/Niko1221/Strata), a third-party engine that runs these GGUFs on a single NVIDIA GPU from 8 GB of VRAM, with a one-click installer and an OpenAI-compatible endpoint on localhost.

---

## Method

| | |
|---|---|
| **GSQ** ([paper](https://arxiv.org/abs/2604.18556), [code](https://github.com/IST-DASLab/GSQ)) | Post-training scalar quantization that learns grid assignments and group scales jointly, closing most of the gap to vector quantization at low bit-widths while staying deployable in standard GGUF types. |
| **RCO** ([paper](https://arxiv.org/abs/2605.00649), [code](https://github.com/IST-DASLab/RCO)) | Budget-constrained optimization on a Riemannian manifold. Used here to choose which experts to remove under one exact budget per layer, and in the base releases to assign per-tensor quantization types. |

Both developed at the [Deep Algorithms and Systems Lab (DASLab)](https://github.com/IST-DASLab), Institute of Science and Technology Austria.

---

## Citation

```bibtex
@article{gsq2026,
  title  = {GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling},
  author = {Dadgarnia, Alireza and Tabesh, Soroush and Nikdan, Mahdi and Helcig, Michael and Kurtic, Eldar and Kleinegger, Maximilian and Alistarh, Dan},
  journal= {arXiv preprint arXiv:2604.18556},
  year   = {2026}
}

@article{rco2026,
  title  = {Model Compression with Exact Budget Constraints via Riemannian Manifolds},
  author = {Helcig, Michael and Alistarh, Dan},
  journal= {arXiv preprint arXiv:2605.00649},
  year   = {2026}
}
```

---

## Acknowledgements

We thank [Verda](https://verda.com/) and Scientific Computing at the Institute of Science and Technology Austria for the compute used to produce these models.

## License

These weights inherit the license of the base model (**Qwen3.8-Flash-Next**). The GSQ-RCO tooling is released by the Deep Algorithms and Systems Lab under its repository license.

<div align="center">
<sub>Built with <b>GSQ</b> and <b>RCO</b> at the <a href="https://github.com/IST-DASLab">Deep Algorithms and Systems Lab</a> &middot; Institute of Science and Technology Austria</sub>
</div>
