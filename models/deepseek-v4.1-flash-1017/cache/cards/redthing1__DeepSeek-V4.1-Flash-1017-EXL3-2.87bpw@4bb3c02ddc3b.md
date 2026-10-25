---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
- exl3
- exllamav3
- deepseek
- deepseek_v41
- moe
- quantized
---

# DeepSeek-V4.1-Flash-1017 · EXL3 2.87 bpw

This EXL3 quantization of **DeepSeek-V4.1-Flash-1017** preserves the source model's predictions more faithfully than the existing HeNryous and EAQ releases. It achieves **7.6–23.2% lower full-vocabulary KL than HeNryous 2.75 bpw** and **28.3–42.2% lower than EAQ v1 2.9 bpw**, improving on both across every evaluated history, including held-out solver and compiler projects.

The release was produced with **Quantseek**, our calibration and quantization engine for preserving the original model's reasoning and broader capabilities under compression. We use substantial engineering projects to exercise abilities that matter well beyond software: extended reasoning, consistent handling of constraints, integration of evidence, tool use and recovery from failed approaches. The original model supplies the target behavior throughout.

The assignments require SAT solvers that learn from conflicts and explain contradictory assumptions; register allocators that preserve program behavior while reducing memory spills; minimum-cost flow solvers that handle negative-cost cycles; and text indexes that balance construction memory against query cost. Complete histories retain reasoning, implementation, tool results and debugging. Each model receives identical inputs under the same evaluator, and we compare probabilities across all **129,280 possible next tokens**.

| History | Context | This quant | HeNryous 2.75 | EAQ 2.9 |
| --- | ---: | ---: | ---: | ---: |
| Minimum-cost flow | 36,234 | **0.01994** | 0.02158 | 0.02781 |
| Text indexing | 74,036 | **0.01048** | 0.01258 | 0.01684 |
| Text indexing, long | 150,791 | **0.01040** | 0.01316 | 0.01781 |
| SAT solving | 222,432 | **0.01453** | 0.01817 | 0.02087 |
| Register allocation | 229,704 | **0.01205** | 0.01569 | 0.02085 |

KL divergence measures the difference between the models' next-token probability distributions at the same assistant positions. **Lower means closer to the source model.** SAT and register allocation were held out from fitting and candidate selection.

## Why existing community work falls short

The relevant standard is preservation of the model's capabilities when a problem requires sustained reasoning, consistent decisions and adaptation to new evidence. The published DGX Spark and EAQ evaluations provide weak evidence for that standard:

| Existing evaluation | Published protocol |
| --- | --- |
| [DGX Spark task suites](https://github.com/sudoingX/dgx-spark-deepseek-v41/blob/379edd6aec5b3c41569c45b61745dd034d3df574/BENCHMARKS.md#suites) | HumanEval+, MBPP+ and IFEval with **thinking disabled** and a **2,048-token output cap**. |
| [EAQ long-session fidelity](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw#quality-vs-the-original-model) | **Eight synthetic sessions**, at most **31.2K tokens**; thinking disabled, every assistant turn a scripted tool call. |
| [DGX Spark KL scorer](https://github.com/sudoingX/dgx-spark-deepseek-v41/blob/379edd6aec5b3c41569c45b61745dd034d3df574/scripts/quantize/p4_kld.py) | **Top 128 tokens plus one pooled tail bucket**, collapsing distinctions across the remaining vocabulary. |
| [EAQ fidelity reference](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw#quality-vs-the-original-model) | Omits the source model's **dynamic FP8 activation quantization**, altering the numerical reference. |

Disabling reasoning removes the behavior of interest. Scripted tool calls do not exercise the reasoning required to construct and debug an algorithm. Pooling the vocabulary tail discards distinctions in the predictive distribution. Changing the reference arithmetic also changes what is being preserved. These choices substantially weaken the evidence offered for demanding reasoning workloads.

The [DGX Spark recipe](https://github.com/sudoingX/dgx-spark-deepseek-v41/blob/379edd6aec5b3c41569c45b61745dd034d3df574/README.md#how-it-works) uses activation-based fitting and expert-specific precision. It assigns importance using a layer's sensitivity and each expert's share of its output. These estimates do not directly measure what a particular expert gains from an actual stored precision change. Later layers are calibrated on inputs produced by earlier quantized layers, accounting for changes in those inputs. The fitting objective still concerns the local approximation; it does not directly minimize the distortion of the model's final predictions.

[EAQ v1's published allocation](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw#conversion-what-was-actually-done) retains the main-model bit widths of the earlier standard EXL3 recipe: routed experts use two bits in layers 18–22 and three bits elsewhere. That schedule cannot assign different precision to sensitive and insensitive experts within the same layer. The EAQ label describes a conversion change, not a new main-model precision allocation.

A local error estimate, an importance ranking and an average bit count cannot establish the fidelity of the finished model. Quantseek evaluates the stored weights through complete forward passes and requires gains across separate workloads. The advantage measured above is present in the finished artifact.

## The Quantseek method

EXL3 defines how compressed weights are stored and reconstructed; it leaves substantial freedom in how those weights are chosen. Quantseek controls calibration, fitting, precision allocation and output refinement within that representation. The optimization is directed toward the behavior of the complete stored model, and the result retains ordinary EXL3 compatibility.

Calibration records the internal signals produced as the source model processes a workload, including their magnitudes and which signals occur together. Two equally small changes to a weight matrix can have very different effects on the computations the model actually performs. Our fitting uses those measured relationships to preserve layer outputs on substantive reasoning and tool-use histories. Complete context is retained: a late debugging step is conditioned on the work that preceded it. The reference also retains the original model's activation quantization, including the rounding applied to intermediate signals.

DeepSeek routes each token through selected expert networks, whose importance varies with the input. More bits allow a more accurate approximation at a higher storage cost. Giving every expert the same precision spends bytes on approximations that may already be adequate while leaving more consequential errors unresolved. Even within one expert, the gate/up projections that construct an intermediate representation and the down projection that returns it to the model have different precision requirements.

Quantseek compares stored alternatives and their error reduction relative to their storage cost when revising the allocation. The release assigns **2–5 bits** across **46,080 routed matrices**, averaging **2.87 bpw**. Gate/up projections share a precision choice; down projections are allocated separately. The output head uses **6 bits**. The average describes the budget; its placement determines which computations that budget protects.

Errors interact as information passes through the network. A matrix can become a better local approximation while the complete model's predictions become worse. A correction at the output stage can compensate for some of the error accumulated earlier. Our output-aware refinement compares the source and quantized models' probabilities across the entire vocabulary, including alternatives beyond the single most likely token. The output head—the final projection that produces token scores—is refined using states from the quantized model. The correction therefore addresses accumulated compression error. It is incorporated into the EXL3 weights; inference requires no additional correction module.

Compression can erase a gain measured in a higher-precision working representation. We therefore reconstruct candidates from the codes and scaling factors actually saved in the files, then measure complete-model outputs. A promising intermediate fit earns no credit unless the improvement survives in the stored weights. Selection also tests transfer across separate workloads: more examples from the same history cannot substitute for a different problem family. The held-out SAT and register-allocation histories check whether the selected result extends to unfamiliar solver and compiler work.

The native DSpark drafter is independently quantized from DeepSeek's official checkpoint. Its calibration records signals inside the actual drafting computation, using features produced by this quantized target. Its routed experts use 2 bits; the other quantized projections use 5 bits. It retains the released architecture and shares the target's token embeddings and output projection. We evaluate proposal acceptance and the confidence-based policy that selects how many proposals the target verifies on each round.

## Download

**2.87 bpw** is the average precision of the target model's routed experts. Download size is approximately **206.7 GB**, plus **203.1 GB** for the two unchanged Engram shards from the official 1017 checkpoint.

```sh
model_dir=./DeepSeek-V4.1-Flash-1017-EXL3-2.87bpw
hf download redthing1/DeepSeek-V4.1-Flash-1017-EXL3-2.87bpw --local-dir "$model_dir"
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir "$model_dir"
```

Load the directory with an ExLlamaV3 runtime supporting DeepSeek V4.1/1017. The index already includes Engram; no edits are needed. Vision and drafter weights are included. Verify downloaded files with `sha256sum --check SHA256SUMS` from the model directory.

MIT license. Base model by [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017); format and runtime by [ExLlamaV3](https://github.com/turboderp-org/exllamav3). [EAQ comparison](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw).
