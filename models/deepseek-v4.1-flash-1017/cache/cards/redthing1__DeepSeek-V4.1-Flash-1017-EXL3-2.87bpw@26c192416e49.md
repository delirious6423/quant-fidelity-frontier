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

This EXL3 quantization of **DeepSeek-V4.1-Flash-1017** delivers **superior source-model fidelity**, achieving **6.8–46.7% lower full-vocabulary KL** across a broad comparison of existing EXL3 releases. The advantage is present in the finished weights on complete reasoning and tool-use histories.

The release was produced with **Quantseek**, our calibration and quantization engine for preserving the original model's reasoning and broader capabilities under compression. We use substantial engineering projects to exercise abilities that matter well beyond software: extended reasoning, consistent handling of constraints, integration of evidence, tool use and recovery from failed approaches. The original model supplies the target behavior throughout.

The assignments require SAT solvers that learn from conflicts and explain contradictory assumptions; register allocators that preserve program behavior while reducing memory spills; minimum-cost flow solvers that handle negative-cost cycles; and text indexes that balance construction memory against query cost. Complete histories retain reasoning, implementation, tool results and debugging. Each model receives identical inputs under the same evaluator, and we compare probabilities across all **129,280 possible next tokens**.

| Release | Flow | Text | Long text | SAT | Register |
| --- | ---: | ---: | ---: | ---: | ---: |
| This quant | **0.01994** | **0.01048** | **0.01040** | **0.01453** | **0.01205** |
| [LibertAI 2.9](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | 0.02912 | 0.01780 | 0.01888 | 0.02237 | 0.02263 |
| [jayleaton 2.8](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw) | 0.02861 | 0.01705 | 0.01762 | 0.02082 | 0.02159 |
| [sudoingx 2.8](https://huggingface.co/sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw) | 0.02141 | 0.01260 | 0.01313 | 0.01773 | 0.01590 |
| [HeNryous 2.75](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark) | 0.02158 | 0.01258 | 0.01316 | 0.01817 | 0.01569 |
| [EAQ v1 2.9](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw) | 0.02781 | 0.01684 | 0.01781 | 0.02087 | 0.02085 |
| [EAQ v1.1 2.9](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw) | 0.02652 | 0.01635 | 0.01770 | 0.02057 | 0.02072 |

**Lower KL means closer agreement with the source model's next-token predictions.**

## Why existing community work falls short

The central weakness in existing community work is the distance between what its evaluations measure and the capabilities users need preserved. Short-text perplexity, reasoning-disabled task suites and isolated component checks leave substantial compression damage unexamined. The published protocols make those omissions concrete:

| Release | Published evaluation | What it fails to establish |
| --- | --- | --- |
| [LibertAI 2.9](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/blob/0be63c3337efbb2a0061d5b8bb4006fa186d433f/README.md) | Perplexity on two sets of **32 × 2,048-token** code/prose excerpts. Its separate agentic benchmark uses the **original weights**. | Preservation of extended reasoning in the released quant. Benchmarking the base model supplies no evidence about damage introduced by this conversion. |
| [sudoingx 2.8](https://huggingface.co/sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw/blob/aba08ea002ae271ee2a2724bcc0c6dc6e1c457b8/README.md) | HumanEval+, MBPP+ and IFEval with **thinking disabled** and a **2,048-token output cap**. | Preservation of the reasoning behavior those settings explicitly exclude. Its separate native-reply fidelity panels do not repair that task-evaluation omission. |
| [EAQ v1](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw#quality-vs-the-original-model) | Long-session evaluation: **eight synthetic sessions**, up to **31.2K tokens**; thinking disabled and assistant turns scripted. Its evaluation port fails above **32,768 tokens** and omits dynamic FP8 activation quantization. | Fidelity during sustained problem solving, or fidelity to the original numerical reference. Scripted tool calls do not exercise the reasoning needed to construct and debug an algorithm. |
| [EAQ v1.1](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw/blob/aee640e2afba93be683bb5b9b420115fe7fbac20/README.md) | Perfect indexer recall on **source-model hidden states**; roughly **0.73 recall** on each quant's own states, similar to v1. | Recovery of decisions inside the complete quantized model. Agreement on source inputs does not establish agreement on the quant's own inputs. |
| [jayleaton 2.8](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw/blob/859396591c340195526aaf61f94b4ad5323a5e18/README.md) | **356 thinking-enabled task items**, comparing 1017 and September-based packs; every reported task-difference interval includes zero. Its fidelity table also uses different source checkpoints. | A resolved quantization-quality ranking. Changing the base checkpoint confounds the task comparison; the reported task margins establish no advantage. |

**The reference and metric must also withstand scrutiny.** The [earlier DGX Spark scorer](https://github.com/sudoingX/dgx-spark-deepseek-v41/blob/379edd6aec5b3c41569c45b61745dd034d3df574/scripts/quantize/p4_kld.py) keeps 128 tokens and collapses the remaining vocabulary into one tail bucket. That measures a coarser distribution and hides redistribution within the tail. EAQ v1's altered activation arithmetic changes the reference being approximated. Neither choice is interchangeable with full-vocabulary fidelity to the released model's numerical computation.

**The quantization recipes have corresponding limitations.** LibertAI and EAQ v1 retain the layer-wide routed-expert schedule: two bits in layers 18–22, three bits elsewhere. All 1,152 routed matrices in a layer receive the same width, despite serving different experts and different projection roles. That schedule has no way to spend an extra bit on a sensitive matrix while taking it from an insensitive neighbor. A new conversion label does not create a new precision allocation.

LibertAI calibrates on **250 sequences of 2,048 tokens**, exposing every expert to every calibration token. Those statistics describe short excerpts and a broad expert-input distribution; they do not measure how that expert behaves after an extended chain of reasoning, tool results and corrections. Its DSpark drafter is explicitly encoded **without calibration data**. Target-model perplexity provides no assessment of the resulting proposal acceptance.

sudoingx uses per-expert precision and calibrates later layers on quantized-prefix states. Its allocation still combines layer sensitivity with an expert's share of routed output. Output contribution does not measure how much error a particular matrix would lose by moving from one stored precision to another. Its encoder fits local reconstructions; a better local approximation does not guarantee better final predictions.

jayleaton reallocates bits toward code and science, but imports layer sensitivities measured on the September checkpoint. An unchanged router does not establish unchanged sensitivity when the expert weights have changed. Its cross-checkpoint comparisons cannot isolate whether transferring those priorities improved this quant. More elaborate allocation requires a stronger finished-model result, not merely a more elaborate rationale.

EAQ v1.1 illustrates the same problem at the component level. Restoring an indexer to source precision removes its weight approximation, but the upstream quantized network still supplies different inputs. Its own reported recall on those inputs exposes that distinction. Perfect agreement under source inputs is insufficient evidence of recovered attention decisions in the actual quant.

A finished quant must demonstrate its advantage with the same source checkpoint, the same complete histories and the same numerical reference. Quantseek measures the stored artifact across all 129,280 vocabulary entries and separate problem families. The results above measure the errors that remain after compression, rather than awarding credit for a method name, a nominal bit count or an isolated intermediate check.

## The Quantseek method

EXL3 defines how compressed weights are stored and reconstructed; it leaves substantial freedom in how those weights are chosen. Quantseek controls calibration, fitting, precision allocation and output refinement within that representation. The optimization is directed toward the behavior of the complete stored model, and the result retains ordinary EXL3 compatibility.

Calibration records the internal signals produced as the source model processes a workload, including their magnitudes and which signals occur together. Two equally small changes to a weight matrix can have very different effects on the computations the model actually performs. Our fitting uses those measured relationships to preserve layer outputs on substantive reasoning and tool-use histories. Complete context is retained: a late debugging step is conditioned on the work that preceded it. The reference also retains the original model's activation quantization, including the rounding applied to intermediate signals.

DeepSeek routes each token through selected expert networks, whose importance varies with the input. More bits allow a more accurate approximation at a higher storage cost. Giving every expert the same precision spends bytes on approximations that may already be adequate while leaving more consequential errors unresolved. Even within one expert, the gate/up projections that construct an intermediate representation and the down projection that returns it to the model have different precision requirements.

Quantseek compares stored alternatives and their error reduction relative to their storage cost when revising the allocation. The release assigns **2–5 bits** across **46,080 routed matrices**, averaging **2.87 bpw**. Gate/up projections share a precision choice; down projections are allocated separately. The output head uses **6 bits**. The average describes the budget; its placement determines which computations that budget protects.

Errors interact as information passes through the network. A matrix can become a better local approximation while the complete model's predictions become worse. A correction at the output stage can compensate for some of the error accumulated earlier. Our output-aware refinement compares the source and quantized models' probabilities across the entire vocabulary, including alternatives beyond the single most likely token. The output head—the final projection that produces token scores—is refined using states from the quantized model. The correction therefore addresses accumulated compression error. It is incorporated into the EXL3 weights; inference requires no additional correction module.

Compression can erase a gain measured in a higher-precision working representation. We reconstruct candidates from the codes and scaling factors actually saved in the files, then measure complete-model outputs across distinct problem families. A promising intermediate fit earns no credit unless the improvement survives in the stored weights and extends beyond the workload used to obtain it.

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

Load the directory with an ExLlamaV3 runtime supporting DeepSeek V4.1/1017. The index already includes Engram; no edits are needed. Vision and drafter weights are included.

MIT license. Base model by [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017); format and runtime by [ExLlamaV3](https://github.com/turboderp-org/exllamav3).
