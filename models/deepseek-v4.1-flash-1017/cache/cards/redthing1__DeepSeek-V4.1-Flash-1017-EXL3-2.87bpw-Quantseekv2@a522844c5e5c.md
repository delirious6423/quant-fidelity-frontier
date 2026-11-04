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

# DeepSeek-V4.1-Flash-1017 EXL3 2.87 bpw, Quantseek v2

[v1](https://huggingface.co/redthing1/DeepSeek-V4.1-Flash-1017-EXL3-2.87bpw) had 6.8–46.7% lower full-vocabulary KL than the
other EXL3 releases we compared. v2 is **15–35% lower than v1** on every family below and removes **68% of the divergence v1 left above the noise floor**, at the same size and memory.

KL is measured on the finished weights, on complete reasoning and tool-use histories, against the original model, over all
129,280 next tokens. Lower is closer.

| Family | Role | v1 | **v2** | Change |
| --- | --- | ---: | ---: | ---: |
| Sparse SPD solver | selection | 0.01520 | **0.00989** | **−34.9%** |
| Incremental SAT | selection | 0.01408 | **0.00985** | **−30.1%** |
| Register allocation | selection | 0.01407 | **0.01192** | **−15.2%** |
| Incremental DAG | selection | 0.01305 | **0.01113** | **−14.7%** |
| Link-cut forest | audit | 0.01630 | **0.01162** | **−28.7%** |
| Epoch skiplist | audit | 0.01486 | **0.01158** | **−22.1%** |
| Loop fusion | audit | 0.01296 | **0.01094** | **−15.6%** |
| SSA construction | audit | 0.01284 | **0.01090** | **−15.1%** |

Selection families chose between candidates. Audit families were never fitted on, and a pack is scored on them only after its
weights are frozen. The top token matches the original more often than v1 on every family.

## Floor

Changing one weight of the original by one unit in the last place already gives 0.007–0.011 KL on the selection families.
That's the floor. v2 removes 68% of what v1 had above it (60–74% per family). None of the releases below report a floor, so
their KL numbers can't say how much is left to remove.

## Long tasks

At maximum thinking, v2 solves all six long algorithm tasks (Myers diff, dominance frontiers, minimum mean cycles; two seeds each). Every run finishes on
its own after 26,000–55,000 generated tokens and passes every test (201/201, 121/121, 151/151), with at most 3.1% repeated 32-grams.

## Method

**Calibration.** Each matrix is fitted to input statistics captured inside the original model's own computation, on seven
complete reasoning and tool-use histories (1.13M tokens). Long context is kept: a late debugging step is conditioned on
the work before it.

**Measuring damage.** We run the original model forward and backward over those histories, FP8 activation rounding included.
The backward pass gives the gradient of the output at each routed expert. Combined with that expert's stored error, it gives the expert's Fisher contribution: the KL its error causes.

Within a layer, the same error costs a sensitive expert 10× what it costs an insensitive one (median), and up to 119×. The differences repeat on unrelated families (median correlation 0.75). Layer schedules can't see them. Neither can
allocations based on traffic alone.

**Allocating bits.** The starting allocation comes from whole-model sensitivity sweeps, each expert's realized error at every
rate, and its traffic. Then each expert's gate/up and down projections are re-priced by Fisher contribution, choosing from
2, 2.5, 3, 3.5, 4, 5 and 6 bits. Each layer keeps its stored bytes and its weight traffic, counted as how often an expert is
read times its bytes.

Fitted on five families and tested on two others, this step predicted a 35% cut in the routed contribution. In the complete
model, divergence above the floor fell 26%. Moving the output head to 8 bits cut another 7%.

**Scoring.** Candidates are rebuilt from the codes and scales saved in the files and run through the complete model.

v2 uses 2–6 bits across 46,080 routed matrices. Attention and shared-expert projections use 5–8 bits, the output head 8. Same 2.87-bpw budget as v1. The DSpark drafter is v1's.

## Other releases

None of their evaluations measures what needs preserving: fidelity on complete, long reasoning histories.

| Release | Evaluation | Problem |
| --- | --- | --- |
| [LibertAI 2.9](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/blob/0be63c3337efbb2a0061d5b8bb4006fa186d433f/README.md) | Perplexity on 32 × 2,048-token excerpts. Agentic benchmark on the **original weights**. | Says nothing about long reasoning in the quant. |
| [sudoingx 2.8](https://huggingface.co/sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw/blob/aba08ea002ae271ee2a2724bcc0c6dc6e1c457b8/README.md) | HumanEval+, MBPP+, IFEval. Thinking off, 2,048-token cap. | Excludes reasoning. |
| [EAQ v1](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw#quality-vs-the-original-model) | Eight synthetic sessions up to 31.2K tokens. Thinking off, assistant turns scripted. Port fails above 32,768 tokens and omits dynamic FP8 activation quantization. | No sustained problem solving. Different numerical reference. |
| [EAQ v1.1](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw/blob/aee640e2afba93be683bb5b9b420115fe7fbac20/README.md) | Perfect indexer recall on source hidden states. About 0.73 on each quant's own states, similar to EAQ v1. | The quant runs on its own states. |
| [jayleaton 2.8](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw/blob/859396591c340195526aaf61f94b4ad5323a5e18/README.md) | 356 thinking-on task items across 1017 and September packs. Every interval includes zero. | No ranking. The base checkpoint changes too. |

Scorers:

- The [earlier DGX Spark scorer](https://github.com/sudoingX/dgx-spark-deepseek-v41/blob/379edd6aec5b3c41569c45b61745dd034d3df574/scripts/quantize/p4_kld.py)
  keeps 128 tokens and collapses the rest into one bucket.
- EAQ v1 changes the activation arithmetic, so it approximates a different reference.

Recipes:

- LibertAI and EAQ v1 give all 1,152 routed matrices in a layer the same width: 2 bits in layers 18–22, 3 elsewhere.
- LibertAI calibrates on 250 sequences of 2,048 tokens. Its drafter has no calibration data.
- sudoingx allocates by layer sensitivity and routed-output share, which doesn't measure what a matrix loses between
  precisions.
- jayleaton uses layer sensitivities from the September checkpoint.
- EAQ v1.1 restores the indexer to source precision. The quantized network still feeds it different inputs.

Ours: same source checkpoint, same complete histories, same numerical reference, all 129,280 tokens.

## Download

206.9 GB, plus 203.1 GB for the two unchanged Engram shards from the official checkpoint.

```sh
model_dir=./DeepSeek-V4.1-Flash-1017-EXL3-2.87bpw-Quantseekv2
hf download redthing1/DeepSeek-V4.1-Flash-1017-EXL3-2.87bpw-Quantseekv2 --local-dir "$model_dir"
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir "$model_dir"
```

Needs an ExLlamaV3 runtime with DeepSeek V4.1/1017 and fractional (2.5- and 3.5-bit) EXL3 rates. The index already includes
Engram. Vision and drafter weights are included. `SHA256SUMS` covers this repository, `SHA256SUMS.engram` the Engram shards,
and `quantseek-formats.json` lists every matrix's stored format.

MIT license. Base model by [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). Format and runtime by
[ExLlamaV3](https://github.com/turboderp-org/exllamav3).
