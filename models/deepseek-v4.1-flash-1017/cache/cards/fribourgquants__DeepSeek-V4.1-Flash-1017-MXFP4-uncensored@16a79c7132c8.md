---
library_name: vllm
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
- pqhaz/DeepSeek-V4.1-Flash-1017-heretic
tags:
- deepseek
- mxfp4
- text-generation
- tool-use
pipeline_tag: text-generation
---

<!-- visual-comparisons -->
# Visual comparisons: Native mixed MXFP4

[Native mixed MXFP4](https://huggingface.co/Infatoshi/DeepSeek-V4.1-Flash-1017-MXFP4-uncensored) · [NVFP4 target](https://huggingface.co/Infatoshi/DeepSeek-V4.1-Flash-1017-NVFP4-uncensored) · [EXL3 2.875-bit experts](https://huggingface.co/Infatoshi/DeepSeek-V4.1-Flash-1017-EXL3-2.875bpw-uncensored) · [EXL3 3.5-bit experts](https://huggingface.co/Infatoshi/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-uncensored)

Four formats, one checkpoint family. Dark charts show measured behavior, coding, speed, latency and context comparisons; full numeric tables and protocols follow the figures.

### Did the edit change harmful-response behavior?

![Did the edit change harmful-response behavior?](figures/comparisons/chat_harmbench.png)

300 HarmBench prompts. Higher green share means more harmful compliance, not a generic refusal rate.

### JailbreakBench prompts: chat-mode comparison

![JailbreakBench prompts: chat-mode comparison](figures/comparisons/chat_jbb.png)

100 JBB prompts; auxiliary HarmBench grading, not official JBB scoring. Higher green share means more harmful compliance, not a generic refusal rate.

### JailbreakBench prompts: reasoning-mode comparison

![JailbreakBench prompts: reasoning-mode comparison](figures/comparisons/thinking_jbb.png)

100 JBB prompts; auxiliary HarmBench grading, not official JBB scoring. Higher green share means more harmful compliance, not a generic refusal rate.

### Capability retention

![Capability retention](figures/comparisons/capability.png)

HumanEval: network-disabled execution, 164 tasks. Strict MMLU: eight-token cap, format-sensitive. Small differences are not proof of superiority.

### Coding and harness checks

![Coding and harness checks](figures/comparisons/software.png)

HumanEval: one greedy completion, no repair, 3 s CPU/wall limit per case, 512 MiB address-space limit. Editing checks are bounded fixtures, not SWE-bench.

### Chat: throughput and interactivity

![Chat: throughput and interactivity](figures/comparisons/chat.png)

Three measured legs per point; bars show observed min–max, not confidence intervals. Native repeat has measured endpoints only.

### Warm agent: throughput and interactivity

![Warm agent: throughput and interactivity](figures/comparisons/warm.png)

Three measured legs per point; bars show observed min–max, not confidence intervals. Native repeat has measured endpoints only.

### Reasoning agent: throughput and response latency

![Reasoning agent: throughput and response latency](figures/comparisons/thinking.png)

Three measured legs per point; bars show observed min–max, not confidence intervals. Native repeat has measured endpoints only.

### Cold agent context: concurrency costs

![Cold agent context: concurrency costs](figures/comparisons/cold.png)

Two measured legs per point; bars show observed min–max, not confidence intervals.

### Draft acceptance by target format

![Draft acceptance by target format](figures/comparisons/format-drafting.png)

Three measured legs. Acceptance is a server-counter metric, not an end-to-end speedup guarantee.

### Drafter comparison on the native target

![Drafter comparison on the native target](figures/comparisons/drafter-ablation.png)

Three measured legs. Drafting-off reasoning was not measured. Same three-stage MTP weights; NVFP4 is a representation/calibration experiment.

### Long-input prefill

![Long-input prefill](figures/comparisons/prefill.png)

Three measurements per length. Effective rate is input tokens divided by first-event latency, not an isolated kernel rate. OS file cache was not standardized.

### Storage and context

![Storage and context](figures/comparisons/capacity-context.png)

300K total context includes input + output. Thirty-two scheduler slots do not prove 32 simultaneous 300K sessions. Harness fixtures are bounded checks.

### Inference settings

![Inference settings](figures/comparisons/settings.png)

Exact flags, package versions, image IDs and cache metadata are in comparison-data.json. Main target quantization differs from the optional NVFP4 drafter.

## How these comparisons were measured

These charts compare the same DeepSeek V4.1 Flash 1017 checkpoint family on four RTX PRO 6000 Blackwell 96GB GPUs. They include the original base and alternative edits where measured. Main expert quantization and serving implementations differ; this is a comparison of complete serving configurations, not an isolated quantization experiment.

**Reading the behavior charts:** harmful-response success means the pinned classifier judged that the final answer complied with the harmful benchmark request. It is not a generic refusal rate. HarmBench uses 300 direct prompts; JailbreakBench rows use 100 prompts with an auxiliary HarmBench rubric, not official JBB grading. The classifier passed 95/100 balanced validation controls. Unknown and empty cases remain visible; output-capped counts are disclosed.

**Reading the performance charts:** points are means of per-leg metrics; error bars show the observed minimum and maximum, not confidence intervals. Latency panels average each leg's median (or explicitly labeled percentile). Three legs were measured except cold-context waves, which used two. Natural output lengths and unrelated shared-GPU workloads vary. The white native-repeat markers are measured C1/C32 endpoints only.

All four released formats use the native DSpark multi-token prediction (MTP) drafter with five speculative positions. The NVFP4 **drafter** experiment changes drafting on the native target; it is distinct from the NVFP4 **target** checkpoint. Its gains are not assumed to transfer to EXL3. Both EXL3 implementations retain native MTP weights.

The coding evidence is executed HumanEval plus two real editing fixtures in each of Claude Code, OpenCode and OMP. HumanEval ran in a network-disabled, read-only container with isolated per-case CPU, memory and time limits. These are function-level coding and bounded integration checks. Repository-level SWE-bench has not been measured.

Native/MXFP4 is one mixed-format baseline. EXL3 bit rates refer to the main routed experts, not the whole model. Runtime-reported KV capacity and 32 scheduler slots do not establish 32 simultaneous full 300K sessions. Source measurements are from October 17–18, 2026.


## Numeric data and inference settings

The text tables below contain the values plotted above. [Download complete source measurements and exact runtime settings](evaluation/visual-comparison.json), including all individual legs, image IDs, package versions, server flags, cache metadata, classifier counts and source hashes. [Plain-text tables](evaluation/visual-comparison-tables.txt) are also available.

The four-format timing sweeps use a 300,000-token total context window and a maximum of 32 active sequences. Chat/warm/cold sweeps use temperature 0, top-p 1 and a 768-token output cap. Reasoning sweeps use temperature 1, top-p 1 and a 16,384-token cap at concurrency 1, 8 and 32. Prefill uses unique public-code prefixes and a short READY reply. Quality and behavior protocols have separate budgets; their exact settings are retained in the detailed text and JSON.

Recreate the figures after downloading the JSON and plotting script:

```bash
uv run --no-project --with matplotlib --with numpy python evaluation/plot-comparisons.py --bundle evaluation/visual-comparison.json --out comparison-figures
```

<details>
<summary>Full numeric comparison tables</summary>

### Behavior benchmark counts

| Configuration | Protocol | Successes | Attempted | Classified incl. automatic empty handling | Unknown | Empty | Output-capped |
|---|---|---|---|---|---|---|---|
| Unmodified base | chat_harmbench | 30 | 300 | 300 | 0 | 0 | 1 |
| Native MXFP4 | chat_harmbench | 287 | 300 | 300 | 0 | 0 | 20 |
| Native rank-2 edit | chat_harmbench | 285 | 300 | 299 | 1 | 0 | 18 |
| Native + NVFP4 drafter | chat_harmbench | 285 | 300 | 299 | 1 | 0 | 19 |
| NVFP4 target | chat_harmbench | 285 | 300 | 299 | 1 | 0 | 17 |
| EXL3 2.875 | chat_harmbench | 287 | 300 | 299 | 1 | 0 | 17 |
| EXL3 3.5 | chat_harmbench | 283 | 300 | 299 | 1 | 0 | 17 |
| Unmodified base | chat_jbb | 1 | 100 | 100 | 0 | 0 | 0 |
| Native MXFP4 | chat_jbb | 90 | 100 | 100 | 0 | 0 | 3 |
| Native rank-2 edit | chat_jbb | 94 | 100 | 100 | 0 | 0 | 1 |
| Native + NVFP4 drafter | chat_jbb | 91 | 100 | 100 | 0 | 0 | 2 |
| NVFP4 target | chat_jbb | 91 | 100 | 100 | 0 | 0 | 3 |
| EXL3 2.875 | chat_jbb | 94 | 100 | 100 | 0 | 0 | 3 |
| EXL3 3.5 | chat_jbb | 89 | 100 | 100 | 0 | 0 | 1 |
| Unmodified base | thinking_jbb | 2 | 100 | 100 | 0 | 0 | 1 |
| Native MXFP4 | thinking_jbb | 88 | 100 | 100 | 0 | 4 | 6 |
| Native rank-2 edit | thinking_jbb | 91 | 100 | 100 | 0 | 3 | 5 |
| Native + NVFP4 drafter | thinking_jbb | 89 | 100 | 100 | 0 | 5 | 7 |
| NVFP4 target | thinking_jbb | 85 | 100 | 100 | 0 | 6 | 10 |
| EXL3 2.875 | thinking_jbb | 91 | 100 | 100 | 0 | 2 | 5 |
| EXL3 3.5 | thinking_jbb | 87 | 100 | 100 | 0 | 4 | 6 |

### Capability outcomes

| Configuration | Suite | Correct | Items | Accuracy (%) |
|---|---|---|---|---|
| Unmodified base | gsm8k | 1279 | 1319 | 96.967 |
| Native MXFP4 | gsm8k | 1276 | 1319 | 96.74 |
| NVFP4 target | gsm8k | 1277 | 1319 | 96.816 |
| EXL3 2.875 | gsm8k | 1279 | 1319 | 96.967 |
| EXL3 3.5 | gsm8k | 1280 | 1319 | 97.043 |
| Unmodified base | humaneval | 157 | 164 | 95.732 |
| Native MXFP4 | humaneval | 156 | 164 | 95.122 |
| NVFP4 target | humaneval | 155 | 164 | 94.512 |
| EXL3 2.875 | humaneval | 158 | 164 | 96.341 |
| EXL3 3.5 | humaneval | 156 | 164 | 95.122 |
| Unmodified base | mmlu | 11920 | 14042 | 84.888 |
| Native MXFP4 | mmlu | 11824 | 14042 | 84.205 |
| NVFP4 target | mmlu | 11796 | 14042 | 84.005 |
| EXL3 2.875 | mmlu | 11859 | 14042 | 84.454 |
| EXL3 3.5 | mmlu | 11787 | 14042 | 83.941 |

### Real editing fixtures

| Configuration | Harness | Passed | Tasks |
|---|---|---|---|
| Native MXFP4 | claude | 2 | 2 |
| Native MXFP4 | opencode | 2 | 2 |
| Native MXFP4 | omp | 2 | 2 |
| NVFP4 target | claude | 2 | 2 |
| NVFP4 target | opencode | 2 | 2 |
| NVFP4 target | omp | 2 | 2 |
| EXL3 2.875 | claude | 2 | 2 |
| EXL3 2.875 | opencode | 2 | 2 |
| EXL3 2.875 | omp | 2 | 2 |
| EXL3 3.5 | claude | 2 | 2 |
| EXL3 3.5 | opencode | 2 | 2 |
| EXL3 3.5 | omp | 2 | 2 |

### Prefill measurements

| Configuration | Input target | Metric | Actual input tokens mean | Mean | Min | Max | Legs |
|---|---|---|---|---|---|---|---|
| Native MXFP4 | prefill-8000 | TTFT seconds | 8013.0 | 0.976 | 0.9253 | 1.058 | 3 |
| Native MXFP4 | prefill-32000 | TTFT seconds | 32377.0 | 3.9249 | 3.8259 | 4.0313 | 3 |
| Native MXFP4 | prefill-128000 | TTFT seconds | 124877.0 | 15.6061 | 15.0326 | 15.9036 | 3 |
| Native MXFP4 | prefill-285000 | TTFT seconds | 284026.0 | 37.8746 | 37.4236 | 38.151 | 3 |
| Native MXFP4 | prefill-8000 | input_tokens / TTFT | 8013.0 | 8238.9651 | 7573.673 | 8659.7106 | 3 |
| Native MXFP4 | prefill-32000 | input_tokens / TTFT | 32377.0 | 8252.9512 | 8031.4523 | 8462.6923 | 3 |
| Native MXFP4 | prefill-128000 | input_tokens / TTFT | 124877.0 | 8007.3205 | 7852.1359 | 8307.0651 | 3 |
| Native MXFP4 | prefill-285000 | input_tokens / TTFT | 284026.0 | 7499.6568 | 7444.7921 | 7589.4803 | 3 |
| NVFP4 target | prefill-8000 | TTFT seconds | 8013.0 | 0.9783 | 0.945 | 1.0143 | 3 |
| NVFP4 target | prefill-32000 | TTFT seconds | 32377.0 | 3.8418 | 3.7863 | 3.8957 | 3 |
| NVFP4 target | prefill-128000 | TTFT seconds | 124877.0 | 15.518 | 15.1835 | 15.8454 | 3 |
| NVFP4 target | prefill-285000 | TTFT seconds | 284026.0 | 37.1828 | 36.4498 | 38.0823 | 3 |
| NVFP4 target | prefill-8000 | input_tokens / TTFT | 8013.0 | 8197.3493 | 7900.4066 | 8479.5015 | 3 |
| NVFP4 target | prefill-32000 | input_tokens / TTFT | 32377.0 | 8428.8 | 8310.8607 | 8551.0841 | 3 |
| NVFP4 target | prefill-128000 | input_tokens / TTFT | 124877.0 | 8049.669 | 7880.9705 | 8224.4946 | 3 |
| NVFP4 target | prefill-285000 | input_tokens / TTFT | 284026.0 | 7641.1495 | 7458.2083 | 7792.2453 | 3 |
| EXL3 2.875 | prefill-8000 | TTFT seconds | 8013.0 | 1.684 | 1.6287 | 1.7419 | 3 |
| EXL3 2.875 | prefill-32000 | TTFT seconds | 32377.0 | 6.6477 | 6.4399 | 6.812 | 3 |
| EXL3 2.875 | prefill-128000 | TTFT seconds | 124877.0 | 26.3662 | 25.8326 | 26.9622 | 3 |
| EXL3 2.875 | prefill-285000 | TTFT seconds | 284026.0 | 60.5797 | 60.4318 | 60.7007 | 3 |
| EXL3 2.875 | prefill-8000 | input_tokens / TTFT | 8013.0 | 4761.8931 | 4600.0884 | 4919.8566 | 3 |
| EXL3 2.875 | prefill-32000 | input_tokens / TTFT | 32377.0 | 4873.0904 | 4752.9613 | 5027.556 | 3 |
| EXL3 2.875 | prefill-128000 | input_tokens / TTFT | 124877.0 | 4737.7179 | 4631.5648 | 4834.085 | 3 |
| EXL3 2.875 | prefill-285000 | input_tokens / TTFT | 284026.0 | 4688.4836 | 4679.1187 | 4699.9414 | 3 |
| EXL3 3.5 | prefill-8000 | TTFT seconds | 8013.0 | 1.7413 | 1.6327 | 1.7984 | 3 |
| EXL3 3.5 | prefill-32000 | TTFT seconds | 32377.0 | 6.5681 | 6.507 | 6.6358 | 3 |
| EXL3 3.5 | prefill-128000 | TTFT seconds | 124877.0 | 25.5807 | 25.2658 | 26.162 | 3 |
| EXL3 3.5 | prefill-285000 | TTFT seconds | 284026.0 | 59.7046 | 59.3943 | 60.1065 | 3 |
| EXL3 3.5 | prefill-8000 | input_tokens / TTFT | 8013.0 | 4610.9991 | 4455.7296 | 4907.9283 | 3 |
| EXL3 3.5 | prefill-32000 | input_tokens / TTFT | 32377.0 | 4929.7343 | 4879.1062 | 4975.7265 | 3 |
| EXL3 3.5 | prefill-128000 | input_tokens / TTFT | 124877.0 | 4882.9334 | 4773.2174 | 4942.5365 | 3 |
| EXL3 3.5 | prefill-285000 | input_tokens / TTFT | 284026.0 | 4757.3068 | 4725.3811 | 4782.0381 | 3 |

### Storage and capacity

| Configuration | Weight file bytes (decimal GB) | Runtime-reported KV capacity (million tokens) |
|---|---|---|
| Native MXFP4 | 510.296708 | 4.531484 |
| NVFP4 target | 527.293386 | 1.571012 |
| EXL3 2.875 | 417.561185 | 12.781811 |
| EXL3 3.5 | 460.028517 | 7.410551 |

### Long-context harness checks

| Configuration | Harness | Maximum observed input tokens | Compactions |
|---|---|---|---|
| Native MXFP4 | claude | 260205 | 0 |
| Native MXFP4 | opencode | 281805 | 0 |
| Native MXFP4 | omp | 259066 | 0 |
| NVFP4 target | claude | 258996 | 0 |
| NVFP4 target | opencode | 281560 | 0 |
| NVFP4 target | omp | 266240 | 0 |
| EXL3 2.875 | claude | 259666 | 0 |
| EXL3 2.875 | opencode | 281712 | 0 |
| EXL3 2.875 | omp | 253572 | 0 |
| EXL3 3.5 | claude | 259901 | 0 |
| EXL3 3.5 | opencode | 281801 | 0 |
| EXL3 3.5 | omp | 254469 | 0 |

### Inference configuration

| Setting | Native MXFP4 | NVFP4 target | EXL3 2.875 | EXL3 3.5 |
|---|---|---|---|---|
| Total context | 300,000 | 300,000 | 300,000 | 300,000 |
| Max active sequences | 32 | 32 | 32 | 32 |
| Native DSpark positions | 5 | 5 | 5 | 5 |
| KV dtype / prefix cache | FP8 / on | FP8 / on | FP8 / on | FP8 / on |
| Engram tables | Disk | Disk | Disk | Disk |
| Parallelism | TP4 | TP4 | TP4 + experts | TP4 + experts |
| Main expert format | Native MXFP4 mix | NVFP4, 4-bit acts | EXL3 2.875 bpw | EXL3 3.5 bpw |
| GPU memory utilization | 0.90 | 0.90 | 0.90 | 0.90 |
| Special numerical setting | Native B12X | Layer-max scales | Marlin + adapters | Marlin + adapters |
| EXL batch-token cap | n/a | n/a | 2,048 | 2,048 |

### Measured image identities

| Configuration | Total context | Max sequences | Drafter / positions | KV dtype | Engram | Parallelism | Measured image |
|---|---|---|---|---|---|---|---|
| Native MXFP4 | 300,000 | 32 | DSpark / 5 | FP8 | disk | TP4 | sha256:55e477ad62ae15a77c9b869e8fb8e2d958f6edcc4d95e306adfa89dee9ed19df |
| NVFP4 target | 300,000 | 32 | DSpark / 5 | FP8 | disk | TP4 | sha256:55e477ad62ae15a77c9b869e8fb8e2d958f6edcc4d95e306adfa89dee9ed19df |
| EXL3 2.875 | 300,000 | 32 | DSpark / 5 | FP8 | disk | TP4 + EP | sha256:6094b86f17a7523bdd9c95c291926c68f5071c643bf0b29b543c4fcf60b64804 |
| EXL3 3.5 | 300,000 | 32 | DSpark / 5 | FP8 | disk | TP4 + EP | sha256:6094b86f17a7523bdd9c95c291926c68f5071c643bf0b29b543c4fcf60b64804 |

### Four-format performance

| Configuration | Run | Workload | Metric | Unit | Concurrency | Mean | Min | Max | Legs |
|---|---|---|---|---|---|---|---|---|---|
| Native MXFP4 | performance-native | chat | aggregate_tokens_per_second | tokens / second | 1 | 146.84336 | 143.00183 | 152.41851 | 3 |
| Native MXFP4 | performance-native | chat | aggregate_tokens_per_second | tokens / second | 2 | 253.7539 | 239.67981 | 265.62086 | 3 |
| Native MXFP4 | performance-native | chat | aggregate_tokens_per_second | tokens / second | 4 | 353.40721 | 334.52948 | 380.12166 | 3 |
| Native MXFP4 | performance-native | chat | aggregate_tokens_per_second | tokens / second | 8 | 484.24483 | 459.64369 | 497.42598 | 3 |
| Native MXFP4 | performance-native | chat | aggregate_tokens_per_second | tokens / second | 16 | 632.00051 | 608.54906 | 644.91414 | 3 |
| Native MXFP4 | performance-native | chat | aggregate_tokens_per_second | tokens / second | 32 | 831.66003 | 817.35856 | 839.54636 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | aggregate_tokens_per_second | tokens / second | 1 | 101.5976 | 100.06832 | 104.32571 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | aggregate_tokens_per_second | tokens / second | 2 | 203.95368 | 182.45824 | 228.38624 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | aggregate_tokens_per_second | tokens / second | 4 | 317.37498 | 300.91128 | 328.00717 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | aggregate_tokens_per_second | tokens / second | 8 | 456.15917 | 440.04571 | 465.41321 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | aggregate_tokens_per_second | tokens / second | 16 | 602.27637 | 581.29177 | 631.66025 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | aggregate_tokens_per_second | tokens / second | 32 | 796.13354 | 776.68612 | 829.22279 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | aggregate_tokens_per_second | tokens / second | 1 | 125.86127 | 124.47899 | 126.65517 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | aggregate_tokens_per_second | tokens / second | 2 | 205.50923 | 201.15824 | 208.74818 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | aggregate_tokens_per_second | tokens / second | 4 | 308.74081 | 284.65356 | 323.66751 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | aggregate_tokens_per_second | tokens / second | 8 | 338.37693 | 330.36224 | 348.45852 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | aggregate_tokens_per_second | tokens / second | 16 | 423.10986 | 420.78376 | 425.60505 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | aggregate_tokens_per_second | tokens / second | 32 | 452.1589 | 448.04842 | 459.61018 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | aggregate_tokens_per_second | tokens / second | 1 | 129.30634 | 125.41362 | 136.35202 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | aggregate_tokens_per_second | tokens / second | 2 | 212.17761 | 205.26047 | 219.28676 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | aggregate_tokens_per_second | tokens / second | 4 | 277.95531 | 241.27994 | 306.73139 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | aggregate_tokens_per_second | tokens / second | 8 | 377.43362 | 339.46319 | 445.36018 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | aggregate_tokens_per_second | tokens / second | 16 | 433.61672 | 422.31715 | 450.00648 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | aggregate_tokens_per_second | tokens / second | 32 | 480.83948 | 476.23933 | 487.80518 | 3 |
| Native MXFP4 | performance-native | chat | time_to_content_p50 | seconds | 1 | 0.34266 | 0.33973 | 0.34812 | 3 |
| Native MXFP4 | performance-native | chat | time_to_content_p50 | seconds | 2 | 0.37674 | 0.33581 | 0.40198 | 3 |
| Native MXFP4 | performance-native | chat | time_to_content_p50 | seconds | 4 | 0.39836 | 0.37094 | 0.43444 | 3 |
| Native MXFP4 | performance-native | chat | time_to_content_p50 | seconds | 8 | 0.68985 | 0.57431 | 0.84347 | 3 |
| Native MXFP4 | performance-native | chat | time_to_content_p50 | seconds | 16 | 0.97568 | 0.77676 | 1.22938 | 3 |
| Native MXFP4 | performance-native | chat | time_to_content_p50 | seconds | 32 | 2.29967 | 2.10043 | 2.5757 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | time_to_content_p50 | seconds | 1 | 0.32442 | 0.31269 | 0.34651 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | time_to_content_p50 | seconds | 2 | 0.32688 | 0.25389 | 0.38465 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | time_to_content_p50 | seconds | 4 | 0.44332 | 0.37964 | 0.51376 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | time_to_content_p50 | seconds | 8 | 0.71395 | 0.44652 | 0.8623 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | time_to_content_p50 | seconds | 16 | 0.75337 | 0.52444 | 1.01211 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | time_to_content_p50 | seconds | 32 | 1.64073 | 1.55907 | 1.7074 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | time_to_content_p50 | seconds | 1 | 0.40811 | 0.39799 | 0.41968 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | time_to_content_p50 | seconds | 2 | 0.47041 | 0.46389 | 0.47814 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | time_to_content_p50 | seconds | 4 | 0.63093 | 0.48042 | 0.87726 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | time_to_content_p50 | seconds | 8 | 0.99986 | 0.93829 | 1.03243 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | time_to_content_p50 | seconds | 16 | 1.59909 | 1.30562 | 1.95452 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | time_to_content_p50 | seconds | 32 | 2.49889 | 2.08404 | 2.79295 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | time_to_content_p50 | seconds | 1 | 0.39463 | 0.35322 | 0.41724 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | time_to_content_p50 | seconds | 2 | 0.43001 | 0.38232 | 0.46608 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | time_to_content_p50 | seconds | 4 | 0.90732 | 0.80164 | 1.09043 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | time_to_content_p50 | seconds | 8 | 1.16691 | 0.94012 | 1.35132 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | time_to_content_p50 | seconds | 16 | 1.99368 | 1.88977 | 2.07182 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | time_to_content_p50 | seconds | 32 | 2.83945 | 2.4565 | 3.28051 | 3 |
| Native MXFP4 | performance-native | chat | seconds_p50 | seconds | 1 | 2.07962 | 2.00165 | 2.14744 | 3 |
| Native MXFP4 | performance-native | chat | seconds_p50 | seconds | 2 | 2.34247 | 2.17701 | 2.63299 | 3 |
| Native MXFP4 | performance-native | chat | seconds_p50 | seconds | 4 | 3.26501 | 3.05553 | 3.3793 | 3 |
| Native MXFP4 | performance-native | chat | seconds_p50 | seconds | 8 | 4.79355 | 4.71843 | 4.84687 | 3 |
| Native MXFP4 | performance-native | chat | seconds_p50 | seconds | 16 | 7.08973 | 6.85927 | 7.33351 | 3 |
| Native MXFP4 | performance-native | chat | seconds_p50 | seconds | 32 | 10.69993 | 10.39476 | 10.91367 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | seconds_p50 | seconds | 1 | 2.93173 | 2.74426 | 3.03016 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | seconds_p50 | seconds | 2 | 2.86831 | 2.53622 | 3.18621 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | seconds_p50 | seconds | 4 | 3.41571 | 3.14781 | 3.55025 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | seconds_p50 | seconds | 8 | 4.75976 | 4.29092 | 5.02014 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | seconds_p50 | seconds | 16 | 6.99896 | 6.38491 | 7.40025 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | seconds_p50 | seconds | 32 | 10.98337 | 10.81062 | 11.12165 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | seconds_p50 | seconds | 1 | 2.55601 | 2.49825 | 2.62658 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | seconds_p50 | seconds | 2 | 3.12043 | 3.09266 | 3.13648 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | seconds_p50 | seconds | 4 | 3.97198 | 3.71712 | 4.42378 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | seconds_p50 | seconds | 8 | 7.33614 | 7.2457 | 7.4225 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | seconds_p50 | seconds | 16 | 11.72988 | 11.50558 | 12.11176 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | seconds_p50 | seconds | 32 | 22.09488 | 21.73117 | 22.28491 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | seconds_p50 | seconds | 1 | 2.55346 | 2.46812 | 2.59867 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | seconds_p50 | seconds | 2 | 3.00548 | 2.75349 | 3.17767 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | seconds_p50 | seconds | 4 | 4.62882 | 4.20985 | 5.11545 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | seconds_p50 | seconds | 8 | 6.96049 | 5.7422 | 7.6873 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | seconds_p50 | seconds | 16 | 11.98044 | 11.74153 | 12.11481 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | seconds_p50 | seconds | 32 | 21.4683 | 21.31897 | 21.56155 | 3 |
| Native MXFP4 | performance-native | chat | tpot_p50 | milliseconds / token | 1 | 5.65401 | 5.39469 | 5.9322 | 3 |
| Native MXFP4 | performance-native | chat | tpot_p50 | milliseconds / token | 2 | 6.1796 | 5.74259 | 6.82685 | 3 |
| Native MXFP4 | performance-native | chat | tpot_p50 | milliseconds / token | 4 | 9.10782 | 8.55911 | 9.47948 | 3 |
| Native MXFP4 | performance-native | chat | tpot_p50 | milliseconds / token | 8 | 12.97839 | 12.25244 | 14.32146 | 3 |
| Native MXFP4 | performance-native | chat | tpot_p50 | milliseconds / token | 16 | 19.25342 | 17.55287 | 20.46761 | 3 |
| Native MXFP4 | performance-native | chat | tpot_p50 | milliseconds / token | 32 | 27.08697 | 25.576 | 27.92241 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | tpot_p50 | milliseconds / token | 1 | 8.72184 | 8.29458 | 8.93867 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | tpot_p50 | milliseconds / token | 2 | 8.04853 | 7.26558 | 8.75767 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | tpot_p50 | milliseconds / token | 4 | 9.41197 | 8.95171 | 9.80549 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | tpot_p50 | milliseconds / token | 8 | 13.36939 | 13.09081 | 13.68433 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | tpot_p50 | milliseconds / token | 16 | 20.96795 | 19.49804 | 22.11697 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | tpot_p50 | milliseconds / token | 32 | 29.85348 | 29.64301 | 30.00723 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | tpot_p50 | milliseconds / token | 1 | 6.64315 | 6.5422 | 6.69918 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | tpot_p50 | milliseconds / token | 2 | 7.9197 | 7.31351 | 8.30356 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | tpot_p50 | milliseconds / token | 4 | 10.07289 | 9.47472 | 10.82855 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | tpot_p50 | milliseconds / token | 8 | 19.86588 | 18.79043 | 20.4629 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | tpot_p50 | milliseconds / token | 16 | 31.3563 | 30.69159 | 31.70635 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | tpot_p50 | milliseconds / token | 32 | 59.95507 | 58.13466 | 61.21949 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | tpot_p50 | milliseconds / token | 1 | 6.52053 | 6.2742 | 6.67442 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | tpot_p50 | milliseconds / token | 2 | 7.67266 | 7.19342 | 8.23751 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | tpot_p50 | milliseconds / token | 4 | 11.07429 | 10.14218 | 12.58982 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | tpot_p50 | milliseconds / token | 8 | 17.34095 | 14.94947 | 19.41495 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | tpot_p50 | milliseconds / token | 16 | 30.40305 | 28.94073 | 31.37351 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | tpot_p50 | milliseconds / token | 32 | 56.5971 | 56.10198 | 56.95601 | 3 |
| Native MXFP4 | performance-native | agent-warm | aggregate_tokens_per_second | tokens / second | 1 | 130.52694 | 111.49517 | 163.9676 | 3 |
| Native MXFP4 | performance-native | agent-warm | aggregate_tokens_per_second | tokens / second | 2 | 188.39133 | 177.22634 | 200.91851 | 3 |
| Native MXFP4 | performance-native | agent-warm | aggregate_tokens_per_second | tokens / second | 4 | 296.49899 | 288.16012 | 305.27513 | 3 |
| Native MXFP4 | performance-native | agent-warm | aggregate_tokens_per_second | tokens / second | 8 | 407.58421 | 385.45154 | 425.79087 | 3 |
| Native MXFP4 | performance-native | agent-warm | aggregate_tokens_per_second | tokens / second | 16 | 576.39912 | 540.00026 | 624.94714 | 3 |
| Native MXFP4 | performance-native | agent-warm | aggregate_tokens_per_second | tokens / second | 32 | 740.89793 | 730.18056 | 756.63061 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | aggregate_tokens_per_second | tokens / second | 1 | 91.07435 | 86.38192 | 96.4983 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | aggregate_tokens_per_second | tokens / second | 2 | 172.15616 | 160.68037 | 185.44617 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | aggregate_tokens_per_second | tokens / second | 4 | 300.6907 | 285.43398 | 310.87107 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | aggregate_tokens_per_second | tokens / second | 8 | 424.71374 | 413.06429 | 443.54963 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | aggregate_tokens_per_second | tokens / second | 16 | 583.2482 | 572.53802 | 597.29954 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | aggregate_tokens_per_second | tokens / second | 32 | 768.23411 | 745.3165 | 789.53805 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | aggregate_tokens_per_second | tokens / second | 1 | 99.24683 | 98.96848 | 99.70826 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | aggregate_tokens_per_second | tokens / second | 2 | 162.5906 | 162.17425 | 162.90685 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | aggregate_tokens_per_second | tokens / second | 4 | 222.03222 | 218.03538 | 225.68481 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | aggregate_tokens_per_second | tokens / second | 8 | 284.61942 | 278.45988 | 291.15043 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | aggregate_tokens_per_second | tokens / second | 16 | 354.7244 | 338.10297 | 386.06694 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | aggregate_tokens_per_second | tokens / second | 32 | 394.01271 | 391.61436 | 395.91315 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | aggregate_tokens_per_second | tokens / second | 1 | 101.71798 | 100.58921 | 102.47016 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | aggregate_tokens_per_second | tokens / second | 2 | 167.47528 | 161.28186 | 173.92958 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | aggregate_tokens_per_second | tokens / second | 4 | 234.92273 | 225.60065 | 248.76625 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | aggregate_tokens_per_second | tokens / second | 8 | 284.34379 | 275.35014 | 291.25758 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | aggregate_tokens_per_second | tokens / second | 16 | 362.21754 | 340.22932 | 399.06459 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | aggregate_tokens_per_second | tokens / second | 32 | 398.04558 | 392.76068 | 405.30399 | 3 |
| Native MXFP4 | performance-native | agent-warm | time_to_content_p50 | seconds | 1 | 0.39946 | 0.27548 | 0.46488 | 3 |
| Native MXFP4 | performance-native | agent-warm | time_to_content_p50 | seconds | 2 | 0.53012 | 0.51484 | 0.5558 | 3 |
| Native MXFP4 | performance-native | agent-warm | time_to_content_p50 | seconds | 4 | 0.64647 | 0.53421 | 0.84383 | 3 |
| Native MXFP4 | performance-native | agent-warm | time_to_content_p50 | seconds | 8 | 0.90003 | 0.68961 | 1.08792 | 3 |
| Native MXFP4 | performance-native | agent-warm | time_to_content_p50 | seconds | 16 | 1.13929 | 1.06693 | 1.27712 | 3 |
| Native MXFP4 | performance-native | agent-warm | time_to_content_p50 | seconds | 32 | 1.61797 | 1.5089 | 1.68171 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | time_to_content_p50 | seconds | 1 | 0.46477 | 0.4459 | 0.48494 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | time_to_content_p50 | seconds | 2 | 0.49962 | 0.48613 | 0.51801 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | time_to_content_p50 | seconds | 4 | 0.58397 | 0.56306 | 0.62252 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | time_to_content_p50 | seconds | 8 | 0.90218 | 0.8818 | 0.94005 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | time_to_content_p50 | seconds | 16 | 1.25193 | 1.12351 | 1.33682 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | time_to_content_p50 | seconds | 32 | 1.86091 | 1.597 | 2.24528 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | time_to_content_p50 | seconds | 1 | 0.46625 | 0.45883 | 0.47654 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | time_to_content_p50 | seconds | 2 | 0.4899 | 0.48002 | 0.50422 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | time_to_content_p50 | seconds | 4 | 0.78631 | 0.53225 | 0.93015 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | time_to_content_p50 | seconds | 8 | 0.93493 | 0.74315 | 1.06133 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | time_to_content_p50 | seconds | 16 | 1.16504 | 1.09807 | 1.24581 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | time_to_content_p50 | seconds | 32 | 1.95558 | 1.81513 | 2.21232 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | time_to_content_p50 | seconds | 1 | 0.44381 | 0.43159 | 0.45745 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | time_to_content_p50 | seconds | 2 | 0.46973 | 0.44478 | 0.48883 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | time_to_content_p50 | seconds | 4 | 0.57087 | 0.52453 | 0.64476 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | time_to_content_p50 | seconds | 8 | 0.78605 | 0.77014 | 0.79932 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | time_to_content_p50 | seconds | 16 | 1.08251 | 0.82574 | 1.28174 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | time_to_content_p50 | seconds | 32 | 1.90188 | 1.63553 | 2.2205 | 3 |
| Native MXFP4 | performance-native | agent-warm | seconds_p50 | seconds | 1 | 1.9272 | 1.54117 | 2.15092 | 3 |
| Native MXFP4 | performance-native | agent-warm | seconds_p50 | seconds | 2 | 2.63223 | 2.53685 | 2.79752 | 3 |
| Native MXFP4 | performance-native | agent-warm | seconds_p50 | seconds | 4 | 3.15681 | 2.83432 | 3.35548 | 3 |
| Native MXFP4 | performance-native | agent-warm | seconds_p50 | seconds | 8 | 4.84493 | 4.61544 | 5.151 | 3 |
| Native MXFP4 | performance-native | agent-warm | seconds_p50 | seconds | 16 | 6.44046 | 6.00151 | 6.6731 | 3 |
| Native MXFP4 | performance-native | agent-warm | seconds_p50 | seconds | 32 | 9.75683 | 9.59071 | 9.91722 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | seconds_p50 | seconds | 1 | 2.93226 | 2.87764 | 3.00125 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | seconds_p50 | seconds | 2 | 3.18985 | 2.93064 | 3.35779 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | seconds_p50 | seconds | 4 | 3.28137 | 3.01799 | 3.42274 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | seconds_p50 | seconds | 8 | 4.98683 | 4.71811 | 5.23141 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | seconds_p50 | seconds | 16 | 6.94453 | 6.65759 | 7.28863 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | seconds_p50 | seconds | 32 | 9.87734 | 9.78521 | 9.94193 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | seconds_p50 | seconds | 1 | 2.18971 | 2.08889 | 2.34071 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | seconds_p50 | seconds | 2 | 2.51549 | 2.47191 | 2.58913 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | seconds_p50 | seconds | 4 | 3.80743 | 3.57168 | 4.02832 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | seconds_p50 | seconds | 8 | 6.17082 | 6.07172 | 6.24152 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | seconds_p50 | seconds | 16 | 9.9741 | 9.62781 | 10.16752 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | seconds_p50 | seconds | 32 | 17.81818 | 17.70017 | 17.93869 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | seconds_p50 | seconds | 1 | 2.26397 | 2.18385 | 2.35574 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | seconds_p50 | seconds | 2 | 2.6639 | 2.57866 | 2.74945 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | seconds_p50 | seconds | 4 | 3.93626 | 3.56585 | 4.51049 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | seconds_p50 | seconds | 8 | 6.24781 | 6.16154 | 6.37154 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | seconds_p50 | seconds | 16 | 9.59321 | 8.25522 | 10.35341 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | seconds_p50 | seconds | 32 | 18.71774 | 18.33827 | 19.14632 | 3 |
| Native MXFP4 | performance-native | agent-warm | tpot_p50 | milliseconds / token | 1 | 6.16486 | 5.00254 | 6.86703 | 3 |
| Native MXFP4 | performance-native | agent-warm | tpot_p50 | milliseconds / token | 2 | 7.68464 | 6.5501 | 8.78323 | 3 |
| Native MXFP4 | performance-native | agent-warm | tpot_p50 | milliseconds / token | 4 | 9.79436 | 9.45242 | 10.12384 | 3 |
| Native MXFP4 | performance-native | agent-warm | tpot_p50 | milliseconds / token | 8 | 14.46394 | 13.55239 | 16.18054 | 3 |
| Native MXFP4 | performance-native | agent-warm | tpot_p50 | milliseconds / token | 16 | 20.52346 | 18.4226 | 22.18795 | 3 |
| Native MXFP4 | performance-native | agent-warm | tpot_p50 | milliseconds / token | 32 | 31.88416 | 31.64098 | 32.081 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | tpot_p50 | milliseconds / token | 1 | 9.04545 | 8.61781 | 9.40601 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | tpot_p50 | milliseconds / token | 2 | 9.05254 | 8.9195 | 9.13917 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | tpot_p50 | milliseconds / token | 4 | 9.83629 | 9.37789 | 10.75167 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | tpot_p50 | milliseconds / token | 8 | 14.1708 | 13.22101 | 14.75885 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | tpot_p50 | milliseconds / token | 16 | 20.40439 | 20.20244 | 20.52522 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | tpot_p50 | milliseconds / token | 32 | 29.64431 | 28.84454 | 31.0172 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | tpot_p50 | milliseconds / token | 1 | 7.72442 | 7.67305 | 7.81863 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | tpot_p50 | milliseconds / token | 2 | 9.2558 | 9.09254 | 9.45878 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | tpot_p50 | milliseconds / token | 4 | 13.57831 | 13.16682 | 14.01451 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | tpot_p50 | milliseconds / token | 8 | 23.04193 | 22.71527 | 23.34042 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | tpot_p50 | milliseconds / token | 16 | 37.33603 | 33.07772 | 39.73994 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | tpot_p50 | milliseconds / token | 32 | 69.60419 | 68.06706 | 70.78899 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | tpot_p50 | milliseconds / token | 1 | 7.97066 | 7.83409 | 8.12041 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | tpot_p50 | milliseconds / token | 2 | 9.58847 | 8.96415 | 10.28063 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | tpot_p50 | milliseconds / token | 4 | 13.39637 | 12.64157 | 14.09931 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | tpot_p50 | milliseconds / token | 8 | 23.19567 | 22.47555 | 23.94472 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | tpot_p50 | milliseconds / token | 16 | 36.79221 | 33.05104 | 39.31548 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | tpot_p50 | milliseconds / token | 32 | 67.88742 | 67.2325 | 68.91056 | 3 |
| Native MXFP4 | performance-native-thinking-16k | agent-warm | aggregate_tokens_per_second | tokens / second | 1 | 131.54642 | 123.70557 | 138.70554 | 3 |
| Native MXFP4 | performance-native-thinking-16k | agent-warm | aggregate_tokens_per_second | tokens / second | 8 | 425.18581 | 406.97905 | 436.23525 | 3 |
| Native MXFP4 | performance-native-thinking-16k | agent-warm | aggregate_tokens_per_second | tokens / second | 32 | 830.05849 | 815.20054 | 857.75188 | 3 |
| NVFP4 target | performance-nvtarget-layermax-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 1 | 86.58328 | 85.26911 | 89.08693 | 3 |
| NVFP4 target | performance-nvtarget-layermax-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 8 | 363.04805 | 358.81172 | 367.14952 | 3 |
| NVFP4 target | performance-nvtarget-layermax-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 32 | 780.69475 | 715.85104 | 831.54859 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 1 | 101.0078 | 95.91425 | 104.51652 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 8 | 330.33182 | 319.23785 | 336.32766 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 32 | 526.01245 | 511.48635 | 536.53162 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 1 | 102.37938 | 100.27231 | 104.5859 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 8 | 318.245 | 297.12008 | 331.04593 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 32 | 522.61707 | 509.91344 | 541.27268 | 3 |
| Native MXFP4 | performance-native-thinking-16k | agent-warm | time_to_content_p50 | seconds | 1 | 9.65073 | 6.35064 | 14.41695 | 3 |
| Native MXFP4 | performance-native-thinking-16k | agent-warm | time_to_content_p50 | seconds | 8 | 16.76038 | 14.67187 | 19.68185 | 3 |
| Native MXFP4 | performance-native-thinking-16k | agent-warm | time_to_content_p50 | seconds | 32 | 30.13698 | 28.98528 | 31.87614 | 3 |
| NVFP4 target | performance-nvtarget-layermax-thinking | agent-warm | time_to_content_p50 | seconds | 1 | 11.68203 | 10.55912 | 12.91683 | 3 |
| NVFP4 target | performance-nvtarget-layermax-thinking | agent-warm | time_to_content_p50 | seconds | 8 | 17.87678 | 15.70633 | 21.63886 | 3 |
| NVFP4 target | performance-nvtarget-layermax-thinking | agent-warm | time_to_content_p50 | seconds | 32 | 34.46783 | 34.05073 | 35.14498 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1-thinking | agent-warm | time_to_content_p50 | seconds | 1 | 9.87362 | 6.88966 | 15.83938 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1-thinking | agent-warm | time_to_content_p50 | seconds | 8 | 25.21921 | 21.61221 | 27.55211 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1-thinking | agent-warm | time_to_content_p50 | seconds | 32 | 58.10381 | 54.98353 | 63.79067 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1-thinking | agent-warm | time_to_content_p50 | seconds | 1 | 8.97122 | 7.75622 | 10.97217 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1-thinking | agent-warm | time_to_content_p50 | seconds | 8 | 21.41952 | 20.76303 | 21.89749 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1-thinking | agent-warm | time_to_content_p50 | seconds | 32 | 56.1705 | 50.28275 | 59.14061 | 3 |
| Native MXFP4 | performance-native-thinking-16k | agent-warm | seconds_p50 | seconds | 1 | 10.82997 | 7.89417 | 15.21746 | 3 |
| Native MXFP4 | performance-native-thinking-16k | agent-warm | seconds_p50 | seconds | 8 | 19.9709 | 18.2953 | 23.18694 | 3 |
| Native MXFP4 | performance-native-thinking-16k | agent-warm | seconds_p50 | seconds | 32 | 34.52788 | 33.53185 | 36.3199 | 3 |
| NVFP4 target | performance-nvtarget-layermax-thinking | agent-warm | seconds_p50 | seconds | 1 | 13.74261 | 12.10677 | 15.07679 | 3 |
| NVFP4 target | performance-nvtarget-layermax-thinking | agent-warm | seconds_p50 | seconds | 8 | 20.9148 | 18.99231 | 24.56765 | 3 |
| NVFP4 target | performance-nvtarget-layermax-thinking | agent-warm | seconds_p50 | seconds | 32 | 40.25575 | 39.44607 | 41.37337 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1-thinking | agent-warm | seconds_p50 | seconds | 1 | 11.5135 | 8.60881 | 17.14334 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1-thinking | agent-warm | seconds_p50 | seconds | 8 | 29.15695 | 25.78508 | 31.09867 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1-thinking | agent-warm | seconds_p50 | seconds | 32 | 66.4614 | 62.69332 | 72.68506 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1-thinking | agent-warm | seconds_p50 | seconds | 1 | 10.65959 | 9.50648 | 12.55364 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1-thinking | agent-warm | seconds_p50 | seconds | 8 | 25.7245 | 25.27752 | 26.31983 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1-thinking | agent-warm | seconds_p50 | seconds | 32 | 64.31096 | 58.20681 | 69.14341 | 3 |
| Native MXFP4 | performance-native-thinking-16k | agent-warm | ttft_p50 | seconds | 1 | 0.42973 | 0.39265 | 0.44904 | 3 |
| Native MXFP4 | performance-native-thinking-16k | agent-warm | ttft_p50 | seconds | 8 | 0.57659 | 0.54334 | 0.60286 | 3 |
| Native MXFP4 | performance-native-thinking-16k | agent-warm | ttft_p50 | seconds | 32 | 0.68577 | 0.65981 | 0.70401 | 3 |
| NVFP4 target | performance-nvtarget-layermax-thinking | agent-warm | ttft_p50 | seconds | 1 | 0.47224 | 0.42333 | 0.53543 | 3 |
| NVFP4 target | performance-nvtarget-layermax-thinking | agent-warm | ttft_p50 | seconds | 8 | 0.55125 | 0.52666 | 0.57333 | 3 |
| NVFP4 target | performance-nvtarget-layermax-thinking | agent-warm | ttft_p50 | seconds | 32 | 0.72161 | 0.70045 | 0.73574 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1-thinking | agent-warm | ttft_p50 | seconds | 1 | 0.46551 | 0.42622 | 0.49055 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1-thinking | agent-warm | ttft_p50 | seconds | 8 | 0.55802 | 0.55155 | 0.56897 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1-thinking | agent-warm | ttft_p50 | seconds | 32 | 0.79293 | 0.79177 | 0.79508 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1-thinking | agent-warm | ttft_p50 | seconds | 1 | 0.43042 | 0.39875 | 0.44835 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1-thinking | agent-warm | ttft_p50 | seconds | 8 | 0.56629 | 0.54607 | 0.58045 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1-thinking | agent-warm | ttft_p50 | seconds | 32 | 0.78931 | 0.77584 | 0.80397 | 3 |
| Native MXFP4 | performance-native-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 1 | 37.09545 | 34.2548 | 39.93611 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 2 | 42.26265 | 35.4083 | 49.11699 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 4 | 35.71412 | 34.18516 | 37.24308 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 8 | 37.52894 | 37.38545 | 37.67243 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 16 | 38.34666 | 36.71134 | 39.98197 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 32 | 39.19946 | 39.1334 | 39.26552 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 1 | 36.73101 | 36.03015 | 37.43187 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 2 | 35.80428 | 34.51568 | 37.09287 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 4 | 37.88717 | 37.46852 | 38.30581 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 8 | 38.19647 | 37.79818 | 38.59477 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 16 | 38.69983 | 37.83821 | 39.56145 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 32 | 37.33845 | 36.26023 | 38.41667 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 1 | 26.56683 | 21.5744 | 31.55926 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 2 | 28.32738 | 26.98783 | 29.66694 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 4 | 28.51585 | 27.66892 | 29.36277 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 8 | 27.16716 | 26.20939 | 28.12493 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 16 | 29.01148 | 26.8516 | 31.17136 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 32 | 28.94365 | 28.74247 | 29.14483 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 1 | 24.36912 | 23.58587 | 25.15237 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 2 | 26.42711 | 26.21068 | 26.64355 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 4 | 25.51421 | 25.35099 | 25.67744 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 8 | 29.54467 | 29.22103 | 29.86831 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 16 | 29.1293 | 28.07231 | 30.1863 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | aggregate_tokens_per_second | tokens / second | 32 | 28.76608 | 27.89788 | 29.63427 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | time_to_content_p50 | seconds | 1 | 5.18192 | 4.98182 | 5.38203 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | time_to_content_p50 | seconds | 2 | 5.68228 | 5.52343 | 5.84112 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | time_to_content_p50 | seconds | 4 | 11.62945 | 11.53337 | 11.72554 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | time_to_content_p50 | seconds | 8 | 24.70868 | 24.30405 | 25.1133 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | time_to_content_p50 | seconds | 16 | 51.84103 | 51.62704 | 52.05503 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | time_to_content_p50 | seconds | 32 | 104.61823 | 104.43082 | 104.80565 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | time_to_content_p50 | seconds | 1 | 5.32908 | 5.19356 | 5.46461 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | time_to_content_p50 | seconds | 2 | 5.56762 | 5.42286 | 5.71237 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | time_to_content_p50 | seconds | 4 | 13.22583 | 13.14096 | 13.3107 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | time_to_content_p50 | seconds | 8 | 27.96082 | 27.884 | 28.03765 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | time_to_content_p50 | seconds | 16 | 55.21617 | 52.80502 | 57.62733 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | time_to_content_p50 | seconds | 32 | 115.86818 | 115.23244 | 116.50391 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | time_to_content_p50 | seconds | 1 | 8.44667 | 8.33225 | 8.56108 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | time_to_content_p50 | seconds | 2 | 8.54046 | 8.0668 | 9.01412 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | time_to_content_p50 | seconds | 4 | 17.48073 | 16.92448 | 18.03699 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | time_to_content_p50 | seconds | 8 | 34.71388 | 34.4651 | 34.96265 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | time_to_content_p50 | seconds | 16 | 69.95614 | 69.69839 | 70.21389 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | time_to_content_p50 | seconds | 32 | 140.43261 | 140.25519 | 140.61004 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | time_to_content_p50 | seconds | 1 | 8.47765 | 8.33383 | 8.62147 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | time_to_content_p50 | seconds | 2 | 8.66083 | 8.33916 | 8.9825 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | time_to_content_p50 | seconds | 4 | 16.73162 | 16.02546 | 17.43778 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | time_to_content_p50 | seconds | 8 | 33.62977 | 33.47659 | 33.78295 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | time_to_content_p50 | seconds | 16 | 69.03042 | 68.91225 | 69.14859 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | time_to_content_p50 | seconds | 32 | 133.70944 | 129.65858 | 137.7603 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | seconds_p50 | seconds | 1 | 6.71391 | 6.39239 | 7.03543 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | seconds_p50 | seconds | 2 | 9.0402 | 9.0147 | 9.06569 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | seconds_p50 | seconds | 4 | 14.52441 | 14.49223 | 14.55659 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | seconds_p50 | seconds | 8 | 27.52614 | 27.11041 | 27.94187 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | seconds_p50 | seconds | 16 | 54.31305 | 53.93642 | 54.68968 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | seconds_p50 | seconds | 32 | 107.59325 | 107.57229 | 107.61422 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | seconds_p50 | seconds | 1 | 8.07807 | 7.74692 | 8.40922 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | seconds_p50 | seconds | 2 | 10.08086 | 9.68668 | 10.47503 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | seconds_p50 | seconds | 4 | 18.10612 | 17.63694 | 18.57531 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | seconds_p50 | seconds | 8 | 32.21708 | 31.95068 | 32.48348 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | seconds_p50 | seconds | 16 | 60.52938 | 58.15115 | 62.9076 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | seconds_p50 | seconds | 32 | 120.13035 | 119.62033 | 120.64037 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | seconds_p50 | seconds | 1 | 10.52263 | 10.33574 | 10.70953 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | seconds_p50 | seconds | 2 | 18.32142 | 17.98262 | 18.66022 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | seconds_p50 | seconds | 4 | 35.33009 | 34.41276 | 36.24742 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | seconds_p50 | seconds | 8 | 60.87809 | 60.38619 | 61.37 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | seconds_p50 | seconds | 16 | 102.08741 | 97.08785 | 107.08697 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | seconds_p50 | seconds | 32 | 171.28692 | 168.75462 | 173.81922 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | seconds_p50 | seconds | 1 | 10.38284 | 10.33627 | 10.42942 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | seconds_p50 | seconds | 2 | 18.29279 | 17.45791 | 19.12768 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | seconds_p50 | seconds | 4 | 34.77896 | 34.25765 | 35.30027 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | seconds_p50 | seconds | 8 | 59.76957 | 58.693 | 60.84614 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | seconds_p50 | seconds | 16 | 96.54082 | 95.40399 | 97.67766 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | seconds_p50 | seconds | 32 | 162.14771 | 158.07962 | 166.2158 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | ttft_p90 | seconds | 1 | 5.18192 | 4.98182 | 5.38203 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | ttft_p90 | seconds | 2 | 12.26395 | 12.10403 | 12.42386 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | ttft_p90 | seconds | 4 | 24.89197 | 24.62592 | 25.15802 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | ttft_p90 | seconds | 8 | 50.36035 | 49.84033 | 50.88038 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | ttft_p90 | seconds | 16 | 95.90339 | 94.45903 | 97.34775 | 2 |
| Native MXFP4 | performance-native-cold-context | agent-cold | ttft_p90 | seconds | 32 | 190.80896 | 190.63279 | 190.98513 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | ttft_p90 | seconds | 1 | 5.32908 | 5.19356 | 5.46461 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | ttft_p90 | seconds | 2 | 12.77519 | 12.36238 | 13.188 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | ttft_p90 | seconds | 4 | 28.11998 | 27.87153 | 28.36844 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | ttft_p90 | seconds | 8 | 57.26009 | 57.25461 | 57.26557 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | ttft_p90 | seconds | 16 | 106.77585 | 104.42536 | 109.12633 | 2 |
| NVFP4 target | performance-nvtarget-layermax-cold-context | agent-cold | ttft_p90 | seconds | 32 | 211.07386 | 209.9392 | 212.20851 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | ttft_p90 | seconds | 1 | 8.44667 | 8.33225 | 8.56108 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | ttft_p90 | seconds | 2 | 16.90326 | 16.39156 | 17.41496 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | ttft_p90 | seconds | 4 | 34.63736 | 33.86985 | 35.40487 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | ttft_p90 | seconds | 8 | 69.85017 | 69.41577 | 70.28457 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | ttft_p90 | seconds | 16 | 128.25204 | 125.32789 | 131.17619 | 2 |
| EXL3 2.875 | performance-exl3-sparsechunks1-cold-context | agent-cold | ttft_p90 | seconds | 32 | 253.98491 | 253.06146 | 254.90836 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | ttft_p90 | seconds | 1 | 8.47765 | 8.33383 | 8.62147 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | ttft_p90 | seconds | 2 | 16.83853 | 16.06121 | 17.61585 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | ttft_p90 | seconds | 4 | 33.88283 | 33.15767 | 34.60799 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | ttft_p90 | seconds | 8 | 68.17744 | 67.69177 | 68.6631 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | ttft_p90 | seconds | 16 | 128.93706 | 128.32091 | 129.55321 | 2 |
| EXL3 3.5 | performance-exl35-sparsechunks1-cold-context | agent-cold | ttft_p90 | seconds | 32 | 246.28565 | 241.12561 | 251.44569 | 2 |
| Native MXFP4 | performance-native | chat | draft_acceptance_fraction | fraction | 1 | 0.7801 | 0.77032 | 0.78717 | 3 |
| Native MXFP4 | performance-native | chat | draft_acceptance_fraction | fraction | 2 | 0.67325 | 0.65582 | 0.69242 | 3 |
| Native MXFP4 | performance-native | chat | draft_acceptance_fraction | fraction | 4 | 0.64758 | 0.64349 | 0.65475 | 3 |
| Native MXFP4 | performance-native | chat | draft_acceptance_fraction | fraction | 8 | 0.71185 | 0.70188 | 0.72142 | 3 |
| Native MXFP4 | performance-native | chat | draft_acceptance_fraction | fraction | 16 | 0.75915 | 0.74867 | 0.76503 | 3 |
| Native MXFP4 | performance-native | chat | draft_acceptance_fraction | fraction | 32 | 0.80676 | 0.80168 | 0.80969 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | draft_acceptance_fraction | fraction | 1 | 0.77769 | 0.77029 | 0.78809 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | draft_acceptance_fraction | fraction | 2 | 0.74746 | 0.7418 | 0.7522 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | draft_acceptance_fraction | fraction | 4 | 0.62609 | 0.62045 | 0.62929 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | draft_acceptance_fraction | fraction | 8 | 0.62305 | 0.61838 | 0.62708 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | draft_acceptance_fraction | fraction | 16 | 0.74476 | 0.73967 | 0.74843 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | draft_acceptance_fraction | fraction | 32 | 0.80624 | 0.8027 | 0.81002 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | draft_acceptance_fraction | fraction | 1 | 0.64391 | 0.63942 | 0.64967 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | draft_acceptance_fraction | fraction | 2 | 0.64418 | 0.64045 | 0.64765 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | draft_acceptance_fraction | fraction | 4 | 0.63121 | 0.625 | 0.63739 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | draft_acceptance_fraction | fraction | 8 | 0.63683 | 0.62905 | 0.64814 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | draft_acceptance_fraction | fraction | 16 | 0.63377 | 0.62674 | 0.63833 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | draft_acceptance_fraction | fraction | 32 | 0.63058 | 0.62846 | 0.63193 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | draft_acceptance_fraction | fraction | 1 | 0.65068 | 0.64394 | 0.6602 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | draft_acceptance_fraction | fraction | 2 | 0.64784 | 0.6387 | 0.65546 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | draft_acceptance_fraction | fraction | 4 | 0.65143 | 0.64854 | 0.65683 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | draft_acceptance_fraction | fraction | 8 | 0.65349 | 0.64802 | 0.6566 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | draft_acceptance_fraction | fraction | 16 | 0.65941 | 0.65513 | 0.6626 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | draft_acceptance_fraction | fraction | 32 | 0.65373 | 0.65085 | 0.65672 | 3 |
| Native MXFP4 | performance-native | chat | accepted_draft_tokens_per_step | tokens / step | 1 | 2.81675 | 2.77504 | 2.86435 | 3 |
| Native MXFP4 | performance-native | chat | accepted_draft_tokens_per_step | tokens / step | 2 | 3.11759 | 3.08 | 3.18792 | 3 |
| Native MXFP4 | performance-native | chat | accepted_draft_tokens_per_step | tokens / step | 4 | 3.20194 | 3.1808 | 3.23717 | 3 |
| Native MXFP4 | performance-native | chat | accepted_draft_tokens_per_step | tokens / step | 8 | 3.09469 | 3.07831 | 3.11044 | 3 |
| Native MXFP4 | performance-native | chat | accepted_draft_tokens_per_step | tokens / step | 16 | 2.9491 | 2.92233 | 2.96514 | 3 |
| Native MXFP4 | performance-native | chat | accepted_draft_tokens_per_step | tokens / step | 32 | 2.77817 | 2.75071 | 2.81129 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | accepted_draft_tokens_per_step | tokens / step | 1 | 2.36467 | 2.31433 | 2.39528 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | accepted_draft_tokens_per_step | tokens / step | 2 | 2.51097 | 2.47779 | 2.5622 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | accepted_draft_tokens_per_step | tokens / step | 4 | 3.06056 | 3.03765 | 3.07569 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | accepted_draft_tokens_per_step | tokens / step | 8 | 3.09214 | 3.062 | 3.11003 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | accepted_draft_tokens_per_step | tokens / step | 16 | 2.92955 | 2.88165 | 2.96841 | 3 |
| NVFP4 target | performance-nvtarget-layermax | chat | accepted_draft_tokens_per_step | tokens / step | 32 | 2.75217 | 2.74643 | 2.75553 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | accepted_draft_tokens_per_step | tokens / step | 1 | 3.21953 | 3.19712 | 3.24836 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | accepted_draft_tokens_per_step | tokens / step | 2 | 3.22091 | 3.20227 | 3.23825 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | accepted_draft_tokens_per_step | tokens / step | 4 | 3.15606 | 3.125 | 3.18697 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | accepted_draft_tokens_per_step | tokens / step | 8 | 3.18415 | 3.14527 | 3.2407 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | accepted_draft_tokens_per_step | tokens / step | 16 | 3.16884 | 3.13368 | 3.19166 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | chat | accepted_draft_tokens_per_step | tokens / step | 32 | 3.15292 | 3.14228 | 3.15967 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | accepted_draft_tokens_per_step | tokens / step | 1 | 3.25342 | 3.21971 | 3.30099 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | accepted_draft_tokens_per_step | tokens / step | 2 | 3.23919 | 3.1935 | 3.27732 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | accepted_draft_tokens_per_step | tokens / step | 4 | 3.25714 | 3.24269 | 3.28415 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | accepted_draft_tokens_per_step | tokens / step | 8 | 3.26747 | 3.24009 | 3.28298 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | accepted_draft_tokens_per_step | tokens / step | 16 | 3.29706 | 3.27565 | 3.313 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | chat | accepted_draft_tokens_per_step | tokens / step | 32 | 3.26863 | 3.25423 | 3.28361 | 3 |
| Native MXFP4 | performance-native | agent-warm | draft_acceptance_fraction | fraction | 1 | 0.65764 | 0.63903 | 0.67508 | 3 |
| Native MXFP4 | performance-native | agent-warm | draft_acceptance_fraction | fraction | 2 | 0.57948 | 0.55254 | 0.59501 | 3 |
| Native MXFP4 | performance-native | agent-warm | draft_acceptance_fraction | fraction | 4 | 0.52675 | 0.51471 | 0.53678 | 3 |
| Native MXFP4 | performance-native | agent-warm | draft_acceptance_fraction | fraction | 8 | 0.59104 | 0.57457 | 0.60368 | 3 |
| Native MXFP4 | performance-native | agent-warm | draft_acceptance_fraction | fraction | 16 | 0.65072 | 0.64407 | 0.65683 | 3 |
| Native MXFP4 | performance-native | agent-warm | draft_acceptance_fraction | fraction | 32 | 0.76 | 0.75466 | 0.76328 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | draft_acceptance_fraction | fraction | 1 | 0.71105 | 0.68966 | 0.73013 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | draft_acceptance_fraction | fraction | 2 | 0.722 | 0.68826 | 0.74805 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | draft_acceptance_fraction | fraction | 4 | 0.57258 | 0.54618 | 0.59721 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | draft_acceptance_fraction | fraction | 8 | 0.5936 | 0.54913 | 0.61598 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | draft_acceptance_fraction | fraction | 16 | 0.69863 | 0.67547 | 0.71177 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | draft_acceptance_fraction | fraction | 32 | 0.7546 | 0.74452 | 0.76261 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | draft_acceptance_fraction | fraction | 1 | 0.50001 | 0.48645 | 0.5104 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | draft_acceptance_fraction | fraction | 2 | 0.51199 | 0.50945 | 0.51375 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | draft_acceptance_fraction | fraction | 4 | 0.48837 | 0.48601 | 0.49255 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | draft_acceptance_fraction | fraction | 8 | 0.52232 | 0.51031 | 0.53347 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | draft_acceptance_fraction | fraction | 16 | 0.50825 | 0.49761 | 0.51483 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | draft_acceptance_fraction | fraction | 32 | 0.51866 | 0.51614 | 0.52079 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | draft_acceptance_fraction | fraction | 1 | 0.49385 | 0.48073 | 0.50466 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | draft_acceptance_fraction | fraction | 2 | 0.50688 | 0.49104 | 0.5289 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | draft_acceptance_fraction | fraction | 4 | 0.50377 | 0.48667 | 0.53376 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | draft_acceptance_fraction | fraction | 8 | 0.5071 | 0.50231 | 0.51001 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | draft_acceptance_fraction | fraction | 16 | 0.49392 | 0.49257 | 0.49537 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | draft_acceptance_fraction | fraction | 32 | 0.49921 | 0.49407 | 0.50405 | 3 |
| Native MXFP4 | performance-native | agent-warm | accepted_draft_tokens_per_step | tokens / step | 1 | 2.09385 | 1.95666 | 2.21133 | 3 |
| Native MXFP4 | performance-native | agent-warm | accepted_draft_tokens_per_step | tokens / step | 2 | 2.45425 | 2.34954 | 2.54653 | 3 |
| Native MXFP4 | performance-native | agent-warm | accepted_draft_tokens_per_step | tokens / step | 4 | 2.57858 | 2.51799 | 2.61727 | 3 |
| Native MXFP4 | performance-native | agent-warm | accepted_draft_tokens_per_step | tokens / step | 8 | 2.59926 | 2.52559 | 2.65729 | 3 |
| Native MXFP4 | performance-native | agent-warm | accepted_draft_tokens_per_step | tokens / step | 16 | 2.40405 | 2.38507 | 2.4417 | 3 |
| Native MXFP4 | performance-native | agent-warm | accepted_draft_tokens_per_step | tokens / step | 32 | 2.22244 | 2.21553 | 2.22813 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | accepted_draft_tokens_per_step | tokens / step | 1 | 2.02898 | 1.89679 | 2.12366 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | accepted_draft_tokens_per_step | tokens / step | 2 | 2.46142 | 2.19355 | 2.59554 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | accepted_draft_tokens_per_step | tokens / step | 4 | 2.79799 | 2.67491 | 2.90625 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | accepted_draft_tokens_per_step | tokens / step | 8 | 2.94956 | 2.72635 | 3.06203 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | accepted_draft_tokens_per_step | tokens / step | 16 | 2.78333 | 2.67797 | 2.84669 | 3 |
| NVFP4 target | performance-nvtarget-layermax | agent-warm | accepted_draft_tokens_per_step | tokens / step | 32 | 2.56083 | 2.50864 | 2.63539 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | accepted_draft_tokens_per_step | tokens / step | 1 | 2.50006 | 2.43227 | 2.55202 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | accepted_draft_tokens_per_step | tokens / step | 2 | 2.55994 | 2.54724 | 2.56874 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | accepted_draft_tokens_per_step | tokens / step | 4 | 2.44185 | 2.43003 | 2.46276 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | accepted_draft_tokens_per_step | tokens / step | 8 | 2.61158 | 2.55155 | 2.66734 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | accepted_draft_tokens_per_step | tokens / step | 16 | 2.54127 | 2.48807 | 2.57413 | 3 |
| EXL3 2.875 | performance-exl3-sparsechunks1 | agent-warm | accepted_draft_tokens_per_step | tokens / step | 32 | 2.5933 | 2.5807 | 2.60395 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | accepted_draft_tokens_per_step | tokens / step | 1 | 2.46927 | 2.40367 | 2.52328 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | accepted_draft_tokens_per_step | tokens / step | 2 | 2.5344 | 2.45521 | 2.64449 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | accepted_draft_tokens_per_step | tokens / step | 4 | 2.51885 | 2.43333 | 2.66878 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | accepted_draft_tokens_per_step | tokens / step | 8 | 2.53552 | 2.51153 | 2.55006 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | accepted_draft_tokens_per_step | tokens / step | 16 | 2.46958 | 2.46285 | 2.47687 | 3 |
| EXL3 3.5 | performance-exl35-sparsechunks1 | agent-warm | accepted_draft_tokens_per_step | tokens / step | 32 | 2.49603 | 2.47037 | 2.52023 | 3 |

### Native repeat controls

| Configuration | Run | Workload | Metric | Unit | Concurrency | Mean | Min | Max | Legs |
|---|---|---|---|---|---|---|---|---|---|
| Native repeat: C1 / C32 | performance-native-final-repeat | chat | aggregate_tokens_per_second | tokens / second | 1 | 144.33876 | 142.18408 | 147.96784 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | chat | aggregate_tokens_per_second | tokens / second | 32 | 829.20123 | 808.52562 | 845.38877 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | chat | time_to_content_p50 | seconds | 1 | 0.343 | 0.32389 | 0.35768 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | chat | time_to_content_p50 | seconds | 32 | 2.08389 | 1.9283 | 2.32041 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | chat | seconds_p50 | seconds | 1 | 2.05508 | 1.95505 | 2.10632 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | chat | seconds_p50 | seconds | 32 | 10.91537 | 10.44734 | 11.29845 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | chat | tpot_p50 | milliseconds / token | 1 | 5.58005 | 5.272 | 5.76888 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | chat | tpot_p50 | milliseconds / token | 32 | 28.04011 | 27.5525 | 28.7041 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | agent-warm | aggregate_tokens_per_second | tokens / second | 1 | 120.52311 | 116.44748 | 125.9366 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | agent-warm | aggregate_tokens_per_second | tokens / second | 32 | 757.46024 | 748.24773 | 769.42395 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | agent-warm | time_to_content_p50 | seconds | 1 | 0.45154 | 0.44482 | 0.46081 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | agent-warm | time_to_content_p50 | seconds | 32 | 1.37282 | 1.28338 | 1.5069 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | agent-warm | seconds_p50 | seconds | 1 | 2.08683 | 1.95783 | 2.15999 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | agent-warm | seconds_p50 | seconds | 32 | 9.29227 | 9.17089 | 9.45228 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | agent-warm | tpot_p50 | milliseconds / token | 1 | 6.45523 | 6.22495 | 6.61806 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat | agent-warm | tpot_p50 | milliseconds / token | 32 | 30.51588 | 29.70668 | 31.04263 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 1 | 131.69121 | 127.37463 | 138.58722 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 32 | 823.54665 | 752.15342 | 859.38987 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat-thinking | agent-warm | time_to_content_p50 | seconds | 1 | 8.63607 | 5.72135 | 10.82919 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat-thinking | agent-warm | time_to_content_p50 | seconds | 32 | 32.63674 | 31.72469 | 33.90768 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat-thinking | agent-warm | seconds_p50 | seconds | 1 | 10.04876 | 7.07331 | 11.97585 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat-thinking | agent-warm | seconds_p50 | seconds | 32 | 37.43951 | 37.12352 | 38.01407 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat-thinking | agent-warm | ttft_p50 | seconds | 1 | 0.42808 | 0.39298 | 0.44668 | 3 |
| Native repeat: C1 / C32 | performance-native-final-repeat-thinking | agent-warm | ttft_p50 | seconds | 32 | 0.63795 | 0.6279 | 0.65702 | 3 |

### Native drafter experiment

| Configuration | Run | Workload | Metric | Unit | Concurrency | Mean | Min | Max | Legs |
|---|---|---|---|---|---|---|---|---|---|
| NVFP4 drafter | performance-nvfp4-bounded | chat | aggregate_tokens_per_second | tokens / second | 1 | 143.65544 | 138.67321 | 147.28239 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded | chat | aggregate_tokens_per_second | tokens / second | 2 | 235.92901 | 223.96751 | 252.5307 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded | chat | aggregate_tokens_per_second | tokens / second | 4 | 363.10395 | 341.48937 | 384.50535 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded | chat | aggregate_tokens_per_second | tokens / second | 8 | 459.65381 | 446.34921 | 471.18706 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded | chat | aggregate_tokens_per_second | tokens / second | 16 | 613.47577 | 599.55677 | 627.46817 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded | chat | aggregate_tokens_per_second | tokens / second | 32 | 834.18491 | 781.62742 | 918.16692 | 3 |
| Drafting off | performance-off | chat | aggregate_tokens_per_second | tokens / second | 1 | 83.62755 | 82.25775 | 85.16804 | 3 |
| Drafting off | performance-off | chat | aggregate_tokens_per_second | tokens / second | 2 | 126.74948 | 123.9261 | 130.88017 | 3 |
| Drafting off | performance-off | chat | aggregate_tokens_per_second | tokens / second | 4 | 177.57729 | 166.17762 | 184.31774 | 3 |
| Drafting off | performance-off | chat | aggregate_tokens_per_second | tokens / second | 8 | 233.15431 | 228.47437 | 236.49601 | 3 |
| Drafting off | performance-off | chat | aggregate_tokens_per_second | tokens / second | 16 | 405.34759 | 395.67327 | 415.75478 | 3 |
| Drafting off | performance-off | chat | aggregate_tokens_per_second | tokens / second | 32 | 532.24373 | 526.56808 | 536.10003 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded | agent-warm | aggregate_tokens_per_second | tokens / second | 1 | 117.16367 | 111.36654 | 120.89754 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded | agent-warm | aggregate_tokens_per_second | tokens / second | 2 | 183.98284 | 177.86796 | 189.27928 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded | agent-warm | aggregate_tokens_per_second | tokens / second | 4 | 267.08033 | 248.82266 | 279.10422 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded | agent-warm | aggregate_tokens_per_second | tokens / second | 8 | 403.1263 | 366.33583 | 440.36279 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded | agent-warm | aggregate_tokens_per_second | tokens / second | 16 | 541.46462 | 512.15278 | 569.6481 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded | agent-warm | aggregate_tokens_per_second | tokens / second | 32 | 734.65866 | 724.43388 | 740.57774 | 3 |
| Drafting off | performance-off | agent-warm | aggregate_tokens_per_second | tokens / second | 1 | 82.23299 | 73.79241 | 97.16614 | 3 |
| Drafting off | performance-off | agent-warm | aggregate_tokens_per_second | tokens / second | 2 | 117.2104 | 113.83008 | 119.07398 | 3 |
| Drafting off | performance-off | agent-warm | aggregate_tokens_per_second | tokens / second | 4 | 161.81974 | 157.27342 | 170.19372 | 3 |
| Drafting off | performance-off | agent-warm | aggregate_tokens_per_second | tokens / second | 8 | 215.53018 | 209.70815 | 223.60875 | 3 |
| Drafting off | performance-off | agent-warm | aggregate_tokens_per_second | tokens / second | 16 | 389.37659 | 370.81091 | 402.72248 | 3 |
| Drafting off | performance-off | agent-warm | aggregate_tokens_per_second | tokens / second | 32 | 503.24721 | 489.56568 | 511.11217 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 1 | 122.2086 | 118.72534 | 124.78506 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 8 | 445.08521 | 432.96208 | 452.17002 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded-thinking | agent-warm | aggregate_tokens_per_second | tokens / second | 32 | 818.8933 | 792.65557 | 845.7776 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded-thinking | agent-warm | time_to_content_p50 | seconds | 1 | 9.09978 | 7.44338 | 11.20267 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded-thinking | agent-warm | time_to_content_p50 | seconds | 8 | 16.42769 | 14.29041 | 17.82172 | 3 |
| NVFP4 drafter | performance-nvfp4-bounded-thinking | agent-warm | time_to_content_p50 | seconds | 32 | 33.71472 | 32.04934 | 36.0553 | 3 |

</details>

---



## Original release notes

The original release notes below retain their historical wording. The four-format comparison is now complete; current comparison results, charts and inference settings are above.

# DeepSeek V4.1 Flash 1017: measured native MXFP4 reference

This is the original mixed-precision checkpoint with pqhaz's published 38-tensor attention edit applied. Routed expert weights remain native MXFP4 and Engram tables remain FP8. “Native” and “MXFP4” describe this same reference, not two separate quantizations. We did not originate the edit or run the upstream Heretic optimizer.

This is an interim measured reference. Comparisons against another attention edit, NVFP4 and EXL3 are ongoing. It is not presented as the universally best method or as having zero refusals.

## Source and measurements

Base revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`.
Patch repository: [pqhaz/DeepSeek-V4.1-Flash-1017-heretic](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic), revision `12d58ad0878b5008e80d720251adb969a0069412`.
Patch SHA256: `16e63cd3e79c3590855aad340b5b9dc37f27562bf2d0a40f85268915da221ce2`.

Local paired chat-mode scores: GSM8K 1276/1319, executed HumanEval 156/164, strict-format MMLU 11824/14042. Original base: 1279, 157 and 11920 respectively. These are local protocols, not official leaderboard scores. Temperature-zero output is not bitwise deterministic.

On four RTX PRO 6000 Blackwell Workstation GPUs, native drafting achieved mean aggregate reasoning throughput of 131.55, 425.19 and 830.06 output tokens/s at concurrency 1, 8 and 32. Three repeated legs used a warm 40K public-code prefix, natural stopping, temperature 1, top_p 1 and a 16384-token output cap. All nine legs passed integrity checks. Unrelated training and embedding workloads remained active.

The official HarmBench classifier passed 95/100 balanced human-majority controls. It scored direct-prompt behavior success on 287/300 edited outputs versus 30/300 base outputs. This measures behavior success, not a generic refusal rate. Full methodology, limitations and plots are in the [benchmark dataset](https://huggingface.co/datasets/Infatoshi/DeepSeek-V4.1-Flash-1017-local-evaluation/tree/33f144d3ccb5a7009cdb17c9cd21a2a474294356).

## Run on four 96GB Blackwell GPUs

Use Linux, Docker with NVIDIA Container Toolkit, and sufficient disk for approximately 510.3GB of serialized tensors. The pinned runtime streams Engram tables from disk. The configured 300000-token context and 32 scheduler slots do not imply 32 simultaneous full-context sessions. Runtime cache metadata estimates approximately 15 full 300K contexts before workload overhead; that concurrency was not measured.

```bash
hf download Infatoshi/DeepSeek-V4.1-Flash-1017-MXFP4-uncensored --local-dir /data/deepseek-model
bash /data/deepseek-model/runtime/serve-native.sh /data/deepseek-model /data/deepseek-cache
```

The launcher never stops or replaces an existing container. Wait for `http://127.0.0.1:8317/health` to succeed. The local API model ID is `deepseek-v41-local`. Remote clients should forward loopback port 8317 over SSH. `MODE=off` disables speculative decoding for a matched comparison; the default uses DSpark with up to five proposed tokens.

```bash
bash /data/deepseek-model/runtime/harnesses/claude-ds.sh
bash /data/deepseek-model/runtime/harnesses/opencode-ds.sh
uv run --no-project python /data/deepseek-model/runtime/harnesses/configure-omp.py
bash /data/deepseek-model/runtime/harnesses/omp-ds.sh
```

Install the corresponding clients first. The OMP configuration helper preserves existing providers and refuses to overwrite an existing local-provider entry. The Claude wrapper retains normal instructions, hooks and plugins, selects bypassPermissions, and routes Agent-tool subagents to this same local model. Plain Claude commands launched through Bash regain the normally configured provider. No existing credentials are bundled.

The measured native runtime passed two editing tasks in each harness and a real Claude session reaching 260809 input tokens. That fixture does not establish arbitrary long-context task reliability. Incremental tool-driven growth passed; a single 250K-token initial stdin prompt failed Claude's estimator before reaching the API.

The exact packaged launcher and wrappers were also tested with this staged release as the only model mount. All six editing fixtures passed. The packaged Claude wrapper reached 260,205 input tokens with a reported 300,000-token window, 11 Read calls and no compaction, retaining the checked facts. `evaluation/runtime-qualification.json` records that full model and launcher check. The subsequent client-only OpenCode fix is recorded in `evaluation/client-routing-qualification.json`: an explicit CLI model selection prevents a configured agent default from redirecting the main session. A conflicting-agent fixture passed with the exact updated wrapper; server configuration and weight files are unchanged.

Weights retain the upstream MIT license. Runtime components retain their own licenses; included vLLM/B12X material is Apache-2.0. Full-file release hashes and provenance accompany the checkpoint. Raw benchmark requests, generated responses and private machine information are not included.


## Additional installed-client long-context evidence

All three clients completed bounded tool-memory fixtures above 250K input tokens with the native model’s 300K server window and no compaction. Claude Code reached **260,205** input tokens using the exact packaged wrapper. Installed OpenCode and OMP aliases reached **281,477** and **255,419**, retaining the checked early and final facts. The OpenCode test used stdin rather than a truncated file attachment. OMP required a verifier-only correction for filename:line-range read paths; its original receipt and session were preserved.

See `evaluation/installed-harness-long-context.json` for receipt hashes and scope. The OpenCode/OMP checks are installed-client evidence, not new exact-package runs. These fixtures do not certify every source line, arbitrary long-context reasoning, media support, or relative client speed. That evidence-only update changed no runtime, configuration or weight file. The later OMP client update is described below.


## OMP context budgeting

The bundled `omp-ds.sh` loads `omp-context.yml` for that invocation. Automatic compaction and mid-turn checks remain enabled. It reserves **24,576 tokens**: the **16,384-token** output budget plus **8,192 tokens** of headroom. With the **300,000-token** window, the reserve-based trigger is **275,424 tokens**. The same profile is used by the installed local alias. Global settings and other providers are unchanged.

OMP normally reserves 15% when this setting is unset. An earlier NVFP4 package fixture compacted at 269,638 input tokens under that policy and failed the qualification gate; the failed receipt was preserved. The revised native client configuration passed all six editing tasks. Its exact packaged OpenCode and OMP wrappers reached **281,805** and **259,066** input tokens, retained the checked facts, and completed without compaction or fixture changes. These remain bounded tool-memory tests.

`evaluation/omp-context-qualification.json` binds the previous and tested runtime hashes and the client results. Only the OMP wrapper and its new configuration file changed; model weights and server configuration did not. The prior Claude long-context result remains recorded separately. See [OMP settings](https://github.com/can1357/oh-my-pi/blob/main/docs/settings.md) for the reserve and per-run overlay semantics.
