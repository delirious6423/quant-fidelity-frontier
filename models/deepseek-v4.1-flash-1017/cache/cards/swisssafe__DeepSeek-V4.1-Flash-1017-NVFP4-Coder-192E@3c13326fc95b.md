---
license: mit
base_model:
  - deepseek-ai/DeepSeek-V4.1-Flash-1017
  - nvidia/DeepSeek-V4.1-Flash-1017-NVFP4
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - coding
  - agentic-coding
  - nvfp4
  - expert-pruning
  - deepseek-v41
  - vllm
  - on-premises
---

# DeepSeek V4.1 Flash 1017 NVFP4 Coder 192E

**Private AI coding model for a team, optimized for NVIDIA and coding, ideal on 2x RTX 6000.**

Excluding engrams (that can be SSD/RAM based), it only requires 55% of the VRAM compared to the full model, or only as much as Unsloth's 2-bit UD-IQ2_XXS, while performing as well as the full model for coding.

In our testing, it outperforms GPT-5.6 Sol and Claude Opus 5 in coding, based on the DeepSWE1.1 benchmark. It also delivered much better results than any popular quants from Unsloth or GSQ-RCO (which we love for smaller setups).

The model keeps only half the experts, but adds no additional quantization. This, together with our harness optimizations, contributed to the final results published below.

Full credit for the NVFP4 version this is based on goes to NVIDIA, and for the expert selection to the great work brought about by GSQ-RCO.

## Coding benchmark results

| Reference | DeepSWE v1.1 | Terminal-Bench 2.1 |
|---|---:|---:|
| **Swiss Safe 192-expert NVFP4** | **79.6% (90/113)** | **86.5% (77/89)** |
| [DeepSeek V4.1 Flash 1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | 77.0% | — |
| [DeepSeek V4.1 Flash, earlier release](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 74.2% | 90.6% |
| [OpenAI GPT-5.6 Sol, as reported by DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 73.0% | 88.8% |
| [Anthropic Claude Opus 5, as reported by DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 74.0% | 89.1% |
| [NVIDIA 1017 NVFP4 with MXFP4 Engram](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) | — | 81.11% |
| [Unsloth 8-bit reference](https://unsloth.ai/docs/models/deepseek-v4.1-flash#dynamic-v3.1-update-november-2nd-2026) | 71.4% | 72.6% |
| [Best Unsloth low-bit variant per suite in its published chart](https://unsloth.ai/docs/models/deepseek-v4.1-flash#dynamic-v3.1-update-november-2nd-2026) | 68.1% | 74.4% |
| [Best GSQ-RCO variant per suite in Unsloth’s tests](https://unsloth.ai/docs/models/deepseek-v4.1-flash#dynamic-v3.1-update-november-2nd-2026) | 61.9% | 69.1% |

Swiss Safe tested multiple harnesses with its own optimized configuration before achieving those final results, which outperform even the original published result of the full model on DeepSWE v1.1.

## Tested setup and throughput on two RTX PRO 6000 Blackwell GPUs

| Serving profile and workload | Active prefill | Active generation |
|---|---:|---: |
| RAM-backed Engram, one active client  | 12,501 tok/s | 112 tok/s |
| RAM-backed Engram, up to 20 active clients  | 11,100 tok/s shared | 214 tok/s combined |
| SSD-backed Engram, one client  | 8,868 tok/s | 95 tok/s |
| SSD-backed Engram, up to 4 active clients  | 7,000 tok/s shared | 95 tok/s combined |

Active rates use the measured logical batch intervals across both pipeline stages. Prefill excludes cached prompt tokens. Generation counts accepted output and excludes rejected draft tokens. Mixed batches contribute time to both rates. Team results are aggregate. The backlog runs used 144 coding requests per profile, with six requests in each separate single-client reference. We used distinct optimized settings for the one-slot RAM, 20-slot RAM, and four-slot SSD profiles.

We tested on Linux with a vLLM-based runtime using two-stage pipeline parallelism, FP8 KV cache, and DSpark speculative decoding. The figures above identify which runs kept Engram in RAM and which used SSD-backed Engram.

Swiss Safe’s 192-expert NVFP4 build runs perfect on **two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs**, each with 96 GB of VRAM, with a **1M-token context window**.

Our SSD-backed profile reported a **3.78M-token GPU KV cache** shared across **up to four active requests**. By token count, that pool has room for three full-length contexts. Four requests can share it when their combined contexts fit. 

With Engram in system RAM and part of the KV cache offloaded to RAM, the same two GPUs can serve a coding team efficiently. In our team benchmark, 24 clients shared **up to 20 active request slots**, reaching **214 output tokens per second in aggregate during active generation**.

Teams can use it for multi-file changes, debugging, and long coding sessions while keeping inference, source code, and prompts inside their own infrastructure. For organizations with steady coding workloads, two workstation GPUs make it practical to evaluate the cost of a shared private deployment against hosted inference.

In our development evaluations, the build solved **90 of 113 DeepSWE v1.1 tasks (79.6%)** and **77 of 89 Terminal-Bench 2.1 tasks (86.5%)**. The DeepSWE coverage is numerically above the published GPT-5.6 Sol and Opus 5 references and the popular low-bit quants of this model (see table). Our harness configuration also helped get coverage above NVIDIA’s NVFP4 reference this model bases on. The comparison table gives the published reference scores for context.

## Model lineage

- **Original model.** DeepSeek AI developed and released DeepSeek V4.1 Flash 1017, including native text and image support. [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- **Weight source.** We started from NVIDIA’s NVFP4 checkpoint at revision `480550d349ede47a048eb65a276c5c01a06256e2`. We preserved the retained weight and scale tensors and left the attention, shared experts, Engram tables, vision weights, and speculative draft weights in their source formats. [NVIDIA model card](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4)
- **Expert selection.** We used ISTA-DASLab’s published GSQ-RCO Coder expert selection at revision `54229d38d588a66d1c6cff08af7b281b9326c656`. We applied those IDs to the NVIDIA weights, retaining **192 of 384 routed experts in each of 40 target layers** and remapping the router weights and biases. Top-6 routing is preserved. [ISTA-DASLab Coder model card](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF)

## Checkpoint and deployment

The checkpoint contains **276.0 GB** of Safetensors weights in 48 shards, including the vision encoder and projector and embedded DSpark/MTP weights. Of that, **104.5 GB (97.3 GiB)** is Engram lookup tables and their scales. The full tables can stay on SSD, with optional partial caching in system RAM, or be loaded entirely into RAM. Running the model requires a DeepSeek V4.1-compatible serving runtime.

In our two-GPU throughput tests, measured VRAM use was about **188 GiB combined** with SSD-backed Engram and **190 GiB combined** with RAM-backed Engram. These figures include the model, GPU KV cache, and runtime allocations.

The checkpoint is distributed under the MIT license. See [LICENSE](LICENSE) for the upstream notice and terms.

## Draft decoding fix

To make draft decoding work and significantly improve generation with high draft-token acceptance, we gave the drafter its own quantization config and selected the native MXFP4 kernels.

The base vLLM build inherited the main model's NVFP4 settings when loading the drafter, even though NVIDIA had preserved the draft weights at their original precision.

## Software integration

This paragraph cites the base setup as shipped in the NVIDIA release, before any of our fixes and adjustments to optimize it for the 2x RTX 6000. If you would like us to publish our reference setup, please contact us (see link bellow).

**Supported Runtime Engine(s):** <br>
* **vLLM** (needs MXFP4 Engram support, [vllm-project/vllm#61530](https://github.com/vllm-project/vllm/pull/61530)) <br>

This checkpoint stores the Engram lookup tables as MXFP4, which stock vLLM images cannot load. Use a vLLM build that includes the pull request above.

* **SGLang** <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Blackwell <br>

**Preferred Operating System(s):** <br>
* Linux <br>

## Deploy private AI with Swiss Safe

Reach out to Swiss Safe for help running this model on your infrastructure or building safe private AI capabilities more broadly. We can help size and deploy the system, integrate it into your workflows, and design access controls for sensitive code and data. [Swiss Safe Agent](https://swisssafe.com/agent)
