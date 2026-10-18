---
license: other
base_model: zai-org/GLM-5.3-Flash
tags:
- moe
- pruned
- glm
- glm-5.3
- model-compression
- rantal
- polish-nlp
---

# GLM-5.3-Flash-Compact-35B

<p align="center">
  <b>Architected & Pruned by <a href="https://huggingface.co/rantal-ai">Rantal</a></b><br>
  <i>Powering advanced analytical projects like <a href="https://slowasejmu.pl">SłowaSejmu.pl</a></i>
</p>

## Overview
This is an efficient, compact derivative of **GLM-5.3-Flash** from Zhipu AI (`zai-org/GLM-5.3-Flash`). 

By applying strategic **MoE Expert Slicing**, the parameter count has been reduced by approximately **85–90%**, transforming a ~320B parameter frontier model into an agile **~35B parameter architecture** runnable on single consumer and workstation GPUs (such as RTX 4090, A100, or Apple Silicon Mac Studio).

## Architecture & Pruning Specifications
- **Base Model:** `zai-org/GLM-5.3-Flash` (originally 288 routed experts + 1 shared expert)
- **Compact Routing:** Sliced to **16 routed experts**
- **Shared Experts:** 1 (retained in full to maintain fundamental linguistic coherence)
- **Top-K Routing (`num_experts_per_tok`):** Reduced from 8 to **4**
- **Layers:** 45 hidden layers (intact self-attention backbone)

## Methodology
The expert reduction was executed directly on the weight tensors via stream-based shard slicing. The gating/router layers were linearly projected to the top surviving expert indices to prevent out-of-bounds routing dispatch.

---
*Created by **Rantal** · Research & Analytical Engineering.*
