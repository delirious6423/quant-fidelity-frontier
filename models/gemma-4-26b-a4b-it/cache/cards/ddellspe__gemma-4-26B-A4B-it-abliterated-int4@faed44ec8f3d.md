---
language:
- en
license: gemma
base_model: google/gemma-4-26B-A4B-it
tags:
- gemma4
- abliterated
- uncensored
- representation-engineering
- int4
- compressed-tensors
pipeline_tag: text-generation
---

# Gemma 4 26B INT4 (Abliterated)

This repository contains an INT4-quantized, abliterated (refusal-direction ablated) variant of **Google Gemma 4 (26B-A4B-it)**.

## Model Summary
- **Base Model:** [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **Architecture:** Hybrid Attention (Local Sliding Window + Global) with 128 Mixture-of-Experts (MoE) layers (4 active per token, 26B total parameters / 4B active parameters).
- **Quantization:** Compressed-Tensors group-64 symmetric INT4 ($W4A16$).
- **Weight Footprint:** ~15.2 GB total weights (enables full 256k context serving inside ~28 GB VRAM).
- **Context Window:** 262,144 tokens (256k).

---

## Abliteration Methodology

Alignment fine-tuning (RLHF/DPO) often induces excessive false refusals on benign, technical, or exploratory queries (e.g. penetration testing, vulnerability analysis, administrative scripting, creative writing).

Following the mechanistic interpretability framework established by *Arditi et al. (2024: "Refusal in Language Models Is Mediated by a Single Direction")*, refusal behavior is mediated by a low-dimensional subspace—predominantly a single direction vector $\\vec{r} \\in \\mathbb{R}^{d_{\\text{model}}}$ in the residual stream. When an input triggers refusal, intermediate hidden activations align along $\\vec{r}$, signaling downstream layers to output refusal stock phrases. By mathematically projecting out this vector from target weight matrices, the model loses its capacity to represent the refusal state while retaining 100% of its reasoning, coding, and factual knowledge.

### 1. Contrastive Activation Harvesting
Hidden states are recorded during forward passes over paired contrastive prompts (structurally and syntactically balanced harmful vs. harmless pairs across multiple domains).

For each target layer $l$, the mean refusal direction is computed from the final token activations:
$$\\vec{r}_l = \\mu_{\\text{harmful}, l} - \\mu_{\\text{harmless}, l}$$
$$\\hat{r}_l = \\frac{\\vec{r}_l}{\\|\\vec{r}_l\\|_2}$$

### 2. Mathematical Formulation & Orthogonal Projection
In Gemma 4 (26B-A4B-it), the hidden dimension is $d_{\\text{model}} = 2816$. Each transformer block $l$ updates the residual stream $x_l$ as:

$$x_{l+1} = x_l + W_{\\text{o}} h_{\\text{attn}} + W_{\\text{mlp\\_down}} h_{\\text{mlp}} + \\sum_{e \\in \\text{top\\_k}} g_e W_{\\text{expert\\_down}, e} h_{e}$$

Every path writing into the residual stream is a linear projection whose output dimension is $d_{\\text{model}} = 2816$. For any target projection matrix $W \\in \\mathbb{R}^{d_{\\text{out}} \\times d_{\\text{in}}}$ with $d_{\\text{out}} = 2816$, the refusal component is projected out:

$$W^' = (I - \\hat{r} \\hat{r}^T) W = W - \\hat{r} (\\hat{r}^T W)$$

### 3. Target Tensors in Gemma 4
1. **Self-Attention Output Projections:**
   - Tensor: `model.language_model.layers.{l}.self_attn.o_proj.weight` ($[2816, 4096]$)
   - $W \\leftarrow W - \\hat{r} (\\hat{r}^T W)$
2. **Shared MLP Down-Projections:**
   - Tensor: `model.language_model.layers.{l}.mlp.down_proj.weight` ($[2816, 2112]$)
   - $W \\leftarrow W - \\hat{r} (\\hat{r}^T W)$
3. **MoE Sparse Down-Projections:**
   - Tensor: `model.language_model.layers.{l}.experts.down_proj` ($[128, 2816, 704]$)
   - Batched Einstein summation across all 128 experts:
     $$\\text{proj}_{e, m} = \\sum_{d} \\hat{r}_d W_{e, d, m}$$
     $$W_{e, d, m} \\leftarrow W_{e, d, m} - \\hat{r}_d \\cdot \\text{proj}_{e, m}$$

### 4. Target Layer Selection
Gemma 4 has 30 transformer layers ($0-29$).
- **Layers 0–9 (Untouched):** Preserves low-level token embeddings, lexical representations, and syntax trees.
- **Layers 10–24 (Orthogonalized):** The intermediate semantic representation layers where refusal conditioning and steering occurs.
- **Layers 25–29 (Untouched):** Preserves softcapping, output normalization, and vocabulary logit mapping.

---

## INT4 Quantization Details
After orthogonalizing the refusal subspace across layers 10–24, the model was requantized using **Neural Magic Compressed-Tensors**:
- **Format:** Group-64 symmetric INT4 ($W4A16$)
- **Granularity:** Per-channel scales with 64-element block grouping
- **Layer-by-Layer Quantization:** Processed sequentially on unified memory to preserve precision
- **Footprint Reduction:** Reduced memory footprint by ~68% (from 48.2 GB to 15.2 GB) without degradation in perplexity or reasoning capability.

---

## Serving with vLLM

Gemma 4 requires the `gemma4` tool-call parser and reasoning parser:

```bash
vllm serve ddellspe/gemma-4-26B-A4B-it-abliterated-int4 \
  --trust-remote-code \
  --port 8000 \
  --max-model-len 262144 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4
```
