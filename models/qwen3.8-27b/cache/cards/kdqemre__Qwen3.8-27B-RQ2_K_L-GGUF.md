---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B

tags:
- qwen
- qwen3_8
- reasoning
- code
- vision
- conversational
- gguf
---

![image(1)](https://cdn-uploads.huggingface.co/production/uploads/642b0a75508b7246f9d42420/on_gglhl6l5jMJIkyUrA4.jpeg)

You need this repository to run this model. Standart llama.cpp will not work on this.

https://github.com/kdqemre/llama.cpp-rq

---

**Speed and size mean nothing if the model's quality is destroyed.** 

The new RQ series (RQ2, RQ3, RQ4) in my llama.cpp fork compresses models to low bits while preserving high-tier quality. Quality is secured through a Walsh–Hadamard Transform (WHT). Speed is maintained via fused CUDA kernels.

Here is how I did it. 👇

**📉 1. The Low-Bit Problem**
LLM weight matrices contain outliers. Standard quantization sets its step size based on the block's maximum value. A single outlier inflates this step, crushing ordinary weights onto the lowest codes. At low bits, most of the block's usable information is lost.

**🔄 2. The Solution: Rotated Quantization (The Math)**
To keep outliers from hijacking the range, RQ rotates the sub-block before quantization. Let's look at a toy 4-bit symmetric example \(n=4\) on a weight vector \\(\mathbf{w}\\) with a moderate outlier:

$$ \mathbf{w} = \begin{bmatrix} 0.1 & 0.4 & 0.7 & 2.8 \end{bmatrix}^\top $$

In plain symmetric 4-bit, we have 15 levels (-7 to 7).

 **Standard Symmetric 4-bit:**
- Max magnitude is 2.8. Step size \\(s\\):

$$ s = \frac{2.8}{7} = 0.4 $$

- Quantize \\(\mathbf{q} = \mathrm{round}(\mathbf{w} / s)\\) and Dequantize \\(\hat{\mathbf{w}} = \mathbf{q} \cdot s\\):

$$ \mathbf{q} = \begin{bmatrix} 0 & 1 & 2 & 7 \end{bmatrix}^\top \implies \hat{\mathbf{w}} = \begin{bmatrix} 0.0 & 0.4 & 0.8 & 2.8 \end{bmatrix}^\top $$

*Result:* The delicate 0.1 detail was completely crushed to 0.0. The information is permanently lost.

 **RQ (WHT):**
RQ applies a Walsh–Hadamard Transform matrix (\\(H\\)). To preserve the L2 norm and spread the outlier's energy, the rotation is normalized by \\(\frac{1}{\sqrt{n}}\\).
- Rotated vector \\(\mathbf{w}' = \frac{1}{\sqrt{n}} H\mathbf{w}\\):

$$ \mathbf{w}' = \begin{bmatrix} 2.0 & -1.2 & -1.5 & 0.9 \end{bmatrix}^\top $$

- Max magnitude dropped from 2.8 to 2.0! New step size \\(s'\\):

$$ s' = \frac{2.0}{7} \approx 0.285 $$

- Quantize \\(\mathbf{q}' = \mathrm{round}(\mathbf{w}' / s')\\) and Dequantize \\(\hat{\mathbf{w}}' = \mathbf{q}' \cdot s'\\):

$$ \mathbf{q}' = \begin{bmatrix} 7 & -4 & -5 & 3 \end{bmatrix}^\top \implies \hat{\mathbf{w}}' = \begin{bmatrix} 2.00 & -1.14 & -1.42 & 0.85 \end{bmatrix}^\top $$

- Inverse WHT back to original domain \\(\hat{\mathbf{w}} = \frac{1}{\sqrt{n}} H^\top \hat{\mathbf{w}}'\\):

$$ \hat{\mathbf{w}} = \begin{bmatrix} 0.14 & 0.43 & 0.71 & 2.71 \end{bmatrix}^\top $$

*Result:* The 0.1 survived as 0.14. The small weights are preserved, and the Mean Squared Error (MSE) is roughly halved in this toy example.
*(Note: Production RQ operates on 32-weight blocks, applies an offline-optimized ±1 sign diagonal before the WHT, and uses true K-quant formats).*

**⚡ 3. Fused Kernels for Inference Speed**

Applying WHT per weight during generation would kill performance. Instead, I shifted the cost:

- **Activation prep:** WHT is fused into the activation quantizer. It is paid once per token, not per weight.
- **Matmul:** Weights unpack directly onto native dp4a / MMA (RQ4's block layout is byte-identical to standard Q4_K).

> **(Note: These fused kernels and custom inference paths are currently CUDA-only).**

---

**📊 4. The Results (Qwen3.8-27B on RTX 4090)**

**Setup:** 8-shot GSM8K-100 (Exact Match), greedy decode, MTP off. Hardware: RTX 4090 (**power-limited to 150W**).

I compared my mixed 2-bit bulk recipe against the high-bit baseline. Here are the results:

| Metric | Q5_K_M (Baseline) | RQ2_K_L (Mixed 2-bit) |
| :--- | :--- | :--- |
| **Bits per Weight** | 5.72 bpw | 3.70 bpw |
| **Model Size** | 19.5 GB | **12.6 GB** |
| **Accuracy Score** | 98/100 | **98/100** |
| **Decode Speed** | 29.2 t/s | **32.3 t/s** |

💡 **The Bottom Line:** On this benchmark, the RQ recipe matches Q5_K_M's quality while being **6.9 GB smaller** and decoding **faster**.

---

**🛠️ 5. Custom Modular Mixing & NVFP4**

RQ is fully modular. You can mix RQ2, RQ3, and RQ4 across different layers using regex targeting (e.g., keeping the FFN bulk at RQ2, while elevating sensitive tensors to RQ3/RQ4).

Furthermore, the fork supports **NVFP4**. You can generate and run models entirely in NVFP4, or mix it per-tensor alongside RQ types to build highly customized quantization recipes.

*Quality first. Then the bits. Then the kernels.*

Code and build notes are here:
🔗 [https://github.com/kdqemre/llama.cpp-rq](https://github.com/kdqemre/llama.cpp-rq)